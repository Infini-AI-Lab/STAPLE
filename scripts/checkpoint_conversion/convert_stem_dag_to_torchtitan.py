# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert STEM-pt stem_dag checkpoint to torchtitan DCP format.

Handles:
1. Backbone weights: renames keys from STEM-pt naming to torchtitan naming
2. Stem embeddings: reassembles sharded stem embeddings and maps stem_idx to layer_idx
3. Optimizer states: converts both backbone and stem optimizer states

The output is a DCP checkpoint directory that can be loaded directly by torchtitan's
CheckpointManager via `--checkpoint.initial_load_path`.

Source format (STEM-pt consolidated checkpoint):
  - consolidated/consolidated.pth: {model: {...}, optim: {...}}
  - stem_shards/stem_model_mp{i}.pt: {stem_embeddings.{stem_idx}.weight: Tensor}
  - stem_shards/stem_optim_mp{i}.pt: Adam state per stem embedding
  - params.json: model config with stem_layers list

Target format (torchtitan DCP):
  - DCP directory with model state using torchtitan key names
  - Stem embeddings included in the model state dict

Usage:
    python apps/main/convert_stem_dag_to_torchtitan.py \
        --ckpt-path ~/checkpoints/om1b-midfine-stemL12-continual/0000017000 \
        --output-dir ~/checkpoints/om1b-stemL12-torchtitan
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key mapping: STEM-pt -> torchtitan
# ---------------------------------------------------------------------------

KEY_MAP = {
    "lm_transformer.layers.{}.attention.wq.weight": "layers.{}.attention.qkv_linear.wq.weight",
    "lm_transformer.layers.{}.attention.wk.weight": "layers.{}.attention.qkv_linear.wk.weight",
    "lm_transformer.layers.{}.attention.wv.weight": "layers.{}.attention.qkv_linear.wv.weight",
    "lm_transformer.layers.{}.attention.wo.weight": "layers.{}.attention.wo.weight",
    "lm_transformer.layers.{}.attention.q_norm.weight": "layers.{}.attention.q_norm.weight",
    "lm_transformer.layers.{}.attention.k_norm.weight": "layers.{}.attention.k_norm.weight",
    "lm_transformer.layers.{}.feed_forward.w1.weight": "layers.{}.feed_forward.w1.weight",
    "lm_transformer.layers.{}.feed_forward.w2.weight": "layers.{}.feed_forward.w2.weight",
    "lm_transformer.layers.{}.feed_forward.w3.weight": "layers.{}.feed_forward.w3.weight",
    "lm_transformer.layers.{}.post_attention_norm.weight": "layers.{}.attention_norm.weight",
    "lm_transformer.layers.{}.post_feedforward_norm.weight": "layers.{}.ffn_norm.weight",
    "lm_transformer.tok_embeddings.weight": "tok_embeddings.weight",
    "lm_transformer.output.weight": "lm_head.weight",
    "lm_transformer.norm.weight": "norm.weight",
}


def rename_key(src_key: str) -> str | None:
    """Convert a STEM-pt key to torchtitan format. Returns None if key should be skipped."""
    if "layers." in src_key:
        import re

        match = re.search(r"layers\.(\d+)\.", src_key)
        if not match:
            return None
        layer_num = match.group(1)
        abstract_key = re.sub(r"layers\.\d+\.", "layers.{}.", src_key)
        target_pattern = KEY_MAP.get(abstract_key)
        if target_pattern is None:
            return None
        return target_pattern.format(layer_num)
    else:
        return KEY_MAP.get(src_key)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def consolidate_if_needed(ckpt_path: Path) -> Path:
    """Consolidate DCP checkpoint into consolidated.pth if not already done.

    Returns the path to the consolidated .pth file.
    """
    candidates = [
        ckpt_path / "consolidated" / "consolidated.pth",
        ckpt_path / "consolidated.pth",
    ]
    for path in candidates:
        if path.exists():
            return path

    # Check if this is a DCP checkpoint (has .metadata)
    if (ckpt_path / ".metadata").exists():
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

        consolidate_dir = ckpt_path / "consolidated"
        consolidate_dir.mkdir(parents=True, exist_ok=True)
        output_path = consolidate_dir / "consolidated.pth"
        logger.info(f"Consolidating DCP checkpoint at {ckpt_path} -> {output_path}")
        dcp_to_torch_save(str(ckpt_path), str(output_path))

        # Copy params.json if present
        if (ckpt_path / "params.json").exists():
            import shutil

            shutil.copy2(ckpt_path / "params.json", consolidate_dir / "params.json")

        logger.info("Consolidation complete")
        return output_path

    raise FileNotFoundError(
        f"No consolidated.pth found and no DCP .metadata to consolidate from at {ckpt_path}"
    )


def load_consolidated(ckpt_path: Path) -> dict[str, Any]:
    """Load consolidated.pth from checkpoint directory, consolidating if needed."""
    consolidated_path = consolidate_if_needed(ckpt_path)
    logger.info(f"Loading consolidated checkpoint from {consolidated_path}")
    return torch.load(consolidated_path, map_location="cpu", weights_only=False)


def load_stem_shards(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load and reassemble stem embedding shards along embedding dim.

    Memory-efficient: loads one shard at a time, concatenates per-key, then
    frees the shard before loading the next.
    """
    stem_dir = ckpt_path / "stem_shards"
    if not stem_dir.exists():
        raise FileNotFoundError(f"Stem shards directory not found: {stem_dir}")

    model_shards = sorted(stem_dir.glob("stem_model_mp*.pt"))
    if not model_shards:
        raise FileNotFoundError(f"No stem model shards in {stem_dir}")

    logger.info(f"Loading {len(model_shards)} stem model shards")

    # First pass: get key list from shard 0
    shard0 = torch.load(model_shards[0], map_location="cpu", weights_only=False)
    keys = sorted(shard0.keys())
    # Accumulate parts per key
    parts_per_key = {key: [shard0[key]] for key in keys}
    del shard0

    # Load remaining shards one at a time
    for shard_path in model_shards[1:]:
        logger.info(f"  Loading {shard_path.name}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for key in keys:
            parts_per_key[key].append(shard[key])
        del shard

    # Concatenate along embedding dim
    reassembled = {}
    for key in keys:
        reassembled[key] = torch.cat(parts_per_key[key], dim=1)
        del parts_per_key[key]
        logger.info(f"  Reassembled {key}: {reassembled[key].shape}")

    return reassembled


def load_stem_optim_shards(ckpt_path: Path) -> dict[str, Any] | None:
    """Load and reassemble stem optimizer state shards."""
    stem_dir = ckpt_path / "stem_shards"
    optim_shards = sorted(stem_dir.glob("stem_optim_mp*.pt"))
    if not optim_shards:
        logger.info("No stem optimizer shards found, skipping")
        return None

    logger.info(f"Loading {len(optim_shards)} stem optimizer shards")
    all_shards = []
    for shard_path in optim_shards:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        all_shards.append(shard)

    # Reassemble: state dict has {state: {key: {step, exp_avg, exp_avg_sq}}, param_groups: [...]}
    reassembled = {"state": {}, "param_groups": all_shards[0]["param_groups"]}
    for key in all_shards[0]["state"]:
        entry = {}
        for field_name in all_shards[0]["state"][key]:
            parts = [shard["state"][key][field_name] for shard in all_shards]
            if parts[0].dim() == 0:
                entry[field_name] = parts[0]
            else:
                entry[field_name] = torch.cat(parts, dim=1)
        reassembled["state"][key] = entry

    return reassembled


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_model_state(
    src_model: dict[str, torch.Tensor],
    stem_embeddings: dict[str, torch.Tensor],
    stem_layers: list[int],
    n_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Convert backbone + stem embeddings to torchtitan state dict."""
    target = {}
    skipped = []

    for src_key, tensor in src_model.items():
        dst_key = rename_key(src_key)
        if dst_key is None:
            skipped.append(src_key)
            continue

        # Reshape q_norm/k_norm from [n_heads * head_dim] to [n_heads, head_dim]
        if dst_key.endswith("q_norm.weight") or dst_key.endswith("k_norm.weight"):
            tensor = tensor.view(n_heads, head_dim)

        target[dst_key] = tensor

    if skipped:
        logger.warning(f"Skipped {len(skipped)} source keys: {skipped}")

    # Map stem embeddings: stem_embeddings.{stem_idx} -> _stem_embeddings.{layer_idx}
    for stem_idx, layer_idx in enumerate(stem_layers):
        src_stem_key = f"stem_embeddings.{stem_idx}.weight"
        if src_stem_key not in stem_embeddings:
            logger.warning(f"Missing stem embedding: {src_stem_key}")
            continue
        dst_stem_key = f"_stem_embeddings.{layer_idx}.weight"
        target[dst_stem_key] = stem_embeddings[src_stem_key]
        logger.info(
            f"  Stem idx {stem_idx} (layer {layer_idx}): "
            f"{src_stem_key} -> {dst_stem_key} {stem_embeddings[src_stem_key].shape}"
        )

    return target


def convert_optim_state_flattened(
    src_optim: dict[str, Any],
    stem_optim: dict[str, Any] | None,
    stem_layers: list[int],
    model_keys: list[str],
    n_heads: int,
    head_dim: int,
) -> dict[str, Any] | None:
    """Convert optimizer state to torchtitan's flattened DCP format.

    torchtitan uses `flatten_optimizer_state_dict=True` which produces:
        state.{param_name}.step          - per-param Adam state
        state.{param_name}.exp_avg
        state.{param_name}.exp_avg_sq
        param_groups.{param_name}.lr     - per-param optimizer hyperparams
        param_groups.{param_name}.betas
        ... (all fields the runtime fused AdamW produces)

    These are stored under the "optimizer" key in the DCP.
    """
    src_state = src_optim["state"]

    first_key = next(iter(src_state.keys()))
    if isinstance(first_key, int):
        logger.warning(
            "Integer-indexed optimizer states require the exact parameter ordering "
            "from the training run. Skipping optimizer conversion for backbone. "
            "Consider resuming training with fresh optimizer states."
        )
        return None

    logger.info(f"Optimizer state uses string keys (first: {first_key})")

    flattened = {}
    skipped = []

    for src_key, opt_entry in src_state.items():
        dst_key = rename_key(src_key)
        if dst_key is None:
            skipped.append(src_key)
            continue

        for field_name, tensor in opt_entry.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() > 0:
                    if dst_key.endswith("q_norm.weight") or dst_key.endswith(
                        "k_norm.weight"
                    ):
                        tensor = tensor.view(n_heads, head_dim)
                flat_key = f"state.{dst_key}.{field_name}"
                flattened[flat_key] = tensor

    if skipped:
        logger.warning(f"Skipped {len(skipped)} optim state keys: {skipped}")

    # Add stem optimizer states
    if stem_optim is not None:
        for stem_idx, layer_idx in enumerate(stem_layers):
            src_stem_key = f"stem_embeddings.{stem_idx}.weight"
            dst_stem_key = f"_stem_embeddings.{layer_idx}.weight"
            if src_stem_key in stem_optim["state"]:
                for field_name, tensor in stem_optim["state"][src_stem_key].items():
                    if isinstance(tensor, torch.Tensor):
                        flat_key = f"state.{dst_stem_key}.{field_name}"
                        flattened[flat_key] = tensor
                logger.info(f"  Stem optim: {src_stem_key} -> {dst_stem_key}")

    # Build param_groups: every model parameter needs its optimizer hyperparams.
    # These must match exactly what the runtime AdamW produces via
    # get_optimizer_state_dict(flatten_optimizer_state_dict=True).
    src_param_groups = src_optim["param_groups"]
    defaults = src_param_groups[0] if src_param_groups else {}
    pg_defaults = {
        "lr": defaults.get("lr", 8e-4),
        "betas": tuple(defaults.get("betas", (0.9, 0.95))),
        "eps": defaults.get("eps", 1e-8),
        "weight_decay": defaults.get("weight_decay", 0.1),
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }

    for param_key in model_keys:
        for hp_name, hp_val in pg_defaults.items():
            flat_key = f"param_groups.{param_key}.{hp_name}"
            flattened[flat_key] = hp_val

    logger.info(
        f"Flattened optimizer state: {len(flattened)} keys "
        f"({len(flattened) - len(model_keys) * len(pg_defaults)} state + "
        f"{len(model_keys) * len(pg_defaults)} param_groups)"
    )
    return flattened


# ---------------------------------------------------------------------------
# Saving as DCP
# ---------------------------------------------------------------------------


def save_as_dcp(
    state_dict: dict[str, torch.Tensor],
    output_dir: Path,
):
    """Save state dict as a DCP checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29513")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    logger.info(f"Saving DCP checkpoint to {output_dir} ({len(state_dict)} keys)")
    dcp.save(state_dict, checkpoint_id=str(output_dir))
    logger.info("DCP checkpoint saved successfully")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert STEM-pt stem_dag checkpoint to torchtitan DCP format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="Path to STEM-pt checkpoint directory (containing consolidated/ and stem_shards/)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for torchtitan DCP checkpoint",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only convert model weights, skip optimizer states",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default=None,
        help="Cast weights to this dtype (default: keep original)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ckpt_path = Path(args.ckpt_path)
    output_dir = Path(args.output_dir)

    if not ckpt_path.exists():
        logger.error(f"Checkpoint path does not exist: {ckpt_path}")
        sys.exit(1)

    # ---- 1. Load params.json ----
    params_path = ckpt_path / "params.json"
    if not params_path.exists():
        params_path = ckpt_path / "consolidated" / "params.json"
    if not params_path.exists():
        logger.error("Cannot find params.json")
        sys.exit(1)

    with open(params_path) as f:
        params = json.load(f)

    model_params = params.get("model", params)
    stem_layers = model_params["stem_layers"]
    n_heads = model_params["n_heads"]
    dim = model_params["dim"]
    head_dim = model_params.get("head_dim", dim // n_heads)

    logger.info(f"Model: dim={dim}, n_heads={n_heads}, head_dim={head_dim}")
    logger.info(f"Stem layers: {stem_layers}")

    # ---- 2. Load checkpoint components ----
    checkpoint = load_consolidated(ckpt_path)
    src_model = checkpoint["model"]
    src_optim = checkpoint.get("optim")

    stem_embeddings = load_stem_shards(ckpt_path)
    stem_optim = None
    if not args.model_only:
        stem_optim = load_stem_optim_shards(ckpt_path)

    # ---- 3. Convert model state ----
    logger.info("Converting model state dict...")
    model_state = convert_model_state(
        src_model, stem_embeddings, stem_layers, n_heads, head_dim
    )

    # Optional dtype cast
    if args.dtype:
        target_dtype = getattr(torch, args.dtype)
        logger.info(f"Casting all weights to {target_dtype}")
        model_state = {k: v.to(target_dtype) for k, v in model_state.items()}

    logger.info(f"Converted model state: {len(model_state)} keys")

    # ---- 4. Convert optimizer state (optional) ----
    optim_flattened = None
    if not args.model_only and src_optim is not None:
        logger.info("Converting optimizer state...")
        optim_flattened = convert_optim_state_flattened(
            src_optim,
            stem_optim,
            stem_layers,
            model_keys=list(model_state.keys()),
            n_heads=n_heads,
            head_dim=head_dim,
        )

    # ---- 5. Save as DCP ----
    # torchtitan DCP layout: model keys at top level + "optimizer" dict with
    # flattened state keys like state.{param_name}.{field}
    dcp_state = dict(model_state)
    if optim_flattened is not None:
        dcp_state["optimizer"] = optim_flattened
    save_as_dcp(dcp_state, output_dir)

    # ---- 6. Save conversion metadata ----
    metadata = {
        "source": str(ckpt_path),
        "stem_layers": stem_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "dim": dim,
        "n_layers": model_params.get("n_layers", 16),
        "vocab_size": model_params.get("vocab_size", 100352),
        "ffn_hidden_dim": stem_embeddings[next(iter(stem_embeddings))].shape[1],
        "model_keys": len(model_state),
        "has_optimizer": optim_flattened is not None,
    }
    with open(output_dir / "conversion_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        f"Saved conversion metadata to {output_dir / 'conversion_metadata.json'}"
    )

    # ---- Done ----
    logger.info("")
    logger.info(f"Conversion complete! Output: {output_dir}")
    logger.info("To use in torchtitan:")
    logger.info(f"  --checkpoint.initial_load_path {output_dir}")
    if args.model_only:
        logger.info("  --checkpoint.initial_load_model_only")
    else:
        logger.info("  --checkpoint.no_initial_load_model_only")
        logger.info(
            "  --checkpoint.exclude_from_loading dataloader,lr_scheduler,train_state"
        )


if __name__ == "__main__":
    main()
