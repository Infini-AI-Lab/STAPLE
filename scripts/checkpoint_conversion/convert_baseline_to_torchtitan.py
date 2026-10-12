# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert STEM-pt baseline OLMo2 checkpoint to torchtitan DCP format.

Handles:
1. Backbone weights: renames keys from STEM-pt naming to torchtitan naming
2. Optimizer states: converts Adam state with matching key renames
3. Reshapes q_norm/k_norm from [n_heads*head_dim] to [n_heads, head_dim]

The output is a DCP checkpoint directory that can be loaded directly by torchtitan's
CheckpointManager via `--checkpoint.initial_load_path`.

Source format (STEM-pt consolidated checkpoint):
  - consolidated/consolidated.pth: {model: {...}, optim: {...}}
  - Model keys use STEM-pt naming (no "lm_transformer." prefix for baseline):
      layers.{i}.attention.wq/wk/wv/wo.weight
      layers.{i}.attention.q_norm/k_norm.weight
      layers.{i}.feed_forward.w1/w2/w3.weight
      layers.{i}.post_attention_norm.weight
      layers.{i}.post_feedforward_norm.weight
      tok_embeddings.weight, output.weight, norm.weight

Target format (torchtitan DCP):
  - DCP directory with torchtitan key names:
      layers.{i}.attention.qkv_linear.wq/wk/wv.weight
      layers.{i}.attention.wo.weight
      layers.{i}.attention.q_norm/k_norm.weight  (reshaped to [n_heads, head_dim])
      layers.{i}.feed_forward.w1/w2/w3.weight
      layers.{i}.attention_norm.weight
      layers.{i}.ffn_norm.weight
      tok_embeddings.weight, lm_head.weight, norm.weight

Usage:
    python apps/main/convert_baseline_to_torchtitan.py \
        --ckpt-path ~/checkpoints/olmo2-1b-base-4T-extend100B \
        --output-dir ~/checkpoints/olmo2-1b-base-4T-torchtitan

    # With optimizer states for training resumption:
    python apps/main/convert_baseline_to_torchtitan.py \
        --ckpt-path ~/checkpoints/olmo2-1b-base-4T-extend100B \
        --output-dir ~/checkpoints/olmo2-1b-base-4T-torchtitan \
        --include-optimizer
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key mapping: STEM-pt baseline -> torchtitan
# ---------------------------------------------------------------------------

KEY_MAP = {
    "layers.{}.attention.wq.weight": "layers.{}.attention.qkv_linear.wq.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attention.qkv_linear.wk.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attention.qkv_linear.wv.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attention.wo.weight",
    "layers.{}.attention.q_norm.weight": "layers.{}.attention.q_norm.weight",
    "layers.{}.attention.k_norm.weight": "layers.{}.attention.k_norm.weight",
    "layers.{}.feed_forward.w1.weight": "layers.{}.feed_forward.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.feed_forward.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.feed_forward.w3.weight",
    "layers.{}.post_attention_norm.weight": "layers.{}.attention_norm.weight",
    "layers.{}.post_feedforward_norm.weight": "layers.{}.ffn_norm.weight",
    "tok_embeddings.weight": "tok_embeddings.weight",
    "output.weight": "lm_head.weight",
    "norm.weight": "norm.weight",
}

# Also handle the "lm_transformer." prefix variant (used in stem_dag checkpoints)
KEY_MAP_WITH_PREFIX = {f"lm_transformer.{k}": v for k, v in KEY_MAP.items()}

HF_ASSET_FILENAMES = (
    "config.json",
    "generation_config.json",
    "params.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def rename_key(src_key: str) -> str | None:
    """Convert a STEM-pt key to torchtitan format. Returns None if key should be skipped."""
    # Try with lm_transformer prefix first
    if src_key.startswith("lm_transformer."):
        src_key_stripped = src_key[len("lm_transformer.") :]
    else:
        src_key_stripped = src_key

    if "layers." in src_key_stripped:
        match = re.search(r"layers\.(\d+)\.", src_key_stripped)
        if not match:
            return None
        layer_num = match.group(1)
        abstract_key = re.sub(r"layers\.\d+\.", "layers.{}.", src_key_stripped)
        target_pattern = KEY_MAP.get(abstract_key)
        if target_pattern is None:
            return None
        return target_pattern.format(layer_num)
    else:
        return KEY_MAP.get(src_key_stripped)


def expected_model_keys(n_layers: int) -> set[str]:
    """Return the complete TorchTitan model key set for a baseline checkpoint."""
    expected = set()
    for target_pattern in KEY_MAP.values():
        if "{}" in target_pattern:
            expected.update(target_pattern.format(layer) for layer in range(n_layers))
        else:
            expected.add(target_pattern)
    return expected


def validate_converted_model_state(
    model_state: dict[str, torch.Tensor], n_layers: int
) -> None:
    """Require conversion to produce exactly the expected model namespace."""
    expected = expected_model_keys(n_layers)
    actual = set(model_state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "Converted model state is incomplete or has unexpected keys. "
            f"Missing ({len(missing)}): {missing[:10]}; "
            f"unexpected ({len(unexpected)}): {unexpected[:10]}"
        )


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


def load_params(ckpt_path: Path) -> dict[str, Any]:
    """Load params.json from checkpoint directory."""
    candidates = [
        ckpt_path / "params.json",
        ckpt_path / "consolidated" / "params.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"params.json not found. Tried: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_model_state(
    src_model: dict[str, torch.Tensor],
    n_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Convert baseline model state dict to torchtitan format."""
    target = {}
    skipped = []

    for src_key, tensor in src_model.items():
        dst_key = rename_key(src_key)
        if dst_key is None:
            skipped.append(src_key)
            continue

        # Reshape q_norm/k_norm from [n_heads * head_dim] to [n_heads, head_dim]
        if dst_key.endswith("q_norm.weight") or dst_key.endswith("k_norm.weight"):
            if tensor.dim() == 1 and tensor.shape[0] == n_heads * head_dim:
                tensor = tensor.view(n_heads, head_dim)

        target[dst_key] = tensor

    if skipped:
        logger.warning(f"Skipped {len(skipped)} source keys: {skipped}")

    return target


def convert_optim_state_flattened(
    src_optim: dict[str, Any],
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
            "from the training run. Skipping optimizer conversion. "
            "Consider resuming training with fresh optimizer states."
        )
        return None

    logger.info(f"Optimizer state uses string keys ({len(src_state)} entries)")

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
                        if tensor.shape[0] == n_heads * head_dim:
                            tensor = tensor.view(n_heads, head_dim)
                flat_key = f"state.{dst_key}.{field_name}"
                flattened[flat_key] = tensor

    if skipped:
        logger.warning(f"Skipped {len(skipped)} optim state keys: {skipped}")

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
    os.environ.setdefault("MASTER_PORT", "29514")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    logger.info(f"Saving DCP checkpoint to {output_dir} ({len(state_dict)} keys)")
    dcp.save(state_dict, checkpoint_id=str(output_dir))
    logger.info("DCP checkpoint saved successfully")

    dist.destroy_process_group()


def validate_dcp_output(
    output_dir: Path,
    model_keys: set[str],
    *,
    expect_optimizer: bool,
) -> None:
    """Validate the persisted DCP namespace before declaring conversion complete."""
    metadata = dcp.FileSystemReader(str(output_dir)).read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    missing_model_keys = sorted(model_keys - checkpoint_keys)
    if missing_model_keys:
        raise RuntimeError(
            f"Saved DCP checkpoint is missing {len(missing_model_keys)} model keys: "
            f"{missing_model_keys[:10]}"
        )

    has_optimizer = any(key.startswith("optimizer.") for key in checkpoint_keys)
    if expect_optimizer and not has_optimizer:
        raise RuntimeError("Saved DCP checkpoint is missing optimizer state.")


def copy_hf_assets(ckpt_path: Path, output_dir: Path) -> None:
    """Copy inference assets without touching DCP's reserved metadata files."""
    for filename in HF_ASSET_FILENAMES:
        source = ckpt_path / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert STEM-pt baseline OLMo2 checkpoint to torchtitan DCP format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="Path to STEM-pt checkpoint directory (containing consolidated/)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for torchtitan DCP checkpoint",
    )
    parser.add_argument(
        "--include-optimizer",
        action="store_true",
        help="Also convert optimizer states for training resumption",
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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty to avoid mixing checkpoint files: "
            f"{output_dir}"
        )

    # ---- 1. Load params.json ----
    params = load_params(ckpt_path)
    model_params = params.get("model", params)
    n_heads = model_params["n_heads"]
    dim = model_params["dim"]
    head_dim = model_params.get("head_dim", dim // n_heads)
    n_layers = model_params.get("n_layers", 16)

    logger.info(
        f"Model: dim={dim}, n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}"
    )

    # ---- 2. Load checkpoint ----
    checkpoint = load_consolidated(ckpt_path)
    src_model = checkpoint["model"]
    src_optim = checkpoint.get("optim")

    logger.info(f"Source model: {len(src_model)} keys")

    # ---- 3. Convert model state ----
    logger.info("Converting model state dict...")
    model_state = convert_model_state(src_model, n_heads, head_dim)
    del src_model
    validate_converted_model_state(model_state, n_layers)

    if args.dtype:
        target_dtype = getattr(torch, args.dtype)
        logger.info(f"Casting all weights to {target_dtype}")
        model_state = {k: v.to(target_dtype) for k, v in model_state.items()}

    logger.info(f"Converted model state: {len(model_state)} keys")

    # ---- 4. Convert optimizer state (optional) ----
    optim_flattened = None
    if args.include_optimizer and src_optim is not None:
        logger.info("Converting optimizer state...")
        optim_flattened = convert_optim_state_flattened(
            src_optim,
            model_keys=list(model_state.keys()),
            n_heads=n_heads,
            head_dim=head_dim,
        )
    del src_optim, checkpoint

    # ---- 5. Save as DCP ----
    # torchtitan DCP layout: model keys at top level + "optimizer" dict with
    # flattened state keys like state.{param_name}.{field}
    dcp_state = dict(model_state)
    if optim_flattened is not None:
        dcp_state["optimizer"] = optim_flattened
    save_as_dcp(dcp_state, output_dir)
    copy_hf_assets(ckpt_path, output_dir)
    validate_dcp_output(
        output_dir,
        set(model_state),
        expect_optimizer=optim_flattened is not None,
    )

    # ---- 6. Save conversion metadata ----
    metadata = {
        "source": str(ckpt_path),
        "dim": dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "vocab_size": model_params.get("vocab_size", 100352),
        "model_keys": len(model_state),
        "has_optimizer": optim_flattened is not None,
    }
    with open(output_dir / "conversion_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- Done ----
    logger.info("")
    logger.info(f"Conversion complete! Output: {output_dir}")
    logger.info("To use in torchtitan:")
    logger.info(f"  --checkpoint.initial_load_path {output_dir}")
    if not args.include_optimizer:
        logger.info("  --checkpoint.initial_load_model_only")
    else:
        logger.info("  --checkpoint.no_initial_load_model_only")
        logger.info("  --checkpoint.exclude_from_loading dataloader,lr_scheduler")


if __name__ == "__main__":
    main()
