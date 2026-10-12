# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Upcycle a TorchTitan OLMo2 DCP checkpoint into a granular MoE checkpoint."""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)

DEFAULT_MOE_LAYERS = tuple(range(1, 13))
MODEL_KEY_PREFIXES = (
    "_stem_embeddings.",
    "tok_embeddings.",
    "layers.",
    "norm.",
    "lm_head.",
)
HF_ASSET_FILENAMES = (
    "config.json",
    "generation_config.json",
    "params.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def expert_scale(expansion_rate: int, granularity: int, top_k: int) -> float:
    return (expansion_rate * granularity**2 / top_k) ** (1.0 / 3.0)


def upcycle_ffn_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    *,
    expansion_rate: int,
    granularity: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_dim, dim = w1.shape
    if w3.shape != (hidden_dim, dim) or w2.shape != (dim, hidden_dim):
        raise ValueError(
            f"Incompatible SwiGLU shapes: w1={w1.shape}, w2={w2.shape}, w3={w3.shape}."
        )
    if hidden_dim % granularity != 0:
        raise ValueError(
            f"FFN hidden dimension {hidden_dim} must divide granularity {granularity}."
        )
    if top_k != granularity:
        raise ValueError(
            "Virtual-group initialization requires top_k == granularity, got "
            f"top_k={top_k}, granularity={granularity}."
        )

    expert_hidden_dim = hidden_dim // granularity
    scale = expert_scale(expansion_rate, granularity, top_k)
    w1_shards = w1.reshape(granularity, expert_hidden_dim, dim)
    w3_shards = w3.reshape(granularity, expert_hidden_dim, dim)
    w2_shards = (
        w2.reshape(dim, granularity, expert_hidden_dim).permute(1, 0, 2).contiguous()
    )
    repeats = (expansion_rate, 1, 1)
    return (
        w1_shards.repeat(repeats).mul(scale),
        w2_shards.repeat(repeats).mul(scale),
        w3_shards.repeat(repeats).mul(scale),
    )


def virtual_group_router_weight(
    *,
    dim: int,
    expansion_rate: int,
    granularity: int,
    seed: int,
    std: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prototypes = torch.empty(expansion_rate, dim, dtype=torch.float32)
    prototypes.normal_(mean=0.0, std=std, generator=generator)
    return prototypes.repeat_interleave(granularity, dim=0).to(dtype)


def upcycle_model_state(
    dense_state: dict[str, torch.Tensor],
    *,
    moe_layers: tuple[int, ...] = DEFAULT_MOE_LAYERS,
    expansion_rate: int = 17,
    granularity: int = 8,
    top_k: int = 8,
    router_seed: int = 42,
    router_std: float = 1e-3,
    output_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    output = {
        key: value.to(output_dtype)
        for key, value in dense_state.items()
        if not any(
            key.startswith(f"layers.{layer}.feed_forward.") for layer in moe_layers
        )
    }

    for layer in moe_layers:
        prefix = f"layers.{layer}"
        w1_key = f"{prefix}.feed_forward.w1.weight"
        w2_key = f"{prefix}.feed_forward.w2.weight"
        w3_key = f"{prefix}.feed_forward.w3.weight"
        missing = [key for key in (w1_key, w2_key, w3_key) if key not in dense_state]
        if missing:
            raise KeyError(f"Missing dense FFN keys for layer {layer}: {missing}")

        w1, w2, w3 = upcycle_ffn_weights(
            dense_state[w1_key],
            dense_state[w2_key],
            dense_state[w3_key],
            expansion_rate=expansion_rate,
            granularity=granularity,
            top_k=top_k,
        )
        output[f"{prefix}.moe.experts.w1_EFD"] = w1.to(output_dtype)
        output[f"{prefix}.moe.experts.w2_EDF"] = w2.to(output_dtype)
        output[f"{prefix}.moe.experts.w3_EFD"] = w3.to(output_dtype)
        output[f"{prefix}.moe.router.gate.weight"] = virtual_group_router_weight(
            dim=dense_state[w1_key].shape[1],
            expansion_rate=expansion_rate,
            granularity=granularity,
            seed=router_seed + layer,
            std=router_std,
            dtype=output_dtype,
        )
    return output


def find_static_stem_layers(state: dict[str, torch.Tensor]) -> list[int]:
    prefix = "_stem_embeddings."
    suffix = ".weight"
    layer_ids = []
    for key in state:
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        layer_id = key[len(prefix) : -len(suffix)]
        if not layer_id.isdigit():
            raise ValueError(f"Invalid static STEM checkpoint key: {key}")
        layer_ids.append(int(layer_id))
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"Duplicate static STEM layers in checkpoint: {layer_ids}")
    return sorted(layer_ids)


def load_model_only_dcp(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    reader = dcp.FileSystemReader(str(checkpoint_path))
    metadata = reader.read_metadata()
    state: dict[str, torch.Tensor] = {}
    for key, tensor_metadata in metadata.state_dict_metadata.items():
        if not key.startswith(MODEL_KEY_PREFIXES):
            continue
        size = getattr(tensor_metadata, "size", None)
        properties = getattr(tensor_metadata, "properties", None)
        if size is None or properties is None:
            continue
        state[key] = torch.empty(size, dtype=properties.dtype)
    if not state:
        raise ValueError(f"No model tensors found in {checkpoint_path}.")
    dcp.load(state, storage_reader=reader)
    return state


def copy_assets(source: Path, output: Path) -> None:
    for filename in HF_ASSET_FILENAMES:
        path = source / filename
        if path.is_file():
            shutil.copy2(path, output / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--moe-layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_MOE_LAYERS),
    )
    parser.add_argument("--expansion-rate", type=int, default=17)
    parser.add_argument("--granularity", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--router-seed", type=int, default=42)
    parser.add_argument("--router-std", type=float, default=1e-3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.expansion_rate <= 0 or args.granularity <= 0 or args.top_k <= 0:
        raise ValueError("Expansion rate, granularity, and top-k must be positive.")
    if args.top_k != args.granularity:
        raise ValueError(
            "Virtual-group initialization requires top-k to equal granularity."
        )
    if len(set(args.moe_layers)) != len(args.moe_layers):
        raise ValueError(f"MoE layers must be unique, got {args.moe_layers}.")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dense_state = load_model_only_dcp(args.input_dir)
    logger.info("Loaded %d model tensors without optimizer state.", len(dense_state))
    dense_hidden_dim = dense_state[
        f"layers.{args.moe_layers[0]}.feed_forward.w1.weight"
    ].shape[0]
    stem_layers = find_static_stem_layers(dense_state)
    output_dtype = getattr(torch, args.dtype)
    moe_state = upcycle_model_state(
        dense_state,
        moe_layers=tuple(args.moe_layers),
        expansion_rate=args.expansion_rate,
        granularity=args.granularity,
        top_k=args.top_k,
        router_seed=args.router_seed,
        router_std=args.router_std,
        output_dtype=output_dtype,
    )
    del dense_state
    dcp.save(moe_state, checkpoint_id=str(args.output_dir))
    copy_assets(args.input_dir, args.output_dir)

    metadata = dcp.FileSystemReader(str(args.output_dir)).read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    if any(key.startswith("optimizer.") for key in checkpoint_keys):
        raise RuntimeError(
            "Converted checkpoint unexpectedly contains optimizer state."
        )

    conversion_metadata = {
        "source": str(args.input_dir),
        "model_keys": len(moe_state),
        "moe_layers": args.moe_layers,
        "stem_layers": stem_layers,
        "expansion_rate": args.expansion_rate,
        "granularity": args.granularity,
        "top_k": args.top_k,
        "num_experts": args.expansion_rate * args.granularity,
        "expert_hidden_dim": dense_hidden_dim // args.granularity,
        "expert_scale": expert_scale(
            args.expansion_rate,
            args.granularity,
            args.top_k,
        ),
        "router_seed": args.router_seed,
        "router_std": args.router_std,
        "dtype": args.dtype,
        "has_optimizer": False,
    }
    (args.output_dir / "conversion_metadata.json").write_text(
        json.dumps(conversion_metadata, indent=2) + "\n"
    )
    logger.info("Saved %d model tensors to %s.", len(moe_state), args.output_dir)


if __name__ == "__main__":
    main()
