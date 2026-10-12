# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding additions for OLMo2 MoE STEM inputs and parameters."""

from torch.distributed.tensor import Replicate

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe_sharding import (
    expert_param_placement_dense,
    expert_param_placement_sparse,
)
from torchtitan.protocols.sharding import LocalMapConfig


def set_static_stem_input_sharding(config, *, stem_layers: set[int]) -> None:
    """Give the full static STEM signal the same placements as the MoE input."""
    for layer_id, layer_config in enumerate(config.layers):
        if layer_id not in stem_layers:
            continue
        if layer_config.moe is None:
            continue

        moe_sharding = layer_config.moe.sharding_config
        experts_sharding = layer_config.moe.experts.sharding_config
        if moe_sharding is None or experts_sharding is None:
            raise RuntimeError("OLMo2 MoE sharding must be configured first")

        moe_src = moe_sharding.in_src_shardings or {}
        moe_dst = moe_sharding.in_dst_shardings or {}
        moe_sharding.in_src_shardings = {
            **moe_src,
            "stem_BLF": moe_src["x_BLD"],
        }
        moe_sharding.in_dst_shardings = {
            **moe_dst,
            "stem_BLF": moe_dst["x_BLD"],
        }

        assert experts_sharding.in_dst_shardings is not None
        experts_input_sharding = experts_sharding.in_dst_shardings["x_BLD"]
        experts_sharding.in_dst_shardings["stem_BLF"] = experts_input_sharding
        assert experts_sharding.local_map is not None
        grad_placements = experts_sharding.local_map.in_grad_placements
        experts_sharding.local_map = LocalMapConfig(
            in_grad_placements=(*grad_placements, grad_placements[0])
        )


def set_projected_stem_sharding(
    config,
    *,
    stem_layers: set[int],
    enable_ep: bool,
) -> None:
    """Shard expert tables on EP while keeping the shared projection dense."""
    for layer_id, layer_config in enumerate(config.layers):
        if layer_id not in stem_layers:
            continue
        if layer_config.moe is None:
            raise ValueError(f"STEM layer {layer_id} must be an MoE layer")

        moe_sharding = layer_config.moe.sharding_config
        experts_sharding = layer_config.moe.experts.sharding_config
        if moe_sharding is None or experts_sharding is None:
            raise RuntimeError("OLMo2 MoE sharding must be configured first")

        moe_src = moe_sharding.in_src_shardings or {}
        moe_dst = moe_sharding.in_dst_shardings or {}
        moe_sharding.in_src_shardings = {
            **moe_src,
            "token_ids_BL": moe_src["x_BLD"],
        }
        moe_sharding.in_dst_shardings = {
            **moe_dst,
            "token_ids_BL": moe_dst["x_BLD"],
        }

        experts_sharding.state_shardings["stem_table_EVR"] = (
            expert_param_placement_sparse()
            if enable_ep
            else expert_param_placement_dense(tp_placement=Replicate())
        )
        experts_sharding.state_shardings["stem_up_FR"] = dense_param_placement(
            tp=Replicate()
        )
        assert experts_sharding.in_dst_shardings is not None
        experts_input_sharding = experts_sharding.in_dst_shardings["x_BLD"]
        experts_sharding.in_dst_shardings["token_ids_BL"] = experts_input_sharding
        assert experts_sharding.local_map is not None
        grad_placements = experts_sharding.local_map.in_grad_placements
        experts_sharding.local_map = LocalMapConfig(
            in_grad_placements=(*grad_placements, grad_placements[0])
        )


__all__ = [
    "set_projected_stem_sharding",
    "set_static_stem_input_sharding",
]
