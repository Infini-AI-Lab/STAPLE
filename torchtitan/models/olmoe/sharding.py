# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Shard

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    set_decoder_sharding_config,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.olmoe.model import OLMoEModel, OLMoETransformerBlock


_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, Placement] = {
    "w1_EFD": Shard(1),
    "w2_EDF": Shard(2),
    "w3_EFD": Shard(1),
}


def set_olmoe_sharding_config(
    config: "OLMoEModel.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all OLMoE sub-configs."""
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        _set_olmoe_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_olmoe_layer_sharding(
    layer_cfg: "OLMoETransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one OLMoE transformer layer.

    OLMoE uses pre-norm with GQA (QK-norm per-head) + MoE FFN.
    """
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    attention = layer_cfg.attention

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # OLMoE QK-norm: per-head normalization on the 4D (bs, seq, n_heads, head_dim)
    # tensor. Under TP the heads are sharded on dim=2, so each rank normalizes
    # its local heads independently.
    if attention.qk_norm is not None:
        attention.qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=Shard(0))},
            in_src_shardings={"input": dense_activation_placement(tp=Shard(2))},
            in_dst_shardings={"input": dense_activation_placement(tp=Shard(2))},
            out_dst_shardings=dense_activation_placement(tp=Shard(2)),
        )

    # MoE FFN
    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )
