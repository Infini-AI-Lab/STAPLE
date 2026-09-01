# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.olmo2.model import OLMo2Model, OLMo2TransformerBlock


def set_olmo2_sharding_config(
    config: "OLMo2Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill sharding_config on all OLMo2 sub-configs."""
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        _set_olmo2_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_olmo2_layer_sharding(
    layer_cfg: "OLMo2TransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """Set sharding on one OLMo2 transformer layer.

    OLMo2 uses post-norm: norm is applied to attention/FFN output before the
    residual add. In TP with SP, the post-norms see the attention/FFN output
    in its reduced form (Replicate or Shard(1) depending on SP), same as
    the pre-norm case.
    """
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    attention = layer_cfg.attention
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # OLMo2 QK-norm normalizes across ALL heads (last 2 dims of the 4D tensor).
    # Under TP the heads are sharded on dim=2, so each shard normalizes its
    # local heads independently — same placement as Qwen3's per-head norm.
    if attention.qk_norm is not None:
        attention.qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=Shard(0))},
            in_src_shardings={"input": dense_activation_placement(tp=Shard(2))},
            in_dst_shardings={"input": dense_activation_placement(tp=Shard(2))},
            out_dst_shardings=dense_activation_placement(tp=Shard(2)),
        )

    assert layer_cfg.feed_forward is not None
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_placement=attn_x_placement,
        enable_sp=enable_sp,
    )
