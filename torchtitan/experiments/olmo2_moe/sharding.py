# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configuration for mixed dense/MoE OLMo2 models."""

from torchtitan.models.common.decoder_sharding import set_decoder_sharding_config
from torchtitan.models.olmo2.sharding import _set_olmo2_layer_sharding
from torchtitan.models.olmoe.sharding import _set_olmoe_layer_sharding


def set_olmo2_moe_sharding_config(
    config,
    *,
    loss_parallel: bool,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        if layer_cfg.moe is None:
            _set_olmo2_layer_sharding(layer_cfg, enable_sp=enable_sp)
        else:
            _set_olmoe_layer_sharding(
                layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
            )
