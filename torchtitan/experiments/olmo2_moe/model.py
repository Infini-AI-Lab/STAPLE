# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMo2 post-norm transformer blocks with optional routed MoE FFNs."""

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, VarlenAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.olmoe.moe_aux_loss import clear_aux_losses, init_aux_loss_buffers
from torchtitan.models.utils import get_moe_model_nparams_and_flops


class OLMo2MoETransformerBlock(TransformerBlock):
    """OLMo2 post-norm block supporting either a dense or routed FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.attention_norm(
            self.attention(x, freqs_cis, attention_masks, positions)
        )
        ffn_out = self.moe(h) if self.moe_enabled else self.feed_forward(h)
        return h + self.ffn_norm(ffn_out)


class OLMo2MoEModel(Decoder):
    """Mixed dense/MoE OLMo2 model retaining the dense post-norm architecture."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 100352

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention."
                )
            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "OLMo2 cross-head QK-norm is not supported with "
                    "tensor_parallel_degree > 1."
                )

            from .sharding import set_olmo2_moe_sharding_config

            set_olmo2_moe_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_moe_layers = sum(layer.moe is not None for layer in config.layers)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        assert buffer_device is not None
        init_aux_loss_buffers(self.num_moe_layers, buffer_device)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.training:
            clear_aux_losses()
        return super().forward(*args, **kwargs)
