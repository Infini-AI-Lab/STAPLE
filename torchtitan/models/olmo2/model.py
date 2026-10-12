# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, VarlenAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops


class OLMo2TransformerBlock(TransformerBlock):
    """OLMo2 TransformerBlock with post-norm architecture.

    Unlike Llama3's pre-norm (norm before sublayer), OLMo2 applies
    normalization after the sublayer output, before the residual addition:
        x = x + norm(attention(x))
        x = x + norm(ffn(x))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        # Post-norms: applied after sublayer, before residual
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention_norm(
            self.attention(x, freqs_cis, attention_masks, positions)
        )
        out = h + self.ffn_norm(self.feed_forward(h))
        return out


class OLMo2Model(Decoder):
    """OLMo2 model."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 100352

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    "Varlen attention is not supported with CP."
                )

            if parallelism.tensor_parallel_degree > 1:
                # OLMo2's QK-norm normalizes across ALL heads jointly.
                # Under TP, heads are sharded across ranks, so each rank
                # would normalize only its local heads — not mathematically
                # equivalent. Requires all-gather before norm (future work).
                raise NotImplementedError(
                    "OLMo2's cross-head QK-norm is not yet supported with "
                    "tensor_parallel_degree > 1. Use TP=1 with FSDP sharding."
                )

            from torchtitan.models.olmo2.sharding import set_olmo2_sharding_config

            set_olmo2_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len=seq_len,
                enable_weight_tying=False,
            )

    def __init__(self, config: Config):
        super().__init__(config)
