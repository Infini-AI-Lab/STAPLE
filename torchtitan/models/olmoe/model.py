# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, VarlenAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.utils import get_moe_model_nparams_and_flops

from .moe_aux_loss import (
    clear_aux_losses,
    init_aux_loss_buffers,
    record_load_balancing_data,
    record_router_z_loss,
)


class OLMoERouter(TokenChoiceTopKRouter):
    """OLMoE router that records aux loss data during training.

    Extends TokenChoiceTopKRouter to capture:
    - Full router softmax probabilities (for load-balancing loss)
    - Raw router logits (for z-loss)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        aux_loss_layer_idx: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.aux_loss_layer_idx = config.aux_loss_layer_idx

    def forward(
        self, x_BLD: torch.Tensor, expert_bias_E: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
            scores_BLE = self.gate(x_BLD)

        # Record raw logits for z-loss before softmax
        if self.training:
            record_router_z_loss(
                scores_BLE.view(-1, self.num_experts),
                self.aux_loss_layer_idx,
            )

        if self.score_func == "sigmoid":
            scores_BLE = torch.sigmoid(scores_BLE)
        elif self.score_func == "softmax":
            scores_BLE = F.softmax(scores_BLE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice_BLE = (
            scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_BLE = self._get_node_limited_routing_scores(
                scores_for_choice_BLE
            )
        _, topk_expert_ids_BLK = torch.topk(
            scores_for_choice_BLE, k=self.top_k, dim=-1, sorted=False
        )

        topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

        if self._debug_force_load_balance:
            (
                topk_expert_ids_BLK,
                topk_scores_BLK,
            ) = self._debug_force_load_balance_routing(scores_BLE)

        if self.route_norm:
            denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_BLK = topk_scores_BLK / denominator
        topk_scores_BLK = topk_scores_BLK * self.route_scale

        num_tokens_per_expert_E = torch.histc(
            topk_expert_ids_BLK.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Record data for load-balancing loss
        if self.training:
            record_load_balancing_data(
                num_tokens_per_expert_E.detach(),
                scores_BLE.view(-1, self.num_experts),
                self.num_experts,
                self.top_k,
                self.aux_loss_layer_idx,
            )

        return topk_scores_BLK, topk_expert_ids_BLK, num_tokens_per_expert_E


class OLMoETransformerBlock(TransformerBlock):
    """OLMoE TransformerBlock: pre-norm with MoE FFN.

    Architecture:
        x = x + attention(attention_norm(x))
        x = x + moe(ffn_norm(x))

    OLMoE uses QK-norm (per-head RMSNorm on Q and K before RoPE),
    pre-norm architecture, and top-k=8 MoE with 64 experts per layer.
    """

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
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class OLMoEModel(Decoder):
    """OLMoE-1B-7B: 64 experts, top-8, QK-norm, pre-norm architecture."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 50304

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

            from torchtitan.models.olmoe.sharding import set_olmoe_sharding_config

            set_olmoe_sharding_config(
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

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        n_layers = len(self.config.layers)
        assert buffer_device is not None
        init_aux_loss_buffers(n_layers, buffer_device)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.training:
            clear_aux_losses()
        return super().forward(*args, **kwargs)
