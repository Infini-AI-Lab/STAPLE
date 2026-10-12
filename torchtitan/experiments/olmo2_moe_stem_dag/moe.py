# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed experts with static or projected per-expert STEM signals."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.tensor import DTensor

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.token_dispatcher import (
    _all_to_all_single_maybe_autograd,
    AllToAllDispatchMetadata,
    LocalDispatchMetadata,
    LocalTokenDispatcher,
)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class StaticStemGroupedExperts(GroupedExperts):
    """Granular experts that consume one shard of a full-width STEM signal."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        granularity: int

    def __init__(self, config: Config):
        super().__init__(config)
        if config.granularity <= 0:
            raise ValueError(f"granularity must be positive, got {config.granularity}")
        if config.num_experts % config.granularity != 0:
            raise ValueError(
                f"num_experts ({config.num_experts}) must be divisible by "
                f"granularity ({config.granularity})"
            )
        if self.token_dispatcher.score_before_experts:
            raise ValueError(
                "StaticStemGroupedExperts requires score_before_experts=False"
            )
        self.granularity = config.granularity

    def _dispatch_stem_signal(
        self,
        stem_TGF: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        metadata: LocalDispatchMetadata,
    ) -> torch.Tensor:
        flat_expert_ids_N = topk_expert_ids_TK.reshape(-1)
        assignment_indices_experts_sorted_N = torch.argsort(
            flat_expert_ids_N,
            stable=True,
        )
        expert_ids_experts_sorted_N = flat_expert_ids_N[
            assignment_indices_experts_sorted_N
        ]
        stem_shard_indices_N = expert_ids_experts_sorted_N.remainder(self.granularity)
        stem_NF = stem_TGF[
            metadata.token_indices_experts_sorted_N,
            stem_shard_indices_N,
        ]
        if not isinstance(metadata, AllToAllDispatchMetadata):
            return stem_NF

        dispatcher = self.token_dispatcher
        ep_mesh = getattr(dispatcher, "ep_mesh", None)
        if ep_mesh is None:
            raise RuntimeError("all-to-all dispatch metadata requires an EP mesh")

        stem_RF = _all_to_all_single_maybe_autograd(
            stem_NF,
            metadata.output_splits,
            metadata.input_splits,
            ep_mesh,
        )
        return stem_RF[metadata.permuted_indices]

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        stem_RF: torch.Tensor,
    ) -> torch.Tensor:
        w1_EFD = _local_tensor(self.w1_EFD)
        w2_EDF = _local_tensor(self.w2_EDF)
        w3_EFD = _local_tensor(self.w3_EFD)

        if x_RD.shape[0] == 0:
            weight_dependency = (
                w1_EFD.reshape(-1)[0] + w2_EDF.reshape(-1)[0] + w3_EFD.reshape(-1)[0]
            ) * 0
            return x_RD + weight_dependency.to(x_RD.dtype) + stem_RF.sum() * 0

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        gate_RF = F.silu(
            torch._grouped_mm(
                x_RD.bfloat16(),
                w1_EFD.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            )
        )
        up_RF = torch._grouped_mm(
            x_RD.bfloat16(),
            w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        hidden_RF = gate_RF * (up_RF + stem_RF.bfloat16())
        return torch._grouped_mm(
            hidden_RF,
            w2_EDF.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(x_RD)

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        stem_BLF: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = x_BLD.shape
        if stem_BLF.shape[:2] != (B, L):
            raise ValueError(
                "STEM signal batch and sequence dimensions must match the MoE "
                f"input, got {tuple(stem_BLF.shape[:2])} and {(B, L)}"
            )
        hidden_dim = self.w1_EFD.shape[1]
        expected_stem_dim = self.granularity * hidden_dim
        if stem_BLF.shape[-1] != expected_stem_dim:
            raise ValueError(
                "STEM signal width must equal granularity times the expert FFN "
                f"width, got {stem_BLF.shape[-1]} and "
                f"{self.granularity} * {hidden_dim}"
            )
        K = topk_scores_BLK.size(-1)
        x_TD = x_BLD.view(B * L, D)
        stem_TGF = stem_BLF.view(B * L, self.granularity, hidden_dim)
        topk_scores_TK = topk_scores_BLK.view(B * L, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(B * L, K)

        dispatch_range = (
            torch.cuda.nvtx.range("moe.dispatch") if x_TD.is_cuda else nullcontext()
        )
        with dispatch_range:
            (
                routed_input_RD,
                tokens_per_expert_E,
                metadata,
            ) = self.token_dispatcher.dispatch(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
            )
            routed_stem_RF = self._dispatch_stem_signal(
                stem_TGF,
                topk_expert_ids_TK,
                metadata,
            )

        experts_range = (
            torch.cuda.nvtx.range("moe.experts") if x_TD.is_cuda else nullcontext()
        )
        with experts_range:
            routed_output_RD = self._experts_forward(
                routed_input_RD,
                tokens_per_expert_E,
                routed_stem_RF,
            )

        combine_range = (
            torch.cuda.nvtx.range("moe.combine") if x_TD.is_cuda else nullcontext()
        )
        with combine_range:
            return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)


class ProjectedStemGroupedExperts(GroupedExperts):
    """Experts with low-dimensional per-expert tables and one shared projection."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        vocab_size: int
        stem_dim: int

    def __init__(self, config: Config):
        super().__init__(config)
        if self.token_dispatcher.top_k != 1:
            raise ValueError(
                "ProjectedStemGroupedExperts only supports G=1, T=1; "
                f"got top_k={self.token_dispatcher.top_k}"
            )
        if config.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {config.vocab_size}")
        if config.stem_dim <= 0:
            raise ValueError(f"stem_dim must be positive, got {config.stem_dim}")
        if self.token_dispatcher.score_before_experts:
            raise ValueError(
                "ProjectedStemGroupedExperts requires score_before_experts=False"
            )
        self.vocab_size = config.vocab_size
        self.stem_dim = config.stem_dim
        self.stem_table_EVR = nn.Parameter(
            torch.empty(config.num_experts, config.vocab_size, config.stem_dim)
        )
        self.stem_up_FR = nn.Parameter(torch.empty(config.hidden_dim, config.stem_dim))

    def _dispatch_token_ids(
        self,
        token_ids_T: torch.Tensor,
        metadata: LocalDispatchMetadata,
    ) -> torch.Tensor:
        token_ids_N = token_ids_T[metadata.token_indices_experts_sorted_N]
        if not isinstance(metadata, AllToAllDispatchMetadata):
            return token_ids_N

        dispatcher = self.token_dispatcher
        ep_mesh = getattr(dispatcher, "ep_mesh", None)
        if ep_mesh is None:
            raise RuntimeError("all-to-all dispatch metadata requires an EP mesh")
        token_ids_R = all_to_all_single(
            token_ids_N,
            metadata.output_splits,
            metadata.input_splits,
            group=ep_mesh,
        )
        token_ids_R = torch.ops._c10d_functional.wait_tensor(token_ids_R)
        return token_ids_R[metadata.permuted_indices]

    def _project_stem(
        self,
        token_ids_R: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        stem_table_EVR = _local_tensor(self.stem_table_EVR)
        stem_up_FR = _local_tensor(self.stem_up_FR)
        num_local_experts, vocab_size, stem_dim = stem_table_EVR.shape
        local_expert_ids_R = torch.arange(
            num_local_experts,
            device=token_ids_R.device,
        ).repeat_interleave(num_tokens_per_expert_E.to(torch.long))
        flat_indices_R = local_expert_ids_R * vocab_size + token_ids_R.to(torch.long)
        stem_RR = stem_table_EVR.reshape(-1, stem_dim)[flat_indices_R]
        return F.linear(stem_RR.bfloat16(), stem_up_FR.bfloat16())

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        token_ids_R: torch.Tensor,
    ) -> torch.Tensor:
        w1_EFD = _local_tensor(self.w1_EFD)
        w2_EDF = _local_tensor(self.w2_EDF)
        w3_EFD = _local_tensor(self.w3_EFD)
        stem_table_EVR = _local_tensor(self.stem_table_EVR)
        stem_up_FR = _local_tensor(self.stem_up_FR)

        if x_RD.shape[0] == 0:
            weight_dependency = (
                w1_EFD.reshape(-1)[0]
                + w2_EDF.reshape(-1)[0]
                + w3_EFD.reshape(-1)[0]
                + stem_table_EVR.reshape(-1)[0]
                + stem_up_FR.reshape(-1)[0]
            ) * 0
            return x_RD + weight_dependency.to(x_RD.dtype)

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        gate_RF = F.silu(
            torch._grouped_mm(
                x_RD.bfloat16(),
                w1_EFD.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            )
        )
        up_RF = torch._grouped_mm(
            x_RD.bfloat16(),
            w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        stem_RF = self._project_stem(token_ids_R, num_tokens_per_expert_E)
        hidden_RF = gate_RF * (up_RF + stem_RF)
        return torch._grouped_mm(
            hidden_RF,
            w2_EDF.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(x_RD)

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        token_ids_BL: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = x_BLD.shape
        if token_ids_BL.shape != (B, L):
            raise ValueError(
                "token_ids_BL must match the MoE batch and sequence dimensions, "
                f"got {tuple(token_ids_BL.shape)} and {(B, L)}"
            )
        K = topk_scores_BLK.size(-1)
        x_TD = x_BLD.view(B * L, D)
        token_ids_T = token_ids_BL.reshape(B * L)
        topk_scores_TK = topk_scores_BLK.view(B * L, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(B * L, K)

        dispatch_range = (
            torch.cuda.nvtx.range("moe.dispatch") if x_TD.is_cuda else nullcontext()
        )
        with dispatch_range:
            (
                routed_input_RD,
                tokens_per_expert_E,
                metadata,
            ) = self.token_dispatcher.dispatch(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
            )
            routed_token_ids_R = self._dispatch_token_ids(token_ids_T, metadata)

        experts_range = (
            torch.cuda.nvtx.range("moe.experts") if x_TD.is_cuda else nullcontext()
        )
        with experts_range:
            routed_output_RD = self._experts_forward(
                routed_input_RD,
                tokens_per_expert_E,
                routed_token_ids_R,
            )

        combine_range = (
            torch.cuda.nvtx.range("moe.combine") if x_TD.is_cuda else nullcontext()
        )
        with combine_range:
            return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)


class StaticStemMoE(MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        experts: StaticStemGroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        shared_experts: FeedForward.Config | None = None

    def forward(self, x_BLD: torch.Tensor, stem_BLF: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        topk_scores_BLK, topk_expert_ids_BLK, tokens_per_expert_E = self.router(
            x_BLD, self.expert_bias_E
        )
        with torch.no_grad():
            self.tokens_per_expert_E.add_(tokens_per_expert_E)
        out_TD = self.experts(
            x_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            stem_BLF,
        )
        out_BLD = out_TD.view(B, L, -1)
        if self.shared_experts is not None:
            out_BLD = out_BLD + self.shared_experts(x_BLD)
        return out_BLD


class ProjectedStemMoE(MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        experts: ProjectedStemGroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        shared_experts: FeedForward.Config | None = None

    def forward(self, x_BLD: torch.Tensor, token_ids_BL: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        topk_scores_BLK, topk_expert_ids_BLK, tokens_per_expert_E = self.router(
            x_BLD, self.expert_bias_E
        )
        with torch.no_grad():
            self.tokens_per_expert_E.add_(tokens_per_expert_E)
        out_TD = self.experts(
            x_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            token_ids_BL,
        )
        out_BLD = out_TD.view(B, L, -1)
        if self.shared_experts is not None:
            out_BLD = out_BLD + self.shared_experts(x_BLD)
        return out_BLD


def make_static_stem_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    granularity: int,
    param_init: dict,
    token_dispatcher: LocalTokenDispatcher.Config,
) -> StaticStemGroupedExperts.Config:
    return StaticStemGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        granularity=granularity,
        param_init=param_init,
        token_dispatcher=token_dispatcher,
    )


def make_projected_stem_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    vocab_size: int,
    stem_dim: int,
    num_experts: int,
    param_init: dict,
    token_dispatcher: LocalTokenDispatcher.Config,
) -> ProjectedStemGroupedExperts.Config:
    return ProjectedStemGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        stem_dim=stem_dim,
        num_experts=num_experts,
        param_init=param_init,
        token_dispatcher=token_dispatcher,
    )
