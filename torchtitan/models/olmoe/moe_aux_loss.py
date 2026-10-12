# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMoE auxiliary losses: load-balancing loss and router z-loss.

These losses are computed eagerly per-layer inside the router forward and
accumulated into pre-allocated tensor buffers (one scalar per layer). This
avoids mutating Python lists which breaks torch.compile fullgraph tracing.

Load-balancing loss (Switch Transformer, arXiv:2101.03961):
    L_lb = num_experts * sum_e(f_e * P_e)
    where f_e = fraction of tokens routed to expert e,
          P_e = mean router probability for expert e.
    Scaled by moe_loss_weight (default 0.01).

Router z-loss (ST-MoE, arXiv:2202.08906):
    L_z = mean(logsumexp(logits, dim=-1)^2)
    Penalizes large router logits for training stability.
    Scaled by moe_zloss_weight (default 0.001).
"""

from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, cross_entropy_loss, LossFunction
from torchtitan.config import CompileConfig


# Pre-allocated buffers: one scalar per MoE layer.
# init_aux_loss_buffers() must be called once after model construction to
# allocate the right size. clear_aux_losses() zeros them each forward.
_lb_loss_buf: torch.Tensor | None = None
_rz_loss_buf: torch.Tensor | None = None


def init_aux_loss_buffers(n_layers: int, device: torch.device) -> None:
    """Allocate fixed-size aux loss buffers. Call once after model init."""
    global _lb_loss_buf, _rz_loss_buf
    _lb_loss_buf = torch.zeros(n_layers, dtype=torch.float32, device=device)
    _rz_loss_buf = torch.zeros(n_layers, dtype=torch.float32, device=device)


def clear_aux_losses() -> None:
    """Detach and zero the buffers before building the next forward graph."""
    if _lb_loss_buf is not None:
        _lb_loss_buf.detach_()
        _lb_loss_buf.zero_()
    if _rz_loss_buf is not None:
        _rz_loss_buf.detach_()
        _rz_loss_buf.zero_()


def record_load_balancing_data(
    num_tokens_per_expert_E: torch.Tensor,
    router_probs_TE: torch.Tensor,
    num_experts: int,
    top_k: int,
    aux_loss_layer_idx: int,
) -> None:
    """Compute and store per-layer load-balancing loss into the buffer.

    Called second in the router forward (after record_router_z_loss).

    Args:
        num_tokens_per_expert_E: Token counts per expert, shape [E].
        router_probs_TE: Full softmax router probabilities, shape [T, E].
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        aux_loss_layer_idx: Static index of this MoE layer in the loss buffer.
    """
    if _lb_loss_buf is None:
        return

    assert 0 <= aux_loss_layer_idx < _lb_loss_buf.shape[0]
    n_tokens = router_probs_TE.shape[0]
    if n_tokens > 0:
        f_e = num_tokens_per_expert_E.float() / (n_tokens * top_k)
        p_e = router_probs_TE.float().mean(dim=0)
        _lb_loss_buf[aux_loss_layer_idx] = num_experts * torch.dot(f_e, p_e)


def record_router_z_loss(logits_TE: torch.Tensor, aux_loss_layer_idx: int) -> None:
    """Compute and store per-layer router z-loss into the buffer.

    Called first in the router forward (before record_load_balancing_data).

    Args:
        logits_TE: Raw router logits (before softmax), shape [T, E].
        aux_loss_layer_idx: Static index of this MoE layer in the loss buffer.
    """
    if _rz_loss_buf is None:
        return

    assert 0 <= aux_loss_layer_idx < _rz_loss_buf.shape[0]
    _rz_loss_buf[aux_loss_layer_idx] = (torch.logsumexp(logits_TE, dim=-1) ** 2).mean()


def get_load_balancing_loss() -> torch.Tensor:
    """Return mean load-balancing loss across all layers."""
    if _lb_loss_buf is None:
        return torch.zeros((), dtype=torch.float32)
    return _lb_loss_buf.mean()


def get_router_z_loss() -> torch.Tensor:
    """Return mean router z-loss across all layers."""
    if _rz_loss_buf is None:
        return torch.zeros((), dtype=torch.float32)
    return _rz_loss_buf.mean()


class OLMoELoss(BaseLoss):
    """Cross-entropy + OLMoE auxiliary losses (load-balancing + router z-loss).

    This loss class wraps the standard cross-entropy loss and adds the
    traditional MoE auxiliary losses that OLMoE uses for training.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        moe_loss_weight: float = 0.01
        moe_zloss_weight: float = 0.001
        num_experts: int = 64
        top_k: int = 8

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)
        self.moe_loss_weight = config.moe_loss_weight
        self.moe_zloss_weight = config.moe_zloss_weight
        self.num_experts = config.num_experts
        self.top_k = config.top_k

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ce_loss = self.fn(pred, labels)
        if global_valid_tokens is not None:
            ce_loss = ce_loss / global_valid_tokens

        lb_loss = get_load_balancing_loss()
        z_loss = get_router_z_loss()

        if not hasattr(self, "_accumulated_ce_losses"):
            self._accumulated_ce_losses: list[torch.Tensor] = []
        self._accumulated_ce_losses.append(ce_loss.detach())
        self._last_lb_loss = lb_loss.detach().item()
        self._last_z_loss = z_loss.detach().item()

        total = ce_loss
        if self.moe_loss_weight > 0.0:
            total = total + self.moe_loss_weight * lb_loss
        if self.moe_zloss_weight > 0.0:
            total = total + self.moe_zloss_weight * z_loss

        return total

    def get_logging_loss(self) -> torch.Tensor | None:
        """Return accumulated CE-only loss for logging (excludes aux losses).

        Sums across microbatches and resets the accumulator.
        """
        losses = getattr(self, "_accumulated_ce_losses", None)
        if not losses:
            return None
        total = torch.sum(torch.stack(losses))
        self._accumulated_ce_losses = []
        return total

    def get_aux_metrics(self) -> dict[str, float]:
        """Return aux loss values for logging (wandb/tensorboard)."""
        return {
            "loss_metrics/load_balancing_loss": getattr(self, "_last_lb_loss", 0.0),
            "loss_metrics/router_z_loss": getattr(self, "_last_z_loss", 0.0),
        }
