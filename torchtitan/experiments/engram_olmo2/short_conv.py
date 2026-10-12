# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Short depthwise causal convolution used inside Engram (paper Eq. 5).

After context-aware gating produces the value sequence ``ṽ``, Engram expands
the receptive field and adds non-linearity with a short, depthwise *causal*
convolution::

    Y = SiLU(Conv1D(RMSNorm(ṽ))) + ṽ

The convolution uses kernel size ``w`` (default 4) and dilation ``δ`` set to
the max N-gram order. The conv weight is zero-initialized so that at the start
of training ``Conv1D(...) = 0`` and ``Y = ṽ`` (identity), preserving the
backbone's behavior (paper Section 4.1).

Unlike the multi-branch (mHC) reference, this single-stream variant operates on
``[B, T, D]`` directly (no per-branch group axis).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.nn_modules import RMSNorm, SiLU
from torchtitan.protocols.module import Module

__all__ = ["ShortConv", "DepthwiseConv1d"]


class DepthwiseConv1d(nn.Conv1d, Module):
    """Configurable depthwise (groups=channels) causal ``nn.Conv1d``.

    Defaults to a zero-initialized weight so the surrounding Engram residual is
    an identity at the start of training.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        channels: int
        kernel_size: int
        dilation: int = 1
        bias: bool = False

    def __init__(self, config: Config):
        # Causal left-padding: (kernel_size - 1) * dilation taps of history.
        super().__init__(
            config.channels,
            config.channels,
            config.kernel_size,
            groups=config.channels,
            bias=config.bias,
            padding=(config.kernel_size - 1) * config.dilation,
            dilation=config.dilation,
        )


class ShortConv(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        kernel_size: int = 4
        dilation: int = 1
        norm_eps: float = 1e-5
        activation: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.activation = config.activation
        self.left_pad = (config.kernel_size - 1) * config.dilation

        # Zero-init the conv -> Engram starts as an identity (Y = ṽ).
        self.conv = DepthwiseConv1d.Config(
            channels=config.dim,
            kernel_size=config.kernel_size,
            dilation=config.dilation,
            param_init={"weight": nn.init.zeros_},
        ).build()
        self.norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
            param_init={"weight": nn.init.ones_},
        ).build()
        if self.activation:
            self.act_fn = SiLU.Config().build()

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x``: [B, T, D] -> [B, T, D] (causal).

        When ``positions`` (``[B, T]`` per-token within-document offsets that
        reset to 0 at each packed-document start) is given, the dilated causal
        receptive field is prevented from crossing document boundaries, so each
        packed document is convolved exactly as if processed alone. When
        ``None``, the plain (fast) ``nn.Conv1d`` path runs -- equivalent to
        ``positions`` being a plain ``arange`` (a single document).
        """
        T = x.shape[1]
        x_norm = self.norm(x)
        if positions is None:
            x_bct = x_norm.transpose(1, 2)  # [B, D, T]
            y_bct = self.conv(x_bct)
            # Left-padding adds ``left_pad`` taps on the right too; trim to T
            # for causality.
            y_bct = y_bct[..., :T]
        else:
            y_bct = self._masked_conv(x_norm, positions)  # [B, D, T]
        if self.activation:
            y_bct = self.act_fn(y_bct)
        return y_bct.transpose(1, 2).contiguous()  # [B, T, D]

    def _masked_conv(
        self, x_norm: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Depthwise dilated causal conv that respects document boundaries.

        Computes the same depthwise convolution as ``self.conv`` but masks each
        kernel tap so it never reads across a packed-document boundary. Tap
        ``j`` reads the token ``j * dilation`` steps back (weight column
        ``K-1-j``); that token is in the same document iff
        ``positions[t] >= j * dilation``. Returns ``[B, D, T]`` to match the
        ``nn.Conv1d`` layout used by the fast path.

        With ``positions`` a plain ``arange`` this zeros exactly the taps that
        the fast path's left-padding already zeros, so the two paths agree.
        """
        B, T, D = x_norm.shape
        x_bdt = x_norm.transpose(1, 2)  # [B, D, T]
        # conv weight: [D, 1, K] (depthwise, groups=D). weight[:, 0, K-1-j]
        # multiplies the tap at delay j*dilation.
        weight = self.conv.weight  # [D, 1, K]
        dilation = self.dilation

        # Compare positions in integer space (seq_len can exceed bf16's exact
        # integer range; the tap delays themselves are small).
        pos = positions.to(device=x_bdt.device, dtype=torch.int64)  # [B, T]
        acc = x_bdt.new_zeros(B, D, T)
        for j in range(self.kernel_size):
            delay = j * dilation
            w = weight[:, 0, self.kernel_size - 1 - j].view(1, D, 1)  # [1, D, 1]
            if delay == 0:
                acc = acc + x_bdt * w
                continue
            # Shift x right by ``delay`` along time (history), zero-filling the
            # start. Tap is valid only where the source token is in the same
            # document: positions[t] >= delay.
            shifted = F.pad(x_bdt, (delay, 0))[..., :T]  # [B, D, T]
            valid = (pos >= delay).unsqueeze(1)  # [B, 1, T]
            acc = acc + torch.where(valid, shifted, acc.new_zeros(())) * w
        if self.conv.bias is not None:
            acc = acc + self.conv.bias.view(1, D, 1)
        return acc

    @property
    def history_len(self) -> int:
        """Number of past inputs the causal receptive field reads (``left_pad``).

        A KV-cached decoder must keep this many past ``ṽ`` activations per
        sequence to produce the next token's conv output.
        """
        return self.left_pad

    def decode_step(
        self,
        v_t: torch.Tensor,
        v_hist: torch.Tensor,
        tok_id: torch.Tensor,
    ) -> torch.Tensor:
        """One causal-conv step for KV-cached decoding -- the conv contribution
        ``SiLU(Conv1D(RMSNorm(·)))`` at the current token only.

        Mirrors :meth:`_masked_conv` at a single position so that, fed the same
        ``ṽ`` history, it reproduces the training/prefill output exactly. The
        caller adds the residual ``ṽ_t`` (matching ``value + short_conv(value)``
        in :meth:`EngramLayer.forward`).

        Args:
            v_t: ``[B, D]`` current gated value ``ṽ_t`` (raw, pre-norm).
            v_hist: ``[B, P, D]`` previous ``P = history_len`` raw ``ṽ``,
                oldest first, front-zero-padded before each document's start.
            tok_id: ``[B]`` current within-document offset (``positions[t]``);
                taps reaching before the document start (``tok_id < j*dilation``)
                are masked, exactly as in the packed forward.

        Returns:
            ``[B, D]`` the conv contribution (already SiLU-activated when
            ``activation`` is set); the caller adds ``ṽ_t``.
        """
        P = self.left_pad
        seq = torch.cat([v_hist, v_t.unsqueeze(1)], dim=1)  # [B, P+1, D]
        seq_norm = self.norm(seq)  # RMSNorm is per-position
        weight = self.conv.weight  # [D, 1, K]
        D = v_t.shape[-1]
        tok = tok_id.to(torch.int64)
        out = v_t.new_zeros(v_t.shape)  # [B, D]
        for j in range(self.kernel_size):
            delay = j * self.dilation
            w = weight[:, 0, self.kernel_size - 1 - j].view(1, D)  # [1, D]
            tap = seq_norm[:, P - delay, :]  # [B, D] -- token t - delay
            if delay == 0:
                out = out + tap * w
                continue
            valid = (tok >= delay).unsqueeze(-1)  # [B, 1]
            out = out + torch.where(valid, tap, out.new_zeros(())) * w
        if self.conv.bias is not None:
            out = out + self.conv.bias.view(1, D)
        if self.activation:
            out = self.act_fn(out)
        return out
