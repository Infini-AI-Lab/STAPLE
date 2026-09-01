# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Engram conditional-memory layer (single-stream, no hyper-connections).

For each position, Engram retrieves static N-gram memory ``e_t`` via hashed
lookup, then fuses it into the residual stream through a context-aware gate
(paper Section 2.3)::

    k_t = W_K e_t,   v_t = W_V e_t
    alpha_t = sigmoid( RMSNorm(h_t)^T RMSNorm(k_t) / sqrt(d) )
    ṽ_t = alpha_t * v_t
    Y = SiLU(Conv1D(RMSNorm(ṽ))) + ṽ

``Y`` is returned to the caller, which adds it to the residual stream
(``H <- H + Y``). This variant omits the multi-branch / mHC integration: a
single Key projection, single gate, single Value projection.

The N-gram embedding tables are owned by :class:`MultiHeadEmbedding` and may be
row/vocab-parallel sharded; everything else here is small and FSDP-shardable
like the rest of the backbone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn

from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.protocols.module import Module

from .hashing import compute_layer_multipliers, EngramHasher
from .multi_head_embedding import MultiHeadEmbedding
from .short_conv import ShortConv
from .tokenizer_compression import load_compression_table

__all__ = ["EngramLayer"]

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}


class EngramLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # Backbone hidden size: Engram output is added to the residual stream,
        # so projections map the memory dim to ``dim``.
        dim: int
        layer_id: int

        # N-gram / hashing hyperparameters.
        max_ngram_size: int = 3
        n_head_per_ngram: int = 8
        per_head_dim: int = 64
        # Flat per-head table sizes for *this* layer, ordered to match
        # EngramHasher's head layout. Precomputed offline (primes) and baked in.
        head_table_sizes: list[int] = field(default_factory=list)

        # Compression table: a cached raw->canonical id lookup (built offline
        # via ``tokenizer_compression``). When None, an identity lookup of size
        # ``raw_vocab_size`` is used (debug / tests).
        compression_table_path: str | None = None
        raw_vocab_size: int | None = None
        num_canonical: int | None = None
        pad_id: int = 0
        seed: int = 0

        kernel_size: int = 4

        def __post_init__(self):
            if self.compression_table_path is None and self.raw_vocab_size is None:
                raise ValueError(
                    "EngramLayer.Config requires either compression_table_path "
                    "or raw_vocab_size (for an identity compression lookup)."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.layer_id = config.layer_id
        self.max_ngram_size = config.max_ngram_size
        self.n_head_per_ngram = config.n_head_per_ngram

        # Resolve the compression lookup and canonical vocab size. Force CPU
        # so this fixed data is real even when the model is built under a
        # ``torch.device("meta")`` context (it must survive ``to_empty``).
        if config.compression_table_path is not None:
            lookup, num_canonical = load_compression_table(
                config.compression_table_path
            )
            lookup = lookup.cpu()
        else:
            lookup = torch.arange(
                config.raw_vocab_size, dtype=torch.int64, device="cpu"
            )
            num_canonical = (
                config.num_canonical
                if config.num_canonical is not None
                else config.raw_vocab_size
            )

        # Compress the (raw) pad id once; ids hashed at sequence start pad with it.
        compressed_pad_id = int(lookup[config.pad_id].item())

        multipliers = torch.from_numpy(
            compute_layer_multipliers(
                layer_id=config.layer_id,
                max_ngram_size=config.max_ngram_size,
                tokenizer_vocab_size=num_canonical,
                seed=config.seed,
            )
        )

        # head_table_sizes is flat; reshape to [n_orders, n_head] for the hasher.
        n_orders = config.max_ngram_size - 1
        flat_sizes = list(config.head_table_sizes)
        head_table_sizes_2d = [
            flat_sizes[i * config.n_head_per_ngram : (i + 1) * config.n_head_per_ngram]
            for i in range(n_orders)
        ]

        self.hasher = EngramHasher(
            compression_lookup=lookup,
            multipliers=multipliers,
            head_table_sizes=head_table_sizes_2d,
            max_ngram_size=config.max_ngram_size,
            n_head_per_ngram=config.n_head_per_ngram,
            pad_id=compressed_pad_id,
        )

        self.embedding = MultiHeadEmbedding.Config(
            head_table_sizes=flat_sizes,
            per_head_dim=config.per_head_dim,
        ).build()

        num_heads_total = n_orders * config.n_head_per_ngram
        engram_hidden = num_heads_total * config.per_head_dim
        self.engram_hidden = engram_hidden
        self.scale = 1.0 / math.sqrt(config.dim)

        self.value_proj = Linear.Config(
            in_features=engram_hidden, out_features=config.dim, param_init=_LINEAR_INIT
        ).build()
        self.key_proj = Linear.Config(
            in_features=engram_hidden, out_features=config.dim, param_init=_LINEAR_INIT
        ).build()
        self.q_norm = RMSNorm.Config(
            normalized_shape=config.dim, param_init=_NORM_INIT
        ).build()
        self.k_norm = RMSNorm.Config(
            normalized_shape=config.dim, param_init=_NORM_INIT
        ).build()
        self.short_conv = ShortConv.Config(
            dim=config.dim,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
        ).build()

    def compute_hashes(
        self, tokens: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute this layer's N-gram hash indices ``[B, L, num_heads]``.

        Called by the model **outside** the activation-checkpointed block so the
        integer hashing is not recomputed in backward (autograd does not support
        integer recompute, and the hashes are deterministic anyway).

        ``positions`` (optional ``[B, L]`` per-token within-document offsets)
        keeps suffix N-grams from crossing packed-document boundaries; see
        :meth:`EngramHasher.forward`.
        """
        return self.hasher(tokens, positions)

    def _compute_value(
        self, hidden_states: torch.Tensor, hash_ids: torch.Tensor
    ) -> torch.Tensor:
        """Gated memory value ``ṽ`` (before the short conv): ``[B, L, D]``.

        This is the position-wise part of the layer -- embedding lookup,
        context-aware gate, and value projection -- and depends only on the
        *current* hidden state and hash ids (no cross-token mixing). The
        short-conv (the only part that mixes across positions) is applied
        separately by :meth:`forward`, so KV-cached decoding can reuse this to
        produce ``ṽ_t`` for the current token and feed a cached conv step.
        """
        embeddings = self.embedding(hash_ids).flatten(
            start_dim=-2
        )  # [B, L, engram_hidden]
        # The embedding table is excluded from FSDP's MixedPrecisionPolicy
        # (vocab-parallel sharded separately), so its output stays in the table's
        # dtype. Cast to the projection dtype before the (FSDP-cast) projections.
        embeddings = embeddings.to(self.value_proj.weight.dtype)

        key = self.k_norm(self.key_proj(embeddings))
        query = self.q_norm(hidden_states)
        # Context-aware scalar gate (paper Eq. 4). We follow the paper's plain
        # sigmoid(dot / sqrt(d)) rather than the demo's extra signed-sqrt.
        gate = (key * query).sum(dim=-1) * self.scale  # [B, L]
        gate = torch.sigmoid(gate).unsqueeze(-1)  # [B, L, 1]

        return gate * self.value_proj(embeddings)  # [B, L, D]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``hidden_states``: [B, L, D]; ``hash_ids``: [B, L, num_heads] -> [B, L, D].

        ``positions`` (optional ``[B, L]``) keeps the short-conv receptive field
        from crossing packed-document boundaries; see :meth:`ShortConv.forward`.
        """
        value = self._compute_value(hidden_states, hash_ids)  # [B, L, D]
        return value + self.short_conv(value, positions)
