# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic multi-head N-gram hashing for Engram (paper Section 2.2).

Each Engram layer maps the compressed suffix N-gram at every position to a set
of table indices via a lightweight multiplicative-XOR hash, using ``K`` hash
heads per N-gram order ``n`` (here ``n in {2, ..., max_ngram}``) to mitigate
collisions::

    mix_{t,n} = (x'_{t-n+1} * m_0) XOR (x'_{t-n+2} * m_1) XOR ... XOR (x'_t * m_{n-1})
    z_{t,n,k} = mix_{t,n} mod M_{n,k}

where ``M_{n,k}`` is a distinct prime table size per (n, k) and ``m_*`` are
odd per-layer multipliers. The hash depends only on the input ids, so the same
indices can be precomputed/prefetched (paper Section 2.5).

Prime table sizes use ``sympy`` and are computed **offline** at config-build
time, then baked into the model config. At training time only integer tensor
ops run, keeping the training path free of non-PyTorch dependencies.
"""

from __future__ import annotations

import numpy as np
import torch

from torchtitan.protocols.module import Module

__all__ = [
    "find_next_prime",
    "compute_head_table_sizes",
    "compute_layer_multipliers",
    "ngram_hashes_numpy",
    "EngramHasher",
]

# Seed offset between layers so different Engram layers hash the same N-gram to
# different indices (matches the reference demo).
_LAYER_SEED_PRIME = 10007


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Return the smallest prime > ``start`` not already in ``seen_primes``."""
    from sympy import isprime

    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


def compute_head_table_sizes(
    engram_vocab_size: int | list[int],
    max_ngram_size: int,
    n_head_per_ngram: int,
    layer_ids: list[int],
) -> dict[int, list[list[int]]]:
    """Compute distinct prime table sizes per (layer, n-gram order, head).

    ``engram_vocab_size`` is the base table size; the actual size for each head
    is the next prime at or above it. Primes are kept globally distinct across
    all (layer, n, head) triples (computed in ``layer_ids`` order), matching the
    reference implementation. ``engram_vocab_size`` may be a single int (applied
    to every N-gram order) or a list indexed by ``n - 2``.

    Returns ``{layer_id: [[size for each head] for each n in 2..max_ngram]}``.
    """
    seen_primes: set[int] = set()
    sizes_by_layer: dict[int, list[list[int]]] = {}

    for layer_id in layer_ids:
        per_ngram_sizes: list[list[int]] = []
        for ngram in range(2, max_ngram_size + 1):
            if isinstance(engram_vocab_size, int):
                base = engram_vocab_size
            else:
                base = engram_vocab_size[ngram - 2]
            head_sizes: list[int] = []
            search_start = base - 1
            for _ in range(n_head_per_ngram):
                prime = find_next_prime(search_start, seen_primes)
                seen_primes.add(prime)
                head_sizes.append(prime)
                search_start = prime
            per_ngram_sizes.append(head_sizes)
        sizes_by_layer[layer_id] = per_ngram_sizes

    return sizes_by_layer


def compute_layer_multipliers(
    layer_id: int,
    max_ngram_size: int,
    tokenizer_vocab_size: int,
    seed: int,
) -> np.ndarray:
    """Compute the odd per-position hash multipliers for one Engram layer.

    Multipliers are bounded so that ``id * multiplier`` cannot overflow int64
    for any compressed id in ``[0, tokenizer_vocab_size)``.
    """
    max_long = np.iinfo(np.int64).max
    m_max = int(max_long // tokenizer_vocab_size)
    half_bound = max(1, m_max // 2)

    base_seed = int(seed + _LAYER_SEED_PRIME * int(layer_id))
    rng = np.random.default_rng(base_seed)
    r = rng.integers(low=0, high=half_bound, size=(max_ngram_size,), dtype=np.int64)
    return r * 2 + 1  # force odd


def ngram_hashes_numpy(
    input_ids: np.ndarray,
    multipliers: np.ndarray,
    head_table_sizes: list[list[int]],
    max_ngram_size: int,
    n_head_per_ngram: int,
    pad_id: int,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Reference numpy implementation of multi-head N-gram hashing.

    ``input_ids`` are **already compressed** canonical ids, shape ``[B, T]``.
    Returns ``[B, T, num_heads_total]`` where
    ``num_heads_total = (max_ngram_size - 1) * n_head_per_ngram``.

    ``positions`` (optional, ``[B, T]``) are per-token within-document offsets
    that reset to 0 at each packed-document start. When provided, a suffix
    N-gram that would reach across a document boundary is padded instead, so
    packed documents hash exactly as if each were its own sequence. When
    ``None``, only the global left edge is padded (the legacy behavior).

    Used to validate the tensor implementation in tests.
    """
    x = np.asarray(input_ids, dtype=np.int64)
    B, T = x.shape
    pos = None if positions is None else np.asarray(positions)

    def shift_k(k: int) -> np.ndarray:
        if k == 0:
            return x
        shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=pad_id)[
            :, :T
        ]
        if pos is not None:
            # Token t-k is in the same document iff the current token's
            # within-document offset is at least k.
            shifted = np.where(pos >= k, shifted, pad_id)
        return shifted

    base_shifts = [shift_k(k) for k in range(max_ngram_size)]
    all_hashes = []
    for n in range(2, max_ngram_size + 1):
        mix = base_shifts[0] * multipliers[0]
        for k in range(1, n):
            mix = np.bitwise_xor(mix, base_shifts[k] * multipliers[k])
        for size in head_table_sizes[n - 2]:
            all_hashes.append((mix % int(size)).astype(np.int64, copy=False))
    return np.stack(all_hashes, axis=2)


class EngramHasher(Module):
    """Compresses token ids and produces multi-head N-gram hash indices.

    Subclasses the torchtitan ``Module`` protocol (so it passes
    ``verify_module_protocol``) but is constructed directly by ``EngramLayer``
    rather than via ``Config.build()``, since its inputs include the resolved
    compression lookup. It owns no learnable parameters: its fixed lookup data
    is deterministic from config and recomputable, so it is **not** stored as
    parameters or buffers (which the meta-init + ``to_empty`` path would wipe,
    and which would needlessly enter checkpoints). Multipliers and per-head
    table sizes are kept as Python ints (no device affinity); the compression
    lookup is a plain CPU tensor lazily moved to the compute device on first
    forward.

    Forward consumes raw token ids ``[B, T]`` and returns hash indices
    ``[B, T, num_heads_total]`` (int64), with heads ordered as
    ``(n=2 head 0..K-1, n=3 head 0..K-1, ...)`` to match the embedding table.
    """

    def __init__(
        self,
        *,
        compression_lookup: torch.Tensor,
        multipliers: torch.Tensor,
        head_table_sizes: list[list[int]],
        max_ngram_size: int,
        n_head_per_ngram: int,
        pad_id: int,
    ):
        super().__init__()
        self.max_ngram_size = max_ngram_size
        self.n_head_per_ngram = n_head_per_ngram
        # Compressed pad id (already mapped through the compression table).
        self.pad_id = int(pad_id)

        # Plain CPU tensor (not a buffer): lazily moved to device in forward.
        self._compression_lookup = compression_lookup.to(torch.int64).cpu()
        # Python ints -- multiplying / mod-ing a device tensor by a Python int
        # keeps the result on-device with no host tensor involved.
        self.multipliers = [int(m) for m in multipliers.tolist()]
        self.head_mods = [int(s) for per_ngram in head_table_sizes for s in per_ngram]

    @property
    def num_heads_total(self) -> int:
        return (self.max_ngram_size - 1) * self.n_head_per_ngram

    def _compress(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map raw token ids to canonical ids; negative ids map to pad."""
        if self._compression_lookup.device != tokens.device:
            self._compression_lookup = self._compression_lookup.to(tokens.device)
        safe = tokens.clamp_min(0).to(torch.int64)
        compressed = self._compression_lookup[safe]
        return torch.where(
            tokens >= 0,
            compressed,
            torch.full_like(compressed, self.pad_id),
        )

    def forward(
        self, tokens: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``tokens``: [B, T] raw ids; ``positions``: optional [B, T] per-token
        within-document offsets (reset to 0 at each packed-document start).

        When ``positions`` is given, a suffix N-gram that would reach across a
        document boundary is padded with the compressed pad id, so packed
        documents hash identically to being processed alone -- matching the
        intra-document (``block_causal``) attention mask. When ``None``, only
        the global left edge is padded (the legacy single-document behavior,
        equivalent to ``positions`` being a plain ``arange``).
        """
        compressed = self._compress(tokens)  # [B, T] int64
        B, T = compressed.shape

        if positions is not None:
            # Broadcast over the shift comparisons below; keep on-device int64.
            positions = positions.to(device=compressed.device, dtype=torch.int64)

        def shift_k(k: int) -> torch.Tensor:
            if k == 0:
                return compressed
            pad = compressed.new_full((B, k), self.pad_id)
            shifted = torch.cat([pad, compressed[:, : T - k]], dim=1)
            if positions is not None:
                # Token t-k is in the same document iff positions[t] >= k;
                # otherwise the n-gram crosses a boundary -> pad it.
                same_doc = positions >= k
                shifted = torch.where(
                    same_doc, shifted, shifted.new_full((), self.pad_id)
                )
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        hashes = []
        head_idx = 0
        for n in range(2, self.max_ngram_size + 1):
            mix = base_shifts[0] * self.multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, base_shifts[k] * self.multipliers[k])
            for _ in range(self.n_head_per_ngram):
                hashes.append(torch.remainder(mix, self.head_mods[head_idx]))
                head_idx += 1

        return torch.stack(hashes, dim=2)  # [B, T, num_heads_total]
