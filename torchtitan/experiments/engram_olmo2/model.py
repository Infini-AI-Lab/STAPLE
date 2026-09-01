# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMo2 backbone augmented with Engram conditional-memory layers.

Reuses the core OLMo2 transformer block (post-norm attention + FFN) and adds an
optional :class:`EngramLayer` to selected blocks. Engram needs the raw token
ids (not just hidden states) to compute N-gram hashes, so this model overrides
``forward`` to thread ``tokens`` into the Engram-enabled blocks. Non-Engram
blocks keep the standard signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.olmo2.model import OLMo2Model, OLMo2TransformerBlock

from .engram import EngramLayer
from .multi_head_embedding import MultiHeadEmbedding

__all__ = [
    "EngramOLMo2Model",
    "EngramOLMo2TransformerBlock",
    "EngramGradReducer",
    "register_engram_tables",
    "get_engram_tables",
    "get_engram_grad_reducer",
]

# Module-level registry of the Engram embedding tables, keyed by layer id (the
# string keys of ``model.layers``), used by ``EngramTrainer`` to separate the
# table grads in its mesh-aware grad clipping. Populated by ``parallelize.py``
# after the tables are vocab-parallel sharded, so it holds the sharded params.
_ENGRAM_TABLES_REF: dict[str, MultiHeadEmbedding] | None = None
# Module-level registry for the grad reducer (used by EngramTrainer.train_step).
_ENGRAM_GRAD_REDUCER: "EngramGradReducer | None" = None


def register_engram_tables(tables: dict[str, MultiHeadEmbedding]) -> None:
    global _ENGRAM_TABLES_REF
    _ENGRAM_TABLES_REF = tables


def get_engram_tables() -> dict[str, MultiHeadEmbedding] | None:
    return _ENGRAM_TABLES_REF


def get_engram_grad_reducer() -> "EngramGradReducer | None":
    return _ENGRAM_GRAD_REDUCER


class EngramGradReducer:
    """Coalesced gradient all-reduce of the Engram tables over ``dp_replicate``.

    Under HSDP (``dp_replicate`` > 1), the vocab-parallel Engram tables are
    sharded only over the inner ``fsdp`` axis and are excluded from FSDP
    (``ignored_params``), so FSDP's reduce-scatter / dp_replicate all-reduce
    never touches them. Each replica processes different data and therefore
    produces a *different* table gradient; without an explicit reduction the
    replicas would silently diverge. This reducer all-reduces (mean) the table
    grads across the ``dp_replicate`` group after the final microbatch backward,
    mirroring ``stem_dag``'s ``StemGradReducer``.

    Pure FSDP (``dp_replicate`` == 1) needs no reducer: the single ``fsdp`` axis
    already owns disjoint row shards whose grads are each complete.
    """

    def __init__(
        self,
        tables: dict[str, MultiHeadEmbedding],
        group: dist.ProcessGroup,
        device: torch.device,
    ):
        self._tables = tables
        self._group = group
        self._stream = torch.cuda.Stream(device=device)
        self._sync_event: torch.cuda.Event | None = None

    def reduce(self) -> None:
        """Launch a coalesced SUM all-reduce of the table grads on a side stream
        so it overlaps with FSDP's dp_replicate gradient sync.

        SUM (not mean) matches how the backbone grads are reduced: the trainer
        runs with ``disable_fsdp_gradient_division`` (FSDP gradient_divide_factor
        = 1.0) and divides the loss by the global token count, so each rank's
        grad is already a partial sum and summing across the dp_replicate group
        yields the correct global gradient. A mean here would double-normalize.
        """
        grads = []
        for table in self._tables.values():
            w = table.weight
            if w.grad is None:
                continue
            # Reduce the local row-shard grad in place; DTensor wraps a local
            # tensor we can all-reduce directly over the dp_replicate group.
            grad = w.grad._local_tensor if hasattr(w.grad, "_local_tensor") else w.grad
            grads.append(grad)

        if not grads:
            return

        current_stream = torch.cuda.current_stream()
        self._stream.wait_stream(current_stream)
        with torch.cuda.stream(self._stream):
            dist.all_reduce_coalesced(grads, group=self._group)
        self._sync_event = self._stream.record_event()

    def sync(self) -> None:
        """Block the current stream until the side-stream all-reduce completes.
        Must be called before grad-norm computation or the optimizer step."""
        if self._sync_event is not None:
            torch.cuda.current_stream().wait_event(self._sync_event)
            self._sync_event = None


class EngramOLMo2TransformerBlock(OLMo2TransformerBlock):
    """OLMo2 post-norm block with an optional pre-attention Engram residual.

    When ``engram`` is set, Engram is applied first and added to the residual
    stream (``x <- x + Engram(x, tokens)``), then the standard attention and
    feed-forward sublayers run (paper Section 2.3).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OLMo2TransformerBlock.Config):
        engram: EngramLayer.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.engram = config.engram.build() if config.engram is not None else None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        hash_ids: torch.Tensor | None = None,
    ):
        if self.engram is not None:
            if hash_ids is None:
                raise ValueError(
                    "EngramOLMo2TransformerBlock has an Engram layer but received "
                    "hash_ids=None. The model must pass precomputed hash indices "
                    "into Engram blocks."
                )
            # ``positions`` (per-document offsets, when packing documents) keeps
            # the Engram short-conv from crossing document boundaries, matching
            # the intra-document attention mask. The N-gram hashes were already
            # made boundary-aware with the same positions in ``compute_hashes``.
            x = x + self.engram(x, hash_ids, positions)
        return super().forward(x, freqs_cis, attention_masks, positions)


class EngramOLMo2Model(OLMo2Model):
    """OLMo2 model with Engram layers; forward threads token ids to them."""

    @dataclass(kw_only=True, slots=True)
    class Config(OLMo2Model.Config):
        # Layer indices that carry an Engram module (informational; the actual
        # modules are built from per-layer ``engram`` configs in ``layers``).
        engram_layer_ids: list[int] = field(default_factory=list)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            engram = getattr(layer, "engram", None)
            if engram is not None:
                # Compute hashes here -- outside the layer's activation-checkpoint
                # region -- so the integer hashing is not recomputed in backward.
                # ``positions`` (per-document offsets from the dataloader when
                # packing documents) make the suffix N-grams boundary-aware so a
                # document never hashes tokens from the preceding packed
                # document; the same positions gate the short-conv inside the
                # layer. When None (plain causal / single doc), the legacy
                # global-pad behavior is used.
                hash_ids = engram.compute_hashes(tokens, positions)
                h = layer(
                    h, self.freqs_cis, attention_masks, positions, hash_ids=hash_ids
                )
            else:
                h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h
