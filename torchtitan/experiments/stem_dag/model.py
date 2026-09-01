# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
STEM DAG OLMo2 model.

Architecture:
- Standard OLMo2 transformer (post-norm, full-dim QK-norm)
- Designated "stem layers" replace the standard SwiGLU FFN with a DAG FFN:
    output = w2(silu(w1(x)) * (w3(x) + y))
  where y is a per-layer stem embedding looked up from token IDs.
- Each stem layer has its own embedding table of shape (vocab_size, hidden_dim).
- Stem embeddings use intra-node dimension-parallel sharding on the EP mesh.
  Each GPU holds (V, D/p). Forward: all-gather IDs → local lookup → all-to-all.
- Weight is zero-initialized so the model starts equivalent to the base
  (w3(x) + 0 = w3(x)) when loading from a pretrained checkpoint.
"""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import (
    all_gather_tensor,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.nn_modules import Embedding, Linear
from torchtitan.models.olmo2.model import OLMo2TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.module import Module, ModuleDict

# Module-level registry for gradient norm logging.
_STEM_EMBEDDINGS_REF: ModuleDict | None = None
# Module-level registry for the grad reducer (used by trainer hooks).
_STEM_GRAD_REDUCER: "StemGradReducer | None" = None


def register_stem_embeddings(ref: ModuleDict) -> None:
    global _STEM_EMBEDDINGS_REF
    _STEM_EMBEDDINGS_REF = ref


def get_stem_grad_reducer() -> "StemGradReducer | None":
    return _STEM_GRAD_REDUCER


class StemGradReducer:
    """Coalesced gradient all-reduce for all stem embeddings on a side stream.

    Replaces per-layer post_accumulate_grad_hook with a single coalesced
    all-reduce that:
    1. Only fires on the final gradient accumulation microbatch (called
       explicitly by the trainer after backward).
    2. Coalesces all stem embedding grads into one NCCL call (1× latency
       instead of N×).
    3. Runs on a dedicated CUDA stream so it can overlap with FSDP's
       dp_replicate gradient sync.
    """

    def __init__(
        self, embeddings: ModuleDict, group: dist.ProcessGroup, device: torch.device
    ):
        self._embeddings = embeddings
        self._group = group
        self._stream = torch.cuda.Stream(device=device)
        self._sync_event: torch.cuda.Event | None = None

    def reduce(self) -> None:
        """Launch coalesced all-reduce of all stem grads on side stream."""
        grads = []
        for emb in self._embeddings.values():
            w = emb.weight
            if w.grad is None:
                continue
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
        """Block current stream until the side-stream all-reduce completes.

        Must be called before grad norm computation or optimizer step.
        """
        if self._sync_event is not None:
            torch.cuda.current_stream().wait_event(self._sync_event)
            self._sync_event = None


# ---------------------------------------------------------------------------
# STEM DAG FeedForward
# ---------------------------------------------------------------------------


class StemDagFeedForward(Module):
    """SwiGLU FFN with additive stem signal: w2(silu(w1(x)) * (w3(x) + y)).

    When y is None (non-stem usage or during init), falls back to standard
    SwiGLU: w2(silu(w1(x)) * w3(x)).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.w3 = config.w3.build()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        y_inverse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        if y is not None:
            if y_inverse is not None:
                y = F.embedding(y_inverse, y)
            up = up + y
        return self.w2(gate * up)


# ---------------------------------------------------------------------------
# STEM DAG TransformerBlock (OLMo2 post-norm)
# ---------------------------------------------------------------------------


class StemDagTransformerBlock(OLMo2TransformerBlock):
    """OLMo2 block where the FFN can receive a stem embedding y."""

    @dataclass(kw_only=True, slots=True)
    class Config(OLMo2TransformerBlock.Config):
        is_stem_layer: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self._is_stem_layer = config.is_stem_layer

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        stem_embedding: torch.Tensor | None = None,
        stem_inverse: torch.Tensor | None = None,
    ):
        h = x + self.attention_norm(
            self.attention(x, freqs_cis, attention_masks, positions)
        )
        if self._is_stem_layer and stem_embedding is not None:
            out = h + self.ffn_norm(self.feed_forward(h, stem_embedding, stem_inverse))
        else:
            out = h + self.ffn_norm(self.feed_forward(h))
        return out

    def forward_stem_prefix(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute through both FFN up projections before STEM is required."""
        h = x + self.attention_norm(
            self.attention(x, freqs_cis, attention_masks, positions)
        )
        return h, F.silu(self.feed_forward.w1(h)), self.feed_forward.w3(h)

    def forward_stem_suffix(
        self,
        h: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        stem_embedding: torch.Tensor,
        stem_inverse: torch.Tensor,
    ) -> torch.Tensor:
        """Consume compact STEM rows after asynchronous staging completes."""
        stem = F.embedding(stem_inverse, stem_embedding)
        return h + self.ffn_norm(self.feed_forward.w2(gate * (up + stem)))


# ---------------------------------------------------------------------------
# Stem Embedding
# ---------------------------------------------------------------------------


class StemEmbedding(nn.Embedding, Module):
    """Per-layer embedding with intra-node dimension-parallel sharding.

    Weight is sharded along embedding_dim across the EP group: each GPU holds
    (V, D/p). Forward: all-gather token IDs → local lookup → all-to-all
    (batch↔dim). Zero-initialized so model starts equivalent to base.

    Gradient reduction is handled externally by StemGradReducer (coalesced
    all-reduce on a side stream), not by per-layer hooks.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)
        self._ep_mesh: DeviceMesh | None = None
        self._ep_size: int = 1

    def wire_ep_mesh(self, ep_mesh: DeviceMesh) -> None:
        self._ep_mesh = ep_mesh
        self._ep_size = ep_mesh.size()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._ep_mesh is None or self._ep_size == 1:
            return F.embedding(input, self.weight)

        p = self._ep_size
        B_local, S = input.shape

        # Get local weight shard (V, D/p) from DTensor managed by FSDP.
        weight = self.weight
        if hasattr(weight, "to_local"):
            weight = weight.to_local()

        # 1. All-gather token IDs: [B_local, S] → [B_local*p, S]
        global_input = all_gather_tensor(input, gather_dim=0, group=self._ep_mesh)

        # 2. Local embedding lookup on shard (V, D/p): → [B_local*p, S, D/p]
        local_emb = F.embedding(global_input, weight)
        D_local = local_emb.shape[-1]

        # 3. All-to-all: [p*B_local, S, D/p] → [B_local, S, D]
        #    Flatten for all_to_all_single (equal splits on dim 0).
        #    Chunk i (batch slice i's embeddings) is sent to rank i.
        flat = local_emb.contiguous().view(-1)
        flat_out = all_to_all_single_autograd(flat, None, None, self._ep_mesh)
        # Received p chunks from different D-shards for this rank's batch slice.
        # Reshape [p, B_local, S, D/p] → permute → [B_local, S, D]
        result = flat_out.view(p, B_local, S, D_local)
        result = result.permute(1, 2, 0, 3).contiguous().view(B_local, S, p * D_local)

        return result


# ---------------------------------------------------------------------------
# Low-Rank Stem Embedding
# ---------------------------------------------------------------------------


class LowRankStemEmbedding(Module):
    """Factored stem signal: ``y = up_proj(table(tokens))``.

    Instead of a single (vocab_size, ffn_hidden_dim) table (as in
    ``StemEmbedding``), the per-token signal is produced by a small
    (vocab_size, stem_dim) table followed by a per-layer up-projection
    (stem_dim -> ffn_hidden_dim). With stem_dim << ffn_hidden_dim this shrinks
    the parameter count by ``ffn_hidden_dim / stem_dim`` (e.g. 32x for
    8192/256), so the table is small enough to shard with plain FSDP -- no
    EP dimension-parallel all-to-all is needed.

    The up-projection is zero-initialized so the model starts equivalent to
    the base (``w3(x) + 0 = w3(x)``) when loading a pretrained checkpoint,
    while the table itself uses a normal init and trains freely.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        table: Embedding.Config
        up_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.table = config.table.build()
        self.up_proj = config.up_proj.build()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.table(input))


# ---------------------------------------------------------------------------
# STEM DAG OLMo2 Model
# ---------------------------------------------------------------------------


class StemDagOLMo2Model(Decoder):
    """OLMo2 with STEM DAG layers.

    Stem layers are specified by index. Each stem layer gets:
    - A StemDagFeedForward (with w3 + additive y)
    - A StemEmbedding(vocab_size, ffn_hidden_dim) sharded via FSDP

    The model's forward receives token IDs and internally looks up
    stem embeddings for each stem layer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 100352
        stem_layers: list[int] = field(default_factory=list)
        stem_embeddings: dict[str, StemEmbedding.Config] = field(default_factory=dict)

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "StemDagOLMo2 does not support tensor_parallel_degree > 1 yet."
                )

            from torch.distributed.tensor import Replicate, Shard

            from torchtitan.models.olmo2.sharding import set_olmo2_sharding_config
            from torchtitan.protocols.sharding import ShardingConfig
            from torchtitan.protocols.types import MeshAxisName

            set_olmo2_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
            )

            if parallelism.expert_parallel_degree > 1:
                for emb_cfg in self.stem_embeddings.values():
                    emb_cfg.sharding_config = ShardingConfig(
                        state_shardings={
                            "weight": {
                                MeshAxisName.DP_REPLICATE: Replicate(),
                                MeshAxisName.EFSDP: Replicate(),
                                MeshAxisName.EP: Shard(1),
                            },
                        },
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
        self._stem_layer_set = set(config.stem_layers)
        self._stem_embeddings = ModuleDict()
        for key, emb_config in config.stem_embeddings.items():
            self._stem_embeddings[key] = emb_config.build()

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for i, layer in self.layers.items():
            if int(i) in self._stem_layer_set:
                y = self._stem_embeddings[i](tokens).to(h.dtype)
                h = layer(
                    h,
                    self.freqs_cis,
                    attention_masks,
                    positions=positions,
                    stem_embedding=y,
                )
            else:
                h = layer(h, self.freqs_cis, attention_masks, positions=positions)

        h = self.norm(h) if self.norm is not None else h

        if self._skip_lm_head:
            return h
        output = self.lm_head(h) if self.lm_head is not None else h
        return output


# ---------------------------------------------------------------------------
# Low-Rank STEM DAG OLMo2 Model
# ---------------------------------------------------------------------------


class StemDagLowRankOLMo2Model(StemDagOLMo2Model):
    """OLMo2 with low-rank STEM DAG layers.

    Identical to ``StemDagOLMo2Model`` except each stem layer's signal comes
    from a ``LowRankStemEmbedding`` (small (vocab_size, stem_dim) table +
    per-layer up-projection to ffn_hidden_dim) rather than a full
    (vocab_size, ffn_hidden_dim) ``StemEmbedding``. The factored table is small
    enough to shard with plain FSDP, so there is no EP dimension-parallel
    sharding of the stem tables -- ``update_from_config`` only applies the
    standard OLMo2 sharding to the backbone.

    ``__init__`` and ``forward`` are inherited unchanged: both iterate
    ``_stem_embeddings`` and call ``emb(tokens)``, which the LowRank module
    supports with the same signature.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(StemDagOLMo2Model.Config):
        stem_embeddings: dict[str, LowRankStemEmbedding.Config] = field(
            default_factory=dict
        )

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "StemDagLowRankOLMo2 does not support "
                    "tensor_parallel_degree > 1 yet."
                )

            from torchtitan.models.olmo2.sharding import set_olmo2_sharding_config

            set_olmo2_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
            )
            # The low-rank stem tables/up-projections are small and sharded by
            # plain FSDP (see parallelize_stem_dag_lowrank); unlike the full
            # StemEmbedding they do NOT use EP dimension-parallel sharding, so
            # no per-embedding ShardingConfig is set here.
