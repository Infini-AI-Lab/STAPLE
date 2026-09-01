# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config registry for Engram-OLMo2.

Builds an :class:`EngramOLMo2Model.Config` by taking the standard OLMo2 layer
stack and attaching an :class:`EngramLayer.Config` to the chosen layers. The
expensive, deterministic prime table sizes are computed here (offline, at config
build) and baked into each Engram layer config so the training path stays free
of ``sympy``.
"""

from __future__ import annotations

from functools import partial

import torch.nn as nn

from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .engram import EngramLayer
from .hashing import compute_head_table_sizes
from .model import EngramOLMo2Model, EngramOLMo2TransformerBlock
from .parallelize import parallelize_engram_olmo2
from .state_dict_adapter import EngramOLMo2StateDictAdapter

__all__ = ["model_registry", "engram_olmo2_configs"]

_EPS = 1e-6
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _olmo2_norm(dim: int | list[int]) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _output_linear_init(dim: int) -> dict:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _build_engram_config(
    *,
    layer_id: int,
    dim: int,
    engram_vocab_size: int,
    max_ngram_size: int,
    n_head_per_ngram: int,
    per_head_dim: int,
    head_table_sizes_by_layer: dict[int, list[list[int]]],
    compression_table_path: str | None,
    raw_vocab_size: int | None,
    num_canonical: int | None,
    pad_id: int,
    seed: int,
    kernel_size: int,
) -> EngramLayer.Config:
    # Flatten this layer's [n_orders, n_head] prime sizes into one ordered list.
    flat_sizes = [
        s for per_ngram in head_table_sizes_by_layer[layer_id] for s in per_ngram
    ]
    return EngramLayer.Config(
        dim=dim,
        layer_id=layer_id,
        max_ngram_size=max_ngram_size,
        n_head_per_ngram=n_head_per_ngram,
        per_head_dim=per_head_dim,
        head_table_sizes=flat_sizes,
        compression_table_path=compression_table_path,
        raw_vocab_size=raw_vocab_size,
        num_canonical=num_canonical,
        pad_id=pad_id,
        seed=seed,
        kernel_size=kernel_size,
    )


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    attn_backend: str,
    engram_layer_ids: list[int],
    engram_kwargs: dict,
) -> list[EngramOLMo2TransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)

    # Precompute deterministic prime table sizes for all Engram layers once.
    head_table_sizes_by_layer = compute_head_table_sizes(
        engram_vocab_size=engram_kwargs["engram_vocab_size"],
        max_ngram_size=engram_kwargs["max_ngram_size"],
        n_head_per_ngram=engram_kwargs["n_head_per_ngram"],
        layer_ids=engram_layer_ids,
    )

    layers: list[EngramOLMo2TransformerBlock.Config] = []
    for layer_id in range(n_layers):
        engram_cfg = None
        if layer_id in engram_layer_ids:
            engram_cfg = _build_engram_config(
                layer_id=layer_id,
                dim=dim,
                head_table_sizes_by_layer=head_table_sizes_by_layer,
                **engram_kwargs,
            )
        layers.append(
            EngramOLMo2TransformerBlock.Config(
                attention_norm=_olmo2_norm(dim),
                ffn_norm=_olmo2_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_LINEAR_INIT,
                    inner_attention=inner_attention,
                    mask_type=mask_type,
                    rope_backend="complex",
                    qk_norm=_olmo2_norm([n_heads, head_dim]),
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_LINEAR_INIT,
                ),
                engram=engram_cfg,
            )
        )
    return layers


def _make_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    vocab_size: int,
    ffn_dim_multiplier: float | None,
    attn_backend: str,
    engram_layer_ids: list[int],
    engram_kwargs: dict,
) -> EngramOLMo2Model.Config:
    return EngramOLMo2Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        engram_layer_ids=engram_layer_ids,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=_olmo2_norm(dim),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim, max_seq_len=4096, theta=500000, backend="complex"
        ),
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=256, ffn_dim_multiplier=ffn_dim_multiplier
            ),
            attn_backend=attn_backend,
            engram_layer_ids=engram_layer_ids,
            engram_kwargs=engram_kwargs,
        ),
    )


# Engram defaults: max 3-gram, 8 heads. per_head_dim chosen so the engram tables
# are large but the fusion dim stays modest. Each flavor overrides
# engram_vocab_size (and the debugmodel shrinks the rest).
_ENGRAM_DEFAULTS = dict(
    engram_vocab_size=2358272,
    max_ngram_size=3,
    n_head_per_ngram=8,
    per_head_dim=64,
    compression_table_path=None,
    raw_vocab_size=100352,  # OLMo2 tokenizer vocab; identity compression unless a table is provided
    num_canonical=None,
    pad_id=0,
    seed=0,
    kernel_size=4,
)
_ENGRAM_LAYER_IDS = [2, 7]
_STEM_DAG_L12_MATCHED_VOCAB_SIZE = 4_812_523


def _debugmodel(attn_backend: str) -> EngramOLMo2Model.Config:
    # Small tables so the debug model builds/runs on CPU quickly.
    engram_kwargs = dict(_ENGRAM_DEFAULTS)
    engram_kwargs.update(engram_vocab_size=512, per_head_dim=16, raw_vocab_size=2048)
    return _make_config(
        dim=256,
        n_heads=16,
        n_kv_heads=16,
        head_dim=16,
        n_layers=6,
        vocab_size=2048,
        ffn_dim_multiplier=None,
        attn_backend=attn_backend,
        engram_layer_ids=[2, 4],
        engram_kwargs=engram_kwargs,
    )


def _1b_stem_dag_l12_size(attn_backend: str) -> EngramOLMo2Model.Config:
    """OLMo2-1B with two Engram layers whose tables match STAPLE-12L's size."""
    engram_kwargs = dict(_ENGRAM_DEFAULTS)
    engram_kwargs["engram_vocab_size"] = _STEM_DAG_L12_MATCHED_VOCAB_SIZE
    return _make_config(
        dim=2048,
        n_heads=16,
        n_kv_heads=16,
        head_dim=128,
        n_layers=16,
        vocab_size=100352,
        ffn_dim_multiplier=1.5,
        attn_backend=attn_backend,
        engram_layer_ids=_ENGRAM_LAYER_IDS,
        engram_kwargs=engram_kwargs,
    )


engram_olmo2_configs = {
    "debugmodel": _debugmodel,
    "1B_stem_dag_l12_size": _1b_stem_dag_l12_size,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = engram_olmo2_configs[flavor](attn_backend=attn_backend)
    if converters is not None:
        for c in converters:
            c.build().convert(config)
    return ModelSpec(
        name="engram_olmo2",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_engram_olmo2,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=EngramOLMo2StateDictAdapter,
    )
