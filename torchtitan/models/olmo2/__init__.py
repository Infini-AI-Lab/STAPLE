# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import OLMo2Model, OLMo2TransformerBlock
from .parallelize import parallelize_olmo2
from .state_dict_adapter import OLMo2StateDictAdapter

__all__ = [
    "parallelize_olmo2",
    "OLMo2Model",
    "olmo2_configs",
]

_EPS = 1e-6

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _olmo2_norm(dim: int | list[int]) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _build_olmo2_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    hidden_dim: int,
    head_dim: int,
    attn_backend: str,
    use_depth_scaled_init: bool = False,
) -> list[TransformerBlock.Config]:
    """Build per-layer OLMo2TransformerBlock configs.

    Args:
        use_depth_scaled_init: If True, use depth-scaled std for wo/w2/w3
            (like Llama3). If False (default), use uniform std=0.02 for all
            layers, matching STEM-pt's ``init_std_factor: disabled``.
    """
    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        wo_init = _depth_init(layer_id) if use_depth_scaled_init else _LINEAR_INIT
        w2w3_init = _depth_init(layer_id) if use_depth_scaled_init else _LINEAR_INIT
        layers.append(
            OLMo2TransformerBlock.Config(
                # Post-norms (applied after sublayer output)
                attention_norm=_olmo2_norm(dim),
                ffn_norm=_olmo2_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=wo_init,
                    inner_attention=inner_attention,
                    mask_type=mask_type,
                    rope_backend="complex",
                    # OLMo2 QK-norm: normalize across all heads jointly.
                    # Shape [n_heads, head_dim] normalizes the last 2 dims of
                    # the (bs, seq, n_heads, head_dim) tensor together.
                    qk_norm=_olmo2_norm([n_heads, head_dim]),
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=w2w3_init,
                ),
            )
        )
    return layers


def _debugmodel(attn_backend: str) -> OLMo2Model.Config:
    dim = 256
    n_heads = 16
    n_kv_heads = 16
    head_dim = dim // n_heads
    n_layers = 6
    vocab_size = 2048
    return OLMo2Model.Config(
        dim=dim,
        vocab_size=vocab_size,
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
            dim=head_dim,
            max_seq_len=4096,
            theta=500000,
            backend="complex",
        ),
        layers=_build_olmo2_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            attn_backend=attn_backend,
        ),
    )


def _7b(attn_backend: str) -> OLMo2Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 32
    head_dim = 128
    n_layers = 32
    vocab_size = 100352
    return OLMo2Model.Config(
        dim=dim,
        vocab_size=vocab_size,
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
            dim=head_dim,
            max_seq_len=4096,
            theta=500000,
            backend="complex",
        ),
        layers=_build_olmo2_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            attn_backend=attn_backend,
        ),
    )


def _1b(attn_backend: str) -> OLMo2Model.Config:
    dim = 2048
    n_heads = 16
    n_kv_heads = 16
    head_dim = 128
    n_layers = 16
    vocab_size = 100352
    return OLMo2Model.Config(
        dim=dim,
        vocab_size=vocab_size,
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
            dim=head_dim,
            max_seq_len=4096,
            theta=500000,
            backend="complex",
        ),
        layers=_build_olmo2_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=256, ffn_dim_multiplier=1.5
            ),
            attn_backend=attn_backend,
        ),
    )


olmo2_configs = {
    "debugmodel": _debugmodel,
    "1B": _1b,
    "7B": _7b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = olmo2_configs[flavor](attn_backend=attn_backend)
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            c.build().convert(config)
    return ModelSpec(
        name="olmo2",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_olmo2,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=OLMo2StateDictAdapter,
    )
