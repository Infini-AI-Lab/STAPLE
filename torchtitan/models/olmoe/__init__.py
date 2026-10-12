# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, TransformerBlock
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_experts_config,
    make_gqa_config,
    make_moe_config,
)
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import OLMoEModel, OLMoERouter, OLMoETransformerBlock
from .parallelize import parallelize_olmoe
from .state_dict_adapter import OLMoEStateDictAdapter

__all__ = [
    "parallelize_olmoe",
    "OLMoEModel",
    "olmoe_configs",
]

_EPS = 1e-5

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


def _olmoe_norm(dim: int | list[int]) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _experts_init() -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
        "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
    }


def _build_olmoe_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    num_experts: int,
    top_k: int,
    expert_hidden_dim: int,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
) -> list[TransformerBlock.Config]:
    """Build per-layer OLMoETransformerBlock configs.

    All layers are MoE layers (no dense layers in OLMoE-1B-7B).
    """
    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            OLMoETransformerBlock.Config(
                attention_norm=_olmoe_norm(dim),
                ffn_norm=_olmoe_norm(dim),
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
                    # OLMoE QK-norm: per-head normalization.
                    # Shape [n_heads, head_dim] normalizes last 2 dims of
                    # the (bs, seq, n_heads, head_dim) tensor together.
                    qk_norm=_olmoe_norm([n_heads, head_dim]),
                ),
                moe=make_moe_config(
                    num_experts=num_experts,
                    router=OLMoERouter.Config(
                        num_experts=num_experts,
                        aux_loss_layer_idx=layer_id,
                        gate=Linear.Config(
                            in_features=dim,
                            out_features=num_experts,
                            bias=False,
                            param_init=_LINEAR_INIT,
                        ),
                        top_k=top_k,
                        score_func="softmax",
                        route_norm=False,
                        route_scale=1.0,
                    ),
                    experts=make_experts_config(
                        dim=dim,
                        hidden_dim=expert_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        param_init=_experts_init(),
                        score_before_experts=False,
                        comm_backend=moe_comm_backend,
                        non_blocking_capacity_factor=non_blocking_capacity_factor,
                    ),
                    shared_experts=None,
                    load_balance_coeff=None,
                ),
            )
        )
    return layers


def _debugmodel(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> OLMoEModel.Config:
    dim = 256
    n_heads = 16
    n_kv_heads = 16
    head_dim = dim // n_heads
    n_layers = 4
    vocab_size = 2048
    num_experts = 8
    top_k = 2
    expert_hidden_dim = 128

    return OLMoEModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=_olmoe_norm(dim),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=10000.0,
            backend="complex",
        ),
        layers=_build_olmoe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        ),
    )


def _1b_7b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> OLMoEModel.Config:
    """OLMoE-1B-7B: 2048 dim, 16 layers, 16 heads, 64 experts, top-8."""
    dim = 2048
    n_heads = 16
    n_kv_heads = 16
    head_dim = 128
    n_layers = 16
    vocab_size = 50304
    num_experts = 64
    top_k = 8
    expert_hidden_dim = 1024

    return OLMoEModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=_olmoe_norm(dim),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=10000.0,
            backend="complex",
        ),
        layers=_build_olmoe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        ),
    )


olmoe_configs = {
    "debugmodel": _debugmodel,
    "1B-7B": _1b_7b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = olmoe_configs[flavor](
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            c.build().convert(config)
    return ModelSpec(
        name="olmoe",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_olmoe,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=OLMoEStateDictAdapter,
    )
