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
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
)
from torchtitan.models.olmoe.model import OLMoERouter
from torchtitan.models.olmoe.parallelize import parallelize_olmoe
from torchtitan.protocols.model_spec import ModelSpec

from .model import OLMo2MoEModel, OLMo2MoETransformerBlock

_EPS = 1e-6
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _norm(dim: int | list[int]) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _experts_init() -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
        "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
    }


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    dense_hidden_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    moe_layers: list[int],
    attn_backend: str,
    moe_comm_backend: str,
) -> list[TransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    moe_layer_set = set(moe_layers)
    aux_loss_layer_idx = 0
    layers = []
    for layer_id in range(n_layers):
        attention_norm = _norm(dim)
        ffn_norm = _norm(dim)
        attention = make_gqa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            wqkv_param_init=_LINEAR_INIT,
            wo_param_init=_LINEAR_INIT,
            inner_attention=inner_attention,
            mask_type=mask_type,
            rope_backend="complex",
            qk_norm=_norm([n_heads, head_dim]),
        )
        if layer_id in moe_layer_set:
            layer = OLMo2MoETransformerBlock.Config(
                attention_norm=attention_norm,
                ffn_norm=ffn_norm,
                attention=attention,
                moe=make_moe_config(
                    num_experts=num_experts,
                    router=OLMoERouter.Config(
                        num_experts=num_experts,
                        aux_loss_layer_idx=aux_loss_layer_idx,
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
                    ),
                    shared_experts=None,
                    load_balance_coeff=None,
                ),
            )
            aux_loss_layer_idx += 1
        else:
            layer = OLMo2MoETransformerBlock.Config(
                attention_norm=attention_norm,
                ffn_norm=ffn_norm,
                attention=attention,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=dense_hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_LINEAR_INIT,
                ),
            )
        layers.append(layer)
    return layers


def _make_model(
    *,
    dim: int,
    n_heads: int,
    n_layers: int,
    vocab_size: int,
    dense_hidden_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    moe_layers: list[int],
    attn_backend: str,
    moe_comm_backend: str,
) -> OLMo2MoEModel.Config:
    head_dim = dim // n_heads
    return OLMo2MoEModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=_norm(dim),
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
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            head_dim=head_dim,
            dense_hidden_dim=dense_hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            moe_layers=moe_layers,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
        ),
    )


def _debugmodel(attn_backend: str, moe_comm_backend: str) -> OLMo2MoEModel.Config:
    return _make_model(
        dim=128,
        n_heads=8,
        n_layers=4,
        vocab_size=256,
        dense_hidden_dim=512,
        expert_hidden_dim=256,
        num_experts=4,
        top_k=2,
        moe_layers=[1, 2],
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
    )


def _1b_e8g1t1_8l(attn_backend: str, moe_comm_backend: str) -> OLMo2MoEModel.Config:
    return _make_model(
        dim=2048,
        n_heads=16,
        n_layers=16,
        vocab_size=100352,
        dense_hidden_dim=8192,
        expert_hidden_dim=8192,
        num_experts=8,
        top_k=1,
        moe_layers=[1, 3, 5, 7, 9, 11, 13, 15],
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
    )


def _1b_e8g8t8_8l(attn_backend: str, moe_comm_backend: str) -> OLMo2MoEModel.Config:
    return _make_model(
        dim=2048,
        n_heads=16,
        n_layers=16,
        vocab_size=100352,
        dense_hidden_dim=8192,
        expert_hidden_dim=1024,
        num_experts=64,
        top_k=8,
        moe_layers=[1, 3, 5, 7, 9, 11, 13, 15],
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
    )


_MODEL_CONFIGS = {
    "debugmodel": _debugmodel,
    "1B-E8G1T1-8L": _1b_e8g1t1_8l,
    "1B-E8G8T8-8L": _1b_e8g8t8_8l,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
) -> ModelSpec:
    config = _MODEL_CONFIGS[flavor](attn_backend, moe_comm_backend)
    return ModelSpec(
        name="olmo2_moe",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_olmoe,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


__all__ = ["OLMo2MoEModel", "model_registry"]
