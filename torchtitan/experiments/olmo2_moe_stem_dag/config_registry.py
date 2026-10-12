# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for independently scaling OLMo2 MoE and STEM capacity."""

import math
from collections.abc import Callable
from dataclasses import fields, replace
from functools import partial
from typing import TypedDict

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.olmo2_moe.config_registry import (
    olmo2_moe_1b_e8g1t1_8l_midfine,
    olmo2_moe_1b_e8g1t1_8l_warmup,
    olmo2_moe_1b_e8g8t8_8l_midfine,
    olmo2_moe_1b_e8g8t8_8l_warmup,
)
from torchtitan.experiments.olmo2_moe.trainer import OLMo2MoETrainer
from torchtitan.experiments.stem_dag.model import StemDagFeedForward, StemEmbedding
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, TransformerBlock
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_experts_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_token_dispatcher_config,
)
from torchtitan.models.olmoe.model import OLMoERouter
from torchtitan.models.olmoe.moe_aux_loss import OLMoELoss
from torchtitan.protocols.model_spec import ModelSpec

from .model import OLMo2MoEStemDagModel, OLMo2MoEStemDagTransformerBlock, StemStrategy
from .moe import (
    make_projected_stem_experts_config,
    make_static_stem_experts_config,
    ProjectedStemMoE,
    StaticStemMoE,
)
from .parallelize import parallelize_olmo2_moe_stem_dag
from .trainer import OLMo2MoEStemDagTrainer, StemEmbeddingHyperparams

_EPS = 1e-6
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


class _Topology(TypedDict):
    dim: int
    n_heads: int
    n_layers: int
    vocab_size: int
    dense_hidden_dim: int
    expert_hidden_dim: int
    num_experts: int
    granularity: int
    top_k: int
    moe_layers: list[int]


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


def _static_stem_ffn(dim: int, hidden_dim: int) -> StemDagFeedForward.Config:
    return StemDagFeedForward.Config(
        w1=Linear.Config(
            in_features=dim,
            out_features=hidden_dim,
            param_init=_LINEAR_INIT,
        ),
        w2=Linear.Config(
            in_features=hidden_dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
        ),
        w3=Linear.Config(
            in_features=dim,
            out_features=hidden_dim,
            param_init=_LINEAR_INIT,
        ),
    )


def _router(
    *,
    dim: int,
    num_experts: int,
    top_k: int,
    aux_loss_layer_idx: int,
) -> OLMoERouter.Config:
    return OLMoERouter.Config(
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
    )


def _routed_moe(
    *,
    strategy: StemStrategy,
    with_stem: bool,
    dim: int,
    dense_hidden_dim: int,
    expert_hidden_dim: int,
    vocab_size: int,
    num_experts: int,
    granularity: int,
    top_k: int,
    aux_loss_layer_idx: int,
    moe_comm_backend: str,
):
    router = _router(
        dim=dim,
        num_experts=num_experts,
        top_k=top_k,
        aux_loss_layer_idx=aux_loss_layer_idx,
    )
    token_dispatcher = make_token_dispatcher_config(
        num_experts=num_experts,
        top_k=top_k,
        score_before_experts=False,
        comm_backend=moe_comm_backend,
    )
    if not with_stem:
        return make_moe_config(
            num_experts=num_experts,
            router=router,
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
        )
    if moe_comm_backend not in {"standard", "deduplicated"}:
        raise ValueError(
            "STEM routing supports moe_comm_backend='standard' or "
            f"'deduplicated', got {moe_comm_backend!r}"
        )
    if strategy == "static":
        return StaticStemMoE.Config(
            num_experts=num_experts,
            router=router,
            experts=make_static_stem_experts_config(
                dim=dim,
                hidden_dim=expert_hidden_dim,
                num_experts=num_experts,
                granularity=granularity,
                param_init=_experts_init(),
                token_dispatcher=token_dispatcher,
            ),
            shared_experts=None,
            load_balance_coeff=None,
        )

    if granularity != 1 or top_k != 1 or expert_hidden_dim != dense_hidden_dim:
        raise ValueError(
            "per_expert_projected STEM only supports granularity=1, top_k=1, "
            "and full-width experts"
        )
    stem_dim = 8 * math.ceil(dense_hidden_dim / (8 * num_experts))
    projected_init = {
        **_experts_init(),
        "stem_table_EVR": partial(
            nn.init.normal_,
            std=1.0 / math.sqrt(stem_dim),
        ),
        "stem_up_FR": nn.init.zeros_,
    }
    return ProjectedStemMoE.Config(
        num_experts=num_experts,
        router=router,
        experts=make_projected_stem_experts_config(
            dim=dim,
            hidden_dim=dense_hidden_dim,
            vocab_size=vocab_size,
            stem_dim=stem_dim,
            num_experts=num_experts,
            param_init=projected_init,
            token_dispatcher=token_dispatcher,
        ),
        shared_experts=None,
        load_balance_coeff=None,
    )


def _build_layers(
    *,
    strategy: StemStrategy,
    n_layers: int,
    dim: int,
    n_heads: int,
    head_dim: int,
    vocab_size: int,
    dense_hidden_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    granularity: int,
    top_k: int,
    moe_layers: list[int],
    stem_layers: list[int],
    attn_backend: str,
    moe_comm_backend: str,
) -> list[TransformerBlock.Config]:
    moe_layer_set = set(moe_layers)
    stem_layer_set = set(stem_layers)
    if len(moe_layer_set) != len(moe_layers):
        raise ValueError("moe_layers must not contain duplicates")
    if len(stem_layer_set) != len(stem_layers):
        raise ValueError("stem_layers must not contain duplicates")
    valid_layer_ids = set(range(n_layers))
    invalid_moe_layers = sorted(moe_layer_set - valid_layer_ids)
    if invalid_moe_layers:
        raise ValueError(f"MoE layers out of range: {invalid_moe_layers}")
    invalid_stem_layers = sorted(stem_layer_set - valid_layer_ids)
    if invalid_stem_layers:
        raise ValueError(f"STEM layers out of range: {invalid_stem_layers}")
    if strategy == "per_expert_projected" and not stem_layer_set <= moe_layer_set:
        invalid_layers = sorted(stem_layer_set - moe_layer_set)
        raise ValueError(
            f"projected per-expert STEM layers must be MoE layers, got {invalid_layers}"
        )

    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    aux_loss_layer_idx = 0
    for layer_id in range(n_layers):
        attention = make_gqa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            head_dim=head_dim,
            wqkv_param_init=_LINEAR_INIT,
            wo_param_init=_LINEAR_INIT,
            inner_attention=inner_attention,
            mask_type=mask_type,
            rope_backend="complex",
            qk_norm=_norm([n_heads, head_dim]),
        )
        common = {
            "attention_norm": _norm(dim),
            "ffn_norm": _norm(dim),
            "attention": attention,
        }
        if layer_id in moe_layer_set:
            layer = OLMo2MoEStemDagTransformerBlock.Config(
                **common,
                moe=_routed_moe(
                    strategy=strategy,
                    with_stem=layer_id in stem_layer_set,
                    dim=dim,
                    dense_hidden_dim=dense_hidden_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    vocab_size=vocab_size,
                    num_experts=num_experts,
                    granularity=granularity,
                    top_k=top_k,
                    aux_loss_layer_idx=aux_loss_layer_idx,
                    moe_comm_backend=moe_comm_backend,
                ),
            )
            aux_loss_layer_idx += 1
        else:
            feed_forward = (
                _static_stem_ffn(dim, dense_hidden_dim)
                if strategy == "static" and layer_id in stem_layer_set
                else make_ffn_config(
                    dim=dim,
                    hidden_dim=dense_hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_LINEAR_INIT,
                )
            )
            layer = OLMo2MoEStemDagTransformerBlock.Config(
                **common,
                feed_forward=feed_forward,
            )
        layers.append(layer)
    return layers


def _make_model(
    *,
    strategy: StemStrategy,
    dim: int,
    n_heads: int,
    n_layers: int,
    vocab_size: int,
    dense_hidden_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    granularity: int,
    top_k: int,
    moe_layers: list[int],
    stem_layers: list[int] | None,
    attn_backend: str,
    moe_comm_backend: str,
) -> OLMo2MoEStemDagModel.Config:
    if strategy not in {"static", "per_expert_projected"}:
        raise ValueError(f"unknown stem_strategy {strategy!r}")
    if granularity <= 0:
        raise ValueError(f"granularity must be positive, got {granularity}")
    if num_experts <= 0 or num_experts % granularity != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be positive and divisible by "
            f"granularity ({granularity})"
        )
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
    if strategy == "static" and dense_hidden_dim != granularity * expert_hidden_dim:
        raise ValueError(
            f"dense_hidden_dim ({dense_hidden_dim}) must equal granularity "
            f"({granularity}) times expert_hidden_dim ({expert_hidden_dim})"
        )
    if stem_layers is None:
        stem_layers = list(moe_layers)
    head_dim = dim // n_heads
    return OLMo2MoEStemDagModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        stem_strategy=strategy,
        stem_layers=stem_layers,
        stem_embeddings=(
            {
                str(layer_id): StemEmbedding.Config(
                    num_embeddings=vocab_size,
                    embedding_dim=dense_hidden_dim,
                )
                for layer_id in stem_layers
            }
            if strategy == "static"
            else {}
        ),
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
            strategy=strategy,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            dense_hidden_dim=dense_hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            granularity=granularity,
            top_k=top_k,
            moe_layers=moe_layers,
            stem_layers=stem_layers,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
        ),
    )


_TOPOLOGIES: dict[str, _Topology] = {
    "debugmodel": {
        "dim": 128,
        "n_heads": 8,
        "n_layers": 4,
        "vocab_size": 256,
        "dense_hidden_dim": 512,
        "expert_hidden_dim": 512,
        "num_experts": 4,
        "granularity": 1,
        "top_k": 1,
        "moe_layers": [1, 2],
    },
    "debugmodel-E2G2T2-STEM2": {
        "dim": 128,
        "n_heads": 8,
        "n_layers": 4,
        "vocab_size": 256,
        "dense_hidden_dim": 512,
        "expert_hidden_dim": 256,
        "num_experts": 4,
        "granularity": 2,
        "top_k": 2,
        "moe_layers": [1, 3],
    },
    "1B-E8G1T1-8L": {
        "dim": 2048,
        "n_heads": 16,
        "n_layers": 16,
        "vocab_size": 100352,
        "dense_hidden_dim": 8192,
        "expert_hidden_dim": 8192,
        "num_experts": 8,
        "granularity": 1,
        "top_k": 1,
        "moe_layers": [1, 3, 5, 7, 9, 11, 13, 15],
    },
    "1B-E8G8T8-8L-STEM12": {
        "dim": 2048,
        "n_heads": 16,
        "n_layers": 16,
        "vocab_size": 100352,
        "dense_hidden_dim": 8192,
        "expert_hidden_dim": 1024,
        "num_experts": 64,
        "granularity": 8,
        "top_k": 8,
        "moe_layers": [1, 3, 5, 7, 9, 11, 13, 15],
    },
}

_STATIC_GRANULAR_STEM_LAYERS = {
    "debugmodel-E2G2T2-STEM2": [1, 2],
    "1B-E8G8T8-8L-STEM12": list(range(1, 13)),
}


def model_registry(
    flavor: str,
    *,
    stem_strategy: StemStrategy = "static",
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
    stem_layers: list[int] | None = None,
) -> ModelSpec:
    try:
        topology = _TOPOLOGIES[flavor]
    except KeyError as error:
        raise ValueError(f"unknown OLMo2 MoE STEM flavor {flavor!r}") from error
    if flavor in _STATIC_GRANULAR_STEM_LAYERS:
        if stem_strategy != "static":
            raise ValueError(f"{flavor} only supports static STEM")
        if stem_layers is None:
            stem_layers = _STATIC_GRANULAR_STEM_LAYERS[flavor]
    config = _make_model(
        strategy=stem_strategy,
        stem_layers=stem_layers,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        **topology,
    )
    return ModelSpec(
        name="olmo2_moe_stem_dag",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_olmo2_moe_stem_dag,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _debug_config(
    strategy: StemStrategy,
    *,
    flavor: str = "debugmodel",
) -> OLMo2MoEStemDagTrainer.Config:
    topology = _TOPOLOGIES[flavor]
    return OLMo2MoEStemDagTrainer.Config(
        loss=OLMoELoss.Config(
            moe_loss_weight=0.01,
            moe_zloss_weight=0.001,
            num_experts=topology["num_experts"],
            top_k=topology["top_k"],
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry(
            flavor,
            stem_strategy=strategy,
            moe_comm_backend="deduplicated",
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(local_batch_size=1, seq_len=128, steps=2),
        parallelism=ParallelismConfig(expert_parallel_degree=1),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
    )


def olmo2_moe_stem_dag_static_debugmodel() -> OLMo2MoEStemDagTrainer.Config:
    return _debug_config("static")


def olmo2_moe_stem_dag_projected_debugmodel() -> OLMo2MoEStemDagTrainer.Config:
    return _debug_config("per_expert_projected")


def olmo2_moe_stem_dag_static_granular_debugmodel() -> (OLMo2MoEStemDagTrainer.Config):
    return _debug_config("static", flavor="debugmodel-E2G2T2-STEM2")


def olmo2_moe_stem_dag_debugmodel() -> OLMo2MoEStemDagTrainer.Config:
    """Compatibility alias for the static debug model."""
    return olmo2_moe_stem_dag_static_debugmodel()


def _e8g1t1_8l_config(
    strategy: StemStrategy,
    *,
    warmup: bool,
) -> OLMo2MoEStemDagTrainer.Config:
    flavor = "1B-E8G1T1-8L"
    base = (
        olmo2_moe_1b_e8g1t1_8l_warmup() if warmup else olmo2_moe_1b_e8g1t1_8l_midfine()
    )
    values = {
        item.name: getattr(base, item.name) for item in fields(OLMo2MoETrainer.Config)
    }
    values["model_spec"] = model_registry(
        flavor,
        stem_strategy=strategy,
        attn_backend="sdpa",
        moe_comm_backend="deduplicated",
    )
    missing_prefixes: list[str] = []
    if warmup:
        missing_prefixes = (
            ["_stem_embeddings."]
            if strategy == "static"
            else [
                f"layers.{layer_id}.moe.experts.stem_"
                for layer_id in _TOPOLOGIES[flavor]["moe_layers"]
            ]
        )
    values["checkpoint"] = replace(
        base.checkpoint,
        initial_load_allowed_missing_model_key_prefixes=missing_prefixes,
    )
    return OLMo2MoEStemDagTrainer.Config(
        **values,
        stem_hyperparams=StemEmbeddingHyperparams(
            lr=4e-4 if warmup else 3e-4,
            weight_decay=1e-4,
        ),
    )


def olmo2_moe_stem_dag_static_1b_e8g1t1_8l_midfine() -> OLMo2MoEStemDagTrainer.Config:
    """Add static STEM to the E8G1T1 eight-layer midfine base config."""
    return _e8g1t1_8l_config("static", warmup=False)


def olmo2_moe_stem_dag_static_1b_e8g1t1_8l_warmup() -> OLMo2MoEStemDagTrainer.Config:
    """Warm up static STEM on the E8G1T1 eight-layer MoE."""
    return _e8g1t1_8l_config("static", warmup=True)


def olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_midfine() -> (
    OLMo2MoEStemDagTrainer.Config
):
    """Add projected per-expert STEM to the E8G1T1 eight-layer base config."""
    return _e8g1t1_8l_config("per_expert_projected", warmup=False)


def olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_warmup() -> OLMo2MoEStemDagTrainer.Config:
    """Warm up projected per-expert STEM on the E8G1T1 eight-layer MoE."""
    return _e8g1t1_8l_config("per_expert_projected", warmup=True)


def _static_e8g8t8_config(
    *,
    base: OLMo2MoETrainer.Config,
    flavor: str,
    stem_lr: float,
    allow_missing_stem: bool = False,
) -> OLMo2MoEStemDagTrainer.Config:
    values = {
        item.name: getattr(base, item.name) for item in fields(OLMo2MoETrainer.Config)
    }
    values["model_spec"] = model_registry(
        flavor,
        stem_strategy="static",
        attn_backend="sdpa",
        moe_comm_backend="deduplicated",
    )
    values["checkpoint"] = replace(
        base.checkpoint,
        initial_load_allowed_missing_model_key_prefixes=(
            ["_stem_embeddings."] if allow_missing_stem else []
        ),
    )
    return OLMo2MoEStemDagTrainer.Config(
        **values,
        stem_hyperparams=StemEmbeddingHyperparams(
            lr=stem_lr,
            weight_decay=1e-4,
        ),
    )


def olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_warmup() -> (
    OLMo2MoEStemDagTrainer.Config
):
    """Add zero-initialized STAPLE-12 tables to an E8G8T8 checkpoint."""
    return _static_e8g8t8_config(
        base=olmo2_moe_1b_e8g8t8_8l_warmup(),
        flavor="1B-E8G8T8-8L-STEM12",
        stem_lr=4e-4,
        allow_missing_stem=True,
    )


def olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_midfine() -> (
    OLMo2MoEStemDagTrainer.Config
):
    """Continue a warmed-up STAPLE-12 E8G8T8 model on the STEM mixture."""
    return _static_e8g8t8_config(
        base=olmo2_moe_1b_e8g8t8_8l_midfine(),
        flavor="1B-E8G8T8-8L-STEM12",
        stem_lr=3e-4,
    )


__all__ = [
    "OLMo2MoEStemDagModel",
    "model_registry",
]
