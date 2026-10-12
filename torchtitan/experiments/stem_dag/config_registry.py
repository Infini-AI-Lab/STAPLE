# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config registry for STEM DAG OLMo2 experiments."""

import math
import os
from functools import partial

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.jsonl_datasets import JSONLDataLoader
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.config_utils import get_attention_config, make_gqa_config
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler

from .loss import StemDagLoss
from .model import (
    LowRankStemEmbedding,
    StemDagFeedForward,
    StemDagLowRankOLMo2Model,
    StemDagOLMo2Model,
    StemDagTransformerBlock,
    StemEmbedding,
)
from .parallelize import parallelize_stem_dag, parallelize_stem_dag_lowrank
from .trainer import StemDagTrainer, StemEmbeddingHyperparams

# ---------------------------------------------------------------------------
# Init helpers (same as olmo2/__init__.py)
# ---------------------------------------------------------------------------

_EPS = 1e-6
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _olmo2_norm(dim: int | list[int]) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


# MLflow tracking for STEM experiments. Logs to the Nile MLflow server under the
# "rjsadhuk/stem" experiment; both can be overridden via the standard MLflow env
# vars (MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME) at launch time.
_MLFLOW_EXPERIMENT_NAME = "rjsadhuk/stem"
_MLFLOW_TRACKING_URI = "https://prod.us-west-2.mlflow.nile.amazon.dev"


def _stem_metrics_config(log_freq: int = 10) -> MetricsProcessor.Config:
    return MetricsProcessor.Config(
        log_freq=log_freq,
        enable_tensorboard=False,
        enable_mlflow=True,
        mlflow_experiment_name=_MLFLOW_EXPERIMENT_NAME,
        mlflow_tracking_uri=_MLFLOW_TRACKING_URI,
    )


def _make_stem_dag_ffn_config(dim: int, hidden_dim: int) -> StemDagFeedForward.Config:
    return StemDagFeedForward.Config(
        w1=Linear.Config(
            in_features=dim, out_features=hidden_dim, param_init=_LINEAR_INIT
        ),
        w2=Linear.Config(
            in_features=hidden_dim, out_features=dim, param_init=_LINEAR_INIT
        ),
        w3=Linear.Config(
            in_features=dim, out_features=hidden_dim, param_init=_LINEAR_INIT
        ),
    )


def _build_stem_dag_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    hidden_dim: int,
    head_dim: int,
    attn_backend: str,
    stem_layers: list[int],
):
    from torchtitan.models.common.config_utils import make_ffn_config

    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        is_stem = layer_id in stem_layers
        if is_stem:
            ffn_config = _make_stem_dag_ffn_config(dim, hidden_dim)
        else:
            ffn_config = make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_LINEAR_INIT,
            )
        layers.append(
            StemDagTransformerBlock.Config(
                is_stem_layer=is_stem,
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
                feed_forward=ffn_config,
            )
        )
    return layers


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

# Default stem layers
_DEFAULT_STEM_LAYERS_7B = list(range(1, 17))
_DEFAULT_STEM_LAYERS_1B = list(range(1, 13))


def _1b_stem_dag(
    attn_backend: str = "sdpa", stem_layers: list[int] | None = None
) -> StemDagOLMo2Model.Config:
    dim = 2048
    n_heads = 16
    n_kv_heads = 16
    head_dim = 128
    n_layers = 16
    vocab_size = 100352
    hidden_dim = compute_ffn_hidden_dim(dim, multiple_of=256, ffn_dim_multiplier=1.5)
    if stem_layers is None:
        stem_layers = _DEFAULT_STEM_LAYERS_1B

    stem_embeddings = {
        str(i): StemEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            param_init={"weight": nn.init.zeros_},
        )
        for i in stem_layers
    }

    return StemDagOLMo2Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        stem_layers=stem_layers,
        stem_embeddings=stem_embeddings,
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
        layers=_build_stem_dag_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
            stem_layers=stem_layers,
        ),
    )


# Default low-rank stem embedding dimension (table is vocab_size x stem_dim,
# up-projected per-layer to ffn_hidden_dim). Much smaller than the full
# StemEmbedding (vocab_size x ffn_hidden_dim).
_DEFAULT_STEM_DIM = 256


def _make_lowrank_stem_embeddings(
    *, vocab_size: int, stem_dim: int, hidden_dim: int, stem_layers: list[int]
) -> dict[str, LowRankStemEmbedding.Config]:
    """Build per-layer factored stem embeddings.

    ``table``: (vocab_size, stem_dim) with normal init so it trains freely.
    ``up_proj``: (stem_dim -> hidden_dim), bias-free, zero-initialized so the
    stem signal ``y = up_proj(table(tokens))`` is 0 at init and the model
    starts equivalent to the base when loading a pretrained checkpoint.
    """
    return {
        str(i): LowRankStemEmbedding.Config(
            table=Embedding.Config(
                num_embeddings=vocab_size,
                embedding_dim=stem_dim,
                param_init={
                    "weight": partial(nn.init.normal_, std=1.0 / math.sqrt(stem_dim))
                },
            ),
            up_proj=Linear.Config(
                in_features=stem_dim,
                out_features=hidden_dim,
                bias=False,
                param_init={"weight": nn.init.zeros_},
            ),
        )
        for i in stem_layers
    }


def _1b_stem_dag_lowrank(
    attn_backend: str = "sdpa",
    stem_layers: list[int] | None = None,
    stem_dim: int = _DEFAULT_STEM_DIM,
) -> StemDagLowRankOLMo2Model.Config:
    """OLMo2-1B low-rank STEM DAG: (vocab x stem_dim) table + per-layer up-proj.

    Mirrors ``_1b_stem_dag`` but replaces each full (vocab x hidden_dim)
    StemEmbedding with a factored LowRankStemEmbedding.
    """
    dim = 2048
    n_heads = 16
    n_kv_heads = 16
    head_dim = 128
    n_layers = 16
    vocab_size = 100352
    hidden_dim = compute_ffn_hidden_dim(dim, multiple_of=256, ffn_dim_multiplier=1.5)
    if stem_layers is None:
        stem_layers = _DEFAULT_STEM_LAYERS_1B

    stem_embeddings = _make_lowrank_stem_embeddings(
        vocab_size=vocab_size,
        stem_dim=stem_dim,
        hidden_dim=hidden_dim,
        stem_layers=stem_layers,
    )

    return StemDagLowRankOLMo2Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        stem_layers=stem_layers,
        stem_embeddings=stem_embeddings,
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
        layers=_build_stem_dag_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
            stem_layers=stem_layers,
        ),
    )


def _7b_stem_dag(
    attn_backend: str = "sdpa", stem_layers: list[int] | None = None
) -> StemDagOLMo2Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 32
    head_dim = 128
    n_layers = 32
    vocab_size = 100352
    hidden_dim = compute_ffn_hidden_dim(dim, multiple_of=256)
    if stem_layers is None:
        stem_layers = _DEFAULT_STEM_LAYERS_7B

    stem_embeddings = {
        str(i): StemEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            param_init={"weight": nn.init.zeros_},
        )
        for i in stem_layers
    }

    return StemDagOLMo2Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        stem_layers=stem_layers,
        stem_embeddings=stem_embeddings,
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
        layers=_build_stem_dag_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
            stem_layers=stem_layers,
        ),
    )


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    stem_layers: list[int] | None = None,
    stem_dim: int | None = None,
) -> ModelSpec:
    configs = {
        "1B_stem_dag": _1b_stem_dag,
        "7B_stem_dag": _7b_stem_dag,
        "1B_stem_dag_lowrank": _1b_stem_dag_lowrank,
    }
    # Low-rank flavors use plain FSDP for the (small) factored stem tables;
    # the full-table flavors use the EP dimension-parallel parallelize_fn.
    lowrank_flavors = {"1B_stem_dag_lowrank"}

    kwargs = {"attn_backend": attn_backend}
    if stem_layers is not None:
        kwargs["stem_layers"] = stem_layers
    if stem_dim is not None:
        if flavor not in lowrank_flavors:
            raise ValueError(
                f"stem_dim is only valid for low-rank flavors, got flavor={flavor!r}."
            )
        kwargs["stem_dim"] = stem_dim
    config = configs[flavor](**kwargs)

    parallelize_fn = (
        parallelize_stem_dag_lowrank
        if flavor in lowrank_flavors
        else parallelize_stem_dag
    )
    return ModelSpec(
        name="stem_dag",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_fn,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


# ---------------------------------------------------------------------------
# Data sources (same as olmo2 config_registry)
# ---------------------------------------------------------------------------

_STEM_SOURCES: dict[str, float] = {
    "cranecode": 10.0,
    "stack_edu": 10.0,
    "dolmino-math": 10.7,
    "cranemath": 5.63,
    "megamatt": 1.73,
    "tinymath-mind": 0.9,
    "tinymath-pot": 0.24,
    "reddit_to_flashcards": 5.9,
    "nemotron-synth-qa": 5.0,
    "wiki_to_rcqa": 3.0,
    "qwq-reasoning-traces": 1.87,
    "general_reasoning_mix": 1.87,
    "openthoughts2": 1.25,
    "code-meta-reasoning": 0.46,
    "math-meta-reasoning": 0.38,
    "gemini-reasoning-traces": 0.25,
    "program_verifiable": 0.16,
    "dolmino_1-flan": 5.0,
    "tulu-3-sft": 1.1,
    "common_crawl-high-quality": 22.5,
    "stem-heavy-crawl": 5.0,
    "olmocr_science_pdfs": 5.0,
}


# ---------------------------------------------------------------------------
# Trainer configs
# ---------------------------------------------------------------------------

# Token embeddings (``tok_embeddings.weight``) train without weight decay, as in
# the base OLMo2 recipe. This is disjoint from the stem-embedding param group
# (``_stem_embeddings.``) that StemDagTrainer prepends from stem_hyperparams, so
# the stem tables keep their own (typically nonzero) weight decay.
_NO_WD_TOKEN_EMBEDDINGS = ParamGroupConfig(
    pattern=r"tok_embeddings\.",
    weight_decay_multiplier=0.0,
)


def stem_dag_1b_prefine() -> StemDagTrainer.Config:
    """STEM DAG OLMo2-1B with JSONL dataloader.

    12 stem layers [1, ..., 12], each with a (vocab_size x 8192) embedding table.
    Uses full data mixture with deficit-driven source mixing.
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/pretraining",
    )
    return StemDagTrainer.Config(
        # Freeze the stem up-projection (w3) during the prefine/warmup phase.
        # w3 stays in the optimizer so its (unchanged) optimizer state is still
        # checkpointed and can be resumed by a later stage that unfreezes it.
        freeze_stem_up_proj=True,
        loss=StemDagLoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/olmo2-1b-base-token4T",
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("1B_stem_dag"),
        stem_hyperparams=StemEmbeddingHyperparams(
            lr=4e-4,
            weight_decay=1e-4,
            warmup_steps=500,
            min_lr_factor=0.6,
        ),
        optimizer=OptimizersContainer.Config(
            lr=7.44e-5,
            weight_decay=0.1,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.91,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=50000,
        ),
        dataloader=JSONLDataLoader.Config(
            root_dir=data_root,
            sources={
                "dclm-baseline_shuffled": 0.963,
                # "starcoder_shuffled": 0.022,
                # "pes2o_shuffled": 0.015,
            },
            seed=777,
            add_bos=True,
            add_eos=True,
            n_views=2,
            prefetch_size=1024,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            data_parallel_replicate_degree=1,
            tensor_parallel_degree=1,
            expert_parallel_degree=8,
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            interval=10000,
            initial_load_allowed_missing_model_key_prefixes=["_stem_embeddings."],
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def stem_dag_1b_l4_prefine() -> StemDagTrainer.Config:
    """Same as ``stem_dag_1b_prefine`` but with stem layers [1..4] only."""
    config = stem_dag_1b_prefine()
    config.model_spec = model_registry("1B_stem_dag", stem_layers=list(range(1, 5)))
    return config


def stem_dag_1b_l8_prefine() -> StemDagTrainer.Config:
    """Same as ``stem_dag_1b_prefine`` but with stem layers [1..8] only."""
    config = stem_dag_1b_prefine()
    config.model_spec = model_registry("1B_stem_dag", stem_layers=list(range(1, 9)))
    return config


def _stem_dag_lowrank_1b_prefine(stem_dim: int) -> StemDagTrainer.Config:
    config = stem_dag_1b_prefine()
    config.model_spec = model_registry("1B_stem_dag_lowrank", stem_dim=stem_dim)
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=1,
        tensor_parallel_degree=1,
        expert_parallel_degree=1,
    )
    return config


def stem_dag_lowrank_1b_r512_prefine() -> StemDagTrainer.Config:
    return _stem_dag_lowrank_1b_prefine(512)


def _stem_dag_lowrank_1b_midfine(stem_dim: int) -> StemDagTrainer.Config:
    config = stem_dag_1b_midfine()
    config.model_spec = model_registry("1B_stem_dag_lowrank", stem_dim=stem_dim)
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=1,
        tensor_parallel_degree=1,
        expert_parallel_degree=1,
    )
    return config


def stem_dag_lowrank_1b_r512_midfine() -> StemDagTrainer.Config:
    return _stem_dag_lowrank_1b_midfine(512)


def stem_dag_1b_midfine() -> StemDagTrainer.Config:
    """STEM DAG OLMo2-1B with JSONL dataloader.

    12 stem layers [1, ..., 12], each with a (vocab_size x 8192) embedding table.
    Uses full data mixture with deficit-driven source mixing.
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled",
    )
    return StemDagTrainer.Config(
        loss=StemDagLoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/olmo2-1b-base-token4T",
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("1B_stem_dag"),
        stem_hyperparams=StemEmbeddingHyperparams(
            lr=3e-4,
            weight_decay=1e-4,
        ),
        optimizer=OptimizersContainer.Config(
            lr=7.44e-5,
            weight_decay=0.1,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=1.0,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=4096,
            steps=50000,
        ),
        dataloader=JSONLDataLoader.Config(
            root_dir=data_root,
            sources=_STEM_SOURCES,
            seed=777,
            add_bos=True,
            add_eos=True,
            n_views=2,
            prefetch_size=1024,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            data_parallel_replicate_degree=1,
            tensor_parallel_degree=1,
            expert_parallel_degree=8,
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            interval=10000,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def stem_dag_1b_l4_midfine() -> StemDagTrainer.Config:
    """Same as ``stem_dag_1b_midfine`` but with stem layers [1..4] only."""
    config = stem_dag_1b_midfine()
    config.model_spec = model_registry("1B_stem_dag", stem_layers=list(range(1, 5)))
    return config


def stem_dag_1b_l8_midfine() -> StemDagTrainer.Config:
    """Same as ``stem_dag_1b_midfine`` but with stem layers [1..8] only."""
    config = stem_dag_1b_midfine()
    config.model_spec = model_registry("1B_stem_dag", stem_layers=list(range(1, 9)))
    return config


def stem_dag_7b_prefine() -> StemDagTrainer.Config:
    """STEM DAG OLMo2-7B with JSONL dataloader.

    16 stem layers [1, ..., 16], each with a vocab_size x hidden_dim embedding
    table. Uses deficit-driven source mixing and ZeRO-2.
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/pretraining",
    )
    return StemDagTrainer.Config(
        # Freeze the stem up-projection (w3) during the prefine/warmup phase.
        # w3 stays in the optimizer so its (unchanged) optimizer state is still
        # checkpointed and can be resumed by a later stage that unfreezes it.
        freeze_stem_up_proj=True,
        stem_hyperparams=StemEmbeddingHyperparams(
            lr=6e-4,
            weight_decay=1e-4,
            warmup_steps=500,
            min_lr_factor=0.333,
        ),
        loss=StemDagLoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMo2-7B-base-token5T",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("7B_stem_dag"),
        optimizer=OptimizersContainer.Config(
            lr=6.1858e-5,
            weight_decay=0.1,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.9,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=1024,
            seq_len=4096,
            steps=25000,
        ),
        dataloader=JSONLDataLoader.Config(
            root_dir=data_root,
            sources={
                "dclm-baseline_shuffled": 0.963,
                "starcoder_shuffled": 0.022,
                "pes2o_shuffled": 0.015,
            },
            seed=777,
            add_bos=True,
            add_eos=True,
            n_views=2,
            prefetch_size=1024,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            data_parallel_replicate_degree=1,
            tensor_parallel_degree=1,
            expert_parallel_degree=8,
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=1000,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def stem_dag_7b_midfine() -> StemDagTrainer.Config:
    """STEM DAG OLMo2-7B with JSONL dataloader.

    16 stem layers [1, ..., 16], each with a vocab_size x hidden_dim embedding
    table. Uses deficit-driven source mixing and ZeRO-2.
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled",
    )
    return StemDagTrainer.Config(
        loss=StemDagLoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMo2-7B-base-token5T",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=50,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=5,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("7B_stem_dag"),
        optimizer=OptimizersContainer.Config(
            lr=6.1858e-5,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=1.0,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=25000,
        ),
        dataloader=JSONLDataLoader.Config(
            root_dir=data_root,
            sources=_STEM_SOURCES,
            seed=777,
            add_bos=True,
            add_eos=True,
            n_views=2,
            prefetch_size=1024,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            data_parallel_replicate_degree=1,
            tensor_parallel_degree=1,
            expert_parallel_degree=8,
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            enable=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )
