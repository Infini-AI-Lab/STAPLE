# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for Engram-OLMo2.

Mirrors the core OLMo2 trainer configs, adding an optimizer param group for the
Engram embedding tables: per the paper, table embeddings train with a higher LR
(5x) and no weight decay (paper Section 4.1).
"""

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.jsonl_datasets import JSONLDataLoader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler

from .model_registry import model_registry
from .trainer import EngramEmbeddingHyperparams, EngramTrainer

# Token embeddings train without weight decay (OLMo2 recipe).
_NO_WD_TOKEN_EMBEDDINGS = ParamGroupConfig(
    pattern=r"tok_embeddings\.",
    weight_decay_multiplier=0.0,
)
# Engram embedding tables train with a higher LR (paper Section 4.1: 5x) and no
# weight decay, on their own warmup schedule. EngramTrainer injects the param
# group + per-group LR lambda from these absolute values (matching stem_dag's
# StemEmbeddingHyperparams) and logs the group's lr as ``lr/engram_embeddings``.

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


def engram_olmo2_debugmodel() -> EngramTrainer.Config:
    return EngramTrainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(
            lr=8e-4,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
        ),
        engram_hyperparams=EngramEmbeddingHyperparams(
            lr=4e-3,  # 5x backbone lr (paper Section 4.1)
            weight_decay=0.0,
            warmup_steps=4,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def engram_olmo2_1b_prefine() -> EngramTrainer.Config:
    """Engram-OLMo2 1B: OLMo2-1B backbone with Engram at layers 2 and 7.

    Set STEM_DATA_ROOT to override the JSONL data root. The backbone weights can
    be initialized from a standard OLMo2-1B checkpoint (Engram params init fresh).
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/pretraining",
    )
    return EngramTrainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/olmo2-1b-base-token4T",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("1B"),
        optimizer=OptimizersContainer.Config(
            lr=7.44e-5,
            weight_decay=0.1,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
        ),
        engram_hyperparams=EngramEmbeddingHyperparams(
            lr=4e-4,  # ~5x backbone lr (paper Section 4.1)
            weight_decay=0.0,
            warmup_steps=1000,  # warm the fresh memory tables in slower than backbone
            min_lr_factor=0.6,
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
            sources={"dclm-baseline_shuffled": 1.0},
            seed=777,
            add_bos=True,
            add_eos=True,
            n_views=2,
            prefetch_size=1024,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            fsdp_reshard_after_forward="default",
        ),
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            interval=10000,
            initial_load_allowed_missing_model_key_prefixes=[
                "layers.2.engram.",
                "layers.7.engram.",
            ],
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def engram_olmo2_1b_stem_dag_l12_prefine() -> EngramTrainer.Config:
    """Two-layer Engram prefine matched to full-table STEM-DAG L12 size."""
    config = engram_olmo2_1b_prefine()
    config.model_spec = model_registry("1B_stem_dag_l12_size")
    return config


def engram_olmo2_1b_midfine() -> EngramTrainer.Config:
    """Engram-OLMo2 1B: OLMo2-1B backbone with Engram at layers 2 and 7.

    Set STEM_DATA_ROOT to override the JSONL data root. The backbone weights can
    be initialized from a standard OLMo2-1B checkpoint (Engram params init fresh).
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled",
    )
    return EngramTrainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/olmo2-1b-base-token4T",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("1B"),
        optimizer=OptimizersContainer.Config(
            lr=7.44e-5,
            weight_decay=0.1,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
        ),
        engram_hyperparams=EngramEmbeddingHyperparams(
            lr=3e-4,
            weight_decay=0.0,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=1.0,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
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
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            fsdp_reshard_after_forward="default",
        ),
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            interval=10000,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def engram_olmo2_1b_stem_dag_l12_midfine() -> EngramTrainer.Config:
    """Two-layer Engram midfine matched to full-table STEM-DAG L12 size."""
    config = engram_olmo2_1b_midfine()
    config.model_spec = model_registry("1B_stem_dag_l12_size")
    return config
