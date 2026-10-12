# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from datasets import load_dataset

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
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.jsonl_datasets import JSONLDataLoader
from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry


# ---------------------------------------------------------------------------
# STEM-pt midtraining dataset registration
# ---------------------------------------------------------------------------
# Each source is a directory of pre-shuffled JSONL chunk files under root_dir.
# We register them as HuggingFace "json" datasets loaded from local glob paths.

# Default root; override via STEM_DATA_ROOT env var
_STEM_DATA_ROOT = os.environ.get("STEM_DATA_ROOT", "/dev/shm/dolmino-mix_shuffled")

# Source name -> sampling weight (from stem_dag_olmo2_7B_midfine.yaml)
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


def _process_jsonl_text(sample: dict) -> str:
    """Extract text from JSONL sample (supports 'text' or 'content' keys)."""
    if "text" in sample:
        return sample["text"]
    return sample["content"]


def _load_jsonl_source(path: str):
    """Load a local JSONL directory as a HuggingFace dataset."""
    return load_dataset("json", data_files=os.path.join(path, "*.jsonl"), split="train")


def _register_stem_datasets() -> None:
    """Register all STEM-pt midtraining sources in torchtitan's DATASETS registry."""
    for source_name in _STEM_SOURCES:
        dataset_key = f"stem_{source_name}"
        if dataset_key not in DATASETS:
            DATASETS[dataset_key] = DatasetConfig(
                path=os.path.join(_STEM_DATA_ROOT, source_name),
                loader=_load_jsonl_source,
                sample_processor=_process_jsonl_text,
            )


_register_stem_datasets()


# Token embeddings (``tok_embeddings.weight``) are trained without weight decay,
# following the OLMo2 recipe (and qwen3's param_groups). The multiplier (vs. a
# literal wd=0) keeps it at exactly zero even if the global weight_decay changes.
_NO_WD_TOKEN_EMBEDDINGS = ParamGroupConfig(
    pattern=r"tok_embeddings\.",
    weight_decay_multiplier=0.0,
)


def olmo2_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(
            lr=8e-4,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
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


def olmo2_1b_prefine() -> Trainer.Config:
    """OLMo2 1B config matching STEM-pt olmo2_1B_midfine training setup."""
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/pretraining",
    )
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/olmo2-1b-base-token4T",
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
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.91,
        ),
        training=TrainingConfig(
            local_batch_size=8,
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
            tensor_parallel_degree=1,
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            interval=10000,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def olmo2_1b_midfine() -> Trainer.Config:
    """OLMo2 1B config matching STEM-pt olmo2_1B_midfine training setup."""
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled",
    )
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/olmo2-1b-base-token4T",
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
            tensor_parallel_degree=1,
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            interval=10000,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
    )


def olmo2_7b_prefine() -> Trainer.Config:
    """OLMo2 7B with JSONL-chunk dataloader (deficit-driven source mixing).

    Replicates STEM-pt profile-om7b-base.sh settings:
    - Model: OLMo2-7B (4096 dim, 32 layers, 32 heads, vocab 100352)
    - Data: dolmino-mix_shuffled JSONL chunks, 22 sources with deficit tracking
    - Seq len: 4096, local batch: 2, prefetch: 1024
    - FSDP full_shard, dp_shard=8, compile=True
    - 10 profiling steps

    Set STEM_DATA_ROOT env var to override data path
    (default: /shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled).
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/pretraining",
    )
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMo2-7B-base-token5T",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=10,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("7B"),
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
            fsdp_reshard_after_forward="default",
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=1000,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def olmo2_7b_midfine() -> Trainer.Config:
    """OLMo2 7B with JSONL-chunk dataloader (deficit-driven source mixing).

    Replicates STEM-pt profile-om7b-base.sh settings:
    - Model: OLMo2-7B (4096 dim, 32 layers, 32 heads, vocab 100352)
    - Data: dolmino-mix_shuffled JSONL chunks, 22 sources with deficit tracking
    - Seq len: 4096, local batch: 2, prefetch: 1024
    - FSDP full_shard, dp_shard=8, compile=True
    - 10 profiling steps

    Set STEM_DATA_ROOT env var to override data path
    (default: /shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled).
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled",
    )
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMo2-7B-base-token5T",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=5,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("7B"),
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
            fsdp_reshard_after_forward="default",
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            enable=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )
