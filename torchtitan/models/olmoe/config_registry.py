# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.jsonl_datasets import JSONLDataLoader
from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry
from .moe_aux_loss import OLMoELoss


# ---------------------------------------------------------------------------
# STEM-pt midtraining dataset registration
# ---------------------------------------------------------------------------

_STEM_DATA_ROOT = os.environ.get("STEM_DATA_ROOT", "/dev/shm/dolmino-mix_shuffled")

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


def _register_stem_datasets() -> None:
    """Register all STEM-pt midtraining sources in torchtitan's DATASETS registry."""
    from datasets import load_dataset

    for source_name in _STEM_SOURCES:
        dataset_key = f"stem_{source_name}"
        if dataset_key not in DATASETS:
            DATASETS[dataset_key] = DatasetConfig(
                path=os.path.join(_STEM_DATA_ROOT, source_name),
                loader=lambda p: load_dataset(
                    "json", data_files=os.path.join(p, "*.jsonl"), split="train"
                ),
                sample_processor=lambda s: s.get("text", s.get("content", "")),
            )


_register_stem_datasets()


# ---------------------------------------------------------------------------
# Config entries
# ---------------------------------------------------------------------------


def olmoe_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMoE-1B-7B-0924/",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def olmoe_debugmodel_ep() -> Trainer.Config:
    config = olmoe_debugmodel()
    config.model_spec = model_registry("debugmodel")
    return config


def olmoe_debugmodel_flex_attn() -> Trainer.Config:
    config = olmoe_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    return config


def olmoe_1b_7b() -> Trainer.Config:
    """OLMoE-1B-7B config with traditional aux losses for training."""
    return Trainer.Config(
        loss=OLMoELoss.Config(
            moe_loss_weight=0.01,
            moe_zloss_weight=0.001,
            num_experts=64,
            top_k=8,
        ),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMoE-1B-7B-0924/",
        model_spec=model_registry("1B-7B", attn_backend="flex"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=4e-4, weight_decay=0.1),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def olmoe_1b_7b_infer() -> Trainer.Config:
    """OLMoE-1B-7B config with plain CE loss (for inference/generation)."""
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMoE-1B-7B-0924/",
        model_spec=model_registry("1B-7B", attn_backend="sdpa"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=4e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


# ---------------------------------------------------------------------------
# Pretraining / midtraining configs
# ---------------------------------------------------------------------------


def olmoe_1b_7b_prefine() -> Trainer.Config:
    """OLMoE-1B-7B pretraining config.

    Model architecture (from allenai/OLMoE-1B-7B-0924):
      - 2048 dim, 16 layers, 16 heads, head_dim 128, vocab 50304
      - 64 experts, top-8, expert_hidden_dim 1024
      - RMSNorm (eps=1e-5), SwiGLU, RoPE (theta=10000)
      - QK-norm, no bias, no GQA

    Training hyperparameters (placeholder LR/schedule — set before launch):
      - Cosine schedule, warmup ~2500 steps (≈10.5B tokens at 4M tokens/step)
      - Weight decay 0.1, grad clip 1.0
      - Seq len 4096, global batch 1024 seqs = ~4.19M tokens/step
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/pretraining",
    )
    return Trainer.Config(
        loss=OLMoELoss.Config(
            moe_loss_weight=0.01,
            moe_zloss_weight=0.001,
            num_experts=64,
            top_k=8,
        ),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMoE-1B-7B-0924/",
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("1B-7B", attn_backend="sdpa"),
        optimizer=OptimizersContainer.Config(
            lr=4e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2500,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=1024,
            seq_len=4096,
            steps=100000,
        ),
        dataloader=JSONLDataLoader.Config(
            root_dir=data_root,
            sources={
                "dclm-baseline_shuffled": 0.963
            },  # "starcoder_shuffled": 0.022, "pes2o_shuffled": 0.015
            seed=777,
            add_bos=True,
            add_eos=True,
            n_views=2,
            prefetch_size=1024,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            expert_parallel_degree=1,
            fsdp_reshard_after_forward="default",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
        checkpoint=CheckpointManager.Config(
            interval=1000,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def olmoe_1b_7b_midfine() -> Trainer.Config:
    """OLMoE-1B-7B midtraining config with STEM-pt data mix.

    Model architecture: same as prefine (loaded from HF checkpoint).

    Training hyperparameters (placeholder LR/schedule — set before launch):
      - Linear decay to 0, no warmup (continuing from pretrained)
      - Weight decay 0.1, grad clip 1.0
      - Seq len 4096, local batch 4
    """
    data_root = os.environ.get(
        "STEM_DATA_ROOT",
        "/shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled",
    )
    return Trainer.Config(
        loss=OLMoELoss.Config(
            moe_loss_weight=0.01,
            moe_zloss_weight=0.001,
            num_experts=64,
            top_k=8,
        ),
        hf_assets_path="/shared/dev/rjsadhuk/checkpoints/OLMoE-1B-7B-0924/",
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
        ),
        model_spec=model_registry("1B-7B", attn_backend="sdpa"),
        optimizer=OptimizersContainer.Config(
            lr=4.04e-5,
            weight_decay=0.1,
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
            expert_parallel_degree=8,
            fsdp_reshard_after_forward="default",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
        checkpoint=CheckpointManager.Config(
            interval=5000,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )
