# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configurations for granular OLMo2 MoE upcycling."""

import os

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
from torchtitan.hf_datasets.jsonl_datasets import JSONLDataLoader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.olmoe.moe_aux_loss import OLMoELoss

from . import model_registry
from .trainer import ExpertHyperparams, OLMo2MoETrainer

# Tokenizer source for the 1B configs (upcycling keeps the OLMo2-1B tokenizer).
# Launch scripts point --hf-assets-path at the upcycled checkpoint instead.
_HF_ASSETS_PATH = "/shared/dev/rjsadhuk/checkpoints/olmo2-1b-base-token4T"


_NO_WD_TOKEN_EMBEDDINGS = ParamGroupConfig(
    pattern=r"tok_embeddings\.",
    weight_decay_multiplier=0.0,
)
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


def olmo2_moe_debugmodel() -> OLMo2MoETrainer.Config:
    return OLMo2MoETrainer.Config(
        loss=OLMoELoss.Config(
            moe_loss_weight=0.01,
            moe_zloss_weight=0.001,
            num_experts=4,
            top_k=2,
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
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


def _olmo2_moe_1b_training_config(
    *,
    midtraining: bool,
    model_flavor: str,
    num_experts: int,
    top_k: int,
    moe_comm_backend: str = "deduplicated",
) -> OLMo2MoETrainer.Config:
    if midtraining:
        data_root = os.environ.get(
            "STEM_DATA_ROOT",
            "/shared/dev/rjsadhuk/data/midtraining/dolmino-mix_shuffled",
        )
        sources = _STEM_SOURCES
        warmup_steps = 0
        decay_type = "linear"
        min_lr_factor = 0.0
    else:
        data_root = os.environ.get(
            "STEM_DATA_ROOT",
            "/shared/dev/rjsadhuk/data/pretraining",
        )
        sources = {"dclm-baseline_shuffled": 1.0}
        # The converted checkpoint has no Adam moments. Ramp the first optimizer
        # updates instead of applying the full continued-training LR immediately.
        warmup_steps = 500
        decay_type = "cosine"
        min_lr_factor = 0.91

    return OLMo2MoETrainer.Config(
        # Routed experts follow the global schedule at a 2e-4 peak in both stages
        # (MoE expert rows of the STAPLE paper's Table 3).
        expert_hyperparams=ExpertHyperparams(lr=2e-4),
        loss=OLMoELoss.Config(
            moe_loss_weight=0.01,
            moe_zloss_weight=0.001,
            num_experts=num_experts,
            top_k=top_k,
        ),
        hf_assets_path=_HF_ASSETS_PATH,
        metrics=MetricsProcessor.Config(log_freq=10, enable_tensorboard=False),
        model_spec=model_registry(
            model_flavor,
            attn_backend="sdpa",
            moe_comm_backend=moe_comm_backend,
        ),
        dataloader=JSONLDataLoader.Config(
            root_dir=data_root,
            sources=sources,
            seed=777,
            add_bos=True,
            add_eos=True,
            n_views=2,
            prefetch_size=1024,
        ),
        optimizer=OptimizersContainer.Config(
            lr=7.44e-5,
            weight_decay=0.1,
            param_groups=[_NO_WD_TOKEN_EMBEDDINGS],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps,
            decay_ratio=1.0,
            decay_type=decay_type,
            min_lr_factor=min_lr_factor,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            global_batch_size=512,
            seq_len=4096,
            steps=50000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            data_parallel_replicate_degree=1,
            expert_parallel_degree=4,
            fsdp_reshard_after_forward="default",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
        checkpoint=CheckpointManager.Config(interval=10000),
        # Selective AC currently fails while recomputing integer MoE routing state.
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
    )


def olmo2_moe_1b_e8g1t1_8l_midfine() -> OLMo2MoETrainer.Config:
    """Train the eight-alternating-layer E8G1T1 model on the STEM mixture."""
    return _olmo2_moe_1b_training_config(
        midtraining=True,
        model_flavor="1B-E8G1T1-8L",
        num_experts=8,
        top_k=1,
    )


def olmo2_moe_1b_e8g1t1_8l_warmup() -> OLMo2MoETrainer.Config:
    """Warm up an upcycled eight-alternating-layer E8G1T1 model."""
    return _olmo2_moe_1b_training_config(
        midtraining=False,
        model_flavor="1B-E8G1T1-8L",
        num_experts=8,
        top_k=1,
    )
