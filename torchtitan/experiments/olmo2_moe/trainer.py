# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configuration for OLMo2 MoE-specific optimizer groups."""

from dataclasses import dataclass, field

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.trainer import Trainer


_EXPERT_WEIGHTS_PATTERN = r"(?:^|\.)moe\.experts\.(?:w1_EFD|w2_EDF|w3_EFD)$"


@dataclass(kw_only=True, slots=True)
class ExpertHyperparams:
    """Optimizer overrides for routed expert weights.

    Unset fields inherit the corresponding global optimizer configuration.
    """

    lr: float | None = None


def _set_expert_param_group(config: "OLMo2MoETrainer.Config") -> None:
    expert_peak_lr = config.expert_hyperparams.lr
    if expert_peak_lr is None:
        return
    if expert_peak_lr < 0:
        raise ValueError(
            f"expert_hyperparams.lr must be non-negative, got {expert_peak_lr}"
        )

    base_peak_lr = config.optimizer.lr
    if expert_peak_lr == base_peak_lr:
        return
    if base_peak_lr <= 0:
        raise ValueError(
            "optimizer.lr must be positive when expert_hyperparams.lr differs from it"
        )

    expert_group = ParamGroupConfig(
        pattern=_EXPERT_WEIGHTS_PATTERN,
        lr_multiplier=expert_peak_lr / base_peak_lr,
    )
    config.optimizer.param_groups = [
        expert_group,
        *config.optimizer.param_groups,
    ]


class OLMo2MoETrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        expert_hyperparams: ExpertHyperparams = field(default_factory=ExpertHyperparams)

    def __init__(self, config: Config):
        _set_expert_param_group(config)
        super().__init__(config)
