# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.experiments.olmo2_moe.trainer import (
    _set_expert_param_group,
    ExpertHyperparams,
    OLMo2MoETrainer,
)


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1_EFD = nn.Parameter(torch.empty(2, 3, 4))
        self.w2_EDF = nn.Parameter(torch.empty(2, 4, 3))
        self.w3_EFD = nn.Parameter(torch.empty(2, 3, 4))


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _Experts()
        self.router = nn.Linear(4, 2, bias=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(8, 4)
        self.moe = _MoE()
        self.dense = nn.Linear(4, 4, bias=False)


def _names_by_lr(model: nn.Module, groups: list[dict]) -> dict[float, set[str]]:
    param_names = {param: name for name, param in model.named_parameters()}
    result: dict[float, set[str]] = {}
    for group in groups:
        result.setdefault(group["lr"], set()).update(
            param_names[param] for param in group["params"]
        )
    return result


class TestOLMo2MoEExpertPeakLR(unittest.TestCase):
    def test_unset_preserves_existing_param_groups(self):
        token_embeddings_group = ParamGroupConfig(
            pattern=r"tok_embeddings\.",
            weight_decay_multiplier=0.0,
        )
        config = OLMo2MoETrainer.Config(
            optimizer=OptimizersContainer.Config(
                lr=7.44e-5,
                param_groups=[token_embeddings_group],
            )
        )

        _set_expert_param_group(config)

        self.assertEqual(config.optimizer.param_groups, [token_embeddings_group])

    def test_distinct_lr_targets_only_expert_weights(self):
        model = _Model()
        config = OLMo2MoETrainer.Config(
            expert_hyperparams=ExpertHyperparams(lr=1.0e-4),
            optimizer=OptimizersContainer.Config(
                lr=7.44e-5,
                weight_decay=0.1,
                param_groups=[
                    ParamGroupConfig(
                        pattern=r"tok_embeddings\.",
                        weight_decay_multiplier=0.0,
                    )
                ],
            ),
        )
        _set_expert_param_group(config)
        optimizer_kwargs = OptimizersContainer._build_optimizer_kwargs(config.optimizer)
        groups = OptimizersContainer._build_param_groups(
            model,
            config.optimizer,
            optimizer_kwargs,
        )
        names_by_lr = _names_by_lr(model, groups)

        self.assertEqual(
            names_by_lr[1.0e-4],
            {
                "moe.experts.w1_EFD",
                "moe.experts.w2_EDF",
                "moe.experts.w3_EFD",
            },
        )
        self.assertIn("moe.router.weight", names_by_lr[7.44e-5])
        self.assertIn("dense.weight", names_by_lr[7.44e-5])

    def test_negative_lr_is_rejected(self):
        config = OLMo2MoETrainer.Config(expert_hyperparams=ExpertHyperparams(lr=-1.0))

        with self.assertRaisesRegex(ValueError, "non-negative"):
            _set_expert_param_group(config)


if __name__ == "__main__":
    unittest.main()
