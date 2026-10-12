# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import pytest
import torch

from torchtitan.experiments.stem_dag.config_registry import (
    stem_dag_1b_midfine,
    stem_dag_1b_stem_wsd_midfine,
)
from torchtitan.experiments.stem_dag.trainer import (
    _clear_non_target_gradients,
    _initialize_adam_state_for_params,
    _set_optimizer_lr,
    StemDagTrainer,
)


def test_initialize_adam_state_for_frozen_params():
    module = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    initial_weight = module.weight.detach().clone()

    _initialize_adam_state_for_params(
        iter([optimizer]),
        [module.weight],
        use_bf16_moments=False,
    )

    state = optimizer.state[module.weight]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert state["step"].item() == 0
    assert torch.count_nonzero(state["exp_avg"]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0
    assert module.bias not in optimizer.state

    module.weight.grad = None
    optimizer.step()

    torch.testing.assert_close(module.weight, initial_weight)
    assert state["step"].item() == 0
    assert torch.count_nonzero(state["exp_avg"]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0


def test_initialize_all_adam_state_supports_complete_checkpoint_loads():
    module = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    params = [param for group in optimizer.param_groups for param in group["params"]]

    _initialize_adam_state_for_params(
        iter([optimizer]),
        params,
        use_bf16_moments=False,
    )

    assert set(optimizer.state) == set(module.parameters())
    assert all(state["step"].item() == 0 for state in optimizer.state.values())


def test_initialize_adam_state_uses_bf16_moments():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)

    _initialize_adam_state_for_params(
        iter([optimizer]),
        [parameter],
        use_bf16_moments=True,
    )

    assert optimizer.state[parameter]["exp_avg"].dtype == torch.bfloat16
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.bfloat16


def test_zero_lr_updates_only_target_adam_state():
    module = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    initial_weight = module.weight.detach().clone()
    initial_bias = module.bias.detach().clone()

    _initialize_adam_state_for_params(
        iter([optimizer]),
        list(module.parameters()),
        use_bf16_moments=False,
    )
    module(torch.ones(2, 4)).sum().backward()
    _clear_non_target_gradients([module], [module.weight])
    _set_optimizer_lr(iter([optimizer]), 0.0)
    optimizer.step()

    torch.testing.assert_close(module.weight, initial_weight, rtol=0, atol=0)
    torch.testing.assert_close(module.bias, initial_bias, rtol=0, atol=0)
    assert optimizer.state[module.weight]["step"].item() == 1
    assert torch.count_nonzero(optimizer.state[module.weight]["exp_avg"]) > 0
    assert torch.count_nonzero(optimizer.state[module.weight]["exp_avg_sq"]) > 0
    assert optimizer.state[module.bias]["step"].item() == 0
    assert torch.count_nonzero(optimizer.state[module.bias]["exp_avg"]) == 0
    assert torch.count_nonzero(optimizer.state[module.bias]["exp_avg_sq"]) == 0


def test_stem_wsd_scheduler_does_not_change_backbone_schedule():
    config = stem_dag_1b_stem_wsd_midfine()
    assert config.stem_hyperparams.lr is not None
    backbone_param = torch.nn.Parameter(torch.ones(1))
    stem_param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [backbone_param], "lr": config.optimizer.lr},
            {"params": [stem_param], "lr": config.stem_hyperparams.lr},
        ]
    )
    trainer = cast(Any, object.__new__(StemDagTrainer))
    trainer.optimizers = [optimizer]
    trainer._stem_param_group_idx = 1

    schedulers = trainer._build_stem_lr_schedulers(
        config,
        config.stem_hyperparams,
    )
    scheduler = cast(torch.optim.lr_scheduler.LambdaLR, schedulers.schedulers[0])
    backbone_schedule, stem_schedule = scheduler.lr_lambdas

    assert backbone_schedule(0) == 1.0
    assert backbone_schedule(30000) == pytest.approx(0.4)
    assert stem_schedule(0) == 1.0
    assert stem_schedule(30000) == 1.0
    assert stem_schedule(30001) == pytest.approx(0.999956)
    assert stem_schedule(50000) == 0.12
    assert config.stem_hyperparams.lr * stem_schedule(50000) == pytest.approx(3e-5)


def test_stem_wsd_midfine_inherits_base_recipe():
    base = stem_dag_1b_midfine()
    stem_wsd = stem_dag_1b_stem_wsd_midfine()

    assert stem_wsd.optimizer == base.optimizer
    assert stem_wsd.lr_scheduler == base.lr_scheduler
    assert stem_wsd.training == base.training
    assert stem_wsd.dataloader == base.dataloader
    assert stem_wsd.parallelism == base.parallelism
    assert stem_wsd.model_spec.name == base.model_spec.name
    assert stem_wsd.model_spec.model.stem_layers == base.model_spec.model.stem_layers

    assert base.stem_hyperparams.lr == 3e-4
    assert base.stem_hyperparams.decay_ratio is None
    assert stem_wsd.stem_hyperparams.lr == 2.5e-4
    assert stem_wsd.stem_hyperparams.weight_decay == 1e-4
    assert stem_wsd.stem_hyperparams.warmup_steps == 0
    assert stem_wsd.stem_hyperparams.decay_ratio == 0.4
    assert stem_wsd.stem_hyperparams.decay_type == "linear"
    assert stem_wsd.stem_hyperparams.min_lr_factor == 0.12
