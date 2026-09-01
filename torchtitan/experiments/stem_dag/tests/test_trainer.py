# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.experiments.stem_dag.trainer import (
    _clear_non_target_gradients,
    _initialize_adam_state_for_params,
    _set_optimizer_lr,
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
