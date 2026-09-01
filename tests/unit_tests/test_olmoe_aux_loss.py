# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.models.olmoe.moe_aux_loss import (
    clear_aux_losses,
    get_load_balancing_loss,
    get_router_z_loss,
    init_aux_loss_buffers,
    record_load_balancing_data,
    record_router_z_loss,
)


def test_aux_losses_use_static_layer_indices() -> None:
    init_aux_loss_buffers(2, torch.device("cpu"))
    clear_aux_losses()

    logits_0 = torch.tensor([[0.0, 0.0]])
    logits_1 = torch.tensor([[1.0, -1.0]])
    record_router_z_loss(logits_1, 1)
    record_router_z_loss(logits_0, 0)

    probs_0 = logits_0.softmax(dim=-1)
    probs_1 = logits_1.softmax(dim=-1)
    record_load_balancing_data(torch.tensor([1.0, 0.0]), probs_1, 2, 1, 1)
    record_load_balancing_data(torch.tensor([0.0, 1.0]), probs_0, 2, 1, 0)

    expected_z_loss = torch.stack(
        [
            torch.logsumexp(logits_0, dim=-1).square().mean(),
            torch.logsumexp(logits_1, dim=-1).square().mean(),
        ]
    ).mean()
    expected_load_balancing_loss = torch.stack(
        [2 * probs_0[0, 1], 2 * probs_1[0, 0]]
    ).mean()

    torch.testing.assert_close(get_router_z_loss(), expected_z_loss)
    torch.testing.assert_close(get_load_balancing_loss(), expected_load_balancing_loss)


def test_aux_losses_propagate_router_gradients_across_steps() -> None:
    init_aux_loss_buffers(1, torch.device("cpu"))

    for _ in range(2):
        clear_aux_losses()
        logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
        probs = logits.softmax(dim=-1)
        record_router_z_loss(logits, 0)
        record_load_balancing_data(torch.tensor([1.0, 0.0]), probs, 2, 1, 0)

        loss = get_router_z_loss() + get_load_balancing_loss()
        loss.backward()

        assert logits.grad is not None
        assert torch.count_nonzero(logits.grad) > 0
