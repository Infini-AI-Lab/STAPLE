# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch


class TestGroupedExpertsCompile(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_grouped_mm_compiles_and_runs(self):
        from torchtitan.models.common.moe import GroupedExperts
        from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

        num_experts = 8
        dim = 128
        hidden_dim = 256
        experts = GroupedExperts(
            GroupedExperts.Config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                token_dispatcher=LocalTokenDispatcher.Config(
                    num_experts=num_experts,
                    top_k=1,
                ),
            )
        ).cuda()
        num_tokens_per_expert = torch.tensor(
            [10, 8, 12, 9, 11, 7, 10, 13], dtype=torch.int32, device="cuda"
        )
        total_tokens = num_tokens_per_expert.sum().item()
        x = torch.randn(total_tokens, dim, device="cuda", requires_grad=True)

        compiled_experts_forward = torch.compile(
            experts._experts_forward, backend="inductor"
        )
        output = compiled_experts_forward(x, num_tokens_per_expert)
        output.sum().backward()

        self.assertEqual(output.shape, x.shape)
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(param.grad is not None for param in experts.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_grouped_mm_empty_tokens_forward_backward(self):
        from torchtitan.models.common.moe import GroupedExperts
        from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

        num_experts = 8
        dim = 128
        experts_config = GroupedExperts.Config(
            dim=dim,
            hidden_dim=256,
            num_experts=num_experts,
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=1,
            ),
        )
        num_tokens_per_expert = torch.zeros(
            num_experts, dtype=torch.int32, device="cuda"
        )

        for compile_experts in (False, True):
            with self.subTest(compile_experts=compile_experts):
                experts = GroupedExperts(experts_config).cuda()
                x = torch.empty(0, dim, device="cuda", requires_grad=True)
                experts_forward = experts._experts_forward
                if compile_experts:
                    experts_forward = torch.compile(experts_forward, backend="inductor")

                output = experts_forward(x, num_tokens_per_expert)
                output.sum().backward()

                self.assertEqual(output.shape, x.shape)
                self.assertIsNotNone(x.grad)
                for param in experts.parameters():
                    assert param.grad is not None
                    torch.testing.assert_close(param.grad, torch.zeros_like(param.grad))


if __name__ == "__main__":
    unittest.main()
