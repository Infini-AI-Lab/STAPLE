# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed.checkpoint as dcp

from scripts.checkpoint_conversion.upcycle_olmo2_to_moe import (
    expert_scale,
    load_model_only_dcp,
    upcycle_ffn_weights,
    upcycle_model_state,
    virtual_group_router_weight,
)


class TestOLMo2MoEUpcycle(unittest.TestCase):
    def test_target_parameter_counts(self):
        """E8G1T1-8L total/active counts; the total matches the built model."""
        dim = 2048
        vocab_size = 100352
        num_layers = 16
        num_moe_layers = 8
        dense_hidden_dim = 8192
        expert_hidden_dim = 8192
        num_experts = 8
        top_k = 1

        dense_non_ffn = (
            2 * vocab_size * dim + dim + num_layers * (4 * dim * dim + 4 * dim)
        )
        dense_ffn = (num_layers - num_moe_layers) * 3 * dim * dense_hidden_dim
        routers = num_moe_layers * dim * num_experts
        experts = num_moe_layers * num_experts * 3 * dim * expert_hidden_dim
        active_experts = num_moe_layers * top_k * 3 * dim * expert_hidden_dim

        self.assertEqual(
            dense_non_ffn + dense_ffn + routers + experts,
            4_303_620_096,
        )
        self.assertEqual(
            dense_non_ffn + dense_ffn + routers + active_experts,
            1_485_047_808,
        )

    def test_ffn_sharding_replication_and_orientation(self):
        dim, hidden_dim, granularity, expansion = 3, 8, 2, 3
        w1 = torch.arange(hidden_dim * dim, dtype=torch.float32).reshape(
            hidden_dim, dim
        )
        w2 = torch.arange(dim * hidden_dim, dtype=torch.float32).reshape(
            dim, hidden_dim
        )
        w3 = w1 + 100
        scale = expert_scale(expansion, granularity, granularity)

        ew1, ew2, ew3 = upcycle_ffn_weights(
            w1,
            w2,
            w3,
            expansion_rate=expansion,
            granularity=granularity,
            top_k=granularity,
        )

        self.assertEqual(ew1.shape, (6, 4, 3))
        self.assertEqual(ew2.shape, (6, 3, 4))
        torch.testing.assert_close(ew1[0], w1[:4] * scale)
        torch.testing.assert_close(ew1[2], ew1[0])
        torch.testing.assert_close(ew2[1], w2[:, 4:] * scale)
        torch.testing.assert_close(ew3[4], ew3[0])

    def test_virtual_group_router_repeats_prototypes(self):
        weight = virtual_group_router_weight(
            dim=16,
            expansion_rate=3,
            granularity=2,
            seed=7,
            std=1e-3,
            dtype=torch.float32,
        )
        self.assertEqual(weight.shape, (6, 16))
        torch.testing.assert_close(weight[0], weight[1])
        torch.testing.assert_close(weight[2], weight[3])
        self.assertFalse(torch.equal(weight[0], weight[2]))

        logits = torch.randn(5, 16) @ weight.T
        selected = logits.softmax(-1).topk(2, dim=-1).indices
        for token_experts in selected:
            groups = torch.div(token_experts, 2, rounding_mode="floor")
            shards = token_experts.remainder(2)
            self.assertEqual(groups.unique().numel(), 1)
            self.assertEqual(set(shards.tolist()), {0, 1})

    def test_model_state_conversion_is_model_only_and_preserves_dense_layers(self):
        dim, hidden_dim = 4, 8
        state = {
            "tok_embeddings.weight": torch.randn(16, dim),
            "layers.0.feed_forward.w1.weight": torch.randn(hidden_dim, dim),
            "layers.0.feed_forward.w2.weight": torch.randn(dim, hidden_dim),
            "layers.0.feed_forward.w3.weight": torch.randn(hidden_dim, dim),
            "layers.1.feed_forward.w1.weight": torch.randn(hidden_dim, dim),
            "layers.1.feed_forward.w2.weight": torch.randn(dim, hidden_dim),
            "layers.1.feed_forward.w3.weight": torch.randn(hidden_dim, dim),
        }
        output = upcycle_model_state(
            state,
            moe_layers=(1,),
            expansion_rate=2,
            granularity=2,
            top_k=2,
            output_dtype=torch.float32,
        )

        torch.testing.assert_close(
            output["layers.0.feed_forward.w1.weight"],
            state["layers.0.feed_forward.w1.weight"],
        )
        self.assertNotIn("layers.1.feed_forward.w1.weight", output)
        self.assertEqual(output["layers.1.moe.experts.w1_EFD"].shape, (4, 4, 4))
        self.assertEqual(output["layers.1.moe.router.gate.weight"].shape, (4, 4))
        torch.testing.assert_close(
            output["tok_embeddings.weight"], state["tok_embeddings.weight"]
        )
        self.assertFalse(any(key.startswith("optimizer.") for key in output))

    def test_requested_e4_topologies(self):
        dim, hidden_dim = 4, 8
        state = {
            "layers.0.feed_forward.w1.weight": torch.randn(hidden_dim, dim),
            "layers.0.feed_forward.w2.weight": torch.randn(dim, hidden_dim),
            "layers.0.feed_forward.w3.weight": torch.randn(hidden_dim, dim),
        }

        for granularity, top_k, expected_shape in (
            (1, 1, (4, 8, 4)),
            (8, 8, (32, 1, 4)),
        ):
            with self.subTest(granularity=granularity):
                output = upcycle_model_state(
                    state,
                    moe_layers=(0,),
                    expansion_rate=4,
                    granularity=granularity,
                    top_k=top_k,
                    output_dtype=torch.float32,
                )
                self.assertEqual(
                    output["layers.0.moe.experts.w1_EFD"].shape,
                    expected_shape,
                )
                self.assertEqual(
                    output["layers.0.moe.router.gate.weight"].shape,
                    (4 * granularity, dim),
                )

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "top_k == granularity"):
            upcycle_ffn_weights(
                torch.empty(8, 4),
                torch.empty(4, 8),
                torch.empty(8, 4),
                expansion_rate=2,
                granularity=2,
                top_k=1,
            )

    def test_dcp_loader_ignores_optimizer_state(self):
        with TemporaryDirectory() as checkpoint_dir:
            dcp.save(
                {
                    "tok_embeddings.weight": torch.full((4, 8), 2.0),
                    "norm.weight": torch.arange(4, dtype=torch.float32),
                    "optimizer.state.weight.exp_avg": torch.ones(4),
                    "train_state.step": torch.tensor(10),
                },
                checkpoint_id=checkpoint_dir,
            )
            state = load_model_only_dcp(Path(checkpoint_dir))

        self.assertEqual(set(state), {"tok_embeddings.weight", "norm.weight"})
        torch.testing.assert_close(
            state["norm.weight"], torch.arange(4, dtype=torch.float32)
        )
        torch.testing.assert_close(
            state["tok_embeddings.weight"],
            torch.full((4, 8), 2.0),
        )


if __name__ == "__main__":
    unittest.main()
