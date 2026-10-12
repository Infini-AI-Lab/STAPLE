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
    find_static_stem_layers,
    load_model_only_dcp,
    upcycle_ffn_weights,
    upcycle_model_state,
    virtual_group_router_weight,
)


class TestOLMo2MoEUpcycle(unittest.TestCase):
    def test_target_parameter_counts(self):
        dim = 2048
        vocab_size = 100352
        num_layers = 16
        num_moe_layers = 12
        dense_hidden_dim = 8192
        expert_hidden_dim = 1024
        num_experts = 136
        top_k = 8

        dense_non_ffn = (
            2 * vocab_size * dim + dim + num_layers * (4 * dim * dim + 4 * dim)
        )
        dense_ffn = (num_layers - num_moe_layers) * 3 * dim * dense_hidden_dim
        routers = num_moe_layers * dim * num_experts
        experts = num_moe_layers * num_experts * 3 * dim * expert_hidden_dim
        active_experts = num_moe_layers * top_k * 3 * dim * expert_hidden_dim

        self.assertEqual(
            dense_non_ffn + dense_ffn + routers + experts,
            11_151_935_488,
        )
        self.assertEqual(
            dense_non_ffn + dense_ffn + routers + active_experts,
            1_488_259_072,
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
            "_stem_embeddings.1.weight": torch.randn(16, hidden_dim),
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
            output["_stem_embeddings.1.weight"],
            state["_stem_embeddings.1.weight"],
        )
        self.assertFalse(any(key.startswith("optimizer.") for key in output))

    def test_static_stem_layers_are_detected_and_sorted(self):
        state = {
            "_stem_embeddings.12.weight": torch.empty(2, 4),
            "_stem_embeddings.1.weight": torch.empty(2, 4),
            "layers.0.feed_forward.w1.weight": torch.empty(4, 2),
        }
        self.assertEqual(find_static_stem_layers(state), [1, 12])

    def test_staple12_alternating_layer_conversion(self):
        dim, hidden_dim = 2, 8
        moe_layers = tuple(range(1, 16, 2))
        state = {
            f"_stem_embeddings.{layer}.weight": torch.randn(16, hidden_dim)
            for layer in range(1, 13)
        }
        for layer in range(16):
            state[f"layers.{layer}.feed_forward.w1.weight"] = torch.randn(
                hidden_dim, dim
            )
            state[f"layers.{layer}.feed_forward.w2.weight"] = torch.randn(
                dim, hidden_dim
            )
            state[f"layers.{layer}.feed_forward.w3.weight"] = torch.randn(
                hidden_dim, dim
            )

        output = upcycle_model_state(
            state,
            moe_layers=moe_layers,
            expansion_rate=2,
            granularity=2,
            top_k=2,
            output_dtype=torch.float32,
        )

        self.assertEqual(find_static_stem_layers(output), list(range(1, 13)))
        for layer in moe_layers:
            self.assertNotIn(f"layers.{layer}.feed_forward.w1.weight", output)
            self.assertIn(f"layers.{layer}.moe.experts.w1_EFD", output)
        for layer in set(range(16)) - set(moe_layers):
            self.assertIn(f"layers.{layer}.feed_forward.w1.weight", output)

    def test_staple12_last_eight_layer_conversion(self):
        dim, hidden_dim = 2, 8
        moe_layers = tuple(range(8, 16))
        state = {
            f"_stem_embeddings.{layer}.weight": torch.randn(16, hidden_dim)
            for layer in range(1, 13)
        }
        for layer in range(16):
            state[f"layers.{layer}.feed_forward.w1.weight"] = torch.randn(
                hidden_dim, dim
            )
            state[f"layers.{layer}.feed_forward.w2.weight"] = torch.randn(
                dim, hidden_dim
            )
            state[f"layers.{layer}.feed_forward.w3.weight"] = torch.randn(
                hidden_dim, dim
            )

        output = upcycle_model_state(
            state,
            moe_layers=moe_layers,
            expansion_rate=2,
            granularity=2,
            top_k=2,
            output_dtype=torch.float32,
        )

        self.assertEqual(find_static_stem_layers(output), list(range(1, 13)))
        for layer in range(8):
            self.assertIn(f"layers.{layer}.feed_forward.w1.weight", output)
        for layer in moe_layers:
            self.assertNotIn(f"layers.{layer}.feed_forward.w1.weight", output)
            self.assertIn(f"layers.{layer}.moe.experts.w1_EFD", output)

    def test_staple12_four_layer_conversions(self):
        dim, hidden_dim = 2, 8
        state = {
            f"_stem_embeddings.{layer}.weight": torch.randn(16, hidden_dim)
            for layer in range(1, 13)
        }
        for layer in range(16):
            state[f"layers.{layer}.feed_forward.w1.weight"] = torch.randn(
                hidden_dim, dim
            )
            state[f"layers.{layer}.feed_forward.w2.weight"] = torch.randn(
                dim, hidden_dim
            )
            state[f"layers.{layer}.feed_forward.w3.weight"] = torch.randn(
                hidden_dim, dim
            )

        for moe_layers in ((2, 6, 10, 14), (8, 10, 12, 14)):
            with self.subTest(moe_layers=moe_layers):
                output = upcycle_model_state(
                    state,
                    moe_layers=moe_layers,
                    expansion_rate=8,
                    granularity=8,
                    top_k=8,
                    output_dtype=torch.float32,
                )

                self.assertEqual(
                    find_static_stem_layers(output),
                    list(range(1, 13)),
                )
                for layer in range(1, 13):
                    key = f"_stem_embeddings.{layer}.weight"
                    torch.testing.assert_close(output[key], state[key])
                for layer in range(16):
                    dense_keys = [
                        f"layers.{layer}.feed_forward.w{projection}.weight"
                        for projection in (1, 2, 3)
                    ]
                    if layer in moe_layers:
                        for key in dense_keys:
                            self.assertNotIn(key, output)
                        self.assertIn(
                            f"layers.{layer}.moe.experts.w1_EFD",
                            output,
                        )
                    else:
                        for key in dense_keys:
                            torch.testing.assert_close(output[key], state[key])

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
                    "_stem_embeddings.1.weight": torch.full((4, 8), 2.0),
                    "norm.weight": torch.arange(4, dtype=torch.float32),
                    "optimizer.state.weight.exp_avg": torch.ones(4),
                    "train_state.step": torch.tensor(10),
                },
                checkpoint_id=checkpoint_dir,
            )
            state = load_model_only_dcp(Path(checkpoint_dir))

        self.assertEqual(set(state), {"_stem_embeddings.1.weight", "norm.weight"})
        torch.testing.assert_close(
            state["norm.weight"], torch.arange(4, dtype=torch.float32)
        )
        torch.testing.assert_close(
            state["_stem_embeddings.1.weight"],
            torch.full((4, 8), 2.0),
        )


if __name__ == "__main__":
    unittest.main()
