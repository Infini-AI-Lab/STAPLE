# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.experiments.olmo2_moe.config_registry import (
    olmo2_moe_1b_e8g1t1_8l_midfine,
    olmo2_moe_1b_e8g1t1_8l_warmup,
)

# (config_fn, flavor, moe_layer_ids, num_experts, top_k, expert_hidden_dim, num_params)
_VARIANTS = (
    (
        olmo2_moe_1b_e8g1t1_8l_midfine,
        "1B-E8G1T1-8L",
        [1, 3, 5, 7, 9, 11, 13, 15],
        8,
        1,
        8192,
        4_303_620_096,
    ),
)


class TestOLMo2MoEVariantConfigs(unittest.TestCase):
    def test_architecture(self):
        for (
            config_fn,
            flavor,
            expected_moe_layer_ids,
            num_experts,
            top_k,
            expert_hidden_dim,
            num_parameters,
        ) in _VARIANTS:
            with self.subTest(flavor=flavor):
                config = config_fn()
                model_config = config.model_spec.model
                moe_layer_ids = [
                    layer_id
                    for layer_id, layer in enumerate(model_config.layers)
                    if layer.moe is not None
                ]

                self.assertEqual(config.model_spec.flavor, flavor)
                self.assertEqual(moe_layer_ids, expected_moe_layer_ids)
                self.assertEqual(config.loss.num_experts, num_experts)
                self.assertEqual(config.loss.top_k, top_k)
                for layer_id in moe_layer_ids:
                    moe = model_config.layers[layer_id].moe
                    assert moe is not None
                    self.assertEqual(moe.num_experts, num_experts)
                    self.assertEqual(moe.router.top_k, top_k)
                    self.assertEqual(moe.experts.hidden_dim, expert_hidden_dim)
                with torch.device("meta"):
                    model = model_config.build()
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    num_parameters,
                )

    def test_warmup_and_midfine_use_different_data_and_schedules(self):
        for warmup_fn, midfine_fn in (
            (olmo2_moe_1b_e8g1t1_8l_warmup, olmo2_moe_1b_e8g1t1_8l_midfine),
        ):
            warmup, midfine = warmup_fn(), midfine_fn()
            with self.subTest(flavor=warmup.model_spec.flavor):
                self.assertEqual(warmup.model_spec.flavor, midfine.model_spec.flavor)
                # Warmup ramps in from a converted checkpoint with no Adam moments;
                # midfine decays to zero on the STEM mixture.
                self.assertNotEqual(
                    warmup.dataloader.sources, midfine.dataloader.sources
                )
                self.assertGreater(warmup.lr_scheduler.warmup_steps, 0)
                self.assertEqual(midfine.lr_scheduler.warmup_steps, 0)
                self.assertEqual(midfine.lr_scheduler.decay_type, "linear")
                self.assertEqual(warmup.lr_scheduler.decay_type, "cosine")


if __name__ == "__main__":
    unittest.main()
