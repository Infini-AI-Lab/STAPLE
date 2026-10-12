# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.experiments.engram_olmo2.config_registry import (
    engram_olmo2_1b_stem_dag_l12_midfine,
    engram_olmo2_1b_stem_dag_l12_prefine,
)
from torchtitan.experiments.engram_olmo2.model_registry import engram_olmo2_configs


class TestEngramModelSize(unittest.TestCase):
    def test_prefine_allows_only_new_engram_parameters(self):
        config = engram_olmo2_1b_stem_dag_l12_prefine()

        self.assertEqual(
            config.checkpoint.initial_load_allowed_missing_model_key_prefixes,
            ["layers.2.engram.", "layers.7.engram."],
        )

    def test_midfine_uses_stem_dag_l12_matched_model(self):
        config = engram_olmo2_1b_stem_dag_l12_midfine()

        self.assertEqual(config.model_spec.flavor, "1B_stem_dag_l12_size")

    def test_two_layer_variant_matches_stem_dag_l12_size(self):
        config = engram_olmo2_configs["1B_stem_dag_l12_size"](attn_backend="sdpa")
        with torch.device("meta"):
            model = config.build()

        num_parameters = sum(parameter.numel() for parameter in model.parameters())

        # Full-table STEM-DAG L12 has 1,484,916,736 backbone parameters plus
        # twelve [100352, 8192] STEM tables. Prime-sized Engram tables make an
        # exact match impossible; this variant is only 1,536 parameters larger.
        self.assertEqual(num_parameters, 11_349_921_280)


if __name__ == "__main__":
    unittest.main()
