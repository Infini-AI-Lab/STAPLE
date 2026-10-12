# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the low-rank STEM DAG building blocks.

These cover the factored stem embedding (``LowRankStemEmbedding``) and its
wiring into the config registry, with no distributed runtime required.
"""

import math
import unittest

import torch

from torchtitan.experiments.stem_dag.model import LowRankStemEmbedding

VOCAB = 512
STEM_DIM = 16
HIDDEN_DIM = 128


def _build_lowrank_stem(*, zero_init_up_proj: bool = True) -> LowRankStemEmbedding:
    """Build a LowRankStemEmbedding mirroring the config-registry recipe.

    ``table`` gets a normal init; ``up_proj`` is bias-free and zero-init by
    default so the stem signal starts at 0 (base-equivalent).
    """
    from torchtitan.experiments.stem_dag.config_registry import (
        _make_lowrank_stem_embeddings,
    )

    config = _make_lowrank_stem_embeddings(
        vocab_size=VOCAB,
        stem_dim=STEM_DIM,
        hidden_dim=HIDDEN_DIM,
        stem_layers=[1],
    )["1"]
    if not zero_init_up_proj:
        config.up_proj.param_init = {
            "weight": lambda weight: torch.nn.init.normal_(weight, std=0.02)
        }
    module = config.build()
    module.init_states()
    return module


class TestLowRankStemEmbedding(unittest.TestCase):
    def test_forward_shape(self):
        module = _build_lowrank_stem()
        tokens = torch.randint(0, VOCAB, (2, 9), dtype=torch.int64)
        y = module(tokens)
        self.assertEqual(tuple(y.shape), (2, 9, HIDDEN_DIM))

    def test_zero_init_up_proj_yields_zero_signal(self):
        """With a zero-init up-projection the stem signal must be exactly 0 so
        the model starts equivalent to the base (``w3(x) + 0 = w3(x)``)."""
        module = _build_lowrank_stem(zero_init_up_proj=True)
        tokens = torch.randint(0, VOCAB, (4, 7), dtype=torch.int64)
        y = module(tokens)
        self.assertTrue(torch.all(y == 0.0))

    def test_table_init_has_rank_scaled_rms(self):
        torch.manual_seed(0)
        module = _build_lowrank_stem()
        expected_rms = 1.0 / math.sqrt(STEM_DIM)
        actual_rms = module.table.weight.square().mean().sqrt().item()
        self.assertAlmostEqual(actual_rms, expected_rms, delta=0.02)

    def test_table_trains_freely(self):
        """The table is normal-initialized (not zero), so it carries gradient
        even though the up-projection starts at zero."""
        module = _build_lowrank_stem(zero_init_up_proj=True)
        # A zero up_proj kills the gradient to the table, so to verify the
        # table is a free parameter we give up_proj a nonzero value first.
        with torch.no_grad():
            torch.nn.init.normal_(module.up_proj.weight, std=0.02)
        self.assertFalse(torch.all(module.table.weight == 0.0))

        tokens = torch.randint(0, VOCAB, (2, 5), dtype=torch.int64)
        module(tokens).sum().backward()
        self.assertIsNotNone(module.table.weight.grad)
        self.assertIsNotNone(module.up_proj.weight.grad)

    def test_no_up_proj_bias(self):
        module = _build_lowrank_stem()
        self.assertIsNone(module.up_proj.bias)

    def test_param_count_smaller_than_full_table(self):
        """The factored path must be smaller than a full (vocab, hidden) table
        whenever stem_dim < hidden_dim."""
        module = _build_lowrank_stem()
        lowrank = sum(p.numel() for p in module.parameters())
        full = VOCAB * HIDDEN_DIM
        self.assertLess(lowrank, full)


class TestConfigRegistry(unittest.TestCase):
    def test_7b_uses_eight_stem_layers(self):
        from torchtitan.experiments.stem_dag.config_registry import model_registry

        config = model_registry("7B_stem_dag").model
        self.assertEqual(config.stem_layers, list(range(1, 9)))
        self.assertEqual(set(config.stem_embeddings), {str(i) for i in range(1, 9)})

    def test_lowrank_model_config_builds(self):
        from torchtitan.experiments.stem_dag.config_registry import _1b_stem_dag_lowrank

        cfg = _1b_stem_dag_lowrank(stem_layers=[1, 2], stem_dim=64)
        # One LowRankStemEmbedding config per requested stem layer.
        self.assertEqual(set(cfg.stem_embeddings.keys()), {"1", "2"})
        emb = cfg.stem_embeddings["1"]
        self.assertIsInstance(emb, LowRankStemEmbedding.Config)
        self.assertEqual(emb.table.embedding_dim, 64)
        # up_proj maps stem_dim -> ffn hidden_dim and is bias-free.
        self.assertEqual(emb.up_proj.in_features, 64)
        self.assertFalse(emb.up_proj.bias)
        # out_features is the OLMo2-1B ffn hidden_dim (2048 -> 8192).
        self.assertEqual(emb.up_proj.out_features, 8192)

    def test_model_registry_rejects_stem_dim_for_full_flavor(self):
        from torchtitan.experiments.stem_dag.config_registry import model_registry

        with self.assertRaises(ValueError):
            model_registry("1B_stem_dag", stem_dim=128)

    def test_rank_specific_trainer_configs(self):
        from torchtitan.experiments.stem_dag import config_registry

        for stage in ("prefine", "midfine"):
            for stem_dim in (128, 512, 2048):
                with self.subTest(stage=stage, stem_dim=stem_dim):
                    config_fn = getattr(
                        config_registry,
                        f"stem_dag_lowrank_1b_r{stem_dim}_{stage}",
                    )
                    config = config_fn()
                    stem_embedding = config.model_spec.model.stem_embeddings["1"]
                    self.assertEqual(stem_embedding.table.embedding_dim, stem_dim)
                    self.assertEqual(stem_embedding.up_proj.in_features, stem_dim)

    def test_prefine_authorizes_only_missing_stem_parameters(self):
        from torchtitan.experiments.stem_dag.config_registry import (
            stem_dag_lowrank_1b_r128_prefine,
        )

        config = stem_dag_lowrank_1b_r128_prefine()
        self.assertEqual(
            config.checkpoint.initial_load_allowed_missing_model_key_prefixes,
            ["_stem_embeddings."],
        )


if __name__ == "__main__":
    unittest.main()
