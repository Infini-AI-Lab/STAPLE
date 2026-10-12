# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.olmo2_moe.config_registry import (
    olmo2_moe_1b_e8g1t1_8l_midfine,
    olmo2_moe_1b_e8g1t1_8l_warmup,
    olmo2_moe_1b_e8g8t8_8l_midfine,
    olmo2_moe_1b_e8g8t8_8l_warmup,
)
from torchtitan.experiments.olmo2_moe_stem_dag.config_registry import (
    _routed_moe,
    model_registry,
    olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_midfine,
    olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_warmup,
    olmo2_moe_stem_dag_static_1b_e8g1t1_8l_midfine,
    olmo2_moe_stem_dag_static_1b_e8g1t1_8l_warmup,
    olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_midfine,
    olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_warmup,
)
from torchtitan.experiments.olmo2_moe_stem_dag.moe import (
    ProjectedStemGroupedExperts,
    ProjectedStemMoE,
    StaticStemGroupedExperts,
    StaticStemMoE,
)
from torchtitan.experiments.olmo2_moe_stem_dag.parallelize import (
    _register_static_stem_grad_reducer,
)
from torchtitan.experiments.olmo2_moe_stem_dag.trainer import (
    _set_stem_param_group,
    OLMo2MoEStemDagTrainer,
    StemEmbeddingHyperparams,
)
from torchtitan.experiments.stem_dag.model import StemDagFeedForward
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


class TestOLMo2MoEStemDagModel(unittest.TestCase):
    EXPECTED_PARAMETERS = {
        ("1B-E8G1T1-8L", "static"): 10_880_288_768,
        ("1B-E8G1T1-8L", "per_expert_projected"): 10_947_397_632,
        ("1B-E8G8T8-8L-STEM12", "static"): 14_169_540_608,
    }

    def test_parameter_counts(self):
        for (flavor, strategy), expected in self.EXPECTED_PARAMETERS.items():
            with self.subTest(flavor=flavor, strategy=strategy):
                config = model_registry(flavor, stem_strategy=strategy).model
                with torch.device("meta"):
                    model = config.build()
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    expected,
                )

    def test_projected_dimensions_follow_num_experts(self):
        expected = {
            "1B-E8G1T1-8L": (8, 8192, 1024),
        }
        for flavor, (num_experts, expert_dim, stem_dim) in expected.items():
            with self.subTest(flavor=flavor):
                config = model_registry(
                    flavor,
                    stem_strategy="per_expert_projected",
                ).model
                for layer_id in config.stem_layers:
                    experts = config.layers[layer_id].moe.experts
                    self.assertIsInstance(experts, ProjectedStemGroupedExperts.Config)
                    self.assertEqual(experts.num_experts, num_experts)
                    self.assertEqual(experts.hidden_dim, expert_dim)
                    self.assertEqual(experts.stem_dim, stem_dim)

    def test_projected_dimension_rounds_up_to_multiple_of_eight(self):
        moe = _routed_moe(
            strategy="per_expert_projected",
            with_stem=True,
            dim=32,
            dense_hidden_dim=30,
            expert_hidden_dim=30,
            vocab_size=64,
            num_experts=3,
            granularity=1,
            top_k=1,
            aux_loss_layer_idx=0,
            moe_comm_backend="standard",
        )
        self.assertIsInstance(moe, ProjectedStemMoE.Config)
        self.assertEqual(moe.experts.stem_dim, 16)

    def test_projected_rejects_granular_static_flavor(self):
        for flavor in ("1B-E8G8T8-8L-STEM12",):
            with self.subTest(flavor=flavor):
                with self.assertRaisesRegex(ValueError, "only supports static"):
                    model_registry(
                        flavor,
                        stem_strategy="per_expert_projected",
                    )

    def test_debug_model_is_granularity_one_top_one(self):
        config = model_registry("debugmodel").model
        for layer_id in config.stem_layers:
            moe = config.layers[layer_id].moe
            self.assertEqual(moe.router.top_k, 1)
            self.assertEqual(moe.experts.token_dispatcher.top_k, 1)
            self.assertEqual(moe.experts.hidden_dim, 512)

    def test_static_uses_one_full_width_table_per_stem_layer(self):
        config = model_registry(
            "1B-E8G1T1-8L",
            stem_strategy="static",
        ).model
        self.assertEqual(
            set(config.stem_embeddings),
            {"1", "3", "5", "7", "9", "11", "13", "15"},
        )
        for embedding in config.stem_embeddings.values():
            self.assertEqual(embedding.num_embeddings, 100352)
            self.assertEqual(embedding.embedding_dim, 8192)
        with torch.device("meta"):
            model = config.build()
        for layer_id in config.stem_layers:
            self.assertIsInstance(model.layers[str(layer_id)].moe, StaticStemMoE)

    def test_static_staple12_mixed_layer_topology(self):
        config = model_registry("1B-E8G8T8-8L-STEM12").model
        moe_layers = {
            layer_id
            for layer_id, layer in enumerate(config.layers)
            if layer.moe is not None
        }
        self.assertEqual(config.stem_layers, list(range(1, 13)))
        self.assertEqual(
            moe_layers,
            {1, 3, 5, 7, 9, 11, 13, 15},
        )
        self.assertEqual(
            set(config.stem_embeddings),
            {str(layer_id) for layer_id in range(1, 13)},
        )
        for embedding in config.stem_embeddings.values():
            self.assertEqual(embedding.embedding_dim, 8192)

        with torch.device("meta"):
            model = config.build()
        for layer_id in (1, 3, 5, 7, 9, 11):
            moe = model.layers[str(layer_id)].moe
            self.assertIsInstance(moe, StaticStemMoE)
            self.assertEqual(moe.experts.granularity, 8)
            self.assertEqual(moe.experts.w1_EFD.shape[1], 1024)
            self.assertEqual(moe.router.top_k, 8)
        for layer_id in (2, 4, 6, 8, 10, 12):
            self.assertIsInstance(
                model.layers[str(layer_id)].feed_forward,
                StemDagFeedForward,
            )
        for layer_id in (13, 15):
            self.assertNotIsInstance(model.layers[str(layer_id)].moe, StaticStemMoE)

    def test_projected_has_one_shared_projection_per_layer(self):
        config = model_registry(
            "1B-E8G1T1-8L",
            stem_strategy="per_expert_projected",
        ).model
        with torch.device("meta"):
            model = config.build()
        for layer_id in config.stem_layers:
            moe = model.layers[str(layer_id)].moe
            self.assertIsInstance(moe, ProjectedStemMoE)
            experts = moe.experts
            self.assertEqual(tuple(experts.stem_table_EVR.shape), (8, 100352, 1024))
            self.assertEqual(tuple(experts.stem_up_FR.shape), (8192, 1024))

    def test_projected_up_projection_is_zero_initialized(self):
        config = model_registry(
            "debugmodel",
            stem_strategy="per_expert_projected",
        ).model
        experts = config.layers[1].moe.experts
        assert experts.param_init is not None
        weight = torch.empty(experts.hidden_dim, experts.stem_dim)
        experts.param_init["stem_up_FR"](weight)
        torch.testing.assert_close(weight, torch.zeros_like(weight))

    def test_static_table_is_zero_initialized(self):
        config = model_registry("debugmodel", stem_strategy="static").model
        table = config.stem_embeddings["1"].build()
        table.reset_parameters()
        torch.testing.assert_close(table.weight, torch.zeros_like(table.weight))

    def test_checkpoint_allows_only_strategy_specific_stem_prefixes(self):
        static_warmup = olmo2_moe_stem_dag_static_1b_e8g1t1_8l_warmup()
        self.assertEqual(
            static_warmup.checkpoint.initial_load_allowed_missing_model_key_prefixes,
            ["_stem_embeddings."],
        )
        static_midfine = olmo2_moe_stem_dag_static_1b_e8g1t1_8l_midfine()
        self.assertEqual(
            static_midfine.checkpoint.initial_load_allowed_missing_model_key_prefixes,
            [],
        )

        expected_projected_prefixes = [
            "layers.1.moe.experts.stem_",
            "layers.3.moe.experts.stem_",
            "layers.5.moe.experts.stem_",
            "layers.7.moe.experts.stem_",
            "layers.9.moe.experts.stem_",
            "layers.11.moe.experts.stem_",
            "layers.13.moe.experts.stem_",
            "layers.15.moe.experts.stem_",
        ]
        projected_warmup = olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_warmup()
        self.assertEqual(
            projected_warmup.checkpoint.initial_load_allowed_missing_model_key_prefixes,
            expected_projected_prefixes,
        )
        projected_midfine = olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_midfine()
        self.assertEqual(
            projected_midfine.checkpoint.initial_load_allowed_missing_model_key_prefixes,
            [],
        )
        staple_warmup = olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_warmup()
        staple_midfine = olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_midfine()
        self.assertEqual(
            staple_warmup.checkpoint.initial_load_allowed_missing_model_key_prefixes,
            ["_stem_embeddings."],
        )
        self.assertEqual(
            staple_midfine.checkpoint.initial_load_allowed_missing_model_key_prefixes,
            [],
        )

    def test_training_configs_inherit_e8g1t1_8l_base(self):
        cases = (
            (
                olmo2_moe_1b_e8g1t1_8l_warmup(),
                olmo2_moe_stem_dag_static_1b_e8g1t1_8l_warmup(),
            ),
            (
                olmo2_moe_1b_e8g1t1_8l_warmup(),
                olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_warmup(),
            ),
            (
                olmo2_moe_1b_e8g1t1_8l_midfine(),
                olmo2_moe_stem_dag_static_1b_e8g1t1_8l_midfine(),
            ),
            (
                olmo2_moe_1b_e8g1t1_8l_midfine(),
                olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_midfine(),
            ),
        )
        for base, config in cases:
            with self.subTest(
                strategy=config.model_spec.model.stem_strategy,
                dataset=config.dataloader.root_dir,
            ):
                self.assertEqual(config.training, base.training)
                self.assertEqual(config.parallelism, base.parallelism)
                self.assertEqual(config.loss, base.loss)
                self.assertEqual(config.dataloader, base.dataloader)
                self.assertEqual(config.optimizer, base.optimizer)
                self.assertEqual(config.lr_scheduler, base.lr_scheduler)
                self.assertEqual(config.model_spec.flavor, "1B-E8G1T1-8L")
                self.assertEqual(
                    config.model_spec.model.stem_layers,
                    [1, 3, 5, 7, 9, 11, 13, 15],
                )

    def test_warmup_and_midfine_use_different_data_and_schedules(self):
        for warmup, midfine in (
            (
                olmo2_moe_stem_dag_static_1b_e8g1t1_8l_warmup(),
                olmo2_moe_stem_dag_static_1b_e8g1t1_8l_midfine(),
            ),
            (
                olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_warmup(),
                olmo2_moe_stem_dag_projected_1b_e8g1t1_8l_midfine(),
            ),
            (
                olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_warmup(),
                olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_midfine(),
            ),
        ):
            with self.subTest(strategy=warmup.model_spec.model.stem_strategy):
                self.assertEqual(warmup.lr_scheduler.warmup_steps, 500)
                self.assertEqual(warmup.lr_scheduler.decay_type, "cosine")
                self.assertEqual(midfine.lr_scheduler.warmup_steps, 0)
                self.assertEqual(midfine.lr_scheduler.decay_type, "linear")
                self.assertEqual(warmup.stem_hyperparams.lr, 4e-4)
                self.assertEqual(midfine.stem_hyperparams.lr, 3e-4)
                self.assertEqual(warmup.stem_hyperparams.weight_decay, 1e-4)
                self.assertEqual(midfine.stem_hyperparams.weight_decay, 1e-4)
                self.assertNotEqual(
                    warmup.dataloader.sources,
                    midfine.dataloader.sources,
                )

    def test_staple12_training_configs_inherit_e8g8t8_base(self):
        cases = (
            (
                olmo2_moe_1b_e8g8t8_8l_warmup(),
                olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_warmup(),
            ),
            (
                olmo2_moe_1b_e8g8t8_8l_midfine(),
                olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_midfine(),
            ),
        )
        for base, config in cases:
            with self.subTest(dataset=config.dataloader.root_dir):
                self.assertEqual(config.training, base.training)
                self.assertEqual(config.parallelism, base.parallelism)
                self.assertEqual(config.loss, base.loss)
                self.assertEqual(config.dataloader, base.dataloader)
                self.assertEqual(config.optimizer, base.optimizer)
                self.assertEqual(config.lr_scheduler, base.lr_scheduler)
                self.assertEqual(
                    config.model_spec.flavor,
                    "1B-E8G8T8-8L-STEM12",
                )
                self.assertEqual(
                    config.model_spec.model.stem_layers,
                    list(range(1, 13)),
                )

    def test_stem_layers_reject_opaque_dispatch_backends(self):
        for strategy in ("static", "per_expert_projected"):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(ValueError, "standard.*deduplicated"):
                    model_registry(
                        "debugmodel",
                        stem_strategy=strategy,
                        moe_comm_backend="deepep",
                    )


class TestStaticStemDispatch(unittest.TestCase):
    def test_local_dispatch_routes_full_width_stem(self):
        experts = StaticStemGroupedExperts(
            StaticStemGroupedExperts.Config(
                dim=3,
                hidden_dim=4,
                num_experts=2,
                granularity=1,
                token_dispatcher=LocalTokenDispatcher.Config(
                    num_experts=2,
                    top_k=1,
                    score_before_experts=False,
                ),
            )
        )
        x_TD = torch.arange(12, dtype=torch.float32).view(4, 3)
        stem_TF = torch.arange(16, dtype=torch.float32).view(4, 4)
        scores_TK = torch.ones(4, 1)
        expert_ids_TK = torch.tensor([[1], [0], [1], [0]])
        _, _, metadata = experts.token_dispatcher.dispatch(
            x_TD,
            scores_TK,
            expert_ids_TK,
        )
        routed_stem = experts._dispatch_stem_signal(
            stem_TF.view(4, 1, 4),
            expert_ids_TK,
            metadata,
        )
        expected = stem_TF[metadata.token_indices_experts_sorted_N]
        torch.testing.assert_close(routed_stem, expected)

    def test_granular_dispatch_selects_expert_modulo_shard(self):
        experts = StaticStemGroupedExperts(
            StaticStemGroupedExperts.Config(
                dim=3,
                hidden_dim=2,
                num_experts=4,
                granularity=2,
                token_dispatcher=LocalTokenDispatcher.Config(
                    num_experts=4,
                    top_k=2,
                    score_before_experts=False,
                ),
            )
        )
        x_TD = torch.zeros(3, 3)
        stem_TGF = torch.arange(12, dtype=torch.float32).view(3, 2, 2)
        scores_TK = torch.ones(3, 2)
        expert_ids_TK = torch.tensor([[3, 0], [2, 1], [1, 2]])
        _, _, metadata = experts.token_dispatcher.dispatch(
            x_TD,
            scores_TK,
            expert_ids_TK,
        )
        actual = experts._dispatch_stem_signal(
            stem_TGF,
            expert_ids_TK,
            metadata,
        )
        flat_expert_ids = expert_ids_TK.reshape(-1)
        assignment_order = torch.argsort(flat_expert_ids, stable=True)
        expected = stem_TGF[
            assignment_order // 2,
            flat_expert_ids[assignment_order] % 2,
        ]
        torch.testing.assert_close(actual, expected)

    def test_rejects_num_experts_not_divisible_by_granularity(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            StaticStemGroupedExperts(
                StaticStemGroupedExperts.Config(
                    dim=3,
                    hidden_dim=2,
                    num_experts=3,
                    granularity=2,
                    token_dispatcher=LocalTokenDispatcher.Config(
                        num_experts=3,
                        top_k=1,
                        score_before_experts=False,
                    ),
                )
            )

    def test_rejects_incompatible_stem_width(self):
        experts = StaticStemGroupedExperts(
            StaticStemGroupedExperts.Config(
                dim=3,
                hidden_dim=2,
                num_experts=4,
                granularity=2,
                token_dispatcher=LocalTokenDispatcher.Config(
                    num_experts=4,
                    top_k=1,
                    score_before_experts=False,
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "granularity times"):
            experts(
                torch.zeros(1, 1, 3),
                torch.ones(1, 1, 1),
                torch.zeros(1, 1, 1, dtype=torch.long),
                torch.zeros(1, 1, 3),
            )


class TestProjectedStemDispatch(unittest.TestCase):
    def _experts(self):
        return ProjectedStemGroupedExperts(
            ProjectedStemGroupedExperts.Config(
                dim=3,
                hidden_dim=4,
                vocab_size=8,
                stem_dim=2,
                num_experts=2,
                token_dispatcher=LocalTokenDispatcher.Config(
                    num_experts=2,
                    top_k=1,
                    score_before_experts=False,
                ),
            )
        )

    def test_lookup_uses_routed_token_and_local_expert(self):
        experts = self._experts()
        with torch.no_grad():
            values = torch.arange(2 * 8 * 2, dtype=torch.float32).view(2, 8, 2)
            experts.stem_table_EVR.copy_(values)
            experts.stem_up_FR.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [1.0, -1.0],
                    ]
                )
            )

        x_TD = torch.zeros(4, 3)
        token_ids_T = torch.tensor([3, 2, 1, 0])
        scores_TK = torch.ones(4, 1)
        expert_ids_TK = torch.tensor([[1], [0], [1], [0]])
        _, counts, metadata = experts.token_dispatcher.dispatch(
            x_TD,
            scores_TK,
            expert_ids_TK,
        )
        routed_ids = experts._dispatch_token_ids(token_ids_T, metadata)
        actual = experts._project_stem(routed_ids, counts)

        expected_codes = torch.stack(
            (
                values[0, 2],
                values[0, 0],
                values[1, 3],
                values[1, 1],
            )
        )
        expected = torch.nn.functional.linear(expected_codes, experts.stem_up_FR)
        torch.testing.assert_close(actual.float(), expected)

    def test_rejects_top_k_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "G=1, T=1"):
            ProjectedStemGroupedExperts(
                ProjectedStemGroupedExperts.Config(
                    dim=3,
                    hidden_dim=4,
                    vocab_size=8,
                    stem_dim=2,
                    num_experts=2,
                    token_dispatcher=LocalTokenDispatcher.Config(
                        num_experts=2,
                        top_k=2,
                        score_before_experts=False,
                    ),
                )
            )


class _OptimizerFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self._stem_embeddings = nn.ModuleDict({"1": nn.Embedding(8, 4)})
        self.stem_table_EVR = nn.Parameter(torch.empty(2, 8, 2))
        self.stem_up_FR = nn.Parameter(torch.empty(4, 2))
        self.dense = nn.Linear(4, 4, bias=False)


class TestStemOptimizerGroups(unittest.TestCase):
    def test_static_and_projected_parameters_share_stem_group(self):
        model = _OptimizerFixture()
        config = OLMo2MoEStemDagTrainer.Config(
            stem_hyperparams=StemEmbeddingHyperparams(
                lr=3e-4,
                weight_decay=1e-4,
            ),
            optimizer=OptimizersContainer.Config(
                lr=7.44e-5,
                weight_decay=0.1,
            ),
        )
        _set_stem_param_group(config)
        optimizer_kwargs = OptimizersContainer._build_optimizer_kwargs(config.optimizer)
        groups = OptimizersContainer._build_param_groups(
            model,
            config.optimizer,
            optimizer_kwargs,
        )
        names = {parameter: name for name, parameter in model.named_parameters()}
        names_by_lr = {
            group["lr"]: {names[parameter] for parameter in group["params"]}
            for group in groups
        }
        self.assertEqual(
            names_by_lr[3e-4],
            {
                "_stem_embeddings.1.weight",
                "stem_table_EVR",
                "stem_up_FR",
            },
        )
        self.assertIn("dense.weight", names_by_lr[7.44e-5])


class _RecordingReducer:
    def __init__(self, events):
        self.events = events

    def reduce(self):
        self.events.append("reduce")

    def sync(self):
        self.events.append("sync")


class TestStaticStemGradReducer(unittest.TestCase):
    @mock.patch("torchtitan.experiments.olmo2_moe_stem_dag.parallelize.StemGradReducer")
    @mock.patch(
        "torchtitan.experiments.olmo2_moe_stem_dag.parallelize.torch.cuda.current_device",
        return_value=3,
    )
    def test_registers_one_reducer_over_flattened_edp_mesh(
        self,
        _current_device,
        reducer_cls,
    ):
        model = mock.MagicMock()
        model._stem_embeddings = mock.sentinel.embeddings
        edp_mesh = mock.MagicMock()
        edp_mesh.size.return_value = 8
        edp_mesh.ndim = 2
        flat_mesh = edp_mesh._flatten.return_value
        reducer = reducer_cls.return_value

        _register_static_stem_grad_reducer(model, edp_mesh)

        edp_mesh._flatten.assert_called_once_with("stem_edp")
        reducer_cls.assert_called_once_with(
            mock.sentinel.embeddings,
            flat_mesh.get_group.return_value,
            torch.device("cuda", 3),
        )
        self.assertIs(model._static_stem_grad_reducer, reducer)

    def test_reduces_once_after_all_accumulated_backwards(self):
        events = []
        reducer = _RecordingReducer(events)
        model = nn.Linear(2, 2, bias=False)
        model._static_stem_grad_reducer = reducer

        trainer = object.__new__(OLMo2MoEStemDagTrainer)
        trainer.model_parts = [model]
        trainer.optimizers = mock.MagicMock()
        trainer.lr_schedulers = mock.MagicMock()
        trainer.lr_schedulers.schedulers[0].get_last_lr.return_value = [7.44e-5]
        trainer.parallel_dims = mock.MagicMock()
        trainer.parallel_dims.dp_enabled = False
        trainer.parallel_dims.dp_cp_enabled = False
        trainer.parallel_dims.ep_enabled = True
        trainer.parallel_dims.get_optional_mesh.return_value = None
        trainer.gradient_accumulation_steps = 2
        trainer.device = torch.device("cpu")
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(max_norm=1.0),
        )
        trainer.checkpointer = mock.MagicMock()
        trainer.metrics_processor = mock.MagicMock()
        trainer.metrics_processor.should_log.return_value = False
        trainer.loss_fn = object()
        trainer.step = 1
        trainer.forward_backward_step = mock.MagicMock(
            side_effect=(torch.tensor(1.0), torch.tensor(2.0)),
        )
        trainer._set_gradient_sync = mock.MagicMock()
        data_iterator = iter(
            (
                ({"input": torch.ones(1, 2)}, torch.tensor([[1, 2]])),
                ({"input": torch.ones(1, 2)}, torch.tensor([[3, 4]])),
            )
        )

        def record_clip(*args, **kwargs):
            events.append("clip")
            return torch.tensor(0.0)

        with mock.patch(
            "torchtitan.experiments.olmo2_moe_stem_dag.trainer."
            "dist_utils.clip_grad_norm_",
            side_effect=record_clip,
        ):
            trainer.train_step(data_iterator)

        self.assertEqual(events, ["reduce", "sync", "clip"])
        self.assertEqual(trainer.forward_backward_step.call_count, 2)
        self.assertEqual(
            trainer._set_gradient_sync.call_args_list,
            [mock.call(False), mock.call(True)],
        )
        trainer.optimizers.zero_grad.assert_called_once_with()
        trainer.optimizers.step.assert_called_once_with()
        trainer.lr_schedulers.step.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
