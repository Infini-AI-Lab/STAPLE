# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from tests.integration_tests import OverrideDefinitions


def build_model_tests_list() -> list[OverrideDefinitions]:
    """
    Build a dictionary of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will broadly cover
    all the supported model parallelism patterns on all the supported models.

    Returns:
        A dictionary where each key is a model name and value is a list of OverrideDefinitions
    """
    model_tests = [
        OverrideDefinitions(
            [
                [
                    "--module olmo2_moe_stem_dag "
                    "--config olmo2_moe_stem_dag_static_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "OLMo2 MoE static STEM DAG FSDP+EP",
            "olmo2_moe_static_stem_dag_fsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module olmo2_moe_stem_dag "
                    "--config olmo2_moe_stem_dag_static_granular_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "OLMo2 MoE granular static STEM DAG FSDP+EP",
            "olmo2_moe_granular_static_stem_dag_fsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module olmo2_moe_stem_dag "
                    "--config olmo2_moe_stem_dag_projected_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "OLMo2 MoE projected STEM DAG FSDP+EP",
            "olmo2_moe_projected_stem_dag_fsdp+ep",
            ngpu=4,
        ),
        # Integration Test Cases for Llama 4
        # TODO: re-enable compile after fixing
        # https://github.com/pytorch/torchtitan/issues/2771
        OverrideDefinitions(
            [
                [
                    "--module llama4 --config llama4_debugmodel_ep",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    # "--compile.enable",
                ],
            ],
            "Llama 4 PP+FSDP+TP+EP+compile",
            "llama4_pp+fsdp+tp+ep+compile",
            ngpu=8,
        ),
    ]

    return model_tests
