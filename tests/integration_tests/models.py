# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from tests.integration_tests import OverrideDefinitions


def build_model_tests_list() -> list[OverrideDefinitions]:
    """
    Build the integration tests: one debugmodel per supported module, run with
    the parallelism its real 1B configs use (HSDP for the dense backbone,
    FSDP + EP for STAPLE tables and MoE experts, FSDP for Engram), compile on
    where the real configs compile.

    Returns:
        A list of OverrideDefinitions, one per test.
    """
    model_tests = [
        OverrideDefinitions(
            [
                [
                    "--module olmo2 --config olmo2_debugmodel",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--compile.enable",
                ],
            ],
            "OLMo2 HSDP+compile",
            "olmo2_hsdp+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module stem_dag --config stem_dag_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 4",
                    "--compile.enable",
                ],
            ],
            "STAPLE warmup FSDP+EP+compile",
            "stem_dag_fsdp+ep+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module engram_olmo2 --config engram_olmo2_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "Engram FSDP",
            "engram_olmo2_fsdp",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--module olmo2_moe --config olmo2_moe_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                    "--compile.components model,loss",
                    "--compile.no-fullgraph",
                ],
            ],
            "OLMo2 MoE FSDP+EP+compile",
            "olmo2_moe_fsdp+ep+compile",
            ngpu=4,
        ),
    ]

    return model_tests
