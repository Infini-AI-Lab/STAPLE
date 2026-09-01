# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, validate_config
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.olmo2.model import OLMo2Model
from torchtitan.tools.logging import logger


def parallelize_olmo2(
    model: OLMo2Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the OLMo2 model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallelism.full_dtensor:
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    else:
        if parallel_dims.cp_enabled:
            apply_cp_to_forward(
                [block.attention.inner_attention for block in model.layers.values()],
                parallel_dims.get_mesh("cp"),
            )
        if parallel_dims.tp_enabled:
            model.parallelize(parallel_dims)
    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config)

    # Always run apply_fsdp -- with shard_degree=1 it is a no-op for the
    # all-gather but still installs the MixedPrecisionPolicy.
    if parallelism.full_dtensor:
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        dp_mesh_dims = None

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    dp_mesh_dims: "DataParallelMeshDims | None" = None,
    enable_symm_mem: bool = False,
):
    """Apply data parallelism (via FSDP2) to the OLMo2 model."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.norm is not None and model.lm_head is not None:
        fully_shard(
            [model.norm, model.lm_head],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)

    disable_fsdp_gradient_division(model)
