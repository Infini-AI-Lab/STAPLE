# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelism for static and projected per-expert OLMo2 MoE STEM."""

from typing import Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard

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
from torchtitan.experiments.stem_dag.model import StemGradReducer
from torchtitan.tools.logging import logger

from .model import OLMo2MoEStemDagModel


def _get_fsdp_mesh_info(mesh: DeviceMesh):
    from torch.distributed.fsdp._fully_shard._fsdp_common import (
        FSDPMeshInfo,
        HSDPMeshInfo,
    )

    if mesh.ndim == 1:
        return FSDPMeshInfo(mesh=mesh, shard_mesh_dim=0)
    if mesh.ndim == 2:
        return HSDPMeshInfo(mesh=mesh, replicate_mesh_dim=0, shard_mesh_dim=1)
    raise ValueError(f"Expected 1D or 2D FSDP mesh, got {mesh.ndim}D mesh")


def _expert_params(experts: nn.Module) -> set[nn.Parameter]:
    """Return parameters following expert storage, excluding the shared STEM up-proj."""
    return {
        parameter
        for name, parameter in experts.named_parameters(recurse=False)
        if name != "stem_up_FR"
    }


def _register_static_stem_grad_reducer(
    model: OLMo2MoEStemDagModel,
    edp_mesh: DeviceMesh | None,
) -> None:
    model._static_stem_grad_reducer = None
    if edp_mesh is None or edp_mesh.size() <= 1:
        return
    flat_mesh = edp_mesh._flatten("stem_edp") if edp_mesh.ndim > 1 else edp_mesh
    device = torch.device("cuda", torch.cuda.current_device())
    model._static_stem_grad_reducer = StemGradReducer(
        model._stem_embeddings,
        flat_mesh.get_group(),
        device,
    )


def _apply_fsdp(
    model: OLMo2MoEStemDagModel,
    *,
    dp_mesh: DeviceMesh,
    edp_mesh: DeviceMesh | None,
    ep_degree: int,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool,
    reshard_after_forward_policy: str,
    enable_symm_mem: bool,
) -> None:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
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

    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            continue

        experts = transformer_block.moe.experts
        expert_params = _expert_params(experts)
        num_experts = experts.num_experts
        if ep_degree > 1:
            assert edp_mesh is not None
            efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
        else:
            efsdp_ep_size = dp_mesh.size()
        expert_shard_placement = Shard(1) if efsdp_ep_size > num_experts else Shard(0)

        if ep_degree == 1 and expert_shard_placement == Shard(0):
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        elif ep_degree == 1:

            def expert_placement(
                parameter: nn.Parameter,
                expert_parameters: set[nn.Parameter] = expert_params,
            ):
                return Shard(1) if parameter in expert_parameters else None

            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=expert_placement,
            )
        else:
            from torch.distributed.fsdp._fully_shard._fsdp_common import (
                ShardPlacementResult,
            )

            assert edp_mesh is not None
            edp_mesh_info = _get_fsdp_mesh_info(edp_mesh)
            dp_mesh_info = _get_fsdp_mesh_info(dp_mesh)

            def mixed_placement(
                parameter: nn.Parameter,
                expert_parameters: set[nn.Parameter] = expert_params,
                expert_placement: Shard = expert_shard_placement,
            ):
                if parameter in expert_parameters:
                    return ShardPlacementResult(
                        placement=expert_placement,
                        mesh_info=edp_mesh_info,
                    )
                return ShardPlacementResult(
                    placement=Shard(0),
                    mesh_info=dp_mesh_info,
                )

            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=mixed_placement,
            )

    static_stem_params: set[nn.Parameter] | None = None
    if model.stem_strategy == "static":
        if ep_degree > 1:
            static_stem_params = set(model._stem_embeddings.parameters())
        else:
            for embedding in model._stem_embeddings.values():
                fully_shard(embedding, **fsdp_config, reshard_after_forward=False)

    fully_shard(model, **fsdp_config, ignored_params=static_stem_params)
    if enable_symm_mem:
        enable_fsdp_symm_mem(model)
    disable_fsdp_gradient_division(model)

    if ep_degree == 1:
        return

    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]
    if model.tok_embeddings is not None and transformer_blocks:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])
    for transformer_block, next_transformer_block in zip(
        transformer_blocks,
        next_transformer_blocks,
    ):
        if next_transformer_block is not None:
            transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif model.norm is not None and model.lm_head is not None:
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.lm_head]
            )

    reversed_blocks = list(reversed(transformer_blocks))
    previous_blocks = reversed_blocks[1:] + [None]
    if model.norm is not None and model.lm_head is not None and reversed_blocks:
        model.lm_head.set_modules_to_backward_prefetch([reversed_blocks[0]])
    for transformer_block, previous_block in zip(reversed_blocks, previous_blocks):
        if previous_block is not None:
            transformer_block.set_modules_to_backward_prefetch([previous_block])
        elif model.tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])


def parallelize_olmo2_moe_stem_dag(
    model: OLMo2MoEStemDagModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply CP, EP, compilation, and mixed dense/expert FSDP layouts."""
    if training.seq_len % parallel_dims.seq_len_divisor != 0:
        raise ValueError(
            f"Sequence length {training.seq_len} must be divisible by "
            f"{parallel_dims.seq_len_divisor}"
        )
    if parallelism.full_dtensor:
        raise NotImplementedError("full_dtensor is not supported")
    if parallel_dims.tp_enabled:
        raise NotImplementedError("tensor parallelism is not supported")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("pipeline parallelism is not supported")

    if parallel_dims.cp_enabled:
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )
    if parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if model.stem_strategy == "static" and parallel_dims.ep_enabled:
        ep_mesh = parallel_dims.get_mesh("ep")
        for embedding in model._stem_embeddings.values():
            embedding.wire_ep_mesh(ep_mesh)

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

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    _apply_fsdp(
        model,
        dp_mesh=dp_mesh,
        edp_mesh=edp_mesh,
        ep_degree=parallel_dims.ep,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    if model.stem_strategy == "static":
        _register_static_stem_grad_reducer(model, edp_mesh)

    logger.info(
        "Applied OLMo2 MoE STEM parallelism: strategy=%s, stem_layers=%d",
        model.stem_strategy,
        len(model._stem_layer_set),
    )
    return model


__all__ = ["parallelize_olmo2_moe_stem_dag"]
