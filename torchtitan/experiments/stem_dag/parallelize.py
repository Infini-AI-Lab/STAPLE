# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for StemDagOLMo2Model.

Stem embeddings use intra-node dimension-parallel sharding on the EP mesh.
Each GPU holds (V, D/p) where p = EP degree. The custom forward handles
all-gather of input_ids + local lookup + all-to-all redistribution.
FSDP wraps stem embeddings on the edp_mesh (dp_replicate + efsdp) to mark
them as managed and handle cross-node gradient sync.
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
)

import torchtitan.experiments.stem_dag.model as _model_module

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
from torchtitan.tools.logging import logger
from .model import (
    register_stem_embeddings,
    StemDagLowRankOLMo2Model,
    StemDagOLMo2Model,
    StemGradReducer,
)


def _wire_stem_ep_mesh(
    model: StemDagOLMo2Model,
    ep_mesh: DeviceMesh,
) -> None:
    """Wire EP mesh onto stem embeddings for custom dim-parallel forward."""
    for emb in model._stem_embeddings.values():
        emb.wire_ep_mesh(ep_mesh)


def parallelize_stem_dag(
    model: StemDagOLMo2Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"Sequence length {training.seq_len} must be divisible by the product of TP degree "
        f"({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

    if parallelism.full_dtensor:
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    else:
        if parallel_dims.cp_enabled:
            apply_cp_to_forward(
                [block.attention.inner_attention for block in model.layers.values()],
                parallel_dims.get_mesh("cp"),
            )
        if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
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

    # --- Dimension-parallel sharding for stem embeddings ---
    ep_mesh: DeviceMesh | None = None
    edp_mesh: DeviceMesh | None = None
    if parallel_dims.ep_enabled:
        ep_mesh = parallel_dims.get_mesh("ep")
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        _wire_stem_ep_mesh(model, ep_mesh)

    # --- FSDP for all parameters (transformer layers + stem embeddings) ---
    if parallelism.full_dtensor:
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        dp_mesh_dims = None

    _apply_fsdp(
        model,
        dp_mesh=dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
        ep_enabled=parallel_dims.ep_enabled,
    )

    # --- Coalesced grad reducer for stem embeddings ---
    # Single coalesced all-reduce on a side stream, called explicitly by the
    # trainer after the final microbatch's backward completes.
    if edp_mesh is not None and edp_mesh.size() > 1:
        if edp_mesh.ndim > 1:
            flat_mesh = edp_mesh._flatten("edp")
        else:
            flat_mesh = edp_mesh
        group = flat_mesh.get_group()
        device = torch.device("cuda", torch.cuda.current_device())
        reducer = StemGradReducer(model._stem_embeddings, group, device)
        _model_module._STEM_GRAD_REDUCER = reducer

    register_stem_embeddings(model._stem_embeddings)

    num_stem = len(model._stem_embeddings)
    logger.info(f"Applied FSDP to model with {num_stem} stem embeddings.")

    return model


def parallelize_stem_dag_lowrank(
    model: StemDagLowRankOLMo2Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Parallelize StemDagLowRankOLMo2Model.

    The factored stem tables ((vocab_size, stem_dim)) and per-layer
    up-projections are small, so they are sharded with plain FSDP alongside
    the backbone -- no EP dimension-parallel sharding, custom embedding
    forward, or coalesced StemGradReducer (those exist for the full
    (vocab_size, ffn_hidden_dim) StemEmbedding). FSDP gradient sync handles
    the stem params like any other parameter.
    """
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"Sequence length {training.seq_len} must be divisible by the product of TP degree "
        f"({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

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

    if parallelism.full_dtensor:
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        dp_mesh_dims = None

    _apply_fsdp(
        model,
        dp_mesh=dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
        ep_enabled=False,
    )

    # No StemGradReducer: stem params are plain FSDP-managed parameters, so
    # FSDP handles their gradient reduction. register_stem_embeddings is still
    # called so StemDagLoss can report the stem grad norm.
    register_stem_embeddings(model._stem_embeddings)

    num_stem = len(model._stem_embeddings)
    logger.info(f"Applied FSDP to low-rank model with {num_stem} stem embeddings.")

    return model


def _apply_fsdp(
    model: StemDagOLMo2Model,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    dp_mesh_dims: "DataParallelMeshDims | None" = None,
    enable_symm_mem: bool = False,
    ep_enabled: bool = False,
):
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
    for transformer_block in model.layers.values():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Stem embeddings: distributed as DTensors via Module.parallelize() with
    # Shard(1) on the EP axis. No FSDP wrapping — exclude from root via
    # ignored_params so FSDP doesn't try to all-gather/reshard them.
    stem_params: set[nn.Parameter] | None = None
    if ep_enabled:
        stem_params = set()
        for emb in model._stem_embeddings.values():
            stem_params.update(emb.parameters())
    else:
        for emb in model._stem_embeddings.values():
            fully_shard(emb, **fsdp_config, reshard_after_forward=False)

    fully_shard(model, **fsdp_config, ignored_params=stem_params)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)

    disable_fsdp_gradient_division(model)
