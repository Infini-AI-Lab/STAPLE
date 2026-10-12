# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for the Engram-augmented OLMo2 model.

The OLMo2 backbone is parallelized exactly as upstream (TP/CP/AC/compile/FSDP via
``parallelize_olmo2``'s helpers). The only Engram-specific concern is the giant
N-gram embedding tables: they are **row/vocab-parallel sharded** across the FSDP
(``dp_shard``/``cp``) group rather than FSDP-sharded.

Each table parameter is converted to a ``Shard(0)`` DTensor over that group, and
then handed to ``fully_shard`` via ``ignored_params`` so FSDP does not try to
all-gather or re-shard it. The small Engram projections / norms / conv shard
normally with the rest of the block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_tensor, Shard

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
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.tools.logging import logger

from . import model as _model_module
from .model import EngramGradReducer, EngramOLMo2Model, register_engram_tables
from .multi_head_embedding import MultiHeadEmbedding

__all__ = ["parallelize_engram_olmo2"]


def _shard_engram_tables(
    model: EngramOLMo2Model,
    table_mesh: DeviceMesh,
) -> set[nn.Parameter]:
    """Convert each Engram table to a ``Shard(0)`` DTensor over ``table_mesh``.

    Returns the set of resulting parameters so they can be excluded from FSDP.

    The All-to-All gather assumes uniform rows/rank (it derives the owning rank
    as ``global_id // rows_per_rank`` with a single scalar ``rows_per_rank``), so
    the table must shard into equal contiguous slices. When ``total_rows`` is not
    a multiple of the mesh size, the table is padded with trailing dead rows up
    to the next multiple. Hash ids only ever address ``[0, total_rows)`` (the
    per-head offsets cumsum maxes at ``total_rows - 1``), so the padding rows are
    never gathered, never receive gradient, and stay at their init value -- they
    only cost ``< world`` extra rows of memory.
    """
    assert table_mesh.ndim == 1, (
        f"Engram table mesh must be 1D, got {table_mesh.ndim}D "
        f"({table_mesh.mesh_dim_names})."
    )
    world = table_mesh.size()
    sharded: set[nn.Parameter] = set()
    for module in model.modules():
        if not isinstance(module, MultiHeadEmbedding):
            continue
        weight = module.weight
        pad_rows = (-module.total_rows) % world
        if pad_rows:
            # Grow the (meta) weight param to a multiple of ``world`` so Shard(0)
            # produces uniform slices. Built on meta, so allocate a new meta
            # param of the padded shape; real values are filled by the trainer's
            # later to_empty + init_weights pass.
            padded = torch.empty(
                module.total_rows + pad_rows,
                module.per_head_dim,
                device=weight.device,
                dtype=weight.dtype,
            )
            weight = nn.Parameter(padded, requires_grad=weight.requires_grad)
            module.register_parameter("weight", weight)
            logger.info(
                f"Padded Engram table from {module.total_rows} to "
                f"{module.total_rows + pad_rows} rows for uniform sharding over "
                f"mesh size {world} (+{pad_rows} dead rows)."
            )
        # distribute_tensor handles meta-device params (model built on meta).
        dt = distribute_tensor(weight, table_mesh, [Shard(0)])
        new_param = nn.Parameter(dt, requires_grad=weight.requires_grad)
        module.register_parameter("weight", new_param)
        sharded.add(new_param)
    return sharded


def parallelize_engram_olmo2(
    model: EngramOLMo2Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply TP, AC, compile, FSDP to the backbone; vocab-shard Engram tables."""
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"Sequence length {training.seq_len} must be divisible by the product of "
        f"TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

    if parallel_dims.full_dtensor:
        raise NotImplementedError(
            "Engram-OLMo2 does not yet support full_dtensor mode; the Engram "
            "tables use manual All-to-All sharding outside the full-DTensor path."
        )
    if parallel_dims.pp_enabled:
        # Engram needs raw token ids at its layer; tokens are a stage-0 extra
        # input not forwarded across PP stages. Supporting PP requires threading
        # tokens to the Engram stage(s) explicitly. TODO.
        raise NotImplementedError(
            "Engram-OLMo2 does not yet support pipeline parallelism."
        )

    if parallel_dims.cp_enabled:
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )
    if parallel_dims.tp_enabled:
        # OLMo2's cross-head QK-norm is not TP-safe (see olmo2/model.py); the
        # backbone config validation already rejects TP>1. Keep async-TP hook
        # for completeness if that ever changes.
        model.parallelize(parallel_dims)
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

    # Resolve the FSDP / table-shard mesh.
    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    dp_mesh = parallel_dims.get_mesh(names)
    # Tables shard over the pure FSDP (dp_shard * cp) axis -- a 1D mesh -- so the
    # All-to-All has a single, well-defined process group. (dp_replicate would
    # add a redundant replica axis we don't want for the row shards.)
    table_mesh = parallel_dims.get_optional_mesh("fsdp")

    engram_table_params: set[nn.Parameter] = set()
    if table_mesh is not None and table_mesh.size() > 1:
        engram_table_params = _shard_engram_tables(model, table_mesh)
        logger.info(
            f"Vocab-parallel sharded {len(engram_table_params)} Engram table(s) "
            f"over fsdp mesh (size {table_mesh.size()})."
        )
    else:
        logger.info(
            "Engram tables left replicated (no FSDP sharding axis > 1); "
            "embedding uses the plain-gather path."
        )

    # Register the (now-sharded) tables for gradient-norm logging, keyed by the
    # layer they belong to (see ``loss.EngramLoss``).
    engram_tables = {
        layer_id: layer.engram.embedding
        for layer_id, layer in model.layers.items()
        if getattr(layer, "engram", None) is not None
    }
    register_engram_tables(engram_tables)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ignored_params=engram_table_params,
    )

    # Under HSDP (dp_replicate > 1) the vocab-parallel tables are sharded only
    # over the inner fsdp axis and excluded from FSDP, so nothing reduces their
    # grads across the dp_replicate replicas. Install a side-stream reducer that
    # the trainer fires after the final microbatch backward. Pure FSDP needs no
    # reducer (the single fsdp axis already owns complete row-shard grads).
    _model_module._ENGRAM_GRAD_REDUCER = None
    if parallel_dims.dp_replicate_enabled and engram_table_params and engram_tables:
        replicate_mesh = parallel_dims.get_mesh("dp_replicate")
        device = torch.device("cuda", torch.cuda.current_device())
        _model_module._ENGRAM_GRAD_REDUCER = EngramGradReducer(
            engram_tables, replicate_mesh.get_group(), device
        )
        logger.info(
            "Applied HSDP to the Engram-OLMo2 model; installed EngramGradReducer "
            f"over dp_replicate (size {replicate_mesh.size()})."
        )
    elif parallel_dims.dp_replicate_enabled:
        logger.info(
            "Applied HSDP to the Engram-OLMo2 model (Engram tables replicated; "
            "no cross-replica reducer needed)."
        )
    else:
        logger.info("Applied FSDP to the Engram-OLMo2 model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ignored_params: set[nn.Parameter] | None = None,
):
    """Apply FSDP2 to the model, excluding vocab-parallel Engram tables.

    The Engram tables are passed as ``ignored_params`` so FSDP leaves their
    existing ``Shard(0)`` placement intact (no all-gather / re-shard). All other
    parameters -- including the small Engram projections, norms, and conv --
    shard normally.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {
        "mesh": dp_mesh,
        "mp_policy": mp_policy,
        "ignored_params": ignored_params,
    }
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

    fully_shard(model, **fsdp_config)

    disable_fsdp_gradient_division(model)
