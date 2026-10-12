# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-head Engram embedding table with optional vocab-parallel sharding.

A single Engram layer concatenates ``K`` hash heads across ``max_ngram - 1``
N-gram orders into one big embedding table of
``total_rows = sum(head_table_sizes)`` rows, each of width ``per_head_dim``.
Per-head index spaces are made disjoint by adding fixed row offsets before the
gather (paper Eq. 2).

These tables dominate the parameter count (billions of rows), so they are
**row/vocab-parallel sharded** across a device-mesh axis rather than replicated
or FSDP-sharded: rank ``r`` owns the contiguous row-slice
``[r * rows_per_rank, (r + 1) * rows_per_rank)``. Lookups use an All-to-All:
each rank dispatches its requested row ids to the owning rank, the owner gathers
the rows from its local shard, and the rows are returned via a reverse
All-to-All. The backward dispatches output gradients back along the same path
and ``index_add_``-accumulates them into the owner's local shard gradient
(paper Section 2.5, Figure 2a). Because requested ids are disjoint *by owner*
but the same owned row may be requested by many ranks, the reverse All-to-All in
the backward already sums each row's contributions across the global batch, so
no extra all-reduce over the shard axis is needed.

When the table is not sharded (single rank / replicated), forward falls back to
a plain ``F.embedding`` gather.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.protocols.module import Module

__all__ = ["MultiHeadEmbedding"]


def _all_to_all_single(
    x: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """``all_to_all_single`` wrapper returning a freshly-allocated output."""
    out = x.new_empty((sum(output_split_sizes), *x.shape[1:]))
    dist.all_to_all_single(
        out,
        x.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return out


class _VocabParallelGather(torch.autograd.Function):
    """All-to-All gather of rows from a row-sharded embedding table.

    Forward inputs:
        weight_local: [rows_per_rank, D] this rank's contiguous row shard.
        global_ids:   [N] int64 global row ids requested by this rank.
        rows_per_rank, world_size, group: sharding metadata.

    Returns rows [N, D] in the original ``global_ids`` order. Gradient flows
    only to ``weight_local`` (indices are non-differentiable).
    """

    @staticmethod
    def forward(ctx, weight_local, global_ids, rows_per_rank, world_size, group):
        owner = torch.div(global_ids, rows_per_rank, rounding_mode="floor")
        local_row = global_ids - owner * rows_per_rank

        # Stable sort groups requests by owner rank while preserving a
        # deterministic order within each bucket (needed to invert later).
        order = torch.argsort(owner, stable=True)
        sorted_owner = owner[order]
        sorted_local_row = local_row[order]

        send_counts = torch.bincount(sorted_owner, minlength=world_size)
        send_counts_list = send_counts.tolist()

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        recv_counts_list = recv_counts.tolist()

        # Send each owner the local-row indices it must look up.
        recv_local_rows = _all_to_all_single(
            sorted_local_row, recv_counts_list, send_counts_list, group
        )
        # Local gather on the owner, then return rows along the reverse path.
        gathered = F.embedding(recv_local_rows, weight_local)
        sorted_rows = _all_to_all_single(
            gathered, send_counts_list, recv_counts_list, group
        )

        # Invert the sort so rows line up with the original global_ids order.
        out = torch.empty_like(sorted_rows)
        out[order] = sorted_rows

        ctx.save_for_backward(order, recv_local_rows, send_counts, recv_counts)
        ctx.rows_per_rank = rows_per_rank
        ctx.world_size = world_size
        ctx.group = group
        ctx.weight_shape = weight_local.shape
        ctx.weight_dtype = weight_local.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        order, recv_local_rows, send_counts, recv_counts = ctx.saved_tensors
        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()

        # Re-apply the forward permutation to grad rows, dispatch to owners.
        sorted_grad = grad_out[order]
        recv_grad = _all_to_all_single(
            sorted_grad, recv_counts_list, send_counts_list, ctx.group
        )
        # Accumulate each owned row's grads (summed across all requesters).
        grad_weight_local = torch.zeros(
            ctx.weight_shape, dtype=grad_out.dtype, device=grad_out.device
        )
        grad_weight_local.index_add_(0, recv_local_rows, recv_grad)
        if grad_weight_local.dtype != ctx.weight_dtype:
            grad_weight_local = grad_weight_local.to(ctx.weight_dtype)

        return grad_weight_local, None, None, None, None


class MultiHeadEmbedding(Module):
    """Concatenated multi-head N-gram embedding table.

    ``head_table_sizes`` is the flat list of per-head row counts, ordered to
    match :class:`EngramHasher`'s head layout. The output is
    ``[..., num_heads, per_head_dim]``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_table_sizes: list[int] = field(default_factory=list)
        per_head_dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.head_table_sizes = list(config.head_table_sizes)
        self.num_heads = len(self.head_table_sizes)
        self.per_head_dim = config.per_head_dim
        self.total_rows = sum(self.head_table_sizes)

        offsets = [0]
        for n in self.head_table_sizes[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False
        )

        self.weight = nn.Parameter(torch.empty(self.total_rows, self.per_head_dim))

    def _init_self_parameters(self) -> None:
        weight = self.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        nn.init.normal_(weight, mean=0.0, std=0.02)

    def _shard_info(self) -> tuple[DeviceMesh, int] | None:
        """Return ``(mesh, rows_per_rank)`` if the table is row-sharded."""
        weight = self.weight
        if not isinstance(weight, DTensor):
            return None
        placements = weight.placements
        if len(placements) != 1 or not isinstance(placements[0], Shard):
            raise ValueError(
                "MultiHeadEmbedding only supports a single Shard(0) placement "
                f"for the table, got {placements}."
            )
        if placements[0].dim != 0:
            raise ValueError(
                f"MultiHeadEmbedding table must be Shard(0), got Shard({placements[0].dim})."
            )
        mesh = weight.device_mesh
        rows_per_rank = weight.to_local().shape[0]
        return mesh, rows_per_rank

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """``hash_ids``: [..., num_heads] int64 -> [..., num_heads, per_head_dim]."""
        global_ids = hash_ids.to(torch.int64) + self.offsets

        shard_info = self._shard_info()
        if shard_info is None:
            # Replicated / single-rank: plain gather.
            weight = self.weight
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            return F.embedding(global_ids, weight)

        mesh, rows_per_rank = shard_info
        group = mesh.get_group()
        world_size = mesh.size()
        # grad_placements=Shard(0): the local grad returned by the custom
        # autograd Function is exactly this rank's row-shard gradient, already
        # summed over the global batch by the reverse All-to-All.
        weight_local = self.weight.to_local(grad_placements=[Shard(0)])

        flat_ids = global_ids.reshape(-1)
        rows = _VocabParallelGather.apply(
            weight_local, flat_ids, rows_per_rank, world_size, group
        )
        return rows.reshape(*global_ids.shape, self.per_head_dim)
