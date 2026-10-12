# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-rank test: vocab-parallel Engram embedding matches a replicated ref.

Verifies both the All-to-All forward gather and the gradient dispatch in the
backward against a single-table reference computed on the full (unsharded)
weight. Self-contained launcher (no torch test-internal deps); run with::

    torchrun --nproc_per_node=2 \\
        torchtitan/experiments/engram_olmo2/tests/test_vocab_parallel_embedding.py

Exits non-zero on mismatch. Requires a NCCL-capable multi-GPU environment.
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

from torchtitan.experiments.engram_olmo2.multi_head_embedding import MultiHeadEmbedding

HEAD_SIZES = [8, 8, 8, 8]  # total_rows = 32, divisible by world sizes 2/4
PER_HEAD_DIM = 6


def _reference(weight_full, hash_ids, offsets, grad_out):
    weight_full = weight_full.detach().clone().requires_grad_(True)
    global_ids = hash_ids + offsets
    out = F.embedding(global_ids, weight_full)
    out.backward(grad_out)
    return out.detach(), weight_full.grad.detach()


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = f"cuda:{local_rank}"
    mesh = init_device_mesh("cuda", (world_size,))

    # Identical full weight on every rank via a fixed seed.
    torch.manual_seed(0)
    emb = MultiHeadEmbedding.Config(
        head_table_sizes=HEAD_SIZES, per_head_dim=PER_HEAD_DIM
    ).build()
    emb.init_states(buffer_device=torch.device(device))
    emb.to(device)
    # Reference computed on CPU; capture the full weight + offsets on CPU.
    weight_full = emb.weight.detach().cpu().clone()
    offsets = emb.offsets.detach().cpu().clone()

    # Each rank holds a DISTINCT batch (seeded by rank). The sharded table's
    # gradient for a given row sums contributions across ALL ranks, so the
    # reference must aggregate every rank's inputs -- which is what the
    # backward's reverse All-to-All does.
    num_heads = len(HEAD_SIZES)
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    hash_ids = torch.stack(
        [
            torch.randint(0, HEAD_SIZES[h], (2, 7), device=device, generator=gen)
            for h in range(num_heads)
        ],
        dim=-1,
    )
    grad_out = torch.randn(2, 7, num_heads, PER_HEAD_DIM, device=device, generator=gen)

    # Shard the table row-wise and run the All-to-All forward/backward.
    sharded = distribute_tensor(emb.weight.detach(), mesh, [Shard(0)])
    emb.weight = torch.nn.Parameter(sharded, requires_grad=True)

    out = emb(hash_ids)
    out.backward(grad_out)

    # Gather every rank's inputs to build the full-table reference.
    all_hash = [torch.empty_like(hash_ids) for _ in range(world_size)]
    all_grad = [torch.empty_like(grad_out) for _ in range(world_size)]
    dist.all_gather(all_hash, hash_ids)
    dist.all_gather(all_grad, grad_out)
    cat_hash = torch.cat([h.cpu() for h in all_hash], dim=0)
    cat_grad = torch.cat([g.cpu() for g in all_grad], dim=0)

    ref_out_all, ref_grad = _reference(weight_full, cat_hash, offsets, cat_grad)

    # This rank's forward output corresponds to its own slice of the batch.
    ref_out = ref_out_all[rank * 2 : rank * 2 + 2]
    fwd_ok = torch.allclose(out.cpu(), ref_out, atol=1e-5)

    local_grad = emb.weight.grad.to_local().cpu()
    rows_per_rank = weight_full.shape[0] // world_size
    start = rank * rows_per_rank
    ref_slice = ref_grad[start : start + rows_per_rank]
    bwd_ok = torch.allclose(local_grad, ref_slice, atol=1e-5)

    status = torch.tensor([int(fwd_ok and bwd_ok)], device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            f"[vocab-parallel embedding] world={world_size} "
            f"forward={'OK' if fwd_ok else 'FAIL'} "
            f"backward={'OK' if bwd_ok else 'FAIL'} "
            f"-> {'PASS' if status.item() else 'FAIL'}"
        )
    dist.destroy_process_group()
    if status.item() != 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
