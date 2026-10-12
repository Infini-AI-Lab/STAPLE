# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read actual optimizer-state tensor VALUES directly from a DCP checkpoint.

This is independent of any model build or distributed init: it asks DCP to
load a handful of optimizer state tensors (step / exp_avg / exp_avg_sq) by
exact FQN into correctly-shaped tensors, then reports their norms. A non-zero
exp_avg_sq norm proves the checkpoint genuinely contains trained optimizer
moments (not just zero placeholders or bare keys).

Run (single process, no torchrun):
    python -m torchtitan._probe_ckpt_content /path/to/checkpoint
"""

import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def main() -> None:
    ckpt = sys.argv[1]
    reader = FileSystemReader(ckpt)
    md = reader.read_metadata()
    sdm = md.state_dict_metadata

    all_keys = list(sdm.keys())
    opt_state_keys = [k for k in all_keys if k.startswith("optimizer.state.")]
    model_keys = [
        k
        for k in all_keys
        if not k.startswith("optimizer.") and not k.startswith("state.")
    ]

    print(f"checkpoint: {ckpt}")
    print(f"  total keys           : {len(all_keys)}")
    print(f"  model (non-optimizer): {len(model_keys)}")
    print(f"  optimizer.state.*    : {len(opt_state_keys)}")

    # Sample a spread of optimizer-state tensors: a few early, a few late.
    def pick(suffix: str) -> list[str]:
        ks = [k for k in opt_state_keys if k.endswith(suffix)]
        return ks[:3] + ks[-3:]

    sample = pick(".step") + pick(".exp_avg") + pick(".exp_avg_sq")
    # Also include a model weight to confirm model tensors are real too.
    sample += model_keys[:2]

    state_dict = {}
    for k in sample:
        m = sdm[k]
        size = tuple(m.size)
        dtype = m.properties.dtype
        state_dict[k] = torch.empty(size, dtype=dtype)

    # No dist init -> DCP loads in no_dist (single-process) mode.
    dcp.load(state_dict, checkpoint_id=ckpt)

    print("\n--- loaded tensor values (norm > 0 means real trained data) ---")
    for k in sample:
        t = state_dict[k].float()
        n = t.numel()
        norm = t.norm().item() if n else 0.0
        first = t.reshape(-1)[0].item() if n else float("nan")
        print(
            f"  {k}\n"
            f"      shape={tuple(state_dict[k].shape)} dtype={state_dict[k].dtype} "
            f"numel={n} norm={norm:.6e} first={first:.6e}"
        )


if __name__ == "__main__":
    main()
