# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Faithful end-to-end test of optimizer-state loading via the real trainer path.

This mirrors ``torchtitan.train``: it parses the SAME ``--module/--config`` +
CLI flags through ``ConfigManager``, calls the SAME ``config.build()`` to get a
real ``Trainer`` (real ``CheckpointManager`` + real ``OptimizersContainer``),
then runs the SAME ``checkpointer.load()`` the trainer would run first thing in
``train()``. The only difference: instead of training, it introspects the live
optimizer state to prove whether optimizer moments were actually loaded.

Discriminator (AdamW), given DCP primes un-stepped optimizers via a zero-grad
step at lr=0 (-> step=1, exp_avg=exp_avg_sq=0):
  * LOADED          -> step == saved big value (e.g. ~1.9e6), exp_avg_sq norm > 0
  * SILENTLY SKIPPED -> step == 1,                            exp_avg_sq norm == 0

Run under torchrun with the same flags as the real run script, e.g.:
  torchrun --nproc_per_node=8 -m torchtitan._verify_optim_load \
      --module olmo2 --config olmo2_1b_midfine \
      --checkpoint.enable \
      --checkpoint.initial_load_path <CKPT> \
      --checkpoint.no_initial_load_model_only \
      --checkpoint.exclude_from_loading dataloader,lr_scheduler,train_state
"""


import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger


def _rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _classify(name: str, container) -> None:
    """Print step + moment norms for representative params of each INNER optimizer.

    OptimizersContainer delegates state_dict/load_state_dict to its inner
    per-model-part optimizers (container.optimizers), so the primed/loaded state
    lives there, NOT on the container's own Optimizer.state.
    """
    n_total = 0
    n_zero = 0
    n_loaded = 0
    for oi, optim in enumerate(container.optimizers):
        for gi, group in enumerate(optim.param_groups):
            params = group["params"]
            idxs = sorted(set([0, len(params) - 1])) if params else []
            for which, pi in zip(("first", "last"), idxs):
                p = params[pi]
                st = optim.state.get(p, {})
                n_total += 1
                if not st:
                    logger.info(f"[{name}] opt{oi}.group{gi} {which}: NO STATE (empty)")
                    continue
                step = st.get("step", None)
                step_v = (
                    float(_local(step).float().reshape(-1)[0])
                    if step is not None
                    else float("nan")
                )
                ea = st.get("exp_avg", None)
                eas = st.get("exp_avg_sq", None)
                ea_n = (
                    float(_local(ea).float().norm()) if ea is not None else float("nan")
                )
                eas_n = (
                    float(_local(eas).float().norm())
                    if eas is not None
                    else float("nan")
                )
                shape = tuple(p.shape)
                verdict = "LOADED" if eas_n > 0 else "ZERO(skip?)"
                if eas_n > 0:
                    n_loaded += 1
                else:
                    n_zero += 1
                logger.info(
                    f"[{name}] opt{oi}.group{gi} lr={group.get('lr'):.3e} {which} "
                    f"shape={shape} step={step_v:.6g} "
                    f"exp_avg_norm={ea_n:.6e} exp_avg_sq_norm={eas_n:.6e} -> {verdict}"
                )
    logger.info(
        f"[{name}] SAMPLED {n_total} params: {n_loaded} LOADED (nonzero moments), "
        f"{n_zero} ZERO/skipped"
    )

    # Full sweep: count, across ALL params, how many have nonzero exp_avg_sq.
    total = 0
    nonzero = 0
    step_vals = set()
    for optim in container.optimizers:
        for p, st in optim.state.items():
            total += 1
            eas = st.get("exp_avg_sq", None)
            if eas is not None and float(_local(eas).float().norm()) > 0:
                nonzero += 1
            step = st.get("step", None)
            if step is not None:
                step_vals.add(round(float(_local(step).float().reshape(-1)[0])))
    logger.info(
        f"[{name}] FULL SWEEP: {total} params have optimizer state; "
        f"{nonzero} have nonzero exp_avg_sq; distinct step values={sorted(step_vals)}"
    )


def main() -> None:
    init_logger()
    config = ConfigManager().parse_args()

    trainer = config.build()
    try:
        if _rank0():
            logger.info("=" * 78)
            logger.info("BEFORE load: live optimizer state (expect empty/unprimed)")
            logger.info("=" * 78)
        opt = trainer.optimizers
        # Before load the optimizer state is empty (no .step(), no priming yet).
        n_state = sum(len(o.state) for o in opt.optimizers)
        if _rank0():
            logger.info(f"inner-optimizer params with state BEFORE load: {n_state}")

        # This is exactly what Trainer.train() does first.
        loaded = trainer.checkpointer.load(step=config.checkpoint.load_step)

        if _rank0():
            logger.info("=" * 78)
            logger.info(f"checkpointer.load() returned: {loaded}")
            logger.info("AFTER load: live optimizer state")
            logger.info("  LOADED      => step ~ saved big value, exp_avg_sq_norm > 0")
            logger.info(
                "  ZERO(skip?) => step == 1, exp_avg_sq_norm == 0 (silent skip)"
            )
            logger.info("=" * 78)
            _classify("optim", opt)

            # Also report the trainer step (train_state) — excluded => stays 0.
            logger.info(f"trainer.step after load = {trainer.step}")
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
