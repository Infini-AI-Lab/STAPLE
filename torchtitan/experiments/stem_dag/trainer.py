# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""StemDagTrainer: inserts coalesced stem gradient all-reduce into the
training loop, fixing three performance issues:

1. Suppresses stem grad all-reduce during non-final gradient accumulation
   microbatches (the base trainer's set_requires_gradient_sync only
   suppresses FSDP, not the stem embedding hooks).
2. Coalesces all 16 stem layer gradients into a single NCCL all-reduce
   call (1× latency instead of 16×).
3. Runs the coalesced all-reduce on a side CUDA stream so it overlaps
   with the FSDP dp_replicate gradient sync on the main stream.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .model import get_stem_grad_reducer

_STEM_EMBEDDING_PATTERN = r".*_stem_embeddings\."


def _initialize_adam_state_for_params(
    optimizers: Iterator[torch.optim.Optimizer],
    params: list[torch.nn.Parameter],
    *,
    use_bf16_moments: bool,
) -> None:
    """Materialize zero Adam slots for the selected parameters."""
    param_ids = {id(param) for param in params}
    initialized_param_ids: set[int] = set()

    for optimizer in optimizers:
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise ValueError("STEM optimizer-state setup requires Adam or AdamW.")
        for group in optimizer.param_groups:
            for param in group["params"]:
                param_id = id(param)
                if param_id not in param_ids:
                    continue
                initialized_param_ids.add(param_id)
                state = optimizer.state[param]
                if state:
                    continue

                step_device = (
                    param.device
                    if group.get("capturable") or group.get("fused")
                    else torch.device("cpu")
                )
                state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
                moment_dtype = torch.bfloat16 if use_bf16_moments else param.dtype
                state["exp_avg"] = torch.zeros_like(
                    param,
                    dtype=moment_dtype,
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param,
                    dtype=moment_dtype,
                    memory_format=torch.preserve_format,
                )
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param,
                        dtype=moment_dtype,
                        memory_format=torch.preserve_format,
                    )

    assert (
        param_ids == initialized_param_ids
    ), "Selected parameters are missing from the optimizer."


def _clear_non_target_gradients(
    model_parts: list[torch.nn.Module],
    target_params: list[torch.nn.Parameter],
) -> None:
    """Keep gradients only for the selected parameters."""
    target_param_ids = {id(param) for param in target_params}
    for model in model_parts:
        for param in model.parameters():
            if id(param) not in target_param_ids:
                param.grad = None


def _set_optimizer_lr(
    optimizers: Iterator[torch.optim.Optimizer],
    lr: float,
) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = lr


@dataclass(kw_only=True, slots=True)
class StemEmbeddingHyperparams:
    """Per-group optimizer/scheduler overrides for STEM embeddings.

    Any field left as None falls back to the global config value.
    """

    lr: float | None = None
    weight_decay: float | None = None
    warmup_steps: int | None = None
    decay_ratio: float | None = None
    decay_type: Literal["linear", "sqrt", "cosine"] | None = None
    min_lr_factor: float | None = None


class StemDagTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        stem_hyperparams: StemEmbeddingHyperparams = field(
            default_factory=StemEmbeddingHyperparams
        )
        freeze_stem_up_proj: bool = False
        """Freeze w3 (up-projection) in the DAG-STEM stem layers, e.g. during a
        warmup phase. Implemented by discarding w3's gradient after backward
        rather than ``requires_grad=False``, so w3 stays in its optimizer param
        group. Its zero optimizer state is materialized before training so it is
        saved to every checkpoint and can be restored when a later stage
        unfreezes w3."""
        warmup_stem_up_proj_optimizer_state: bool = False
        """Update only the STEM-layer w3 Adam state while applying zero parameter
        updates. This is intended to migrate legacy warmup checkpoints whose
        frozen w3 parameters have no optimizer slots."""

    def __init__(self, config: Config):
        if config.freeze_stem_up_proj and config.warmup_stem_up_proj_optimizer_state:
            raise ValueError(
                "freeze_stem_up_proj and warmup_stem_up_proj_optimizer_state "
                "cannot both be enabled."
            )

        stem = config.stem_hyperparams
        global_lr = config.optimizer.lr
        global_wd = config.optimizer.weight_decay
        has_stem_scheduler_override = any(
            value is not None
            for value in (
                stem.warmup_steps,
                stem.decay_ratio,
                stem.decay_type,
                stem.min_lr_factor,
            )
        )

        # Inject ParamGroupConfig for stem embeddings into the optimizer config
        # so the base Trainer builds a distinct group for any stem optimizer or
        # scheduler override.
        if (
            stem.lr is not None
            or stem.weight_decay is not None
            or has_stem_scheduler_override
        ):
            lr_mult = (stem.lr / global_lr) if stem.lr is not None else 1.0
            wd_mult = (
                (stem.weight_decay / global_wd)
                if stem.weight_decay is not None and global_wd > 0
                else 1.0
            )
            stem_pg = ParamGroupConfig(
                pattern=_STEM_EMBEDDING_PATTERN,
                lr_multiplier=lr_mult,
                weight_decay_multiplier=wd_mult,
            )
            config.optimizer.param_groups = [stem_pg] + list(
                config.optimizer.param_groups
            )

        super().__init__(config)

        # Index of the optimizer param group holding the stem embedding params,
        # matched by parameter identity (the group is NOT necessarily at a fixed
        # position; see _build_stem_lr_schedulers). None if there is no distinct
        # stem group. Used both to rebuild per-group LR schedulers and to log the
        # stem group's LR alongside the global LR.
        self._stem_param_group_idx = self._find_stem_param_group_idx()

        # Rebuild LR schedulers with per-group lambdas if stem schedule differs.
        if has_stem_scheduler_override:
            self.lr_schedulers = self._build_stem_lr_schedulers(config, stem)
            # Re-point the checkpointer at the rebuilt scheduler. super().__init__
            # registered the original object; without this the checkpointer
            # save/loads the orphaned original and the LR schedule restarts from
            # step 0 on every resume. "lr_scheduler" is the CheckpointManager key
            # (torchtitan/components/checkpoint.py LR_SCHEDULER).
            self.checkpointer.states["lr_scheduler"] = self.lr_schedulers

        # Collect only stem-layer up-projection (w3) params. Non-stem layers also
        # have a w3, so use the model's stem-layer set instead of name matching.
        self._stem_up_proj_params: list[torch.nn.Parameter] = []
        stem_up_proj_fqns: list[str] = []
        if config.freeze_stem_up_proj or config.warmup_stem_up_proj_optimizer_state:
            for model in self.model_parts:
                stem_layer_set = getattr(model, "_stem_layer_set", set())
                for layer_idx, layer in model.layers.items():
                    if int(layer_idx) not in stem_layer_set:
                        continue
                    feed_forward = layer.feed_forward
                    if hasattr(feed_forward, "w3"):
                        for param_name, param in feed_forward.w3.named_parameters():
                            self._stem_up_proj_params.append(param)
                            stem_up_proj_fqns.append(
                                f"layers.{layer_idx}.feed_forward.w3.{param_name}"
                            )

            optimizers = list(self.optimizers)
            optimizer_params = [
                param
                for optimizer in optimizers
                for group in optimizer.param_groups
                for param in group["params"]
            ]
            _initialize_adam_state_for_params(
                iter(optimizers),
                optimizer_params,
                use_bf16_moments=(
                    config.optimizer.implementation == "fused_opt_states_bf16"
                ),
            )

            if self.checkpointer.enable:
                legacy_prefixes = tuple(
                    f"optimizer.state.{fqn}." for fqn in stem_up_proj_fqns
                )
                configured_prefixes = (
                    self.checkpointer.resume_allowed_missing_optimizer_state_key_prefixes
                )
                self.checkpointer.resume_allowed_missing_optimizer_state_key_prefixes = tuple(
                    dict.fromkeys(configured_prefixes + legacy_prefixes)
                )

            if config.freeze_stem_up_proj:
                logger.info(
                    "freeze_stem_up_proj=True: freezing %d stem up-projection "
                    "(w3) parameter tensor(s) by discarding their gradients each "
                    "step; their zero-initialized optimizer state is saved in "
                    "checkpoints.",
                    len(self._stem_up_proj_params),
                )
            else:
                logger.info(
                    "warmup_stem_up_proj_optimizer_state=True: updating Adam "
                    "state only for %d stem up-projection (w3) parameter tensor(s) "
                    "with optimizer learning rates forced to zero.",
                    len(self._stem_up_proj_params),
                )

    def _find_stem_param_group_idx(self) -> int | None:
        """Return the index of the optimizer param group holding the stem
        embedding params, matched by parameter identity.

        OptimizersContainer._build_param_groups emits the unmatched DEFAULT
        (backbone) group first, then matched groups in pattern order, so the stem
        group is NOT necessarily at a fixed index. Returns None if no group
        contains stem params (e.g. all stem optimizer and scheduler overrides
        are unset, so stem params share the default group)."""
        stem_param_ids = {
            id(p)
            for model in self.model_parts
            for p in getattr(
                model, "_stem_embeddings", torch.nn.ModuleDict()
            ).parameters()
        }
        if not stem_param_ids:
            return None
        # All optimizers share the same param-group layout, so inspect the first.
        first_optimizer = next(iter(self.optimizers))
        for idx, group in enumerate(first_optimizer.param_groups):
            if any(id(p) in stem_param_ids for p in group["params"]):
                # A distinct stem group only exists if it doesn't also hold the
                # backbone (i.e. stem.lr/weight_decay injected a dedicated group).
                if not all(id(p) in stem_param_ids for p in group["params"]):
                    continue
                return idx
        return None

    def _build_stem_lr_schedulers(
        self,
        config: Config,
        stem: StemEmbeddingHyperparams,
    ) -> LRSchedulersContainer:
        """Build LR schedulers with distinct per-group lambdas for stem embeddings."""
        total_steps = (
            config.lr_scheduler.total_steps
            if config.lr_scheduler.total_steps is not None
            else config.training.steps
        )

        def _make_lambda(warmup_steps, decay_ratio, decay_type, min_lr_factor):
            if decay_ratio is not None:
                decay_steps = round(total_steps * decay_ratio)
                if warmup_steps + decay_steps > total_steps:
                    decay_steps = total_steps - warmup_steps
            else:
                decay_steps = total_steps - warmup_steps
            stable_steps = total_steps + 1 - warmup_steps - decay_steps

            def lr_lambda(current_step):
                warmup_stable_steps = warmup_steps + stable_steps
                if current_step < warmup_steps:
                    return float((current_step + 1) / warmup_steps)
                elif current_step < warmup_stable_steps:
                    return 1.0
                else:
                    progress = float(
                        (current_step + 1 - warmup_stable_steps) / decay_steps
                    )
                    if decay_type == "linear":
                        adj = 1 - progress
                    elif decay_type == "sqrt":
                        adj = 1 - math.sqrt(progress)
                    elif decay_type == "cosine":
                        adj = 0.5 * (1.0 + math.cos(math.pi * progress))
                    else:
                        raise ValueError(f"Unknown decay_type: {decay_type}")
                    return min_lr_factor + (1 - min_lr_factor) * adj

            return lr_lambda

        # Global lambda (same as LRSchedulersContainer.Config.build would produce)
        global_lambda = _make_lambda(
            warmup_steps=config.lr_scheduler.warmup_steps,
            decay_ratio=config.lr_scheduler.decay_ratio,
            decay_type=config.lr_scheduler.decay_type,
            min_lr_factor=config.lr_scheduler.min_lr_factor,
        )

        # Stem lambda with overrides
        stem_lambda = _make_lambda(
            warmup_steps=(
                stem.warmup_steps
                if stem.warmup_steps is not None
                else config.lr_scheduler.warmup_steps
            ),
            decay_ratio=(
                stem.decay_ratio
                if stem.decay_ratio is not None
                else config.lr_scheduler.decay_ratio
            ),
            decay_type=(
                stem.decay_type
                if stem.decay_type is not None
                else config.lr_scheduler.decay_type
            ),
            min_lr_factor=(
                stem.min_lr_factor
                if stem.min_lr_factor is not None
                else config.lr_scheduler.min_lr_factor
            ),
        )

        # Assign stem_lambda to the optimizer param group holding the stem
        # embedding params (located by parameter identity in
        # _find_stem_param_group_idx), and global_lambda to every other group.
        # When there is no distinct stem group, stem params share the default
        # group and can't be distinguished here, so use the global schedule.
        stem_idx = self._stem_param_group_idx
        schedulers = []
        for optimizer in self.optimizers:
            if stem_idx is not None:
                lambdas = [
                    stem_lambda if i == stem_idx else global_lambda
                    for i in range(len(optimizer.param_groups))
                ]
            else:
                lambdas = [global_lambda] * len(optimizer.param_groups)
            schedulers.append(LambdaLR(optimizer, lambdas))

        container = LRSchedulersContainer.__new__(LRSchedulersContainer)
        container.schedulers = schedulers
        return container

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # With per-group lambdas, get_last_lr() returns one value per group.
        # Use the last entry (the base/default group) for logging.
        last_lrs = self.lr_schedulers.schedulers[0].get_last_lr()
        lr = last_lrs[-1]
        # Stem embeddings train with their own lr/warmup; log it separately so
        # the stem schedule is visible distinctly from the backbone lr.
        stem_lr = (
            last_lrs[self._stem_param_group_idx]
            if self._stem_param_group_idx is not None
            else None
        )

        parallel_dims = self.parallel_dims

        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                microbatches.append((input_dict, labels))
        sl.log_trace_scalar({"local_valid_tokens": int(local_valid_tokens)})

        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        reducer = get_stem_grad_reducer()

        accumulated_losses = []
        num_microbatches = len(microbatches)
        for mb_idx, (input_dict, labels) in enumerate(microbatches):
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            is_last_microbatch = mb_idx == num_microbatches - 1
            if not is_last_microbatch:
                self._set_gradient_sync(False)

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

            if not is_last_microbatch:
                self._set_gradient_sync(True)

        # Freeze the stem up-projection (w3): discard its accumulated gradient
        # so the optimizer skips it (Adam/AdamW no-ops on params with grad=None)
        # and it is also excluded from the grad-norm computation below. w3 stays
        # in the optimizer, so its optimizer state is preserved unchanged.
        if self.config.freeze_stem_up_proj:
            for param in self._stem_up_proj_params:
                param.grad = None

        # After final backward: launch single coalesced stem grad all-reduce
        # on a side stream.
        if reducer is not None:
            reducer.reduce()

        with sl.log_trace_span("optim"):
            # Sync the side-stream before grad norm computation.
            if reducer is not None:
                reducer.sync()

            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            if self.config.warmup_stem_up_proj_optimizer_state:
                _clear_non_target_gradients(
                    self.model_parts,
                    self._stem_up_proj_params,
                )
                _set_optimizer_lr(iter(self.optimizers), 0.0)
                lr = 0.0
                stem_lr = 0.0
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()
            if self.config.warmup_stem_up_proj_optimizer_state:
                # Keep the saved optimizer LR at zero as well as the LR used by
                # the update. The scheduler state still advances normally.
                _set_optimizer_lr(iter(self.optimizers), 0.0)

        loss = torch.sum(torch.stack(accumulated_losses))

        if not self.metrics_processor.should_log(self.step):
            if hasattr(self.loss_fn, "get_logging_loss"):
                self.loss_fn.get_logging_loss()
            return

        with sl.log_trace_span("collect_dist_metrics"):
            sl.log_trace_scalar({"global_valid_tokens": int(global_valid_tokens)})

            logging_loss = loss
            if hasattr(self.loss_fn, "get_logging_loss"):
                ce_only = self.loss_fn.get_logging_loss()
                if ce_only is not None:
                    logging_loss = ce_only

            if parallel_dims.dp_cp_enabled:
                logging_loss = logging_loss.detach()
                loss_mesh = parallel_dims.get_optional_mesh("loss")

                local_avg_loss = logging_loss * global_valid_tokens / local_valid_tokens
                global_avg_loss, global_max_loss, global_ntokens_seen = (
                    dist_utils.dist_sum(logging_loss, loss_mesh),
                    dist_utils.dist_max(local_avg_loss, loss_mesh),
                    dist_utils.dist_sum(
                        torch.tensor(
                            self.ntokens_seen, dtype=torch.int64, device=self.device
                        ),
                        loss_mesh,
                    ),
                )
            else:
                global_avg_loss = global_max_loss = float(logging_loss.detach().item())
                global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        if stem_lr is not None:
            extra_metrics["lr/stem_embeddings"] = stem_lr
        if hasattr(self.loss_fn, "get_aux_metrics"):
            extra_metrics.update(self.loss_fn.get_aux_metrics())
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )
