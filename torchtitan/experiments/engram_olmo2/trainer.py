# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EngramTrainer: gives the Engram embedding tables their own optimizer/LR
schedule (separate lr, weight decay, warmup, and min-lr floor) on top of the
backbone schedule, and logs the Engram tables' learning rate during training.

Mirrors ``stem_dag``'s ``StemDagTrainer`` schedule handling but without any of
its grad-reduction / freeze machinery: Engram tables are vocab-parallel sharded
(not FSDP) and dispatch their own gradients, so the base training loop is reused
verbatim. We only:

1. Inject a :class:`ParamGroupConfig` so the Engram tables form a distinct
   optimizer param group with their own lr/weight-decay.
2. Optionally rebuild the LR schedulers with a per-group lambda so the Engram
   tables can warm up (and floor out) on a different schedule than the backbone.
3. Log the Engram param group's current lr as ``lr/engram_embeddings``.
"""

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torch.distributed.tensor import DTensor
from torch.optim.lr_scheduler import LambdaLR

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer

from .model import get_engram_grad_reducer, get_engram_tables

# Matches the ``MultiHeadEmbedding`` table param FQN
# ``layers.{i}.engram.embedding.weight``.
_ENGRAM_EMBEDDING_PATTERN = r"\.engram\.embedding\.weight$"


@dataclass(kw_only=True, slots=True)
class EngramEmbeddingHyperparams:
    """Per-group optimizer/scheduler overrides for the Engram embedding tables.

    Any field left as None falls back to the corresponding global config value
    (``optimizer.lr`` / ``optimizer.weight_decay`` / ``lr_scheduler.warmup_steps``
    / ``lr_scheduler.min_lr_factor``). ``lr`` and ``weight_decay`` are absolute
    values (converted to multipliers over the global values internally), matching
    ``stem_dag``'s ``StemEmbeddingHyperparams``.
    """

    lr: float | None = None
    weight_decay: float | None = None
    warmup_steps: int | None = None
    min_lr_factor: float | None = None


class EngramTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        engram_hyperparams: EngramEmbeddingHyperparams = field(
            default_factory=EngramEmbeddingHyperparams
        )

    def __init__(self, config: Config):
        engram = config.engram_hyperparams
        global_lr = config.optimizer.lr
        global_wd = config.optimizer.weight_decay

        # Inject a ParamGroupConfig for the Engram tables so the base Trainer
        # builds a distinct param group with the right lr/weight-decay. We
        # prepend it so it takes precedence over any broader user patterns; the
        # pattern is specific enough (table weight FQN) that ordering is moot in
        # practice.
        if engram.lr is not None or engram.weight_decay is not None:
            lr_mult = (engram.lr / global_lr) if engram.lr is not None else 1.0
            wd_mult = (
                (engram.weight_decay / global_wd)
                if engram.weight_decay is not None and global_wd > 0
                else 1.0
            )
            engram_pg = ParamGroupConfig(
                pattern=_ENGRAM_EMBEDDING_PATTERN,
                lr_multiplier=lr_mult,
                weight_decay_multiplier=wd_mult,
            )
            config.optimizer.param_groups = [engram_pg] + list(
                config.optimizer.param_groups
            )

        super().__init__(config)

        # Identity set of the Engram table params (the big vocab-parallel
        # tables). Prefer the registry populated by parallelize.py (the actual
        # sharded params); fall back to the FQN pattern. Used by the mesh-aware
        # clip to separate table grads (1D fsdp mesh) from the rest.
        tables = get_engram_tables()
        if tables:
            self._engram_table_param_ids = {id(t.weight) for t in tables.values()}
        else:
            engram_re = re.compile(_ENGRAM_EMBEDDING_PATTERN)
            self._engram_table_param_ids = {
                id(p)
                for model in self.model_parts
                for name, p in model.named_parameters()
                if engram_re.search(name)
            }

        # Index of the optimizer param group holding the Engram table params,
        # matched by parameter identity (the group is NOT necessarily at a fixed
        # position). None if there is no distinct Engram group. Used both to
        # rebuild per-group LR schedulers and to log the Engram group's lr.
        self._engram_param_group_idx = self._find_engram_param_group_idx()

        # Rebuild LR schedulers with per-group lambdas if the Engram schedule
        # differs from the backbone schedule.
        if engram.warmup_steps is not None or engram.min_lr_factor is not None:
            self.lr_schedulers = self._build_engram_lr_schedulers(config, engram)
            # super().__init__ already registered the ORIGINAL lr_schedulers with
            # the checkpointer (checkpointer.states["lr_scheduler"]). We just
            # replaced self.lr_schedulers with a new object, so re-point the
            # checkpointer at it -- otherwise checkpoint save/load operate on the
            # orphaned original and the LR schedule (warmup + decay) restarts from
            # step 0 on every resume. "lr_scheduler" is the key set in
            # CheckpointManager (torchtitan/components/checkpoint.py LR_SCHEDULER).
            self.checkpointer.states["lr_scheduler"] = self.lr_schedulers

        # Most recent pre-step Engram lr, captured in train_step and injected
        # into the logged metrics. None until the first step / when there is no
        # distinct Engram group.
        self._engram_lr: float | None = None

        # Wrap the metrics processor's log() so the Engram lr rides along with
        # the standard metrics without duplicating the base train_step.
        if self._engram_param_group_idx is not None:
            self._wrap_metrics_log_with_engram_lr()

    def _find_engram_param_group_idx(self) -> int | None:
        """Return the index of the optimizer param group holding the Engram
        table params, matched by parameter identity.

        OptimizersContainer._build_param_groups emits the unmatched DEFAULT
        (backbone) group first, then matched groups in pattern order, so the
        Engram group is NOT necessarily at a fixed index. Returns None if no
        group contains Engram params (e.g. engram.lr / engram.weight_decay both
        unset, so the tables share the default group)."""
        engram_re = re.compile(_ENGRAM_EMBEDDING_PATTERN)
        engram_param_ids = {
            id(p)
            for model in self.model_parts
            for name, p in model.named_parameters()
            if engram_re.search(name)
        }
        if not engram_param_ids:
            return None
        # All optimizers share the same param-group layout, so inspect the first.
        first_optimizer = next(iter(self.optimizers))
        for idx, group in enumerate(first_optimizer.param_groups):
            group_ids = {id(p) for p in group["params"]}
            # A distinct Engram group is one whose params are exactly (a subset
            # of) the Engram tables -- not the backbone group that merely also
            # contains them when no dedicated group was injected.
            if group_ids and group_ids <= engram_param_ids:
                return idx
        return None

    def _build_engram_lr_schedulers(
        self,
        config: Config,
        engram: EngramEmbeddingHyperparams,
    ) -> LRSchedulersContainer:
        """Build LR schedulers with a distinct per-group lambda for the Engram
        tables (warmup / min-lr floor overrides), reusing the backbone schedule
        for every other group."""
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

        # Backbone lambda (identical to what LRSchedulersContainer.Config.build
        # would produce for every group).
        global_lambda = _make_lambda(
            warmup_steps=config.lr_scheduler.warmup_steps,
            decay_ratio=config.lr_scheduler.decay_ratio,
            decay_type=config.lr_scheduler.decay_type,
            min_lr_factor=config.lr_scheduler.min_lr_factor,
        )

        # Engram lambda with warmup / min-lr overrides (decay shape inherited).
        engram_lambda = _make_lambda(
            warmup_steps=(
                engram.warmup_steps
                if engram.warmup_steps is not None
                else config.lr_scheduler.warmup_steps
            ),
            decay_ratio=config.lr_scheduler.decay_ratio,
            decay_type=config.lr_scheduler.decay_type,
            min_lr_factor=(
                engram.min_lr_factor
                if engram.min_lr_factor is not None
                else config.lr_scheduler.min_lr_factor
            ),
        )

        # Assign engram_lambda to the Engram param group (located by parameter
        # identity), global_lambda to every other group. When there is no
        # distinct Engram group, fall back to a uniform schedule. (To get an
        # Engram-specific schedule you must also set engram lr or weight_decay.)
        engram_idx = self._engram_param_group_idx
        schedulers = []
        for optimizer in self.optimizers:
            if engram_idx is not None:
                lambdas = [
                    engram_lambda if i == engram_idx else global_lambda
                    for i in range(len(optimizer.param_groups))
                ]
            else:
                lambdas = [global_lambda] * len(optimizer.param_groups)
            schedulers.append(LambdaLR(optimizer, lambdas))

        container = LRSchedulersContainer.__new__(LRSchedulersContainer)
        container.schedulers = schedulers
        return container

    def _wrap_metrics_log_with_engram_lr(self) -> None:
        """Wrap ``metrics_processor.log`` to inject ``lr/engram_embeddings``.

        This avoids duplicating the base ``train_step`` just to add one metric;
        the Engram lr is captured pre-step in ``train_step`` (consistent with how
        the base captures ``lr``) and stitched in here at log time."""
        original_log = self.metrics_processor.log

        def log_with_engram_lr(*args, extra_metrics=None, **kwargs):
            if self._engram_lr is not None:
                extra_metrics = dict(extra_metrics or {})
                extra_metrics["lr/engram_embeddings"] = self._engram_lr
            return original_log(*args, extra_metrics=extra_metrics, **kwargs)

        self.metrics_processor.log = log_with_engram_lr

    def _clip_grad_norm_mesh_aware(self, max_norm: float) -> torch.Tensor:
        """Clip grads when the Engram tables live on a different mesh than the
        backbone (HSDP: tables on the 1D ``fsdp`` mesh, FSDP-sharded params on
        the 2D ``(dp_replicate, fsdp)`` mesh).

        ``get_total_norm`` would ``torch.stack`` per-parameter norm DTensors and
        fail on mixed meshes. We instead split the params into Engram-table vs
        the rest, take each group's total norm (reducing any DTensor to a full
        scalar), combine the two scalars, and clip both groups by the combined
        norm. This mirrors ``dist_utils._clip_grad_norm_with_ep``.

        Assumes the Engram-table grads have already been reduced across
        ``dp_replicate`` by the EngramGradReducer, so each table grad is the
        correct global gradient before its norm is taken.
        """
        norm_type = 2.0
        engram_table_ids = self._engram_table_param_ids
        engram_grads, other_grads = [], []
        engram_params, other_params = [], []
        for m in self.model_parts:
            for p in m.parameters():
                if p.grad is None:
                    continue
                if id(p) in engram_table_ids:
                    engram_params.append(p)
                    engram_grads.append(p.grad)
                else:
                    other_params.append(p)
                    other_grads.append(p.grad)

        def _group_norm(grads):
            norm = torch.nn.utils.get_total_norm(
                grads, norm_type, error_if_nonfinite=False, foreach=True
            )
            if isinstance(norm, DTensor):
                # full_tensor reduces _NormPartial over the group's mesh.
                norm = norm.full_tensor()
            return norm

        engram_norm = _group_norm(engram_grads)
        other_norm = _group_norm(other_grads)
        total_norm = (engram_norm**norm_type + other_norm**norm_type) ** (
            1.0 / norm_type
        )

        torch.nn.utils.clip_grads_with_norm_(
            engram_params, max_norm, total_norm, foreach=True
        )
        torch.nn.utils.clip_grads_with_norm_(
            other_params, max_norm, total_norm, foreach=True
        )
        return total_norm

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        # Fast path: no distinct Engram group AND no cross-replica reducer
        # (pure FSDP, single mesh) -> the base train_step is correct as-is. We
        # still capture the Engram lr for logging via the metrics wrapper.
        reducer = get_engram_grad_reducer()
        if self._engram_param_group_idx is not None:
            self._engram_lr = self.lr_schedulers.schedulers[0].get_last_lr()[
                self._engram_param_group_idx
            ]
        if reducer is None:
            return super().train_step(data_iterator)

        # HSDP path: own the optim sequence so we can (1) reduce the Engram table
        # grads across dp_replicate after the final backward and (2) clip across
        # the two distinct param meshes. Mirrors the base train_step otherwise.
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
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

        # Reduce Engram table grads across dp_replicate (side stream), then sync
        # before clip so every replica clips with the same (global) grad norm.
        reducer.reduce()

        with sl.log_trace_span("optim"):
            reducer.sync()
            grad_norm = self._clip_grad_norm_mesh_aware(self.config.training.max_norm)
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

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
        if self._engram_lr is not None:
            extra_metrics["lr/engram_embeddings"] = self._engram_lr
        if hasattr(self.loss_fn, "get_aux_metrics"):
            extra_metrics.update(self.loss_fn.get_aux_metrics())
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )
