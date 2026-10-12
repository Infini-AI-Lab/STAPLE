# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer groups for static and projected per-expert STEM parameters."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.olmo2_moe.trainer import OLMo2MoETrainer
from torchtitan.experiments.stem_dag.model import StemGradReducer
from torchtitan.observability import structured_logger as sl


@dataclass(kw_only=True, slots=True)
class StemEmbeddingHyperparams:
    """Optimizer overrides for STEM tables and projections."""

    lr: float | None = None
    weight_decay: float | None = None


def _set_stem_param_group(config: "OLMo2MoEStemDagTrainer.Config") -> None:
    stem = config.stem_hyperparams
    if stem.lr is None and stem.weight_decay is None:
        return

    global_lr = config.optimizer.lr
    global_weight_decay = config.optimizer.weight_decay
    if stem.lr is not None and stem.lr < 0:
        raise ValueError(f"stem_hyperparams.lr must be non-negative, got {stem.lr}")
    if stem.weight_decay is not None and stem.weight_decay < 0:
        raise ValueError(
            "stem_hyperparams.weight_decay must be non-negative, got "
            f"{stem.weight_decay}"
        )
    if stem.lr is not None and global_lr <= 0:
        raise ValueError(
            "optimizer.lr must be positive when stem_hyperparams.lr is set"
        )
    if (
        stem.weight_decay is not None
        and global_weight_decay <= 0
        and stem.weight_decay != global_weight_decay
    ):
        raise ValueError(
            "optimizer.weight_decay must be positive when "
            "stem_hyperparams.weight_decay differs from it"
        )

    weight_decay_multiplier = 1.0
    if stem.weight_decay is not None and stem.weight_decay != global_weight_decay:
        weight_decay_multiplier = stem.weight_decay / global_weight_decay

    config.optimizer.param_groups = [
        ParamGroupConfig(
            pattern=(
                r"(?:^|\.)_stem_embeddings\.|" r"(?:^|\.)(?:stem_table_EVR|stem_up_FR)$"
            ),
            lr_multiplier=stem.lr / global_lr if stem.lr is not None else 1.0,
            weight_decay_multiplier=weight_decay_multiplier,
        ),
        *config.optimizer.param_groups,
    ]


class OLMo2MoEStemDagTrainer(OLMo2MoETrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OLMo2MoETrainer.Config):
        stem_hyperparams: StemEmbeddingHyperparams = field(
            default_factory=StemEmbeddingHyperparams
        )

    def __init__(self, config: Config):
        _set_stem_param_group(config)
        super().__init__(config)

    def _get_static_stem_grad_reducer(self) -> StemGradReducer | None:
        reducers = [
            reducer
            for model in self.model_parts
            if (reducer := getattr(model, "_static_stem_grad_reducer", None))
            is not None
        ]
        if len(reducers) > 1:
            raise RuntimeError("expected at most one static STEM gradient reducer")
        return reducers[0] if reducers else None

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        reducer = self._get_static_stem_grad_reducer()
        if reducer is None:
            return super().train_step(data_iterator)

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
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(self.device)
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

        reducer.reduce()

        with sl.log_trace_span("optim"):
            reducer.sync()
            grad_norm = dist_utils.clip_grad_norm_(
                [
                    parameter
                    for model in self.model_parts
                    for parameter in model.parameters()
                ],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
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
                            self.ntokens_seen,
                            dtype=torch.int64,
                            device=self.device,
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
        if hasattr(self.loss_fn, "get_aux_metrics"):
            extra_metrics.update(self.loss_fn.get_aux_metrics())
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )
