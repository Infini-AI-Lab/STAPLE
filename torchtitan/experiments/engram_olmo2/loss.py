# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loss with Engram embedding-table gradient-norm reporting.

Mirrors ``stem_dag``'s ``StemDagLoss``: a thin ``ChunkedCELoss`` subclass whose
``get_aux_metrics`` is picked up by the base trainer (``Trainer.train_step``
calls ``loss_fn.get_aux_metrics()`` when present) and merged into the logged
``extra_metrics``. We report the gradient norm of the big N-gram tables, which
are excluded from the model-wide ``clip_grad_norm_`` reporting path's intuition
(they train at 5x LR with no weight decay), so a separate signal is useful to
watch whether the memory tables are actually learning.
"""

from dataclasses import dataclass

import torch

from torchtitan.components.loss import ChunkedCELoss

from . import model as _model_module


class EngramLoss(ChunkedCELoss):
    """ChunkedCELoss with additional Engram embedding-table grad-norm reporting."""

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedCELoss.Config):
        pass

    def get_aux_metrics(self) -> dict[str, float]:
        tables = _model_module.get_engram_tables()
        if not tables:
            return {}

        metrics: dict[str, float] = {}
        all_grads = []
        for layer_id, table in tables.items():
            grad = table.weight.grad
            if grad is None:
                continue
            all_grads.append(grad)
            # Per-table norm. The grad is a Shard(0) DTensor over the FSDP axis
            # (one row-shard per rank); ``get_total_norm`` returns a DTensor and
            # ``full_tensor`` reduces across shards to the true global norm.
            norm = torch.nn.utils.get_total_norm([grad], norm_type=2.0)
            if hasattr(norm, "full_tensor"):
                norm = norm.full_tensor()
            metrics[f"grad_norm/engram_table_layer_{layer_id}"] = float(norm.item())

        if not all_grads:
            return {}

        total_norm = torch.nn.utils.get_total_norm(all_grads, norm_type=2.0)
        if hasattr(total_norm, "full_tensor"):
            total_norm = total_norm.full_tensor()
        metrics["grad_norm/engram_tables"] = float(total_norm.item())
        return metrics
