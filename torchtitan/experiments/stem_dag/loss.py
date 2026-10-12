# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.components.loss import ChunkedCELoss

from . import model as _model_module


class StemDagLoss(ChunkedCELoss):
    """ChunkedCELoss with additional stem embedding gradient norm reporting."""

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedCELoss.Config):
        pass

    def get_aux_metrics(self) -> dict[str, float]:
        ref = _model_module._STEM_EMBEDDINGS_REF
        if ref is None:
            return {}
        # Collect grads from every stem parameter. This covers both the full
        # StemEmbedding (single ``weight``) and the factored LowRankStemEmbedding
        # (``table.weight`` + ``up_proj.weight``).
        grads = [p.grad for p in ref.parameters() if p.grad is not None]
        if not grads:
            return {}
        total_norm = torch.nn.utils.get_total_norm(grads, norm_type=2.0)
        if hasattr(total_norm, "full_tensor"):
            total_norm = total_norm.full_tensor()
        return {"grad_norm/stem_embeddings": float(total_norm.item())}
