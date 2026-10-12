# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Post-norm OLMo2 MoE with static or projected per-expert STEM."""

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import Replicate, Shard

from torchtitan.experiments.olmo2_moe.model import (
    OLMo2MoEModel,
    OLMo2MoETransformerBlock,
)
from torchtitan.experiments.stem_dag.model import StemDagFeedForward, StemEmbedding
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.olmoe.moe_aux_loss import clear_aux_losses
from torchtitan.protocols.module import ModuleDict
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.protocols.types import MeshAxisName

from .moe import ProjectedStemMoE, StaticStemMoE


StemStrategy = Literal["static", "per_expert_projected"]


class OLMo2MoEStemDagTransformerBlock(OLMo2MoETransformerBlock):
    """OLMo2 block whose routed FFN can consume one of the STEM inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(OLMo2MoETransformerBlock.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        stem_embedding: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.attention_norm(
            self.attention(x, freqs_cis, attention_masks, positions)
        )
        if self.moe_enabled and isinstance(self.moe, StaticStemMoE):
            if stem_embedding is None:
                raise RuntimeError("static STEM layer requires a STEM embedding")
            ffn_out = self.moe(h, stem_embedding)
        elif self.moe_enabled and isinstance(self.moe, ProjectedStemMoE):
            if token_ids is None:
                raise RuntimeError("projected per-expert STEM layer requires token IDs")
            ffn_out = self.moe(h, token_ids)
        elif not self.moe_enabled and stem_embedding is not None:
            if not isinstance(self.feed_forward, StemDagFeedForward):
                raise RuntimeError(
                    "dense static STEM layer requires a StemDagFeedForward"
                )
            ffn_out = self.feed_forward(h, stem_embedding)
        else:
            ffn_out = self.moe(h) if self.moe_enabled else self.feed_forward(h)
        return h + self.ffn_norm(ffn_out)


class OLMo2MoEStemDagModel(OLMo2MoEModel):
    """OLMo2 MoE augmented with static or projected per-expert STEM."""

    @dataclass(kw_only=True, slots=True)
    class Config(OLMo2MoEModel.Config):
        stem_strategy: StemStrategy = "static"
        stem_layers: list[int] = field(default_factory=list)
        stem_embeddings: dict[str, StemEmbedding.Config] = field(default_factory=dict)

        def update_from_config(self, *, config, **kwargs) -> None:
            OLMo2MoEModel.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "olmo2_moe_stem_dag does not support pipeline parallelism"
                )

            from .sharding import (
                set_projected_stem_sharding,
                set_static_stem_input_sharding,
            )

            if self.stem_strategy == "static":
                set_static_stem_input_sharding(
                    self,
                    stem_layers=set(self.stem_layers),
                )
                if parallelism.expert_parallel_degree > 1:
                    for embedding in self.stem_embeddings.values():
                        embedding.sharding_config = ShardingConfig(
                            state_shardings={
                                "weight": {
                                    MeshAxisName.DP_REPLICATE: Replicate(),
                                    MeshAxisName.EFSDP: Replicate(),
                                    MeshAxisName.EP: Shard(1),
                                }
                            }
                        )
            elif self.stem_strategy == "per_expert_projected":
                set_projected_stem_sharding(
                    self,
                    stem_layers=set(self.stem_layers),
                    enable_ep=parallelism.expert_parallel_degree > 1,
                )
            else:
                raise ValueError(f"unknown stem_strategy {self.stem_strategy!r}")

        def get_nparams_and_flops(
            self,
            model: nn.Module,
            seq_len: int,
        ) -> tuple[int, int]:
            nparams, num_flops_per_token = OLMo2MoEModel.Config.get_nparams_and_flops(
                self,
                model,
                seq_len,
            )
            if self.stem_strategy == "per_expert_projected":
                for layer_id in self.stem_layers:
                    experts = model.layers[str(layer_id)].moe.experts
                    num_flops_per_token += (
                        2 * experts.stem_up_FR.shape[0] * experts.stem_up_FR.shape[1]
                    )
            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__(config)
        self.stem_strategy = config.stem_strategy
        self._stem_layer_set = set(config.stem_layers)
        expected_keys = (
            {str(layer_id) for layer_id in self._stem_layer_set}
            if config.stem_strategy == "static"
            else set()
        )
        if set(config.stem_embeddings) != expected_keys:
            raise ValueError(
                "stem_embeddings keys do not match the selected strategy: "
                f"expected {sorted(expected_keys)}, got "
                f"{sorted(config.stem_embeddings)}"
            )
        self._stem_embeddings = ModuleDict(
            {
                layer_id: embedding_config.build()
                for layer_id, embedding_config in config.stem_embeddings.items()
            }
        )
        for layer_id in self._stem_layer_set:
            layer = self.layers[str(layer_id)]
            if config.stem_strategy == "static":
                if layer.moe_enabled and not isinstance(layer.moe, StaticStemMoE):
                    raise ValueError(
                        f"static STEM MoE layer {layer_id} must use StaticStemMoE"
                    )
                if not layer.moe_enabled and not isinstance(
                    layer.feed_forward, StemDagFeedForward
                ):
                    raise ValueError(
                        f"dense static STEM layer {layer_id} must use "
                        "StemDagFeedForward"
                    )
            elif not layer.moe_enabled:
                raise ValueError(
                    f"projected per-expert STEM layer {layer_id} must be an MoE layer"
                )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training:
            clear_aux_losses()

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        for layer_id, layer in self.layers.items():
            if int(layer_id) not in self._stem_layer_set:
                h = layer(h, self.freqs_cis, attention_masks, positions=positions)
            elif self.stem_strategy == "static":
                stem_embedding = self._stem_embeddings[layer_id](tokens).to(h.dtype)
                h = layer(
                    h,
                    self.freqs_cis,
                    attention_masks,
                    positions=positions,
                    stem_embedding=stem_embedding,
                )
            else:
                h = layer(
                    h,
                    self.freqs_cis,
                    attention_masks,
                    positions=positions,
                    token_ids=tokens,
                )

        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h
