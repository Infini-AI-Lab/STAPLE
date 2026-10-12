# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter

from .model import OLMoEModel


class OLMoEStateDictAdapter(MoEStateDictAdapter):
    """Converts between HuggingFace OLMoE and torchtitan state dict formats.

    HF OLMoE stores experts as separate modules (model.layers.X.mlp.experts.Y.*).
    Torchtitan stores experts as 3D parameters (layers.X.moe.experts.w{1,2,3}_EFD).

    HF uses "rotate_half" RoPE layout while torchtitan uses "complex" (interleaved
    pairs). Q and K weight matrices need permutation during conversion.
    """

    def __init__(
        self,
        model_config: OLMoEModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        # HF OLMoE -> torchtitan mapping
        # HF OLMoE key naming convention:
        #   model.layers.X.self_attn.q_proj.weight
        #   model.layers.X.self_attn.k_proj.weight
        #   model.layers.X.self_attn.v_proj.weight
        #   model.layers.X.self_attn.o_proj.weight
        #   model.layers.X.self_attn.q_norm.weight  (QK-norm)
        #   model.layers.X.self_attn.k_norm.weight  (QK-norm)
        #   model.layers.X.input_layernorm.weight   (pre-attn norm)
        #   model.layers.X.post_attention_layernorm.weight (pre-FFN norm)
        #   model.layers.X.mlp.gate.weight          (router)
        #   model.layers.X.mlp.experts.Y.gate_proj.weight (expert gate/w1)
        #   model.layers.X.mlp.experts.Y.up_proj.weight   (expert up/w3)
        #   model.layers.X.mlp.experts.Y.down_proj.weight (expert down/w2)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # Norms
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1_EFD",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3_EFD",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2_EDF",
            # Final
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_config.layers[0].attention.n_heads
        n_kv_heads = self.model_config.layers[0].attention.n_kv_heads or n_heads
        dim = self.model_config.dim
        head_dim = dim // n_heads

        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    moe_layer = next(
                        l for l in self.model_config.layers if l.moe is not None
                    )
                    split_values = self._split_experts_weights(
                        value, moe_layer.moe.num_experts
                    )
                    for expert_num in range(moe_layer.moe.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
                if "q_norm" in key or "k_norm" in key:
                    value = self._flatten_qk_norm(value)
                if abstract_key == "layers.{}.attention.qkv_linear.wq.weight":
                    value = self._permute(value, n_heads)
                elif abstract_key == "layers.{}.attention.qkv_linear.wk.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)
                hf_state_dict[new_key] = value

            else:
                new_key = to_hf_map.get(key)
                if new_key is None:
                    continue
                hf_state_dict[new_key] = value

        return hf_state_dict

    def _permute(
        self, w: Any, n_heads: int, dim1: int | None = None, dim2: int | None = None
    ) -> Any:
        """Convert torchtitan (complex/interleaved) Q/K layout to HF (rotate_half)."""
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(
        self, w: Any, n_heads: int, dim1: int | None = None, dim2: int | None = None
    ) -> Any:
        """Convert HF (rotate_half) Q/K layout to torchtitan (complex/interleaved)."""
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def _reshape_qk_norm(self, value: Any) -> Any:
        """Reshape and interleave QK-norm weight for complex RoPE layout.

        HF OLMoE stores q_norm/k_norm as 1D [n_heads * head_dim] in rotate_half
        order: [h0_0, h0_1, ..., h0_{d/2-1}, h0_{d/2}, ..., h0_{d-1}, h1_0, ...]
        Torchtitan with complex RoPE expects interleaved order within each head:
        [h0_0, h0_{d/2}, h0_1, h0_{d/2+1}, ...] reshaped to [n_heads, head_dim].
        """
        import torch

        if isinstance(value, torch.Tensor) and value.dim() == 1:
            n_heads = self.model_config.layers[0].attention.n_heads
            head_dim = value.numel() // n_heads
            # Reshape to [n_heads, head_dim], then interleave within each head
            value = value.view(n_heads, head_dim)
            # Split each head into two halves and interleave
            # [n_heads, 2, head_dim//2] -> transpose -> [n_heads, head_dim//2, 2] -> flatten last two dims
            value = (
                value.view(n_heads, 2, head_dim // 2)
                .transpose(1, 2)
                .reshape(n_heads, head_dim)
            )
            return value
        return value

    def _flatten_qk_norm(self, value: Any) -> Any:
        """De-interleave and flatten QK-norm weight back to HF format.

        Torchtitan stores [n_heads, head_dim] in interleaved order.
        HF expects 1D [n_heads * head_dim] in rotate_half order.
        """
        import torch

        if isinstance(value, torch.Tensor) and value.dim() == 2:
            n_heads, head_dim = value.shape
            # De-interleave: [n_heads, head_dim//2, 2] -> transpose -> [n_heads, 2, head_dim//2] -> flatten
            value = (
                value.view(n_heads, head_dim // 2, 2)
                .transpose(1, 2)
                .reshape(n_heads, head_dim)
            )
            return value.view(-1)
        return value

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_config.layers[0].attention.n_heads
        n_kv_heads = self.model_config.layers[0].attention.n_kv_heads or n_heads
        dim = self.model_config.dim
        head_dim = dim // n_heads

        state_dict = {}
        expert_weights_by_layer = {}

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map.get(abstract_key)
                if titan_abstract_key is None:
                    continue
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:
                    moe_layer = next(
                        l for l in self.model_config.layers if l.moe is not None
                    )
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        moe_layer.moe.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
                if "q_norm" in key or "k_norm" in key:
                    value = self._reshape_qk_norm(value)
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                elif abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    continue
                state_dict[new_key] = value

        return state_dict
