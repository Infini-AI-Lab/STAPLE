# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import OLMo2Model


class OLMo2StateDictAdapter(StateDictAdapter):
    """Converts between HuggingFace OLMo2 and torchtitan state dict formats.

    HF OLMo2 uses "rotate_half" RoPE layout while torchtitan uses "complex"
    (interleaved pairs). Q and K weight matrices need permutation during
    conversion. QK-norm weights also need reshaping between 1D (HF) and 2D
    (torchtitan) with interleaving for the complex RoPE layout.
    """

    def __init__(
        self,
        model_config: OLMo2Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        # HF OLMo2 -> torchtitan mapping
        # OLMo2 HF names use "post_attention_layernorm" for the post-attn norm
        # and "input_layernorm" doesn't exist — OLMo2 uses post-norm.
        # HF OLMo2 key names:
        #   model.layers.X.self_attn.q_norm.weight -> QK norm on Q
        #   model.layers.X.self_attn.k_norm.weight -> QK norm on K
        #   model.layers.X.post_attention_layernorm.weight -> post-attn norm
        #   model.layers.X.post_feedforward_layernorm.weight -> post-ffn norm
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def _permute(
        self,
        w: torch.Tensor,
        n_heads: int,
        dim1: int | None = None,
        dim2: int | None = None,
    ) -> torch.Tensor:
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
        self,
        w: torch.Tensor,
        n_heads: int,
        dim1: int | None = None,
        dim2: int | None = None,
    ) -> torch.Tensor:
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

    def _reshape_qk_norm(self, value: torch.Tensor) -> torch.Tensor:
        """Reshape and interleave QK-norm weight for complex RoPE layout.

        HF stores q_norm/k_norm as 1D [n_heads * head_dim] in rotate_half order.
        Torchtitan with complex RoPE expects [n_heads, head_dim] in interleaved
        order within each head.
        """
        if value.dim() == 1:
            n_heads = self.model_config.layers[0].attention.n_heads
            head_dim = value.numel() // n_heads
            value = value.view(n_heads, head_dim)
            value = (
                value.view(n_heads, 2, head_dim // 2)
                .transpose(1, 2)
                .reshape(n_heads, head_dim)
            )
            return value
        return value

    def _flatten_qk_norm(self, value: torch.Tensor) -> torch.Tensor:
        """De-interleave and flatten QK-norm weight back to HF format.

        Torchtitan stores [n_heads, head_dim] in interleaved order.
        HF expects 1D [n_heads * head_dim] in rotate_half order.
        """
        if value.dim() == 2:
            n_heads, head_dim = value.shape
            value = (
                value.view(n_heads, head_dim // 2, 2)
                .transpose(1, 2)
                .reshape(n_heads, head_dim)
            )
            return value.view(-1)
        return value

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_config.layers[0].attention.n_heads
        n_kv_heads = self.model_config.layers[0].attention.n_kv_heads or n_heads
        dim = self.model_config.dim
        head_dim = dim // n_heads

        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
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

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_config.layers[0].attention.n_heads
        n_kv_heads = self.model_config.layers[0].attention.n_kv_heads or n_heads
        dim = self.model_config.dim
        head_dim = dim // n_heads

        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
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
