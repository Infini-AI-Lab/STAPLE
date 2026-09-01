# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Efficient KV-cached generation and scoring for GQAttention decoder models.

Model-agnostic: works with any model built from the shared
``torchtitan.models.common.attention.GQAttention`` (olmo2, olmo2_moe, stem_dag,
engram_olmo2). It implements generation and loglikelihood scoring by using a
packed-sequence KV cache similar to the STEM-pt approach. Multiple prompts are
packed into a single sequence dimension, with block-diagonal document masks
preventing cross-sequence attention.

The KV cache uses layout (1, max_tokens, n_kv_heads, head_dim) where all
documents share the sequence dimension. Each document is allocated a
contiguous slice of [prompt_len + max_gen_len] in the cache. This avoids
padding waste and naturally supports variable-length sequences.

Loglikelihood scoring (:meth:`PackedKVCacheGenerator.score_token_ids`) is the
same packed flex prefill with no autoregressive decode (STEM-pt expresses this
as generation with ``max_gen_len=1``).

Used by lm_eval_wrapper.py; there is no standalone entry point.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torchtitan.models.common.attention import GQAttention
from tqdm import tqdm


# Compiled flex_attention used for the packed block-diagonal *prefill* (and the
# loglikelihood prefill, which is the same forward with no decode). We call this
# directly rather than going through the model's `inner_attention`, because the
# model is built with the "sdpa" backend whose forward ignores attention_masks
# (see _gqattention_forward_with_cache for the full explanation).
_compiled_flex_attention = torch.compile(flex_attention)


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


class KVCache(nn.Module):
    """Fixed-size KV cache for packed-sequence generation.

    Layout: (1, max_tokens, n_kv_heads, head_dim) — all documents share the
    sequence dimension. Each document writes to its allocated offset via
    index_copy_ on dim=1. This avoids batch-dimension padding and aligns with
    how the packed attention mask works.
    """

    def __init__(
        self,
        max_tokens: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        shape = (1, max_tokens, n_kv_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.offset: torch.Tensor | int = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.offset = 0

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor, tok_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V at positions (offset + tok_idx) and return full cache."""
        positions = self.offset + tok_idx
        self.k_cache.index_copy_(1, positions, k_val)
        self.v_cache.index_copy_(1, positions, v_val)
        return self.k_cache, self.v_cache


# ---------------------------------------------------------------------------
# Attention forward with KV cache
# ---------------------------------------------------------------------------


def _gqattention_forward_with_cache(
    self: GQAttention,
    x: torch.Tensor,
    rope_cache: torch.Tensor,
    attention_masks,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Monkey-patched GQAttention.forward that uses KV cache during generation."""
    from torchtitan.models.common.rope import (
        apply_rotary_emb_complex,
        apply_rotary_emb_cos_sin,
    )

    bs, seqlen, _ = x.shape
    xq, xk, xv = self.qkv_linear(x)

    if self.q_norm is not None or self.k_norm is not None:
        assert self.q_norm is not None and self.k_norm is not None
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

    if self.use_rope:
        if self.rope_backend == "cos_sin":
            xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)
        else:
            xq, xk = apply_rotary_emb_complex(
                xq, xk, freqs_cis=rope_cache, positions=positions
            )

    if hasattr(self, "_kv_cache") and self._kv_cache is not None:
        k_full, v_full = self._kv_cache.update(xk, xv, self._tok_idx)
        # Use the full cache for attention
        xk, xv = k_full, v_full

    if isinstance(attention_masks, dict):
        mask_key = "rope" if self.use_rope else "nope"
        attention_masks = attention_masks[mask_key]

    # During generation with cache, we use SDPA with explicit mask
    if hasattr(self, "_kv_cache") and self._kv_cache is not None:
        # xq: (1, n_seqs, n_heads, head_dim) — packed queries
        # xk, xv: (1, max_tokens, n_kv_heads, head_dim) — full cache
        # attention_masks: (n_seqs, max_tokens) bool or BlockMask

        if (
            isinstance(attention_masks, torch.Tensor)
            and attention_masks.dtype == torch.bool
        ):
            # Generation step: use SDPA with explicit bool mask
            # Transpose to SDPA layout: (batch, heads, seq, head_dim)
            xq_t = xq.transpose(1, 2)  # (1, n_heads, n_seqs, head_dim)
            xk_t = xk.transpose(1, 2)  # (1, n_kv_heads, max_tokens, head_dim)
            xv_t = xv.transpose(1, 2)  # (1, n_kv_heads, max_tokens, head_dim)

            # attention_masks is (n_seqs, max_tokens) bool
            # SDPA expects attn_mask broadcastable to (batch, heads, q_len, kv_len)
            # Shape: (1, 1, n_seqs, max_tokens) broadcasts over batch and heads
            attn_bias = torch.where(
                attention_masks.unsqueeze(0).unsqueeze(0),
                torch.zeros(1, device=xq.device, dtype=xq.dtype),
                torch.finfo(xq.dtype).min,
            )
            output = F.scaled_dot_product_attention(
                xq_t,
                xk_t,
                xv_t,
                attn_mask=attn_bias,
                scale=self.scaling,
                enable_gqa=self.enable_gqa,
            )
            # Back to (1, n_seqs, n_heads, head_dim) then flatten heads
            output = output.transpose(1, 2).contiguous()
            output = output.view(bs, seqlen, -1)
            return self.wo(output)
        else:
            # Prefill (also used by loglikelihood, which is prefill with
            # max_gen_len=1): use flex_attention with the packed block-diagonal
            # BlockMask. We call the compiled flex_attention directly rather
            # than self.inner_attention because the model is built with the
            # "sdpa" backend, whose ScaledDotProductAttention.forward drops
            # attention_masks into **kwargs and applies plain is_causal=True —
            # silently ignoring the packed mask and corrupting every document
            # except the first.
            # flex_attention layout is (batch, heads, seq, head_dim).
            xq_t = xq.transpose(1, 2)  # (1, n_heads, total_prefill, head_dim)
            xk_t = xk.transpose(1, 2)  # (1, n_kv_heads, max_tokens, head_dim)
            xv_t = xv.transpose(1, 2)
            output = _compiled_flex_attention(
                xq_t,
                xk_t,
                xv_t,
                block_mask=attention_masks,
                scale=self.scaling,
                enable_gqa=self.enable_gqa,
            )
            output = output.transpose(1, 2).contiguous()
            output = output.view(bs, seqlen, -1)
            return self.wo(output)

    # Standard forward without cache (should not normally be reached)
    output = self.inner_attention(
        xq,
        xk,
        xv,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
    ).contiguous()
    output = output.view(bs, seqlen, -1)
    return self.wo(output)


# ---------------------------------------------------------------------------
# Engram decode cache
# ---------------------------------------------------------------------------
#
# Engram layers (experiments/engram_olmo2) fuse a hashed N-gram memory into the
# residual stream and pass the gated value through a short dilated *causal*
# conv. Both pieces look back across positions, so packed/KV-cached decoding --
# where each model step sees only one token per document, packed along the
# "sequence" axis -- needs the same per-document history the packed prefill had:
#
#   * Hashing needs the last ``max_ngram-1`` token ids of each document. These
#     are plain ids the generator already owns, so we keep a small rolling id
#     window and rebuild each document's suffix N-gram from it (no real state).
#   * The short conv needs the last ``(kernel-1)*dilation`` gated values ``ṽ``
#     of each document. Those depend on past hidden states (gone once the KV
#     cache replaces them), so they MUST be cached -- seeded from the prefill
#     and rolled one step at a time.
#
# We monkey-patch each Engram module's ``compute_hashes`` and ``forward`` (the
# same approach as the attention KV cache above) and dispatch on a per-module
# ``_dec_mode`` flag. Models without Engram layers are unaffected: the generator
# simply finds no Engram modules and installs nothing.


def _is_engram_module(m: nn.Module) -> bool:
    """Duck-type an EngramLayer without importing the experiment package."""
    return (
        hasattr(m, "compute_hashes")
        and hasattr(m, "_compute_value")
        and hasattr(m, "short_conv")
        and hasattr(m, "hasher")
        and hasattr(m, "max_ngram_size")
    )


def _engram_compute_hashes_with_cache(self, tokens, positions=None):
    """Patched ``EngramLayer.compute_hashes`` that is decode-cache aware.

    Prefill (``_dec_mode != "decode"``): delegate to the original boundary-aware
    hashing over the packed prompt. Decode: rebuild each document's own suffix
    N-gram window from the cached id history + the current token, hash it, and
    return only the current token's indices ``(1, n_seqs, num_heads)``.
    """
    if getattr(self, "_dec_mode", "prefill") != "decode":
        return self._orig_compute_hashes(tokens, positions)

    n = self.max_ngram_size
    cur = tokens.reshape(-1, 1)  # (n_seqs, 1) current id per document
    window = torch.cat([self._dec_id_window, cur], dim=1)  # (n_seqs, n)
    tok_id = self._dec_tok_id  # (n_seqs,) current within-document offset
    rel = torch.arange(-(n - 1), 1, device=tok_id.device)  # (n,)
    # Per-token within-document offsets for the window; the hasher masks any
    # tap with offset < k (reaching before the document start) to pad, exactly
    # as the packed forward does -- so short documents / sequence starts get the
    # same padded N-gram they would standalone.
    window_pos = tok_id.unsqueeze(1) + rel  # (n_seqs, n)
    hashes = self.hasher(window, window_pos)  # (n_seqs, n, num_heads)

    # Roll the id history to include the current token (drop the oldest). Each
    # Engram layer holds an identical copy and is called exactly once per step,
    # so rolling here advances the window once per decode step.
    if n > 1:
        self._dec_id_window = torch.cat([self._dec_id_window[:, 1:], cur], dim=1)

    return hashes[:, -1].unsqueeze(0)  # (1, n_seqs, num_heads)


def _engram_forward_with_cache(self, hidden_states, hash_ids, positions=None):
    """Patched ``EngramLayer.forward`` that is decode-cache aware.

    Prefill: run the normal boundary-aware forward, but stash the gated value
    ``ṽ`` so the generator can seed the conv history cache. Decode: compute the
    current token's ``ṽ_t`` and run one cached causal-conv step, then roll the
    ``ṽ`` history.
    """
    if getattr(self, "_dec_mode", "prefill") != "decode":
        value = self._compute_value(hidden_states, hash_ids)  # [1, T, D]
        self._prefill_v = value
        return value + self.short_conv(value, positions)

    value = self._compute_value(hidden_states, hash_ids)  # [1, n_seqs, D]
    v_t = value[0]  # [n_seqs, D]
    conv_out = self.short_conv.decode_step(v_t, self._v_cache, self._dec_tok_id)
    # Roll the ṽ history (drop oldest, append current) for the next step.
    self._v_cache = torch.cat([self._v_cache[:, 1:], v_t.unsqueeze(1)], dim=1)
    return (v_t + conv_out).unsqueeze(0)  # [1, n_seqs, D]


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    topk_value, _ = torch.topk(probs, k)
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
    shape = logits.shape
    logits = logits.flatten(end_dim=-2)
    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_p is not None:
            next_token = sample_top_p(probs, top_p)
        elif top_k is not None:
            next_token = sample_top_k(probs, top_k)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token.view(*shape[:-1])


# ---------------------------------------------------------------------------
# Packed sequence utilities
# ---------------------------------------------------------------------------


def lengths_to_start_ids(lengths: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum: [2, 3, 4] -> [0, 2, 5]."""
    return torch.cat(
        [
            torch.zeros(1, dtype=lengths.dtype, device=lengths.device),
            lengths.cumsum(0)[:-1],
        ]
    )


def lengths_to_local_ids(lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For packed tokens, return (doc_id, local_position_in_doc) for each token."""
    total = lengths.sum().item()
    doc_id = torch.repeat_interleave(
        torch.arange(len(lengths), device=lengths.device), lengths
    )
    starts = lengths_to_start_ids(lengths)
    offsets = torch.repeat_interleave(starts, lengths)
    tok_id = torch.arange(total, device=lengths.device) - offsets
    return doc_id, tok_id


def pack_prompts(prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack list of token lists into a single tensor + lengths."""
    tensors = [torch.tensor(p, dtype=torch.long) for p in prompts]
    lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)
    packed = torch.cat(tensors)
    return packed, lengths


def batch_prompts(
    prompts: list[list[int]], max_tokens: int, max_gen_len: int
) -> list[list[list[int]]]:
    """Group prompts into batches that fit within max_tokens."""
    batches = []
    current_batch = []
    current_count = 0

    for p in prompts:
        padded_size = len(p) + max_gen_len
        if current_count + padded_size <= max_tokens:
            current_batch.append(p)
            current_count += padded_size
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch = [p]
            current_count = padded_size

    if current_batch:
        batches.append(current_batch)
    return batches


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    max_gen_len: int = 128
    max_tokens: int = 8192
    max_prompt_len: int | None = None
    until: list[str] = field(default_factory=list)
    dtype: str = "bf16"


class PackedKVCacheGenerator:
    """Efficient generation with KV caching for packed sequences.

    Multiple prompts are packed into a single sequence dimension. Each prompt
    gets an allocated cache region of [prompt_len + max_gen_len]. During
    prefill, a block-diagonal attention mask ensures documents don't attend
    to each other. During generation, each document generates one token per
    step and the cache is updated via index_copy_.
    """

    def __init__(
        self,
        cfg: GeneratorConfig,
        model: nn.Module,
        tokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg.dtype]

        self._original_forwards: dict[str, callable] = {}

    def _install_kv_caches(self, max_tokens: int) -> None:
        """Attach KV caches to all GQAttention modules.

        If caches are already installed and large enough, this is a no-op.
        If the existing caches are too small, they are reallocated.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, GQAttention):
                existing = getattr(module, "_kv_cache", None)
                if existing is not None and existing.k_cache.shape[1] >= max_tokens:
                    continue
                n_kv_heads = module.n_kv_heads
                head_dim = module.head_dim
                module._kv_cache = KVCache(
                    max_tokens, n_kv_heads, head_dim, self.dtype, self.device
                )
                module._tok_idx = None
                if name not in self._original_forwards:
                    self._original_forwards[name] = module.forward
                    module.forward = _gqattention_forward_with_cache.__get__(
                        module, type(module)
                    )

    def _reset_kv_caches(self) -> None:
        """Zero all KV caches between batches without removing them."""
        for module in self.model.modules():
            if isinstance(module, GQAttention) and hasattr(module, "_kv_cache"):
                module._kv_cache.reset()

    def _remove_kv_caches(self) -> None:
        """Remove KV caches and restore original forward methods."""
        for name, module in self.model.named_modules():
            if isinstance(module, GQAttention):
                if hasattr(module, "_kv_cache"):
                    del module._kv_cache
                if hasattr(module, "_tok_idx"):
                    del module._tok_idx
                if name in self._original_forwards:
                    module.forward = self._original_forwards[name]
        self._original_forwards.clear()

    def _set_cache_offsets(self, offset: torch.Tensor) -> None:
        """Set the KV cache offset for all attention layers."""
        for module in self.model.modules():
            if isinstance(module, GQAttention) and hasattr(module, "_kv_cache"):
                module._kv_cache.offset = offset

    def _set_tok_idx(self, tok_idx: torch.Tensor) -> None:
        """Set the token index for all attention layers."""
        for module in self.model.modules():
            if isinstance(module, GQAttention) and hasattr(module, "_kv_cache"):
                module._tok_idx = tok_idx

    # ------------------------------------------------------------------
    # Engram decode cache (no-op for models without Engram layers)
    # ------------------------------------------------------------------

    def _engram_modules(self):
        return [m for m in self.model.modules() if _is_engram_module(m)]

    def _install_engram_caches(self) -> None:
        """Patch Engram modules' compute_hashes/forward for decode caching.

        Idempotent. No-op when the model has no Engram layers, so non-Engram
        models pay nothing.
        """
        for name, module in self.model.named_modules():
            if not _is_engram_module(module):
                continue
            if getattr(module, "_engram_patched", False):
                continue
            module._orig_compute_hashes = module.compute_hashes
            module._orig_engram_forward = module.forward
            module.compute_hashes = _engram_compute_hashes_with_cache.__get__(
                module, type(module)
            )
            module.forward = _engram_forward_with_cache.__get__(module, type(module))
            module._dec_mode = "prefill"
            module._prefill_v = None
            module._engram_patched = True

    def _set_engram_mode(self, mode: str) -> None:
        for m in self._engram_modules():
            m._dec_mode = mode

    def _seed_engram_caches(
        self,
        lengths: torch.Tensor,
        prompt_ids: list[list[int]],
    ) -> None:
        """Seed each Engram layer's per-document conv-``ṽ`` and id history from
        the just-run prefill, so the first generated token continues the
        document rather than restarting it.

        ``_prefill_v`` on each module is the packed ``ṽ`` ``(1, total, D)``
        captured during the prefill forward. For each document we take its last
        ``history_len`` values (front-zero-padded if the document is shorter)
        and its last ``max_ngram-1`` token ids.
        """
        modules = self._engram_modules()
        if not modules:
            return
        n_seqs = len(prompt_ids)
        starts = lengths_to_start_ids(lengths).tolist()
        lens = lengths.tolist()

        for m in modules:
            P = m.short_conv.history_len
            v = m._prefill_v[0]  # (total_prefill, D)
            D = v.shape[-1]
            v_cache = v.new_zeros(n_seqs, P, D)
            for i in range(n_seqs):
                s, L = int(starts[i]), int(lens[i])
                take = min(P, L)
                if take > 0:
                    v_cache[i, P - take :] = v[s + L - take : s + L]
            m._v_cache = v_cache  # (n_seqs, P, D)
            m._prefill_v = None

            n = m.max_ngram_size
            id_window = torch.zeros(
                n_seqs, max(n - 1, 0), dtype=torch.long, device=self.device
            )
            for i in range(n_seqs):
                ids = prompt_ids[i]
                take = min(n - 1, len(ids))
                if take > 0:
                    id_window[i, (n - 1) - take :] = torch.tensor(
                        ids[-take:], dtype=torch.long, device=self.device
                    )
            m._dec_id_window = id_window  # (n_seqs, n-1)

    def _set_engram_tok_id(self, tok_id: torch.Tensor) -> None:
        for m in self._engram_modules():
            m._dec_tok_id = tok_id

    def _reset_engram_caches(self) -> None:
        for m in self._engram_modules():
            m._dec_mode = "prefill"
            m._prefill_v = None
            for attr in ("_v_cache", "_dec_id_window", "_dec_tok_id"):
                if hasattr(m, attr):
                    delattr(m, attr)

    def _remove_engram_caches(self) -> None:
        for m in self._engram_modules():
            if not getattr(m, "_engram_patched", False):
                continue
            m.compute_hashes = m._orig_compute_hashes
            m.forward = m._orig_engram_forward
            for attr in (
                "_orig_compute_hashes",
                "_orig_engram_forward",
                "_dec_mode",
                "_prefill_v",
                "_v_cache",
                "_dec_id_window",
                "_dec_tok_id",
                "_engram_patched",
            ):
                if hasattr(m, attr):
                    delattr(m, attr)

    @torch.inference_mode()
    def generate(self, prompts: list[str]) -> list[str]:
        """Generate completions for a list of prompts with KV caching."""
        cfg = self.cfg

        # Tokenize
        encoded = [
            self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts
        ]

        # Truncate
        max_prompt_len = cfg.max_prompt_len or (cfg.max_tokens - cfg.max_gen_len)
        encoded = [p[-max_prompt_len:] for p in encoded]

        # Batch prompts
        batches = batch_prompts(encoded, cfg.max_tokens, cfg.max_gen_len)

        # Install caches once with max_tokens capacity
        self._install_kv_caches(cfg.max_tokens)
        # Engram models also need a per-document N-gram/short-conv decode cache
        # (no-op for non-Engram models).
        self._install_engram_caches()

        all_generations = []
        try:
            batch_iter = tqdm(batches, desc="generate", unit="batch")
            for batch in batch_iter:
                self._reset_kv_caches()
                self._reset_engram_caches()
                generations = self._generate_batch(batch)
                all_generations.extend(generations)
        finally:
            self._remove_kv_caches()
            self._remove_engram_caches()

        return all_generations

    @torch.inference_mode()
    def score_token_ids(
        self, encoded: list[list[int]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-token loglikelihood + greedy match for pre-tokenized prompts.

        This is the loglikelihood analog of :meth:`generate`: it runs the same
        packed block-diagonal flex prefill but does no autoregressive decode
        (STEM-pt expresses this as generation with ``max_gen_len=1``). Returns
        one ``(token_ll, token_greedy)`` pair per prompt, each of length
        ``len(ids) - 1`` and aligned so that index ``t`` scores the prediction
        of input token ``t+1``. Callers tokenize (and choose bos/eos)
        themselves.
        """
        # Truncate. With no generation, the full cache budget is for the prompt.
        max_prompt_len = self.cfg.max_prompt_len or self.cfg.max_tokens
        encoded = [p[-max_prompt_len:] for p in encoded]

        # Scoring is prefill with no decode: temporarily drop max_gen_len to 1
        # so the cache reserves no generation tail (STEM-pt does the same).
        # batch_prompts and _prefill_batch must use the same max_gen_len for the
        # "batch fits in max_tokens" invariant to hold.
        saved_max_gen_len = self.cfg.max_gen_len
        self.cfg.max_gen_len = 1
        self._install_kv_caches(self.cfg.max_tokens)

        all_scores = []
        try:
            batches = batch_prompts(encoded, self.cfg.max_tokens, self.cfg.max_gen_len)
            for batch in tqdm(batches, desc="score", unit="batch"):
                self._reset_kv_caches()
                all_scores.extend(self._score_batch(batch))
        finally:
            self._remove_kv_caches()
            self.cfg.max_gen_len = saved_max_gen_len

        return all_scores

    def _prefill_batch(self, batch: list[list[int]]) -> tuple[torch.Tensor, dict]:
        """Pack a batch and run the block-diagonal flex prefill.

        Returns the prefill logits ``(1, total_prefill, vocab)`` and a state
        dict with the per-document index tensors needed for decoding. This is
        the single packed flex forward shared by both generation
        (:meth:`_generate_batch`) and loglikelihood scoring
        (:meth:`_score_batch`) — loglikelihood is just this forward with no
        subsequent decode (STEM-pt sets ``max_gen_len=1``).
        """
        cfg = self.cfg
        n_seqs = len(batch)

        packed, lengths = pack_prompts(batch)
        packed = packed.to(self.device)
        lengths = lengths.to(self.device)

        padded_lengths = lengths + cfg.max_gen_len
        max_tokens = max(cfg.max_tokens, padded_lengths.sum().item())
        padded_lengths[-1] += max_tokens - padded_lengths.sum()

        # Ensure caches are large enough for this batch
        self._install_kv_caches(max_tokens)

        # --- Prefill ---
        padded_doc_start = lengths_to_start_ids(padded_lengths)
        prefill_offset = torch.repeat_interleave(padded_doc_start, lengths)

        self._set_cache_offsets(prefill_offset)

        _, prefill_tok_id = lengths_to_local_ids(lengths)
        self._set_tok_idx(prefill_tok_id)

        # Build prefill positions for RoPE (local position in each doc)
        prefill_positions = prefill_tok_id.unsqueeze(0)  # (1, total_tokens)

        # Build prefill attention mask (block-diagonal causal). Query indices
        # live in the compact packed space (total_prefill); key indices live in
        # the padded cache space (max_tokens, with a per-doc max_gen_len gap).
        padded_doc_ids, padded_tok_ids = lengths_to_local_ids(padded_lengths)
        query_doc_ids = torch.repeat_interleave(
            torch.arange(n_seqs, device=self.device), lengths
        )
        local_positions = prefill_tok_id

        def prefill_mask_mod(b, h, q_idx, kv_idx):
            q_doc = query_doc_ids[q_idx]
            kv_doc = padded_doc_ids[kv_idx]
            kv_local = padded_tok_ids[kv_idx]
            q_local = local_positions[q_idx]
            return (q_doc == kv_doc) & (q_local >= kv_local)

        total_prefill = lengths.sum().item()
        prefill_mask = create_block_mask(
            prefill_mask_mod, 1, None, total_prefill, max_tokens
        )

        # Engram decode caches are installed only on the generate() path (not
        # loglikelihood scoring, which is prefill-only and already correct).
        engram_decoding = any(
            getattr(m, "_engram_patched", False) for m in self._engram_modules()
        )
        if engram_decoding:
            # Run the prefill in "prefill" mode so each Engram layer captures
            # its packed gated value ``ṽ`` for seeding the decode cache.
            self._set_engram_mode("prefill")

        # Run prefill (flex_attention with the block-diagonal BlockMask)
        prefill_logits = self.model(
            packed.unsqueeze(0),
            attention_masks=prefill_mask,
            positions=prefill_positions,
        )

        if engram_decoding:
            # Seed each layer's conv-``ṽ`` history + N-gram id window from the
            # prompt, then switch to decode mode for the autoregressive steps.
            self._seed_engram_caches(lengths, batch)
            self._set_engram_mode("decode")

        state = {
            "lengths": lengths,
            "padded_doc_ids": padded_doc_ids,
            "padded_tok_ids": padded_tok_ids,
            "padded_doc_start": padded_doc_start,
            "n_seqs": n_seqs,
        }
        return prefill_logits, state

    def _score_batch(
        self, batch: list[list[int]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-document token loglikelihood + greedy match from one prefill.

        Mirrors STEM-pt: for each packed document, ``token_ll[t]`` is the log
        probability of predicting token ``t+1`` from position ``t``. Returns
        one ``(token_ll, token_greedy)`` pair per document, each of length
        ``len(doc) - 1``.
        """
        prefill_logits, state = self._prefill_batch(batch)
        lengths = state["lengths"]
        logits = prefill_logits[0]  # (total_prefill, vocab)

        results = []
        for doc_tokens, doc_logits in zip(batch, logits.split(lengths.tolist())):
            x = doc_logits[:-1].float()
            y = torch.tensor(doc_tokens[1:], device=x.device)
            token_ll = -F.cross_entropy(x, y, reduction="none")
            token_greedy = x.argmax(dim=-1) == y
            results.append((token_ll.cpu(), token_greedy.cpu()))
        return results

    def _generate_batch(self, batch: list[list[int]]) -> list[str]:
        """Generate for a single batch of packed prompts."""
        cfg = self.cfg

        prefill_logits, state = self._prefill_batch(batch)
        lengths = state["lengths"]
        padded_doc_ids = state["padded_doc_ids"]
        padded_tok_ids = state["padded_tok_ids"]
        padded_doc_start = state["padded_doc_start"]
        n_seqs = state["n_seqs"]

        # Sample first generated token from the last position of each prompt
        cumulative = lengths.cumsum(0) - 1
        last_logits = prefill_logits[0, cumulative]  # (n_seqs, vocab)
        start_tokens = sample_tokens(
            last_logits, cfg.temperature, cfg.top_p, cfg.top_k
        )  # (n_seqs,)

        # --- Generation ---
        self._set_cache_offsets(padded_doc_start)

        current_tok_id = lengths.clone()
        generated_tokens = [[tok.item()] for tok in start_tokens]
        is_done = [False] * n_seqs
        max_until_size = max(len(e) for e in cfg.until) if cfg.until else 1

        current_token = start_tokens  # (n_seqs,)

        for step in range(1, cfg.max_gen_len):
            self._set_tok_idx(current_tok_id)
            # Engram decode uses the same within-document offset as the KV cache
            # (no-op for non-Engram models).
            self._set_engram_tok_id(current_tok_id)

            # Build generation mask: (n_seqs, max_tokens)
            # Each sequence attends to its doc and up to current position
            current_doc_id = torch.arange(n_seqs, device=self.device)
            doc_mask = current_doc_id.unsqueeze(1) == padded_doc_ids.unsqueeze(0)
            caus_mask = current_tok_id.unsqueeze(1) >= padded_tok_ids.unsqueeze(0)
            gen_mask = doc_mask & caus_mask  # (n_seqs, max_tokens)

            # Positions for RoPE
            gen_positions = current_tok_id.unsqueeze(0)  # (1, n_seqs)

            # Forward pass
            next_logits = self.model(
                current_token.unsqueeze(0),  # (1, n_seqs)
                attention_masks=gen_mask,
                positions=gen_positions,
            )

            # Sample
            next_tokens = sample_tokens(
                next_logits[0], cfg.temperature, cfg.top_p, cfg.top_k
            )  # (n_seqs,)

            # Update state
            current_tok_id += 1

            for seq_id, tok in enumerate(next_tokens.tolist()):
                if not is_done[seq_id]:
                    generated_tokens[seq_id].append(tok)
                    current_end_str = self.tokenizer.decode(
                        generated_tokens[seq_id][-max_until_size:]
                    )
                    contains_end = any(e in current_end_str for e in cfg.until)
                    is_done[seq_id] = contains_end or tok == self.tokenizer.eos_id

            if all(is_done):
                break

            current_token = next_tokens

        return [self.tokenizer.decode(g) for g in generated_tokens]
