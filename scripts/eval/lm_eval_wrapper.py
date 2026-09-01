# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
lm-eval-harness LM wrapper for TorchTitan models.

Scores loglikelihood requests with one packed block-diagonal prefill and serves
generate_until requests with a packed KV-cached decoder (both from
generate_kv.py). Models without GQAttention fall back to padded direct forward
passes.
"""

import logging

import torch
import torch.distributed as dist
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class _MockAccelerator:
    """Minimal accelerator shim expected by lm_eval for multi-GPU gathering."""

    def gather(self, tensor):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered)

    def wait_for_everyone(self):
        if dist.is_initialized():
            dist.barrier()


def _truncate_at_stop(text: str, stops: list[str]) -> str:
    """Truncate generation at first stop sequence occurrence."""
    stop_positions = [text.find(stop) for stop in stops if stop]
    stop_positions = [pos for pos in stop_positions if pos >= 0]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


class TorchTitanEvalLM(LM):
    """lm-eval-harness LM interface over an already-loaded TorchTitan model.

    Args:
        model: TorchTitan model (already loaded, on device, in eval mode).
        tokenizer: HuggingFaceTokenizer instance.
        batch_size: Batch size for the padded loglikelihood fallback.
        max_length: Maximum sequence length for the model.
        gen_max_tokens: Token budget for packed prefill/generation batches.
        show_progress: Show tqdm progress bars on the fallback paths.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: HuggingFaceTokenizer,
        *,
        batch_size: int = 8,
        max_length: int = 4096,
        gen_max_tokens: int = 65536,
        show_progress: bool = False,
    ):
        super().__init__()
        self.accelerator = _MockAccelerator()
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._gen_max_tokens = gen_max_tokens
        self._show_progress = show_progress
        self._device = next(model.parameters()).device
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return self._max_length

    def tok_encode(self, string: str, add_special_tokens: bool = True) -> list[int]:
        return self._tokenizer.encode(string, add_bos=add_special_tokens, add_eos=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    # ------------------------------------------------------------------
    # Loglikelihood
    # ------------------------------------------------------------------
    #
    # Preferred path: a single packed block-diagonal flex prefill via
    # PackedKVCacheGenerator (mirrors STEM-pt eval.py, where loglikelihood is
    # generation with max_gen_len=1). This works for any model built from the
    # shared GQAttention (olmo2, olmo2_moe, stem_dag, engram_olmo2). For other
    # attention layers we fall back to the padded direct-forward path, which is
    # also correct.

    def _has_gqattention(self) -> bool:
        from torchtitan.models.common.attention import GQAttention

        return any(isinstance(m, GQAttention) for m in self._model.modules())

    def _make_score_generator(self):
        from generate_kv import GeneratorConfig, PackedKVCacheGenerator

        gen_cfg = GeneratorConfig(
            temperature=0.0,
            max_gen_len=1,
            max_tokens=self._gen_max_tokens,
            max_prompt_len=self._max_length,
            dtype="bf16",
        )
        return PackedKVCacheGenerator(
            gen_cfg, self._model, self._tokenizer, self._device
        )

    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        if self._has_gqattention():
            try:
                return self._loglikelihood_packed(requests)
            except Exception:
                logger.warning(
                    "Packed-flex loglikelihood failed, falling back to padded "
                    "direct forward",
                    exc_info=True,
                )
        return self._loglikelihood_padded(requests)

    def _loglikelihood_packed(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        """Score (context + continuation) with one packed flex prefill, then
        keep only the continuation tokens.

        Per-token ll[t] scores the prediction of input token t+1, so the
        continuation's logprobs are the last ``cont_len`` entries of token_ll.
        """
        full_texts = []
        cont_token_counts = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok_encode(context)
            cont_ids = self._tokenizer.encode(
                continuation, add_bos=False, add_eos=False
            )
            full_ids = (ctx_ids + cont_ids)[-self._max_length :]
            cont_len = min(len(cont_ids), len(full_ids))
            full_texts.append(full_ids)
            cont_token_counts.append(cont_len)

        scorer = self._make_score_generator()
        scored = scorer.score_token_ids(full_texts)

        results = []
        for (token_ll, token_greedy), cont_len in zip(scored, cont_token_counts):
            cont_ll = token_ll[-cont_len:]
            cont_greedy = token_greedy[-cont_len:]
            results.append((cont_ll.sum().item(), bool(cont_greedy.all().item())))
        return results

    def _loglikelihood_padded(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        results = []
        batches = range(0, len(requests), self._batch_size)
        if self._show_progress:
            batches = tqdm(batches, desc="loglikelihood", unit="batch")
        for i in batches:
            batch = requests[i : i + self._batch_size]
            results.extend(self._loglikelihood_padded_batch(batch))
        return results

    def _loglikelihood_padded_batch(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        encoded = []
        cont_lengths = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok_encode(context)
            cont_ids = self._tokenizer.encode(
                continuation, add_bos=False, add_eos=False
            )
            full_ids = (ctx_ids + cont_ids)[-self._max_length :]
            # Recalculate cont_len after truncation
            cont_len = min(len(cont_ids), len(full_ids))
            encoded.append(full_ids)
            cont_lengths.append(cont_len)

        max_len = max(len(ids) for ids in encoded)
        input_tensors = []
        for ids in encoded:
            padded = ids + [0] * (max_len - len(ids))
            input_tensors.append(padded)

        input_ids = torch.tensor(input_tensors, dtype=torch.long, device=self._device)
        logits = self._model(input_ids)

        results = []
        for j, (ids, cont_len) in enumerate(zip(encoded, cont_lengths)):
            seq_len = len(ids)
            cont_start = seq_len - cont_len
            log_probs = torch.nn.functional.log_softmax(
                logits[j, cont_start - 1 : seq_len - 1, :], dim=-1
            )
            target_ids = input_ids[j, cont_start:seq_len]
            token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

            total_ll = token_log_probs.sum().item()
            greedy_tokens = log_probs.argmax(dim=-1)
            is_greedy = (greedy_tokens == target_ids).all().item()
            results.append((total_ll, is_greedy))

        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        if self._has_gqattention():
            try:
                texts = [
                    self.tok_encode(req.args[0])[-self._max_length :]
                    for req in requests
                ]
                scorer = self._make_score_generator()
                scored = scorer.score_token_ids(texts)
                return [(token_ll.sum().item(),) for token_ll, _ in scored]
            except Exception:
                logger.warning(
                    "Packed-flex loglikelihood_rolling failed, falling back to "
                    "padded direct forward",
                    exc_info=True,
                )

        results = []
        for req in requests:
            (text,) = req.args
            ids = self.tok_encode(text)[-self._max_length :]
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
            logits = self._model(input_ids)

            log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
            targets = input_ids[0, 1:]
            token_log_probs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
            results.append((token_log_probs.sum().item(),))
        return results

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate with the packed KV-cached decoder.

        Falls back to naive quadratic generation if the KV-cached generator
        fails (e.g., the model uses non-GQAttention layers).
        """
        try:
            return self._generate_until_kv_cached(requests)
        except Exception:
            logger.warning(
                "KV-cached generation failed, falling back to naive generation",
                exc_info=True,
            )
            return self._generate_until_naive(requests)

    def _generate_until_kv_cached(self, requests: list[Instance]) -> list[str]:
        """Efficient generation using PackedKVCacheGenerator.

        lm-eval pools generate_until requests from ALL generation tasks into a
        single call, and tasks have heterogeneous gen_args (e.g. gsm8k stops at
        'Question:', humaneval stops at '\\ndef'/'\\nif'/'\\nprint', mbpp at
        '[DONE]', with different max_gen_toks). The generator applies one shared
        stop set + max length per batch, so we MUST group requests by gen_args
        and run one packed batch per group — otherwise one task's stop strings
        truncate another task's generation (e.g. humaneval's '\\n#'/'\\nprint'
        cut gsm8k chains-of-thought before the answer, collapsing gsm8k to 0).
        See apps/main/eval.py (STEM-pt) for the same grouping.
        """
        from collections import OrderedDict

        from generate_kv import GeneratorConfig, PackedKVCacheGenerator

        def _gen_args_key(gen_args: dict):
            # Hashable, order-independent key over the fields that affect
            # generation. List values (e.g. until, a set of stop strings whose
            # order is irrelevant) are normalized to sorted tuples so the same
            # config groups together regardless of element ordering.
            def _norm(v):
                if isinstance(v, list):
                    return tuple(sorted(v, key=str))
                return v

            return tuple(
                sorted(
                    ((k, _norm(v)) for k, v in gen_args.items()),
                    key=lambda kv: kv[0],
                )
            )

        # group key -> (gen_args, list of (original_index, prompt))
        groups: "OrderedDict" = OrderedDict()
        for i, req in enumerate(requests):
            prompt, gen_args = req.args
            key = _gen_args_key(gen_args)
            if key not in groups:
                groups[key] = (gen_args, [])
            groups[key][1].append((i, prompt))

        results: list[str] = [None] * len(requests)
        for gen_args, indexed_prompts in groups.values():
            until = gen_args.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_len = gen_args.get("max_gen_toks", 256)

            gen_cfg = GeneratorConfig(
                temperature=gen_args.get("temperature", 0.0),
                top_p=gen_args.get("top_p", None),
                top_k=gen_args.get("top_k", None),
                max_gen_len=max_gen_len,
                max_tokens=self._gen_max_tokens,
                max_prompt_len=self._max_length - max_gen_len,
                until=list(until),
                dtype="bf16",
            )
            generator = PackedKVCacheGenerator(
                gen_cfg, self._model, self._tokenizer, self._device
            )
            group_prompts = [p for _, p in indexed_prompts]
            generations = generator.generate(group_prompts)

            for (orig_idx, _), gen_text in zip(indexed_prompts, generations):
                results[orig_idx] = _truncate_at_stop(gen_text, until)

        return results

    def _generate_until_naive(self, requests: list[Instance]) -> list[str]:
        """Naive O(n²) generation fallback (recomputes full sequence each step)."""
        results = []
        it = requests
        if self._show_progress:
            it = tqdm(it, desc="generate_until", unit="req")
        for req in it:
            prompt, gen_args = req.args
            max_gen_tokens = gen_args.get("max_gen_toks", 256)
            until = gen_args.get("until", [])
            if isinstance(until, str):
                until = [until]
            temperature = gen_args.get("temperature", 0.0)

            ids = self.tok_encode(prompt)[-self._max_length :]
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)

            generated = self._greedy_generate(
                input_ids, max_new_tokens=max_gen_tokens, temperature=temperature
            )
            new_token_ids = generated[0, len(ids) :].tolist()
            text = self.tok_decode(new_token_ids)
            text = _truncate_at_stop(text, until)
            results.append(text)
        return results

    @torch.no_grad()
    def _greedy_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Simple autoregressive generation (no KV cache, O(n²))."""
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            if generated.shape[1] > self._max_length:
                generated = generated[:, -self._max_length :]
            logits = self._model(generated)
            next_logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if (
                self._tokenizer.eos_id is not None
                and next_token.item() == self._tokenizer.eos_id
            ):
                break
        return generated
