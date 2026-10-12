# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared chat-tokenization utilities for supervised fine-tuning (SFT).

Turns a list of chat ``messages`` into ``(tokens, loss_mask)`` where
``loss_mask[j] == 1`` iff token ``j`` belongs to an assistant turn (its content
and the trailing EOS). Both the HuggingFace-datasets SFT path
(:class:`~torchtitan.hf_datasets.text_datasets.ChatDataset`) and the JSONL SFT
path (:mod:`torchtitan.hf_datasets.jsonl_datasets`) build their per-token loss
masks from this helper so the prompt-masking logic lives in exactly one place.
"""

from typing import Any

from torchtitan.components.tokenizer import BaseTokenizer

# Roles permitted in a conversation. Only ``assistant`` turns contribute loss.
_PROMPT_ROLES = ("system", "user")
_ASSISTANT_ROLE = "assistant"


def validate_messages(messages: list[dict[str, str]]) -> None:
    """Validate a (possibly multi-turn) conversation.

    Requires every role to be one of ``system``/``user``/``assistant`` and at
    least one ``assistant`` turn (otherwise the sample would be fully masked and
    contribute no loss).
    """
    if not messages:
        raise ValueError("messages must be a non-empty list of chat turns")
    has_assistant = False
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == _ASSISTANT_ROLE:
            has_assistant = True
        elif role not in _PROMPT_ROLES:
            raise ValueError(
                f"messages[{i}] has unsupported role {role!r}; expected one of "
                f"{_PROMPT_ROLES + (_ASSISTANT_ROLE,)}"
            )
    if not has_assistant:
        raise ValueError("conversation must contain at least one assistant turn")


def encode_chat_sample(
    tokenizer: BaseTokenizer,
    messages: list[dict[str, Any]],
    *,
    seq_len: int,
    validate: bool = True,
) -> tuple[list[int], list[int]] | None:
    """Tokenize a conversation and build a per-token assistant-response mask.

    Args:
        tokenizer: tokenizer with a chat template and a valid ``eos_id``.
        messages: list of ``{"role", "content"}`` turns (multi-turn supported).
        seq_len: maximum input length; samples whose input (``len(tokens) - 1``,
            accounting for the next-token shift) exceeds this are dropped
            (returns ``None``) to avoid training on truncated responses.
        validate: run :func:`validate_messages` first. ``ChatDataset`` performs
            its own stricter single-turn validation and passes ``False``.

    Returns:
        ``(tokens, loss_mask)`` where ``tokens`` is the full id sequence
        (``bos ... eos``) and ``loss_mask[j] == 1`` iff token ``j`` is an
        assistant-response token (content + its EOS), else ``0``. Returns
        ``None`` if the sample exceeds ``seq_len``.

    The mask is built by incremental prefix re-tokenization (the same technique
    :class:`ChatDataset` uses for the single-turn case): for each assistant turn
    ``t`` the response span ``[start, end)`` is delimited by the encoded length
    of the conversation prefix with a generation prompt (``start``) and through
    the assistant turn (``end``). This avoids BPE merge errors at the
    prompt/response boundary.
    """
    if validate:
        validate_messages(messages)

    full_text = tokenizer.apply_chat_template(messages).rstrip("\n")
    tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)
    # Ensure the sequence ends with EOS exactly once so the model learns to stop.
    if tokenizer.eos_id is not None and tokens[-1] != tokenizer.eos_id:
        tokens.append(tokenizer.eos_id)

    # Drop (don't truncate) over-length samples: the input is tokens[:-1].
    if len(tokens) - 1 > seq_len:
        return None

    loss_mask = [0] * len(tokens)
    last_idx = len(messages) - 1
    for t, msg in enumerate(messages):
        if msg.get("role") != _ASSISTANT_ROLE:
            continue

        # start: length of the prefix ending in this turn's generation prompt
        # ("...assistant\n"), i.e. the first response token of turn t.
        prompt_prefix = tokenizer.apply_chat_template(
            messages[:t], add_generation_prompt=True
        )
        start = len(tokenizer.encode(prompt_prefix, add_bos=True, add_eos=False))

        # end: length through this assistant turn (content + its EOS). For the
        # final turn, extend to the very end so any appended EOS stays a
        # response token — this makes the single-turn case bit-identical to
        # ChatDataset's "mask only the prompt prefix" behavior.
        if t == last_idx:
            end = len(tokens)
        else:
            resp_prefix = tokenizer.apply_chat_template(messages[: t + 1]).rstrip("\n")
            end = len(tokenizer.encode(resp_prefix, add_bos=True, add_eos=False))

        start = min(start, len(loss_mask))
        end = min(end, len(loss_mask))
        for j in range(start, end):
            loss_mask[j] = 1

    return tokens, loss_mask
