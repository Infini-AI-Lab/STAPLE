# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
JSONL-chunk dataloader adapted from STEM-pt/lingua/data.py.

Reads pre-shuffled JSONL chunk files, applies weighted multi-source sampling
with deficit-driven token tracking, packs tokens into fixed-length sequences,
and prefetches/shuffles batches. Supports deterministic resume via nested state
dicts and optional async loading via a producer process.

Expected data layout:
    root_dir/
        source_a/
            source_a.chunk.00.jsonl
            source_a.chunk.01.jsonl
            ...
        source_b/
            source_b.chunk.00.jsonl
            ...

Each JSONL line must have a "text" or "content" key.
"""

import contextlib
import json
import os
import pickle
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import tyro

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.chat_tokenize import encode_chat_sample
from torchtitan.tools.logging import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_DATA_FILE_PATTERN = "*.chunk.*.jsonl"


# ---------------------------------------------------------------------------
# Source-weight file loading
# ---------------------------------------------------------------------------


def load_source_weights(path: str) -> dict[str, float]:
    """Load a {source: weight} mapping from a YAML or JSON file.

    The file must contain a top-level mapping of source subdirectory name to a
    non-negative sampling weight, e.g.::

        cranecode: 20.0
        stack_edu: 15.0
        dolmino-math: 12.0

    Weights are relative; the dataloader normalizes them internally, so they
    need not sum to 1. ``.yaml``/``.yml`` files require PyYAML (imported lazily
    so it stays an optional dependency); ``.json`` uses the standard library.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    f"Reading source weights from {path} requires PyYAML "
                    "(pip install pyyaml), or use a .json file instead."
                ) from e
            loaded = yaml.safe_load(f)
        elif ext == ".json":
            loaded = json.load(f)
        else:
            raise ValueError(
                f"Unsupported sources_file extension '{ext}' for {path}; "
                "expected .yaml, .yml, or .json."
            )

    if not isinstance(loaded, dict):
        raise ValueError(
            f"sources_file {path} must contain a top-level mapping of "
            f"{{source: weight}}, got {type(loaded).__name__}."
        )

    weights: dict[str, float] = {}
    for source, weight in loaded.items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(
                f"sources_file {path}: weight for '{source}' must be a number, "
                f"got {weight!r}."
            )
        if weight < 0:
            raise ValueError(
                f"sources_file {path}: weight for '{source}' must be "
                f"non-negative, got {weight}."
            )
        weights[str(source)] = float(weight)

    if not weights:
        raise ValueError(f"sources_file {path} contains no sources.")
    return weights


# ---------------------------------------------------------------------------
# State TypedDicts (mirrors STEM-pt/lingua/data.py)
# ---------------------------------------------------------------------------
# Using plain dicts for state to stay serialization-friendly.


# ---------------------------------------------------------------------------
# JSONL iteration
# ---------------------------------------------------------------------------


def read_jsonl(
    file_path: str,
    position: int,
    block_size: int,
    offset: int,
    current_iter: int,
):
    if (offset < 0) or (offset >= block_size):
        raise RuntimeError("JSONL iterator offset value is invalid")
    current_line = offset + 1 if position > 0 else 0

    state = dict(
        file_path=file_path,
        position=position,
        block_size=block_size,
        offset=offset,
        current_iter=current_iter,
    )
    with open(file_path, "r") as file:
        file.seek(position)
        while line := file.readline():
            if not line.strip():
                continue
            current_line += 1
            if (current_line - 1) % block_size == offset:
                state = dict(
                    file_path=file_path,
                    position=file.tell(),
                    block_size=block_size,
                    offset=offset,
                    current_iter=current_iter,
                )
                yield json.loads(line), state


def loop_on_jsonl(
    file_path: str,
    position: int,
    block_size: int,
    offset: int,
    current_iter: int,
):
    try:
        while True:
            it = read_jsonl(file_path, position, block_size, offset, current_iter)
            for content, jsonl_state in it:
                yield content, jsonl_state
            current_iter += 1
            position = 0
    finally:
        it.close()


def loop_on_jsonl_chunks(
    chunk_states: list[dict],
    current_chunk_idx: int,
):
    assert len(chunk_states) > 0, "chunk_states must be non-empty"
    states = deepcopy(chunk_states)
    iters = [
        loop_on_jsonl(
            s["file_path"],
            s["position"],
            s["block_size"],
            s["offset"],
            s["current_iter"],
        )
        for s in states
    ]

    idx = current_chunk_idx % len(iters)
    try:
        while True:
            content, state = next(iters[idx])
            states[idx] = state
            next_idx = (idx + 1) % len(iters)
            yield content, dict(
                chunk_states=deepcopy(states),
                current_chunk_idx=next_idx,
            )
            idx = next_idx
    finally:
        for it in iters:
            it.close()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_tagged(
    iterator,
    tokenizer: BaseTokenizer,
    add_bos: bool,
    add_eos: bool,
):
    for content, doc_source, mc_state in iterator:
        assert (
            "text" in content or "content" in content
        ), "JSON line must contain either text or content key"
        content_key = "text" if ("text" in content) else "content"
        text = content[content_key]
        tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        yield tokens, doc_source, dict(
            it_state=mc_state,
            add_bos=add_bos,
            add_eos=add_eos,
        )


def tokenize_chat_tagged(
    iterator,
    tokenizer: BaseTokenizer,
    sample_processor: Callable,
    seq_len: int,
):
    """SFT counterpart of :func:`tokenize_tagged`.

    Converts each JSONL line to chat ``messages`` via ``sample_processor`` and
    emits ``(tokens, loss_mask, doc_source, state)`` where ``loss_mask[j] == 1``
    iff token ``j`` is an assistant-response token (see
    :func:`~torchtitan.hf_datasets.chat_tokenize.encode_chat_sample`). Samples
    whose input exceeds ``seq_len`` are dropped. bos/eos are owned by the chat
    template, so the pretraining ``add_bos``/``add_eos`` flags do not apply here.
    """
    for content, doc_source, mc_state in iterator:
        result = encode_chat_sample(
            tokenizer, sample_processor(content), seq_len=seq_len
        )
        if result is None:
            continue
        tokens, loss_mask = result
        yield tokens, loss_mask, doc_source, dict(it_state=mc_state, sft=True)


# ---------------------------------------------------------------------------
# Source sampling (deficit-driven)
# ---------------------------------------------------------------------------


def choose_source_tagged(
    source_to_iterator: dict[str, Iterator],
    source_to_state: dict[str, Any],
    root_dir: str,
    sources: dict[str, float],
    rng_state: dict[str, Any],
    source_counts: dict[str, int],
):
    n_sources = len(sources)
    possible_sources = list(sources.keys())
    target_fracs = np.array(list(sources.values()), dtype=np.float64)
    target_fracs = target_fracs / target_fracs.sum()
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    while True:
        emitted = np.array(
            [source_counts[s] for s in possible_sources], dtype=np.float64
        )
        total = emitted.sum()
        deficit = np.maximum(0.0, target_fracs * total - emitted)
        deficit_sum = deficit.sum()
        if deficit_sum > 0.0:
            weights = deficit / deficit_sum
        else:
            weights = target_fracs

        source_choice = possible_sources[rng.choice(n_sources, p=weights)]
        seq, state = next(source_to_iterator[source_choice])
        source_to_state = {**source_to_state, source_choice: state}
        multi_choice_state = dict(
            root_dir=root_dir,
            sources=sources,
            source_to_state=source_to_state,
            rng_state=rng.bit_generator.state,
        )
        yield seq, source_choice, multi_choice_state


# ---------------------------------------------------------------------------
# Token packing with source counting
# ---------------------------------------------------------------------------


def _get_empty_buffer_state(start_token, states):
    seq_to_resume_from = -1
    while start_token < 0:
        seq_to_resume_from -= 1
        start_token += states[seq_to_resume_from]["seq_len"]
    resume_state = deepcopy(states[seq_to_resume_from])
    resume_state["start_token"] = start_token
    del states[:seq_to_resume_from]
    if "seq_len" in resume_state:
        del resume_state["seq_len"]
    return resume_state


def _normalize_positions(positions: list[int]) -> list[int]:
    """Shift a leading partial document so the window starts at position 0.

    When a packed window begins in the middle of a document (because the
    document's earlier tokens filled a previous window), the first tokens carry
    nonzero within-document offsets. Subtracting that offset from the leading run
    -- up to the first genuine document start (a 0) -- keeps every position in
    ``[0, output_seq_len)`` and, crucially, avoids a spurious ``position == 0``
    appearing mid-window, which :func:`get_document_mask_mod` would otherwise
    misread as a false document boundary. Subsequent documents already begin at 0
    (the packer records offset 0 at each fresh document), so only the leading
    run is touched. Mirrors the convention in ``hf_datasets/text_datasets.py``.
    """
    offset = positions[0]
    if offset > 0:
        for i, p in enumerate(positions):
            if p == 0:
                break
            positions[i] = p - offset
    return positions


def pack_tokens_with_source_counts(
    iterator,
    empty_buffer_state: dict,
    source_counts: dict[str, int],
):
    buffer: list[int] = []
    # Within-document offset of each buffered token (0 at every document start),
    # kept in lockstep with ``buffer``. Emitted as an extra channel so flex /
    # varlen block_causal attention can mask across packed-document boundaries
    # and RoPE sees per-document positions. Derived purely from packing
    # structure (not token ids), so it is correct even when bos_id == eos_id.
    pos_buffer: list[int] = []
    # SFT only: per-token assistant-response mask, also kept in lockstep with
    # ``buffer``. Emitted as a second extra channel and consumed downstream to
    # set prompt-token labels to IGNORE_INDEX. Gated on ``sft`` so the
    # pretraining tensor width (n_views + 1) and numerics stay unchanged.
    sft = empty_buffer_state.get("sft", False)
    mask_buffer: list[int] = []
    states: list[dict] = []
    output_seq_len = empty_buffer_state["output_seq_len"]
    n_views = empty_buffer_state["n_views"]
    start_token = empty_buffer_state["start_token"]
    previous_state = empty_buffer_state["it_state"]
    buffer_size = output_seq_len + n_views - 1
    for _i, item in enumerate(iterator):
        if sft:
            tokens, loss_mask, doc_source, state = item
        else:
            tokens, doc_source, state = item
            loss_mask = None
        if _i == 0 and start_token >= len(tokens):
            logger.warning(
                "Resume desync in pack_tokens_with_source_counts: "
                f"saved start_token={start_token} >= len(first_doc)={len(tokens)}. "
                "Discarding stale intra-doc offset and packing from token 0."
            )
            start_token = 0
            previous_state = state
        sample_is_read = False
        while not sample_is_read:
            assert start_token < len(
                tokens
            ), f"Start token index {start_token} bigger than sequence {len(tokens)}"
            free_space = buffer_size - len(buffer)
            seq_len = min(free_space, len(tokens) - start_token)
            end_token = start_token + seq_len
            buffer.extend(tokens[start_token:end_token])
            # Tokens [start_token:end_token] are the (start_token .. end_token-1)
            # th tokens of their document, so their within-document offsets are
            # exactly that range.
            pos_buffer.extend(range(start_token, end_token))
            if sft:
                mask_buffer.extend(loss_mask[start_token:end_token])
            start_token = end_token

            source_counts[doc_source] += seq_len

            states.append(
                dict(
                    start_token=start_token,
                    seq_len=seq_len,
                    it_state=previous_state,
                    output_seq_len=output_seq_len,
                    n_views=n_views,
                    sft=sft,
                )
            )
            assert len(buffer) <= buffer_size, "Buffer overflow"

            if len(buffer) == buffer_size:
                out = np.array(buffer)
                assert out.ndim == 1
                out = np.lib.stride_tricks.sliding_window_view(out, n_views, axis=0)

                # Per-input-token positions (the input column is buffer[:output_seq_len]),
                # normalized so a leading partial document starts at 0. Appended as
                # one extra column on `out` so the downstream row shuffle carries it
                # with its sequence without drawing from the prefetch RNG.
                positions = np.array(
                    _normalize_positions(list(pos_buffer[:output_seq_len])),
                    dtype=out.dtype,
                ).reshape(output_seq_len, 1)
                if sft:
                    # Label-aligned response mask: take the same sliding window
                    # as the tokens and keep column n_views-1, so the mask at
                    # output row i corresponds to the target token (buffer[i +
                    # n_views - 1]) — the prediction target loss is computed on.
                    # NOTE: unlike positions (which index the input column and
                    # use buffer[:output_seq_len]), the mask indexes the label
                    # column, so it must use the window, not a leading slice.
                    mask_view = np.lib.stride_tricks.sliding_window_view(
                        np.array(mask_buffer), n_views, axis=0
                    )[:, n_views - 1].reshape(output_seq_len, 1)
                    out = np.concatenate([out, positions, mask_view], axis=1)
                else:
                    out = np.concatenate([out, positions], axis=1)

                rewinded_idx = start_token - (n_views - 1)
                empty_buffer_state = _get_empty_buffer_state(rewinded_idx, states)
                buffer = buffer[output_seq_len:]
                pos_buffer = pos_buffer[output_seq_len:]
                if sft:
                    mask_buffer = mask_buffer[output_seq_len:]
                assert len(buffer) == (n_views - 1)

                yield out, empty_buffer_state

            if start_token == len(tokens):
                start_token = 0
                sample_is_read = True
                previous_state = state


# ---------------------------------------------------------------------------
# Prefetch and batch shuffle
# ---------------------------------------------------------------------------


def batch_and_shuffle_prefetched_sequences(
    data_loader,
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    state: dict,
    sft: bool = False,
):
    # Last dim is n_views token columns + 1 trailing per-token positions column,
    # plus (SFT only) 1 more for the per-token response mask — all appended by
    # pack_tokens_with_source_counts. The row shuffle below keeps each
    # sequence's extra columns aligned with its tokens.
    last_dim = n_views + (2 if sft else 1)
    prefetch_buffer = -1 * np.ones(
        (prefetch_size * batch_size, seq_len, last_dim), dtype=int
    )
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]

    seq_idx = state["seq_idx"]
    assert (
        0 <= seq_idx < prefetch_size
    ), "Prefetch state seq_idx should be in 0 <= seq_idx < prefetch_size."

    _rng_state = state["rng_state"]
    _it_state = state["it_state"]

    for i in range(prefetch_size * batch_size):
        prefetch_buffer[i], next_it_state = next(data_loader)
    rng.shuffle(prefetch_buffer, axis=0)
    for i in range(seq_idx * batch_size):
        prefetch_buffer[i], _ = next(data_loader)

    idx = seq_idx
    while True:
        if idx == prefetch_size - 1:
            _it_state = next_it_state
            _rng_state = rng.bit_generator.state

        current_state = dict(
            it_state=_it_state,
            seq_idx=(idx + 1) % prefetch_size,
            rng_state=_rng_state,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
        )

        yield prefetch_buffer[
            idx * batch_size : (idx + 1) * batch_size
        ].copy(), current_state

        for i in range(batch_size):
            prefetch_buffer[idx * batch_size + i], pack_state = next(data_loader)

        if idx == prefetch_size - 1:
            next_it_state = pack_state
            rng.shuffle(prefetch_buffer, axis=0)

        idx = (idx + 1) % prefetch_size


# ---------------------------------------------------------------------------
# Data distribution
# ---------------------------------------------------------------------------


def find_and_sanitize_chunks(dataset_path: str, world_size: int, file_pattern: str):
    dataset_chunks = sorted(str(p) for p in Path(dataset_path).glob(file_pattern))
    n_chunks = len(dataset_chunks)
    if n_chunks <= world_size:
        assert (
            world_size % n_chunks == 0
        ), "World size should be a multiple of number of chunks"
    assert n_chunks > 0, f"No valid chunks in {dataset_path}"
    return dataset_chunks


def distribute_data_to_rank(
    dataset_path: str, rank: int, world_size: int, file_pattern: str
):
    dataset_chunks = find_and_sanitize_chunks(dataset_path, world_size, file_pattern)
    n_chunks = len(dataset_chunks)

    if n_chunks > world_size:
        assigned_chunks = dataset_chunks[rank::world_size]
        assert len(assigned_chunks) > 0
        return dict(
            chunk_states=[
                dict(
                    file_path=chunk_path,
                    position=0,
                    block_size=1,
                    offset=0,
                    current_iter=0,
                )
                for chunk_path in assigned_chunks
            ],
            current_chunk_idx=0,
        )

    n_ranks_per_chunk = world_size // n_chunks
    rank_to_params = []
    for chunk_path in dataset_chunks:
        for i in range(n_ranks_per_chunk):
            rank_to_params.append(
                dict(
                    file_path=chunk_path,
                    position=0,
                    block_size=n_ranks_per_chunk,
                    offset=i,
                    current_iter=0,
                )
            )
    return rank_to_params[rank]


# ---------------------------------------------------------------------------
# State initialization
# ---------------------------------------------------------------------------


def _init_choice_state(
    root_dir: str,
    sources: dict[str, float],
    seed: int,
    rank: int,
    world_size: int,
    file_pattern: str,
):
    data_path_to_jsonl_state = {}
    for dataset_path in sources:
        jsonl_state = distribute_data_to_rank(
            os.path.join(root_dir, dataset_path), rank, world_size, file_pattern
        )
        data_path_to_jsonl_state[dataset_path] = jsonl_state

    multi_rng_state = np.random.default_rng(
        (seed, rank, world_size)
    ).bit_generator.state

    return dict(
        root_dir=root_dir,
        sources=sources,
        source_to_state=data_path_to_jsonl_state,
        rng_state=multi_rng_state,
    )


def _multi_choice_state(prefetch_state: dict) -> dict:
    """Navigate a prefetch_state down to its (innermost) multi-source choice
    state, which holds ``sources`` (the mix) and ``source_to_state`` (per-source
    read positions). Nesting is: prefetch -> pack -> tokenizer -> multi_choice.
    """
    return prefetch_state["it_state"]["it_state"]["it_state"]


def _init_prefetch_state(
    root_dir: str,
    sources: dict[str, float],
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    seed: int,
    rank: int,
    world_size: int,
    add_bos: bool,
    add_eos: bool,
    file_pattern: str,
    sft: bool = False,
):
    multi_choice_state = _init_choice_state(
        root_dir=root_dir,
        sources=sources,
        seed=seed,
        rank=rank,
        world_size=world_size,
        file_pattern=file_pattern,
    )
    tokenizer_state = dict(
        it_state=multi_choice_state,
        add_bos=add_bos,
        add_eos=add_eos,
        sft=sft,
    )
    pack_state = dict(
        start_token=0,
        it_state=tokenizer_state,
        output_seq_len=seq_len,
        n_views=n_views,
        seq_len=0,
        sft=sft,
    )

    prefetch_rng_state = np.random.default_rng(
        (seed + 1, rank, world_size)
    ).bit_generator.state

    return dict(
        it_state=pack_state,
        seq_idx=0,
        rng_state=prefetch_rng_state,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    )


# ---------------------------------------------------------------------------
# Dataloader builder (context manager)
# ---------------------------------------------------------------------------


def _setup_sources(multi_state):
    path_to_iter = {}
    for source in multi_state["sources"]:
        jsonl_state = multi_state["source_to_state"][source]
        if "chunk_states" in jsonl_state:
            path_to_iter[source] = loop_on_jsonl_chunks(
                jsonl_state["chunk_states"],
                jsonl_state["current_chunk_idx"],
            )
        else:
            path_to_iter[source] = loop_on_jsonl(
                jsonl_state["file_path"],
                jsonl_state["position"],
                jsonl_state["block_size"],
                jsonl_state["offset"],
                jsonl_state["current_iter"],
            )
    return path_to_iter


@contextlib.contextmanager
def _build_pipeline(
    state: dict,
    source_counts: dict[str, int],
    tokenizer: BaseTokenizer,
    sample_processor: Callable | None = None,
):
    pack_state = state["it_state"]
    tokenizer_state = pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]
    sft = tokenizer_state.get("sft", False)

    path_to_iter = _setup_sources(multi_state)

    data_it = choose_source_tagged(
        source_to_iterator=path_to_iter,
        source_to_state=multi_state["source_to_state"],
        root_dir=multi_state["root_dir"],
        sources=multi_state["sources"],
        rng_state=multi_state["rng_state"],
        source_counts=source_counts,
    )
    if sft:
        assert sample_processor is not None, "SFT pipeline requires a sample_processor"
        data_it = tokenize_chat_tagged(
            data_it,
            tokenizer,
            sample_processor,
            pack_state["output_seq_len"],
        )
    else:
        data_it = tokenize_tagged(
            data_it,
            tokenizer,
            tokenizer_state["add_bos"],
            tokenizer_state["add_eos"],
        )
    data_it = pack_tokens_with_source_counts(
        data_it,
        pack_state,
        source_counts,
    )
    data_it = batch_and_shuffle_prefetched_sequences(
        data_loader=data_it,
        seq_len=pack_state["output_seq_len"],
        n_views=pack_state["n_views"],
        batch_size=state["batch_size"],
        prefetch_size=state["prefetch_size"],
        state=state,
        sft=sft,
    )
    yield data_it
    for it in path_to_iter.values():
        it.close()
    data_it.close()


# ---------------------------------------------------------------------------
# TorchTitan-compatible DataLoader
# ---------------------------------------------------------------------------


class JSONLDataLoader(BaseDataLoader):
    """DataLoader that reads pre-shuffled JSONL chunks with deficit-driven
    multi-source mixing, token packing, and async prefetching.

    Adapted from STEM-pt/lingua/data.py to match torchtitan's BaseDataLoader
    interface. Yields ({"input": Tensor, "positions": Tensor}, labels_Tensor)
    per step.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        root_dir: str = ""
        """Root directory containing source subdirectories with JSONL chunks."""

        sources: dict[str, float] = field(default_factory=dict)
        """Mapping from subdirectory name to sampling weight.

        Ignored when ``sources_file`` is set (a warning is emitted in that case).
        """

        sources_file: str = ""
        """Optional path to a YAML/JSON file with a {source: weight} mapping.

        When set, it fully overrides the ``sources`` field above, letting you
        swap source ratios without editing config_registry code, e.g.
        ``--dataloader.sources_file=mixes/mathcode_heavy.yaml``. See
        :func:`load_source_weights` for the expected file format.
        """

        file_pattern: str = TRAIN_DATA_FILE_PATTERN
        """Glob pattern for discovering chunk files."""

        seed: int = 42
        """Random seed for reproducibility."""

        add_bos: bool = True
        """Add beginning-of-sequence token."""

        add_eos: bool = True
        """Add end-of-sequence token."""

        n_views: int = 2
        """Number of shifted views (2 means input+target)."""

        prefetch_size: int = 64
        """Number of batches to prefetch and shuffle."""

        sft: bool = False
        """Enable supervised fine-tuning: chat-template tokenization with
        prompt-token loss masking (prompt labels set to IGNORE_INDEX). Requires
        ``sample_processor`` and a tokenizer with a chat template. When True,
        ``add_bos``/``add_eos`` are owned by the chat template and ignored."""

        sample_processor: Annotated[Callable | None, tyro.conf.Suppress] = None
        """Callable(line_dict) -> list[message_dict] for SFT. Required when
        ``sft=True``; set in config functions (e.g. ``lambda s: s["messages"]``)."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps=None,
        **kwargs,
    ):
        self._config = config
        self._tokenizer = tokenizer
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._seq_len = seq_len
        self._batch_size = local_batch_size
        self._rank_id = f"dp_rank_{dp_rank}"

        # A sources_file, when provided, fully overrides the registry's
        # `sources` dict so ratios can be swapped without editing config code.
        if config.sources_file:
            sources = load_source_weights(config.sources_file)
            if config.sources:
                logger.warning(
                    "JSONLDataLoader: both `sources` and `sources_file` are set; "
                    "`sources_file` (%s) takes precedence and `sources` is ignored.",
                    config.sources_file,
                )
            logger.info(
                "JSONLDataLoader: loaded %d source weights from %s",
                len(sources),
                config.sources_file,
            )
        else:
            sources = config.sources
        self._sources = sources

        self._source_counts: dict[str, int] = {k: 0 for k in sorted(sources.keys())}

        if config.sft:
            if config.sample_processor is None:
                raise ValueError(
                    "JSONLDataLoader: sft=True requires a sample_processor "
                    "(line_dict -> list of chat messages)."
                )
            if config.add_bos or config.add_eos:
                logger.warning(
                    "JSONLDataLoader: add_bos/add_eos are ignored when sft=True; "
                    "the chat template owns bos/eos."
                )

        self._state = _init_prefetch_state(
            root_dir=config.root_dir,
            sources=sources,
            batch_size=local_batch_size,
            prefetch_size=config.prefetch_size,
            seq_len=seq_len,
            n_views=config.n_views,
            seed=config.seed,
            rank=dp_rank,
            world_size=dp_world_size,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
            file_pattern=config.file_pattern,
            sft=config.sft,
        )

        self._context = None
        self._iterator = None

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        with _build_pipeline(
            self._state,
            self._source_counts,
            self._tokenizer,
            self._config.sample_processor,
        ) as raw_iter:
            for batch_np, prefetch_state in raw_iter:
                self._state = prefetch_state

                # batch_np shape: (batch_size, seq_len, n_views + 1) for
                # pretraining, or (..., n_views + 2) for SFT. Columns
                # 0..n_views-1 are the shifted token views (for n_views=2:
                # column 0 is input, column 1 is target); column n_views is
                # per-token positions that reset to 0 at each packed document
                # boundary; the SFT-only column n_views+1 is the label-aligned
                # assistant-response mask (see pack_tokens_with_source_counts).
                batch_tensor = torch.from_numpy(batch_np.astype(np.int64))

                n_views = self._config.n_views
                inputs = batch_tensor[:, :, 0]  # (batch_size, seq_len)
                labels = batch_tensor[:, :, n_views - 1]  # (batch_size, seq_len)
                positions = batch_tensor[:, :, n_views].to(
                    torch.int32
                )  # (batch_size, seq_len)

                if self._config.sft:
                    # Mask prompt-token labels so loss is computed only on
                    # assistant-response tokens (trainer counts valid tokens as
                    # labels != IGNORE_INDEX and normalizes loss accordingly).
                    loss_mask = batch_tensor[:, :, n_views + 1]
                    labels = labels.clone()
                    labels[loss_mask == 0] = IGNORE_INDEX

                yield {"input": inputs, "positions": positions}, labels

    def state_dict(self) -> dict[str, Any]:
        return {
            self._rank_id: pickle.dumps(
                {
                    "prefetch_state": self._state,
                    "source_counts": self._source_counts,
                }
            ),
            "world_size": self._dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if self._rank_id not in state_dict:
            logger.warning(
                f"JSONLDataLoader state is empty for dp rank {self._dp_rank}, "
                f"expected key {self._rank_id}"
            )
            return
        assert self._dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        loaded = pickle.loads(state_dict[self._rank_id])
        loaded_state = loaded["prefetch_state"]
        loaded_counts = loaded.get("source_counts", self._source_counts)

        # The prefetch buffer's last-dim width is (n_views + 1), or (n_views + 2)
        # in SFT mode. It is reallocated from the current config on each __iter__,
        # so resuming with a different n_views / sft than the checkpoint would
        # silently mismatch the saved per-sequence arrays. Guard explicitly.
        saved_pack = loaded_state["it_state"]
        saved_n_views = saved_pack.get("n_views", self._config.n_views)
        saved_sft = saved_pack.get("sft", False)
        if saved_n_views != self._config.n_views or saved_sft != self._config.sft:
            raise ValueError(
                "JSONLDataLoader: checkpoint was written with "
                f"n_views={saved_n_views}, sft={saved_sft} but the current config "
                f"has n_views={self._config.n_views}, sft={self._config.sft}; "
                "resuming across a packing-shape change is not supported."
            )

        saved_multi = _multi_choice_state(loaded_state)
        saved_sources = saved_multi["sources"]

        if saved_sources == self._sources:
            # Same mix: ordinary resume (within-stage / preemption). Restore the
            # full saved state and the deficit-mixer counts verbatim.
            self._state = loaded_state
            self._source_counts = loaded_counts
            return

        # Mix changed (a continual-learning stage boundary). Carry per-source
        # read positions so overlapping sources don't re-read documents the
        # model already saw, but switch to the NEW mix and RESET the
        # deficit-driven token counts so the new ratios apply to this stage's
        # tokens only (not to the cumulative history of prior stages).
        logger.info(
            "JSONLDataLoader: data mix changed at resume "
            "(%d -> %d sources); carrying read positions for overlapping "
            "sources and resetting source token counts for the new mix.",
            len(saved_sources),
            len(self._sources),
        )

        # self._state was freshly built in __init__ from the new mix, so it
        # already has fresh read positions for every new source. Overwrite those
        # with the saved positions wherever a source carries over.
        new_multi = _multi_choice_state(self._state)
        saved_positions = saved_multi["source_to_state"]
        carried = 0
        for source in new_multi["source_to_state"]:
            if source in saved_positions:
                new_multi["source_to_state"][source] = deepcopy(saved_positions[source])
                carried += 1
        logger.info(
            "JSONLDataLoader: carried read positions for %d/%d sources "
            "(%d new sources start fresh).",
            carried,
            len(self._sources),
            len(self._sources) - carried,
        )

        # Fresh deficit-mixer counts for exactly the new source set.
        self._source_counts = {k: 0 for k in sorted(self._sources.keys())}
