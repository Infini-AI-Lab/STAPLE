# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the SFT (prompt-masked) path of the JSONL dataloader.

Covers the shared chat-tokenization helper, end-to-end mask alignment through
the packer, multi-turn masking, packing across a window boundary, that the
pretraining path is unchanged when ``sft=False``, and resume round-trips.
"""

import json
import os
import tempfile
import unittest

import torch
from datasets import Dataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.chat_tokenize import encode_chat_sample
from torchtitan.hf_datasets.jsonl_datasets import JSONLDataLoader
from torchtitan.hf_datasets.text_datasets import ChatDataset


_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
_TOKENIZER_PATH = os.path.join(_ASSETS_DIR, "tokenizer")


def _load_tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)


def _messages(i):
    # The test tokenizer is char-level, so keep content short: each sample is
    # ~15-20 tokens. Tests pick seq_len comfortably above this so samples are
    # packed (not dropped). A digit varies per sample for deterministic checks.
    d = i % 10
    return [
        {"role": "user", "content": f"q{d}"},
        {"role": "assistant", "content": f"a{d}"},
    ]


def _write_chunk(root, source, n_lines, make_messages=_messages):
    """Write a single JSONL chunk laid out as the loader expects:
    root/<source>/<source>.chunk.00.jsonl, one {"messages": [...]} per line."""
    src_dir = os.path.join(root, source)
    os.makedirs(src_dir, exist_ok=True)
    path = os.path.join(src_dir, f"{source}.chunk.00.jsonl")
    with open(path, "w") as f:
        for i in range(n_lines):
            f.write(json.dumps({"messages": make_messages(i)}) + "\n")
    return path


def _build_loader(root, *, sft, seq_len=32, batch_size=2, n_views=2, seed=7, **cfg_kw):
    cfg = JSONLDataLoader.Config(
        root_dir=root,
        sources={"sftsrc": 1.0},
        sft=sft,
        sample_processor=(lambda s: s["messages"]) if sft else None,
        seed=seed,
        n_views=n_views,
        prefetch_size=4,
        **cfg_kw,
    )
    return JSONLDataLoader(
        cfg,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=_load_tokenizer(),
        seq_len=seq_len,
        local_batch_size=batch_size,
    )


class TestEncodeChatSampleMatchesChatDataset(unittest.TestCase):
    """The refactored helper must reproduce ChatDataset's label masking exactly."""

    def test_single_turn_bit_identical(self):
        tokenizer = _load_tokenizer()
        for i in range(5):
            messages = _messages(i)
            tokens, loss_mask = encode_chat_sample(
                tokenizer, messages, seq_len=2048, validate=False
            )
            # Helper-derived (input_ids, label_ids).
            input_ids = tokens[:-1]
            label_ids = [
                t if m else IGNORE_INDEX for t, m in zip(tokens[1:], loss_mask[1:])
            ]

            # ChatDataset's own path on the same sample.
            ds = ChatDataset(
                dataset=Dataset.from_list([{"messages": messages}]),
                tokenizer=tokenizer,
                sample_processor=lambda s: s["messages"],
                seq_len=2048,
                infinite=False,
            )
            ref_input, ref_label = ds._tokenize_sample({"messages": messages})
            self.assertEqual(input_ids, ref_input)
            self.assertEqual(label_ids, ref_label)


class TestMaskAlignment(unittest.TestCase):
    def test_unmasked_label_equals_next_input(self):
        with tempfile.TemporaryDirectory() as root:
            _write_chunk(root, "sftsrc", 60)
            dl = _build_loader(root, sft=True, seq_len=32, batch_size=2)
            inp, labels = next(iter(dl))
            tokens = inp["input"]

            # Width carries the SFT mask column (n_views + 2 == 4 internally).
            self.assertEqual(tokens.shape, labels.shape)

            # Some prompt tokens masked, some response tokens unmasked.
            self.assertGreater(int((labels == IGNORE_INDEX).sum()), 0)
            self.assertGreater(int((labels != IGNORE_INDEX).sum()), 0)

            # For every unmasked label position i, label[i] must equal the next
            # input token (the teacher-forcing target), proving the mask column
            # is aligned with the label column.
            for b in range(tokens.shape[0]):
                t = tokens[b].tolist()
                lab = labels[b].tolist()
                for i in range(len(t) - 1):
                    if lab[i] != IGNORE_INDEX:
                        self.assertEqual(lab[i], t[i + 1])


class TestMultiTurnMasking(unittest.TestCase):
    def test_all_assistant_turns_unmasked(self):
        tokenizer = _load_tokenizer()
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "first question here"},
            {"role": "assistant", "content": "first answer response"},
            {"role": "user", "content": "second question here"},
            {"role": "assistant", "content": "second answer response"},
        ]
        tokens, loss_mask = encode_chat_sample(tokenizer, messages, seq_len=2048)
        self.assertEqual(len(tokens), len(loss_mask))
        # At least two disjoint unmasked spans (one per assistant turn).
        self.assertIn(1, loss_mask)
        self.assertIn(0, loss_mask)
        # The system/user prefix (first turn) is always masked.
        self.assertEqual(loss_mask[0], 0)
        # Count contiguous unmasked runs; expect >= 2 for two assistant turns.
        runs = 0
        prev = 0
        for m in loss_mask:
            if m == 1 and prev == 0:
                runs += 1
            prev = m
        self.assertGreaterEqual(runs, 2)


class TestPackingAcrossWindowBoundary(unittest.TestCase):
    def test_no_seam_artifact(self):
        # Small seq_len so a sample's response straddles packed windows.
        with tempfile.TemporaryDirectory() as root:
            _write_chunk(root, "sftsrc", 80)
            dl = _build_loader(root, sft=True, seq_len=8, batch_size=2)
            it = iter(dl)
            for _ in range(5):
                inp, labels = next(it)
                tokens = inp["input"]
                # Alignment must hold in every window (no off-by-one at seams).
                for b in range(tokens.shape[0]):
                    t = tokens[b].tolist()
                    lab = labels[b].tolist()
                    for i in range(len(t) - 1):
                        if lab[i] != IGNORE_INDEX:
                            self.assertEqual(lab[i], t[i + 1])


class TestPretrainingPathUnchanged(unittest.TestCase):
    def test_no_ignore_and_same_values_when_sft_false(self):
        # A pretraining JSONL line uses a "text" key.
        with tempfile.TemporaryDirectory() as root:
            src_dir = os.path.join(root, "sftsrc")
            os.makedirs(src_dir)
            with open(os.path.join(src_dir, "sftsrc.chunk.00.jsonl"), "w") as f:
                for i in range(100):
                    f.write(
                        json.dumps({"text": f"document number {i} body text"}) + "\n"
                    )

            dl = _build_loader(root, sft=False, seq_len=16, batch_size=2)
            inp, labels = next(iter(dl))
            # No masking in pretraining mode.
            self.assertEqual(int((labels == IGNORE_INDEX).sum()), 0)
            # input/labels are shifted views of the same packed buffer.
            tokens = inp["input"]
            self.assertEqual(tokens.shape, labels.shape)
            for b in range(tokens.shape[0]):
                t = tokens[b].tolist()
                lab = labels[b].tolist()
                for i in range(len(t) - 1):
                    self.assertEqual(lab[i], t[i + 1])


class TestResumeRoundTrip(unittest.TestCase):
    def test_resume_reproduces_subsequent_batches(self):
        with tempfile.TemporaryDirectory() as root:
            _write_chunk(root, "sftsrc", 200)

            dl1 = _build_loader(root, sft=True, seq_len=16, batch_size=2)
            it1 = iter(dl1)
            for _ in range(3):
                next(it1)
            state = dl1.state_dict()
            after = [next(it1) for _ in range(3)]

            dl2 = _build_loader(root, sft=True, seq_len=16, batch_size=2)
            dl2.load_state_dict(state)
            it2 = iter(dl2)
            after2 = [next(it2) for _ in range(3)]

            for (i1, l1), (i2, l2) in zip(after, after2):
                self.assertTrue(torch.equal(i1["input"], i2["input"]))
                self.assertTrue(torch.equal(l1, l2))

    def test_resume_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as root:
            _write_chunk(root, "sftsrc", 50)
            dl1 = _build_loader(root, sft=True, seq_len=16, batch_size=2)
            next(iter(dl1))
            state = dl1.state_dict()

            # Resuming as a pretraining (sft=False) loader must be rejected.
            dl2 = _build_loader(root, sft=False, seq_len=16, batch_size=2)
            with self.assertRaises(ValueError):
                dl2.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
