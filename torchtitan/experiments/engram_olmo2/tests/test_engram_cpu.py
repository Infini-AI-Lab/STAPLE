# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the Engram building blocks (no distributed required)."""

import unittest

import numpy as np
import torch

from torchtitan.experiments.engram_olmo2.hashing import (
    compute_head_table_sizes,
    compute_layer_multipliers,
    EngramHasher,
    ngram_hashes_numpy,
)
from torchtitan.experiments.engram_olmo2.multi_head_embedding import MultiHeadEmbedding
from torchtitan.experiments.engram_olmo2.short_conv import ShortConv

MAX_NGRAM = 3
N_HEAD = 8
LAYER_IDS = [2, 7]
VOCAB = 4096
PAD_ID = 0


def _build_hasher(layer_id: int) -> tuple[EngramHasher, np.ndarray, list[list[int]]]:
    sizes_by_layer = compute_head_table_sizes(
        engram_vocab_size=VOCAB,
        max_ngram_size=MAX_NGRAM,
        n_head_per_ngram=N_HEAD,
        layer_ids=LAYER_IDS,
    )
    head_sizes = sizes_by_layer[layer_id]
    multipliers = compute_layer_multipliers(
        layer_id=layer_id,
        max_ngram_size=MAX_NGRAM,
        tokenizer_vocab_size=VOCAB,
        seed=0,
    )
    # Identity compression for the test.
    lookup = torch.arange(VOCAB, dtype=torch.int64)
    hasher = EngramHasher(
        compression_lookup=lookup,
        multipliers=torch.from_numpy(multipliers),
        head_table_sizes=head_sizes,
        max_ngram_size=MAX_NGRAM,
        n_head_per_ngram=N_HEAD,
        pad_id=PAD_ID,
    )
    return hasher, multipliers, head_sizes


class TestHashing(unittest.TestCase):
    def test_primes_distinct_and_at_least_vocab(self):
        sizes_by_layer = compute_head_table_sizes(VOCAB, MAX_NGRAM, N_HEAD, LAYER_IDS)
        all_primes = [
            s
            for per_layer in sizes_by_layer.values()
            for per_ngram in per_layer
            for s in per_ngram
        ]
        # Globally distinct.
        self.assertEqual(len(all_primes), len(set(all_primes)))
        # Each is >= base vocab size.
        self.assertTrue(all(p >= VOCAB - 1 for p in all_primes))

    def test_tensor_matches_numpy_reference(self):
        hasher, multipliers, head_sizes = _build_hasher(layer_id=2)
        torch.manual_seed(0)
        tokens = torch.randint(0, VOCAB, (3, 17), dtype=torch.int64)

        got = hasher(tokens).numpy()
        expected = ngram_hashes_numpy(
            tokens.numpy(),
            multipliers,
            head_sizes,
            MAX_NGRAM,
            N_HEAD,
            PAD_ID,
        )
        np.testing.assert_array_equal(got, expected)

    def test_layers_hash_differently(self):
        h2, _, _ = _build_hasher(layer_id=2)
        h7, _, _ = _build_hasher(layer_id=7)
        tokens = torch.randint(0, VOCAB, (2, 12), dtype=torch.int64)
        self.assertFalse(torch.equal(h2(tokens), h7(tokens)))

    def test_hash_indices_in_range(self):
        hasher, _, _ = _build_hasher(layer_id=2)
        tokens = torch.randint(0, VOCAB, (4, 20), dtype=torch.int64)
        hashes = hasher(tokens)
        sizes = torch.tensor(hasher.head_mods)  # [num_heads]
        self.assertTrue((hashes >= 0).all())
        self.assertTrue((hashes < sizes).all())

    def test_arange_positions_match_legacy(self):
        """positions == arange (a single document) must reproduce the legacy
        global-left-pad hashing exactly (no regression for plain-causal)."""
        hasher, _, _ = _build_hasher(layer_id=2)
        tokens = torch.randint(0, VOCAB, (3, 19), dtype=torch.int64)
        positions = torch.arange(19).expand(3, 19).contiguous()
        self.assertTrue(torch.equal(hasher(tokens), hasher(tokens, positions)))

    def test_numpy_reference_matches_with_positions(self):
        hasher, multipliers, head_sizes = _build_hasher(layer_id=2)
        torch.manual_seed(1)
        # Two packed docs per row, positions reset at the boundary.
        tokens = torch.randint(0, VOCAB, (2, 7), dtype=torch.int64)
        positions = torch.tensor(
            [[0, 1, 2, 0, 1, 2, 3], [0, 1, 0, 1, 2, 3, 4]], dtype=torch.int64
        )
        got = hasher(tokens, positions).numpy()
        expected = ngram_hashes_numpy(
            tokens.numpy(),
            multipliers,
            head_sizes,
            MAX_NGRAM,
            N_HEAD,
            PAD_ID,
            positions=positions.numpy(),
        )
        np.testing.assert_array_equal(got, expected)

    def test_packed_hashing_equals_per_document(self):
        """Hashing a packed row with per-doc positions, sliced per document,
        equals hashing each document standalone -- no cross-document leakage."""
        hasher, _, _ = _build_hasher(layer_id=2)
        torch.manual_seed(2)
        doc_a = torch.randint(0, VOCAB, (1, 3), dtype=torch.int64)
        doc_b = torch.randint(0, VOCAB, (1, 4), dtype=torch.int64)
        packed = torch.cat([doc_a, doc_b], dim=1)  # [1, 7]
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], dtype=torch.int64)

        packed_hashes = hasher(packed, positions)

        # Standalone hashing (positions=None) pads the global left edge, which
        # for a lone document is exactly its document-boundary behavior.
        hashes_a = hasher(doc_a)
        hashes_b = hasher(doc_b)
        self.assertTrue(torch.equal(packed_hashes[:, :3], hashes_a))
        self.assertTrue(torch.equal(packed_hashes[:, 3:], hashes_b))


class TestShortConv(unittest.TestCase):
    def test_zero_init_is_identity(self):
        conv = ShortConv.Config(dim=32, kernel_size=4, dilation=MAX_NGRAM).build()
        conv.init_states(buffer_device=torch.device("cpu"))
        x = torch.randn(2, 16, 32)
        # Conv weight zero-init -> conv output is zero, so the module returns 0.
        self.assertTrue(torch.allclose(conv(x), torch.zeros_like(x), atol=1e-6))

    def test_causal(self):
        conv = ShortConv.Config(dim=8, kernel_size=4, dilation=MAX_NGRAM).build()
        conv.init_states(buffer_device=torch.device("cpu"))
        # Make the conv non-trivial.
        torch.nn.init.normal_(conv.conv.weight)
        x = torch.randn(1, 32, 8)
        y_full = conv(x)
        # Perturbing a future position must not change an earlier output.
        x2 = x.clone()
        x2[:, 20:, :] += 5.0
        y2 = conv(x2)
        self.assertTrue(torch.allclose(y_full[:, :17], y2[:, :17], atol=1e-5))

    def test_arange_positions_match_plain_conv(self):
        """With positions == arange (single doc), the masked-conv path must
        reproduce the fast nn.Conv1d path bit-for-bit (no regression)."""
        conv = ShortConv.Config(dim=8, kernel_size=4, dilation=MAX_NGRAM).build()
        conv.init_states(buffer_device=torch.device("cpu"))
        torch.nn.init.normal_(conv.conv.weight)
        x = torch.randn(2, 16, 8)
        positions = torch.arange(16).expand(2, 16).contiguous()
        self.assertTrue(torch.allclose(conv(x), conv(x, positions), atol=1e-6))

    def test_packed_conv_equals_per_document(self):
        """A packed sequence convolved with per-doc positions, sliced per
        document, equals each document convolved standalone."""
        conv = ShortConv.Config(dim=8, kernel_size=4, dilation=MAX_NGRAM).build()
        conv.init_states(buffer_device=torch.device("cpu"))
        torch.nn.init.normal_(conv.conv.weight)

        doc_a = torch.randn(1, 5, 8)
        doc_b = torch.randn(1, 11, 8)
        packed = torch.cat([doc_a, doc_b], dim=1)  # [1, 16, 8]
        positions = torch.cat([torch.arange(5), torch.arange(11)]).unsqueeze(
            0
        )  # [1, 16]

        packed_out = conv(packed, positions)
        # Standalone (positions=None) uses the plain conv, which left-pads each
        # document's own start -- the per-document ground truth.
        out_a = conv(doc_a)
        out_b = conv(doc_b)
        self.assertTrue(torch.allclose(packed_out[:, :5], out_a, atol=1e-5))
        self.assertTrue(torch.allclose(packed_out[:, 5:], out_b, atol=1e-5))

    def test_decode_step_matches_full_conv(self):
        """One-token-at-a-time decode_step (seeded from a prompt, rolling the
        ṽ history) reproduces the full causal short conv over the whole doc --
        the KV-cached-decode equivalence the conv must satisfy."""
        conv = ShortConv.Config(dim=8, kernel_size=4, dilation=MAX_NGRAM).build()
        conv.init_states(buffer_device=torch.device("cpu"))
        torch.nn.init.normal_(conv.conv.weight)

        L_prompt, L_gen = 6, 8
        v_full = torch.randn(1, L_prompt + L_gen, 8)
        # Ground truth: full conv contribution (the short_conv output minus the
        # residual ṽ that EngramLayer.forward adds; here forward isn't called).
        gt = conv(v_full, torch.arange(L_prompt + L_gen).unsqueeze(0))

        P = conv.history_len
        v_cache = v_full.new_zeros(1, P, 8)
        take = min(P, L_prompt)
        v_cache[0, P - take :] = v_full[0, L_prompt - take : L_prompt]

        for t in range(L_prompt, L_prompt + L_gen):
            out_t = conv.decode_step(v_full[:, t, :], v_cache, torch.tensor([t]))
            # decode_step returns the conv contribution; gt also has the conv
            # contribution only (forward()'s residual add is separate).
            self.assertTrue(torch.allclose(out_t, gt[:, t, :], atol=1e-5))
            v_cache = torch.cat([v_cache[:, 1:], v_full[:, t, :].unsqueeze(1)], dim=1)


class TestMultiHeadEmbedding(unittest.TestCase):
    def test_offsets_reinitialized_after_meta_to_empty(self):
        head_sizes = [101, 103, 107]
        with torch.device("meta"):
            emb = MultiHeadEmbedding.Config(
                head_table_sizes=head_sizes, per_head_dim=5
            ).build()

        emb.to_empty(device=torch.device("cpu"))
        emb.init_states(buffer_device=torch.device("cpu"))

        expected = torch.tensor([0, 101, 101 + 103])
        self.assertTrue(torch.equal(emb.offsets, expected))
        self.assertNotIn("offsets", emb.state_dict())

    def test_replicated_gather(self):
        head_sizes = [101, 103, 107]
        emb = MultiHeadEmbedding.Config(
            head_table_sizes=head_sizes, per_head_dim=5
        ).build()
        emb.init_states(buffer_device=torch.device("cpu"))

        hash_ids = torch.tensor([[[0, 1, 2], [100, 102, 106]]], dtype=torch.int64)
        out = emb(hash_ids)
        self.assertEqual(tuple(out.shape), (1, 2, 3, 5))

        # Manually replicate the offset gather.
        offsets = torch.tensor([0, 101, 101 + 103])
        global_ids = hash_ids + offsets
        expected = torch.nn.functional.embedding(global_ids, emb.weight)
        self.assertTrue(torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
