# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenizer compression for Engram N-gram lookup.

Standard subword tokenizers assign disjoint ids to semantically equivalent
surface forms (e.g. ``Apple`` vs. ``▁apple``). Engram collapses raw token
ids into canonical ids via a surjective map ``P: V -> V'`` built from a text
normalizer (NFKC / NFD / strip-accents / lowercase / whitespace-collapse),
maximizing the semantic density of each N-gram key (paper Section 2.2).

The lookup table is expensive to build (it decodes every id in the vocab), so
it is computed **once offline** and cached to disk. ``transformers`` and
``tokenizers`` are only needed for this offline step, not at training time:
the cached table loads as a plain int64 tensor.

CLI::

    python -m torchtitan.experiments.engram_olmo2.tokenizer_compression \\
        --tokenizer deepseek-ai/DeepSeek-V3 --out compression_lookup.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import torch

__all__ = [
    "build_compression_table",
    "load_compression_table",
]


def _build_normalizer():
    """Construct the text normalizer used to canonicalize token surface forms.

    Imported lazily so ``tokenizers`` is only required for the offline build.
    """
    from tokenizers import normalizers, Regex

    # Sentinel keeps a single space from being stripped, then restored to a
    # space, so a lone-space token does not collapse into the empty string.
    sentinel = ""
    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )


def build_compression_table(
    tokenizer_name_or_path: str,
) -> tuple[np.ndarray, int]:
    """Build the raw-id -> canonical-id lookup table.

    Returns ``(lookup, num_canonical)`` where ``lookup`` is an int64 array of
    length ``raw_vocab_size`` mapping each raw token id to its canonical id in
    ``[0, num_canonical)``. Tokens that do not decode cleanly (contain the
    Unicode replacement char) are keyed by their raw token string so distinct
    byte-fallback tokens stay distinct.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=True
    )
    normalizer = _build_normalizer()

    key_to_canonical: dict[str, int] = {}
    raw_vocab_size = len(tokenizer)
    lookup = np.empty(raw_vocab_size, dtype=np.int64)

    for raw_id in range(raw_vocab_size):
        text = tokenizer.decode([raw_id], skip_special_tokens=False)
        if "�" in text:
            # Undecodable (byte-fallback) token: key by the raw token string.
            key = tokenizer.convert_ids_to_tokens(raw_id)
        else:
            norm = normalizer.normalize_str(text)
            key = norm if norm else text

        canonical = key_to_canonical.get(key)
        if canonical is None:
            canonical = len(key_to_canonical)
            key_to_canonical[key] = canonical
        lookup[raw_id] = canonical

    return lookup, len(key_to_canonical)


def _cache_metadata(tokenizer_name_or_path: str) -> dict:
    digest = hashlib.sha256(tokenizer_name_or_path.encode("utf-8")).hexdigest()[:16]
    return {"tokenizer": tokenizer_name_or_path, "tokenizer_hash": digest}


def save_compression_table(
    tokenizer_name_or_path: str,
    out_path: str,
) -> None:
    """Build and serialize the compression table to ``out_path``."""
    lookup, num_canonical = build_compression_table(tokenizer_name_or_path)
    payload = {
        "lookup": torch.from_numpy(lookup),
        "num_canonical": int(num_canonical),
        **_cache_metadata(tokenizer_name_or_path),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)


def load_compression_table(path: str) -> tuple[torch.Tensor, int]:
    """Load a cached compression table.

    Returns ``(lookup, num_canonical)`` with ``lookup`` an int64 ``torch.Tensor``.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    lookup = payload["lookup"]
    if not isinstance(lookup, torch.Tensor):
        lookup = torch.as_tensor(lookup, dtype=torch.int64)
    return lookup.to(torch.int64), int(payload["num_canonical"])


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer name or path (e.g. deepseek-ai/DeepSeek-V3).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the cached compression table (.pt).",
    )
    args = parser.parse_args()
    save_compression_table(args.tokenizer, args.out)
    lookup, num_canonical = load_compression_table(args.out)
    reduction = 1.0 - num_canonical / lookup.numel()
    print(
        f"Wrote {args.out}: raw_vocab={lookup.numel()} "
        f"canonical={num_canonical} reduction={reduction:.1%}"
    )


if __name__ == "__main__":
    _main()
