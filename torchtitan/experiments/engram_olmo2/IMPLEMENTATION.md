# Engram on OLMo2 — Implementation Notes

This document explains how the Engram conditional-memory module (DeepSeek,
*"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large
Language Models"*) is implemented on top of the core OLMo2 backbone in
torchtitan, the design decisions behind it, and how it has been validated.

It is written for someone who knows the paper at a high level and wants to
understand *this* implementation: where each piece lives, why it is shaped the
way it is, and what the non-obvious correctness concerns were.

---

## 1. What Engram is (one paragraph)

Engram adds a second axis of sparsity alongside MoE: **conditional memory**.
Instead of activating parameters to *compute* (MoE), it activates a tiny number
of rows in a massive embedding table to *retrieve* static knowledge in O(1).
For each token position it forms the suffix N-grams (here 2- and 3-grams),
hashes each into `K` independent prime-sized tables (multi-head hashing to
suppress collisions), concatenates the retrieved row vectors, and fuses that
static "memory" vector into the residual stream through a context-aware gate.
A short depthwise causal convolution widens the receptive field. The module is
applied only at a couple of early layers, so it offloads the backbone's
bottom-layer "static reconstruction" work while leaving the embedding / un-embedding
and the rest of the backbone intact.

The retrieved memory vector for position `t`:

```
g_{t,n}        = (x'_{t-n+1}, ..., x'_t)            # compressed suffix n-gram
z_{t,n,k}      = phi_{n,k}(g_{t,n})                  # multiplicative-XOR hash, mod M_{n,k}
e_{t,n,k}      = E_{n,k}[z_{t,n,k}]                  # row lookup
e_t            = concat over n in {2..N}, k in {1..K}   # the memory vector
```

Fusion into the residual stream (single-stream variant — see §3):

```
k_t      = W_K e_t,   v_t = W_V e_t
alpha_t  = sigmoid( RMSNorm(h_t)^T RMSNorm(k_t) / sqrt(d) )   # scalar gate in (0,1)
ṽ_t      = alpha_t * v_t
Y        = SiLU(Conv1D(RMSNorm(ṽ))) + ṽ
H        = H + Y                                              # residual add
```

---

## 2. Design decisions (and the constraints that drove them)

| Decision | Why |
|---|---|
| **New `experiments/engram_olmo2/` package, no core edits** | torchtitan's rules forbid leaking experiments into core. We *subclass* OLMo2 and *reuse* common components rather than branching core files. |
| **No mHC / hyper-connections** (explicit requirement) | The paper's default uses Manifold-Constrained Hyper-Connections (M=4 parallel branches, M key projections). We use a **single residual stream**: one `W_K`, one gate, one `W_V`, `ShortConv` over `[B,L,D]` with no group axis. This is the paper's "w/o multi branch" configuration. |
| **Output dim = backbone hidden size** | The fusion result is *added* to the residual stream, so `W_V`/`W_K` map the internal memory dim to the backbone `dim` (2048 for 1B). The internal `per_head_dim` is a separate, tunable knob. |
| **Tables vocab/row-parallel sharded, not FSDP** (explicit requirement) | The tables can reach billions of rows and not fit on one GPU. FSDP would all-gather the full table every step — catastrophic. We shard rows across GPUs and gather only the *active* rows via All-to-All (paper §2.5, Fig. 2a). |
| **Hashing on-GPU from token ids in forward** | The hash depends only on input ids, so it can run model-side with no dataloader changes. Prime table sizes (needing `sympy`) are computed **offline at config-build time** and baked into the config, keeping the training path free of non-PyTorch deps. |
| **Compression table cached to disk** | Building it decodes every id in the tokenizer vocab (slow). It's a one-time offline step (`transformers`/`tokenizers` only needed there); training loads a plain int64 tensor. |

---

## 3. File-by-file walkthrough

```
torchtitan/experiments/engram_olmo2/
├── tokenizer_compression.py   # P: V → V' (offline build + cached load)
├── hashing.py                 # prime sizes (offline) + EngramHasher (torch-native)
├── multi_head_embedding.py    # concatenated table; replicated + vocab-parallel paths
├── short_conv.py              # depthwise causal Conv1d (zero-init) + RMSNorm + SiLU
├── engram.py                  # EngramLayer: gate + projections + residual output
├── model.py                   # EngramOLMo2Model / ...TransformerBlock (subclass OLMo2)
├── parallelize.py             # backbone FSDP + table Shard(0) via ignored_params
├── model_registry.py          # builds the model config (debugmodel / 1B_stem_dag_l12_size)
├── config_registry.py         # trainer configs (debugmodel / 1b_stem_dag_l12 prefine+midfine)
├── trainer.py                 # EngramTrainer: table param group + mesh-aware grad clipping
├── state_dict_adapter.py      # reuse OLMo2 HF mapping; Engram params pass through
├── tests/                     # CPU unit tests + multi-GPU embedding test
└── README.md / IMPLEMENTATION.md
```

### 3.1 `tokenizer_compression.py` — vocabulary projection `P: V → V'`

Standard subword tokenizers give disjoint ids to semantically equivalent surface
forms (`Apple` vs `▁apple`). Engram collapses raw ids into **canonical** ids via
a text normalizer (NFKC → NFD → strip accents → lowercase → whitespace-collapse).

- `build_compression_table(tokenizer)` decodes each id, normalizes it, and assigns
  one canonical id per distinct normalized string. Byte-fallback tokens that don't
  decode cleanly (`"�"`) are keyed by their raw token string so they stay distinct.
- This is **expensive** (one decode per vocab entry), so it is an **offline CLI**
  that caches `{lookup, num_canonical}` to a `.pt`:
  ```bash
  python -m torchtitan.experiments.engram_olmo2.tokenizer_compression \
      --tokenizer deepseek-ai/DeepSeek-V3 --out compression_lookup.pt
  ```
- Training loads it with `load_compression_table` as a plain int64 tensor. If no
  path is configured, an **identity** lookup is used (no merging) — handy for
  debug/tests.

### 3.2 `hashing.py` — deterministic multi-head N-gram hashing

Two offline (config-time) helpers and one runtime module:

- `compute_head_table_sizes(...)` — for each `(layer, n-gram order, head)` picks the
  next prime ≥ the base table size, **globally distinct** across all heads (uses
  `sympy.isprime`). Computed once when the config is built and baked into the
  config as a flat list of ints.
- `compute_layer_multipliers(...)` — per-layer odd multipliers `m_k = 2r+1`, seeded
  per layer (so different Engram layers hash the same n-gram differently), bounded
  so `id * m_k` cannot overflow int64.
- `EngramHasher` (a torchtitan `Module`, but constructed directly, not via
  `Config.build()`): given raw `tokens [B,L]`, it
  1. compresses ids through the lookup (negative/pad ids map to the compressed pad),
  2. forms shifted copies for the suffix n-grams,
  3. computes `mix = (s_0 * m_0) XOR (s_1 * m_1) XOR ...` and `z = mix % M_{n,k}`
     for every head, returning `hash_ids [B, L, num_heads_total]`.

  **Why it owns no parameters/buffers:** the lookup and multipliers are fixed,
  deterministic, and recomputable. Registering them as buffers would (a) make them
  meta tensors under torchtitan's meta-device build and get wiped by `to_empty()`,
  and (b) needlessly bloat checkpoints. Instead, multipliers/table-sizes are Python
  ints (no device affinity; `int * device_tensor` stays on device) and the
  compression lookup is a plain CPU tensor lazily moved to the compute device on
  first forward.

`ngram_hashes_numpy(...)` is a reference numpy implementation used only to test the
torch path bit-for-bit.

### 3.3 `multi_head_embedding.py` — the big table + vocab parallelism

`MultiHeadEmbedding` holds **one** `nn.Parameter` of shape `[sum(head_table_sizes),
per_head_dim]` (all heads concatenated). Per-head index spaces are made disjoint by
adding fixed row `offsets` before the gather. Output is `[..., num_heads, per_head_dim]`.

It has **two forward paths**, selected by whether `weight` is a sharded DTensor:

- **Replicated / single-rank** (`weight` is a plain tensor or `Replicate` DTensor):
  a plain `F.embedding` gather. Used on 1 GPU and in CPU tests.
- **Vocab-parallel** (`weight` is `Shard(0)` over a 1-D mesh): each rank owns the
  contiguous row slice `[r·rows_per_rank, (r+1)·rows_per_rank)`. Lookup uses a custom
  autograd Function `_VocabParallelGather`:
  - **Forward**: compute each requested global id's owner rank, stable-sort requests
    by owner, exchange per-owner counts, **All-to-All** the local-row indices to
    owners, each owner does a local `F.embedding` on its shard, **reverse All-to-All**
    returns the rows, then invert the sort to restore the caller's order.
  - **Backward**: re-apply the forward permutation to the output grads, **All-to-All**
    them to the owners, and `index_add_` them into the owner's local-shard grad. The
    reverse All-to-All already **sums each owned row's contributions across the global
    batch**, so no extra all-reduce over the shard axis is needed.

  `to_local(grad_placements=[Shard(0)])` tells autograd the local gradient *is* this
  rank's row-shard gradient (per the project's DTensor-gradient-placement rule).

### 3.4 `short_conv.py` — receptive-field convolution (paper Eq. 5)

`ShortConv` = `RMSNorm` → depthwise causal `Conv1d` (`groups=channels`, kernel 4,
dilation = max n-gram) → `SiLU`, then the caller adds the input (`Y = ṽ + SiLU(Conv(RMSNorm(ṽ)))`).
Causality is enforced by left-padding `(k-1)·dilation` and trimming the right tail
to length `T`. The conv weight is **zero-initialized**, so at step 0 the conv output
is 0 and the module is an identity over `ṽ` — the paper's "preserve identity at init"
trick. A small `DepthwiseConv1d` configurable wrapper is added so the module conforms
to torchtitan's `Module` protocol (which requires every submodule to be a `Module`).

### 3.5 `engram.py` — the layer that ties it together

`EngramLayer`:
- builds the hasher, the multi-head embedding, `value_proj`/`key_proj`
  (`Linear: engram_hidden → dim`), `q_norm`/`k_norm` (`RMSNorm(dim)`), and `ShortConv`;
- `compute_hashes(tokens)` runs the hasher (called by the model **outside** the
  activation-checkpoint region — see §4.3);
- `forward(hidden_states, hash_ids)` does: gather + flatten → cast to projection dtype
  → single-stream gate `sigmoid(RMSNorm(h)·RMSNorm(W_K e)/√d)` → `ṽ = gate · W_V e` →
  `ṽ + short_conv(ṽ)`. We use the paper's plain sigmoid gate (not the reference demo's
  extra signed-sqrt).

The dtype cast exists because the table is excluded from FSDP's mixed-precision policy
(it's sharded separately and stays in its own dtype), while the projections are FSDP-cast
to bf16; the cast aligns them.

### 3.6 `model.py` — subclassing OLMo2 to thread token ids

The core `Decoder.forward` only passes hidden states to each block, but Engram needs
the raw token ids. Rather than edit core:
- `EngramOLMo2TransformerBlock(OLMo2TransformerBlock)` adds an optional `engram` and,
  when present, does `x = x + engram(x, hash_ids)` *before* the standard OLMo2 post-norm
  attention + FFN.
- `EngramOLMo2Model(OLMo2Model)` overrides `forward` to, for each Engram layer, compute
  `hash_ids = engram.compute_hashes(tokens)` and pass them in. Everything else is the
  stock OLMo2 forward.

### 3.7 `parallelize.py` — backbone FSDP + table sharding

The backbone is parallelized like upstream OLMo2 (CP / AC / compile / FSDP). The Engram
tables are handled specially:
- `_shard_engram_tables` converts each table to a `Shard(0)` DTensor over the **pure FSDP
  axis** (`dp_shard × cp`, a 1-D mesh → a single, well-defined All-to-All process group).
  It pads `total_rows` up to a multiple of `world` with trailing dead rows so the gather's
  uniform-rows/rank assumption holds for any FSDP degree (padding rows are never addressed).
- `apply_fsdp` then FSDP-wraps every block **but** passes the table params via
  `ignored_params=` so FSDP never all-gathers or re-shards them. The small Engram
  projections/norms/conv shard normally with their block.

This is the crux of "the large table can't fit on one GPU, and FSDP is the wrong tool":
the table is model-parallel (row-sharded), the rest is data-parallel.

`full_dtensor` and pipeline parallelism currently raise `NotImplementedError` (PP would
need token ids threaded to the Engram stage; documented TODO). TP is already rejected by
the OLMo2 backbone (cross-head QK-norm isn't TP-safe).

### 3.8 `model_registry.py` / `config_registry.py` / `state_dict_adapter.py`

- `model_registry` builds `debugmodel` / `1B` / `7B` configs: standard OLMo2 layer stack
  with an `EngramLayer.Config` attached at the chosen layers (2 and 7 for real flavors;
  2 and 4 for the tiny debugmodel). Prime table sizes are computed here, once.
- `config_registry` provides trainer configs. Engram tables get an optimizer param group
  with **5× LR and no weight decay** (paper §4.1), matched by FQN
  `\.engram\.embedding\.weight$`. The module is registered in
  `experiments/__init__.py::_supported_experiments` so `--module engram_olmo2` resolves.
- `state_dict_adapter` subclasses OLMo2's: the backbone keeps OLMo2's exact HF mapping,
  so **a vanilla OLMo2 checkpoint loads into the backbone unchanged**; Engram params are
  new and init fresh.

---

## 4. Non-obvious correctness issues encountered (and fixes)

These were found and fixed during bring-up on real GPUs; they're the parts most likely
to bite a future change.

### 4.1 Meta-device build wiping fixed state
torchtitan builds the model under `with torch.device("meta")` then `to_empty()`s it.
Anything created as a buffer/parameter becomes meta and is wiped. The hasher's lookup,
multipliers, and `.item()` on the pad id all broke under this. **Fix:** keep the hasher's
fixed data as Python ints + a plain CPU tensor (not buffers), and force the compression
lookup to CPU at config-build (`torch.arange(..., device="cpu")` / `.cpu()` after load).

### 4.2 dtype mismatch at the table → projection boundary
The table is `ignored_params` for FSDP, so it isn't cast to bf16, but the projections are.
`F.linear(float_embeddings, bf16_weight)` errors. **Fix:** cast embeddings to
`value_proj.weight.dtype` in the Engram forward.

### 4.3 Integer recompute under activation checkpointing
With selective AC, the integer hashing inside the checkpointed block was recomputed in
backward and hit `RuntimeError: Autograd not support dtype: Long`. **Fix:** compute hashes
in the **model forward, outside** the per-block AC region, and pass `hash_ids` into the
block. This is also closer to the paper (hashes are deterministic and prefetchable).

### 4.4 Backward gradient summation across ranks
The vocab-parallel backward must sum a row's gradient over *all* ranks that requested it,
not just the local batch. The reverse All-to-All in `_VocabParallelGather.backward` does
exactly this. (An early version of the *test* — not the kernel — missed this and looked
like a failure; the fix was to make the reference aggregate every rank's inputs.)

---

## 5. Validation

All run against the GPU box's torch nightly (the local env is torch 2.6 and too old for
torchtitan's common modules; FSDP2 `ignored_params` only exists on the nightly).

- **CPU unit tests** (`tests/test_engram_cpu.py`, 7 tests, all pass):
  - torch hashing **matches the numpy reference bit-for-bit**;
  - prime sizes are globally distinct and ≥ base vocab;
  - different layers hash differently; indices stay in range;
  - `ShortConv` zero-init is exactly identity; conv is causal;
  - replicated multi-head embedding equals a manual offset-gather.
- **Multi-GPU embedding** (`tests/test_vocab_parallel_embedding.py`): vocab-parallel
  forward **and** backward-gradient shards match a full-table replicated reference on
  **2 and 4 GPUs**.
- **End-to-end training** (`--config engram_olmo2_debugmodel`):
  - 2-GPU FSDP: tables shard (log confirms), loss decreases 4.23 → 3.60, both ranks
    produce identical loss/grad_norm;
  - selective AC works (memory drops vs AC off);
  - single-GPU replicated path + `torch.compile` both run.
- **Lint:** `pre-commit run` passes (flake8, µfmt, pydoclint, codespell, license, etc.;
  only `pyrefly` is unavailable in the local env).

---

## 6. Defaults and knobs

- Engram layers: **2 and 7**; `max_ngram_size=3` (2- and 3-grams); `n_head_per_ngram=8`.
- `engram_vocab_size=2358272` base table size (each head rounds up to the next distinct
  prime).
- `per_head_dim` (internal memory column width) is tunable; `engram_hidden =
  (max_ngram-1) · n_head · per_head_dim` and the projections map `engram_hidden → dim`.
- Tables shard over the FSDP axis; `total_rows` is padded up to a multiple of that degree
  with dead rows (never addressed), so any FSDP degree works.
- Optimizer: token embeddings and Engram tables use no weight decay; Engram tables use 5× LR.

## 7. Known limitations / TODO

- No pipeline parallelism and no `full_dtensor` (both raise clearly).
- TP unsupported by the OLMo2 backbone (cross-head QK-norm).
- The `1B`/`7B` flavors default to the OLMo2 tokenizer vocab with **identity** compression;
  to use real tokenizer compression at the full 2.36M vocab, run the offline CLI and set
  `compression_table_path` on the Engram configs.
- Inference-time host-memory offload / prefetching (paper §2.5) is not implemented; this
  is training-focused.
