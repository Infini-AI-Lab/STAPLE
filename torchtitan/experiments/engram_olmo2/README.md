# Engram on OLMo2

Implements the **Engram** conditional-memory module (DeepSeek, *"Conditional
Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"*)
on top of the core OLMo2 backbone, as a self-contained `experiments/` package
that does not modify core torchtitan.

Engram augments selected transformer layers with large, hash-addressed N-gram
embedding tables. For each position it retrieves static N-gram memory via O(1)
hashed lookup and fuses it into the residual stream through a context-aware
gate, relieving the backbone's early layers from reconstructing static local
patterns.

## What's here

| File | Purpose |
|------|---------|
| `tokenizer_compression.py` | Offline builder + cached loader for the raw→canonical id map (`P: V → V'`). CLI caches the table to disk. |
| `hashing.py` | Deterministic prime table sizes (offline, `sympy`) + `EngramHasher`: torch-native multiplicative-XOR multi-head N-gram hashing. |
| `multi_head_embedding.py` | Concatenated multi-head table with a replicated plain-gather path and a **vocab-parallel All-to-All** path (custom autograd backward dispatches grads). |
| `short_conv.py` | Depthwise causal `Conv1d` (zero-init) + RMSNorm + SiLU (paper Eq. 5). |
| `engram.py` | `EngramLayer`: single-stream gate, value/key projections to backbone `dim`, residual output. **No mHC / hyper-connections.** |
| `model.py` | `EngramOLMo2Model` / `EngramOLMo2TransformerBlock`: OLMo2 backbone, hashes computed in the model forward (outside the AC region) and passed into Engram layers. |
| `parallelize.py` | Backbone TP/CP/AC/compile/FSDP; Engram tables sharded `Shard(0)` over the FSDP axis and excluded from FSDP via `ignored_params`. |
| `model_registry.py` / `config_registry.py` | Model + trainer config registries. Engram tables train at 5× LR, no weight decay, on their own warmup schedule (set via `engram_hyperparams`). |
| `trainer.py` | `EngramTrainer`: injects the Engram-table optimizer param group (lr/weight-decay) and a per-group LR lambda (separate warmup / min-lr floor) from `EngramEmbeddingHyperparams`, and logs the tables' lr as `lr/engram_embeddings`. |
| `loss.py` | `EngramLoss` = `ChunkedCELoss` + per-table / combined Engram embedding gradient-norm reporting (`grad_norm/engram_table_layer_{i}`, `grad_norm/engram_tables`); surfaced via wandb/tensorboard. |
| `state_dict_adapter.py` | Reuses OLMo2's HF mapping; Engram params init fresh (vanilla OLMo2 checkpoints load into the backbone unchanged). |

## Defaults

- Engram at **layers 2 and 7**, `max_ngram=3` (2- and 3-grams), `n_head_per_ngram=8`.
- `engram_vocab_size=2358272` (base table size; each head uses the next distinct prime).
- Tables are **row/vocab-parallel** sharded across the FSDP group (not FSDP-sharded);
  forward gathers rows via All-to-All, backward dispatches gradients back.

## Build the compression table (offline, once)

```bash
python -m torchtitan.experiments.engram_olmo2.tokenizer_compression \
    --tokenizer <hf-tokenizer> --out <path>/compression_lookup.pt
```

Then set `compression_table_path` on the Engram configs. Without it, an identity
compression (no merging) of size `raw_vocab_size` is used — fine for debugging.

## Run

```bash
# 2-GPU FSDP debug run
torchrun --nproc_per_node=2 -m torchtitan.train \
    --module engram_olmo2 --config engram_olmo2_debugmodel \
    --parallelism.data_parallel_shard_degree 2

# 1B
torchrun --nproc_per_node=8 -m torchtitan.train \
    --module engram_olmo2 --config engram_olmo2_1b
```

## Tests

```bash
# CPU unit tests (hash determinism vs numpy, conv identity/causality, gather)
python -m unittest torchtitan.experiments.engram_olmo2.tests.test_engram_cpu

# Multi-GPU vocab-parallel embedding forward+backward correctness
PYTHONPATH=$PWD torchrun --nproc_per_node=2 \
    torchtitan/experiments/engram_olmo2/tests/test_vocab_parallel_embedding.py
```

## Limitations / TODO

- **No pipeline parallelism** and **no `full_dtensor`** yet (`parallelize.py`
  raises). Engram needs raw token ids at its layer; PP would require threading
  tokens to the relevant stage(s).
- TP is unsupported by the OLMo2 backbone (cross-head QK-norm); use FSDP/CP.
- Table sharding pads `total_rows` up to a multiple of the FSDP-axis size with
  trailing dead rows (never addressed by any hash id), so any FSDP degree works.
