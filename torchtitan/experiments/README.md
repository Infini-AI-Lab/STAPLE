Each subfolder here is one model family built on the core OLMo2 backbone in
[`models/olmo2`](../models/olmo2/). They reuse core components (trainer,
dataloaders, checkpointing, parallelism) through their own `ModelSpec`, and core
never imports from `experiments/`.

| Experiment | `--module` | What it adds |
| ----- | ----- | ----- |
| [stem_dag](./stem_dag/) | `stem_dag` | STAPLE: selected layers add a token-indexed embedding (zero-initialized) to the FFN up-projection, `w2(silu(w1(x)) * (w3(x) + y))`; full-table and low-rank variants |
| [engram_olmo2](./engram_olmo2/) | `engram_olmo2` | Engram: hashed N-gram memory tables fused into selected layers |
| [olmo2_moe](./olmo2_moe/) | `olmo2_moe` | MoE upcycled from OLMo2 (E8G1T1, MoE in every other layer) |

Every experiment provides warmup (`*_prefine` / `*_warmup`) and midtraining
(`*_midfine`) configs, plus a `*_debugmodel` config used by the integration tests.
