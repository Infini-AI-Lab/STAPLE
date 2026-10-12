# OLMo2 MoE STEM-DAG Implementation

## Supported Shape

Static STEM supports granular experts and arbitrary top-k routing. If the
dense FFN width is `H`, granularity is `G`, and expert width is `F=H/G`, the
static table still stores one full `H`-wide row per token. Routed expert `e`
receives contiguous shard `e % G` from that row.

Projected per-expert STEM remains limited to `G=1, T=1`. Its tables and shared
projection have different storage semantics and are intentionally unchanged.

The experiment remains separate from `stem_dag_moe`.

## Static E8G8T8 STAPLE12 Model

The `1B-E8G8T8-8L-STEM12` topology in
[`config_registry.py`](config_registry.py) has:

- 16 transformer layers with model dimension `D=2048`.
- Dense FFN width `H=8192`.
- 64 routed experts with expert width `F=1024`.
- Top-8 routing without selected-score renormalization.
- MoE layers `[1,3,5,7,9,11,13,15]`.
- Static STEM layers `[1,2,...,12]`.
- One `[100352,8192]` static table for every STEM layer.

The intersection of the MoE and STEM layer sets determines which layers use
STEM experts:

| Layers | FFN type |
| --- | --- |
| `0` | Dense FFN without STEM |
| `1,3,5,7,9,11` | Static STEM MoE |
| `2,4,6,8,10,12` | Dense STEM DAG FFN |
| `13,15` | Ordinary MoE |
| `14` | Ordinary dense FFN |

All 16 layers retain the same OLMo2 attention block. `_build_layers()` creates
these categories by testing membership in `moe_layers` and `stem_layers`.

## Distributed Topology

The production launcher
`scripts_gl/_olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12.sh` uses:

```text
dp_replicate = NNODES
dp_shard     = 8
ep           = 4
tp = cp = pp = 1
```

`ParallelDims.build_mesh()` derives `efsdp = dp_shard / ep = 2`, giving:

```text
dense mesh:  [dp_replicate=NNODES, fsdp=8]
sparse mesh: [dp_replicate=NNODES, efsdp=2, ep=4]
```

Dense transformer parameters are FSDP-sharded across the eight GPUs of each
data-parallel replica. The replica dimension normally spans nodes.

For expert parameters, EP shards dimension zero of the `[E,...]` tensors.
Each EP rank therefore handles 16 of the 64 expert identities. eFSDP then
storage-shards those expert tensors over two ranks. Persistent expert storage
per rank is consequently one eighth of the complete expert tensor, although
an EP rank has slices for 16 expert identities at rest. FSDP gathers the
complete weights needed by those experts before their computation.

Each static STEM table is sharded on its width over EP:

```text
global table: [100352,8192]
local shard:  [100352,2048]
```

The table is replicated over `efsdp` and `dp_replicate`. It is excluded from
FSDP because its forward implements the EP feature sharding explicitly.

## End-to-End Forward Pass

`OLMo2MoEStemDagModel.forward()` first clears the per-layer MoE auxiliary-loss
buffers and embeds the input token IDs:

```text
h = token_embedding(tokens)
```

It then visits the layers in order. A layer outside `stem_layers` receives only
`h`. For static STEM layer `l`, the model performs:

```text
y_l = stem_table_l[tokens]
h = layer_l(h, stem_embedding=y_l)
```

The lookup always uses the original token IDs, not a representation produced
by an earlier transformer layer.

Every block follows OLMo2 post-normalization:

```text
h   = x + attention_norm(attention(x))
out = h + ffn_norm(ffn_or_moe(h))
```

### Distributed STEM Lookup

`StemEmbedding.forward()` turns a width-sharded table into full-width STEM
vectors for the rank's local tokens:

1. All-gather local `[B,L]` token IDs across the EP group.
2. Look up the gathered IDs in the local `[V,H/EP]` table shard.
3. Use an autograd-enabled all-to-all to transpose batch sharding into feature
   sharding.
4. Reassemble `[B_local,L,H]` for the original local batch.

Conceptually, the all-gather lets every feature shard see every EP rank's token
IDs. The all-to-all then sends the corresponding feature pieces back to each
token-owning rank.

### Dense STEM Layer

For layers `2,4,6,8,10,12`, `StemDagFeedForward.forward()` computes:

```text
gate   = silu(w1(h))
up     = w3(h)
output = w2(gate * (up + y_l))
```

The `[B,L,8192]` STEM vector has the same width as the dense FFN up branch.
FSDP gathers the dense block parameters before this computation and may
reshard them after the forward according to the configured policy.

### Static STEM MoE Layer

For layers `1,3,5,7,9,11`, `StaticStemMoE.forward()` first invokes the router.
`OLMoERouter` computes `[B,L,64]` logits and softmax probabilities in FP32,
then chooses eight experts per token:

```text
topk_scores:     [B,L,8]
topk_expert_ids: [B,L,8]
```

`route_norm=False`, so the eight selected probabilities are not normalized to
sum to one. The router also records raw logits for z-loss and full softmax
probabilities plus assignment counts for load-balancing loss.

`StaticStemGroupedExperts.forward()` flattens the batch and sequence dimensions
and reshapes each full STEM vector:

```text
x:    [B,L,2048] -> [T,2048]
stem: [B,L,8192] -> [T,8,1024]
```

For an assignment to global expert `e`, it selects STEM shard `e % 8`.
Therefore expert `e` computes:

```text
gate_e = silu(x @ w1_e.T)
up_e   = x @ w3_e.T
out_e  = (gate_e * (up_e + stem[:, e % 8])) @ w2_e.T
```

The hidden-state dispatcher uses
`DeduplicatedAllToAllTokenDispatcher`. If one token selects several experts
owned by the same EP rank, its `[2048]` hidden state is sent to that rank once
and expanded into assignment rows after receipt. This reduces hidden-state
traffic without changing expert-major grouped-GEMM input.

The selected STEM `[1024]` slices still follow assignment-level dispatch.
They are sorted by expert, sent with an autograd-enabled all-to-all, and
permuted into the same expert-major order as the hidden states.

After the three grouped GEMMs, combine:

1. Reverses the expert-major permutation.
2. Performs the reverse EP all-to-all.
3. Multiplies each assignment output by its selected router score.
4. Scatter-adds the eight expert contributions into one output per token.

The block applies `ffn_norm`, adds the residual, and passes the result to the
next layer. After layer 15, the model applies its final norm and language-model
head.

## Backward Pass

The trainer computes cross entropy with sum reduction and divides by the
number of valid tokens across every data-parallel rank and every accumulated
microbatch. The OLMoE loss then adds:

```text
0.01  * load_balancing_loss
0.001 * router_z_loss
```

For an MoE layer, autograd traverses the combine path in reverse:

1. Scatter-add backward produces gradients for the selected assignment
   outputs.
2. The combine all-to-all backward sends those gradients to the ranks that
   computed the experts.
3. Grouped-GEMM backward computes gradients for `w1`, `w2`, `w3`, routed
   hidden states, and selected STEM slices.
4. The STEM dispatch all-to-all backward returns feature-shard gradients
   toward the ranks holding the corresponding static table shards.
5. Hidden-state dispatch backward returns assignment gradients to the source
   ranks.

The deduplicated dispatcher has a custom autograd function. Its backward uses
the ordinary assignment-level reverse all-to-all, then accumulates all
duplicate assignment gradients into the original token's hidden-state row.

Routing scores receive gradients because combine multiplies expert outputs by
the selected scores. The integer top-k expert IDs are nondifferentiable.
Load-balancing loss and z-loss add router gradients through the complete
softmax probabilities and raw logits, respectively.

As backward reaches each FSDP-wrapped block, FSDP regathers any parameters
resharded after forward. It reduce-scatters dense gradients over `fsdp=8` and
expert gradients over `efsdp=2`, with the HSDP replica reduction over
`dp_replicate`. Explicit forward and backward prefetch links overlap adjacent
block parameter gathers.

FSDP gradient division is disabled because cross entropy is already divided
by the global valid-token count. During gradient accumulation, the trainer
also disables FSDP gradient synchronization on every non-final microbatch.

## Static STEM Gradient Reduction

Static tables are not managed by FSDP. The parallelizer installs one
`StemGradReducer` over the replicated `dp_replicate x efsdp` mesh. The trainer
launches its coalesced all-reduce after the final gradient-accumulation
microbatch and synchronizes the side stream before gradient clipping and the
optimizer step.

Delaying the reduction until all local microbatch gradients have accumulated
avoids repeatedly reducing earlier contributions. It also replaces one NCCL
call per STEM layer with a single coalesced operation:

```text
local_accumulated_gradient
    = microbatch_1_gradient + ... + microbatch_N_gradient
global_gradient
    = all_reduce(local_accumulated_gradient)
```

With the production launcher:

```text
gradient_accumulation
    = global_batch / (local_batch * data_parallel_degree)
    = 512 / (4 * 8 * NNODES)
    = 16 / NNODES
```

A four-node run therefore accumulates four local microbatch gradients before
performing one replicated-table reduction.

## Common Expert Computation

For token hidden state `x_i` and its selected expert `n_i`, both strategies
modify the SwiGLU up branch:

```math
h_i =
\operatorname{SiLU}(W_{1,n_i}x_i)
\odot
\left(W_{3,n_i}x_i+s_i\right),
```

```math
y_i=p_iW_{2,n_i}h_i.
```

`score_before_experts=False` keeps the routing score `p_i` in the dispatcher
combine step.

The relevant tensor shapes are:

- Activations `x_BLD`: `[B, L, D]`
- Static STEM signal `stem_BLF`: `[B, L, H]`, reshaped to `[B, L, G, F]`
- Projected tables `stem_table_EVR`: `[N, V, R]`
- Shared projection `stem_up_HR`: stored as `stem_up_FR`, `[H, R]`
- Selected expert IDs: `[B, L, T]`

## Configuration Flow

`_build_layers()` separates `moe_layers` from `stem_layers`. Static STEM allows
the sets to overlap partially: overlapping layers use `StaticStemMoE`, while
STEM-only layers use `StemDagFeedForward`. Projected STEM still requires every
STEM layer to be an MoE layer.

`_routed_moe()` creates routers, dispatchers, and expert matrices from the
topology's `num_experts`, `top_k`, `granularity`, and `expert_hidden_dim`.

Non-STEM MoE layers use the existing TorchTitan grouped experts. STEM layers
use either `StaticStemMoE` or `ProjectedStemMoE`.

Production topologies include:

- `1B-E8G1T1-8L`, with MoE and STEM layers `[1,3,...,15]`.
- `1B-E8G8T8-8L-STEM12`, with STEM layers `[1..12]` and MoE layers
  `[1,3,...,15]`.

The static and projected warmup/midfine functions call the corresponding
public E8G1T1-L8 base functions:

- `olmo2_moe_1b_e8g1t1_8l_warmup()`
- `olmo2_moe_1b_e8g1t1_8l_midfine()`

They preserve the base training, data, optimizer, scheduler, loss, and
parallelism settings, then replace:

- The model spec with the selected STEM strategy.
- The allowed missing checkpoint prefixes with strategy-specific STEM keys.
- The STEM learning-rate and weight-decay overrides.

Following the STAPLE stage recipes, the STEM parameter group uses learning
rate `4e-4` during warmup and `3e-4` during midfine, with `1e-4` weight decay
in both stages.

## Static STEM

### Parameters

Each static STEM layer owns one zero-initialized embedding:

```math
S_l \in \mathbb{R}^{V \times H}.
```

The embeddings live in `model._stem_embeddings`, keyed by transformer layer
ID.

### Forward Path

At a static STEM layer:

1. `OLMo2MoEStemDagModel` looks up the original token IDs to produce
   `[B, L, H]`.
2. `StaticStemMoE` computes top-k routing.
3. `StaticStemGroupedExperts` reshapes each row to `[G, F]`.
4. Each assignment selects shard `expert_id % G`.
5. Selected shards follow the activation assignments' stable expert ordering,
   all-to-all splits, and expert-major permutation.
6. The expert adds the routed shard to `W3x`.

The grouped expert validates `H = G * F` and `num_experts % G == 0`.

## Projected Per-Expert STEM

### Parameters

Each projected layer stores:

```math
S_l \in \mathbb{R}^{N \times V \times R},
```

```math
U_l \in \mathbb{R}^{H \times R},
```

with:

```math
R=8\left\lceil\frac{H}{8N}\right\rceil.
```

The implementation calculates this directly:

```python
stem_dim = 8 * ceil(dense_hidden_dim / (8 * num_experts))
```

There is no `T`-dependent expansion calculation.

### Forward Path

The projected implementation dispatches token IDs instead of materializing
all expert STEM vectors before routing:

1. `_dispatch_token_ids()` repeats token IDs in top-1 assignment order.
2. EP all-to-all uses the activation dispatch metadata.
3. `permuted_indices` converts received IDs to expert-major order.
4. Local expert IDs are generated from `num_tokens_per_expert_E`.
5. Each assignment looks up `S_l[n_i, v_i, :]`.
6. The layer-shared `U_l` projects the row from `R` to `H`.
7. The projected vector is added to `W3x`.

The expert tables use normal initialization with standard deviation
`1/sqrt(R)`. `U_l` is zero-initialized, preserving base-MoE output at
initialization.

## Model Integration

`OLMo2MoEStemDagTransformerBlock` leaves attention unchanged. After attention:

- `StaticStemMoE` receives a full-width static embedding.
- Dense static STEM layers pass the embedding to `StemDagFeedForward`.
- `ProjectedStemMoE` receives token IDs.
- Non-STEM MoE layers use the original MoE call.
- Dense layers use the original feed-forward call.

The model configuration adds:

- `stem_strategy`
- `stem_layers`
- `stem_embeddings`

Only static STEM populates `stem_embeddings`; projected tables are direct
parameters of each grouped-expert module.

## Sharding and FSDP

### Static Tables

Static `[V, H]` tables are feature-sharded over EP. `StemEmbedding.forward()`:

1. All-gathers token IDs over EP.
2. Looks up each local `[V, H/EP]` feature shard.
3. Uses all-to-all to reconstruct full `[B, L, H]` rows for local tokens.

The tables are replicated over EDP, excluded from FSDP, and use an explicit
coalesced EDP gradient all-reduce after the final accumulation microbatch.

### Projected Parameters

`stem_table_EVR` follows the same sparse EP/EDP placement as routed expert
weights. EP shards dimension zero, so each rank stores tables for its local
experts.

`stem_up_FR` is excluded from the expert parameter set in the mixed FSDP
placement callback. It therefore uses the dense DP/FSDP mesh and remains
shared across experts within the layer.

The rest of the mixed FSDP implementation mirrors Llama 4:

- Dense parameters use `dp_mesh`.
- Expert weights and projected tables use `edp_mesh`.
- The shared STEM projection uses `dp_mesh`.

Tensor parallelism, pipeline parallelism, and full-DTensor mode remain
unsupported. Context parallelism uses the existing attention integration.

## Checkpoint and Optimizer Behavior

Static tables and projected shared projections start at zero, so both
strategies can load a base MoE checkpoint without changing its initial output.

Warmup loads the E8G1T1-L8 base checkpoint model-only. The fully initialized
STEM model is constructed first, and partial loading is limited to:

- Static: `_stem_embeddings.*`
- Projected: `layers.<id>.moe.experts.stem_*`

The missing STEM parameters retain the initialization described above. Midfine
loads the corresponding warmup checkpoint with no missing-key allowlist, so
every model parameter must be present.

The STAPLE12 E8G8T8 converter preserves `_stem_embeddings.*`, converts only
the selected alternating FFNs, and emits a complete model-only checkpoint.
Its warmup therefore loads strictly without a missing-key allowlist.

```bash
python scripts/checkpoint_conversion/upcycle_olmo2_to_moe.py \
  STAPLE_CHECKPOINT OUTPUT_CHECKPOINT \
  --moe-layers 1 3 5 7 9 11 13 15 \
  --expansion-rate 8 --granularity 8 --top-k 8
```

The STEM optimizer group includes:

- Static embedding weights.
- `stem_table_EVR`.
- `stem_up_FR`.

It can override learning rate and weight decay independently from backbone and
expert weights.

## Parameter Accounting

For `L_stem` layers:

```math
P_{\mathrm{static}}=L_{\mathrm{stem}}VH,
```

```math
P_{\mathrm{projected}}
=L_{\mathrm{stem}}(NVR+HR).
```

Projected STEM adds `2HR` FLOPs per token and STEM layer. Top-1 routing removes
the previous top-k multiplier.

## Tests

The focused unit test covers:

- Parameter totals and layer categories for E8G1T1-L8 and STAPLE12 E8G8T8.
- Inheritance from the corresponding public MoE warmup/midfine configs.
- Projected rank and shape calculations.
- Rank rounding to a multiple of eight.
- Zero initialization.
- Granular static shard selection and validation.
- Expert-specific projected lookup.
- Preservation of all static STEM tables during checkpoint conversion.
- Checkpoint missing-key prefixes.
- STEM optimizer grouping.

The integration registry contains four-GPU FSDP plus EP cases for static
top-1, static E2G2T2, and projected top-1 debug models.

Dedicated Greenland wrappers cover STAPLE12 E8G8T8 warmup and midfine in
addition to the existing E8G1T1 static/projected jobs.

Runtime unit, GPU integration, deterministic parity, and convergence tests
still require TorchTitan's supported PyTorch environment.
