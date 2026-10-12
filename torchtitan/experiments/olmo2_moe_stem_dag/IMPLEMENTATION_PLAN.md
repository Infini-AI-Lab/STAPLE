# OLMo2 MoE STEM-DAG Design

## Scope

Static STEM supports granular experts and top-k routing. For granularity `G`,
the full static row is split into `G` contiguous expert-width shards, and
expert `e` consumes shard `e % G`. Projected per-expert STEM remains limited
to granularity 1 and top-1 routing.

Let:

- `D = 2048` be the transformer hidden dimension.
- `H = 8192` be the dense FFN and full static STEM width.
- `V = 100352` be the vocabulary size.
- `N` be the total number of routed experts.
- `G` be the static STEM granularity.
- `T` be router top-k.
- `F = H/G` be the granular expert width.
- `R = 8 ceil(H / (8N))` be the projected per-expert STEM dimension.

For token `v_i`, hidden state `x_i`, and selected expert `n_i`:

```math
h_i =
\operatorname{SiLU}(W_{1,n_i}x_i)
\odot
\left(W_{3,n_i}x_i+s_i\right),
```

```math
y_i=p_iW_{2,n_i}h_i.
```

## Static STEM

Each STEM layer owns one table:

```math
S_l \in \mathbb{R}^{V \times H}.
```

The selected expert receives its granularity shard:

```math
s_i=\operatorname{reshape}(S_l[v_i], [G,F])[n_i\bmod G].
```

For `G=1`, this reduces to the complete row. For `L_stem` layers:

```math
P_{\mathrm{static}}=L_{\mathrm{stem}}VH.
```

## Projected Per-Expert STEM

Each STEM layer owns:

```math
S_l \in \mathbb{R}^{N \times V \times R},
\qquad
U_l \in \mathbb{R}^{H \times R}.
```

After top-1 routing:

```math
s_i=U_lS_l[n_i,v_i,:].
```

`U_l` is bias-free and shared by all experts in that layer. The additional
parameter count is:

```math
P_{\mathrm{projected}}
=L_{\mathrm{stem}}(NVR+HR).
```

When `H/N` is already a multiple of eight, `NVR=VH`. The projected strategy
then has the same aggregate embedding-table size as static STEM plus one
`H x R` projection per layer.

The expert tables use normal initialization with standard deviation
`1/sqrt(R)`. The shared projection is initialized to zero, so the initial
model output matches the corresponding base MoE.

## Storage

For `b` bytes per parameter:

```math
M_{\mathrm{static}}=bL_{\mathrm{stem}}VH,
```

```math
M_{\mathrm{projected}}
=bL_{\mathrm{stem}}(NVR+HR).
```

For 12 E8G1T1 STEM layers, `N=8` and `R=1024`:

| Strategy | Additional parameters | BF16 | FP32 |
|---|---:|---:|---:|
| Static | 9,865,003,008 | 18.375 GiB | 36.750 GiB |
| Projected per-expert | 9,965,666,304 | 18.562 GiB | 37.125 GiB |

## Focus Configurations

The existing static/projected E8G1T1 configurations derive from the
corresponding E8G1T1-L8 warmup or midfine function in the OLMo2 MoE registry:

- Static warmup
- Static midfine
- Projected per-expert warmup
- Projected per-expert midfine

- Flavor: `1B-E8G1T1-8L`
- Experts: 8
- Routing: top-1
- Expert width: 8192
- MoE and STEM layers: 1, 3, 5, 7, 9, 11, 13, 15
- Projected STEM rank: 1024

| Base parameters | Static total | Projected total |
|---:|---:|---:|
| 4,303,620,096 | 10,880,288,768 | 10,947,397,632 |

The STAPLE upcycling configuration is:

- Flavor: `1B-E8G8T8-8L-STEM12`
- Experts: 64
- Routing: top-8
- Expert width: 1024
- STEM layers: `1..12`
- MoE layers: `1,3,5,7,9,11,13,15`
- Total parameters: 14,169,540,608

## Distributed Layout

Static tables are feature-sharded over EP. Their lookup reconstructs the full
`H`-wide row for each local token before routing it to the selected expert.

Projected tables follow expert ownership on EP/EDP. Token IDs follow the
activation dispatch metadata, and the receiving rank performs the local
expert lookup. The shared projection remains on the dense DP/FSDP mesh.

Both standard and deduplicated all-to-all preserve matching activation,
token-ID, and routing-score ordering.
