# L9 — Distributed training

> **Milestone:** [L9](../course/ROADMAP.md#milestones) · **Builds:** P13 (tiny multiprocess equivalence and restart test) in [projects](../course/PROJECTS.md) · **Prerequisites:** [L8](08-efficiency.md) for memory budgets and the roofline, [L3](03-pretraining.md) for checkpoint and resume, [L1](01-neural-learning.md) for gradients and AdamW

When one device cannot hold the model or cannot process enough tokens in the time available, training is split across several processes ("ranks"). Every scheme in this module splits one of three things — the data, the model's stored state, or the computation inside a layer — and pays for it in communication. The skill is to keep the mathematics of a single-process step unchanged while it is spread out, to predict what the communication costs, and to prove equivalence on a small CPU setup before trusting any large run.

## What you should be able to do

- Define all-reduce, reduce-scatter, all-gather, broadcast and all-to-all, and compute the bytes each rank sends for each.
- Explain how DDP keeps replicas identical: initial broadcast, gradient all-reduce, bucketing and overlap with backward.
- Normalize the loss correctly when ranks see different numbers of tokens, and show with numbers what goes wrong otherwise.
- State what ZeRO stages 1, 2 and 3 and FSDP shard, when parameters are gathered, and the resulting memory per rank.
- Split a linear layer by columns or by rows for tensor parallelism, and say where the collective goes.
- Compute the pipeline bubble for a given number of stages and micro-batches.
- Describe sequence/context parallelism and expert parallelism (all-to-all routing) at the level of what is split and what is exchanged.
- List the state a distributed checkpoint must hold, and explain how a restart can silently go wrong.
- Design a CPU (gloo) test that compares a multiprocess update with a single-process baseline, and say what such a test cannot show about GPU scaling.

## Core ideas

### 1. A map of the strategies

| Strategy | What is split | What each rank holds | Main communication |
|---|---|---|---|
| Data parallel (DDP) | the batch | full model, full optimizer | all-reduce of gradients |
| ZeRO / FSDP | the batch, plus model states | a shard of optimizer state (and gradients, and parameters) | reduce-scatter of gradients, all-gather of parameters |
| Tensor parallel | each layer's weight matrices | a slice of every layer | all-reduce (or reduce-scatter + all-gather) of activations, inside every layer |
| Pipeline parallel | the stack of layers | a contiguous group of layers | point-to-point send/receive of activations between stages |
| Sequence / context parallel | the sequence dimension | some positions of every sequence | exchange of keys/values or all-to-all |
| Expert parallel | the experts of an MoE layer | some experts | all-to-all of routed tokens |

Large runs combine several. The total number of ranks is the product of the degrees, e.g. 64 ranks = 4-way tensor × 2-way pipeline × 8-way data parallel. A common placement keeps the most communication-heavy split (tensor parallel) on ranks joined by the fastest links; the right layout depends on measured topology.

### 2. Collective operations and their cost

A **collective** is an operation that every rank in a group must call, in the same order. Let $W$ be the number of ranks and $M$ the size in bytes of each rank's buffer.

| Collective | Before | After | Bytes sent per rank (ring algorithm) |
|---|---|---|---|
| Broadcast | root has $x$ | every rank has $x$ | ≈ $M$ (pipelined) |
| Reduce-scatter | rank $r$ has $x_r$ | rank $r$ has chunk $r$ of $\sum_r x_r$ | $\frac{W-1}{W} M$ |
| All-gather | rank $r$ has chunk $r$ (size $M/W$) | every rank has all chunks | $\frac{W-1}{W} M$ |
| All-reduce | rank $r$ has $x_r$ | every rank has $\sum_r x_r$ | $2\frac{W-1}{W} M$ |
| All-to-all | rank $r$ has a distinct chunk for each other rank | rank $r$ has the chunks addressed to it | $\frac{W-1}{W} M$ |

**All-reduce = reduce-scatter + all-gather.** In a ring, split the buffer into $W$ chunks. In $W-1$ steps, each rank passes one chunk to its neighbor and adds what it receives; afterwards each rank owns one fully reduced chunk. In $W-1$ more steps, the reduced chunks circulate until everyone has all of them.

Tiny example with $W = 3$ and three-element buffers: rank 0 has $[1, 2, 3]$, rank 1 $[4, 5, 6]$, rank 2 $[7, 8, 9]$. After reduce-scatter, rank 0 holds 12 (first chunk), rank 1 holds 15, rank 2 holds 18. After all-gather, every rank holds $[12, 15, 18]$; dividing by $W$ gives the average $[4, 5, 6]$.

**Cost model.** A simple model charges a latency $\alpha$ per message step and $1/\beta$ seconds per byte:

$$t_{\text{all-reduce}} \approx 2(W-1)\,\alpha + 2\frac{W-1}{W}\frac{M}{\beta}.$$

The bandwidth term barely grows with $W$, which is why ring all-reduce scales; the latency term grows linearly, which is why libraries also use tree algorithms for small messages and many ranks.

**Worked example (predicted).** A 1.5 B-parameter model with bf16 gradients has $M = 3$ GB. With $W = 8$, each rank sends $2 \cdot \tfrac{7}{8} \cdot 3 = 5.25$ GB per step. At an assumed effective bandwidth of 100 GB/s per rank, that is ≈ 52 ms, ignoring latency. Whether that matters depends on whether it overlaps with a backward pass that takes, say, 300 ms.

**Backends.** PyTorch's `torch.distributed` offers **NCCL** (NVIDIA GPUs) and **gloo** (CPU, and a fallback for GPUs). On CPU, gloo gives you the same semantics for the main collectives, which is what P13 needs — but not the same performance.

### 3. Data parallelism and DDP

In **data parallelism** every rank holds a full replica, processes a different slice of the global batch, and must end each step with identical parameters. If every rank applies the same optimizer step to the same parameters with the same gradient, they stay identical; so the job is to give every rank the **average gradient** over the global batch.

PyTorch's `DistributedDataParallel` (DDP) does this as follows:

1. **At construction** it broadcasts parameters and buffers from rank 0, so all replicas start identical even if their random initializations differed.
2. **During backward**, a hook fires as each parameter's gradient becomes ready. Gradients are grouped into **buckets** of a few tens of MB (`bucket_cap_mb` defaults to 25 MiB in current PyTorch 2.x source; check the pinned release). When every gradient in a bucket is ready, DDP launches an asynchronous all-reduce for that bucket.
3. **Overlap:** because buckets are filled roughly in reverse layer order — the order backward produces them — communication for late layers runs while backward is still computing early layers.
4. After backward, DDP waits for all buckets and divides by $W$, so `.grad` holds the average. Each rank then runs its own, identical, optimizer step.

Bucketing matters because one all-reduce per parameter would pay the latency $\alpha$ hundreds of times; one giant all-reduce at the end would forfeit overlap.

**Practical details.**

- **Sharding the data.** Each rank must see different examples. `DistributedSampler` assigns indices by rank; call `set_epoch` each epoch or every epoch reuses the same shuffle. Unless `drop_last=True`, it pads the index list by repeating examples so every rank gets the same count — a few examples are then seen twice per epoch.
- **Accumulation.** Wrap all but the last micro-batch in `model.no_sync()` so DDP accumulates locally and all-reduces once.
- **Unused parameters.** If some parameters receive no gradient in a step (a branch not taken), DDP waits for them forever unless told to look for unused parameters, which costs extra work.
- **Uneven numbers of batches.** If one rank runs out of data, the others block at their next all-reduce: the job hangs. Make shard lengths equal or use PyTorch's join utilities.

### 4. Loss normalization with uneven tokens across ranks

DDP averages gradients across ranks with equal weight. If each rank's loss is its **local** token mean, the global objective becomes the mean of per-rank means — which is not the token mean over the global batch when ranks have different numbers of counted tokens (after padding and loss masks). This is the [L8 accumulation trap](08-efficiency.md#7-gradient-accumulation-and-the-normalization-subtlety) in another form.

**Worked example.** Two ranks, intended objective: mean over all counted tokens in the step.

| Rank | Counted tokens $n_r$ | Sum of token losses $S_r$ | Local mean |
|---|---|---|---|
| 0 | 600 | 1500 | 2.5 |
| 1 | 200 | 300 | 1.5 |

- Single-process baseline on all 800 tokens: $(1500 + 300) / 800 = 2.25$.
- Naive DDP (local means, then averaged): $(2.5 + 1.5) / 2 = 2.0$.
- Per-token gradient weight under the naive scheme: rank 0 tokens get $\tfrac{1}{2} \cdot \tfrac{1}{600} = \tfrac{1}{1200}$; rank 1 tokens get $\tfrac{1}{2} \cdot \tfrac{1}{200} = \tfrac{1}{400}$. A rank-1 token counts three times as much as a rank-0 token, instead of every token getting $\tfrac{1}{800}$.

**Correct normalization.** Before backward, all-reduce the token counts to get the global count $n = \sum_r n_r$ (a single integer, not differentiated). Each rank uses

$$\mathcal{L}_r = \frac{W \cdot S_r}{n}.$$

Then DDP's division by $W$ yields gradients of $\tfrac{1}{n}\sum_r S_r$, the global token mean. With the numbers above: $\mathcal{L}_0 = 2 \cdot 1500 / 800 = 3.75$ and $\mathcal{L}_1 = 2 \cdot 300 / 800 = 0.75$; their average is 2.25, matching the baseline. The logged per-rank losses look strange; log the all-reduced sum and count instead.

With accumulation, $n$ must count tokens across all micro-batches *and* all ranks in the optimizer step. Whether a given trainer does this by default is version-dependent; check the pinned source rather than assuming.

### 5. ZeRO and FSDP: sharding model states

DDP replicates everything, so its memory per rank is the full [L8 model-state budget](08-efficiency.md#4-the-training-memory-budget). With mixed-precision AdamW and $\Psi$ parameters: 16-bit parameters $2\Psi$, 16-bit gradients $2\Psi$, and $K\Psi$ bytes of optimizer state with $K = 12$ (fp32 master weights, $m$, $v$). **ZeRO** (Zero Redundancy Optimizer) removes the redundancy step by step:

| Stage | Sharded across $W$ ranks | Model-state bytes per rank | Communication vs DDP |
|---|---|---|---|
| DDP | nothing | $16\Psi$ | all-reduce of gradients ($2\Psi$-byte buffer) |
| ZeRO-1 | optimizer states | $4\Psi + 12\Psi/W$ | same volume |
| ZeRO-2 | + gradients | $2\Psi + 14\Psi/W$ | same volume |
| ZeRO-3 | + parameters | $16\Psi/W$ | about 1.5× |

**Worked example (predicted).** $\Psi = 1.5$ B, $W = 8$: DDP 24 GB; ZeRO-1 $6 + 2.25 = 8.25$ GB; ZeRO-2 $3 + 2.63 = 5.63$ GB; ZeRO-3 3 GB. Activations, temporary buffers and the full parameters of whatever unit is currently gathered come on top; ZeRO does not shard activations.

**Why stages 1–2 cost no extra communication.** An all-reduce is a reduce-scatter followed by an all-gather. ZeRO-2 stops after the reduce-scatter: each rank receives only its shard of the summed gradient, updates only its shard of the parameters, then all-gathers the updated parameters. Same bytes, different order.

**When ZeRO-3 / FSDP gathers.** With parameters sharded, a layer's full weights must be assembled before use:

1. Forward of unit $i$: all-gather its parameters, compute, then free the gathered copy (or keep it until backward, trading memory for one less all-gather).
2. Backward of unit $i$: all-gather again, compute gradients, reduce-scatter them so each rank receives its gradient shard, free the full copy.
3. Optimizer step: each rank updates its own shard of fp32 master weights and moments.

Libraries **prefetch** the next unit's all-gather while the current unit computes, so communication overlaps compute when there is enough work per unit.

**FSDP** is PyTorch's implementation of this idea. Its sharding strategies map onto ZeRO stages (full sharding ≈ ZeRO-3; sharding gradients and optimizer state only ≈ ZeRO-2; hybrid sharding shards within a node and replicates across nodes). The **wrapping unit** — usually one Transformer block — sets the granularity of gathering: too coarse and peak memory is high; too fine and many small collectives pay latency. There are two API generations (the original `FullyShardedDataParallel` wrapper and the newer per-parameter `fully_shard`); their checkpoint formats and mixed-precision options differ, so pin the version. DeepSpeed implements the ZeRO stages with its own configuration; the [Accelerate comparison](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed) is scoped to single-node use.

### 6. Tensor parallelism

**Tensor parallelism** splits individual weight matrices across $t$ ranks so each rank does part of every layer.

For $Y = XA$ with $X$ of shape $[\cdot, d_{in}]$ and $A$ of shape $[d_{in}, d_{out}]$:

- **Column split:** $A = [A_1, A_2]$ along output columns. Rank $i$ computes $Y_i = XA_i$, needing the full $X$, and holds a slice of the output columns.
- **Row split:** $A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}$ along input rows, with $X = [X_1, X_2]$ split to match. Rank $i$ computes the partial sum $X_i A_i$; the result is $Y = X_1 A_1 + X_2 A_2$, which needs an **all-reduce**.

Tiny example: $X = [1, 2]$, $A = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$, so $XA = [5, 11]$. Column split: rank 0 has the first column and computes 5; rank 1 computes 11; concatenating gives $[5, 11]$. Row split: rank 0 computes $1 \cdot [1, 3] = [1, 3]$; rank 1 computes $2 \cdot [2, 4] = [4, 8]$; their sum is $[5, 11]$.

**MLP block (Megatron-style).** Make the first linear layer column-parallel and the second row-parallel. The elementwise activation (or gate) acts on each rank's own columns, so no communication is needed between the two layers; one all-reduce produces the block output.

| Tensor | Shape per rank ($t$-way) |
|---|---|
| Input $X$ (replicated) | `[B, T, d]` |
| First weight $A_i$ | `[d, d_ff / t]` |
| Hidden $H_i = \text{act}(XA_i)$ | `[B, T, d_ff / t]` |
| Second weight $C_i$ | `[d_ff / t, d]` |
| Partial output $H_i C_i$ | `[B, T, d]` |
| After all-reduce | `[B, T, d]` (replicated) |

**Attention** splits by heads: the Q/K/V projections are column-parallel, each rank runs $h/t$ heads independently, and the output projection is row-parallel followed by an all-reduce. This needs $h$ (and the number of KV heads, under grouped-query attention) divisible by $t$, or replication of KV heads.

**Cost.** Per layer, forward needs two all-reduces of a $[B, T, d]$ activation and backward two more. Predicted example: $B = 4$, $T = 4096$, $d = 4096$, bf16 gives 134 MB per all-reduce; with $t = 4$ each rank sends $2 \cdot \tfrac{3}{4} \cdot 134 \approx 201$ MB per all-reduce, ≈ 0.8 GB per layer per step. This traffic sits on the critical path of every layer, which is why tensor parallelism is usually confined to ranks with very fast interconnect.

### 7. Pipeline parallelism

**Pipeline parallelism** places consecutive groups of layers ("stages") on different ranks. Stage $s$ sends its output activation, shape `[b, T, d]` for a micro-batch of $b$ sequences, to stage $s+1$; backward sends gradients the other way.

With one batch, only one stage works at a time. Splitting the batch into $m$ **micro-batches** lets stages work on different micro-batches concurrently. In a simple schedule (all forwards, then all backwards), with $p$ stages and per-micro-batch stage times $t_f$ and $t_b$:

$$\text{ideal time} = m\,(t_f + t_b), \qquad \text{bubble} = (p-1)(t_f + t_b), \qquad \frac{\text{bubble}}{\text{ideal}} = \frac{p-1}{m}.$$

| $p$ | $m$ | Bubble / ideal | Idle share of total, $\frac{p-1}{m+p-1}$ |
|---|---|---|---|
| 4 | 8 | 37.5% | 27% |
| 4 | 32 | 9.4% | 8.6% |
| 8 | 32 | 21.9% | 18% |

More micro-batches shrink the bubble but, in the all-forward-first schedule, stage 1 must hold activations for all $m$ micro-batches. The **1F1B** schedule alternates one forward and one backward once the pipeline is full, so each stage holds at most about $p$ micro-batches' activations; the bubble is unchanged. **Interleaved** schedules give each rank several non-adjacent chunks of layers, dividing the bubble by the number of chunks per rank at the cost of more point-to-point messages.

Micro-batches here are gradient accumulation, so the token-count normalization of §4 applies to them too.

### 8. Sequence and context parallelism

Activations scale with $T$, and attention compute with $T^2$, so long sequences eventually exceed one device even with recomputation.

- **Sequence parallelism** (in the Megatron sense) works alongside tensor parallelism: regions such as normalization and dropout, which tensor parallelism would otherwise replicate, are split along the sequence dimension. The all-reduce becomes a reduce-scatter plus an all-gather of the same total volume, and those regions' activation memory is divided by $t$.
- **Context parallelism** splits every sequence across ranks for the whole layer. Projections are local, but each query needs keys and values from all earlier positions. **Ring Attention** passes K/V blocks around a ring of ranks while each rank accumulates its queries' outputs with the [online softmax](08-efficiency.md#10-flashattention-tiling-online-softmax-io-awareness), overlapping transfer with compute. An alternative uses all-to-all to switch from a sequence-split to a head-split layout for the attention itself. Causal masking makes naive splits unbalanced (later positions do more work), so implementations reorder chunks.

### 9. Expert parallelism

In a mixture-of-experts layer ([L10](10-frontier.md#1-dense-versus-mixture-of-experts)), a router sends each token to $k$ of $E$ expert MLPs. **Expert parallelism** places different experts on different ranks:

1. Each rank routes its local tokens.
2. **All-to-all dispatch:** each token's hidden vector is sent to the ranks holding its chosen experts.
3. Each rank runs its experts on the tokens it received.
4. **All-to-all combine:** outputs return to the tokens' home ranks, where they are weighted by the router gates and summed.

**Worked example (predicted).** 8 ranks, 8,192 tokens per rank per MoE layer, top-2, $d = 2048$, bf16: the dispatch buffer per rank is $8192 \cdot 2 \cdot 2048 \cdot 2 \approx 67$ MB; with uniform routing about $7/8$ of it leaves the rank. Combine moves the same amount back, and backward repeats both.

Two problems are specific to this layout: the all-to-all sits on the critical path of every MoE layer, and **load imbalance** makes the busiest rank set the pace — if one rank's experts receive twice the average tokens, every other rank waits. Capacity limits and balancing losses (L10) exist partly for this systems reason.

### 10. Distributed checkpointing and recovery

A checkpoint must let a restarted job continue *as if it had never stopped*. That means saving:

| State | Why |
|---|---|
| Parameters (all shards) | the model |
| Optimizer state (all shards: master weights, moments, step counts) | without it, AdamW restarts from zero moments and the trajectory changes |
| LR scheduler and gradient scaler state | schedule position; current fp16 loss scale |
| Step and token counters | logging, schedule, stopping rule |
| Data-loader position per rank (epoch, sampler state, consumed samples) | otherwise examples are skipped or repeated |
| RNG states per rank (CPU and device) | dropout masks and shuffles reproduce |
| Configuration: world size, parallel layout, versions | to detect or handle a changed layout |

With sharded states, each rank writes its own shards plus shared metadata describing how they fit together. PyTorch's `torch.distributed.checkpoint` does this and can load into a different sharding layout (check the pinned version for what resharding it supports). Write to a temporary location and mark a checkpoint complete only after every rank has finished, so a crash mid-save never leaves a "latest" checkpoint that is partial.

**Recovery.** When one rank fails, collectives on the others hang or error, so the whole group restarts — typically under a launcher such as `torchrun` with restart limits — and every rank loads the last *complete* checkpoint. Silent failure modes are worse than crashes: a resume that restores parameters but not optimizer moments, or not the data position, trains without error but no longer matches the uninterrupted run. That is why P13 compares a resumed run with an uninterrupted one.

### 11. Testing equivalence against a single-process baseline (test design)

P13 asks you to *show* that a multiprocess update equals a single-process one. This section describes the design; the implementation is yours.

**Fixture.**

- A tiny model in float32 (float64 makes tolerances tighter if needed), no dropout, fixed seed, run on CPU with a fixed thread count.
- A fixed global batch whose examples have *different* numbers of counted tokens, so normalization bugs are visible.
- A deterministic split of that batch into per-rank shards, written explicitly rather than through a sampler that might pad or shuffle.

**Baseline.** One process runs the global batch through one (then several) optimizer steps and records loss, gradients and parameters.

**Distributed run.** Spawn $W$ processes on one machine, initialize a process group with the gloo backend and a local rendezvous, confirm that all ranks start from identical parameters (compare a checksum across ranks), give each rank its shard, compute the loss with the §4 normalization, and take the same step.

**Assertions.**

| Check | Expected |
|---|---|
| Parameters across ranks after the step | identical to each other |
| Parameters vs baseline | equal within a stated tolerance; summation order differs, so not bit-exact |
| Gradients vs baseline before the step | equal within tolerance — locates a failure before the optimizer hides it |
| Same test with $W = 2$ and $W = 3$ | both pass, catching anything hard-coded to two ranks |
| Negative control: naive per-rank mean | fails with uneven tokens, passes with equal tokens — evidence the test can detect the bug |
| Accumulation variant | $k$ micro-batches per rank with `no_sync` matches the baseline |

**Restart variant.** Run $S$ steps uninterrupted. Separately run $s < S$ steps, save the full state of §10, terminate the processes, start fresh processes, load and run the remaining $S - s$. Compare final parameters. Add a negative control that omits the optimizer state and confirm the comparison catches it.

**Hygiene.** Use timeouts so a hang fails the test instead of blocking it. Record Python, PyTorch and backend versions with the result.

### 12. What a CPU demonstration does and does not show

| A gloo-on-CPU test can show | It cannot show |
|---|---|
| The update rule is mathematically equivalent to single-process training | Speedup, scaling efficiency or throughput on GPUs |
| Loss normalization and accumulation are correct | Whether communication overlaps compute under NCCL |
| Checkpoint/restart restores the trajectory | Interconnect bandwidth, topology effects or stragglers at scale |
| Collective semantics (what each rank holds after each call) | Memory savings measured on device |

Communication-volume numbers in this note are predictions from the cost model. A GPU scaling claim needs measured step times at several world sizes on stated hardware and software.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Job hangs at the first or a later collective | Ranks call collectives in different orders or numbers; one rank ran out of data; rendezvous misconfigured | Log each rank's collective sequence; check shard lengths; add a timeout |
| Distributed loss differs from the single-process baseline | Per-rank mean normalization; sampler padding duplicated examples; different initial weights; dropout RNG | Compare per-rank token counts and parameter checksums; disable dropout; split data explicitly |
| Every rank effectively trains on the same data | No sharding, or the same seed used for per-rank shuffles | Print the first example IDs per rank |
| Shuffle identical every epoch | `set_epoch` not called on the distributed sampler | Compare epoch-1 and epoch-2 order |
| DDP errors about parameters that did not receive gradients | A branch not used this step | Check which parameters have `grad is None`; restructure, or enable unused-parameter detection |
| FSDP gives little memory reduction | Wrapping unit too coarse; full parameters kept after forward; activations dominate | Inspect wrapping; profile peak memory per phase |
| Throughput barely improves with more GPUs | Communication not overlapped; small buckets or units; slow interconnect; load imbalance | Profile a timeline per rank; compare compute and communication time |
| Loss jumps after resume | Optimizer or scheduler state not restored; data position lost; loss scale reset | Compare resumed and uninterrupted runs step by step |
| Resume fails after changing the number of GPUs | Checkpoint stored per-rank shards without resharding metadata | Use a sharding-aware checkpoint format and test a reshard |

## Check yourself

1. Show by counting chunks why a ring all-reduce sends $2\frac{W-1}{W}M$ bytes per rank. What does the latency term look like, and when does it dominate?
2. In DDP, what guarantees that replicas stay identical after many steps? List every assumption.
3. Recompute the §4 example with three ranks holding 500, 250 and 50 tokens and loss sums of your choosing. Which tokens are over-weighted by the naive scheme, and by how much?
4. Predict, then run (once your P13 harness exists): with equal token counts per rank, does the naive normalization pass the equivalence test? Explain why that makes it a weak test case.
5. For your own model's parameter count, compute model-state memory per rank under DDP and ZeRO-1/2/3 with $W = 4$. Which stage would you choose if activations were the limit, and why would ZeRO not help?
6. Why does ZeRO-2 need no more communication than DDP, while ZeRO-3 needs about 1.5 times as much?
7. Split a $[3, 4]$ weight matrix of your own between two ranks by columns and by rows; compute both results by hand and identify the collective each needs.
8. With 6 pipeline stages, how many micro-batches keep the bubble below 10% of ideal time? What does that choice cost in activation memory under each schedule?
9. In expert parallelism, what happens to step time when one expert receives four times its share of tokens? Which mechanism from L10 addresses this?
10. List the state your P13 restart test must save. For each item, describe the symptom if it were omitted.
11. Which of your P13 results could support a GPU scaling claim? Write the measurement you would need instead.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [PyTorch distributed overview and collectives](https://docs.pytorch.org/docs/stable/distributed.html) | Backends, collective functions, initialization | Exact semantics for gloo/NCCL in your pinned version |
| [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | The collectives page | Diagrams of each collective's before/after state |
| Patarasuk and Yuan, "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations" (JPDC, 2009) | Ring algorithm (optional) | Why the ring is bandwidth-optimal |
| [Li et al., PyTorch Distributed: Experiences on Accelerating Data Parallel Training (2020)](https://arxiv.org/abs/2006.15704) | System design: bucketing, overlap, unused parameters | How DDP is built |
| [PyTorch DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html) | Whole note | Construction broadcast, hooks and buckets |
| [Rajbhandari et al., ZeRO (2019)](https://arxiv.org/abs/1910.02054) | Figure 1; §5 (ZeRO-DP stages) and §7 (communication analysis) | Stage-by-stage memory and communication |
| [Zhao et al., PyTorch FSDP (2023)](https://arxiv.org/abs/2304.11277) | Design sections on sharding, prefetching and wrapping | How FSDP gathers and frees parameters |
| [Accelerate: FSDP vs DeepSpeed](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed) | Whole guide | Configuration and precision differences (single-node scope) |
| [Shoeybi et al., Megatron-LM (2019)](https://arxiv.org/abs/1909.08053) | §3, model-parallel Transformers | Column/row splits for MLP and attention |
| [Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (2021)](https://arxiv.org/abs/2104.04473) | §2.2 pipeline schedules and bubble analysis; combining parallelisms | 1F1B, interleaving and layout trade-offs |
| [Huang et al., GPipe (2018)](https://arxiv.org/abs/1811.06965) | Micro-batch pipelining (optional) | The original synchronous pipeline |
| [Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models (2022)](https://arxiv.org/abs/2205.05198) | §4.2.2 sequence parallelism | Splitting non-tensor-parallel regions along $T$ |
| [Liu, Zaharia and Abbeel, Ring Attention (2023)](https://arxiv.org/abs/2310.01889) | Method section (optional) | Context parallelism by rotating K/V blocks |
| [Lepikhin et al., GShard (2020)](https://arxiv.org/abs/2006.16668) and [Fedus et al., Switch Transformers (2021)](https://arxiv.org/abs/2101.03961) | Switch §2.2 (capacity) and §5 (expert parallelism); GShard's expert-capacity description and §3 implementation | All-to-all routing and its constraints |
| [PyTorch distributed checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) and [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) | API pages for the pinned version | Sharded save/load, resharding and restarts |
| [CS336](https://cs336.stanford.edu/) | Parallelism lectures and systems assignment | Course-length treatment |
