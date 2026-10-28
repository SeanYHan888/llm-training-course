# L8 — Training efficiency

> **Milestone:** [L8](../course/ROADMAP.md#milestones) · **Builds:** P12 (resource budget and one measured optimization) in [projects](../course/PROJECTS.md); the measurement half of [Inference 01 Lab B](../exercises/inference/README.md) · **Prerequisites:** [L2](02-transformer.md) for the model's tensors, [L3](03-pretraining.md) for the training loop, [L1](01-neural-learning.md) for AdamW state

Training a language model is a budget problem: a fixed amount of arithmetic, memory and time must cover a given number of tokens. This module teaches you to count what a training step costs before running it, to find out what it actually costs by measuring it correctly, and to change one thing (precision, batch layout, recomputation, an attention kernel) without changing the mathematics you meant to compute. Throughout, keep three kinds of statement apart: **predicted** (from a formula and stated assumptions), **measured** (on a named device and software stack) and **illustrated** (a CPU or toy run that demonstrates a mechanism but not GPU performance).

## What you should be able to do

- Count the FLOPs of a Transformer training step with the $6N$ rule, add the attention term, and convert the count into a predicted step time and model FLOPs utilization.
- Compute the arithmetic intensity of an operation and place it on a roofline to predict whether it is compute-bound or memory-bound.
- Describe the GPU memory hierarchy (registers, shared memory/SRAM, L2, HBM) and explain why moving bytes, not doing arithmetic, often sets the speed.
- Derive a training memory budget — weights, gradients, optimizer states, activations, temporary buffers — symbolically, then for a concrete configuration, with every assumption stated.
- Explain fp32, fp16, bf16 and fp8 in terms of exponent and mantissa bits; explain loss scaling and why mixed precision keeps fp32 master weights.
- Implement gradient accumulation without changing the loss normalization, and use activation recomputation while showing that gradients are unchanged.
- Profile a training step correctly (warmup, synchronization, repeated trials) and read a `torch.profiler` table or trace.
- Explain FlashAttention's tiling and online softmax, what kernel fusion saves, and what Triton is for.
- For one optimization, state the predicted effect, measure the actual effect, and show that correctness was preserved.

## Core ideas

### 1. Counting FLOPs

A **FLOP** here is one floating-point add or multiply. A matrix product of an $[M, K]$ matrix with a $[K, N]$ matrix does $MN$ dot products of length $K$, each a multiply and an add per term, so it costs $2MKN$ FLOPs.

**Forward pass.** In a dense decoder almost every parameter belongs to a weight matrix that each token multiplies once. If the model has $N$ parameters, the forward pass costs about $2N$ FLOPs per token (one multiply and one add per weight).

**Backward pass.** For a linear layer $Y = XW$, backward needs two products of the same size as the forward one: $\partial \mathcal{L}/\partial X = (\partial \mathcal{L}/\partial Y) W^\top$ and $\partial \mathcal{L}/\partial W = X^\top (\partial \mathcal{L}/\partial Y)$. That is about $4N$ FLOPs per token, so a full training step is

$$\text{FLOPs per token} \approx 6N, \qquad \text{FLOPs per step} \approx 6N \cdot B \cdot T,$$

where $B$ is sequences per step and $T$ is tokens per sequence. The approximation ignores normalization, activations, softmax and the optimizer update, which are small in FLOPs (though not in memory traffic — see §2).

**Two details that change the count.**

- *Embeddings.* An embedding lookup is indexing, not a matrix product, so it costs almost nothing. The output projection to $V$ logits is a real $[d, V]$ product. With tied input/output weights, count those $Vd$ parameters once and the rule still holds.
- *Attention scores.* $QK^\top$ and the weighted sum over $V$ have no parameters, so $6N$ misses them. Per layer and per token, the scores cost $2Td$ FLOPs and the weighted sum another $2Td$ (summing over heads, $h \cdot d_h = d$). Forward: $4Td$ per layer per token; training (×3): $12Td$. A kernel that skips fully masked blocks of a causal mask does about half of this.

Comparing the attention term with the weight term for a block with $N_{layer} \approx 12d^2$ parameters (attention projections $4d^2$, a 4×-wide MLP $8d^2$):

$$\frac{12\,T d}{6 \cdot 12 d^2} = \frac{T}{6d}.$$

Attention FLOPs become significant when $T$ is comparable to $6d$. Short-context, wide models are dominated by matrix products; long-context training is not.

**Worked example (my own configuration, used through this note).** A decoder with $L = 24$ layers, $d = 1024$, $h = 8$ heads ($d_h = 128$), a 4×-wide MLP, $V = 50{,}000$, tied embeddings; batch $B = 8$, sequence $T = 2048$.

| Quantity | Calculation | Value |
|---|---|---|
| Block parameters | $24 \times 12 \times 1024^2$ | 302.0 M |
| Embedding/head parameters | $50{,}000 \times 1024$ | 51.2 M |
| $N$ (ignoring norms, biases) | sum | ≈ 353.2 M |
| Tokens per step | $8 \times 2048$ | 16,384 |
| Weight FLOPs per step | $6 \times 353.2\text{M} \times 16{,}384$ | ≈ $3.47 \times 10^{13}$ |
| Attention FLOPs per step (no masking savings) | $12 \times 24 \times 2048 \times 1024 \times 16{,}384$ | ≈ $9.9 \times 10^{12}$ |
| Total | | ≈ $4.5 \times 10^{13}$ |

Here attention adds about 28%.

**From FLOPs to time.** Given a device's peak throughput $C$ (FLOP/s) for the precision used, a predicted step time is $\text{FLOPs} / (u \cdot C)$, where $u$ is the fraction of peak you expect to achieve. **Model FLOPs utilization (MFU)** turns this around after a measurement:

$$\text{MFU} = \frac{\text{model FLOPs per step} \,/\, \text{measured step time}}{C}.$$

"Model FLOPs" are the FLOPs the mathematics requires; recomputation (§8) does extra work that MFU deliberately does not credit. On a *hypothetical* accelerator with $C = 400$ TFLOP/s and an assumed $u = 0.4$, the step above is predicted to take $4.5 \times 10^{13} / 1.6 \times 10^{14} \approx 0.28$ s. That number is a prediction from two assumptions ($C$ and $u$), not a measurement of anything.

*In code this shows up as* the resource-budget part of P12: write the count from your own model's actual shapes, then compare with a measured step time.

### 2. Arithmetic intensity and the roofline

An operation's **arithmetic intensity** is FLOPs performed per byte moved between main memory and the compute units:

$$I = \frac{\text{FLOPs}}{\text{bytes read} + \text{bytes written}}.$$

A device has a peak compute rate $C$ and a memory bandwidth $W$ (bytes/s). An operation cannot run faster than either limit, so its attainable throughput is

$$\text{attainable FLOP/s} = \min(C,\; I \cdot W).$$

Plotted against $I$ on log axes this is the **roofline**: a sloped line (memory-bound) meeting a flat ceiling (compute-bound) at the **ridge point** $I^* = C / W$. The [inference note](inference-serving.md#2-why-decode-is-memory-bandwidth-bound) uses the same model for decoding.

For a matrix product with $s$-byte elements, assuming each operand is read once and the output written once:

$$I_{\text{matmul}} = \frac{2MKN}{s\,(MK + KN + MN)}.$$

Worked examples on the same hypothetical device, $C = 400$ TFLOP/s and $W = 2$ TB/s, so $I^* = 200$ FLOP/byte; bf16 ($s = 2$):

| Operation | Shapes | FLOPs | Bytes | $I$ | Predicted regime |
|---|---|---|---|---|---|
| MLP up-projection in training | $[16384, 1024] \times [1024, 4096]$ | $1.37 \times 10^{11}$ | 176 MB | ≈ 780 | compute-bound |
| Same layer, one decode token | $[1, 1024] \times [1024, 4096]$ | $8.4 \times 10^{6}$ | 8.4 MB | ≈ 1 | memory-bound |
| Elementwise GELU on $[B, T, 4d]$ | $67.1$ M elements | a few per element | 268 MB | ≈ 1–5 | memory-bound |

The GELU does trivial arithmetic but must still read and write 268 MB, about 0.13 ms at 2 TB/s; the far larger matrix product needs about 0.34 ms of compute. A Transformer step contains dozens of such elementwise and normalization operations per layer, which is why reducing memory traffic (§10–11) matters even though they contribute almost no FLOPs.

**Limits of the model.** The roofline assumes perfect overlap, ideal caching and peak rates that real kernels rarely reach; matrix units also need particular shapes and dtypes to hit their peak. Use it to predict the regime and an upper bound, then measure.

### 3. The GPU memory hierarchy

A GPU is many **streaming multiprocessors (SMs)**, each running many threads. Data lives at several levels; capacities and bandwidths below are orders of magnitude only — read your device's datasheet and the CUDA programming guide for real values.

| Level | Scope | Capacity (order of magnitude) | What it is used for |
|---|---|---|---|
| Registers | one thread | a few hundred bytes per thread | operands of the current instruction |
| Shared memory / L1 (on-chip SRAM) | one SM, shared by a thread block | ~100 KB per SM | tiles that a block reuses many times |
| L2 cache | whole device | tens of MB | automatic caching of HBM traffic |
| HBM (device memory) | whole device | tens to hundreds of GB | weights, activations, optimizer state; TB/s bandwidth |
| Host memory | CPU side, over PCIe or a faster link | hundreds of GB | offload, data loading; much slower than HBM |

Each level down is larger and slower. A **kernel** (one GPU function launch) reads its inputs from HBM and writes outputs back. An efficient kernel loads a tile into SRAM once and reuses it for many operations; an inefficient sequence of kernels writes intermediate results to HBM only to read them back immediately. That one idea motivates tiling in matrix multiplication, kernel fusion and FlashAttention.

**CPU analogy, with care.** A CPU has registers, L1/L2/L3 caches and DRAM, and you can observe roofline-like effects there (a large matrix product achieves far more FLOP/s than a chain of small elementwise operations). The mechanism transfers; the numbers, the crossover points and the best kernel choices do not.

### 4. The training memory budget

During training, device memory holds five kinds of data:

| Component | Size depends on | Lives for |
|---|---|---|
| Weights | $N$, storage dtype | whole run |
| Gradients | $N$, gradient dtype | from backward to optimizer step (longer with accumulation) |
| Optimizer states | $N$; AdamW keeps two moments | whole run |
| Activations saved for backward | $B$, $T$, $d$, $L$, $h$, kernel choices | from forward to backward of that layer |
| Temporary buffers and overhead | largest single operation, allocator behavior, runtime context | transient |

**Model states, bytes per parameter.** For AdamW:

| Setup | Weights | Grads | fp32 master copy | Adam $m$ | Adam $v$ | Total |
|---|---|---|---|---|---|---|
| Plain fp32 | 4 | 4 | — | 4 | 4 | $16N$ |
| Mixed precision, 16-bit working weights + fp32 master | 2 | 2 | 4 | 4 | 4 | $16N$ |
| Autocast-style: fp32 parameters, 16-bit compute | 4 | 4 | — | 4 | 4 | $16N$ + transient 16-bit casts |

Mixed precision does **not** shrink model-state memory for AdamW; the second row is the accounting used in the ZeRO paper. Its savings come from 16-bit activations and faster matrix units. PyTorch's autocast stores parameters in fp32 and casts them inside each operation, which is the third row.

**Activations.** Backward needs tensors from the forward pass: inputs to each linear layer, normalization statistics, the MLP hidden states, attention probabilities if they are materialized, and dropout masks if dropout is used. Symbolically:

$$M_{act} \approx L \cdot \big(c_1 \cdot B T d + c_2 \cdot B h T^2\big) \text{ bytes},$$

where $c_1$ counts bytes saved per element of a $[B, T, d]$ tensor per layer and $c_2$ counts bytes saved per attention-score element. Both depend on your block's design and kernels. One published count for a GPT-style block with 16-bit activations and dropout gives $c_1 = 34$ and $c_2 = 5$ (Korthikanti et al., §4.1, eq. (1)); treat it as an anchor and recount for your own block. The $T^2$ term disappears when attention never materializes the score matrix (§10).

**Temporary buffers.** The largest is often the logits: $[B, T, V]$, and cross-entropy implementations may hold an fp32 copy, the softmax and its gradient. Add the runtime context, workspace for matrix-product libraries and allocator fragmentation (reserved but unused memory).

**Worked example (predicted, same configuration as §1, $h = 8$).** Decimal GB.

| Component | Calculation | Predicted size |
|---|---|---|
| Model states (AdamW, mixed precision) | $16 \times 353.2$ M | 5.65 GB |
| Activations, $c_1$ term | $24 \times 34 \times 8 \times 2048 \times 1024$ | 13.7 GB |
| Activations, attention $T^2$ term | $24 \times 5 \times 8 \times 8 \times 2048^2$ | 32.2 GB |
| Logits, one fp32 copy | $8 \times 2048 \times 50{,}000 \times 4$ | 3.3 GB (expect 2–3 copies) |
| Total before overhead | | ≈ 55 GB with one logits copy; up to ≈ 61 GB with three |

The budget says this configuration is dominated by stored attention scores, and that model states — often the first thing people reduce — are the smallest line. With an attention kernel that does not store scores, the total drops to roughly 23–29 GB depending on logits copies; adding block-level recomputation (§8) drops it further. Each of those is a new prediction to check against a measurement of peak allocated memory.

*In code this shows up as* P12's budget: replace my constants with the shapes and saved tensors of your own model, and list every assumption (dtype per tensor, dropout or not, attention backend).

### 5. Number formats

A binary floating-point number stores a sign, an exponent $e$ and a mantissa (fraction) $f$; a normal value is $(-1)^{\text{sign}} \cdot (1.f)_2 \cdot 2^{e - \text{bias}}$. **Exponent bits set the range; mantissa bits set the precision.** The spacing between neighboring values near 1 is $2^{-m}$ for $m$ mantissa bits (machine epsilon).

| Format | Exponent bits | Mantissa bits | Largest finite | Smallest positive normal | Spacing at 1 |
|---|---|---|---|---|---|
| fp32 | 8 | 23 | ≈ $3.4 \times 10^{38}$ | ≈ $1.2 \times 10^{-38}$ | ≈ $1.2 \times 10^{-7}$ |
| fp16 | 5 | 10 | 65,504 | ≈ $6.1 \times 10^{-5}$ | ≈ $9.8 \times 10^{-4}$ |
| bf16 | 8 | 7 | ≈ $3.4 \times 10^{38}$ | ≈ $1.2 \times 10^{-38}$ | ≈ $7.8 \times 10^{-3}$ |
| fp8 E4M3 | 4 | 3 | 448 | $2^{-6}$ | 0.125 |
| fp8 E5M2 | 5 | 2 | 57,344 | $2^{-14}$ | 0.25 |

fp16 also has subnormals down to $2^{-24} \approx 6 \times 10^{-8}$, with reduced precision. bf16 keeps fp32's exponent, so it rarely overflows or underflows, but it has only about two to three significant decimal digits.

**Worked rounding examples.**

- bf16 near 1 has spacing $2^{-7} \approx 0.0078$, so $1 + 0.001$ rounds back to $1$. A weight update smaller than half the local spacing vanishes.
- bf16 represents every integer up to 256, but between 256 and 512 the spacing is 2. Adding 1 to a bf16 running sum of 256 gives 257, which rounds (to even) back to 256: a sum accumulated in bf16 can stall.
- fp16 near 1 has spacing ≈ 0.00098, so $1 + 0.001$ survives, but $1 + 0.0001$ does not.

These are why matrix units accumulate products in fp32 even when inputs are 16-bit, and why reductions (loss sums, norms, softmax denominators) are usually computed in fp32.

**fp8.** Two variants trade range for precision: E4M3 (more precision) is commonly proposed for weights and activations, E5M2 (more range) for gradients. Because the range is so narrow, fp8 training keeps a **scale factor** per tensor or per block and multiplies values into range before casting. Only matrix-product inputs are typically fp8; accumulation, master weights and many other operations stay in higher precision. Support depends on hardware generation and library version — check the pinned stack before relying on it.

### 6. Mixed precision, master weights and loss scaling

**Mixed precision** runs matrix products and most activations in a 16-bit format while keeping numerically sensitive state in fp32:

1. Keep an fp32 **master copy** of the weights. The optimizer updates this copy, so small updates accumulate instead of rounding away (the first bf16 example above).
2. Run forward and backward with 16-bit casts of the weights and 16-bit activations.
3. Keep reductions, the loss and optimizer moments in fp32.

**Loss scaling (fp16).** Many gradient values in deep networks are tiny. fp16's smallest subnormal is about $6 \times 10^{-8}$, so a gradient of $2 \times 10^{-8}$ becomes 0. Multiplying the loss by a scale $S$ multiplies every gradient by $S$ (backward is linear in the upstream gradient). With $S = 2^{16} = 65{,}536$ that gradient becomes ≈ $1.3 \times 10^{-3}$, comfortably representable. Before the optimizer step, divide by $S$ in fp32 to recover the true value.

The danger is the other end: a gradient of 2.0 scaled by 65,536 overflows fp16's maximum of 65,504 to infinity. **Dynamic loss scaling** handles both: if any gradient is inf/NaN, skip that step and reduce $S$; after a run of clean steps, increase $S$. Any operation that reads gradient magnitudes — clipping by norm, logging the gradient norm — must happen *after* unscaling.

bf16's fp32-sized range usually makes loss scaling unnecessary, which is the main practical reason it is preferred where hardware supports it.

In PyTorch the API pattern looks like this (`torch.amp.GradScaler("cuda")` exists from PyTorch 2.3; earlier releases used `torch.cuda.amp.GradScaler` — check your pinned version):

```python
scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = compute_loss(model, batch)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)          # gradients are now true values
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)              # skipped if inf/NaN was found
scaler.update()
```

On CPU, autocast supports bf16 for a subset of operations; it illustrates dtype flow and rounding, not GPU speed.

### 7. Gradient accumulation and the normalization subtlety

**Gradient accumulation** splits one optimizer step's batch into $k$ micro-batches, runs forward and backward on each (PyTorch adds into `.grad`), then steps once and zeroes gradients. Activation memory scales with the micro-batch; the effective batch is the sum of micro-batches.

The trap is the loss normalization. Suppose your intended loss is the **mean over all non-padding tokens in the step**. Dividing each micro-batch's mean loss by $k$ reproduces that only when every micro-batch has the same number of counted tokens.

**Worked example.** Two micro-batches:

| Micro-batch | Counted tokens | Sum of token losses | Mean |
|---|---|---|---|
| 1 | 300 | 690 | 2.3 |
| 2 | 100 | 150 | 1.5 |

- Intended token mean over the step: $(690 + 150) / 400 = 2.1$.
- Average of micro-batch means: $(2.3 + 1.5) / 2 = 1.9$.

The gradient differs too: each token in micro-batch 1 gets weight $1/(2 \cdot 300) = 1/600$, each token in micro-batch 2 gets $1/200$ — three times as much — instead of $1/400$ for every token. The fix is to divide each micro-batch's *summed* loss by the total token count for the whole step (known before backward if you count tokens when you assemble the step's batches). The same issue reappears across data-parallel ranks in [L9](09-distributed.md).

Other accumulation details: zero gradients once per optimizer step, not per micro-batch; clip after the last micro-batch; count the learning-rate schedule in optimizer steps; and remember that anything computed per micro-batch (such as normalization statistics in batch-dependent layers) is not identical to the large-batch computation.

### 8. Activation recomputation (checkpointing)

**Activation checkpointing** saves only selected tensors in the forward pass — typically each block's input — and recomputes the block's internals during backward when they are needed.

- *Memory:* instead of $L$ blocks' full activations, you store $L$ block inputs ($L \cdot B T d \cdot s$ bytes) plus one block's full activations at a time during backward.
- *Compute:* each checkpointed block runs its forward twice, so a training step costs about $8N$ instead of $6N$ FLOPs per token (≈ 33% more) if every block is checkpointed. Hardware FLOPs rise; model FLOPs and MFU, by definition, do not.
- *Choices:* checkpointing every $\sqrt{L}$ blocks gives $O(\sqrt{L})$ memory (Chen et al.); **selective** recomputation recomputes only cheap-to-recompute but large tensors such as attention scores (Korthikanti et al.).

**Worked example (predicted).** For §4's configuration with 2-byte block inputs: $24 \times 8 \times 2048 \times 1024 \times 2$ bytes ≈ 0.8 GB of stored inputs, plus one block's activations (≈ 0.57 GB without the score term). Compare 13.7 GB for the $c_1$ term without recomputation.

PyTorch exposes this as `torch.utils.checkpoint.checkpoint(fn, *inputs, use_reentrant=False)`; the `use_reentrant` default and recommendations have changed across versions. Correctness points: dropout inside a recomputed region must reproduce the same mask (PyTorch preserves the RNG state by default), and in-place operations or side effects inside `fn` can break recomputation.

**Preserving correctness.** Recomputation must not change gradients beyond floating-point noise. The check is a controlled comparison: same weights, same batch, same seeds, gradients with and without checkpointing, compared with a stated tolerance. That comparison is the "preserve correctness across one optimization" demonstration in L8.

### 9. Profiling and measuring correctly

GPU work is **asynchronous**: a Python call enqueues a kernel and returns immediately. Wall-clock timing around GPU code measures launch time unless you synchronize.

A correct timing protocol:

1. **Warm up.** Run several iterations first. Early iterations include lazy initialization, allocator growth, library autotuning and, with `torch.compile`, compilation (repeated if shapes change).
2. **Synchronize** before starting and before stopping the timer (`torch.cuda.synchronize()`, or CUDA events with `enable_timing=True`; `torch.mps.synchronize()` on Apple GPUs).
3. **Repeat** and report the median and spread, not one run.
4. **Hold everything else fixed:** shapes, dtype, data already on device (or measure data loading separately), same seeds.
5. **Record the environment:** device, driver/runtime, PyTorch version, backend, thread count on CPU.

`torch.utils.benchmark.Timer` handles warmup, synchronization and repetition for small snippets.

**`torch.profiler`** records operators and, on GPU, kernels:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(activities=[ProfilerActivity.CPU],      # add ProfilerActivity.CUDA on a GPU
             schedule=schedule(wait=1, warmup=2, active=3),
             record_shapes=True, profile_memory=True) as prof:
    for _ in range(6):
        train_step()                                   # your own step
        prof.step()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
```

What you can read from it:

| Column or view | Meaning |
|---|---|
| Self time vs total time | time in the operator itself vs including the operators it called |
| CUDA time per op | kernel time on the device, as opposed to CPU-side dispatch |
| Input shapes (`record_shapes`) | which call of an operator is expensive |
| Memory columns (`profile_memory`) | bytes allocated/freed per operator |
| Exported trace (`prof.export_chrome_trace`) | a timeline; gaps between GPU kernels mean the device is waiting on the CPU (launch overhead, data loading, synchronization) |

Many tiny kernels with gaps suggest fusion or CUDA graphs; one dominant matrix product near peak suggests the step is already compute-bound. For activation memory independent of device, `torch.autograd.graph.saved_tensors_hooks` lets you count the bytes autograd saves — but note that GPU attention backends may save different tensors than the CPU path.

**Predicted vs measured.** Write the prediction first (FLOPs, bytes, expected regime), measure, then explain the gap. A large gap is information: an unexpected materialized tensor, a dtype you did not intend, a CPU-bound loop.

### 10. FlashAttention: tiling, online softmax, IO-awareness

Standard attention for one head computes $S = QK^\top / \sqrt{d_h}$ of shape $[T, T]$, writes it to HBM, reads it to compute $P = \operatorname{softmax}(S)$, writes $P$, then reads $P$ again for $O = PV$. For long $T$ the $T^2$ reads and writes dominate the time, and storing $P$ for backward dominates activation memory (§4).

**FlashAttention** computes exactly the same output without materializing $S$ or $P$ in HBM:

- **Tiling.** Load a block of query rows into SRAM, then stream blocks of $K$ and $V$ through SRAM, computing each block of scores on chip.
- **Online softmax.** For each query row, keep a running maximum $m$, a running denominator $\ell$ and a running unnormalized output $o$. When a new score block arrives with maximum $\tilde{m}$, set $m' = \max(m, \tilde{m})$, rescale the old quantities by $e^{m - m'}$, and add the new block's contributions with weights $e^{s_j - m'}$. At the end, $O = o / \ell$.
- **Recompute in backward.** Save only $O$ and the per-row log-sum-exp, then recompute score blocks during backward instead of reading a stored $P$.

The FLOP count is the same or slightly higher; the HBM traffic is much lower, which is the sense in which the algorithm is **IO-aware**.

**Worked online-softmax example.** One query row with scores $[2, 1, 3, 0]$ and scalar values $[1, 2, 3, 4]$, processed in two blocks.

| Step | $m$ | $\ell$ | $o$ |
|---|---|---|---|
| Block 1: scores $[2, 1]$, values $[1, 2]$ | 2 | $e^{0} + e^{-1} = 1.3679$ | $1 \cdot 1 + 0.3679 \cdot 2 = 1.7358$ |
| Block 2 arrives, max 3: rescale by $e^{2-3} = 0.3679$ | 3 | $0.5032 + e^{0} + e^{-3} = 1.5530$ | $0.6386 + 1 \cdot 3 + 0.0498 \cdot 4 = 3.8377$ |
| Final $o / \ell$ | | | 2.4711 |

Computing the full softmax of $[2, 1, 3, 0]$ directly and taking the weighted sum gives the same 2.4711. The running maximum is also what keeps the exponentials from overflowing.

PyTorch's `scaled_dot_product_attention` chooses among flash, memory-efficient and plain ("math") implementations depending on device, dtype, shapes and mask; which one runs is version- and hardware-dependent, so check rather than assume. Implementing FlashAttention is an optional specialization in this course.

### 11. Kernel fusion and Triton

**Kernel fusion** merges a sequence of operations into one kernel so intermediates stay in registers or SRAM instead of making round trips through HBM. Example: bias add → GELU → dropout on a $[B, T, 4d]$ tensor with §1's shapes (67.1 M bf16 elements, 134 MB per full read or write):

| Version | HBM traffic | Predicted time at 2 TB/s |
|---|---|---|
| Three separate kernels (each reads and writes) | ≈ 805 MB | ≈ 0.40 ms |
| One fused kernel (read once, write once) | ≈ 268 MB | ≈ 0.13 ms |

This is a bandwidth-only prediction; it ignores the bias vector, launch overhead and whether the backward pass is also fused.

`torch.compile` performs much of this fusion automatically: its default backend generates Triton kernels for GPUs and C++ for CPUs. Compilation changes timing (first-call cost, recompiles on new shapes) and can change numerics slightly, so it is itself an "optimization that must preserve correctness".

**Triton** is a Python-embedded language and compiler for writing GPU kernels at the level of *blocks* rather than individual threads:

| Concept | What you write | What the compiler handles |
|---|---|---|
| Program instance | a function decorated with `@triton.jit`, launched over a grid; each instance finds its block with `tl.program_id` | mapping instances to hardware |
| Block of indices | offsets from `tl.arange(0, BLOCK)` | vectorization across threads |
| Memory access | `tl.load` / `tl.store` with a mask for the ragged edge | coalescing, use of shared memory |
| Tuning | candidate block sizes and warps in `triton.autotune` | benchmarking candidates |

Triton targets GPUs; setting `TRITON_INTERPRET=1` runs kernels through its interpreter on CPU for debugging (see the debugging guide in the Triton docs for your release). Treat Triton as orientation in the core course: be able to read a simple kernel, say what it loads and stores and predict its bytes moved.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| GPU timings are tiny and nearly identical across sizes | No synchronization; you timed kernel launches | Synchronize before each timer read; use CUDA events or `torch.utils.benchmark` |
| First measured step is many times slower than the rest | Warmup, autotuning or `torch.compile` compilation included | Discard warmup steps; check for recompiles when shapes change |
| Out of memory although the budget predicted it fits | Attention scores materialized by the chosen backend; fp32 logits copies; allocator fragmentation; evaluation without `torch.no_grad()` | Compare `max_memory_allocated` with `max_memory_reserved`; profile memory per op; check which attention backend ran |
| fp16 loss becomes NaN or inf after some steps | Overflow; loss scale too high; an fp16 reduction | Log the scale and count skipped steps; locate the first non-finite tensor |
| bf16 training plateaus early or weights stop changing | No fp32 master weights; updates below bf16 spacing | Check parameter and optimizer-state dtypes; compare the update size with the local spacing |
| Accumulated run diverges from the same batch in one step | Mean-of-means normalization; zeroing gradients per micro-batch; clipping too early | Compare gradients of one large batch against the accumulated micro-batches |
| Clipping has no effect, or every step is clipped, with GradScaler | Norm computed on scaled gradients | Unscale before clipping and logging |
| Checkpointed and plain gradients differ beyond noise | RNG not restored for dropout; in-place op or side effect in the recomputed function | Disable dropout and compare; inspect the recomputed function |
| "Speedup" reported from a CPU run as a GPU result | Illustration mislabeled as measurement | Label each number with device, stack and whether it was predicted or measured |

## Check yourself

1. Count the parameters and training FLOPs per token of your own decoder from its actual shapes. Which modules does $6N$ miss, and at what $T$ does the attention term reach half of the weight term?
2. For a matrix product of shape $[M, K] \times [K, N]$, which single dimension would you shrink to push it from compute-bound to memory-bound? Predict, then check your answer with the intensity formula.
3. Predict, then run: add 1 to a fp16 scalar 3,000 times in a Python loop, with every addition rounded to fp16. Where does the sum stop, and why there?
4. Why does mixed-precision AdamW not reduce model-state memory below $16N$? What does it reduce?
5. Write the activation-memory budget for your own block by listing each tensor autograd saves. Then measure saved bytes with `saved_tensors_hooks` and explain any difference.
6. Predict, then run: compare gradients from one batch with uneven padding against the same batch split into two micro-batches, first with mean-of-means and then with token-count normalization.
7. Turn on activation checkpointing for every block of your model. Predict the change in step time and in saved-activation bytes, measure both, and show that gradients agree within a stated tolerance.
8. Time one training step with and without synchronization, with and without warmup. Report all four numbers and explain which one is the measurement.
9. In the online-softmax recurrence, what would go wrong if you skipped the rescaling by $e^{m - m'}$ when a new maximum arrives?
10. Read a profiler table from your training step. Which operator has the largest self time, is it compute- or memory-bound by your estimate, and what would you change first?
11. Which of your L8 claims could a CPU run support, and which need a GPU measurement? Sort them into the two lists before measuring.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [Kaplan et al., Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361) | §2.1 and its parameter/FLOP table (Table 1) | Origin of the $6N$ accounting and its context term |
| [Hoffmann et al., Training Compute-Optimal Large Language Models (2022)](https://arxiv.org/abs/2203.15556) | Appendix on FLOP computation | A more detailed count including attention; compare with §1 |
| [Chowdhery et al., PaLM (2022)](https://arxiv.org/abs/2204.02311) | Training-efficiency discussion of model FLOPs utilization | Definition of MFU vs hardware FLOPs utilization |
| Williams, Waterman and Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (CACM, 2009) | Whole short paper | The original roofline argument (optional) |
| [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Programming model: thread hierarchy and memory hierarchy | Authoritative description of SMs, shared memory and device memory |
| [Rajbhandari et al., ZeRO (2019)](https://arxiv.org/abs/1910.02054) | §3, where memory goes | Model-state and residual-state accounting behind §4 |
| [Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models (2022)](https://arxiv.org/abs/2205.05198) | §4 activation memory (per-layer count: §4.1, eq. (1)); §5 selective recomputation | The per-layer activation count used as an anchor |
| [Micikevicius et al., Mixed Precision Training (2017)](https://arxiv.org/abs/1710.03740) | §3: master weights, loss scaling, arithmetic precision | The mixed-precision recipe |
| [Micikevicius et al., FP8 Formats for Deep Learning (2022)](https://arxiv.org/abs/2209.05433) | Format definitions and scaling discussion | E4M3/E5M2 trade-offs |
| [Chen et al., Training Deep Nets with Sublinear Memory Cost (2016)](https://arxiv.org/abs/1604.06174) | The checkpointing algorithm sections | The $O(\sqrt{L})$ recomputation idea |
| [Milakov and Gimelshein, Online Normalizer Calculation for Softmax (2018)](https://arxiv.org/abs/1805.02867) | The online normalizer algorithm | Online softmax in isolation |
| [Dao et al., FlashAttention (2022)](https://arxiv.org/abs/2205.14135) | §3.1 algorithm, §3.2 IO analysis | Tiling, recomputation and the IO-complexity argument |
| [Dao, FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691) | Algorithm and parallelism sections (optional) | How work partitioning changes kernel efficiency |
| [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html), [activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html), [benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) | API pages for the pinned version | Exact semantics and current defaults |
| [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | Whole recipe | Reading tables, memory and traces |
| [torch.compile documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html) | Overview | What gets fused and when recompiles happen |
| [Triton documentation](https://triton-lang.org/) | Getting-started tutorials (optional) | Block-level kernel programming |
| [CS336](https://cs336.stanford.edu/) | Systems lectures and assignment handout | Course-length treatment of the same material |
