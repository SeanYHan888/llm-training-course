# Inference and serving

> **Elective** · related milestones: [L2](../course/ROADMAP.md#milestones), [L8](../course/ROADMAP.md#milestones), [L4](../course/ROADMAP.md#milestones) · **Builds:** [Inference 01](../exercises/inference/README.md) (Labs A–C), cached decoding in the [Transformer workshop](../exercises/transformer/README.md) · **Prerequisites:** [L2](02-transformer.md) attention and generation; [L8](08-efficiency.md) memory hierarchy helps but is not required

Training runs a model once over many full sequences; serving runs it token by token for many unrelated users at once. Almost every serving technique — KV caching, paged memory, prefix reuse, continuous batching — exists because generation is a long chain of small steps that is limited by how fast memory can be read, not by arithmetic. This note builds that picture from first principles so you can reason about the Inference 01 labs, read the PagedAttention and SGLang papers, and judge serving benchmarks honestly.

## What you should be able to do

- Explain the difference between **prefill** and **decode**, and argue from bytes and FLOPs why decode is usually memory-bandwidth bound.
- Write the symbolic KV-cache size for a dense decoder and predict how it changes with sequence length, batch, layers, and the number of KV heads (MHA vs GQA vs MQA).
- Explain internal and external fragmentation of KV memory and how **PagedAttention** removes most of it with fixed-size blocks, block tables and copy-on-write sharing.
- Distinguish per-request KV reuse, cross-request **prefix caching**, and whole-response caching; describe how a radix tree (SGLang's RadixAttention) finds reusable prefixes.
- Explain **continuous batching** and **chunked prefill** and the latency/throughput trade-off they manage.
- Define TTFT, inter-token latency, end-to-end latency and throughput, and design a measurement that does not mislead.
- Place quantization, speculative decoding, FlashAttention-style kernels and structured output on a map: which bottleneck each one attacks.

## Core ideas

### 1. Two phases: prefill and decode

Autoregressive generation produces one token at a time. Given a prompt of $T_p$ tokens:

- **Prefill** runs the whole prompt through the model in one forward pass. All $T_p$ positions are processed in parallel (with the causal mask), and the pass ends with logits for the next token.
- **Decode** then loops: sample a token, feed *only that token* through the model, get the next logits, repeat until a stop token or a length limit.

Decode can feed only one token because of the **KV cache**: at every layer, attention for the new position needs the keys and values of all earlier positions, and those do not change once computed (they depend only on earlier tokens, by causality). So the engine stores them and appends one new key and value per layer per step. Without the cache, each decode step would recompute the whole prefix.

| Phase | Tokens per sequence per pass | Attention work for a sequence | What dominates |
|---|---|---|---|
| Prefill | $T_p$ | queries $[T_p, d_h]$ against keys $[T_p, d_h]$ per head — quadratic in $T_p$ | Matrix-multiply compute |
| Decode step $t$ | 1 | one query $[1, d_h]$ against $t$ cached keys per head — linear in $t$ | Reading weights and KV cache from memory |

Shapes for one layer, per sequence, $h$ query heads of size $d_h$, $h_{kv}$ KV heads:

| Tensor | Prefill | One decode step at position $t$ |
|---|---|---|
| Hidden input | `[B, T_p, d]` | `[B, 1, d]` |
| Queries | `[B, h, T_p, d_h]` | `[B, h, 1, d_h]` |
| Keys/values used | `[B, h_kv, T_p, d_h]` (written to cache) | `[B, h_kv, t, d_h]` (read from cache, 1 position appended) |
| Attention scores | `[B, h, T_p, T_p]` | `[B, h, 1, t]` |

In code this shows up as Lab A's two paths: the uncached reference recomputes the whole prefix, while the cached path does one prefill and then extends stored keys and values with an explicit position counter. If the two paths disagree, the usual suspects are the mask and the position index of the new token.

### 2. Why decode is memory-bandwidth bound

A useful mental model is the **roofline**: an operation needs some number of FLOPs and must move some number of bytes between main memory (HBM on a GPU) and the compute units. Its **arithmetic intensity** is

$$I = \frac{\text{FLOPs}}{\text{bytes moved}}.$$

A device with peak compute $C$ (FLOP/s) and memory bandwidth $W$ (bytes/s) is memory-bound for operations with $I < C/W$ and compute-bound above it. For current datacenter accelerators $C/W$ is in the hundreds of FLOPs per byte (check your device's datasheet).

Now count one decode step for a model with $P$ parameters stored at $s$ bytes each, running a batch of $B$ sequences. Each parameter participates in roughly one multiply-add per token, so:

- FLOPs $\approx 2PB$ (ignoring attention over the cache).
- Bytes $\approx sP$ for the weights, read once per step and shared by the whole batch, **plus** each sequence's own KV cache, which cannot be shared.

So for the weight part $I \approx 2B/s$. With 2-byte weights, $I \approx B$: a single sequence does about **one FLOP per byte** read — far below the hundreds a device can sustain. The compute units mostly wait for memory.

Worked example (hypothetical round numbers, not a real device or model): a 3-billion-parameter model in 2-byte precision has 6 GB of weights. On a device with $W = 1$ TB/s and $C = 100$ TFLOP/s:

| Quantity | Value |
|---|---|
| Time to stream weights once | $6\text{ GB} / 1\text{ TB/s} = 6$ ms |
| FLOPs for one token | $2 \times 3\times10^9 = 6$ GFLOP |
| Time to compute one token | $6\text{ GFLOP} / 100\text{ TFLOP/s} = 0.06$ ms |
| Upper bound on decode steps/s at batch 1 | about 167 |

The step is about 100× slower than its arithmetic alone, so batching ~100 sequences would cost little extra time per step — **if** their KV caches fit and are cheap to read. That "if" is why the rest of this note is about KV memory. Prefill, by contrast, reuses each weight across $T_p$ tokens, giving $I \approx T_p$ for the weight reads; long prompts are compute-bound.

Two consequences:

1. Decode throughput grows with batch size until either compute saturates or KV memory runs out. Memory capacity, not FLOPs, usually sets the maximum batch.
2. Attention over the cache has intensity near 1 regardless of batch, because every sequence reads its own keys and values. Long contexts make decode steps slower even at fixed batch.

### 3. KV-cache memory

For a dense decoder with standard attention, each layer stores one key vector and one value vector per KV head per token. Define:

| Symbol | Meaning |
|---|---|
| $L$ | layers |
| $h_{kv}$ | KV heads per layer |
| $d_h$ | head dimension |
| $s$ | bytes per stored element |
| $B$ | sequences in the batch |
| $T$ | tokens held per sequence (prompt + generated so far) |

Then

$$M_{KV} = 2 \cdot L \cdot h_{kv} \cdot d_h \cdot s \cdot B \cdot T \text{ bytes},$$

where the leading 2 counts K and V. The cache grows **linearly** in both $B$ and $T$, which is exactly the product a serving system wants to make large. The per-token cost $2 L h_{kv} d_h s$ is a property of the architecture and precision; it is the number to compute first for any model.

**Head sharing.** The number of query heads $h$ sets the attention pattern's expressiveness; the number of KV heads $h_{kv}$ sets the cache size.

| Variant | $h_{kv}$ | Effect on $M_{KV}$ |
|---|---|---|
| Multi-head attention (MHA) | $h_{kv} = h$ | baseline |
| Grouped-query attention (GQA) | $1 < h_{kv} < h$, each KV head shared by $h/h_{kv}$ query heads | divided by $h/h_{kv}$ |
| Multi-query attention (MQA) | $h_{kv} = 1$ | divided by $h$ |

Worked example (hypothetical model): $L = 24$, $h = 16$ query heads, $d_h = 96$, $s = 2$ bytes, one sequence of $T = 2000$ tokens.

| Variant | Bytes per token $2 L h_{kv} d_h s$ | Cache for 2000 tokens |
|---|---|---|
| MHA, $h_{kv}=16$ | 147,456 B = 144 KiB | 294,912,000 B ≈ 281.25 MiB |
| GQA, $h_{kv}=4$ | 36,864 B = 36 KiB | 73,728,000 B ≈ 70.31 MiB |
| MQA, $h_{kv}=1$ | 9,216 B = 9 KiB | 18,432,000 B ≈ 17.58 MiB |

Notice the query heads do not appear in the formula at all. GQA and MQA trade some modeling capacity for a smaller cache and less memory traffic per decode step; whether quality is preserved is an empirical question settled by training, not by the formula.

What the formula leaves out: model weights, activations and temporary buffers during a step, allocator padding and reserved-but-unused slots, block tables and other metadata, and framework/runtime overhead. Architectures that do not keep one K and V per token per layer (sliding-window layers, compressed latent caches, linear-attention recurrent state as in the [attention-alternatives workshop](../exercises/attention-alternatives/README.md)) need their own accounting.

In code this shows up as Lab B: inspect your cached tensors' shapes and element sizes, check them against the formula, and apply it to the lab's own hypothetical configuration.

### 4. Fragmentation and PagedAttention

The naive allocator gives each request one contiguous KV buffer sized for the **maximum** length it might reach, because the final output length is unknown in advance. Three kinds of waste follow:

| Waste | Cause |
|---|---|
| Reserved but unused | Slots allocated for tokens that may never be generated |
| Internal fragmentation | The request stops early; the unused tail of its buffer is stranded |
| External fragmentation | Free memory exists but is split into gaps too small for a new contiguous buffer |

**PagedAttention** (Kwon et al.) borrows the idea of virtual memory from operating systems:

- The KV pool is divided into **physical blocks** of fixed size $b$ tokens (each block holds K and V for $b$ positions, for every layer and KV head, in the engine's chosen layout).
- Each request sees a sequence of **logical blocks**: logical block $j$ holds positions $jb$ to $jb + b - 1$.
- A per-request **block table** maps logical block $j$ to some physical block id. Physical blocks need not be adjacent.
- New physical blocks are allocated only when the last one fills; finished requests return their blocks to a free list.
- The attention kernel takes the block table as input and gathers keys and values from scattered blocks. That is why it needs a custom kernel rather than an ordinary dense attention call.

Waste becomes small and bounded: at most $b - 1$ empty slots in each request's last block, and no external fragmentation because all blocks are interchangeable.

Small example with $b = 4$: a request holding 10 tokens needs $\lceil 10/4 \rceil = 3$ logical blocks. Its table might be `[7, 2, 9]` — positions 0–3 live in physical block 7, 4–7 in block 2, and 8–9 in the first two slots of block 9, leaving 2 empty slots. Writing position 12 (the 13th token) would trigger allocation of a fourth physical block from the free list.

**Sharing and copy-on-write.** Because the indirection is explicit, two block tables can point at the same physical block. Each physical block keeps a **reference count**. This lets, for example, several samples drawn from one prompt (parallel sampling) share the prompt's blocks. Full blocks are read-only and can be shared freely. When a sharer needs to *write* into a block whose count is above 1 — typically the partially filled last block of the shared prompt — the engine first copies it to a fresh block, points that sharer's table at the copy and decrements the original's count. This is **copy-on-write**: memory is duplicated only at the moment sequences diverge.

The choice of $b$ is a trade-off: small blocks waste fewer slots but need larger tables and less efficient gathers; large blocks do the reverse. Engines expose it as a configuration value whose default is version-dependent.

In code this shows up as the Lab C logical/physical trace. Draw it by hand first; the lab deliberately does not ask for an allocator implementation.

### 5. Prefix caching across requests and radix trees

Many requests share leading tokens: a long system prompt, a few-shot template, a document asked several questions, or the growing history of a chat or agent loop. Their KV tensors for that shared prefix are identical *provided* the token ids **and** positions are identical (and the model and its settings are the same). So an engine can keep KV blocks after a request finishes and reuse them when a later request starts with the same tokens. Prefill then runs only on the new suffix, which cuts time to first token and prefill compute.

Keep three different kinds of reuse separate:

| Reuse | Scope | What is saved |
|---|---|---|
| Per-request KV cache | Within one request, across its decode steps | Recomputing the prefix each step |
| Cross-request prefix cache | Across requests that share leading tokens | Prefill for the shared part |
| Response cache | Identical full requests | The whole computation, by replaying a stored answer (outside the model engine) |

A shared middle or suffix cannot be reused this way: a token's keys and values depend on every token before it.

**Radix-tree reuse (SGLang RadixAttention).** SGLang keeps cached prefixes in a **radix tree** — a trie in which each edge is labeled by a run of tokens and each node stands for the prefix spelled by the path from the root. An incoming request walks the tree from the root to find its **longest matching prefix**; those positions are reused and only the remainder is prefilled, after which the new tokens become a new branch. Nodes in use by running requests are protected by reference counts; when memory is needed, unreferenced leaves are evicted, least recently used first. The SGLang paper also describes a cache-aware scheduler that orders waiting requests so that ones sharing long cached prefixes run close together.

Tiny example (different from Lab C): requests arrive in this order, with unlimited cache.

| Request | Tokens | Matched prefix | New tokens prefilled |
|---|---|---|---|
| R1 | `[40, 41, 42, 43, 44]` | 0 | 5 |
| R2 | `[40, 41, 42, 50]` | 3 (`40 41 42`) | 1 |
| R3 | `[40, 41, 60, 61, 62]` | 2 (`40 41`) | 3 |

Totals: 14 prompt tokens, 5 reused, 9 prefilled. After R3 the tree has a root edge `40 41`, which branches into `42` (itself branching into `43 44` and `50`) and `60 61 62`.

Block-based engines (for example vLLM's automatic prefix caching) achieve cross-request reuse by identifying KV blocks by their content and the prefix before them (vLLM's V1 design notes hash each full block together with its parent block's hash and cache only full blocks); defaults depend on the release, so check the documentation for the version you pin. The historical paper comparison does not mean only SGLang reuses prefixes.

### 6. Scheduling: continuous batching and chunked prefill

**Static (request-level) batching** groups requests, runs them together until the longest one finishes, then admits the next group. Short requests sit finished while their slot idles, and new arrivals wait.

| Decode step | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Request X (needs 6 tokens) | X | X | X | X | X | X |
| Request Y (needs 2) | Y | Y | idle | idle | idle | idle |
| Request Z (needs 3) | Z | Z | Z | idle | idle | idle |

**Continuous batching** (also called iteration-level scheduling; introduced in the Orca system) re-decides batch membership at **every step**. A finished sequence leaves immediately, and a waiting request can join at the next step, provided its KV memory can be allocated. Paged memory makes this practical because admitting a request needs only a few free blocks, not one large contiguous region.

Admitting a new request means running its prefill, and a long prefill inside a step delays every decoding sequence in that step — a visible stall in their token stream. **Chunked prefill** splits a long prompt into fixed-size chunks and processes one chunk per step alongside the ongoing decodes, so each step has a bounded amount of work. This mixes compute-bound prefill tokens with memory-bound decode tokens in the same pass, which can use the hardware better. The chunk size trades the newcomer's time to first token against the steadiness of everyone else's stream.

When KV memory runs out mid-generation, an engine must **preempt** something: either free a request's blocks and recompute them later, or swap them to host memory. Which policies exist and which is the default is release-dependent.

### 7. Metrics and honest measurement

For one streaming request that produces $n$ output tokens:

| Metric | Definition |
|---|---|
| Time to first token (TTFT) | From request sent to first output token received. Dominated by queueing + prefill. |
| Inter-token latency (ITL) | Gap between consecutive output tokens. Dominated by decode step time. |
| Time per output token (TPOT) | Mean ITL over a request: $(E2E - TTFT)/(n - 1)$ |
| End-to-end latency (E2E) | From request sent to last token received |
| Output throughput | Total output tokens across all requests ÷ wall-clock time of the run |
| Request throughput | Completed requests ÷ wall-clock time |

Worked example: a request returns 41 tokens, with TTFT 0.18 s and E2E 2.18 s. Then TPOT $= (2.18 - 0.18)/40 = 0.05$ s, i.e. 50 ms per token, or 20 tokens/s for this one stream. If the server handled 16 such streams concurrently over the same 2.18 s, aggregate output throughput would be $16 \times 41 / 2.18 \approx 301$ tokens/s. Per-user speed and system throughput are different numbers, and raising concurrency usually improves the second while worsening the first.

Rules for measurements that mean something:

1. **Fix the workload.** Same model revision, dtype, tokenizer and chat template, the same prompts as token ids, same sampling settings, same input/output length limits. Force output length (ignore end-of-sequence or set a fixed maximum) when comparing step speed.
2. **Separate startup from steady state.** Model load, compilation and graph capture can take seconds to minutes. Warm up first, and report cold-start separately if it matters.
3. **Control the cache state.** Prefix-cache hits change TTFT drastically. Say whether each condition is cold or warm, and how you cleared or controlled the cache, or that you could not.
4. **Report distributions.** Give median and tail percentiles (p90/p99) across requests and repeats, plus sample size. A mean hides stalls.
5. **Describe the load.** Closed-loop (fixed concurrency, new request when one finishes) and open-loop (requests arrive at a rate regardless of completions) answer different questions. Under open-loop load, latency should include queueing time.
6. **Count tokens, not chunks.** A streamed chunk may contain several tokens or none. Use the tokenizer or server-reported token counts before calling chunk gaps "inter-token latency".
7. **Say where the clock is.** Client-side timings include network and serialization; server-side timings do not. Both are valid; mixing them is not.
8. **Label claims.** Measured on your hardware, simulated (hand traces, toy timelines), or author-reported (paper speedups) are three different kinds of evidence. A CPU timing says nothing about GPU serving.

### 8. Orientation map of other techniques

These are recognized here so you know which problem each solves; none is assigned in Inference 01.

| Technique | Bottleneck it attacks | One-line mechanism | Watch out for |
|---|---|---|---|
| Weight quantization (e.g. 8-bit or 4-bit integer weights) | Bytes of weights read per decode step; memory capacity | Store weights in fewer bits with per-group scales; dequantize inside the matmul | Accuracy loss varies by method and model; speedups need kernels that actually exploit the format |
| KV-cache quantization | Cache size and cache bytes read | Store K/V in fewer bits | Long-context quality; engine support is version-dependent |
| Speculative decoding | Number of sequential, memory-bound steps | A cheap drafter proposes several tokens; the large model checks them in one pass and keeps the accepted prefix | Gains depend on acceptance rate; the exact accept/reject rule keeps the output distribution unchanged only if implemented correctly |
| FlashAttention-style kernels | Memory traffic *inside* attention | Compute attention in tiles held in fast on-chip memory, never writing the full $[T, T]$ score matrix to HBM | Changes how attention is computed, not how much KV must be stored; it complements, not replaces, paging |
| Structured / constrained output | Invalid outputs (e.g. malformed JSON) | At each step, mask logits of tokens that would violate a grammar or schema | Masking changes the distribution; tokenizer boundaries make grammar-to-token mapping nontrivial |
| Distributed inference (tensor/pipeline parallel) | Model or cache too large for one device | Split weights and KV across devices | Communication per step; see L9 for the training-side versions |

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Cached and uncached outputs agree for the first token, then diverge | New token given the wrong position index, or the cache appended twice | Print the position counter and cache length after each step; compare with the number of tokens fed |
| Cached path matches but "future" tokens influence the output in the uncached path | Causal mask missing or offset by one in the reference | Perturb a later token and check that earlier outputs do not change |
| Second request produces strange outputs | Cache not reset between requests | Assert cache length is 0 at request start |
| Memory estimate far below observed usage | Only KV counted; weights, temporaries, allocator reservation omitted | List every term the formula excludes; measure tensor bytes separately from process memory |
| Estimate off by exactly 2× | Forgot that K and V are both stored, or used 4-byte instead of 2-byte elements | Recompute element count and dtype from the actual tensors |
| GQA cache sized with $h$ instead of $h_{kv}$ | Confusing query heads with KV heads | Read $h_{kv}$ from the K tensor's shape, not the config's head count |
| Prefix cache "never hits" | Prompts differ in an early token (timestamp, request id, whitespace, different chat template) | Tokenize two prompts and find the first differing id |
| Benchmark shows one engine far faster | Different output lengths, cache warm in one run, compilation counted in the other | Fix output length, report cold/warm, discard warmup, repeat |
| ITL looks bimodal with long spikes | Long prefills admitted into decode steps | Check whether chunked prefill is enabled; correlate spikes with arrivals |

## Check yourself

1. Using your Lab A code, predict which tensors change shape when one token is appended, then run it and confirm.
2. Why is the KV cache correct only for *causal* attention? What would break for a bidirectional encoder?
3. Predict, then measure on your own tiny model: does the cached path beat the uncached one at length 4? At 16? Explain the answer in terms of overhead versus saved work.
4. A model switches from MHA with 32 heads to GQA with 8 KV heads. By what factor does the KV cache shrink? Does the attention-score tensor shrink?
5. Explain in one paragraph, with numbers you choose, why raising decode batch size from 1 to 8 can raise throughput nearly 8× while barely changing per-step time — and what eventually stops this.
6. With block size $b$, what is the worst-case number of wasted slots for a batch of $B$ requests? Why does PagedAttention not need to know output lengths in advance?
7. Two samples share a 9-token prompt with $b = 4$. Which blocks are shared, and at what moment does copy-on-write happen?
8. Change one token in the middle of a cached prefix. Which cached positions are still reusable, and why?
9. Draw a radix tree for four prompts of your own invention, then insert a fifth that shares nothing. What gets evicted first if memory runs out?
10. Design a two-condition experiment showing the effect of prefix caching on TTFT. List every variable you hold fixed and how you will control cache state.
11. A blog post reports "2× faster" for engine A over engine B. Write three questions you would need answered before believing it.
12. Which technique in the orientation map would help most for a single user on a laptop generating long answers? Which for a server with many users and a long shared system prompt? Justify each.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [Hugging Face — How caching works](https://huggingface.co/docs/transformers/en/cache_explanation) | Cache explanation and attention-mask discussion | Concrete library view of prefill/decode and cache objects; version-dependent API |
| [Kwon et al., PagedAttention (2023), v1](https://arxiv.org/abs/2309.06180v1) | §§4.1–4.3, Figures 4–7; §4.4 and Figure 8 for sharing and copy-on-write; §6 optional | Blocks, block tables, sharing and copy-on-write. Speedups are author-reported |
| [Zheng et al., SGLang (v2)](https://arxiv.org/html/2312.07104v2) | §3; §2 for context | RadixAttention, LRU eviction and cache-aware scheduling |
| [vLLM automatic prefix caching docs](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) | Whole page | Block-level cross-request reuse in a current engine; match to your pinned release |
| [Pope et al., Efficiently Scaling Transformer Inference (2022)](https://arxiv.org/abs/2211.05102) | Sections on memory/compute costs of decoding (optional depth) | Careful FLOP/byte analysis of prefill vs decode and of KV memory |
| [Shazeer, Fast Transformer Decoding: One Write-Head is All You Need (2019)](https://arxiv.org/abs/1911.02150) | Introduction and the incremental-decoding cost analysis | Origin of MQA and of the memory-bandwidth argument for decoding |
| [Ainslie et al., GQA (2023)](https://arxiv.org/abs/2305.13245) | Method section | Grouped-query attention and converting MHA checkpoints |
| Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) | Iteration-level scheduling sections (optional) | Origin of continuous batching |
| [Agrawal et al., Sarathi-Serve (2024)](https://arxiv.org/abs/2403.02310) | Problem statement and chunked-prefill design (optional) | Chunked prefill and the stall problem |
| [Dao et al., FlashAttention (2022)](https://arxiv.org/abs/2205.14135) | Abstract, introduction, algorithm overview (optional) | Attention memory-traffic optimization, distinct from KV allocation |
| [Leviathan et al., Fast Inference from Transformers via Speculative Decoding (2022)](https://arxiv.org/abs/2211.17192) | Sections 1–3 (optional) | The draft-and-verify algorithm and why it preserves the distribution |
| [Frantar et al., GPTQ (2022)](https://arxiv.org/abs/2210.17323) | Introduction and method overview (optional) | A representative post-training weight-quantization method |
| [Willard & Louf, Efficient Guided Generation for LLMs (2023)](https://arxiv.org/abs/2307.09702) | Introduction and method overview (optional) | Grammar/regex-constrained decoding via token masks |
| [SGLang request tutorial](https://docs.sglang.io/docs/basic_usage/send_request) | One request example (optional) | Connects paper mechanisms to a current API; version-dependent |
