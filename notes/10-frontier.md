# L10 — Frontier architectures and reports

> **Milestone:** [L10](../course/ROADMAP.md#milestones) · **Builds:** [Attention alternatives workshop](../exercises/attention-alternatives/README.md) (linear attention → KDA → K3 claim ledger), P14 in [projects](../course/PROJECTS.md) · **Prerequisites:** [L2](02-transformer.md) causal attention, RoPE and the KV cache; [L3](03-pretraining.md) for training context; [L8](08-efficiency.md) and [L9](09-distributed.md) for the systems claims; [L7](07-online-rl.md) for post-training claims

Frontier model reports describe systems you cannot rebuild, using components you can. This module covers the components that recur in current reports — mixture-of-experts layers, linear-attention and delta-rule memories such as Kimi Delta Attention, hybrid layouts, long-context training and multimodal objectives — at the level of equations, state shapes and memory arithmetic. It then teaches a disciplined way to read such a report: a claim ledger that separates what the authors report from what a given experiment could actually establish.

## What you should be able to do

- Explain a mixture-of-experts layer: router, top-$k$ selection, gate weights, load-balancing loss, capacity, and the difference between active and total parameters for compute and for memory.
- Write causal linear attention in its quadratic (masked) form and its recurrent state form, including the normalizer, and explain why it is a different attention rule from softmax.
- Write the delta-rule state update, interpret it as online regression, and explain what it fixes relative to purely additive memory.
- Describe scalar and channel-wise gating, and place Kimi Delta Attention (KDA) among these variants, marking which details must be checked in the paper.
- Compare recurrent-state memory with a growing KV cache using explicit element counts, including for hybrid layouts.
- Explain why RoPE models need care to extend context, the main extension families, and why training is staged by context length.
- Outline the main multimodal training objectives and which tokens carry loss.
- Build a claim ledger for one report claim: source location, author-reported evidence, comparison conditions, what a toy can and cannot test, the missing ablation, and what reproduction would require.

## Core ideas

### 1. Dense versus mixture-of-experts

In a **dense** Transformer every token passes through every parameter of every MLP. A **mixture-of-experts (MoE)** layer replaces one MLP with $E$ parallel MLPs ("experts") and a **router** that sends each token to only $k$ of them.

For a token's hidden vector $x \in \mathbb{R}^d$:

$$s = x W_r \in \mathbb{R}^{E}, \qquad \mathcal{T} = \operatorname{top-}k(s), \qquad y = \sum_{i \in \mathcal{T}} g_i \, \text{FFN}_i(x),$$

where $W_r$ is the router matrix `[d, E]` and $g_i$ are gate weights. Two common gate choices: a softmax over the $k$ selected logits (weights sum to 1), or a softmax over all $E$ logits, keeping the selected entries. Some designs add **shared experts** that every token uses, alongside routed ones.

**Worked routing example.** Router logits for one token over four experts: $[1.2, 0.3, 2.0, -0.5]$. Top-2 selects experts 3 and 1. Softmax over the selected pair: $g_3 = e^{2.0}/(e^{2.0} + e^{1.2}) = 1/(1 + e^{-0.8}) \approx 0.690$, $g_1 \approx 0.310$.

**Active versus total parameters.** Per-token compute scales with the parameters a token actually uses (**active**); memory for weights, gradients and optimizer states scales with all of them (**total**).

Worked example (my own numbers): $d = 1024$, $E = 32$ experts, top-2, each expert a gated MLP with hidden width 2816 (three matrices).

| Quantity | Calculation | Value |
|---|---|---|
| Parameters per expert | $3 \times 1024 \times 2816$ | 8.65 M |
| Total expert parameters per layer | $32 \times 8.65$ M | 276.8 M |
| Active expert parameters per token | $2 \times 8.65$ M | 17.3 M |
| Router parameters | $1024 \times 32$ | 32.8 K |
| AdamW model states for the experts (16 bytes/param, [L8](08-efficiency.md#4-the-training-memory-budget)) | $16 \times 276.8$ M | 4.4 GB per layer |

Per token this layer costs about as much as a dense gated MLP of width 5632, but stores sixteen times the parameters. "Active parameters" predicts FLOPs; "total parameters" predicts memory and checkpoint size. For scale, the course's reference notes on Kimi K3 record an author-reported 2.8 T total and 104 B active parameters, about 3.7% active.

**Load balancing.** Left alone, routers tend to collapse onto a few experts, which then get most of the training signal. A common remedy (Switch Transformer form) adds an auxiliary loss

$$\mathcal{L}_{bal} = \alpha \, E \sum_{i=1}^{E} f_i \, P_i,$$

where $f_i$ is the fraction of tokens dispatched to expert $i$ in the batch and $P_i$ is the mean router probability assigned to expert $i$. Under perfect balance $f_i = P_i = 1/E$ and the sum times $E$ equals 1. Example with $E = 4$: $f = [0.5, 0.3, 0.1, 0.1]$ and $P = [0.4, 0.3, 0.15, 0.15]$ give $4 \times (0.20 + 0.09 + 0.015 + 0.015) = 1.28$. Alternatives include a router z-loss that penalizes large logits and auxiliary-loss-free schemes that adjust per-expert biases in the routing decision.

**Capacity.** For fixed-shape computation, each expert gets a buffer of

$$C = \text{capacity factor} \times \frac{k \cdot (\text{tokens in the batch})}{E}$$

slots. With 4,096 tokens, $k = 2$, $E = 32$ and capacity factor 1.25, $C = 1.25 \times 256 = 320$. If 400 tokens choose one expert, 80 overflow: depending on the design they are dropped (only the residual connection carries them forward) or rerouted. Capacity also bounds the [all-to-all](09-distributed.md#9-expert-parallelism) cost per rank.

### 2. Linear attention

Causal softmax attention for one head computes, for position $t$,

$$o_t = \frac{\sum_{j \le t} \exp(q_t^\top k_j / \sqrt{d_k}) \, v_j}{\sum_{j \le t} \exp(q_t^\top k_j / \sqrt{d_k})}.$$

This is a kernel smoother: each value is weighted by a similarity $\text{sim}(q_t, k_j)$, then normalized. **Linear attention** chooses a similarity that factorizes through a feature map $\phi: \mathbb{R}^{d_k} \to \mathbb{R}^{d_\phi}_{>0}$:

$$\text{sim}(q, k) = \phi(q)^\top \phi(k), \qquad o_t = \frac{\phi(q_t)^\top \sum_{j \le t} \phi(k_j) v_j^\top}{\phi(q_t)^\top \sum_{j \le t} \phi(k_j)}.$$

Katharopoulos et al. use $\phi(x) = \operatorname{elu}(x) + 1$, which keeps every feature positive so the denominator stays positive.

**Two equivalent computations of the same rule.**

- *Quadratic (masked) form:* form $A = \phi(Q)\phi(K)^\top$ of shape `[T, T]`, zero entries with $j > t$, divide each row by its sum, multiply by $V$. Cost $O(T^2)$.
- *Recurrent form:* keep a matrix state and a normalizer state,

$$S_t = S_{t-1} + \phi(k_t)\, v_t^\top, \qquad z_t = z_{t-1} + \phi(k_t), \qquad o_t = \frac{S_t^\top \phi(q_t)}{z_t^\top \phi(q_t)},$$

starting from $S_0 = 0$, $z_0 = 0$. Cost $O(T \, d_\phi \, d_v)$, and constant memory per step.

The two must agree to floating-point precision when they use the same $\phi$ and the same inclusive mask ($j \le t$). That agreement — not agreement with softmax — is the P14 check.

| Tensor (one head) | Shape | Grows with $T$? |
|---|---|---|
| $\phi(q_t)$, $\phi(k_t)$ | `[d_phi]` | no |
| $v_t$, $o_t$ | `[d_v]` | no |
| State $S_t$ | `[d_phi, d_v]` | no |
| Normalizer $z_t$ | `[d_phi]` | no |
| Full layer state | `[B, h, d_phi, d_v]` + `[B, h, d_phi]` | no |
| Softmax KV cache (compare) | `[B, h_kv, T, d_h]` for K and again for V | yes |

**Training form.** Running the recurrence token by token is slow on parallel hardware. Implementations split the sequence into chunks: within a chunk use the quadratic form, across chunks pass the state. This chunkwise form is what production kernels optimize; it is outside P14's scope.

### 3. Why linear attention is not softmax attention

Same query and keys, two rules. Query $q = [0, 1]$ sees keys $k_1 = [0, 1]$, $k_2 = [1, 0]$, $k_3 = [0, 2]$.

| | $k_1$ | $k_2$ | $k_3$ |
|---|---|---|---|
| Softmax, scores $q^\top k / \sqrt{2}$ = 0.707, 0, 1.414 | 0.284 | 0.140 | 0.576 |
| Linear, $\phi = \operatorname{elu} + 1$: $\phi(q) = [1, 2]$; similarities 5, 4, 7 | 0.313 | 0.250 | 0.438 |

The linear rule is flatter: the orthogonal key $k_2$ still receives a quarter of the weight. More generally:

- **Sharpness.** The exponential lets softmax concentrate almost all weight on the best match; a low-degree positive feature map cannot do this as easily, which hurts precise retrieval.
- **Finite capacity.** The whole past is compressed into a `[d_phi, d_v]` matrix. At most $d_\phi$ keys can be mutually orthogonal, so once more than $d_\phi$ items are written, reads interfere.
- **No forgetting.** Plain linear attention only adds; old associations are never removed or decayed.
- **The normalizer.** $z_t^\top \phi(q_t)$ can become small relative to the numerator, or grow without bound over long sequences, which affects numerical behavior. Many later variants drop the explicit denominator and rely on normalization elsewhere in the layer.

The next three sections address the last three problems.

### 4. Delta-rule memory

Think of $S$ as an associative memory: writing pair $(k, v)$ should make the read $S^\top k$ return $v$. Additive writes ($S \leftarrow S + k v^\top$) simply pile up. The **delta rule** writes only the *error* between the target value and what the memory currently returns:

$$S_t = S_{t-1} + \beta_t \, k_t \big(v_t - S_{t-1}^\top k_t\big)^\top = \big(I - \beta_t k_t k_t^\top\big) S_{t-1} + \beta_t \, k_t v_t^\top,$$

with write strength $\beta_t \in [0, 1]$ and read $o_t = S_t^\top q_t$. This is one gradient-descent step, with learning rate $\beta_t$, on the regression loss $\tfrac{1}{2}\lVert S^\top k_t - v_t \rVert^2$. Here $S$ is `[d_k, d_v]`; some papers store the transpose and read $S q$ — align orientation before comparing equations. DeltaNet-style layers normalize keys (for example to unit length) so that $I - \beta_t k_t k_t^\top$ cannot amplify the state; check the paper for its exact normalization of queries and keys.

**Worked example ($d_k = 2$, $d_v = 1$, unit keys).**

| Step | Write | Additive memory $S$ | Delta-rule memory $S$ ($\beta = 1$) |
|---|---|---|---|
| 1 | $k_a = [0, 1]$, $v_a = 2$ | $[0, 2]$ | $[0, 2]$ |
| 2 | $k_b = [0.8, 0.6]$, $v_b = -1$ | $[-0.8, 1.4]$ | prediction $1.2$, error $-2.2$ → $[-1.76, 0.68]$ |
| Read $k_b$ | | $0.2$ (target $-1$) | $-1.0$ (exact) |
| Read $k_a$ | | $1.4$ (target 2) | $0.68$ (target 2) |

The delta rule makes the most recent write exact by correcting the interference it would otherwise cause, and rewriting an existing key *replaces* its value instead of adding to it. Because $k_a$ and $k_b$ are not orthogonal, the correction disturbs $k_a$'s value — capacity is still finite. With $\beta < 1$ the write is partial.

### 5. Gated variants

A **gate** multiplies the state by a retention factor before (or as part of) each write, so old content decays:

| Variant | State update (schematic) | Retention granularity |
|---|---|---|
| Linear attention | $S_t = S_{t-1} + k_t v_t^\top$ | none |
| Scalar decay (fixed or data-dependent) | $S_t = \alpha_t S_{t-1} + k_t v_t^\top$ | one $\alpha_t \in (0,1)$ per head |
| Channel-wise decay (e.g. gated linear attention) | $S_t = \operatorname{Diag}(\alpha_t) S_{t-1} + k_t v_t^\top$ | one value per key channel, $\alpha_t \in (0,1)^{d_k}$ |
| Delta rule | $S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ | none |
| Gated delta rule | $S_t = \alpha_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ | one scalar per head |

A scalar decay $\alpha$ shrinks a write's contribution by $\alpha^n$ after $n$ steps: with $\alpha = 0.9$, to 0.35 after 10 steps and 0.005 after 50 — a memory horizon of roughly $1/(1 - \alpha) = 10$ tokens. Channel-wise decay lets some key channels hold information for long spans ($\alpha = 0.99$, horizon ≈ 100) while others refresh quickly ($\alpha = 0.5$, horizon ≈ 2), all in the same head. Data-dependent gates are computed from the current token, so the model can decide when to forget.

### 6. Kimi Delta Attention (KDA)

KDA is described in **Kimi Linear** (arXiv 2510.26692, version v2 is the course's pinned reference). Following the course's [workshop brief](../exercises/attention-alternatives/README.md), read it in three steps: additive memory, the delta-rule correction (Kimi Linear §2.2), then **fine-grained, channel-wise retention** (§3). In the terms of §5, KDA is a gated delta rule whose retention gate acts per key channel instead of as one scalar per head. Schematically, in this note's orientation:

$$S_t = \big(I - \beta_t k_t k_t^\top\big)\operatorname{Diag}(\alpha_t)\, S_{t-1} + \beta_t \, k_t v_t^\top, \qquad o_t = S_t^\top q_t.$$

**Verify in the paper before relying on it:** the state orientation, whether decay is applied before or after the delta correction, how $\alpha_t$ and $\beta_t$ are parameterized, and how $q$ and $k$ are normalized. Cross-check against Kimi K3 v2 §2.1.1, equation (1), as the workshop brief instructs. The schematic above is a reading aid, not a transcription.

| Tensor (one head, one step) | Shape | Role |
|---|---|---|
| $q_t$, $k_t$ (normalized) | `[d_k]` | read and write addresses |
| $v_t$ | `[d_v]` | value written |
| $\beta_t$ | scalar | write strength |
| $\alpha_t$ | `[d_k]` | channel-wise retention (the memory gate) |
| $S_t$ | `[d_k, d_v]` | recurrent state |

A complete KDA layer contains more than this recurrence — input projections, a short convolution, output processing and other gates (per the workshop brief). Keep the **memory-retention gate** separate from those when mapping the toy recurrence to the paper. P14's artifact takes already-projected inputs and gates; full layers and chunkwise kernels are later scope.

### 7. Hybrid architectures

Pure recurrent layers trade retrieval precision for fixed memory. **Hybrid** models interleave them with some full softmax-attention layers, hoping to keep most of the memory savings and most of the retrieval ability. Kimi Linear §4 and K3 §2.1 describe their layouts; read the ratio and the full-attention layer type there rather than assuming.

**Memory arithmetic (my own example, decode state per sequence, element counts).** Each layer has $h = 8$ heads with $d_h = d_k = d_v = 128$.

| | Per layer | $T = 4{,}096$ | $T = 32{,}768$ | $T = 1{,}048{,}576$ |
|---|---|---|---|---|
| Softmax KV cache, $2 h T d_h$ | grows with $T$ | 8.4 M | 67.1 M | 2.15 B |
| Recurrent state, $h d_k d_v$ | constant | 131 K | 131 K | 131 K |

Crossover is at $T = d_v / 2 = 64$ tokens. Now take 24 layers with one full-attention layer in four (my ratio, chosen for arithmetic): at $T = 32{,}768$, six full layers hold $6 \times 67.1$ M ≈ 403 M elements and eighteen recurrent layers hold $18 \times 131$ K ≈ 2.4 M. That is about one quarter of an all-softmax model's 1.61 B — a large saving, but the cache still grows linearly with $T$, set by the full-attention layers. A fixed-size recurrent state does not make the whole model's cache constant.

Two cautions: these are decode-time state sizes, not training activations, which depend on the chunkwise kernel; and storage size is not speed — a hardware speed claim needs a measurement on that hardware ([L8](08-efficiency.md)).

### 8. Long-context training

**RoPE recap.** Rotary position embedding ([L2](02-transformer.md)) rotates each pair $i$ of query/key dimensions by angle $p \, \theta_i$ at position $p$, with $\theta_i = b^{-2i/d_h}$ and base $b$ (often 10,000). The score between positions $p$ and $p'$ then depends only on $p - p'$.

**Why extension is hard.** With $d_h = 128$ and $b = 10{,}000$, the fastest pair turns 1 radian per token (wavelength ≈ 6.3 tokens), while the slowest turns ≈ $1.2 \times 10^{-4}$ radians per token (wavelength ≈ 54,000 tokens). A model trained only at 4,096 tokens sees the slowest pairs sweep less than a tenth of a cycle. At longer contexts those dimensions take angles the model never saw, and attention patterns can break.

**Extension families.**

| Method | Idea | Trade-off |
|---|---|---|
| Position interpolation | Multiply positions by $T_{train}/T_{new}$ so all angles stay in the trained range | Neighboring tokens become closer in angle (with factor 8, adjacent tokens differ by 1/8 rad in the fastest pair); needs fine-tuning |
| Base scaling ("NTK-aware") | Increase $b$, which stretches slow frequencies much more than fast ones | Keeps local resolution; the choice of $b$ is empirical |
| Per-frequency schemes (e.g. YaRN) | Leave fast pairs alone, interpolate slow ones, blend in between; adjust attention temperature | More hyperparameters; usually paired with a short long-context fine-tune (the paper also reports a dynamic-scaling variant used without fine-tuning) |

**Staged context.** Attention FLOPs per token grow with $T$ ([L8](08-efficiency.md#1-counting-flops)) and activations grow with $T$, so reports typically train most tokens at a short context and then run one or more extension stages at longer lengths, with adjusted position encoding and a data mixture containing genuinely long documents. Things to check in any report: tokens per stage, how documents were packed and masked, whether short-context quality was re-measured after extension, and what parallelism made the long stages fit ([L9](09-distributed.md#8-sequence-and-context-parallelism)). Hybrid models must also state how each layer type receives position information; do not assume RoPE everywhere.

**Evaluation caution.** Low perplexity on long documents and success at retrieving a planted fact ("needle" tests) show that information can be reached, not that it can be used for multi-step reasoning across the context. Treat long-context claims as a [L4](04-evaluation.md) evaluation-design question.

### 9. Multimodal objectives (overview)

| Objective | What is trained | Where the loss sits |
|---|---|---|
| Contrastive (e.g. CLIP) | Image and text encoders map matching pairs close together | Symmetric cross-entropy over the $N \times N$ similarity matrix of a batch, with a temperature |
| Encoder + projector + LM (e.g. LLaVA) | A vision encoder's patch features are projected to width $d$ and inserted as tokens in the LM's input | Next-token loss on text tokens only; image positions are masked out of the loss (compare [L5](05-sft-peft.md) loss masks) |
| Cross-attention to visual features (e.g. Flamingo) | New cross-attention layers let a frozen LM attend to image features | Next-token loss on text interleaved with images |
| Native / early fusion | Images enter as patch embeddings or discrete tokens in the same sequence from pretraining onward | Depends on the design; read which positions carry loss |

The contrastive loss, for image embeddings $u_i$ and text embeddings $w_i$ (unit length) and temperature $\tau$, is the average of $-\log \frac{\exp(u_i^\top w_i/\tau)}{\sum_j \exp(u_i^\top w_j/\tau)}$ and its text-to-image counterpart. Questions to ask of a multimodal report: how many tokens one image costs (it eats context length), which components are frozen at each stage, the data mixture across modalities, and which tokens carry loss. The repo records K3's "native vision" as an author-reported claim; this course does not cover multimodal training beyond this overview.

### 10. Data, optimizer and RL-system choices

Reports also make claims about choices that are not architecture. For each one, ask what it was compared against.

| Area | Typical claim types | Course link |
|---|---|---|
| Data | mixture proportions, filtering, deduplication, synthetic data, decontamination | [L3](03-pretraining.md), [L4](04-evaluation.md) |
| Optimizer and schedule | optimizer family (AdamW or newer matrix-aware optimizers), schedule shape, batch-size ramp, stability fixes for logits or attention | [L1](01-neural-learning.md), [L3](03-pretraining.md) |
| Scaling | fitted laws used to choose model size and token count | [L4](04-evaluation.md) |
| RL systems | separate rollout and training engines, weight synchronization, asynchronous (off-policy) rollouts, train/inference log-probability mismatch, long or partial rollouts | [L7](07-online-rl.md), [L9](09-distributed.md) |

A recipe result ("our schedule improved the final model") is a different kind of claim from a mechanism result ("this attention rule retrieves better at equal compute"); the first rarely isolates one cause.

### 11. Reading a frontier report with a claim ledger

**The case study.** The course uses *Kimi K3: Open Frontier Intelligence* (Kimi Team, arXiv 2607.24653, version v2). The repository records that its abstract describes a 2.8 T-parameter MoE with 104 B active parameters, native vision and a million-token context, and that it covers KDA, Attention Residuals, Stable LatentMoE, pretraining data, scaling laws, optimizer and training recipe, context extension, post-training, infrastructure and evaluation. All of these are **author-reported**; none has been reproduced in this course. Do not take definitions of Attention Residuals or Stable LatentMoE from this note — read them in the report.

| Course concept | K3 route (per the workshop brief) | Question to bring |
|---|---|---|
| KDA recurrence, hybrid layout | §2 architecture, especially §2.1 and §2.1.1 eq. (1) | Which parts of the layer does the toy omit? |
| Pretraining, scaling | §3 | What separates a mechanism result from a recipe result here? |
| Post-training | §4 | Which objective, data or rollout choice maps onto L5–L7 code? |
| Systems | §5 | What resource model explains the claim, and what measurement would test it? |
| Evaluation | §6 plus the relevant methods section | Which controls exist, which are missing? |

**Ledger fields** (from the workshop brief): the claim in your own words; paper, version and exact location (section, equation, table); author-reported evidence; baseline and comparison conditions; what your toy tests; what it cannot establish; missing control or ablation; proposed test; conclusion and remaining uncertainty.

**Illustrative row — a fictional "Report X", not K3:**

| Field | Entry |
|---|---|
| Claim | Replacing three of every four attention layers with a gated recurrent layer keeps long-context retrieval while cutting decode memory |
| Location | Report X v1, architecture section, a table comparing hybrid and full-attention models |
| Author-reported evidence | Retrieval scores at several context lengths; memory per token |
| Comparison conditions | Are both models trained on the same tokens, with the same active parameters and the same context-extension recipe? |
| What a toy tests | State size vs KV cache by element count; recurrence correctness; interference as items exceed $d_k$ |
| What it cannot establish | Quality at scale; speed on the reported hardware |
| Missing control | A hybrid with scalar instead of channel-wise gates at matched compute |
| Proposed test | Fixed-size synthetic recall task, varying the number of stored pairs past $d_k$, for both gate types |
| Conclusion | Pending — fill only after the investigation |

**Habits for any report.**

- Tag every number as author-reported, independently reproduced, or measured by you.
- Ask what is held fixed: total compute, tokens, active parameters, data, tuning effort. Ablations run at small scale may not transfer.
- Separate mechanism claims, recipe claims and scale claims.
- List what reproduction would need: code and configs at a revision, data or a faithful description, tokenizer, compute, seeds, number of runs and their variance, and the exact evaluation harness and prompts.
- Write the ledger entry before reading the results table closely, then revise it; the ledger is evidence of your reasoning, not a summary of the paper.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Quadratic and recurrent linear attention disagree | Different $\phi$ in the two paths; normalizer omitted in one; mask excludes the diagonal in one path | Compare position 1 (one key) first; print both denominators |
| "Linear attention is wrong" because it differs from softmax | Expecting equality between different rules | Compare the two linear-attention forms with each other; compare with softmax only to study behavior |
| NaN or huge outputs in linear attention | Feature map not strictly positive; denominator near zero | Check $\min \phi$ and the denominator per position |
| Delta-rule state grows without bound | Keys not normalized or $\beta$ too large, so $I - \beta k k^\top$ amplifies | Check $\beta \lVert k \rVert^2$ is at most 1 (diverges beyond 2) |
| Output matches the paper's shapes but not its numbers | State orientation transposed; decay applied in a different order | Write both update orders by hand for two steps and compare |
| Processing a sequence in two segments differs from one pass | State or gates not carried across the boundary | Compare the state after the full sequence with the state after segment 2 |
| Memory estimate for an MoE model far too low | Used active instead of total parameters | Recompute model states from total parameters |
| A few experts receive almost all tokens | Load-balancing term absent or too weak; router collapse | Histogram $f_i$ per layer over training |
| Long-context claim accepted on a needle test alone | Evaluation probes access, not use | Look for multi-step long-context tasks and short-context regression checks |
| Ledger mixes reported and measured numbers | Missing provenance tags | Tag every number with its source |

## Check yourself

1. For the MoE example in §1, change to $E = 64$ experts of half the hidden width with top-4. What happens to active parameters, total parameters and the all-to-all volume?
2. Recompute the load-balancing value for a distribution of your choice in which $f$ is balanced but $P$ is not. What does the result tell you about what the loss rewards?
3. Show algebraically that the recurrent form of causal linear attention equals its masked quadratic form for the same $\phi$.
4. Predict, then run on your P14 code: how does the maximum disagreement between the two linear-attention forms change from float32 to float64? What would a large disagreement in float64 mean?
5. With $d_k = 4$, how many unit keys can you write with the additive rule and still read every value back exactly? Construct a fifth key that breaks it.
6. Derive the delta-rule update as a gradient step on $\tfrac{1}{2}\lVert S^\top k - v \rVert^2$. What does $\beta = 1$ with a unit key do to the old value stored at that key?
7. For channel-wise retention, what would you expect to see in the state of a channel with $\alpha$ near 1 versus near 0 after a long sequence? Predict, then inspect a trace from your recurrence.
8. Repeat §7's memory table for your own head count, head width and a hybrid ratio of your choice. At what context length does the hybrid's cache exceed the size of its weights?
9. Using the RoPE formula, compute the wavelength of the slowest pair for your model's $d_h$ and base. What fraction of a cycle does it cover at your training length?
10. Write one claim-ledger row for a single claim in K3 v2 §2.1, citing the exact location. Leave the conclusion blank until you have run the proposed test.
11. For that claim, list everything reproduction would require that the report does or does not provide.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [Shazeer et al., Outrageously Large Neural Networks (2017)](https://arxiv.org/abs/1701.06538) | Gating and balancing sections | The sparsely gated MoE layer |
| [Fedus, Zoph and Shazeer, Switch Transformers (2021)](https://arxiv.org/abs/2101.03961) | §2: routing, capacity factor, auxiliary loss | Top-1 routing and the balancing loss used in §1 |
| [Zoph et al., ST-MoE (2022)](https://arxiv.org/abs/2202.08906) | Stability sections (optional) | Router z-loss and training stability |
| [Jiang et al., Mixtral of Experts (2024)](https://arxiv.org/abs/2401.04088) | Architecture section | Top-2 routing with softmax over selected experts |
| [Dai et al., DeepSeekMoE (2024)](https://arxiv.org/abs/2401.06066) | Fine-grained and shared experts | Variants of expert design |
| [DeepSeek-AI, DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437) | Load-balancing section (optional) | Auxiliary-loss-free balancing |
| [Katharopoulos et al., Transformers are RNNs (ICML 2020)](https://proceedings.mlr.press/v119/katharopoulos20a.html) | §3 | Feature-map attention and its recurrent form |
| [Schlag, Irie and Schmidhuber, Linear Transformers Are Secretly Fast Weight Programmers (2021)](https://arxiv.org/abs/2102.11174) | Sections on the delta-rule update | Memory capacity and the delta rule |
| [Yang et al., Parallelizing Linear Transformers with the Delta Rule over Sequence Length (2024)](https://arxiv.org/abs/2406.06484) | Background on DeltaNet; chunkwise form (optional) | Modern delta-rule layers and their training form |
| [Yang et al., Gated Linear Attention Transformers (2023)](https://arxiv.org/abs/2312.06635) | Gating section | Channel-wise decay |
| [Yang, Kautz and Hatamizadeh, Gated Delta Networks (2024)](https://arxiv.org/abs/2412.06464) | Gated delta rule | Combining decay with the delta correction |
| [Kimi Linear v2](https://arxiv.org/html/2510.26692v2) | §2.2 delta-rule foundations, §3 fine-grained gating, [§4 hybrid architecture](https://arxiv.org/html/2510.26692v2#S4) | The KDA mechanism (P14 reference) |
| [Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear) | Optional, unpinned | Implementation route for later scope |
| [Kimi K3 v2](https://arxiv.org/html/2607.24653v2) | [§2.1](https://arxiv.org/html/2607.24653v2#S2.SS1), [§2.1.1 eq. (1)](https://arxiv.org/html/2607.24653v2#S2.SS1.SSS1), then §3–§6 as the course reaches them | The claim-ledger case study; [arXiv record](https://arxiv.org/abs/2607.24653) |
| [Lieber et al., Jamba (2024)](https://arxiv.org/abs/2403.19887) | Architecture section (optional) | Another hybrid layout (state-space + attention + MoE) |
| [Su et al., RoFormer (2021)](https://arxiv.org/abs/2104.09864) | RoPE derivation | The rotation and its relative-position property |
| [Chen et al., Extending Context Window via Positional Interpolation (2023)](https://arxiv.org/abs/2306.15595) | §2 Method | Position interpolation |
| [Peng et al., YaRN (2023)](https://arxiv.org/abs/2309.00071) | §3 Methodology, including §3.1 on "NTK-aware" base scaling | Per-frequency extension |
| [Radford et al., CLIP (2021)](https://arxiv.org/abs/2103.00020) | Approach section | Contrastive image–text objective |
| [Alayrac et al., Flamingo (2022)](https://arxiv.org/abs/2204.14198) and [Liu et al., Visual Instruction Tuning (2023)](https://arxiv.org/abs/2304.08485) | Architecture and training-objective sections (optional) | Cross-attention and projector designs |
| [CS336](https://cs336.stanford.edu/) | Architecture and MoE lectures | Course-length treatment |
