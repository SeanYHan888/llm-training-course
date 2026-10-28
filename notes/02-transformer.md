# L2 — The Transformer Language Model

> **Milestone:** [L2](../course/ROADMAP.md#milestones) · **Builds:** [Transformer workshop](../exercises/transformer/README.md) ([attention.py](../exercises/transformer/attention.py)), P02 tokenizer, P03 attention and P04 tiny decoder in [projects](../course/PROJECTS.md), cached decoding in [Inference 01](../exercises/inference/README.md) · **Prerequisites:** [L0](00-foundations.md) shapes and matrix products; [L1](01-neural-learning.md) softmax, cross-entropy and autograd (needed before training)

A decoder-only Transformer turns a sequence of token IDs into, at every position, a distribution over the next token. This note follows that computation from raw text to logits and back to text: tokenization, embeddings, causal self-attention, heads, residual connections, normalization, position information, the MLP, the output head, the shifted next-token loss, sampling, and the KV cache that makes generation efficient. Each section gives the math and shapes you need to build and test the component yourself; the implementations belong to you.

## What you should be able to do

- Explain byte-level BPE: how merges are learned from a corpus, how they are applied to new text, why decoding round-trips any string, and how special tokens are handled.
- Trace token IDs through embeddings, attention, MLP and output head, stating every tensor's shape.
- Write the equations of causal self-attention with Q/K/V, justify the $1/\sqrt{d_h}$ scale, and explain how the causal mask enforces "no information from the future."
- Describe how multi-head attention splits and recombines the width, including the reshape/transpose step and the output projection.
- Explain the residual stream, LayerNorm versus RMSNorm, and why modern models use pre-norm.
- Compare learned absolute positions with rotary position embeddings (RoPE).
- Describe standard and gated (SwiGLU) MLPs and their parameter counts, and the output head with or without weight tying.
- Build shifted next-token labels and explain what an off-by-one does to training.
- Describe greedy decoding, temperature, top-k and top-p sampling.
- Explain prefill and decode, what the KV cache stores and why reuse is valid, and derive its memory as a formula.
- Trace shapes and masks through a tiny decoder; validate tokenizer round trip and causality; diagnose an off-by-one label or mask bug (the L2 demonstration).

## Core ideas

### The whole computation at a glance

| Stage | Operation | Output shape |
|---|---|---|
| text → IDs | tokenizer | `[B, T]` (`int64`) |
| embedding | row lookup in $E \in \mathbb{R}^{V \times d}$ (plus learned positions, if used) | `[B, T, d]` |
| $L$ decoder blocks | each: $x \leftarrow x + \text{Attn}(\text{Norm}(x))$, then $x \leftarrow x + \text{MLP}(\text{Norm}(x))$ | `[B, T, d]` |
| final norm | Norm | `[B, T, d]` |
| output head | $\cdot\, W_U$, $W_U \in \mathbb{R}^{d \times V}$ | `[B, T, V]` logits |
| training | cross-entropy of logits at $t$ against token $t+1$ | scalar |
| generation | sample from logits at the last position, append, repeat | `[B, T+1]`, … |

The width $d$ stays constant through all blocks. Only the tokenizer, embedding and head touch the vocabulary.

### Tokenization and byte-level BPE

A model cannot read characters directly; it reads integers from a fixed vocabulary. The tokenizer defines that mapping. The design tension: a vocabulary of single characters gives long sequences (attention cost grows with $T^2$), while a vocabulary of whole words cannot represent new words. **Subword** tokenizers sit between them: frequent strings become single tokens, rare strings are spelled out from smaller pieces.

**Bytes as the base alphabet.** Text is first encoded as UTF-8 bytes. Every string, in any language or with any emoji, becomes a sequence of values in $0..255$. ASCII characters take one byte; "é" takes two (`0xC3 0xA9`); most emoji take four. Starting the vocabulary with all 256 byte values guarantees that *any* input can be encoded, with no "unknown" token.

**Byte-pair encoding (BPE)** builds the rest of the vocabulary by repeatedly merging the most frequent adjacent pair of tokens.

*Learning merges (training the tokenizer):*

1. **Pre-tokenize.** Split the training text into chunks (roughly words, with punctuation and whitespace handled by a regular expression) so merges never cross chunk boundaries. GPT-2's tokenizer, for example, keeps a leading space attached to the following word, so " cat" and "cat" are different chunks.
2. Represent each chunk as a sequence of single-byte tokens, and count how often each distinct chunk occurs.
3. Count every adjacent token pair across all chunks, weighted by chunk frequency.
4. Pick the most frequent pair, using a fixed, documented tie-break rule. Create a new token for it, record the merge with its **rank** (1 for the first merge, 2 for the second, …), and replace that pair everywhere it occurs.
5. Repeat steps 3–4 until the vocabulary reaches the target size: $V = 256 + \text{number of merges} + \text{number of special tokens}$.

Recounting all pairs after every merge is slow for real corpora; efficient trainers update only the counts affected by the latest merge. The result is the same.

*Worked example.* Suppose pre-tokenization produced three chunk types: `tata` (4 times), `tap` (3 times) and `pat` (2 times).

| Round | Pair counts (weighted) | Merge chosen | Chunks afterwards |
|---|---|---|---|
| 1 | `t a`: 4·2 + 3 = 11; `a t`: 4 + 2 = 6; `a p`: 3; `p a`: 2 | `t a` → `ta` (rank 1) | `ta ta`, `ta p`, `p a t` |
| 2 | `ta ta`: 4; `ta p`: 3; `p a`: 2; `a t`: 2 | `ta ta` → `tata` (rank 2) | `tata`, `ta p`, `p a t` |
| 3 | `ta p`: 3; `p a`: 2; `a t`: 2 | `ta p` → `tap` (rank 3) | `tata`, `tap`, `p a t` |

The vocabulary is now the 256 bytes plus `ta`, `tata` and `tap`.

*Applying merges (encoding new text).* For each chunk, start from bytes and repeatedly merge the adjacent pair with the **lowest rank** present in the merge table, until no remaining pair appears in the table. Encoding the unseen chunk `patata`:

| Step | Tokens | Ranked pairs present | Action |
|---|---|---|---|
| start | `p a t a t a` | `t a` (rank 1) | merge both occurrences |
| 1 | `p a ta ta` | `ta ta` (rank 2) | merge |
| 2 | `p a tata` | none (`p a` and `a tata` are not merges) | stop |

The result is three tokens, `p`, `a`, `tata`. Note that encoding follows merge *rank*, not frequency in the new text, and it replays the training order: that is what makes encoding deterministic.

**The round-trip property.** Decoding concatenates each token's bytes and decodes the result as UTF-8. Because every token is just a byte string, and encoding only groups bytes without changing them, `decode(encode(s)) == s` for every string `s` (given no text normalization and consistent special-token handling). This is the property P02 asks you to validate on varied inputs: accents, emoji, whitespace runs, code.

The reverse does not hold: `encode(decode(ids))` can differ from `ids`. The ID sequence `[t, a]` decodes to "ta", which re-encodes as the single token `ta`. Models are trained on canonical encodings, so non-canonical sequences are unusual inputs.

A practical consequence for generation: a single token may end in the middle of a multi-byte character. Decoding a partial output with strict UTF-8 raises an error or shows a replacement character; streaming decoders hold back incomplete bytes until the character is complete.

**Special tokens** are vocabulary entries that never come from merges, such as an end-of-text marker that separates documents. They are given reserved IDs (usually after the merge tokens). When encoding, the text is first split on the special-token strings so they are not broken into bytes; the pieces between them are encoded normally. This raises a policy question: if user-supplied text contains the literal string of a special token, should it become the special ID or plain text? Treating it as special lets text inject control tokens. Some libraries refuse such input by default; check the behavior of the release you pin. Chat templates ([L5](05-sft-peft.md)) depend heavily on special tokens.

| Choice | Effect |
|---|---|
| larger $V$ | shorter sequences, larger embedding and head matrices, rarer tokens get fewer training updates |
| smaller $V$ | longer sequences (more attention cost), smaller matrices |
| pre-tokenization regex | controls which strings can ever become one token (for example, whether digits group) |

### Embeddings

The embedding table $E \in \mathbb{R}^{V \times d}$ holds one learned vector per token. Embedding is row lookup: for IDs of shape `[B, T]`, `E[ids]` has shape `[B, T, d]`. It is mathematically the same as multiplying a one-hot vector by $E$, without forming the one-hot.

**Worked example.** With $V = 6$, $d = 3$ and IDs `[[4, 1, 4]]`, the output is rows 4, 1 and 4 of $E$, shape `[1, 3, 3]`. In the backward pass, row 4 receives the *sum* of the gradients from both positions, row 1 receives one, and the other four rows receive zero. Tokens that never appear in a batch get no gradient from it.

`torch.nn.Embedding(V, d)` stores this table; its `weight` has shape `[V, d]`.

### Causal self-attention

Attention lets each position build its new representation from a weighted mix of other positions. For input $X$ of shape `[T, d]` (one sequence; the batch axis just comes along), three learned projections produce

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V,$$

where row $i$ of $Q$ is the **query** of position $i$ (what it is looking for), row $j$ of $K$ is the **key** of position $j$ (what it offers for matching), and row $j$ of $V$ is the **value** of position $j$ (what it contributes if selected). For a single head with $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$:

$$S = \frac{QK^\top}{\sqrt{d_h}}, \qquad S_{ij} \leftarrow -\infty \ \text{ for } j > i, \qquad A = \operatorname{softmax}_{\text{row}}(S), \qquad O = AV.$$

$S_{ij}$ scores how well query $i$ matches key $j$. Setting the scores of future keys ($j > i$) to $-\infty$ before the softmax makes their weights exactly $e^{-\infty} = 0$. Each row of $A$ is then a probability distribution over positions $0..i$, and output row $i$ is a weighted average of the visible value vectors.

| Tensor | Shape (one head, with batch) |
|---|---|
| $X$ | `[B, T, d]` |
| $W_Q$, $W_K$, $W_V$ | `[d, d_h]` each |
| $Q$, $K$, $V$ | `[B, T, d_h]` |
| $S$, mask, $A$ | `[B, T, T]` (mask `[T, T]`, broadcast) |
| $O$ | `[B, T, d_h]` |

Your [attention.py](../exercises/transformer/attention.py) implements this with $d_h = d$, which is why its scale uses the width.

**Why divide by $\sqrt{d_h}$.** If query and key entries are roughly independent with mean 0 and variance 1, the dot product of two $d_h$-dimensional vectors has variance $d_h$ ([L0](00-foundations.md#norms-angles-and-projections)). With $d_h = 100$ the raw scores have a standard deviation of about 10. Softmax of numbers that far apart is nearly one-hot, and the softmax gradient $\operatorname{diag}(p) - pp^\top$ is then nearly zero, so learning stalls. Dividing by $\sqrt{d_h}$ restores unit variance at initialization. The scale depends on the length of the vectors being dotted, which is $d_h$, not the model width, once you have several heads.

**Worked example.** Three positions, $d_h = 4$ (so the scale is $1/2$), with projections already computed:

$$Q = \begin{bmatrix} 1&0&1&0 \\ 0&2&0&0 \\ 1&1&1&1 \end{bmatrix}, \quad K = \begin{bmatrix} 2&0&0&0 \\ 0&1&0&1 \\ 1&0&1&2 \end{bmatrix}, \quad V = \begin{bmatrix} 4&0&0&0 \\ 0&2&0&0 \\ 0&0&2&2 \end{bmatrix}.$$

| Row (query) | Raw $QK^\top$ row | Scaled and masked | Weights $A$ | Output $O$ |
|---|---|---|---|---|
| 0 | $[2, 0, 2]$ | $[1, -\infty, -\infty]$ | $[1, 0, 0]$ | $[4, 0, 0, 0]$ |
| 1 | $[0, 2, 0]$ | $[0, 1, -\infty]$ | $[0.2689, 0.7311, 0]$ | $[1.0758, 1.4621, 0, 0]$ |
| 2 | $[2, 2, 4]$ | $[1, 1, 2]$ | $[0.2119, 0.2119, 0.5761]$ | $[0.8478, 0.4239, 1.1522, 1.1522]$ |

Row 0 would have given position 2 a high raw score, but the mask removes it: position 0 can only copy its own value. Changing anything in row 2 of $K$ or $V$ cannot alter outputs 0 or 1. That is the causality property your checks test.

**Masking details.**

- Use $-\infty$ (or the most negative finite value of the dtype) *before* the softmax. Multiplying weights by a 0/1 mask *after* the softmax leaves rows that no longer sum to 1.
- With a causal mask alone, every row keeps at least its diagonal entry. If you combine it with a padding mask, a row can become fully masked; softmax of all $-\infty$ is `nan`. Decide how such rows are handled.
- With right padding (real tokens first, padding after), the causal mask already prevents real tokens from attending to padding; only the padded positions' own outputs are meaningless, and they are excluded from the loss. Left padding, common in batched generation, needs an explicit padding mask.
- When the number of queries differs from the number of keys (as in cached decoding), "future" has to be defined by absolute positions, not by a square lower-triangular matrix.

PyTorch's `torch.nn.functional.scaled_dot_product_attention` computes the same operation (with `is_causal=True` for the square causal case) and can dispatch to fused kernels. It is a useful independent reference to compare your implementation against; read its documentation for how it interprets boolean versus additive masks.

### Multi-head attention

One attention pattern per layer is limiting: a position may need to look at the previous token for syntax and at a distant name for meaning at the same time. **Multi-head attention** runs $h$ attention operations in parallel on smaller vectors, $d_h = d / h$, and concatenates the results.

Head $i$ uses its own column block of the projection matrices: $W_Q^{(i)}$ is columns $i \cdot d_h$ to $(i+1) \cdot d_h - 1$ of a single $W_Q \in \mathbb{R}^{d \times d}$, and likewise for $K$ and $V$. Each head computes its own scores, mask and weights. The head outputs are concatenated back to width $d$ and mixed by an **output projection** $W_O \in \mathbb{R}^{d \times d}$:

$$\operatorname{MHA}(X) = \big[\,O^{(1)}, \dots, O^{(h)}\,\big]\,W_O, \qquad O^{(i)} = \operatorname{softmax}\!\Big(\frac{Q^{(i)} K^{(i)\top}}{\sqrt{d_h}} + M\Big)\,V^{(i)},$$

where $M$ is the additive causal mask (0 for visible, $-\infty$ for future).

In tensor form the heads become a separate axis so one batched product handles all of them:

| Step | Shape |
|---|---|
| projected $Q$ (all heads together) | `[B, T, d]` |
| split the last axis into heads | `[B, T, h, d_h]` |
| move heads before time | `[B, h, T, d_h]` |
| scores and weights | `[B, h, T, T]` |
| head outputs | `[B, h, T, d_h]` |
| move time back before heads | `[B, T, h, d_h]` |
| merge heads | `[B, T, d]` |
| after $W_O$ | `[B, T, d]` |

The order matters. Reshaping `[B, T, d]` straight into `[B, h, T, d_h]` without the intermediate `[B, T, h, d_h]` and transpose produces the right shape with the wrong contents: it slices the flattened token-and-feature memory into blocks that mix different positions. Shape assertions cannot catch it; an independent per-head reference can. Merging back after a transpose also needs a contiguous tensor (see [L0](00-foundations.md#tensors-shape-dtype-device)).

Parameter count: $W_Q, W_K, W_V, W_O$ are each $d \times d$, so $4d^2$ per attention layer, the same for any $h$ that divides $d$. Heads change how capacity is organized, not how much there is.

**Fewer key/value heads.** In multi-query attention (MQA) all query heads share one key head and one value head; in grouped-query attention (GQA) the $h$ query heads are split into $h_{kv}$ groups, each sharing one K/V head. The K/V projections shrink to $d \times (h_{kv} d_h)$. The motivation is the KV cache below: its size scales with $h_{kv}$, not $h$.

### The residual stream

Each sublayer adds its output to its input instead of replacing it: $x \leftarrow x + f(x)$. The running vector $x$ of shape `[B, T, d]` is called the **residual stream**. Every attention and MLP sublayer reads from it and writes an update into it, and the output head reads its final state.

Two consequences:

- **Gradient flow.** $\partial(x + f(x))/\partial x = I + \partial f/\partial x$. The identity term carries gradient from the loss straight back to early layers, so deep stacks remain trainable.
- **A shared communication channel.** Because updates add, a feature written by layer 2 is still available to layer 10 unless something removes it. Interpretability work ([interpretability](interpretability.md)) leans on this view.

All sublayers must output width $d$ for the addition to work; this is why attention ends with $W_O$ and the MLP ends with a projection back to $d$.

### LayerNorm, RMSNorm and pre-norm

Normalization rescales each token vector so its size stays controlled as many residual updates accumulate. Statistics are computed over the **feature axis of each token separately**, never across the batch or sequence.

**LayerNorm** for a token vector $x \in \mathbb{R}^d$, with learned gain $\gamma$ and bias $\beta$ (both `[d]`) and small $\epsilon$:

$$\mu = \frac{1}{d}\sum_i x_i, \qquad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2, \qquad \operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta.$$

**RMSNorm** drops the mean subtraction and the bias:

$$\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}, \qquad \operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\operatorname{RMS}(x)}.$$

**Worked example.** For $x = [2, 4, 6, 8]$ with $\gamma = 1$, $\beta = 0$ and negligible $\epsilon$:

| | Statistic | Output |
|---|---|---|
| LayerNorm | $\mu = 5$, $\sigma^2 = 5$ | $[-1.342, -0.447, 0.447, 1.342]$ (mean 0, variance 1) |
| RMSNorm | $\operatorname{RMS} = \sqrt{30} \approx 5.477$ | $[0.365, 0.730, 1.095, 1.461]$ (mean square 1, mean not removed) |

RMSNorm is cheaper and works as well in practice for language models, which is why many recent architectures use it. Note that LayerNorm divides by $d$ (the biased variance); a check against `torch.var` must pass `unbiased=False` (or `correction=0`). PyTorch provides `nn.LayerNorm`, and `nn.RMSNorm` since PyTorch 2.4; its default `eps` follows the input dtype, so check the documentation of your pinned version.

**Where the norm goes.** The original Transformer used **post-norm**, $x \leftarrow \operatorname{Norm}(x + f(x))$. Most current decoders use **pre-norm**, $x \leftarrow x + f(\operatorname{Norm}(x))$, with one extra **final norm** before the output head. In pre-norm the residual path from input to output contains no normalization, which keeps gradients well scaled in deep stacks and makes training less sensitive to warmup. Forgetting the final norm in a pre-norm model is a common bug: the head then reads an unnormalized stream whose scale grows with depth.

### Position information

Without positions, attention cannot tell order. Permuting the input tokens permutes the queries, keys and values the same way, and (ignoring the mask) the outputs are permuted identically: "dog bites man" and "man bites dog" would get the same set of representations. The causal mask leaks some order information, but models need explicit positions to work well.

**Learned absolute positions** add a learned vector per position: $x_t = E[\text{id}_t] + P[t]$ with $P \in \mathbb{R}^{T_{max} \times d}$. This is simple and is what the course's first decoder uses. Its limits: the model has no vector for positions at or beyond $T_{max}$, and nothing in the parameterization says that positions 7 and 8 are related. (The original Transformer used fixed sinusoids instead of learned vectors.)

**Rotary position embeddings (RoPE)** encode position by rotating queries and keys, not by adding to the residual stream. Split a head's $d_h$ features into $d_h/2$ pairs. Pair $i$ at position $m$ is rotated by angle $m\theta_i$, with frequencies

$$\theta_i = b^{-2i/d_h}, \qquad i = 0, \dots, \tfrac{d_h}{2} - 1,$$

where the base $b$ is a hyperparameter (10000 in the original RoPE paper; long-context models often change it). For one pair $(u_1, u_2)$:

$$\begin{bmatrix} u_1' \\ u_2' \end{bmatrix} = \begin{bmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{bmatrix} \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}.$$

Rotations satisfy $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$. So for a query at position $m$ and a key at position $n$,

$$\big(R_m q\big)^\top \big(R_n k\big) = q^\top R_{n-m}\, k:$$

the attention score depends on the content and on the **offset** $n - m$, not on absolute positions. Early pairs rotate quickly (sensitive to short offsets); late pairs rotate slowly (long offsets).

**Worked example.** One pair with $\theta = 0.5$ rad and $q = k = [1, 0]$:

| Query position $m$ | Key position $n$ | Offset | Score $\cos((n - m)\theta)$ |
|---|---|---|---|
| 2 | 2 | 0 | 1.000 |
| 5 | 5 | 0 | 1.000 |
| 2 | 4 | 2 | 0.540 |
| 7 | 9 | 2 | 0.540 |

Facts to keep straight when you add RoPE:

- It is applied to $Q$ and $K$ after projection and after splitting into heads, and **not** to $V$.
- Implementations pair features differently: adjacent features $(2i, 2i+1)$ or the two halves of the head $(i, i + d_h/2)$. Both are valid; weights trained with one convention give wrong results under the other.
- In cached decoding, keys are stored already rotated, so the new token must be rotated with its true absolute position. A position counter that restarts at 0 each step is a classic cache bug.

ALiBi and other schemes add position-dependent biases to the scores instead; they are optional depth here.

### The MLP and gated MLPs

After attention mixes information across positions, the **MLP** transforms each position independently (same weights at every position):

$$\operatorname{MLP}(x) = \phi(xW_1)\,W_2, \qquad W_1 \in \mathbb{R}^{d \times d_{ff}},\ W_2 \in \mathbb{R}^{d_{ff} \times d},$$

with $\phi$ = GELU and $d_{ff} = 4d$ as a common default. That is $2 \cdot d \cdot d_{ff} = 8d^2$ parameters (ignoring biases), twice the attention layer.

A **gated** MLP multiplies an activated branch by a second, linear branch elementwise. **SwiGLU** uses SiLU for the activated branch:

$$\operatorname{SwiGLU}(x) = \big(\operatorname{SiLU}(xW_1) \odot xW_3\big)\,W_2, \qquad W_1, W_3 \in \mathbb{R}^{d \times d_{ff}},\ W_2 \in \mathbb{R}^{d_{ff} \times d}.$$

| Tensor | Shape |
|---|---|
| $x$ | `[B, T, d]` |
| $xW_1$, $xW_3$, their product | `[B, T, d_ff]` |
| output | `[B, T, d]` |

With three matrices, the same parameter budget as a $4d$ standard MLP requires $3 \cdot d \cdot d_{ff} = 8d^2$, so $d_{ff} \approx \tfrac{8}{3}d$, usually rounded to a hardware-friendly multiple. The gate lets the network switch features on and off multiplicatively. In Shazeer's experiments, several gated variants (SwiGLU among them) improved quality over ReLU and GELU MLPs with parameters and computation held equal.

### The decoder block and the output head

A pre-norm block with input $x$ of shape `[B, T, d]` computes

$$a = x + \operatorname{MHA}(\operatorname{Norm}_1(x)), \qquad y = a + \operatorname{MLP}(\operatorname{Norm}_2(a)).$$

After $L$ blocks, a final norm and the output head produce logits:

$$\text{logits} = \operatorname{Norm}_f(x_L)\,W_U, \qquad W_U \in \mathbb{R}^{d \times V}, \qquad \text{logits}: \texttt{[B, T, V]}.$$

**Weight tying** sets $W_U = E^\top$: the same matrix embeds tokens and scores them. It saves $V \cdot d$ parameters and couples the input and output meanings of each token. GPT-2 ties them; many larger recent models keep them separate. Both appear in practice; the choice is a recorded configuration decision, not a correctness issue.

**Parameter count** (no biases, $h_{kv} = h$, standard $4d$ MLP, ignoring norm gains): about $12 L d^2$ for the blocks, plus $V d$ for the embedding, plus another $V d$ if the head is untied, plus $T_{max}\,d$ for learned positions. For example, $d = 384$, $L = 6$, $V = 8192$, $T_{max} = 256$ gives $10{,}616{,}832 + 3{,}145{,}728 + 98{,}304 \approx 13.9$M parameters tied, or about $17.0$M untied. In small models the vocabulary matrices are a large share of the total. Compare such an estimate with `sum(p.numel() for p in model.parameters())` for your own decoder.

### Shifted labels: next-token prediction

The logits at position $t$ are a prediction for the token at position $t + 1$. Given one tokenized sequence of length $T + 1$, the training pair is

| | Positions | Example (IDs `[5, 9, 2, 7, 3]`) |
|---|---|---|
| inputs | $0 .. T-1$ | `[5, 9, 2, 7]` |
| targets | $1 .. T$ | `[9, 2, 7, 3]` |

The loss is the mean cross-entropy over all $B \cdot T$ predicted positions ([L1](01-neural-learning.md#negative-log-likelihood-and-cross-entropy)): logits flattened to `[B·T, V]`, targets to `[B·T]`. Equivalently, run the model on all $T + 1$ tokens and drop the last position's logits and the first token's target. Padded or otherwise excluded targets are set to the ignore index.

Because of the causal mask, one forward pass trains $T$ predictions at once: position $t$ sees tokens $0..t$ and predicts $t + 1$. This parallelism during training is a large part of why Transformers displaced recurrent models.

**What off-by-one bugs look like.**

| Bug | What the model can do | Symptom |
|---|---|---|
| no shift (target = current token) | copy its own input, which attention can always see | training loss collapses toward 0 within a few steps; generation repeats the last token |
| shift in the wrong direction (target = previous token) | copy the previous token, also visible | very low training loss; useless generation |
| correct shift, mask missing or wrong | read the answer from position $t+1$ through attention | training loss suspiciously low; generation poor because the future does not exist at inference |
| labels shifted twice (your shift plus a library's) | predict two tokens ahead | loss stays high; plausible-looking but poor text |

A loss that falls far below what is plausible for the data is a bug signal, not a success. The most direct test is the one in your attention checks lifted to the full model: change a token at position $j$ and confirm that logits at positions before $j$ do not change at all. Libraries differ in whether the model shifts labels internally; Hugging Face causal-LM classes do when you pass `labels` (see [L5](05-sft-peft.md)), so check the pinned version's documentation before shifting yourself.

### Generation and sampling

Generation is a loop: run the model on the current sequence, take the logits at the **last** position, choose a token, append it, repeat until an end-of-text token or a length limit. With learned absolute positions, the sequence fed to the model must also stay within $T_{max}$.

Ways to choose the token from logits $z$:

| Method | Rule |
|---|---|
| greedy | $\arg\max_i z_i$; deterministic; tends to loop and repeat |
| temperature $\tau$ | sample from $\operatorname{softmax}(z / \tau)$; $\tau < 1$ sharpens, $\tau > 1$ flattens, $\tau \to 0$ approaches greedy |
| top-k | keep the $k$ highest-probability tokens, renormalize, sample |
| top-p (nucleus) | keep the smallest set of highest-probability tokens whose total probability is at least $p$, renormalize, sample |

**Worked example: temperature.** For logits $[3.0, 2.0, 0.5]$:

| $\tau$ | Probabilities |
|---|---|
| 0.5 | $[0.8756, 0.1185, 0.0059]$ |
| 1 | $[0.6897, 0.2537, 0.0566]$ |
| 2 | $[0.5283, 0.3204, 0.1513]$ |

**Worked example: top-k versus top-p.** For logits $[2.0, 1.5, 1.0, -1.0]$, $\tau = 1$, the probabilities are $[0.494, 0.300, 0.182, 0.025]$ with cumulative sums $0.494, 0.794, 0.975, 1.0$.

| Filter | Kept | Renormalized |
|---|---|---|
| top-k, $k = 2$ | first two | $[0.6225, 0.3775, 0, 0]$ |
| top-p, $p = 0.75$ | first two (0.794 ≥ 0.75) | $[0.6225, 0.3775, 0, 0]$ |
| top-p, $p = 0.9$ | first three (0.975 ≥ 0.9) | $[0.5065, 0.3072, 0.1863, 0]$ |

Top-k keeps a fixed count regardless of how confident the model is; top-p keeps fewer tokens when the distribution is peaked and more when it is flat. Libraries apply these filters in a particular order, and the order can change results. Hugging Face `generate` currently applies temperature, then top-k, then top-p; other libraries may differ, so check your pinned version. For reproducible sampling, fix the random seed and record the full sampling configuration.

Random weights produce random tokens. In early builds, generation tests the mechanics (shapes, lengths, stopping, causal reuse), not text quality.

### The KV cache

Naive generation reruns the whole sequence for every new token. At step $t$, it recomputes the keys and values of positions $0..t-1$ in every layer, even though they are identical to what it computed in the previous step. Over $n$ generated tokens, that repeats a growing amount of work at each step.

**Why reuse is valid.** In a causal decoder, the hidden state at position $j$ in every layer depends only on tokens $0..j$. Appending a token never changes the representation of an earlier position, so the keys and values computed for earlier positions in every layer remain correct. Only the new position needs new computation. Your causality checks are the evidence for this: "changing a future input does not change earlier outputs" is the same statement.

**What is cached.** For each layer, the keys and values of every past position, after projection and (with RoPE) after rotation:

| Cached | Not cached |
|---|---|
| $K$ and $V$ per layer, per K/V head, per position | queries (each is used only once, at its own step) |
| a record of how many positions are filled | attention weights, MLP activations, residual stream values |

**Prefill and decode.** Generation has two phases:

| Phase | Input | Work | Typical bottleneck |
|---|---|---|---|
| prefill | the whole prompt, $T_p$ tokens, in one pass | computes all layers for all prompt positions in parallel, fills the cache with $T_p$ entries per layer, and returns logits for the last position (the first generated token) | arithmetic (large matrix products) |
| decode | one new token per step | computes $q, k, v$ for the new position only, appends $k, v$ to each layer's cache, attends the new query against all cached keys | memory traffic (reading all weights and the whole cache to produce one token) |

During decode the query axis has length 1 while the key axis has length $t$, so the scores have shape `[B, h, 1, t]`. The mask and the position used for the new token must be derived from absolute positions, as noted in the masking and RoPE sections. A cache must also be emptied (or replaced) between independent requests.

**Memory formula.** Count elements, then convert to bytes. For one layer, one sequence and one position, the key holds $h_{kv} \cdot d_h$ numbers and the value holds the same. Multiply by positions, sequences and layers:

$$N_{\text{elements}} = 2 \cdot L \cdot B \cdot T \cdot h_{kv} \cdot d_h, \qquad N_{\text{bytes}} = N_{\text{elements}} \cdot s,$$

where $T$ is the number of cached positions, $h_{kv}$ the number of key/value heads, $d_h$ the head dimension, and $s$ the bytes per element (4 for `float32`, 2 for `bfloat16`/`float16`). The factor 2 is K plus V. Without GQA, $h_{kv} d_h = d$, so $N_{\text{elements}} = 2 L B T d$.

As a tiny check of the counting (not a realistic model): $L = 3$, $B = 1$, $T = 10$, $h_{kv} = 2$, $d_h = 8$ gives $2 \cdot 3 \cdot 1 \cdot 10 \cdot 2 \cdot 8 = 960$ elements, or 3,840 bytes in `float32`.

Read the formula for scaling behavior: the cache grows linearly with sequence length and with batch size, does not depend on $V$ or $d_{ff}$, and shrinks by a factor of $h / h_{kv}$ with grouped-query attention. At long contexts and large batches it can exceed the model weights, which is what serving systems are built around ([inference and serving](inference-serving.md)). Inference Lab B asks you to verify the formula against the actual tensors in your cache and to explain what it leaves out.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| `decode(encode(s)) != s` for some strings | normalization, stripped whitespace, or special-token strings handled inconsistently | test accents, emoji, repeated spaces, newlines and a literal special-token string |
| Tokenizer output differs between two runs or machines | nondeterministic tie-break when choosing merges | log the merge list with ranks and diff it |
| Encoding uses different tokens than expected | applying merges by frequency in the input instead of by learned rank | encode a word by hand from the merge table and compare |
| Attention rows do not sum to 1 | mask applied after softmax, or softmax over the wrong axis | sum `weights` over the last axis |
| `nan` in attention output | a fully masked row (padding combined with causal mask) | find rows whose mask is all false |
| Earlier logits change when a later token changes | missing or mis-oriented causal mask, or a norm/statistic computed across time | perturb one late token and diff earlier logits |
| Multi-head output has the right shape but disagrees with a per-head reference | reshape without the transpose, or merge without making the tensor contiguous | compare one head's output with attention computed from that head's column block |
| Scores saturate; attention almost one-hot at initialization | missing scale, or scale by $d$ instead of $d_h$ | print the standard deviation of raw scores |
| Loss near 0 after a few steps on real text | labels not shifted, or attention can see the target | compare inputs and targets for one row; run the causality test |
| Loss plateaus well above expected | labels shifted twice, or targets misaligned by padding | print the decoded input/target pairs for one row |
| Output depends on sequence length beyond $T_{max}$ / crash at long prompts | learned position table too short | assert sequence length against $T_{max}$ before embedding |
| Cached and uncached outputs differ only after the first decode step | new token's position reset to 0, or mask built for equal query/key lengths | log the position used per step; compare against full-prefix recomputation |
| Second request's output depends on the first | cache not reset between requests | run the same request twice from a fresh and a reused cache |

## Check yourself

1. Train the merge table from the worked example by hand on a different tiny corpus of your own. Then encode a word that never appeared in it. Which merges fired, in which order?
2. Find a string for which byte-level BPE produces more tokens than characters, and one where it produces far fewer. What does that imply about sequence length for different languages or for code?
3. Give an ID sequence that is valid but non-canonical for your tokenizer. What does `encode(decode(ids))` return?
4. Explain what goes wrong if pre-tokenization is skipped and merges can cross spaces.
5. In your `single_head_attention`, predict how the attention weights change if you remove the scale and multiply all inputs by 10. Run it.
6. Your attention mask is built from the score matrix's last two dimensions. Predict what it does if there is one query and six keys. What should the mask be in that case?
7. Write down the shapes at each step of the multi-head reshape for $B = 2$, $T = 5$, $d = 12$, $h = 3$. Which step makes the tensor non-contiguous?
8. Why is the attention parameter count $4d^2$ for any number of heads? How does it change with GQA when $h_{kv} < h$?
9. Compute LayerNorm and RMSNorm by hand for $x = [1, 1, 1, 5]$. Which one leaves the vector's mean in place? Check with PyTorch.
10. Show with one 2-dimensional pair that a RoPE score depends only on the offset: pick $q$, $k$, $\theta$ and two position pairs with the same offset, and compute both scores.
11. Take a working tiny decoder and remove the label shift. Predict the training-loss curve and the generated text before running it. Then restore the shift and break the mask instead; how do the two failures differ?
12. For logits $[1.0, 0.8, 0.6, 0.4, 0.2]$, which tokens survive top-k with $k = 2$ and top-p with $p = 0.5$? With temperature 0.5 applied first?
13. Explain why queries are not cached but keys and values are. What would change if the model were not causal?
14. Using the memory formula, how does the cache size change if you double $T$, halve $B$, and switch from $h_{kv} = h$ to $h_{kv} = h/4$? Derive it before calculating.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | §3.2 (scaled dot-product and multi-head attention), §3.3 (position-wise feed-forward), §3.5 (positional encoding) | the original mechanism; note it describes an encoder–decoder with post-norm, not our decoder-only pre-norm model |
| [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | §2.2 (input representation) | byte-level BPE and why merges are restricted across character categories |
| [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | §3.2 (byte pair encoding) | the original BPE-for-text procedure |
| [Hugging Face LLM course: BPE tokenization](https://huggingface.co/learn/llm-course/chapter6/5) | whole section | an alternative walk-through of training and applying merges |
| [tiktoken](https://github.com/openai/tiktoken) | README; special-token arguments | a production byte-level BPE encoder to compare round trips and special-token handling against |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | §3 (the proposed rotary formulation and its properties) | derivation of RoPE and its relative-position property |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | the method section | what RMSNorm removes and why it still works |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) | the method section | the original LayerNorm definition |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | introduction and analysis sections | why pre-norm trains more stably than post-norm |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | whole paper (short) | SwiGLU and the $\tfrac{2}{3}$ width adjustment |
| [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) | whole paper | the case for weight tying |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | sections introducing nucleus sampling | why greedy and pure sampling fail, and the top-p rule |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | method section | grouped K/V heads and their effect on cache size |
| [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | mask semantics, `is_causal`, scale | a reference implementation to test against; check your pinned version |
| [Hugging Face cache explanation](https://huggingface.co/docs/transformers/en/cache_explanation) | whole page | how a production library organizes the KV cache |
| [CS336 Spring 2026](https://cs336.stanford.edu/) and [assignment 1](https://github.com/stanford-cs336/assignment1-basics) | architecture and tokenizer lectures; the assignment handout's component specifications | optional depth: a full specification of a similar build (respect its AI-use rules) |

Next: [L3 — Pretraining](03-pretraining.md) trains this model on real data. Cached decoding, paging and batching continue in [inference and serving](inference-serving.md).
