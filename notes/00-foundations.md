# L0 — Foundations for Reading and Writing Model Code

> **Milestone:** [L0](../course/ROADMAP.md#milestones) · **Builds:** [Transformer workshop](../exercises/transformer/README.md) (Step 1 shapes and products), P01 in [projects](../course/PROJECTS.md) · **Prerequisites:** comfortable Python; no prior ML assumed

Every later module manipulates the same few objects: tensors with named axes, probability distributions over a vocabulary, gradients of a scalar loss, and loss numbers measured on data the model did not train on. This note covers just enough of each to read and write LLM code without guessing. Each idea is introduced at the point where a language model needs it, so you can return here when a shape, a log, or a gradient in your own code stops making sense.

## What you should be able to do

- Predict the shape of any tensor in a small model, and explain which axis an operation acts on.
- Apply PyTorch broadcasting rules by hand and spot when broadcasting silently produced the wrong shape.
- Choose the right reduction axis (`sum`, `mean`, `max`, `softmax`) and use `keepdim` deliberately.
- Read a matrix product as a batch of dot products, and explain a linear projection, a norm and a cosine similarity.
- Write the probability of a token sequence with the chain rule of probability, and explain why models work with log-probabilities.
- Compute a gradient through a small composition with the chain rule, and describe backpropagation as a sequence of vector–Jacobian products.
- Explain what train, validation and test splits are for, what overfitting looks like in loss curves, and why leakage invalidates a held-out number.
- Trace a small computation in your own code, writing down every intermediate shape and value (the L0 demonstration).

## Core ideas

### Tensors: shape, dtype, device

A tensor is an n-dimensional array with three properties you should always know: its **shape** (size of each axis), its **dtype** (element type, such as `float32`, `bfloat16`, `int64`, `bool`) and its **device** (CPU or a GPU). Most bugs in model code are a wrong value in one of these three.

In an LLM, each axis has a meaning. The course uses this notation throughout:

| Symbol | Meaning |
|---|---|
| $B$ | batch size: number of independent sequences processed together |
| $T$ | sequence length: tokens per sequence |
| $d$ or $d_{model}$ | model width: features per token |
| $h$, $d_h$ | number of attention heads, features per head |
| $V$ | vocabulary size |
| $L$ | number of layers |

The tensors you will meet in the [Transformer workshop](../exercises/transformer/README.md):

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| token IDs | `[B, T]` | `int64` | integers in $[0, V)$; used for lookup, never for arithmetic |
| embeddings / hidden states | `[B, T, d]` | float | one vector per token |
| a projection matrix | `[d, d]` or `[d, k]` | float | learned parameter |
| attention scores | `[B, T, T]` (one head) | float | row = query position, column = key position |
| causal mask | `[T, T]` | `bool` | broadcast over the batch |
| logits | `[B, T, V]` | float | one score per vocabulary entry per position |
| per-token loss | `[B, T]` | float | reduced to a scalar before `backward()` |

Habits that pay off: print `x.shape` after every new line while building; write the expected shape as a comment; and use small, *distinct* sizes in tests (for example $B=3$, $T=5$, $d=6$) so that a swapped axis produces an error instead of a silently valid result. If $B = T$, a transposition bug can pass every shape assertion.

**Views versus copies.** `x.view(...)`, `x.transpose(...)`, slicing and `x.unsqueeze(...)` usually return a *view*: a new tensor header over the same memory. Writing into a view changes the original. `transpose` produces a non-contiguous view, which `view` cannot always reshape; `reshape` will copy when it has to. This matters when you split and merge attention heads in L2.

### Broadcasting

Broadcasting lets an operation combine tensors of different shapes without copying data. PyTorch compares shapes **from the last axis backwards**. Two axes are compatible if they are equal, or one of them is 1, or one tensor has run out of axes. The result takes the larger size on each axis.

| Left | Right | Result | Where it appears |
|---|---|---|---|
| `[B, T, d]` | `[d]` | `[B, T, d]` | adding a bias or a norm gain to every token |
| `[B, T, T]` | `[T, T]` | `[B, T, T]` | applying one causal mask to every sequence |
| `[B, T, d]` | `[T, d]` | `[B, T, d]` | adding learned position vectors |
| `[B, T, 1]` | `[B, T, d]` | `[B, T, d]` | dividing each token by its own norm (needs `keepdim=True`) |
| `[B]` | `[B, 1]` | `[B, B]` | **usually a bug**: an outer "sum" nobody asked for |

The last row is the classic silent failure. The snippet below shows the behavior; nothing errors.

```python
import torch
a = torch.tensor([1.0, 2.0, 3.0])        # shape [3]
b = torch.tensor([[10.0], [20.0], [30.0]])  # shape [3, 1]
print((a - b).shape)                      # torch.Size([3, 3]), not [3]
```

When a loss value looks plausible but training behaves strangely, check whether a target and a prediction broadcast into a square matrix.

### Reductions and the axis question

A reduction collapses one or more axes: `sum`, `mean`, `max`, `argmax`, `logsumexp`, `norm`. `softmax` is not a reduction (its output has the same shape), but it normalizes along one axis, so it raises the same question: *which axis?*

| Operation in an LLM | Axis | Why |
|---|---|---|
| softmax over attention scores | last (keys) | each query distributes weight over the keys it can see |
| softmax over logits | last (vocabulary) | one distribution over next tokens per position |
| LayerNorm / RMSNorm statistics | last (features) | each token is normalized on its own |
| averaging the loss | over valid tokens | padded positions should not count |

`keepdim=True` keeps the reduced axis with size 1, so the result broadcasts back against the input. For `x` of shape `[4, 6]`, `x.sum(dim=-1)` has shape `[4]`, while `x.sum(dim=-1, keepdim=True)` has shape `[4, 1]` and can divide `x` row by row.

**Worked example (masked mean).** Two sequences have per-token losses `[[2.0, 1.0, 3.0], [4.0, 0.5, 0.5]]`, but the last two positions of the second sequence are padding. With mask `[[1, 1, 1], [1, 0, 0]]`, the token-weighted mean is $(2 + 1 + 3 + 4) / 4 = 2.5$. A plain `mean()` would give $11/6 \approx 1.83$, and a mean of per-sequence means would give $(2 + 4)/2 = 3.0$. All three are "the loss"; they weight tokens differently. [L3](03-pretraining.md), [L4](04-evaluation.md) and [L9](09-distributed.md) return to this choice.

### Matrix products as batches of dot products

The dot product of two vectors $u, v \in \mathbb{R}^n$ is $u \cdot v = \sum_{i=1}^{n} u_i v_i$. It is large when the vectors point the same way and are long. A matrix product $C = AB$ with $A$ of shape `[m, n]` and $B$ of shape `[n, p]` is a table of dot products: $C_{ij}$ is row $i$ of $A$ dotted with column $j$ of $B$.

Two readings of the same product appear constantly:

1. **Projection of every token.** For hidden states $X$ of shape `[B, T, d]` and weight $W$ of shape `[d, k]`, `X @ W` has shape `[B, T, k]`. The same linear map is applied to each of the $B \cdot T$ token vectors independently. Leading axes act as batch axes.
2. **All-pairs similarity.** For $Q$ and $K$ both of shape `[B, T, k]`, `Q @ K.transpose(-2, -1)` has shape `[B, T, T]`. Entry $[b, i, j]$ is the dot product of token $i$'s query with token $j$'s key, inside sequence $b$ only. Sequences never mix.

| Expression | Shapes in | Shape out | Contracted axis |
|---|---|---|---|
| `X @ W` | `[B, T, d]`, `[d, k]` | `[B, T, k]` | $d$ |
| `Q @ K.transpose(-2, -1)` | `[B, T, k]`, `[B, k, T]` | `[B, T, T]` | $k$ |
| `A @ V` | `[B, T, T]`, `[B, T, k]` | `[B, T, k]` | key positions |

**Worked example.** Let one token vector be $x = [1, 2, -1]$ and

$$W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 3 \end{bmatrix}.$$

Then $xW = [1\cdot1 + 2\cdot0 + (-1)\cdot2,\; 1\cdot0 + 2\cdot1 + (-1)\cdot3] = [-1, -1]$. Each output feature is a dot product of $x$ with one column of $W$, which is why "a linear layer is a bank of learned feature detectors" is a fair description.

`torch.einsum` writes the contraction explicitly: `torch.einsum("btd,dk->btk", X, W)` names every axis. It is a good way to check your understanding of a product even if you write the final code with `@`.

### Norms, angles and projections

The Euclidean norm is $\lVert x \rVert_2 = \sqrt{\sum_i x_i^2}$. Cosine similarity, $\cos\theta = \dfrac{u \cdot v}{\lVert u \rVert\,\lVert v \rVert}$, compares direction only. A dot product mixes direction with length, which is why the length of query and key vectors matters in attention.

A useful fact for L2: if $u$ and $v$ have $n$ independent entries with mean 0 and variance 1, their dot product has mean 0 and variance $n$, so its typical size grows like $\sqrt{n}$. With $n = 100$, raw dot products are typically around $\pm 10$. Attention divides by $\sqrt{d_h}$ for exactly this reason.

Normalization layers (L2) divide each token vector by a norm-like statistic so that its scale stays controlled as it passes through many layers. Weight decay (L1) shrinks the norm of parameters.

### Probability over a vocabulary

A language model outputs a **categorical distribution** at each position: a vector $p \in \mathbb{R}^V$ with $p_i \ge 0$ and $\sum_i p_i = 1$, where $p_i$ is the model's probability that the next token is $i$.

The **chain rule of probability** turns a sequence probability into a product of next-token probabilities:

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}).$$

Here $x_t$ is the token at position $t$, and $p(x_t \mid x_{<t})$ is a conditional probability: the distribution over position $t$ given everything before it. This identity is exact for any distribution; an autoregressive model chooses to *parameterize* each factor with a neural network. Causal masking (L2) is what makes each factor depend only on the past.

The **expectation** of a function $f$ under distribution $p$ is $\mathbb{E}_{x \sim p}[f(x)] = \sum_x p(x) f(x)$. Training losses are expectations over data that we estimate with averages over minibatches; evaluation numbers are averages over held-out tokens. Both are estimates with noise ([L4](04-evaluation.md) quantifies it).

Three quantities recur from L1 to L7:

| Quantity | Definition | Reading |
|---|---|---|
| entropy | $H(p) = -\sum_i p_i \log p_i$ | how spread out a distribution is |
| cross-entropy | $H(q, p) = -\sum_i q_i \log p_i$ | average surprise of model $p$ on data from $q$; with a one-hot target it is $-\log p_y$ |
| KL divergence | $\mathrm{KL}(q \,\Vert\, p) = \sum_i q_i \log \dfrac{q_i}{p_i}$ | extra surprise from using $p$ instead of $q$; always $\ge 0$ |

They satisfy $H(q, p) = H(q) + \mathrm{KL}(q \,\Vert\, p)$. Minimizing cross-entropy against fixed data therefore minimizes the KL from the data distribution to the model. KL reappears as a penalty in preference learning ([L6](06-preferences.md)) and online RL ([L7](07-online-rl.md)).

### Why models live in log-space

Probabilities of whole sequences are products of many numbers below 1, and they underflow. Suppose each of 200 tokens has probability 0.05. The product is $0.05^{200} \approx 6 \times 10^{-261}$, far below the smallest positive `float32` (about $1.4 \times 10^{-45}$), so it is stored as exactly 0. Every sequence would look equally impossible.

Taking logarithms turns products into sums: $\log p(x_{1:T}) = \sum_t \log p(x_t \mid x_{<t})$. For the example, $200 \ln 0.05 \approx -599.1$, a perfectly ordinary number. Log-probabilities are also what the loss uses directly (negative log-likelihood), and their gradients are better behaved than those of raw probabilities.

The operation you need in log-space is **log-sum-exp**: $\operatorname{LSE}(z) = \log \sum_i e^{z_i}$. Computed naively, $e^{z_i}$ overflows for $z_i$ above about 88 in `float32`. The stable form subtracts the maximum $m = \max_i z_i$ first:

$$\operatorname{LSE}(z) = m + \log \sum_i e^{z_i - m}.$$

Every exponent is now $\le 0$, so nothing overflows, and at least one term equals 1, so the sum cannot underflow to 0. L1 uses this to build a stable softmax and log-softmax. Use `torch.logsumexp` and `torch.log_softmax` rather than composing `log` and `exp` yourself.

### Derivatives, gradients and the chain rule

For a scalar function $f(\theta)$ of a parameter vector $\theta \in \mathbb{R}^n$, the **gradient** $\nabla_\theta f$ is the vector of partial derivatives $\partial f / \partial \theta_i$. It has the same shape as $\theta$ and points in the direction of steepest increase. Training moves parameters a small step against it.

The **chain rule** says that for a composition $f(g(\theta))$, $\dfrac{df}{d\theta} = \dfrac{df}{dg}\,\dfrac{dg}{d\theta}$. A model is a long composition, so its gradient is a long product of local derivatives.

**Worked example.** Take a one-weight model with squared error: $z = wx + b$, $r = z - y$, $f = r^2$. With $w = 2$, $x = 3$, $b = 1$, $y = 4$:

| Step | Forward value | Local derivative | Gradient of $f$ |
|---|---|---|---|
| $z = wx + b$ | $7$ | $\partial z/\partial w = x = 3$, $\partial z/\partial b = 1$ | $\partial f/\partial w = 6 \cdot 3 = 18$, $\partial f/\partial b = 6$ |
| $r = z - y$ | $3$ | $\partial r/\partial z = 1$ | $\partial f/\partial z = 6$ |
| $f = r^2$ | $9$ | $\partial f/\partial r = 2r = 6$ | $\partial f/\partial r = 6$ |

Read the last column bottom-up: that is **backpropagation**, carrying the gradient of the final scalar backwards one step at a time. You can check it numerically: at $w = 2.001$, $f = (7.003 - 4)^2 = 9.018009$, so $(9.018009 - 9)/0.001 \approx 18.0$. L1 turns this check into a tool.

### Jacobians and vector-Jacobian products

When a step maps a vector to a vector, $y = g(x)$ with $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^m$, its derivative is the **Jacobian** $J \in \mathbb{R}^{m \times n}$ with $J_{ij} = \partial y_i / \partial x_j$. Backpropagation never needs to build $J$. It receives the upstream gradient $\bar{y} = \partial f / \partial y$ (same shape as $y$) and needs $\bar{x} = \partial f / \partial x = J^\top \bar{y}$ (same shape as $x$). This **vector–Jacobian product** (VJP) is usually cheap to compute directly from the structure of $g$.

| Step $y = g(x)$ | Jacobian structure | VJP $\bar{x}$ |
|---|---|---|
| elementwise, $y_i = \sigma(x_i)$ | diagonal | $\bar{x}_i = \sigma'(x_i)\,\bar{y}_i$ |
| linear, $y = xW$ with $x$: `[n]`, $W$: `[n, m]` | $J = W^\top$ | $\bar{x} = \bar{y}\,W^\top$, and $\bar{W} = x^\top \bar{y}$ (outer product) |
| softmax, $p = \operatorname{softmax}(z)$ | $\operatorname{diag}(p) - p p^\top$ | $\bar{z} = p \odot (\bar{p} - (\bar{p} \cdot p))$ |
| sum, $y = \sum_i x_i$ | row of ones | $\bar{x}_i = \bar{y}$ for every $i$ |

The shape rule is the most useful check: **the gradient with respect to a tensor always has that tensor's shape.** For a batch $X$ of shape `[N, n]` and $W$ of shape `[n, m]`, $\bar{W} = X^\top \bar{Y}$ has shape `[n, m]`, and it is a *sum* over the $N$ rows. Gradients from different examples add up in shared parameters.

This is why PyTorch's `backward()` without arguments requires a scalar output: it starts the chain with $\bar{f} = 1$. Calling `y.backward(v)` on a non-scalar $y$ supplies $\bar{y} = v$ and computes one VJP. **Forward-mode** differentiation computes the other product, a Jacobian–vector product $J v$, which is efficient when inputs are few and outputs many; training uses reverse mode because there are millions of inputs (parameters) and one output (the loss).

### Generalization: train, validation, test

A model can reduce training loss by fitting general patterns or by memorizing training examples. **Generalization** is performance on new data from the same source. You can only measure it on data the model never trained on.

| Split | Used for | Rule |
|---|---|---|
| train | computing gradients | everything the optimizer sees |
| validation (dev) | choosing learning rate, model size, when to stop, which run is best | look as often as needed; decisions made here are fitted to it |
| test | one final, honest estimate | touch once, after decisions are frozen |

**Overfitting** shows up as training loss that keeps falling while validation loss flattens and then rises. The gap between them is the generalization gap. Small models on large text corpora often see each token once and overfit little; small datasets, many epochs or repeated documents make it appear quickly.

Intentionally overfitting is a debugging tool: a correct model and training loop should drive the loss on one tiny fixed batch close to zero. Failure to do so points at a bug, not at a lack of data. [L3](03-pretraining.md) makes this an explicit check.

**Leakage** is any path by which test information reaches training: the same document (or a near-duplicate) in both splits, a split made after tuning, normalization statistics computed on all data, or a benchmark copied into a web crawl. A leaked held-out number is not wrong arithmetic; it measures memorization instead of generalization. For text, split by *document* (or by source), never by individual tokens of one document. See [L3](03-pretraining.md) for deduplication and split construction.

### Tracing a computation

The L0 demonstration asks you to trace a small computation in your own code. A trace is a table with one row per line of code: the expression, the shape you predicted, the shape you observed, and (for tiny inputs) the values. Write the prediction column *before* running anything; the rows where prediction and observation disagree are what you learn from.

A filled-in example for a toy scorer, two inputs of width 3 and four output classes, shows the format:

| Line | Predicted shape | Observed | Notes |
|---|---|---|---|
| `x` (input) | `[2, 3]` | `[2, 3]` | rows are examples |
| `w` (parameter) | `[3, 4]` | `[3, 4]` | `requires_grad=True` |
| `z = x @ w` | `[2, 4]` | `[2, 4]` | contracted the width axis |
| `lp = torch.log_softmax(z, dim=-1)` | `[2, 4]` | `[2, 4]` | each row's `exp` sums to 1 |
| `lp[torch.arange(2), y]` with `y` of shape `[2]` | `[2]` | `[2]` | one log-probability per example |
| `loss = -(...).mean()` | `[]` (scalar) | `[]` | ready for `backward()` |
| `w.grad` after `backward()` | `[3, 4]` | `[3, 4]` | same shape as `w` |

Useful tools while tracing: `print(t.shape, t.dtype)`, `assert t.shape == (...)`, a debugger breakpoint, and, for gradients, `t.grad_fn` to see which operation produced a tensor. In the Transformer workshop, the same table for [attention.py](../exercises/transformer/attention.py) is a good first piece of evidence.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Shape is `[N, N]` where you expected `[N]` | broadcasting `[N]` against `[N, 1]` | print both operand shapes right before the operation |
| Tests pass but results are wrong for real inputs | test sizes equal ($B = T$ or $T = d$) hide a swapped axis | rerun with distinct sizes such as 3, 5, 6 |
| `RuntimeError` from `view` after `transpose` | tensor is non-contiguous | check `x.is_contiguous()`; use `reshape` or `.contiguous()` |
| Changing a slice changed the original tensor | slices and views share storage | compare `a.data_ptr()` values, or `.clone()` when you need a copy |
| Normalization divides the wrong entries | reduction without `keepdim`, then a broadcast along the wrong axis | print the shape of the statistic; it should be `[..., 1]` |
| Log-likelihood is `-inf` or `nan` | `log` of a probability that underflowed to 0, or `exp` overflow | compute in log-space with `log_softmax` / `logsumexp` |
| Loss value depends on batch padding | mean over all positions instead of valid tokens | recompute with an explicit mask and token count |
| Validation loss suspiciously close to training loss on a tiny dataset | duplicated or overlapping documents across splits | hash documents (or normalized text) and intersect the splits |

## Check yourself

1. In your [attention scaffold](../exercises/transformer/attention.py), write the shape of every intermediate tensor next to its line for an input of shape `[3, 5, 6]`. Then run it and compare.
2. Predict the result shape of `[4, 1, 6] + [5, 1]`. Run it. Which axes were stretched?
3. For a tensor of shape `[2, 3, 4]`, what are the shapes of `x.sum(0)`, `x.sum(-1)`, `x.sum(-1, keepdim=True)` and `x.mean((0, 1))`? Predict, then run.
4. Explain, using the dot-product reading, why `Q @ K.transpose(-2, -1)` can never mix information between two sequences in the batch.
5. Compute $\operatorname{LSE}([50, 51, 52])$ by hand with the max-subtraction trick, then check `torch.logsumexp` in `float32`. Try the naive formula on `[100, 101, 102]` in `float32` and explain what you see.
6. A model assigns probability 0.9 to each of 1,000 tokens. What is the log-probability of the sequence, and what would happen if you multiplied the probabilities in `float32`?
7. Extend the worked chain-rule example: what is $\partial f / \partial x$? Verify your answer with a finite difference in Python.
8. For $y = xW$ with $x$ of shape `[7, 3]` and $W$ of shape `[3, 2]`, what shapes do $\bar{x}$ and $\bar{W}$ have? Write the products that produce them and check with autograd.
9. Why does `loss.backward()` fail if `loss` has shape `[B]`? Give two different ways to make it run, and say how they weight examples differently.
10. You tuned learning rate and model size by checking the test set after each run. What does the final test number now measure, and how would you restore an honest estimate?
11. Describe one way that splitting a text corpus by individual lines (rather than by document) could inflate a validation score.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | "Tensors" and "Autograd" pages | official walkthrough of tensor creation, operations, devices and gradients |
| [PyTorch broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) | whole page | the exact rules, including in-place restrictions |
| [PyTorch tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html) | whole page | which operations return views and what contiguity means |
| [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) | Ch. 2 (linear algebra), Ch. 3 (probability, information theory), Ch. 4.1 (overflow/underflow), Ch. 5.2–5.3 (capacity, overfitting, validation sets) | compact, rigorous background for every section above |
| [Mathematics for Machine Learning (Deisenroth, Faisal, Ong)](https://mml-book.github.io/) | Ch. 5 (vector calculus, Jacobians, backpropagation) | optional: a slower derivation of gradients of vector functions |
| [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) | §3.1–3.2 (Forward Mode, Reverse Mode) | optional: why reverse mode computes vector–Jacobian products |
| [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | "Optional Reading - Vector Calculus using autograd" section | connects the VJP idea to `backward(v)` |

Next: [L1 — Neural learning](01-neural-learning.md) uses these pieces to build and train a model.
