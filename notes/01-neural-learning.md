# L1 — How a Neural Model Learns

> **Milestone:** [L1](../course/ROADMAP.md#milestones) · **Builds:** P01 (training loop and gradient check) in [projects](../course/PROJECTS.md); the training step of P04 in the [Transformer workshop](../exercises/transformer/README.md) · **Prerequisites:** [L0](00-foundations.md)

Training is four operations repeated many times: compute predictions (forward), score them with a scalar loss, compute the gradient of that loss with respect to every parameter (backward), and change the parameters (update). This note explains each operation, the PyTorch machinery that performs it, and the ways it silently breaks. Everything here is architecture-independent: a Transformer is trained exactly like the small MLP used in the examples.

## What you should be able to do

- Distinguish parameters, activations, buffers and optimizer state, and say which ones receive gradients.
- Write the math of a linear layer and an MLP, give their shapes and parameter counts, and explain why a nonlinearity is required.
- Turn logits into probabilities and log-probabilities stably, and compute negative log-likelihood (cross-entropy) and its gradient by hand.
- Explain what autograd records, what a leaf tensor is, why `.grad` accumulates, and what `zero_grad`, `detach` and `no_grad` change.
- Verify an analytic gradient with a central-difference check and interpret the relative error.
- Explain minibatch gradient estimates, and the update rules for SGD, momentum, Adam and AdamW (decoupled weight decay).
- Describe learning-rate warmup and decay, and common regularizers.
- Write your own small training loop with a numerical gradient check, and diagnose absent gradients, accumulated gradients and train/validation divergence (the L1 demonstration).

## Core ideas

### What lives in a model

| Kind | Example | Receives a gradient? | Saved in a checkpoint? |
|---|---|---|---|
| parameter | weight matrices, biases, norm gains, embedding table | yes (if `requires_grad=True`) | yes |
| buffer | fixed position tables, running statistics | no | usually |
| activation | hidden states, attention scores, logits | flows *through* them; not stored in `.grad` | no |
| optimizer state | momentum, Adam's moment estimates, step count | no | yes, if you want exact resume ([L3](03-pretraining.md)) |

In PyTorch, a parameter is a tensor wrapped in `nn.Parameter` and assigned as an attribute of an `nn.Module`; `model.parameters()` finds it. A tensor stored as a plain attribute is not a parameter, so the optimizer never sees it. This is one of the most common causes of "my layer does not learn."

### Linear layers and MLPs

A linear (affine) layer maps an input vector $x \in \mathbb{R}^{d_{in}}$ to $y = xW + b$, with weight $W \in \mathbb{R}^{d_{in} \times d_{out}}$ and bias $b \in \mathbb{R}^{d_{out}}$. It has $d_{in} d_{out} + d_{out}$ parameters. Applied to a batch, the same $W$ and $b$ act on every row.

`torch.nn.Linear(d_in, d_out)` stores its weight with shape `[d_out, d_in]` and computes $x W^\top + b$. The course's attention scaffold instead stores raw matrices of shape `[d_in, d_out]` and computes `x @ w`. Both are fine; mixing the conventions is a source of transposition bugs.

A multilayer perceptron (MLP) stacks linear layers with a nonlinear **activation** $\phi$ in between:

$$\operatorname{MLP}(x) = \phi(xW_1 + b_1)\,W_2 + b_2.$$

| Tensor | Shape |
|---|---|
| input $x$ | `[N, d_in]` |
| $W_1$, $b_1$ | `[d_in, d_hidden]`, `[d_hidden]` |
| hidden $\phi(xW_1 + b_1)$ | `[N, d_hidden]` |
| $W_2$, $b_2$ | `[d_hidden, d_out]`, `[d_out]` |
| output | `[N, d_out]` |

Without $\phi$, two linear layers collapse into one: $(xW_1)W_2 = x(W_1W_2)$. The nonlinearity is what lets depth add expressive power.

| Activation | Formula | Notes |
|---|---|---|
| ReLU | $\max(0, x)$ | gradient is exactly 0 for negative inputs; a unit that is always negative never recovers ("dead") |
| GELU | $x\,\Phi(x)$, $\Phi$ = standard normal CDF | smooth; standard in GPT-style MLPs; a tanh approximation is also common |
| SiLU / Swish | $x\,\sigma(x)$, $\sigma$ = logistic sigmoid | smooth; used inside gated MLPs ([L2](02-transformer.md)) |

**Initialization** sets the scale of the starting weights. If the entries of $x$ have variance 1 and the entries of $W$ are independent with variance $s^2$, each output entry of $xW$ has variance $d_{in}\,s^2$. Choosing $s^2 \approx 1/d_{in}$ keeps activations at a stable scale through many layers. PyTorch modules come with default initializers (check your pinned version); models often override them.

### Logits, softmax and log-softmax

A classifier's (or language model's) final layer outputs **logits** $z \in \mathbb{R}^V$: unbounded real scores, one per class. Softmax turns them into a distribution:

$$p_i = \operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.$$

Adding the same constant $c$ to every logit leaves $p$ unchanged, because $e^{c}$ cancels. That invariance is what makes the stable form possible: subtract $m = \max_j z_j$ before exponentiating.

**Worked example (overflow).** For logits $[1000, 999, 998]$ in `float32`, $e^{1000}$ overflows to `inf` and the naive softmax returns `nan`. After subtracting 1000 the logits are $[0, -1, -2]$; the exponentials are $[1, 0.3679, 0.1353]$ with sum $1.5032$, giving $p = [0.6652, 0.2447, 0.0900]$. The answer was always well defined; only the naive arithmetic failed.

The log of softmax has its own stable form, using log-sum-exp from [L0](00-foundations.md#why-models-live-in-log-space):

$$\log p_i = z_i - \operatorname{LSE}(z).$$

Computing `softmax` and then `log` loses precision for tiny probabilities (they round to 0 and the log becomes `-inf`). Use `torch.log_softmax`, which never forms the small probability explicitly.

### Negative log-likelihood and cross-entropy

For one example with correct class $y$, the **negative log-likelihood** (NLL) loss is

$$\ell = -\log p_y = -z_y + \operatorname{LSE}(z).$$

With a one-hot target this equals the cross-entropy between the target and the model distribution. Over a dataset, the loss is the mean of $\ell$ over examples (for a language model: over predicted tokens).

Its gradient with respect to the logits is remarkably simple:

$$\frac{\partial \ell}{\partial z_i} = p_i - \mathbb{1}[i = y].$$

Training pushes the correct logit up by $1 - p_y$ and every other logit down by its probability.

**Worked example.** Logits $z = [2.0, 0.5, -1.0]$, target $y = 1$.

| Quantity | Value |
|---|---|
| $e^{z}$ | $[7.389, 1.649, 0.368]$, sum $9.406$ |
| $p$ | $[0.7856, 0.1753, 0.0391]$ |
| $\ell = -\log p_1$ | $1.741$ |
| $\partial \ell / \partial z$ | $[0.7856, -0.8247, 0.0391]$ |

The gradient sums to zero across classes (a consequence of the shift invariance above).

Two sanity values are worth memorizing:

- A model that outputs uniform probabilities has loss $\ln V$. With $V = 1000$ that is about $6.91$. A freshly initialized model with small output weights should start near this value; a much larger initial loss suggests over-confident initialization, a much smaller one suggests leakage of the target.
- **Perplexity** is $\exp(\text{mean NLL per token})$, the effective number of equally likely choices. [L4](04-evaluation.md) covers when comparisons of loss or perplexity are valid.

In PyTorch, `torch.nn.functional.cross_entropy(logits, targets)` expects **logits** (it applies log-softmax internally), logits of shape `[N, C]` and integer targets of shape `[N]`. For a language model you flatten `[B, T, V]` to `[B·T, V]` and `[B, T]` to `[B·T]`. Targets equal to `ignore_index` (default `-100`; confirm in your pinned version) are excluded, and the default `reduction="mean"` averages over the remaining targets. Passing probabilities instead of logits changes the meaning silently. Float targets of shape `[N, C]` are read as class probabilities (supported since PyTorch 1.10), and `ignore_index` does not apply to them.

### Autograd: what PyTorch records

When an operation's input has `requires_grad=True`, PyTorch records the operation in a **computation graph**: each result tensor remembers the function that produced it (`tensor.grad_fn`) and its inputs. `loss.backward()` walks this graph from the scalar loss back to its sources, computing vector–Jacobian products ([L0](00-foundations.md#jacobians-and-vector-jacobian-products)) at each node.

- A **leaf tensor** is one the user created directly (not the output of a recorded operation). Parameters are leaves. After `backward()`, gradients are written into `leaf.grad` only for leaves with `requires_grad=True`.
- Intermediate (non-leaf) tensors get gradients during the pass, but their `.grad` stays `None` unless you call `.retain_grad()` on them first.
- By default `backward()` frees the graph's saved tensors. Calling it twice on the same graph raises an error unless the first call used `retain_graph=True`.

**`.grad` accumulates.** Each `backward()` *adds* into `.grad` rather than replacing it:

```python
import torch
w = torch.tensor(3.0, requires_grad=True)
(2 * w).backward()
print(w.grad)   # tensor(2.)
(2 * w).backward()
print(w.grad)   # tensor(4.)  -- summed with the previous gradient
```

This is deliberate (it lets you sum gradients from several micro-batches, or from several losses), but it means you must clear gradients before each new step. `optimizer.zero_grad()` (or `model.zero_grad()`) does that. Since PyTorch 2.0 it sets `.grad` to `None` by default rather than filling zeros (`set_to_none=True`; the default was `False` through 1.13); keep this in mind if your code reads `.grad` directly.

**Stopping gradients.** Three tools look similar but differ:

| Tool | What it does | Typical use |
|---|---|---|
| `t.detach()` | returns a tensor sharing `t`'s data with no graph history | treat a value as a constant: a target, a logged number, a frozen reference model's output |
| `with torch.no_grad():` | operations inside are not recorded | evaluation, manual parameter updates, generation |
| `p.requires_grad_(False)` | the leaf stops requesting gradients | freezing part of a model (adapters in [L5](05-sft-peft.md)) |

```python
a = torch.tensor(2.0, requires_grad=True)
b = a * a                 # b = 4, depends on a
c = b.detach() * a        # first factor is now a constant 4
c.backward()
print(a.grad)             # tensor(4.), not 3 * a**2 = 12
```

Without `detach`, $c = a^3$ and the gradient would be $3a^2 = 12$. A misplaced `detach` gives a gradient that is wrong but not zero, which is harder to spot than a missing one. `torch.inference_mode()` is a stricter, faster variant of `no_grad` for pure inference.

In-place operations (`x += 1`, `x.relu_()`) on tensors that autograd saved for backward raise an error at `backward()` time. Prefer out-of-place versions while learning.

### Numerical gradient checks

An analytic gradient can be checked against the definition of a derivative. For a scalar loss $f(\theta)$ and one coordinate $\theta_i$, the **central difference** is

$$g_i^{num} = \frac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2\epsilon},$$

where $e_i$ is the unit vector for coordinate $i$. Its error shrinks like $\epsilon^2$ (the one-sided difference only like $\epsilon$). Compare it with the autograd value $g_i$ using a **relative error**:

$$\text{rel}_i = \frac{\lvert g_i - g_i^{num} \rvert}{\max(\lvert g_i \rvert, \lvert g_i^{num} \rvert, \delta)},$$

with a small floor $\delta$ (for example $10^{-12}$) to avoid dividing by zero.

Practical rules:

- Run the check in `float64`. In `float32`, rounding error in $f$ divided by a small $2\epsilon$ swamps the result. With `float64` and $\epsilon$ around $10^{-6}$, a correct gradient typically gives relative errors around $10^{-7}$ or smaller; errors near $10^{-2}$ or above indicate a bug.
- Make the function deterministic: fixed inputs, no dropout, no random sampling inside $f$.
- Check a handful of randomly chosen coordinates of each parameter tensor, not only the first one; bugs often affect a slice.
- Avoid kinks: ReLU at exactly 0 or a `max` with ties makes the two sides disagree legitimately.
- Use a tiny model and batch. The check costs two forward passes per coordinate.

PyTorch provides `torch.autograd.gradcheck` for checking a function's gradient automatically. Writing the check yourself for P01 is the point of the exercise; the library function is useful afterwards for checking custom operations.

### Minibatches

The full-data loss is an average over $N$ examples, $F(\theta) = \frac{1}{N}\sum_{n} \ell_n(\theta)$, and its gradient is the average of per-example gradients. A **minibatch** of $B$ randomly chosen examples gives an unbiased estimate of that gradient at a fraction of the cost. Its noise shrinks roughly like $1/\sqrt{B}$: quadrupling the batch halves the standard deviation of the estimate.

| Term | Meaning |
|---|---|
| step | one optimizer update |
| epoch | one pass over the training set |
| shuffling | reordering examples each epoch so batches are not correlated with file order |

With `reduction="mean"`, the gradient size does not depend on $B$. With `reduction="sum"` it grows with $B$, which acts like multiplying the learning rate by $B$. If you sum gradients over several micro-batches before one update (gradient accumulation), divide by the number of micro-batches (or tokens) so the step matches a single large batch. [L9](09-distributed.md) returns to this when gradients are combined across processes.

### SGD and momentum

Plain **stochastic gradient descent** updates every parameter with learning rate $\eta$:

$$\theta \leftarrow \theta - \eta\, g, \qquad g = \nabla_\theta \ell_{\text{batch}}(\theta).$$

**Worked example.** Minimize $f(w) = (w - 3)^2$, so $g = 2(w - 3)$. From $w = 0$ with $\eta = 0.1$: $w$ becomes $0.6$, then $1.08$, then $1.464$. Each step multiplies the distance to the minimum by $1 - 2\eta = 0.8$. With $\eta = 1.1$ the factor is $-1.2$: each step overshoots further and the iterates diverge. Too-large learning rates are the most common cause of loss spikes and `nan`.

**Momentum** keeps a running velocity $v$ (in PyTorch's convention, $v \leftarrow \mu v + g$, then $\theta \leftarrow \theta - \eta v$, with $\mu$ around 0.9). Steps in a consistent direction build up; oscillating components cancel. Other texts scale $g$ by $1 - \mu$; the conventions differ by a rescaling of $\eta$.

### Adam and AdamW

**Adam** keeps two exponential moving averages per parameter: the mean of gradients and the mean of squared gradients. With decay rates $\beta_1$ (typically 0.9) and $\beta_2$ (typically 0.95–0.999), step counter $t$ and small $\epsilon$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.$$

All operations are elementwise. The bias corrections $1 - \beta^t$ undo the fact that $m$ and $v$ start at zero. On the first step, $\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$, so each coordinate moves by about $\eta \cdot \operatorname{sign}(g_1)$, regardless of how large or small the gradient was. Adam's step size is set by $\eta$ much more than by the gradient's scale, which is why it is forgiving across layers with different gradient magnitudes.

Adam stores two extra tensors the size of the model ($m$ and $v$), plus a step count. For a model with $P$ parameters this state is $2P$ numbers, usually in `float32`; [L8](08-efficiency.md) counts it in memory budgets, and [L3](03-pretraining.md) saves it in checkpoints.

**Weight decay** shrinks parameters toward zero to discourage large weights. Adding an L2 penalty $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss adds $\lambda\theta$ to the gradient. In SGD that is equivalent to shrinking $\theta$ by $\eta\lambda\theta$ each step. In Adam it is not: the added $\lambda\theta$ is divided by $\sqrt{\hat{v}}$ like the rest of the gradient, so parameters with large gradient history are decayed less. **AdamW** decouples the two: it takes the Adam step on the loss gradient alone and applies the shrinkage separately,

$$\theta_t = \theta_{t-1} - \eta\,\lambda\,\theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.$$

Implementations differ in details (whether the decay term is multiplied by the current learning rate, default $\lambda$, which parameters are decayed). In PyTorch's `torch.optim.AdamW` the decay is multiplied by the learning rate and the default `weight_decay` is 0.01 (both as of the PyTorch 2.14 documentation); confirm them for your pinned release. Common practice excludes biases and normalization gains from decay by putting them in a separate parameter group; practice for embeddings varies.

### Learning-rate schedules

The learning rate usually changes over training. A standard shape for Transformer training is linear **warmup** followed by **decay**.

With peak rate $\eta_{max}$, minimum $\eta_{min}$, $W$ warmup steps and $S$ total steps:

$$\eta_t = \begin{cases} \eta_{max}\,\dfrac{t}{W} & t < W \\[2ex] \eta_{min} + \tfrac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\pi\,\dfrac{t - W}{S - W}\right)\right) & W \le t \le S \end{cases}$$

**Worked example.** With $\eta_{max} = 3\times10^{-4}$, $\eta_{min} = 3\times10^{-5}$, $W = 100$, $S = 1000$: at $t = 50$, $\eta = 1.5\times10^{-4}$; at $t = 100$, $\eta = 3\times10^{-4}$; at $t = 550$ (halfway through decay) the cosine term is 0, so $\eta = 3\times10^{-5} + 1.35\times10^{-4} = 1.65\times10^{-4}$; at $t = 1000$, $\eta = 3\times10^{-5}$.

Why warmup helps: at the start, Adam's second-moment estimates rest on very few gradients and the network is far from any stable region, so full-size steps can push it into a bad state. Why decay helps: late in training, smaller steps reduce the noise of minibatch updates and let the loss settle. Linear decay and constant-then-decay variants are also used.

In PyTorch, schedulers (for example `torch.optim.lr_scheduler.LambdaLR`) adjust the optimizer's rate each time `scheduler.step()` is called, conventionally once after each `optimizer.step()`. Calling it per epoch instead of per step, or twice per step, produces a schedule that looks fine in code and wrong in the logs. Always log the actual learning rate. [L3](03-pretraining.md) asks you to investigate a broken schedule.

### Regularization

Regularization is anything that reduces overfitting without reducing training data:

| Method | Mechanism | LLM note |
|---|---|---|
| weight decay | shrinks parameters each step | common in AdamW pretraining |
| dropout | zeroes each activation with probability $q$ during training and scales survivors by $1/(1-q)$; does nothing in eval mode | often small or off in large-scale pretraining with abundant data; more common in fine-tuning |
| early stopping | keep the checkpoint with the best validation loss | requires an honest validation split ([L0](00-foundations.md#generalization-train-validation-test)) |
| more or deduplicated data | reduces memorization | the dominant lever for language models ([L3](03-pretraining.md)) |

Dropout makes `model.train()` and `model.eval()` behave differently. Evaluating in train mode gives noisy, pessimistic validation numbers; training in eval mode silently disables the regularizer.

### Order of a training step

The table lists what each phase touches. Your P01 loop must perform these in a correct order; the exact code is yours to write.

| Phase | Reads | Writes |
|---|---|---|
| clear gradients | — | every parameter's `.grad` |
| forward | parameters, batch | activations, graph |
| loss | logits, targets | a scalar tensor |
| backward | graph, scalar loss | `.grad` of each trainable leaf |
| (optional) clip gradients | `.grad` | `.grad` rescaled to a maximum global norm ([L3](03-pretraining.md)) |
| optimizer step | parameters, `.grad`, optimizer state | parameters, optimizer state |
| scheduler step | step count | learning rate in the optimizer |

Evaluation runs only forward and loss, in eval mode and under `no_grad`.

### Diagnosing training failures

**Absent gradients** (`.grad` is `None` or all zeros after `backward()`): the tensor is not a leaf you think it is; it was never used in the loss; something on its path was detached or computed under `no_grad`; `requires_grad` is false; it is a plain attribute instead of an `nn.Parameter`; or the optimizer was built from a different parameter list (for example, before you replaced a layer). A useful habit is to loop over `model.named_parameters()` after one backward pass and print which gradients are `None` or zero.

**Accumulated gradients** (forgotten clearing): the effective gradient becomes the sum of all past gradients, so steps grow and the loss typically decreases briefly, then oscillates or diverges. The check: run two identical backward passes on the same batch without clearing; the gradient doubles exactly.

**Train/validation divergence**: training loss keeps falling while validation loss rises. Distinguish real overfitting (expected on small data, many epochs) from artifacts: validation computed in train mode, a different preprocessing path for validation, a validation set that is drawn from a different distribution, or duplicated examples that make *training* loss misleadingly low. Plot both curves against the same x-axis (steps or tokens seen).

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Loss does not move at all | learning rate 0, parameters not given to the optimizer, or no gradient reaches them | log the learning rate; print `p.grad is None` for each named parameter; compare a parameter before and after one step |
| Loss decreases then grows steadily | gradients never cleared | two backward passes on one batch: does `.grad` double? |
| Loss jumps to `nan` | learning rate too large, `log(0)`, or `exp` overflow | lower $\eta$ by 10×; replace `log(softmax(z))` with `log_softmax(z)`; check inputs to `exp` |
| Initial loss far from $\ln V$ | large output weights or a bug in targets | compute the loss at initialization before any step |
| Gradient check fails only in `float32` | rounding error with small $\epsilon$ | rerun in `float64` |
| Gradient check fails for one parameter | a `detach`, a wrong transpose, or a reused buffer on that path | check coordinates of that tensor only; inspect its `grad_fn` chain |
| Validation loss noisy and higher than expected | evaluation in train mode (dropout active) | call `model.eval()` and compare |
| Training differs between two "identical" runs | unseeded randomness or nondeterministic kernels | fix seeds; compare the first-step loss and gradients |
| Schedule plot does not match the formula | scheduler stepped per epoch or twice per step | log `optimizer.param_groups[0]["lr"]` every step |

## Check yourself

1. Count the parameters of an MLP with $d_{in} = 20$, $d_{hidden} = 80$, $d_{out} = 5$, with biases. Then count them in code with `sum(p.numel() for p in model.parameters())`.
2. Predict `torch.softmax(torch.tensor([88.0, 89.0, 90.0]), -1)` versus your own naive `exp(z) / exp(z).sum()` in `float32`. Run both and explain.
3. Show that adding 5 to every logit leaves the cross-entropy unchanged. Then confirm it numerically.
4. For logits $[1, 3, 0, -2]$ and target index 2, compute the loss and the gradient vector by hand. Check both with autograd.
5. Predict what happens to `w.grad` if you call `backward()` three times on fresh graphs without clearing. Then predict what happens if you call it twice on the *same* graph.
6. In the `detach` example above, move the `detach()` to the second factor. Predict `a.grad`, then run it.
7. Run your gradient check in `float32` with $\epsilon = 10^{-3}$, $10^{-5}$ and $10^{-7}$. Plot or tabulate the relative error and explain the shape of the curve.
8. Deliberately break one line of your P01 model (for example, transpose a weight in the backward-relevant path or detach a hidden layer). Does your gradient check catch it? Which coordinates fail?
9. Using the SGD worked example, find the largest learning rate that still converges for $f(w) = 4(w - 1)^2$. Test values on either side.
10. On the first Adam step, why is the update almost independent of the gradient's magnitude? What does that imply about the learning rate you choose?
11. Explain in your own words why L2 regularization added to the loss is not the same as AdamW's decoupled decay. Which parameters are affected differently?
12. Train your P01 model on a very small training set until validation loss rises. Then change one thing (weight decay, dropout, or data size) and predict its effect on both curves before running.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | whole page | the graph, `backward`, and frozen parameters in code |
| [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | sections on how autograd encodes history, `no_grad`/`inference_mode`, in-place operations | exact behavior of the tools in this note |
| [`torch.nn.functional.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) | argument list, shapes, `ignore_index`, `reduction` | confirm defaults for your pinned version |
| [`torch.optim.AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | the algorithm box | the exact update PyTorch implements |
| [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) | Ch. 6 (feedforward networks, back-propagation), Ch. 7 (regularization), Ch. 8 (optimization) | rigorous background for MLPs, regularizers and optimizers |
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | Algorithm 1 and the bias-correction discussion | the original update and why the corrections exist |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | the problem statement and the AdamW algorithm | why decay and the adaptive step should be separated |
| [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) | the warmup discussion | optional: an early, clear argument for gradual warmup |
| [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) | the cosine-annealing formula | optional: origin of the cosine schedule |

Next: [L2 — The Transformer language model](02-transformer.md) applies this training machinery to next-token prediction.
