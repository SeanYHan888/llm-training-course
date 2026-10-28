# Interpretability

> **Elective** · related milestones: [L1](../course/ROADMAP.md#milestones), [L2](../course/ROADMAP.md#milestones), [L4](../course/ROADMAP.md#milestones) · **Builds:** [Interpretability 01](../exercises/interpretability/README.md) (feature sketch and J-lens observation note), companion to [Inference 01](../exercises/inference/README.md) · **Prerequisites:** [L1](01-neural-learning.md) vectors, activations and logits; [L2](02-transformer.md) residual blocks and the output head; [L4](04-evaluation.md) controlled comparisons

Interpretability asks what information a trained model represents internally and how that information produces its outputs. For someone learning to train models, it is useful twice over: it turns "the loss went down" into questions about what was learned, and it is a strict school in evidence — most interpretability mistakes are claims that outrun their experiments. This note introduces the main objects (residual stream, neurons, features, sparse-autoencoder latents, lens readouts) and, just as importantly, which experiments support which kinds of claim.

## What you should be able to do

- Describe the **residual stream** of a decoder and explain why it is the natural object to study.
- Distinguish a **neuron activation**, a **distributed feature** (a direction), and an **SAE latent**, and say what each one is a property of.
- Explain **superposition** with a small geometric example and why it makes single neurons hard to interpret.
- Write the sparse-autoencoder objective, explain the sparsity penalty, reconstruction error and dead latents, and explain why a latent's label is a hypothesis.
- Explain what the **logit lens** and **tuned lens** compute, and describe the J-lens/J-space at the level stated in the course's sources, deferring details to the paper.
- Distinguish **correlational** evidence (probes, activating examples, lens readouts) from **causal** evidence (activation patching, ablation, steering), and match each kind of claim to the evidence it needs.

## Core ideas

### 1. The residual stream

In a pre-norm decoder, each position carries a vector $x_\ell \in \mathbb{R}^d$ between layers. Every block *reads* a normalized copy of it and *adds* its output back:

$$
x_{\ell}' = x_\ell + \mathrm{Attn}_\ell(\mathrm{Norm}(x_\ell)), \qquad
x_{\ell+1} = x_\ell' + \mathrm{MLP}_\ell(\mathrm{Norm}(x_\ell')).
$$

Here $x_0$ is the token embedding (plus position information if the model adds it there), $\ell = 0, \dots, L-1$ indexes layers, and the logits are $z = W_U\,\mathrm{Norm}_f(x_L)$ with unembedding $W_U \in \mathbb{R}^{V \times d}$.

Unrolling the additions shows that the final vector is a **sum of contributions**:

$$x_L = x_0 + \sum_{\ell} \big(a_\ell + m_\ell\big),$$

where $a_\ell$ and $m_\ell$ are what attention and MLP layer $\ell$ wrote. This is why the residual stream is the central object: it is the shared channel through which every component communicates, and every intermediate $x_\ell$ has the same shape as the final one, so tools that work on one layer can be applied to all.

| Tensor | Shape | Notes |
|---|---|---|
| Residual stream at layer $\ell$ | `[B, T, d]` | One vector per position; the main object of study |
| Attention pattern of layer $\ell$ | `[B, h, T, T]` | Which earlier positions each head reads |
| MLP hidden activations | `[B, T, d_ff]` | The "neurons" usually discussed; $d_{ff}$ often $\approx 4d$ or set by a gated design |
| SAE latents for one site | `[B, T, m]` | $m$ dictionary size, usually several times $d$ |
| Logits | `[B, T, V]` | Read off the final residual stream |

One consequence of the sum: if you know a direction $u \in \mathbb{R}^d$ that the unembedding reads as "token $k$" (roughly row $k$ of $W_U$, ignoring the final norm), you can ask how much each component's output points along it. This is called **direct logit attribution**. It measures only direct effects; a component can also matter by changing what later layers do.

In code this shows up in the [Transformer workshop](../exercises/transformer/README.md) as the residual additions in each block. Being able to save the residual tensor after each block (for example with a forward hook in PyTorch) is the prerequisite for every technique below.

### 2. Neurons versus distributed features

A **neuron** here means one coordinate of a layer's hidden activation, typically after the MLP nonlinearity. It is tempting to read a neuron as a concept detector. Two observations argue against assuming that:

1. Nothing in training rewards aligning concepts with coordinates. If the next layer reads $W v$, rotating $v$ by an invertible matrix and compensating in $W$ computes the same function. The *nonlinearity* is applied per coordinate, which makes the neuron basis somewhat privileged, but not necessarily meaningful.
2. In practice many neurons are **polysemantic**: they activate for several unrelated patterns.

The working alternative is the **linear representation hypothesis**: a **feature** — a property of the input the model tracks — corresponds to a **direction** $u$ in activation space, and its strength on an input is roughly the projection $u^\top x$. A direction can involve many neurons at once; that is what "distributed" means.

| Object | What it is a property of | How you find it |
|---|---|---|
| Neuron activation | The original model, one coordinate | Read it directly |
| Distributed feature (direction) | A hypothesis about the original model | Probe, difference of means, SAE decoder column, or a lens |
| SAE latent | A separately trained analysis model applied to the original model's activations | Train an SAE; inspect its activations |

### 3. Superposition intuition

**Superposition** is the hypothesis that a model represents *more* features than it has dimensions, by assigning them nearly orthogonal directions and tolerating some interference. It works best when features are **sparse** — each is active on only a small fraction of inputs — because interference only hurts when several overlapping features are active together.

Geometric example in $d = 2$ with three features, directions at 120° apart:

$$u_1 = (1, 0), \quad u_2 = (-\tfrac12, \tfrac{\sqrt3}{2}), \quad u_3 = (-\tfrac12, -\tfrac{\sqrt3}{2}), \qquad u_i^\top u_j = -\tfrac12 \ (i \ne j).$$

Store activity $c_i$ as $x = \sum_i c_i u_i$ and read feature $i$ back as $\mathrm{ReLU}(u_i^\top x)$.

| Active features | $x$ | Readouts $(r_1, r_2, r_3)$ | Outcome |
|---|---|---|---|
| Only $c_1 = 1$ | $(1, 0)$ | $(1, 0, 0)$ — the $-0.5$ interference is clipped by ReLU | Perfect |
| $c_1 = 1, c_2 = 1$ | $(0.5, 0.866)$ | $(0.5, 0.5, 0)$ | Both detected but halved: interference |

Three features fit in two dimensions as long as they rarely co-occur. In high dimensions the effect is much stronger: the number of directions that are all *nearly* orthogonal to each other grows very quickly with $d$. The upshot for interpretation: if features are in superposition, individual neurons will look polysemantic, and you need a method that finds an overcomplete set of directions. That motivates sparse autoencoders. Toy Models of Superposition (Elhage et al.) studies this in detail.

### 4. Sparse autoencoders (SAEs)

An SAE is a small network trained, after the fact, to rewrite a model's activation $x \in \mathbb{R}^d$ at one site (say the residual stream after layer $\ell$) as a **sparse** combination of $m \gg d$ learned directions.

A common form:

$$
f = \mathrm{ReLU}\big(W_e (x - b_d) + b_e\big) \in \mathbb{R}^m, \qquad
\hat x = W_d f + b_d \in \mathbb{R}^d,
$$

with encoder $W_e \in \mathbb{R}^{m \times d}$, decoder $W_d \in \mathbb{R}^{d \times m}$ whose columns $w_j$ are the latent directions (often constrained or normalized to unit length), and biases $b_e, b_d$. The training objective over many activations is

$$\mathcal{L} = \underbrace{\lVert x - \hat x \rVert_2^2}_{\text{reconstruction error}} + \lambda \underbrace{\lVert f \rVert_1}_{\text{sparsity penalty}}.$$

- **Reconstruction error** keeps the dictionary faithful to what the model actually computes.
- **Sparsity penalty** pushes most entries of $f$ to zero, so each input is explained by a few latents. $\lambda$ sets the trade-off. The L1 penalty also shrinks the active values toward zero (**shrinkage**); variants such as top-$k$ activations (keep the $k$ largest, zero the rest) replace the penalty with a hard sparsity constraint.
- **Dictionary size** $m$ (often written as an expansion factor $m/d$) controls granularity: larger dictionaries tend to split a broad latent into several narrower ones.

Worked example ($d = 4$, $m = 8$, $b_d = b_e = 0$, $\lambda = 0.01$). Suppose for one activation the encoder gives $f = (0, 1.2, 0, 0, 0, 0.5, 0, 0)$, with decoder columns $w_2 = (0.6, 0, 0.8, 0)$ and $w_6 = (0, -0.6, 0, 0.8)$:

| Quantity | Value |
|---|---|
| $\hat x = 1.2\,w_2 + 0.5\,w_6$ | $(0.72, -0.30, 0.96, 0.40)$ |
| Actual $x$ | $(0.75, -0.28, 0.93, 0.42)$ |
| $x - \hat x$ | $(0.03, 0.02, -0.03, 0.02)$ |
| Reconstruction error | $0.0009 + 0.0004 + 0.0009 + 0.0004 = 0.0026$ |
| $\lVert f \rVert_1$ / penalty | $1.7$ / $0.017$ |
| $L_0$ (active latents) | 2 of 8 |

Here the penalty is larger than the reconstruction error, so training would push the latents to shrink or switch off unless that cost too much reconstruction. Every SAE sits somewhere on this trade-off curve.

How SAEs are judged:

| Metric | Question it answers |
|---|---|
| Mean $L_0$ | How many latents are active per input? |
| Fraction of variance unexplained, $\lVert x - \hat x\rVert^2 / \lVert x - \bar x\rVert^2$ | How much of the activation is missed? |
| Downstream loss when $\hat x$ replaces $x$ in the forward pass | Does the missing part matter to the model's predictions? |
| Dead latents | Latents that never activate on a large sample are wasted capacity; they are tracked and sometimes revived by resampling or auxiliary losses |
| Interpretability of latents | Do activating examples show a consistent, specific pattern? |

**Why labels are hypotheses.** A latent gets a name ("legal citations", "Python indentation") from someone reading its **top-activating examples**. That procedure has known weaknesses:

- Top examples show what activates it *most*; weak activations may follow a different pattern.
- A label describes *sufficient* inputs, not necessarily *all* of them (specificity versus coverage).
- The latent belongs to the SAE, not the model. A different SAE (different $m$, $\lambda$, seed, layer) can split or merge it.
- The reconstruction error is part of the model's activation too; behavior can hide there.
- Activation does not show *use*. Whether the model's output depends on that direction needs an intervention (§7).

In code: no SAE training is assigned in Interpretability 01. The feature sketch asks you to draw activation → sparse latents → reconstruction for a published example and to name one labeling or reconstruction limitation.

### 5. Logit lens and tuned lens

Since every $x_\ell$ lives in the same space as $x_L$, you can ask what the model "would predict" if you stopped at layer $\ell$.

**Logit lens:** apply the final norm and unembedding to an intermediate state,

$$z^{(\ell)} = W_U\,\mathrm{Norm}_f(x_\ell), \qquad p^{(\ell)} = \mathrm{softmax}(z^{(\ell)}).$$

Plotting the top token or the probability of the eventual answer across layers often shows the prediction forming in later layers. Its weakness: earlier layers need not use the same basis the unembedding expects, so early readouts can be noisy or misleading.

**Tuned lens:** learn a per-layer affine **translator** $A_\ell x + c_\ell$, trained so that $\mathrm{softmax}(W_U \mathrm{Norm}_f(A_\ell x_\ell + c_\ell))$ matches the model's final distribution (by KL divergence). This corrects for basis drift between layers at the cost of training translators — and the translator itself could add information, so it needs checks.

| Lens | Trained parameters | Main risk |
|---|---|---|
| Logit lens | None | Misreads early layers whose representation differs from the final layer's |
| Tuned lens | One affine map per layer | Translator may do work the model does not |

Both are **readouts**: they show what is linearly decodable as tokens from a state, not that the model uses it.

### 6. J-space and the J-lens (as described in the course sources)

This section deliberately stays at the level stated in [Interpretability 01](../exercises/interpretability/README.md) and the [reading list](../references/inference-and-interpretability.md). Check every detail against the paper before relying on it.

The sources are Anthropic's 2026 overview, [A global workspace in language models](https://www.anthropic.com/research/global-workspace), and the technical paper by Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html), with an [interactive demo on Neuronpedia](https://www.neuronpedia.org/jlens) and an [official implementation](https://github.com/anthropics/jacobian-lens).

What the course files state:

- **J-space** names the representations being studied; the **J-lens** (Jacobian lens) is the method used to read them. Keep the two terms separate.
- A J-lens readout is a token-associated score derived from a layer's activation using a Jacobian-based lens, which can be inspected across layers and token positions.
- The demo's **Multi-Hop Reasoning** example shows readouts across layers and positions together with a published intervention result; a **Versus Logit-Lens** view compares it with the logit lens of §5.
- The lens is approximate and incomplete. A word appearing in a readout is not an exhaustive transcript of the model's reasoning, and it is not evidence of subjective experience.
- J-space is about representations in activations; it is not a KV-cache or serving technique.

Background you will need for the Methods section, stated generically and *not* as a description of the paper's construction: for a function $g: \mathbb{R}^d \to \mathbb{R}^k$, its **Jacobian** at $x$ is the $k \times d$ matrix $J_{ij} = \partial g_i / \partial x_j$. It gives the best linear approximation of how a small change in $x$ moves each output. Autograd computes Jacobian-vector and vector-Jacobian products efficiently. How the J-lens chooses $g$, which layers and outputs it uses and how it aggregates them are in the paper's Methods, which the course defers until after autograd foundations.

Apply the evidence rules of §7 to the demo: the readout itself is correlational; the published intervention is the causal part. Both are author-reported/demo-provided, not results you reproduced.

### 7. Correlation versus causation: probes and interventions

**Probing.** Train a simple classifier (often linear) to predict a property $y$ from activations $x_\ell$. High held-out accuracy shows the information is **decodable** from that layer. It does not show the model **uses** it: a flexible probe can compute the property itself from correlated information. Two safeguards: keep probes simple, and compare against a **control task** (for example, random labels with the same structure) so that a probe's accuracy can be read relative to what it could memorize.

**Interventions** change an internal state during a forward pass and measure the output.

| Intervention | What you do | What it can show |
|---|---|---|
| Activation patching | Run a *clean* input and a minimally different *corrupted* input. Copy the activation at one site from one run into the other and measure how far the output moves toward the source run's | Whether that site carries information that matters for the difference |
| Ablation | Replace a component's output with zero, its mean over a dataset, or its value on a random other input (resample) | Whether the component is necessary for a behavior, under that replacement |
| Steering | Add $\alpha u$ along a feature direction $u$ during the forward pass | Whether moving along $u$ changes behavior as the label predicts |

Patching worked example. On a fact-recall prompt, the logit difference between the right answer and a distractor is $3.2$ on the clean prompt and $-0.8$ on a corrupted prompt (one key entity swapped). Patching one layer's residual stream at the entity position from clean into corrupted gives $2.0$. The **normalized recovery** is

$$\frac{2.0 - (-0.8)}{3.2 - (-0.8)} = \frac{2.8}{4.0} = 0.70,$$

so that single site restores 70% of the clean behavior on this pair. One prompt pair is an anecdote; report the distribution over many pairs and the choice of corruption.

Pitfalls that make interventions misleading:

- **Off-distribution states.** Zero-ablation can push activations far outside anything seen in training; mean or resample ablation is usually gentler.
- **Compensation.** Later components can partly take over when one is removed, so a small ablation effect does not prove a component is unused.
- **Metric choice.** Probabilities saturate; logit differences behave more linearly. Say which metric you use.
- **Dose.** Steering effects depend on $\alpha$; a large enough push breaks the model in ways unrelated to the feature. Report a sweep of $\alpha$ values.

### 8. Which evidence supports which claim

| Claim | Minimum evidence | Not enough on its own |
|---|---|---|
| "Information about $y$ is present at layer $\ell$" | Simple probe beats a control on held-out data | A single striking example |
| "This latent responds to concept C" | Consistent activating examples across the activation range, plus checks for false positives | Top-10 examples |
| "The model *uses* this direction for behavior B" | Ablation or patching along it changes B on many inputs, with a sensible baseline | Probe accuracy; lens readout |
| "Steering along $u$ produces C" | Dose–response across $\alpha$, with other behaviors checked for collateral damage | One generation at one $\alpha$ |
| "This is the mechanism for task T" | A circuit hypothesis that, when kept and the rest ablated or patched, reproduces T, evaluated on held-out prompts | A plausible story assembled from readouts |
| "The model reasons in a particular internal representation" | Readouts plus controlled interventions, alternative explanations considered, scope stated | A word visible in one lens view |

This table is L4 experimental judgment applied inside the model: baselines, controls, multiple inputs and an explicit statement of scope.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Hooked activations change after the forward pass | Saved a reference to a tensor that is later modified in place | Save `.detach().clone()` and compare |
| Logit-lens readouts look random at every layer | Final norm skipped, or wrong unembedding (tied vs untied weights) | Confirm the lens on the last layer reproduces the model's own logits exactly |
| Probe gets 95% everywhere, even at the embedding layer | Probe too flexible or labels leak through token identity | Run a control task; use a linear probe; check accuracy on the embedding alone |
| SAE has great reconstruction but nearly all latents dense | $\lambda$ too small | Inspect mean $L_0$ |
| Many dead latents | $\lambda$ too large, poor initialization, or too few training activations | Track fraction of latents that never fire over a fixed sample |
| Latent label fails on new text | Labeled only from top activations | Look at mid-range activations and at inputs that should but do not activate it |
| Ablation shows "no effect" | Compensation by later layers, or wrong metric | Patch instead of ablate; use a logit-difference metric; check intermediate layers |
| Demo result treated as your finding | Precomputed, author-reported output | Label it demo-provided; state what you would need to reproduce it |

## Check yourself

1. Write the residual stream at the final layer as a sum of contributions for a 3-layer model. Which terms could direct logit attribution measure, and which effects does it miss?
2. Why does an invertible change of basis in the residual stream leave the function unchanged, and why does that not apply to the MLP hidden layer?
3. Extend the superposition example: put four features at 90° in $d = 2$ (with signs). What happens when features 1 and 3 are both active? What about five features?
4. In the SAE worked example, double $\lambda$. Which term changes, and in which direction would training push $f$?
5. Explain the difference between an SAE latent and a neuron to someone who has seen only the MLP code in your Transformer workshop.
6. Predict, then run on your own small model: apply the logit lens to every layer of one prompt. At which layer does the final top token first appear? Repeat with a different prompt; is the layer the same?
7. You train a linear probe that predicts "is inside a quoted string" with high accuracy at layer 2. State precisely what you may and may not conclude, and design one experiment that would strengthen the claim.
8. For the J-lens Multi-Hop Reasoning example, write down: the prompt, the layer/position you viewed, the readout, the baseline answer, and the changed output after the published intervention. Then give one control and one alternative explanation.
9. Why might zero-ablation overstate a component's importance, and how would resample ablation change the result?
10. Design a steering experiment for a direction of your choice: what $\alpha$ values, what inputs, what metric, and what would count as collateral damage?

## Going deeper

| Source | Read | Why |
|---|---|---|
| [Elhage et al., A Mathematical Framework for Transformer Circuits (2021)](https://transformer-circuits.pub/2021/framework/index.html) | Residual-stream and "virtual weights" sections (optional depth) | The residual stream as a communication channel; direct paths |
| [Elhage et al., Toy Models of Superposition (2022)](https://transformer-circuits.pub/2022/toy_model/index.html) | **Definitions and Motivation: Features, Directions, and Superposition**, **Demonstrating Superposition** | Geometry of more features than dimensions |
| [Anthropic, Decomposing language models into understandable components](https://www.anthropic.com/research/decomposing-language-models-into-understandable-components) | Whole overview | Short introduction to dictionary learning |
| [Bricken et al., Towards Monosemanticity (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | **Problem Setup**, **Features as a Decomposition**; one feature example | The SAE setup and how features are inspected |
| [Cunningham et al., Sparse Autoencoders Find Highly Interpretable Features in Language Models (2023)](https://arxiv.org/abs/2309.08600) | Method and evaluation (optional) | Independent SAE work on residual streams |
| [Templeton et al., Scaling Monosemanticity (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | **Influence on Behavior** (feature steering) and **Limitations, Challenges, and Open Problems** (optional) | SAEs at larger scale; using latents for interventions |
| [Gao et al., Scaling and evaluating sparse autoencoders (2024)](https://arxiv.org/abs/2406.04093) | Top-$k$ SAE method, dead-latent discussion, evaluation metrics (optional) | Alternative sparsity mechanism and how SAEs are measured |
| nostalgebraist, *interpreting GPT: the logit lens* (LessWrong, 2020) | Whole post | The original logit-lens observation |
| [Belrose et al., Eliciting Latent Predictions from Transformers with the Tuned Lens (2023)](https://arxiv.org/abs/2303.08112) | Introduction and method | Learned translators; comparison with the logit lens |
| [Anthropic, A global workspace in language models (2026)](https://www.anthropic.com/research/global-workspace) | **How we found the J-space**, **Claude thinks in its J-space** | Overview of J-space |
| [Gurnee et al., Verbalizable Representations Form a Global Workspace in Language Models (2026)](https://transformer-circuits.pub/2026/workspace/index.html) | **The Jacobian Lens and the J-space**, the multi-hop example in **The J-space mediates internal reasoning**, **Limitations and open questions**; Methods later | The authoritative description of the J-lens; check details here |
| [Neuronpedia J-lens demo](https://www.neuronpedia.org/jlens) | **Multi-Hop Reasoning**; **Versus Logit-Lens** if time | Inspect readouts and a published intervention |
| [Official Jacobian-lens implementation](https://github.com/anthropics/jacobian-lens) | README, later with Methods (optional) | For a later pinned reproduction |
| [Hewitt & Liang, Designing and Interpreting Probes with Control Tasks (2019)](https://arxiv.org/abs/1909.03368) | Introduction and control-task definition | Why probe accuracy needs a baseline |
| [Belinkov, Probing Classifiers: Promises, Shortcomings, and Advances (2021)](https://arxiv.org/abs/2102.12452) | Shortcomings sections (optional) | Survey of what probes can and cannot show |
| [Meng et al., Locating and Editing Factual Associations in GPT (2022)](https://arxiv.org/abs/2202.05262) | Causal-tracing section | A worked example of activation patching |
| [Wang et al., Interpretability in the Wild (2022)](https://arxiv.org/abs/2211.00593) | Method and faithfulness evaluation (optional) | A circuit claim with explicit tests |
| [Heimersheim & Nanda, How to use and interpret activation patching (2024)](https://arxiv.org/abs/2404.15255) | Whole note (optional) | Practical guidance on patching metrics and pitfalls |
| [Turner et al., Steering Language Models With Activation Engineering (2023)](https://arxiv.org/abs/2308.10248) | §3 (how ActAdd works) and the limitations discussion in §5 (optional) | Activation-addition steering |
