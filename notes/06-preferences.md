# L6 — Preference learning: reward models and DPO

> **Milestone:** [L6](../course/ROADMAP.md#milestones) · **Builds:** P09 (toy DPO objective, pinned source comparison and research connection) in [projects](../course/PROJECTS.md) · **Prerequisites:** [L5](05-sft-peft.md) (templates, loss masks, adapters), [L1](01-neural-learning.md) (log-softmax, cross-entropy), [L0](00-foundations.md) (probability in log-space)

SFT teaches a model to imitate demonstrations. Preference learning teaches it from *comparisons*: for a prompt, response A was judged better than response B. Comparisons are easier to collect than perfect demonstrations and carry information about what *not* to do. This module builds the probability model behind comparisons (Bradley–Terry), the objective most preference methods aim at (reward maximization with a KL leash to a reference model), and the derivation that turns that objective into Direct Preference Optimization (DPO), a loss on log-probabilities with no separate reward model and no sampling. You implement the toy objective yourself in P09; this note gives the math and the tensor bookkeeping, not the loss code.

## What you should be able to do

- Describe a preference dataset and judge pair quality: ties, noise, reversed labels, length confounds, on- versus off-policy responses.
- Write the Bradley–Terry likelihood, train-and-evaluate a pairwise reward model conceptually, and explain why rewards are only defined up to a per-prompt offset.
- State the KL-regularized policy objective and derive its optimal policy.
- Derive DPO from that optimum and Bradley–Terry, showing where the partition function cancels, and list the assumptions used.
- Explain the role of the reference model and of $\beta$.
- Compute sequence log-probabilities from logits (in shapes), and explain sum versus mean and its length effects.
- Name common DPO variants and say what each changes, without treating any as a drop-in improvement.
- Locate the objective in a pinned library source and trace a research claim about preference training to its evidence.

## Core ideas

### Preference data

A preference example is a triple $(x, y_w, y_l)$: prompt $x$, preferred ("chosen", winner) response $y_w$, and dispreferred ("rejected", loser) response $y_l$. Write $y_w \succ y_l$ for "$y_w$ is preferred to $y_l$". Sources include human raters, rubric-driven AI judges, and rule-based checks.

Properties that matter more than dataset size:

| Property | Why it matters |
|---|---|
| Who generated $y_w, y_l$ | responses sampled from your own SFT model are **on-policy** for it; responses from other models are **off-policy** and may contain text your model would almost never produce |
| Margin | two nearly identical responses carry little signal and lots of label noise |
| Ties and inconsistent raters | Bradley–Terry has no tie outcome; ties forced into a winner become noise |
| Confounds | if chosen responses are systematically longer, a model can learn "longer is better" instead of "better is better" |
| Rendering | both responses must be rendered with the same template as the prompt and tokenized consistently ([L5](05-sft-peft.md)) |

### Pairwise reward models and Bradley–Terry

A **reward model** $r_\phi(x, y) \in \mathbb{R}$ scores a response. In practice it is a language model backbone with a one-output head read at the last real token of the rendered (prompt + response). The **Bradley–Terry** (BT) model says preferences arise from a latent scalar score through a logistic link:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

Maximum likelihood over a dataset $\mathcal{D}$ gives the pairwise loss

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big).
$$

**Worked example.** $r_\phi(x, y_w) = 1.2$, $r_\phi(x, y_l) = 0.4$. Margin $0.8$; $\sigma(0.8) \approx 0.690$; loss $= -\ln 0.690 \approx 0.371$. The derivative of the loss with respect to the margin is $-(1 - \sigma(0.8)) \approx -0.310$: the gradient pushes the margin up, and pushes harder when the model is more wrong. At margin $0$ the loss is $\ln 2 \approx 0.693$, the "coin flip" value you should see at initialization if the head starts near zero.

**Offset indeterminacy.** Replace $r(x, y)$ by $r(x, y) + c(x)$ for any function $c$ of the prompt alone. Every difference $r(x, y_w) - r(x, y_l)$ is unchanged, so the likelihood is unchanged. The data therefore cannot pin down the absolute level of reward for a prompt, only differences between responses to the same prompt. Consequences:

- An absolute reward value ("this answer scored 3.1") means nothing on its own; compare responses within a prompt.
- Rewards for different prompts are not on a common scale unless you normalize. Some implementations add a small penalty that keeps rewards centered near zero.
- Downstream, the KL-regularized objective below is unaffected by $c(x)$: it shifts every response to $x$ equally and cancels in the normalized optimal policy. Keep this in mind for the DPO derivation.

> **Version note (observed in the TRL docs checked September 2026; re-check at your pinned version):** TRL's reward trainer used a one-output sequence-classification model and its documentation discussed Bradley–Terry offset indeterminacy.

**Evaluating a reward model.** Held-out **preference accuracy** (fraction of pairs where $r(x, y_w) > r(x, y_l)$), the distribution of margins, accuracy on slices (by length difference, by topic, by data source), and adversarial probes (does appending filler text raise reward?). A reward model that is 70% accurate overall but 95% on "chosen is longer" pairs has partly learned length.

| Tensor | Shape |
|---|---|
| chosen and rejected inputs (often stacked) | `[2B, T]` |
| backbone hidden states | `[2B, T, d]` |
| scalar head output at every position | `[2B, T]` |
| reward at the last non-pad token | `[2B]` → split into `[B]` chosen, `[B]` rejected |

### The KL-regularized objective

Given a reward function $r$, the post-training objective used across RLHF and DPO is

$$
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D}} \Big[ \mathbb{E}_{y \sim \pi(\cdot \mid x)} [r(x, y)] - \beta\, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big) \Big].
$$

Symbols: $\pi$ is the policy being trained (a distribution over complete responses $y$ given $x$); $\pi_{\text{ref}}$ is a fixed **reference policy**, usually the SFT model; $\beta > 0$ sets how strongly $\pi$ is held near $\pi_{\text{ref}}$; $\mathrm{KL}(p \,\|\, q) = \sum_y p(y) \log \frac{p(y)}{q(y)}$.

Why the KL term: the reward model is only trustworthy near the data it was trained on. Without a leash, maximizing $r$ finds responses that exploit its errors (reward hacking, [L7](07-online-rl.md)) and the policy drifts from fluent language.

### Deriving the optimal policy

Fix one prompt $x$ and drop it from the notation. The objective for that prompt is

$$
J(\pi) = \sum_y \pi(y) \Big[ r(y) - \beta \log \frac{\pi(y)}{\pi_{\text{ref}}(y)} \Big].
$$

Factor out $-\beta$ and write $r(y)/\beta = \log e^{r(y)/\beta}$:

$$
J(\pi) = -\beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}.
$$

The denominator is not a distribution yet. Define the **partition function** $Z = \sum_y \pi_{\text{ref}}(y) e^{r(y)/\beta}$ and the normalized distribution

$$
\pi^*(y) = \frac{1}{Z}\, \pi_{\text{ref}}(y)\, e^{r(y)/\beta}.
$$

Then $\pi_{\text{ref}}(y) e^{r(y)/\beta} = Z \pi^*(y)$, and

$$
J(\pi) = -\beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi^*(y)} + \beta \log Z = -\beta\, \mathrm{KL}(\pi \,\|\, \pi^*) + \beta \log Z.
$$

$Z$ does not depend on $\pi$, and KL is zero exactly when $\pi = \pi^*$. So $\pi^*$ is the unique maximizer: the reference policy reweighted by exponentiated reward, with temperature $\beta$.

**Tiny check.** Three possible responses, $\pi_{\text{ref}} = (0.5, 0.3, 0.2)$, rewards $(0, 1, 2)$, $\beta = 1$. Unnormalized weights: $0.5 \cdot 1 = 0.5$, $0.3 e \approx 0.815$, $0.2 e^2 \approx 1.478$. $Z \approx 2.793$, so $\pi^* \approx (0.179, 0.292, 0.529)$. With $\beta = 10$ the weights barely move from $\pi_{\text{ref}}$; with $\beta \to 0$ all mass goes to the highest reward. Adding a constant $c$ to all three rewards multiplies every weight and $Z$ by $e^{c/\beta}$: $\pi^*$ is unchanged, which is the offset indeterminacy at work.

The catch: $Z$ sums over every possible response, an astronomically large set, so you cannot compute $\pi^*$ directly. RLHF approximates it by sampling and policy gradients ([L7](07-online-rl.md)). DPO avoids it algebraically.

### From the optimum to DPO

**Step 1, invert.** Take logs of the optimal-policy formula and solve for the reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
$$

Read this as: any reward function corresponds to an optimal policy, and the reward can be written in terms of that policy, up to a term that depends only on $x$. The quantity $\beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is called the **implicit reward** of a policy.

**Step 2, substitute into Bradley–Terry.** BT only uses reward differences for the same $x$:

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \underbrace{\beta \log Z(x) - \beta \log Z(x)}_{=\,0}.
$$

The intractable partition function cancels. This is the same cancellation as the offset indeterminacy: $\beta \log Z(x)$ is just a per-prompt offset.

**Step 3, fit by maximum likelihood.** Parametrize the policy as $\pi_\theta$ and treat it as if it were the optimal policy for some reward. Maximizing the BT likelihood of the observed preferences gives

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \log \sigma\Big( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \Big).
$$

It is a reward-model loss in which the reward model *is* the policy's log-ratio to the reference.

**Gradient intuition.** Write $\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$. Differentiating gives

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}\Big[ \sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big) \big( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \big) \Big].
$$

Each pair raises the chosen log-probability and lowers the rejected one, weighted by how badly the implicit reward currently misorders the pair. Note what is constrained: the *difference*. Both log-probabilities can fall during training as long as the rejected one falls faster, an effect often observed in practice. Watch both, not only the margin.

**Worked example.** One pair, $\beta = 0.1$. Sequence log-probabilities:

| | policy $\log \pi_\theta$ | reference $\log \pi_{\text{ref}}$ | log-ratio | implicit reward $\hat r$ |
|---|---|---|---|---|
| chosen $y_w$ | −12.0 | −13.0 | +1.0 | +0.10 |
| rejected $y_l$ | −15.0 | −14.5 | −0.5 | −0.05 |

Margin $0.15$; $\sigma(0.15) \approx 0.537$; loss $\approx 0.621$. The gradient weight $\sigma(-0.15) \approx 0.463$. At the start of training, when $\pi_\theta = \pi_{\text{ref}}$, every log-ratio is zero, every margin is zero, and the loss is exactly $\ln 2 \approx 0.693$ for every pair. That is a free correctness check for your P09 implementation.

### Assumptions behind DPO

| Assumption | What breaks if it fails |
|---|---|
| Preferences follow Bradley–Terry with a scalar latent reward | intransitive or context-dependent preferences cannot be represented |
| The KL-regularized objective is the right target | DPO optimizes it faithfully, but that may not be what you want |
| The policy class can represent $\pi^*$ (and the derivation sums over all $y$) | with finite capacity and finite data, the fitted policy only matches preferences on the pairs seen |
| Data covers the responses that matter | DPO only sees the responses in the dataset; it never samples. Probability mass can move to responses that appear in no pair, for better or worse |
| $\pi_{\text{ref}}$ is the same policy the derivation assumed | if pairs were generated by a very different model, the log-ratios on those responses can be extreme and the optimization badly conditioned |

DPO is *offline*: the data is fixed. Online or iterative variants regenerate pairs from the current policy to address the coverage problem.

### The reference model and $\beta$

The **reference model** is frozen and usually equals the policy's starting point (the SFT checkpoint). It must see exactly the same rendered tokens as the policy. Because it is frozen, its log-probabilities on the dataset can be computed once and cached, which saves memory during training. With LoRA, the reference can be the base model with the adapter disabled.

> **Version note (observed in the TRL v1.16.0 docs, checked October 2026; re-check at your pinned version):** TRL's DPO trainer used the initial policy as the reference when none was given, offered precomputing reference log-probabilities to reduce memory, and warned that padding-free DPO temporarily fell back to padding after a refactor. Record $\beta$, the length policy and the loss variant in every run manifest.

$\beta$ appears twice: as the KL strength in the objective and as the scale of the implicit reward inside $\sigma$. Small $\beta$ means a weak leash and a sigmoid that saturates only after large log-ratio changes, so the policy is allowed to move far from the reference. Large $\beta$ keeps it close. Values around 0.1 are a common starting point, but $\beta$ interacts with the data and with sum-versus-mean scoring, so treat it as a swept hyperparameter, not a constant.

### Sequence log-probabilities from logits

DPO needs $\log \pi(y \mid x)$ for whole responses. By the chain rule of probability,

$$
\log \pi(y \mid x) = \sum_{t=1}^{n} \log \pi(y_t \mid x, y_{<t}).
$$

In words and shapes (do not mistake this for a loss implementation):

1. Render prompt + response with the template and tokenize once, so the boundary tokenizes the way it will at inference. Build a **response mask** that is 1 on response tokens (including the end-of-turn token) and 0 on prompt and padding. `input_ids`, `attention_mask`, response mask: `[B, T]`. Chosen and rejected are often stacked: `[2B, T]`.
2. Forward the policy: `logits [2B, T, V]`.
3. Align: logits at position $t$ score the token at $t+1$. Drop the last logit and the first token/mask entry: `[2B, T-1, V]` and `[2B, T-1]`.
4. Log-softmax over the vocabulary axis, then select the log-probability of the actual next token at each position: `[2B, T-1]`.
5. Multiply by the shifted response mask and sum over the time axis: `[2B]`. Split into chosen `[B]` and rejected `[B]`.
6. Repeat with the reference model under no-grad (or read cached values).

Every L5 masking bug reappears here: a mask that includes prompt tokens, an off-by-one shift, or padding counted as response. A reversed pair, or chosen/rejected swapped during stacking, produces a model that learns the opposite preference with a perfectly healthy-looking loss.

**Sum versus mean.** The derivation uses the **sum**. The mean (divide by response length) is a different objective. Length effects:

- Summed log-probabilities scale with length: a 40-token response at $-0.5$ per token scores $-20$; a 10-token response at $-1.0$ per token scores $-10$.
- In DPO the reference ratio removes much of this, since both policy and reference pay for length. But a longer response offers more tokens whose probability the policy can raise, so the log-ratio of a long chosen response can grow faster than that of a short one. Combined with datasets where chosen responses are longer, this is one route to learned verbosity.
- Length-normalized variants (SimPO, below) average per token, which changes the target. Whichever you use, report it, and evaluate with length-controlled comparisons ([L4](04-evaluation.md)).

### Pair quality and label noise

Suppose each label is flipped with probability $\varepsilon$. A model that is confident on every pair is then confidently wrong on about a fraction $\varepsilon$ of them, and the log-sigmoid loss gives exactly those pairs its largest gradient weights (near 1, versus near 0 for pairs it already orders correctly). Remedies: clean data (deduplicate, drop near-ties, resolve rater disagreement), and robust losses that assume some flipped labels (for example, label-smoothed "conservative" DPO). Diagnostics on your own data:

- Implicit-reward accuracy on the *training* pairs stuck below 50% after some steps: check for reversed columns.
- Pairs where the reference already prefers the rejected response by a huge margin: possibly off-policy or mislabeled.
- Accuracy split by length difference: if most gain comes from "chosen is longer" pairs, suspect a confound.

### Variants, by name only

Each line says what changes. None is uniformly better: they change the objective, so compare them under a frozen evaluation protocol and the same data, and read the pinned implementation, since library versions differ from papers.

| Variant | What it changes |
|---|---|
| IPO | replaces the log-sigmoid with a squared loss on the log-ratio margin toward a fixed target, to avoid overfitting when preferences are near-deterministic |
| Conservative / label-smoothed DPO | assumes a known label-flip rate and mixes the loss for both orderings |
| KTO | learns from unpaired "good" / "bad" labels using a prospect-theory-inspired value function instead of pairs |
| SimPO | drops the reference model, uses length-averaged log-probabilities and a target margin |
| ORPO | adds an odds-ratio preference term to the SFT loss, removing the separate reference model and SFT stage |
| Online / iterative DPO | regenerates and relabels pairs from the current policy between rounds |

### Tracing a claim to evidence

When a paper or report says a preference method "improves helpfulness", ask: what was the baseline (SFT only? another method with tuned $\beta$?), was the evaluation held out and length-controlled, which judge and prompt were used, how many seeds, and does the improvement survive in the slices that matter? The same discipline applies to your own P09 write-up and connects to the claim ledgers in [L11](11-research.md).

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Initial DPO loss is not $\ln 2$ | policy and reference differ at step 0 (different checkpoint, template, dropout active, adapter not disabled) | compare policy and reference sequence log-probs on one batch before training |
| Margin improves while chosen log-prob falls steadily | expected tendency of the objective; can signal over-optimization | log chosen and rejected log-probs separately; check held-out generations |
| Model learns the opposite preference | chosen/rejected swapped during stacking or loading | decode one stacked batch; confirm which half is chosen |
| Responses grow long over training | length confound in data and/or summed log-probs | plot response length over checkpoints; slice accuracy by length difference |
| Reward model accuracy high, behavior unchanged | reward model learned a shortcut (length, formatting) | adversarial probes; slice accuracy |
| Log-ratios huge from the start | off-policy pairs or reference mismatch | histogram reference log-probs on chosen and rejected |
| Loss saturates near zero quickly | $\beta$ too large for the data, or trivially separable pairs | margin histogram; inspect easiest pairs |
| Different results "with the same config" across library versions | changed defaults (loss type, length handling, padding path) | diff the pinned source path for the objective |

## Check yourself

1. Show that adding $c(x)$ to every reward leaves the Bradley–Terry likelihood unchanged. Then show it leaves $\pi^*$ unchanged.
2. Redo the derivation of $\pi^*$ without looking. At which line does $Z$ appear, and why is it allowed to be ignored when maximizing over $\pi$?
3. In the DPO derivation, which assumption lets you replace $\pi^*$ by $\pi_\theta$? What does that substitution not guarantee?
4. Predict, then run: before any training, what DPO loss should your P09 objective report on any batch? What does a different value tell you?
5. Using shapes only, explain how a missing shift in step 3 of the log-prob recipe changes the result. Would the loss still decrease?
6. Construct a three-pair toy dataset where every chosen response is longer. Predict how sum-based and mean-based scoring treat it.
7. With a fixed log-ratio margin of 2.0, compute the DPO loss for $\beta = 0.05$ and $\beta = 0.5$. Which setting is still pushing the margin, and why?
8. Reverse one pair in your toy data and trace its gradient contribution. How large is it compared with a correctly labeled pair at the same margin?
9. In your pinned TRL version, find the function that computes the DPO objective. Which variant names does it accept, and how does it reduce over tokens?
10. Pick one published preference-learning claim. Write down the baseline, the evaluation, the length control and the number of runs behind it.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) | §3 preliminaries; §4 the derivation and gradient analysis; Appendix A.1 (KL-constrained optimum), A.2 (DPO objective under Bradley–Terry), A.4 (gradient) | the primary derivation and its stated assumptions |
| Bradley and Terry, "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons", *Biometrika* (1952) | the model definition (optional, historical) | origin of the pairwise likelihood |
| [Deep RL from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741) | reward-learning method | reward models fit to pairwise human comparisons |
| [Fine-Tuning LMs from Human Preferences (Ziegler et al., 2019)](https://arxiv.org/abs/1909.08593) | methods | KL-penalized reward optimization for language models |
| [Learning to Summarize from Human Feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325) | reward model and policy sections; over-optimization discussion | the RLHF pipeline at small scale, with careful evaluation |
| [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) | §3.5 Models: reward model loss and the KL-penalized RL objective | the standard pipeline DPO replaces |
| [TRL reward trainer](https://huggingface.co/docs/trl/reward_trainer) and [DPO trainer](https://huggingface.co/docs/trl/dpo_trainer) | dataset format, loss variants, reference handling | read the pinned release before P09's source comparison |
| [IPO (Azar et al., 2023)](https://arxiv.org/abs/2310.12036) · [KTO (Ethayarajh et al., 2024)](https://arxiv.org/abs/2402.01306) · [ORPO (Hong et al., 2024)](https://arxiv.org/abs/2403.07691) · [SimPO (Meng et al., 2024)](https://arxiv.org/abs/2405.14734) | objective sections only (optional) | what each variant changes and why |
