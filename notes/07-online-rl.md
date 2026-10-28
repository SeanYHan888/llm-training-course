# L7 — Online reinforcement learning for language models

> **Milestone:** [L7](../course/ROADMAP.md#milestones) · **Builds:** P10 (reward/verifier audit and hand-traced policy update) and P11 (small online-RL run with fixed rollout/token/time limits) in [projects](../course/PROJECTS.md) · **Prerequisites:** [L6](06-preferences.md) (KL-regularized objective, reference model, sequence log-probs), [L0](00-foundations.md) (expectations, log-derivatives), [L2](02-transformer.md) (sampling)

DPO learns from a fixed set of comparisons. **Online** methods instead let the current model generate responses, score them, and update on its own samples. That makes them able to discover behavior that no dataset contained, and it also makes them able to discover every loophole in the reward. This module builds the vocabulary of reinforcement learning (RL) for language models, derives the policy gradient, and walks from REINFORCE to PPO and GRPO, then spends as much time on rewards, verifiers and failure diagnostics as on the objectives, because that is where small runs go wrong. You hand-trace an update and write the reward audit yourself in P10; this note gives a worked trace on a different toy problem and no trainer code.

## What you should be able to do

- Map RL vocabulary onto language-model generation: policy, state, action, trajectory, reward, return, value, advantage.
- Derive the policy gradient with the log-derivative trick, and show why a baseline leaves it unbiased while reducing variance.
- Explain GAE at the level of what $\lambda$ trades off.
- Explain importance ratios, why PPO clips them, and what the value model is for.
- Distinguish a KL penalty added to the reward from a KL term added to the loss.
- Compute GRPO group-relative advantages and RLOO baselines by hand, and hand-trace one tiny update.
- Design rewards, especially verifiable rewards, and test a verifier before trusting it.
- Recognize reward hacking and diagnose zero-variance groups, truncation, entropy collapse and train/inference log-probability mismatch.

## Core ideas

### RL vocabulary for language models

| RL term | Language-model meaning |
|---|---|
| policy $\pi_\theta(a \mid s)$ | the model's next-token distribution (softmax over $V$ logits) |
| state $s_t$ | the prompt $x$ plus the tokens generated so far, $y_{<t}$ |
| action $a_t$ | the next token $y_t$ |
| trajectory / episode $\tau$ | one complete sampled response $y = (y_1, \dots, y_n)$, ending at EOS or a length limit |
| reward $r_t$ | usually zero for every token except the last, where $r_n = R(x, y)$ from a reward model or verifier |
| return $G_t$ | reward from step $t$ onward, $G_t = \sum_{k \ge t} \gamma^{k-t} r_k$; with $\gamma = 1$ and a terminal reward, $G_t = R(x, y)$ for every $t$ |
| value $V(s_t)$ | expected return from state $s_t$ under the current policy |
| advantage $A_t$ | how much better the action taken was than expected: $A_t = Q(s_t, a_t) - V(s_t)$, estimated in practice |

Two useful views. In the **bandit view**, the whole response is one action and gets one reward. In the **token view**, each token is an action, which lets per-token quantities (KL, value estimates) enter. Both views give the same policy gradient when the reward is terminal.

The objective is expected reward under the policy's own samples:

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\, \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ R(x, y) \big].
$$

The difficulty: $y$ is sampled, sampling is not differentiable, and $R$ is often a black box (a program, a regex, another model). We need a gradient that does not differentiate through either.

### The policy gradient (REINFORCE)

Use the **log-derivative trick**, $\nabla_\theta \pi_\theta(y) = \pi_\theta(y) \nabla_\theta \log \pi_\theta(y)$. For one prompt (dropping $x$):

$$
\nabla_\theta J = \sum_y R(y)\, \nabla_\theta \pi_\theta(y) = \sum_y \pi_\theta(y)\, R(y)\, \nabla_\theta \log \pi_\theta(y) = \mathbb{E}_{y \sim \pi_\theta}\big[ R(y)\, \nabla_\theta \log \pi_\theta(y) \big].
$$

The gradient is an expectation under the policy, so you can estimate it by sampling. $R$ is only multiplied in, never differentiated. Because $\log \pi_\theta(y) = \sum_t \log \pi_\theta(y_t \mid y_{<t})$,

$$
\nabla_\theta J \approx \frac{1}{G} \sum_{i=1}^{G} R(y^{(i)}) \sum_{t} \nabla_\theta \log \pi_\theta\big(y^{(i)}_t \mid y^{(i)}_{<t}\big).
$$

Read it as weighted maximum likelihood on your own samples: raise the log-probability of every token in a sample in proportion to that sample's reward. **Causality** refinement: an action at step $t$ cannot affect rewards received before $t$, so each token's weight can be the reward-to-go $G_t$ rather than the full-episode reward. With a terminal reward these coincide.

### Baselines, advantages and variance

Subtract any $b$ that does not depend on the sampled action:

$$
\mathbb{E}_{y \sim \pi_\theta}\big[ b\, \nabla_\theta \log \pi_\theta(y) \big] = b \sum_y \nabla_\theta \pi_\theta(y) = b\, \nabla_\theta \sum_y \pi_\theta(y) = b\, \nabla_\theta 1 = 0.
$$

So $\mathbb{E}[(R - b) \nabla \log \pi]$ is still the true gradient, but the variance of the estimate can be much lower. $R - b$ (or a refined estimate of it) is the **advantage**. The best simple baseline is close to the expected reward, i.e. the value $V(s)$.

**Why it matters.** Suppose every reward is shifted by $+10$, so rewards are $\{11, 10\}$ instead of $\{1, 0\}$. The true gradient is unchanged (the shift contributes zero in expectation), but each single-sample estimate is now dominated by the $10 \nabla \log \pi$ term pointing in a random direction. Without a baseline you need far more samples to average that noise away. A baseline at the mean reward removes the shift exactly.

**GAE, briefly.** With a learned value function $V$, define the one-step temporal-difference error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. **Generalized Advantage Estimation** uses

$$
\hat A_t = \sum_{l \ge 0} (\gamma \lambda)^l\, \delta_{t+l}.
$$

$\lambda = 0$ gives $\hat A_t = \delta_t$: low variance but biased by errors in $V$. $\lambda = 1$ gives the return minus $V(s_t)$: unbiased but high variance. Values in between trade the two. For a terminal reward and a weak value model, the value model's quality decides whether GAE helps.

### Importance ratios and off-policy correction

Generating samples is expensive, so you want several gradient steps per batch of rollouts. After the first step, the samples came from the old policy $\pi_{\text{old}}$, not the current $\pi_\theta$. Correct with the **importance ratio** per token:

$$
\rho_t(\theta) = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{old}}(y_t \mid s_t)}, \qquad L^{\text{PG}}(\theta) = \mathbb{E}_t\big[ \rho_t(\theta)\, \hat A_t \big].
$$

At $\theta = \theta_{\text{old}}$, $\rho_t = 1$ and $\nabla \rho_t = \nabla \log \pi_\theta$, so the gradient equals the policy gradient. Far from $\theta_{\text{old}}$, the ratio can become large and the estimate unreliable: the samples no longer represent the policy you are updating.

In implementations $\rho_t$ is computed as $\exp(\log \pi_\theta - \log \pi_{\text{old}})$, with $\log \pi_{\text{old}}$ recorded at rollout time and treated as a constant.

### PPO: clipping and the value model

**PPO**'s clipped surrogate limits how much one batch can move each token's probability:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[ \min\big( \rho_t \hat A_t,\; \mathrm{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon)\, \hat A_t \big) \Big].
$$

| Case | Effect of the min/clip |
|---|---|
| $\hat A_t > 0$, $\rho_t \le 1 + \epsilon$ | normal gradient: raise the token's probability |
| $\hat A_t > 0$, $\rho_t > 1 + \epsilon$ | clipped term is a constant: no further push up |
| $\hat A_t < 0$, $\rho_t \ge 1 - \epsilon$ | normal gradient: lower the token's probability |
| $\hat A_t < 0$, $\rho_t < 1 - \epsilon$ | clipped: no further push down |
| ratio moved the *wrong* way (e.g. $\hat A_t > 0$, $\rho_t < 1 - \epsilon$) | the unclipped term is smaller, so it is kept: the gradient still corrects the mistake |

The full PPO loss adds a **value loss** (regress a value head $V_\psi(s_t)$ onto return targets, used for GAE) and often an entropy bonus. For a language model, PPO-style RLHF holds up to four models: the policy, the frozen reference, the reward model, and the value model (often initialized from the reward model). The value model is the costliest piece to remove, which motivates GRPO.

### KL: in the reward or in the loss

The [L6](06-preferences.md) objective penalizes $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$. Online methods estimate it from samples, per token, using $\log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)$. Two placements:

| Placement | How | Consequence |
|---|---|---|
| **KL in the reward** (classic RLHF PPO) | per-token reward $r_t = -\beta(\log \pi_\theta - \log \pi_{\text{ref}})$, plus $R(x, y)$ at the last token | the KL flows through returns, the value model and advantages; it is part of what is clipped |
| **KL in the loss** (GRPO as published) | add $\beta \cdot \widehat{\mathrm{KL}}$ directly to the objective, separate from the advantage | advantages reflect task reward only; the KL is a separate regularizer with its own gradient |

Sample-based KL estimators differ. With $u = \pi_{\text{ref}}(y_t \mid s_t) / \pi_\theta(y_t \mid s_t)$: the naive $k_1 = -\log u$ is unbiased but can be negative on single samples; $k_3 = (u - 1) - \log u$ is unbiased and always non-negative. Example: $\pi_\theta = 0.5$, $\pi_{\text{ref}} = 0.4$ gives $u = 0.8$, $k_1 \approx 0.223$, $k_3 \approx 0.023$. Individual values differ while expectations agree.

### GRPO: group-relative advantages, no value model

**GRPO** samples a group of $G$ responses for each prompt, scores them, and uses the group itself as the baseline:

$$
\hat A_i = \frac{R_i - \mathrm{mean}(R_1, \dots, R_G)}{\mathrm{std}(R_1, \dots, R_G) + \varepsilon_{\text{num}}},
$$

assigned to every token of response $i$. The objective is the PPO-style clipped surrogate with these advantages, averaged over tokens and responses, plus (optionally) a KL term to the reference. No value model is needed, and with a rule-based verifier no reward model either.

Details that vary by implementation and change behavior: population versus sample standard deviation; whether to divide by the std at all; whether the token average is per response (which weights short responses' tokens more) or over all tokens in the batch; whether truncated responses are included. One analysis of these biases is "Understanding R1-Zero-Like Training: A Critical Perspective" (Liu et al., 2025).

> **Version note (observed in the TRL v1.16.0 docs, checked October 2026; re-check at your pinned version):** TRL's GRPO trainer had $\beta = 0$ by default, so no KL term unless explicitly enabled; its documented diagnostics included zero-variance reward groups, truncated completions, entropy and train/inference log-probability mismatch. Read the objective in your pinned version's source rather than assuming it equals the original paper.

### RLOO, briefly

**REINFORCE leave-one-out** also samples $G$ responses per prompt, but the baseline for response $i$ is the mean of the *other* $G - 1$ rewards:

$$
\hat A_i = R_i - \frac{1}{G - 1} \sum_{j \ne i} R_j.
$$

Because the baseline does not include $R_i$, it is independent of response $i$'s sample, keeping the estimator unbiased. It equals $\frac{G}{G-1}(R_i - \bar R)$, so it is the group-mean baseline rescaled, without std normalization.

### Hand-traced update

A one-step "bandit" small enough to do on paper. Vocabulary $\{A, B, C\}$, policy logits $z = (0, 0, 0)$, so $\pi = (\tfrac13, \tfrac13, \tfrac13)$. The reward is 1 for C and 0 otherwise. One prompt, group size $G = 4$. Sampled actions: C, A, B, A.

**Step 1, rewards and advantages.** $R = (1, 0, 0, 0)$, mean $0.25$, population std $\approx 0.433$.

| sample | action | $R_i$ | GRPO $\hat A_i$ (pop. std) | GRPO with sample std | RLOO $\hat A_i$ |
|---|---|---|---|---|---|
| 1 | C | 1 | +1.732 | +1.5 | +1.0 |
| 2 | A | 0 | −0.577 | −0.5 | −0.333 |
| 3 | B | 0 | −0.577 | −0.5 | −0.333 |
| 4 | A | 0 | −0.577 | −0.5 | −0.333 |

**Step 2, per-sample gradients.** For a softmax, $\nabla_z \log \pi(a) = \mathrm{onehot}(a) - \pi$. So for C it is $(-\tfrac13, -\tfrac13, \tfrac23)$, for A $(\tfrac23, -\tfrac13, -\tfrac13)$, for B $(-\tfrac13, \tfrac23, -\tfrac13)$.

**Step 3, combine.** On the first pass $\rho = 1$, so the clipped objective's gradient is $\frac{1}{G} \sum_i \hat A_i \nabla_z \log \pi(a_i)$, using the GRPO (pop. std) column:

$$
g = \tfrac14\Big[1.732(-\tfrac13, -\tfrac13, \tfrac23) - 0.577\big(2(\tfrac23, -\tfrac13, -\tfrac13) + (-\tfrac13, \tfrac23, -\tfrac13)\big)\Big] \approx (-0.289, -0.144, +0.433).
$$

The components sum to zero, as they must for a softmax. A is pushed down twice as hard as B only because it was sampled twice: the estimate is noisy even though A and B are equally bad.

**Step 4, update.** Gradient *ascent* with learning rate 1: $z' \approx (-0.289, -0.144, 0.433)$, so $\pi' \approx (0.237, 0.274, 0.488)$. Expected reward rose from $0.333$ to $0.488$.

**Step 5, a second pass on the same samples** (PPO-style reuse, $\epsilon = 0.2$). Ratios $\rho = \pi'/\pi$: C $\approx 1.465$, A $\approx 0.712$, B $\approx 0.823$.

- C: $\hat A > 0$ and $\rho > 1.2$, clipped, contributes no gradient.
- A (both samples): $\hat A < 0$ and $\rho < 0.8$, clipped, no gradient.
- B: $0.8 \le \rho \le 1.2$, still pushes B down.

Clipping has stopped this batch from moving C and A any further; only B's term remains. A real token-level update is the same arithmetic repeated for every token of every response, with each token in response $i$ sharing $\hat A_i$. Your P10 trace should use your own setup, not this one.

### Reward design

The reward is the specification. The policy optimizes what you wrote, not what you meant.

| Reward type | Example | Typical weakness |
|---|---|---|
| Learned reward model ([L6](06-preferences.md)) | preference-trained scorer | exploitable shortcuts (length, flattery, formatting); over-optimization |
| Verifiable / rule-based | exact-match answer, unit tests pass | brittle parsing; false negatives on correct but differently formatted answers; exploitable checks |
| Format reward | answer appears inside a required tag | easy to satisfy without solving the task |
| Shaping / penalties | length penalty, repetition penalty | changes the optimum; weights need justification |

Combine rewards only with explicit weights, log each component separately, and remember that GRPO-style normalization rescales the sum per group, so a small format reward can dominate when task rewards are all equal.

### Verifiable rewards and verifier testing

**RL with verifiable rewards (RLVR)** uses a program that checks correctness, for example extracting a final numeric answer and comparing it with the reference, or running hidden unit tests. It avoids learned-reward exploits but is only as good as the checker. Test the verifier like any other code before using it as a reward:

| Test case | Expected verdict |
|---|---|
| correct answer in the canonical format | pass |
| correct answer in an equivalent format (`0.5`, `1/2`, `.50`, trailing period, LaTeX) | decide the policy explicitly, then test it |
| wrong answer | fail |
| several candidate answers in one response | fail, or a declared rule (e.g. last one counts) |
| empty, truncated, or missing the answer marker | fail without crashing |
| answer copied from the prompt or a hint | fail if the hint is not the answer |
| extremely long output, or code that never terminates | fail within a time limit; sandboxed |
| response that tampers with tests, prints "PASS", or targets the grader's parsing | fail |

Measure false-positive and false-negative rates on a small hand-labeled set of real model outputs. A verifier with a 5% false-positive rate is a 5% chance per sample of rewarding a wrong answer, and the policy will search for that 5%.

### Reward hacking

**Reward hacking** is the policy increasing measured reward without the intended behavior: padding answers to satisfy a reward model's length preference, emitting the answer format without reasoning, exploiting a loose regex, or editing tests in a coding task. With learned reward models, true quality can rise and then fall as the proxy reward keeps rising (over-optimization). Defenses: KL to the reference, held-out evaluation with a *different* checker or human reading, monitoring reward components and lengths, and reading samples every few steps. Rising training reward alone is not evidence of improvement ([L4](04-evaluation.md)).

### Failure diagnostics

Log these from the first step of P11:

| Signal | What it reveals |
|---|---|
| fraction of **zero-variance groups** | if all $G$ responses get the same reward, every $\hat A_i = 0$ and the prompt contributes no policy gradient. Too-easy or too-hard prompts, or a broken verifier that always says pass/fail |
| fraction of **truncated** completions | responses hitting the length limit have no EOS, usually fail parsing, and get reward 0; this biases toward short answers or punishes long reasoning. Decide whether truncated responses are scored, masked or excluded |
| mean token **entropy** of sampled tokens | **entropy collapse**: the policy becomes near-deterministic, groups become identical, variance vanishes and learning stalls |
| ratio $\rho$ statistics and clip fraction | how far updates move the policy; a large clip fraction means steps are too big |
| **train/inference log-prob mismatch** | rollouts often come from a fast inference engine while gradients use the training forward; different kernels, precision or sampling settings make the log-probs disagree even with identical weights, so the "on-policy" ratio is not 1 |
| KL to reference, response length, each reward component | drift, verbosity, one component dominating |
| held-out task accuracy with a separate checker | whether the reward gain is real |

A quick mismatch test: at the very first optimization step, before any update, the ratio between the training forward's log-probs and the rollout's recorded log-probs should be 1 within numerical tolerance for every token. Also check that log-probs are computed with the same temperature and sampling transform used to generate; scoring at temperature 1 samples drawn at temperature 0.7 is an off-policy mismatch that looks like a bug-free run.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Reward flat, loss near zero | most groups have zero variance | log the zero-variance fraction; inspect prompt difficulty and verifier output |
| Reward rises fast, held-out accuracy flat | reward hacking or verifier false positives | read top-reward samples; run the verifier test table |
| Responses shrink to the minimum or explode to the limit | length bias from normalization, truncation handling or a reward component | plot length distribution and truncation rate over steps |
| Ratio far from 1 on the first step | train/inference mismatch (dtype, kernels, temperature, template) | compare per-token log-probs from both paths on the same tokens |
| Entropy falls to near zero within a few steps | learning rate too high, too many reuse epochs, no KL | entropy and clip fraction curves; reduce step size |
| KL to reference grows without bound | $\beta = 0$ or too small, or KL computed on wrong tokens | confirm the KL term is active and masked to response tokens |
| Loss NaN | zero std without a numerical epsilon, empty responses, or $\exp$ overflow of large log-ratio differences | check group stds and log-ratio ranges |
| Same config, different behavior across library versions | changed normalization, defaults or KL estimator | diff the pinned source path for the objective |

## Check yourself

1. Derive $\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[R(y)]$ without notes. Which step requires that $R$ does not depend on $\theta$?
2. Prove that a constant baseline leaves the policy gradient unbiased. Does a baseline that depends on the sampled response also do so? Why does RLOO exclude $R_i$?
3. Redo the hand trace with rewards $(1, 1, 0, 0)$ for actions C, C, A, B. Predict the direction of each logit change before computing it.
4. Predict, then compute: in the hand trace, what happens to the update if every reward is increased by 5? Compare REINFORCE without a baseline with GRPO.
5. For each row of the PPO clipping table, construct a numerical $(\rho, \hat A)$ example and compute the gradient of the min term with respect to $\rho$.
6. Explain the difference between putting the KL in the reward and in the loss, using where each enters the computation of $\hat A_i$.
7. Write a verifier test table for a task of your choice with at least eight cases, then measure false-positive and false-negative rates on 20 real outputs.
8. In your P11 run, which prompts produced zero-variance groups? What would you change: prompts, $G$, temperature, or the reward?
9. Predict, then run: lower the maximum completion length until a third of responses truncate. What happens to reward, length and the zero-variance fraction?
10. Show, on one batch, that the training forward's per-token log-probs match the rollout engine's. If they do not, list every difference between the two paths.
11. Describe one way your current reward could be hacked, and the held-out check that would catch it.

## Going deeper

| Source | Read | Why |
|---|---|---|
| Sutton and Barto, [*Reinforcement Learning: An Introduction*, 2nd ed.](http://incompleteideas.net/book/the-book-2nd.html) | Ch. 13 (policy gradient methods; §13.3 REINFORCE, §13.4 REINFORCE with baseline) | the textbook derivation |
| Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", *Machine Learning* (1992) | the REINFORCE estimator (optional, historical) | the original estimator |
| [OpenAI Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) | the derivation and the baseline sections | a second, compact derivation |
| [GAE (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438) | §3, advantage function estimation (defines the estimator and $\lambda$) | bias–variance trade-off with a value function |
| [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) | §3 clipped surrogate; §4 adaptive KL penalty; §5 algorithm | the clipped objective and its alternatives |
| [Approximating KL Divergence (Schulman)](http://joschu.net/blog/kl-approx.html) | the whole post | the $k_1$, $k_2$, $k_3$ estimators |
| [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) | §3.5 Models: RL fine-tuning with a per-token KL penalty | KL in the reward, PPO for language models |
| [DeepSeekMath (Shao et al., 2024)](https://arxiv.org/abs/2402.03300) | §4.1 Group Relative Policy Optimization (within §4, Reinforcement Learning) | group-relative advantages, KL in the loss |
| [Back to Basics: REINFORCE-style optimization for RLHF (Ahmadian et al., 2024)](https://arxiv.org/abs/2402.14740) | method sections on REINFORCE and RLOO | leave-one-out baselines for language models |
| [Tülu 3 (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124) | §6 Reinforcement Learning with Verifiable Rewards | verifiable rewards in a full post-training pipeline |
| [DeepSeek-R1 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948) | rule-based reward and RL-only training sections (optional) | a large-scale verifiable-reward case; author-reported results |
| [Scaling Laws for Reward Model Overoptimization (Gao et al., 2022)](https://arxiv.org/abs/2210.10760) | main results (optional) | how proxy and true reward diverge |
| [Concrete Problems in AI Safety (Amodei et al., 2016)](https://arxiv.org/abs/1606.06565) | the reward-hacking section (optional) | the general failure mode |
| [TRL GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer) and [RLOO trainer](https://huggingface.co/docs/trl/rloo_trainer) | objective, loss normalization, logged metrics | read at the pinned release before P11 |
| [Historical TRL PPO docs (v0.11.4)](https://huggingface.co/docs/trl/v0.11.4/ppo_trainer) | concept only | the TRL v1.16.0 docs (checked October 2026) have no PPO trainer page; this does not prove removal from every version, and historical snippets should not be mixed with current APIs |
| [CS336](https://cs336.stanford.edu/) | alignment / reasoning-RL lectures (select at activation) | course backbone for this module |
