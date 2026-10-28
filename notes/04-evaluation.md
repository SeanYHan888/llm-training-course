# L4 — Evaluation and experimental judgment

> **Milestone:** [L4](../course/ROADMAP.md#milestones) · **Builds:** evaluation portion of P06 and the frozen protocol of P15a in [projects](../course/PROJECTS.md); every later comparison in the course · **Prerequisites:** [L3](03-pretraining.md); habits from [L1](01-neural-learning.md)

A training run produces numbers; evaluation decides what those numbers mean. Most wrong conclusions in ML do not come from bad models but from bad comparisons: the wrong baseline, metrics that are not comparable, differences smaller than the noise, judges that prefer longer answers, or test data that leaked into training. This note gives you the tools to build a small evaluation protocol, run a controlled ablation, and recognize a misleading comparison when you see one.

## What you should be able to do

- Choose baselines that make a result interpretable.
- Compute token-normalized loss, perplexity and bits per byte, and explain why perplexity across different tokenizers is not comparable.
- Choose downstream metrics and explain log-likelihood versus generation-based scoring.
- Attach uncertainty to a metric (standard error, bootstrap, multiple seeds) and use paired comparisons.
- Identify length, position and self-preference bias in LLM-as-judge evaluation and design checks for them.
- Detect contamination and leakage with overlap checks and reasoning about the pipeline.
- Explain Kaplan and Chinchilla scaling results, derive $C \approx 6ND$, and list the assumptions an allocation rule rests on.
- Design a controlled ablation and read how lm-evaluation-harness structures a task.
- Demonstration: produce a small evaluation protocol and one controlled ablation; catch a misleading comparison or a contaminated split.

## Core ideas

### Baselines

A number means nothing alone. A **baseline** is a reference result that answers "compared with what?". Useful baselines, from weakest to strongest:

| Baseline | What it tells you | Example value |
|---|---|---|
| Uniform over the vocabulary | Loss of a model that knows nothing: $\ln V$ | $V = 4096$: 8.32 nats/token |
| Unigram frequencies | Loss of a model that knows token frequencies but no context | Entropy of the training token distribution |
| Majority class / random choice | Floor for a classification or multiple-choice task | 25% on four-way multiple choice |
| Previous checkpoint or prior run | Did this change help relative to where you were? | Same protocol, earlier config |
| A tuned standard method | Is the new idea better than doing the ordinary thing well? | Same budget, learning rate tuned equally |

The last row is the one most often skipped. Melis et al. showed that carefully tuned LSTM baselines matched or beat several more recent architectures on language-modeling benchmarks; an untuned baseline makes almost any treatment look good. Give the baseline the same tuning budget as the treatment.

### Token-normalized loss

The training loss is a mean of per-token negative log-likelihoods (NLL). For an evaluation set with counted target tokens $i = 1..N$ (padding and masked positions excluded),

$$\bar\ell = \frac{1}{N}\sum_{i=1}^{N} -\ln p_\theta(x_i \mid x_{<i}).$$

The units are **nats per token** (use $\log_2$ for bits). Two details decide whether numbers are comparable:

**Average over tokens, not over batches.** If batch 1 has 100 counted tokens with mean NLL 2.0 and batch 2 has 300 with mean NLL 3.0, the mean of batch means is 2.5 but the token-weighted mean is $(200 + 900)/400 = 2.75$. Only the second equals the corpus-level quantity. Variable-length batches and the last partial batch make this a real effect.

**Count the same tokens.** Whether EOS is scored, whether the first token of each document (which has no context) is scored, and how long the available context is all change $\bar\ell$. A model evaluated with 512 tokens of context per prediction will usually score better than the same model with 96, so evaluate long documents with a fixed window and stride and record both.

### Perplexity and bits per byte

**Perplexity** is the exponential of mean token NLL:

$$\text{PPL} = \exp(\bar\ell).$$

It reads as an "effective number of equally likely choices per token": a uniform model over $V$ tokens has $\text{PPL} = V$.

The trap: **perplexity is per token, and tokenizers disagree about what a token is.** A tokenizer with longer tokens makes fewer, harder predictions per text; one with shorter tokens makes more, easier predictions. To compare models with different tokenizers, normalize by something both share, such as the raw bytes of the text. **Bits per byte (BPB)**:

$$\text{BPB} = \frac{\sum_i -\ln p_\theta(x_i \mid x_{<i})}{\ln 2 \cdot N_{\text{bytes}}}.$$

The numerator is the total NLL in nats over the whole text, so BPB measures how many bits the model needs to encode each byte. It is comparable across tokenizers on the same text (as long as the byte count is defined the same way, for example UTF-8 bytes of the untokenized text). The Pile paper reports BPB for this reason.

Worked example. The same 2,000-byte text:

| | Tokenizer A | Tokenizer B |
|---|---:|---:|
| Tokens | 500 | 800 |
| Total NLL (nats) | 1600 | 1650 |
| Mean NLL per token | 3.20 | 2.06 |
| Perplexity | 24.5 | 7.87 |
| Bits per byte | 1.154 | 1.190 |

Per-token perplexity makes B look three times better. Per byte, A actually compresses the text slightly better. Never report a perplexity comparison across tokenizers without a byte- or character-normalized metric beside it.

Related metrics: *word perplexity* normalizes by a word count, which depends on how words are split; *byte perplexity* is $2^{\text{BPB}}$.

### Downstream metrics

Language-modeling loss tells you how well a model predicts text; **downstream** metrics test a specific capability. Common ones:

| Metric | Definition | Watch out for |
|---|---|---|
| Accuracy | Fraction of items correct | Chance level; class imbalance |
| Exact match | Output string equals reference after normalization | Normalization rules (case, punctuation, whitespace) change scores |
| Token F1 | Overlap of answer tokens with reference tokens | Rewards partial answers |
| pass@$k$ | Probability that at least one of $k$ samples passes tests | Use the unbiased estimator below; sensitive to temperature |
| Calibration | Whether predicted confidence matches observed accuracy | Needs probabilities, not just answers |

For pass@$k$, Chen et al. generate $n \ge k$ samples per problem, count $c$ correct, and estimate

$$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k},$$

averaged over problems. Example: $n = 10$, $c = 3$, $k = 1$ gives $1 - 7/10 = 0.3$.

A lower loss does not guarantee better downstream scores, and small models are often at chance on hard tasks; pick tasks where your model's size can show a difference.

### Log-likelihood versus generation-based evaluation

There are two ways to ask a model a question.

**Log-likelihood scoring.** For a multiple-choice item with context $c$ and choices $a_1..a_m$, compute $\log p(a_j \mid c)$ for each choice (the sum of the choice's token log-probabilities) and pick the highest. No sampling, no parsing, deterministic given the model.

**Generation-based scoring.** Let the model generate text under fixed decoding settings, extract an answer (regex, filter, or a judge), and compare with the reference.

| | Log-likelihood | Generation |
|---|---|---|
| Needs a working decoder loop | No | Yes |
| Deterministic | Yes | Only with greedy decoding and fixed settings |
| Sensitive to | Choice length, tokenization of choices, leading spaces | Prompt format, stop sequences, temperature, answer extraction |
| Measures | Ranking of given options | Whether the model can produce the answer unaided |
| Good for | Small/base models, early training | Instruction-tuned models, open-ended tasks |

**Length normalization** matters for log-likelihood scoring because longer choices accumulate more negative log-probability. Example: the context ends with "The sky on a clear day is", choice A is " blue" (total log-prob $-3.0$, 5 characters) and choice B is " light blue" (total $-4.2$, 11 characters). Raw sums pick A. Per character, A is $-0.60$ and B is $-0.38$, so normalized scoring picks B. The two are different metrics; report which one you use. In lm-evaluation-harness these appear as `acc` and `acc_norm`; current code divides by each choice's character length, but check the pinned version.

### Uncertainty

Every metric computed on a finite test set is an estimate. Report how uncertain it is.

**Standard error of an accuracy.** With $n$ independent items and observed accuracy $\hat p$,

$$\text{SE} = \sqrt{\frac{\hat p(1-\hat p)}{n}}, \qquad \text{approx. 95\% interval: } \hat p \pm 1.96\,\text{SE}.$$

Example: $\hat p = 0.62$ on $n = 400$ items gives $\text{SE} \approx 0.024$ and an interval of about $[0.57, 0.67]$. A rival model at 0.64 on the same size set is not distinguishable by this number alone.

For loss, the same idea applies to per-document (or per-token) losses: SE is the sample standard deviation divided by $\sqrt{n}$. Use per-*document* units when tokens within a document are strongly correlated.

**Clustered items.** Many benchmarks have several questions about the same passage. Those are not independent, and the simple formula understates the SE. Miller ("Adding Error Bars to Evals") discusses clustered standard errors for this case.

**Bootstrap.** When there is no simple formula (for F1, BPB, a median, a ratio): resample the $n$ items *with replacement* to form a new test set of size $n$, recompute the metric, repeat many times (say 1,000–10,000), and take the 2.5th and 97.5th percentiles as an interval. Resample the *unit that is independent* (documents or passages, not tokens).

**Seeds.** A test-set interval measures sampling noise in the evaluation items. It says nothing about **training noise**: a different seed changes initialization and data order, and can move a small model's score by more than the effect you are testing. When affordable, train several seeds per condition and report the spread. When not, label the result "single run".

### Paired comparisons

When two systems are evaluated on the **same items**, compare them item by item. Much of the variance is item difficulty, which affects both systems equally and cancels in the difference.

Worked example: models A and B on the same 400 items.

| | B correct | B wrong |
|---|---:|---:|
| **A correct** | 228 | 20 |
| **A wrong** | 32 | 120 |

A scores $248/400 = 0.62$, B scores $260/400 = 0.65$. Per-item differences $d_i = \text{B}_i - \text{A}_i$ are $+1$ on 32 items, $-1$ on 20, and 0 otherwise. Mean difference $= 12/400 = 0.03$. The variance of $d$ is about $52/400 - 0.03^2 = 0.129$, so the paired $\text{SE} \approx \sqrt{0.129/400} \approx 0.018$ and the 95% interval is about $0.03 \pm 0.035$, which includes zero.

Treating the two scores as independent would give $\text{SE} \approx \sqrt{0.024^2 + 0.024^2} \approx 0.034$, about twice as large. Pairing helps, but here even the paired interval does not exclude "no difference". Only the 52 **discordant** items carry information; McNemar's test is the classical test built on exactly those counts. A **paired bootstrap** resamples items and recomputes the difference each time.

Pairing also applies to loss: compute per-document loss for both models on the same documents and analyze the differences.

### Length and judge bias in LLM-as-judge

For open-ended outputs, a strong model is often used as a **judge**: shown a prompt and one or two responses, it outputs a score or a preference. Zheng et al. (MT-Bench) documented position, verbosity and self-enhancement bias; the format-and-style row is a related concern to check as well:

| Bias | What happens | Check or mitigation |
|---|---|---|
| Position | Prefers the first (or second) response shown | Evaluate both orders; count only consistent verdicts or average |
| Verbosity / length | Prefers longer answers regardless of quality | Report length per system; length-controlled scoring; compare at matched lengths |
| Self-preference | Prefers outputs from its own model family | Use a different judge family; human spot checks |
| Format and style | Rewards markdown, confidence, lists | Strip formatting in a control run |

Length-controlled AlpacaEval (Dubois et al.) fits a regression that predicts the judge's preference from the length difference and other factors, then reports the preference with the length effect removed. Whatever method you use, record the judge model and revision, the exact judge prompt, the decoding settings, and measure agreement with a small set of human labels before trusting the judge.

A quick diagnostic: if the treatment's mean output length grew by 40% and its judge win rate rose, you have not yet learned whether quality improved.

### Contamination and leakage

**Contamination** means evaluation items (or close variants) appear in training data. **Leakage** is the broader family: any route by which information from the evaluation set influences training or model selection. Both inflate scores and cannot be fixed afterward by more careful statistics.

Detection methods:

- **N-gram overlap.** Tokenize or normalize text, collect long n-grams from each test item, and look them up in an index of training n-grams. The GPT-3 paper flagged items by word n-gram overlap (roughly 13-grams; its Appendix C sets $n$ per benchmark, from 8 to 13 words), then compared scores on clean versus flagged subsets. Choose $n$ long enough to avoid common phrases, short enough to catch paraphrased copies.
- **Near-duplicate search.** MinHash similarity between test items and training documents ([L3](03-pretraining.md#deduplication-exact-and-near-duplicate)).
- **Model-based signals** when training data is unavailable. Min-K% probability (Shi et al.) flags text whose least likely tokens are unusually likely for the model. Oren et al. test whether a model assigns higher likelihood to a benchmark's canonical example ordering than to shuffled orderings; exchangeable data should show no preference. These give evidence, not proof.
- **Clean-subset comparison.** If the score on the flagged subset is much higher than on the clean subset, contamination likely helped.

Pipeline leakage in your own experiments:

| Route | Guard |
|---|---|
| Choosing hyperparameters or checkpoints on test | Decide on dev; open test once |
| Tokenizer, filter or classifier fitted on dev/test | Fit on train only |
| Prompt or few-shot examples tuned on test items | Tune on dev; draw few-shot examples from a train split |
| Train/test near-duplicates | Cross-split dedup before training |
| Re-running until a "good" seed appears | Pre-declare seeds; report all runs |

### Scaling laws

A **scaling law** is an empirical fit of loss as a function of model size $N$ (parameters), data $D$ (training tokens) and compute $C$ (FLOPs).

**Compute per token.** For a dense Transformer, a forward pass costs about 2 FLOPs per parameter per token (one multiply and one add per weight); the backward pass costs about twice the forward. So

$$C \approx 6ND.$$

This ignores attention-score FLOPs (which grow with context length $T$), embeddings and normalization; it is good when $d$ is large relative to $T$. Example: $N = 10^7$, $D = 2\times10^8$ gives $C \approx 1.2\times10^{16}$ FLOPs.

**Kaplan et al. (2020)** fitted power laws $L(N)$, $L(D)$, $L(C)$ over several orders of magnitude and concluded that, for a fixed compute budget, model size should grow much faster than data: train large models and stop well before convergence.

**Hoffmann et al. (2022), "Chinchilla"** used three approaches (fixed model sizes with varied tokens, IsoFLOP profiles, and a parametric fit $L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$) and concluded that $N$ and $D$ should grow in roughly equal proportion with compute. The widely quoted rule of thumb of about 20 training tokens per parameter is not stated as a rule in the paper; it is read off their compute-optimal estimates (for example their Table 3). They attribute part of the disagreement with Kaplan to Kaplan's use of one fixed learning-rate schedule for all runs, whereas they matched the schedule to each run's length (see [L3](03-pretraining.md#learning-rate-schedules-warmup-then-decay)), and part to their inclusion of larger models. A later replication attempt (Besiroglu et al.) reconstructed the data from the paper's figures and reported that the published parametric fit was inconsistent with the other two approaches, while a refit agreed better with them.

Worked allocation with the 20-tokens rule: $C = 6ND$ and $D = 20N$ give $C = 120N^2$, so $N = \sqrt{C/120}$. For $C = 1.2\times10^{18}$ FLOPs: $N = 10^8$ parameters and $D = 2\times10^9$ tokens.

**What an allocation rule assumes.** Before applying one, check:

1. The loss is what you care about (not inference cost, a downstream task, or latency). If the model will serve many requests, a smaller model trained longer may be better overall.
2. Data is unique and plentiful. Repeated epochs give diminishing returns not captured by a single-epoch fit.
3. The architecture, tokenizer, data mixture and optimizer match those used in the fit; constants change with any of them.
4. Each run is tuned (especially the schedule matched to its length).
5. The fit's range covers your scale; extrapolating far outside it is a guess.
6. $N$ is counted the same way (with or without embeddings; total versus active parameters for MoE, see [L10](10-frontier.md)).
7. $C \approx 6ND$ holds, which weakens at long context.

### Designing an ablation

An **ablation** changes one factor and measures the effect with everything else held fixed. Template:

| Element | Decide before running |
|---|---|
| Question | One sentence, one factor (e.g. "does a document mask change held-out BPB?") |
| Baseline and treatment | Exactly what differs; diff of the two configs should be one line |
| Matched budget | Same tokens, or same FLOPs, or same wall time — say which |
| Tuning | Same tuning effort for both; or both untuned with a shared LR sweep |
| Seeds | Number of seeds per arm; which seeds |
| Data and splits | Same split hashes; same data order where possible |
| Metric and protocol | Primary metric; frozen evaluation set and settings |
| Effect of interest | Smallest difference you would act on |
| Decision rule | What result counts as "helps", "hurts", "no detectable effect" |

Two traps: changing a factor that silently changes another (a larger vocabulary also changes tokens per byte, so a "same tokens" budget is not the same text), and testing many variants then reporting the best (each extra comparison raises the chance of a false win; report all of them). [L11](11-research.md) turns this template into a full capstone protocol.

### How lm-evaluation-harness structures tasks

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) separates a **task** (data, prompts, scoring) from a **model** (anything that can answer a few request types). Field names below follow the current docs and may change; read the [task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md) for your pinned release.

A task is a YAML config with, among others:

| Field | Role |
|---|---|
| `dataset_path`, `dataset_name` | Which dataset to load (often from the Hugging Face Hub) |
| `training_split`, `validation_split`, `test_split`, `fewshot_split` | Where items and few-shot examples come from |
| `doc_to_text` | Template turning a dataset row into the prompt |
| `doc_to_target` | The reference answer (string or index) |
| `doc_to_choice` | The list of options for multiple choice |
| `output_type` | `loglikelihood`, `loglikelihood_rolling`, `multiple_choice` or `generate_until` |
| `generation_kwargs` | Decoding settings for generation tasks |
| `filter_list` | Post-processing to extract an answer from generated text |
| `metric_list` | Metrics and how to aggregate them |
| `num_fewshot` | Number of in-context examples |
| `metadata` (version) | Task version; scores are only comparable at the same version |

The model side answers three request types: log-likelihood of a continuation given a context, rolling log-likelihood of a whole text (used for perplexity and BPB), and generation until a stop sequence. That interface is why the harness can evaluate a model you wrote yourself if you implement those requests.

For a reproducible run: pin the harness commit, task versions, model and tokenizer revisions, few-shot count, chat-template setting and seeds; save per-sample outputs (the `--log_samples` option); and treat `--limit` as a debugging aid, not a result. Biderman et al. describe the reproducibility pitfalls the harness was built around.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| New tokenizer "wins" by a large perplexity margin | Per-token metric across tokenizers | Recompute as bits per byte on identical raw text |
| Loss differs between two evaluation scripts on the same model | Mean of batch means, different EOS/first-token handling, different context window | Sum NLL and token counts separately; compare counts |
| Treatment beats baseline by less than seed-to-seed spread | Noise | Rerun baseline with two more seeds |
| Benchmark score far above what loss suggests | Contamination | 13-gram or MinHash overlap with training data; clean-subset score |
| Judge win rate rises with no visible quality change | Length or position bias | Compare mean lengths; swap order; human spot check |
| Multiple-choice accuracy changes when you add a space | Tokenization of choices / target delimiter | Inspect the exact strings scored per choice |
| Results differ from a published number | Task version, prompt, few-shot, normalization | Diff the task config and per-sample logs |
| Scaling-law extrapolation is badly off | Outside fitted range, different data or schedule | Plot your points against the fit; check assumptions list |
| Ablation shows a gain that disappears later | Best-of-many selection, tuned treatment only | Count all variants tried; equalize tuning |

## Check yourself

1. Compute the uniform-model loss and perplexity for your own vocabulary size. Is your model's step-0 dev loss close to it?
2. Your evaluation script averages per-batch losses. Construct a dev set where this gives a different answer from the token-weighted mean by more than 0.1 nats.
3. Predict, then run: evaluate the same model on the same text with context windows of 64 and 256 tokens. Which perplexity is lower, and why does that make the window part of the protocol?
4. Two models use different tokenizers. Write down every quantity you need to compare them in bits per byte.
5. Predict, then run: score a multiple-choice set with raw and length-normalized log-likelihood. On which items do they disagree, and what do those items have in common?
6. A 200-item test gives 0.71 for the treatment and 0.66 for the baseline. Before computing anything, guess whether the difference is meaningful; then compute the paired interval from your per-item results.
7. Why does resampling tokens instead of documents in a bootstrap give an interval that is too narrow?
8. Design a check that would detect a judge's position bias using only 30 prompts.
9. Inject five test documents into your training set, train briefly, and measure their loss compared with other test documents. Which detection method from this note would have found them, and which would not?
10. Using $C \approx 6ND$, estimate the FLOPs of your own tiny run. Which neglected term is largest relative to $6ND$ at your context length?
11. Pick one assumption behind the 20-tokens-per-parameter rule that your setup violates, and predict how it would change the best allocation.
12. Write the ablation table for one change you want to test in your model, including the decision rule, before running anything.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [CS336](https://cs336.stanford.edu/) | Evaluation and scaling lectures for the current year | Course backbone for this module |
| [On the State of the Art of Evaluation in Neural Language Models](https://arxiv.org/abs/1707.05589) (Melis et al.) | Main results and discussion | Why tuned baselines matter |
| [The Pile](https://arxiv.org/abs/2101.00027) (Gao et al.) | Evaluation sections | Bits per byte as a tokenizer-independent metric |
| [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (Chen et al.) | pass@k definition and estimator | Unbiased pass@k |
| [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (Miller) | Whole paper | SE, clustered SE and paired analysis for LLM evals |
| [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098) (Bouthillier et al.) | Sources-of-variance sections | Seeds, data order and other training noise (optional) |
| [The Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/) (Dror et al.) | Test selection guide | Which test fits which metric (optional) |
| [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (Zheng et al.) | Sections on judge limitations | Position, verbosity and self-enhancement bias |
| [Length-Controlled AlpacaEval](https://arxiv.org/abs/2404.04475) (Dubois et al.) | Method section | One way to remove length effects from judge scores |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al.) | §4 (Measuring and Preventing Memorization of Benchmarks) and Appendix C | n-gram overlap and clean-subset comparison |
| [Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789) (Shi et al.) | Min-K% method | Membership signal without training data |
| [Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/abs/2310.17623) (Oren et al.) | Method section | Ordering-based contamination test |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al.) | §1.1–1.2 (summaries) and §6 (Optimal Allocation of the Compute Budget) | The first large-scale power-law fits |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al.) | §3 (three estimation approaches) and the Kaplan comparison in §2 | Equal scaling of $N$ and $D$ |
| [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) (Besiroglu et al.) | Whole paper (short) | How fragile a fitted law can be |
| [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) (Liang et al.) | Introduction and metrics taxonomy | Multi-metric evaluation design (optional) |
| [lm-evaluation-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md) and [new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) | YAML fields, output types, metrics | How a task is specified |
| [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) (Biderman et al.) | Pitfalls and recommendations | Why prompts, versions and samples must be pinned |
