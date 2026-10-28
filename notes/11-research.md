# L11 — Defending a reproducible training investigation

> **Milestone:** [L11](../course/ROADMAP.md#milestones) · **Builds:** capstone P15a (question, frozen protocol, baseline), P15b (treatment and failure analysis), P15c (reproduction package and oral defense) in [projects](../course/PROJECTS.md); claim ledger in the [attention-alternatives workshop](../exercises/attention-alternatives/README.md) · **Prerequisites:** [L4](04-evaluation.md) and the relevant parts of [L3](03-pretraining.md) and L5–L10 ([L5](05-sft-peft.md), [L6](06-preferences.md), [L7](07-online-rl.md), [L8](08-efficiency.md), [L9](09-distributed.md))

This module turns everything earlier into one small piece of research you can defend. The goal is not a big result; it is a *narrow, true* result: a bounded question, a fair baseline and treatment, an evaluation fixed before you look, a run anyone can reproduce within a stated budget, an honest account of what failed, and a conclusion no larger than the evidence. The same skills let you read a frontier technical report and say precisely which of its claims its evidence supports. Acceptance criteria come from [the assessment rubric](../course/ASSESSMENT.md), especially [run evidence](../course/ASSESSMENT.md#run-evidence) and [capstone acceptance](../course/ASSESSMENT.md#capstone-acceptance).

## What you should be able to do

- Turn a vague question into a falsifiable hypothesis with a prediction, a metric and a budget.
- Design a baseline and treatment that differ in one controlled factor.
- Freeze an evaluation protocol before running the treatment, and keep dev and test roles separate.
- Write a run manifest that pins code, environment, data, seeds and hashes well enough for someone else to reproduce the run.
- Log resources, separating measured values from estimates.
- Analyze failures, including null and negative results, as evidence.
- Build a claim ledger for a paper: claim, evidence, conditions, missing controls, proposed test.
- Read a frontier technical report critically.
- Defend the result orally: explain one example from data to update, address an unfamiliar variation, and state the limits.
- Connect the outcome back to a preference-learning ([L6](06-preferences.md)) or frontier-report ([L10](10-frontier.md)) question.

## Core ideas

### From question to falsifiable, bounded experiment

Research questions start vague: "Is document masking better?" That cannot be wrong, so it cannot be tested. Tighten it in four moves:

| Step | Output | Running example |
|---|---|---|
| 1. Name the factor | One thing you will change | Packing with plain causal attention vs a document (block-diagonal) mask |
| 2. Name the measurement | One primary metric on a fixed set | Held-out bits per byte on a frozen test split |
| 3. State a prediction with direction and size | What you expect and what would refute it | "The mask lowers BPB by at least 0.01; a difference within ±0.01 counts as no detectable effect" |
| 4. Bound it | Model size, tokens, seeds, wall time, compute | 4M-parameter decoder, 50M tokens, 3 seeds per arm, CPU or one small GPU, fixed hour cap |

A hypothesis is **falsifiable** when some realistic outcome would count against it. Step 3 does that work: by fixing the **smallest effect of interest** in advance, you decide what "no effect" means before seeing data.

A question is **bounded** when it can be finished within a declared budget, including failed attempts and reruns. The capstone projects P15a–P15c are each sized to fit a short window, so a good capstone question is one where a tiny model can show a difference. Questions whose answer only appears at scale ("does this help at 7B?") are not testable here; restate them as "does the mechanism behave as claimed at toy scale, and what would a larger test require?"

**Check resolvability before committing.** If training noise is larger than the effect you care about, the experiment cannot answer the question. Suppose two pilot seeds of the baseline give dev BPB values with a standard deviation of about $\sigma = 0.012$. With $n$ seeds per arm, the standard error of the difference of means is

$$\text{SE}_{\Delta} = \sigma\sqrt{2/n}.$$

For $n = 3$: $0.012 \times 0.816 \approx 0.0098$, so only differences larger than about $2 \times 0.0098 \approx 0.02$ are clearly resolvable. If you care about 0.01, you need more seeds, a larger evaluation set to reduce evaluation noise, or a different question. This calculation uses dev data from pilot runs only.

### Baseline and treatment design

The **baseline** is the configuration you compare against; the **treatment** differs from it in exactly the factor under study. Everything else is a **control** to hold fixed.

| Control | Why it matters |
|---|---|
| Data, splits and order | Different data is a different experiment; record split hashes and shuffle seed |
| Token budget or compute budget | Pick one and match it; a treatment that is slower per token wins "same steps" but may lose "same FLOPs" |
| Tuning effort | Tune both equally, or neither; a tuned treatment against a default baseline is not a fair test |
| Model size and parameter count | If the treatment adds parameters, either match counts or report both |
| Evaluation | Same protocol, same items, same settings |
| Seeds | Same seed list for both arms enables paired comparison |
| Code path | Treatment should be a flag on the same code, not a separate script |

A practical check: the **diff between the two config files should be one line** (or one small block). Anything else in the diff is a confound to remove or justify.

Common confounds: the treatment changes tokens per byte (a different tokenizer), changes effective batch size (a different packing scheme with different padding), or changes the learning-rate schedule length (more steps for the same tokens). Each one alone can produce the result you are tempted to attribute to the treatment.

### Analysis choices and forking paths

Even with a fixed metric, many small decisions remain: which checkpoint, which seeds to include, whether to drop a run that diverged, how to smooth curves, which subsets to report. Each choice made *after* seeing results is a chance to drift toward the answer you hoped for. With enough such forks, a "significant" difference can appear where none exists.

Worked illustration: if you compare five treatment variants against one baseline and each comparison has a 5% chance of a false positive when there is no real effect, the chance that at least one looks like a win is about $1 - 0.95^5 \approx 0.23$. Reporting only the winner hides that.

Guards:

- Decide exclusion rules in advance (for example, "a run whose loss becomes NaN is rerun once with the same seed; both are reported").
- Report every variant you tried, including the ones you abandoned, in an appendix or log.
- If you explore freely on dev, label the result *exploratory* and confirm the chosen variant once, on test or on fresh seeds.

### Frozen evaluation protocol

A **frozen protocol** is the evaluation written down and fixed *before* the treatment is run. It prevents choosing metrics, subsets or checkpoints after seeing which one favors your hypothesis.

Contents:

- Primary metric and units; secondary metrics listed as secondary.
- Evaluation sets with revisions and split hashes; which is dev (for decisions) and which is test (opened once).
- Exact settings: context window and stride, prompt template, few-shot examples, decoding settings, normalization, judge model and prompt if any.
- Which checkpoint is evaluated (last, or best on dev — never best on test).
- Uncertainty method (paired SE, paired bootstrap, seed spread) and the decision rule from your hypothesis.
- Planned sanity checks: initial loss near $\ln V$, a known-answer item, a contamination check between training data and test.

Store the protocol in the repository and record its file hash in each run manifest. If you must change it after seeing results, keep the original, add a dated amendment saying what changed and why, and report results under both. Lightweight preregistration of this kind is what makes "the evaluation supports the narrow conclusion" checkable.

### Run manifests and reproducibility

A **run manifest** is a small file written at the start of each run that records everything needed to rerun it. Its fields follow the course's [run-evidence requirements](../course/ASSESSMENT.md#run-evidence):

| Group | Fields | Example value |
|---|---|---|
| Code | Commit hash, dirty-tree flag, file hash of key modules | `a1b2c3d`, clean |
| Environment | Python and package lock file, device, driver, backend | lock file hash; CPU, 8 threads |
| Model | Config, parameter count, tokenizer revision and hash | 4.1M params; tokenizer hash |
| Data | Source, revision, license, preprocessing steps, split hashes | SHA-256 of sorted document IDs per split |
| Inputs to training | Template, masks, packing and boundary policy | Document mask on; EOS between documents |
| Optimization | Optimizer and hyperparameters, schedule, precision, effective batch, total tokens | AdamW; warmup 300 steps, cosine; fp32; 64×256 tokens per step |
| Randomness | Every seed and the data-order seed | 11, 12, 13 |
| State | Checkpoint paths and what each contains | Weights, optimizer, scheduler, RNG, data position |
| Execution | Exact command, start/end time, wall time, peak memory | Measured |
| Evaluation | Protocol hash, raw per-item outputs, metric files | Path to per-document losses |
| Provenance | Assistance received, deviations, known limitations | Free text |

**Pins** fix versions (packages, model and tokenizer revisions, dataset revisions, harness commit). **Hashes** prove an artifact is the one recorded: hash the data files, the split ID lists, the protocol and the config. **Seeds** make the random parts repeatable on the same software and hardware but not across platforms ([L3](03-pretraining.md#seeds-and-the-limits-of-determinism)); state the tolerance you expect when someone reruns on a different machine.

A reproduction package is complete when a reader with the declared environment can run one documented command per arm, obtain results within the stated tolerance, and regenerate every table and figure from the saved raw outputs. Test this yourself from a fresh clone before calling it done.

### Resource logging

Record resources for every run, including failed and abandoned ones:

| Quantity | Measured or estimated | How |
|---|---|---|
| Wall time | Measured | Timestamps at start and end |
| Peak memory | Measured | Framework or OS peak-memory counter |
| Tokens processed | Measured | Counter in the training loop |
| Throughput | Measured | Tokens per second over a stable window (exclude warm-up and evaluation) |
| Training FLOPs | Estimated | $6ND$ from [L4](04-evaluation.md#scaling-laws); state what it neglects |
| Hardware utilization | Estimated | Estimated FLOPs ÷ (wall time × hardware peak) |
| Cost | Estimated or invoiced | Rate × time; say which |
| Energy | Estimated unless metered | Power draw × time; label the source |

Always label each number *measured* or *estimated*, and never present a CPU timing as evidence about GPU performance. Strubell et al. is the standard reference for why compute and energy reporting matters. The total including failed runs is your real cost; reporting only the final successful run understates it.

### Failure analysis

Most runs fail in some way. A **failure analysis** explains what went wrong and what the failure tells you; the rubric explicitly accepts a well-investigated failed experiment as evidence of understanding.

Classify the failure first:

| Class | Signature | Next step |
|---|---|---|
| Bug | Tiny-batch overfit fails; shapes or masks wrong; resume spike | Fix, add a check, rerun both arms |
| Instability | Loss spikes, NaN, gradient norm explosions | Inspect LR, clipping, precision; rerun with logging |
| Insufficient budget | Curves still dropping steeply at the end | Report as incomplete; estimate the budget needed |
| Null result | Difference inside the pre-declared "no effect" band | Report it as a result; check resolvability was adequate |
| Negative result | Treatment reliably worse | Report it; look for the mechanism |
| Confound found | A second difference between arms | Remove it, rerun, report both versions |

Then look at examples, not just averages. For loss-based questions, sort per-document loss differences between arms and read the documents at both ends: which kinds of text did the treatment help or hurt? For generation tasks, read a fixed random sample of outputs from each arm before reading the scores. Write the analysis as: what you expected, what happened, which evidence distinguishes the candidate explanations, and what you would test next.

### The claim ledger

A **claim ledger** is a table that ties each claim — yours or a paper's — to its evidence and conditions. It is how you read a paper without absorbing its conclusions wholesale, and how you write one without overclaiming. The attention-alternatives workshop uses the same structure for its [Kimi K3 ledger](../exercises/attention-alternatives/README.md#k3-report-thread).

| Column | Meaning |
|---|---|
| Claim | In your own words, as narrow as the text allows |
| Source location | Paper and version, section, equation, table or figure |
| Evidence offered | What the authors actually show (author-reported unless reproduced) |
| Baseline and conditions | What it is compared with, at what scale, budget and data |
| What a toy test can check | The part a small experiment can probe |
| What it cannot establish | The part that depends on scale, data or infrastructure you lack |
| Missing control or ablation | What would separate this explanation from alternatives |
| Proposed test | A bounded experiment, with prediction |
| Status and uncertainty | Supported, partially supported, unsupported, untested |

Worked example with an invented report (not a real paper): *"Our new mixing layer reduces validation loss by 0.05 nats at equal compute."*

| Column | Entry |
|---|---|
| Claim | Replacing attention in every layer with the mixer lowers validation loss at matched training FLOPs |
| Source location | Hypothetical §3, one results table |
| Evidence offered | One run per arm at one model size |
| Baseline and conditions | Baseline uses the authors' older recipe; FLOPs counted with a formula that omits attention scores |
| What a toy test can check | Whether the mixer trains stably and its loss relative to attention at tiny scale under one shared recipe |
| What it cannot establish | Behavior at the reported scale; long-context quality |
| Missing control | Same recipe and tuning for both arms; multiple seeds; measured rather than formula FLOPs |
| Proposed test | Two arms, shared recipe, 3 seeds, matched measured compute; predict a gain no larger than the seed spread |
| Status | Untested; evidence is single-run and recipe-confounded |

For your own capstone, write a ledger row for every sentence in your conclusion. If a sentence has no evidence cell, delete it or turn it into a stated hypothesis for future work.

### Reading a frontier technical report critically

Frontier reports describe systems far beyond what you can train, so you cannot reproduce their headline numbers. You can still audit them. Questions to ask of each claim:

1. **Mechanism or recipe?** Does the evidence isolate the architectural change, or did data, optimizer, tokenizer, scale and post-training change at the same time?
2. **What is the comparison?** Against whom, at what scale, with what budget matching (tokens, FLOPs, wall time, active versus total parameters)?
3. **How many runs?** Single runs at large scale are normal and expensive; note that variance is unknown.
4. **Is evaluation clean?** Are contamination checks described? Are prompts, few-shot settings and judge details given?
5. **What is missing?** Ablations mentioned but not shown, numbers without settings, results only on favorable benchmarks.
6. **What would reproduction require?** Data, code, compute and details not released.

The course's recurring case is **Kimi K3** ([arXiv 2607.24653v2](https://arxiv.org/html/2607.24653v2)). According to the authors, it is a 2.8T-parameter MoE with 104B active parameters, native vision and million-token context, and the report covers architecture including Kimi Delta Attention (§2), pretraining and scaling (§3), post-training (§4), infrastructure (§5) and evaluation (§6). All of these are author-reported claims; none are reproduced in this course. The workshop route is to build a tiny linear-attention and KDA recurrence, then ledger one narrowly scoped §2.1 attention claim, separating what the toy recurrence tests from what depends on the full layer, the hybrid layout and the training recipe. The same questions apply on the later revisits of §3–§6.

Lipton and Steinhardt's "Troubling Trends in Machine Learning Scholarship" catalogs patterns to watch for when reading, such as explanation offered without evidence and gains not attributed to their actual source.

### Defending a result and stating its limits

The oral defense checks that you understand the result, not just that the scripts ran. Prepare to:

- **Trace one example end to end**: a raw document → tokens → packed row → masks → logits → loss → gradients → the update, in both arms if they differ.
- **Reproduce in front of the reviewer**: rerun a short evaluation from the package and get the recorded number.
- **Handle an unfamiliar variation**: the reviewer changes something (a longer sequence length, a missing optimizer state, an injected train/test overlap, a different seed set). Predict the effect, test it, explain the outcome.
- **Separate what was measured from what was inferred**, and the result from its interpretation.

Write the conclusion as one sentence with its conditions attached:

> Under *[model size, data, token budget, seeds, hardware]*, *[treatment]* changed *[primary metric]* by *[effect ± uncertainty]* relative to *[baseline]*; this does not show *[the nearest overclaim]*.

For the running example, a well-formed limit would be: "this does not show that document masking helps at larger scale, longer context, or on downstream tasks." The rubric is explicit that no frontier-quality or general superiority claim follows from a toy result.

**Connect back.** End by stating what the result means for a larger question you care about: for example, what it implies about which tokens should carry loss in a preference objective ([L6](06-preferences.md)), or which K3 claim it makes more or less plausible and which test would be next ([L10](10-frontier.md)). That connection is a hypothesis, not a finding; label it so.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Result disappears on rerun | Single seed; effect below noise floor | Compute seed spread; compare with effect size |
| Treatment wins but config diff is large | Confounded comparison | Diff the configs; remove everything but the factor |
| Metric or subset chosen after seeing results | Protocol not frozen | Compare protocol hash in manifests with the reported one |
| Reviewer cannot rerun the package | Unpinned dependencies, missing data revision, hidden local files | Fresh clone in a clean environment |
| Reported cost far below reality | Failed runs and pilots omitted | Sum resource logs over all runs |
| Conclusion broader than evidence | No claim ledger for the write-up | One ledger row per sentence |
| Test score improved by checkpoint selection | Best-on-test chosen | Confirm selection used dev only |
| Null result reported as "does not work" | Resolvability not checked | Show the SE of the difference against the effect of interest |
| Paper claim accepted at face value | Mechanism and recipe changes conflated | Ledger the comparison conditions |

## Check yourself

1. Write your capstone question in one sentence. Now write the outcome that would make you abandon the hypothesis. Is it realistic within your budget?
2. From two pilot baseline seeds, estimate $\sigma$ and compute the smallest difference you could resolve with your planned number of seeds. Does your question survive?
3. Diff your baseline and treatment configs. Explain every line of the diff.
4. Name a quantity your treatment might change besides the one you intend (tokens per byte, effective batch, parameter count, schedule length). How will you hold it fixed or report it?
5. Have someone else (or a fresh clone in a clean environment) reproduce one run from your manifest alone. What was missing?
6. Which numbers in your resource log are measured and which are estimated? Would a reader be able to tell?
7. Predict, then run: rerun your baseline with one restored-state item deliberately missing (for example the scheduler). Can your failure analysis identify it from the logs alone?
8. Sort per-document loss differences between arms. Read the five documents at each end. Do they suggest a mechanism, or a confound?
9. Build a claim-ledger row for one claim in a paper you have read. What is the most important missing control?
10. For one K3 §2.1 claim, separate what your tiny recurrence can test from what it cannot.
11. Write your conclusion sentence with its conditions. Then write the most tempting overclaim and explain why the evidence does not support it.
12. What unfamiliar variation would you least like a reviewer to ask about? Run it before the defense.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [Course assessment rubric](../course/ASSESSMENT.md) | Run evidence; capstone acceptance | The criteria this module is assessed against |
| [Show Your Work: Improved Reporting of Experimental Results](https://arxiv.org/abs/1909.03004) (Dodge et al.) | Expected validation performance and reporting checklist | Reporting results as a function of tuning budget |
| [Improving Reproducibility in Machine Learning Research](https://arxiv.org/abs/2003.12206) (Pineau et al.) | Reproducibility checklist sections | What reviewers expect a reproduction package to contain |
| [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist) | Experiments, statistical significance and compute items | A concrete checklist to self-audit a write-up |
| [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560) (Henderson et al.) | Seeds and reporting sections | How seed variance can reverse conclusions |
| [On the State of the Art of Evaluation in Neural Language Models](https://arxiv.org/abs/1707.05589) (Melis et al.) | Main results | Equal tuning for baselines |
| [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098) (Bouthillier et al.) | Recommendations | Which sources of variance to randomize (optional) |
| [Troubling Trends in Machine Learning Scholarship](https://arxiv.org/abs/1807.03341) (Lipton and Steinhardt) | Whole paper | Patterns to avoid when writing and to catch when reading |
| [Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02243) (Strubell et al.) | Methods for estimating cost and energy | Why resource logging matters |
| [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (Mitchell et al.) | Card sections | Template for documenting a trained artifact (optional) |
| [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) (Biderman et al.) | Recommendations | Evaluation reproducibility specifically for LMs |
| [Kimi K3: Open Frontier Intelligence, v2](https://arxiv.org/html/2607.24653v2) (Kimi Team) | §2.1 first; §3–§6 on later revisits | Frontier-report case for the claim ledger |
| [Attention alternatives workshop](../exercises/attention-alternatives/README.md) | K3 report thread and ledger fields | How the course applies the ledger to K3 |
