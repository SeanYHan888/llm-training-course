# L3 — Pretraining and resume

> **Milestone:** [L3](../course/ROADMAP.md#milestones) · **Builds:** [Transformer workshop](../exercises/transformer/README.md) (training step), P04 one-batch overfit, P05 reproducible run and checkpoint-resume comparison, data portion of P06 in [projects](../course/PROJECTS.md) · **Prerequisites:** [L1](01-neural-learning.md), [L2](02-transformer.md)

Pretraining turns a tiny decoder that can compute logits into a model that has actually learned from a corpus. Most of the difficulty is not in the model: it is in deciding which text goes in, keeping evaluation text out, feeding tokens efficiently, keeping the optimizer stable, and being able to stop and restart without silently changing the experiment. This note covers the path from raw documents to a resumable training run and the evidence you read along the way.

## What you should be able to do

- Record where a corpus came from, under what license or terms, and which revision you used.
- Explain heuristic and model-based quality filters and what each can remove that you wanted to keep.
- Remove exact and near-duplicate documents, and explain how MinHash estimates similarity.
- Split train/dev/test by document *before* any tuning, and name the ways leakage creeps back in.
- Tokenize a corpus once, store it compactly and pack it into fixed-length training rows, handling document boundaries and EOS deliberately.
- Describe the full training step and loop, including AdamW state, gradient clipping and a warmup/decay schedule.
- Say what a seed does and does not guarantee.
- List every piece of state a checkpoint needs, and predict what goes wrong when one is missing on restart.
- Use "overfit a tiny batch" as a debugging tool and read a loss curve for the usual failure shapes.
- Demonstration: train a tiny model with held-out loss, overfit a tiny batch, restart and explain what state was restored, and investigate a leakage or broken-schedule case.

## Core ideas

### Corpus provenance and licensing

**Provenance** is the record of where each piece of data came from and what was done to it. A corpus is not "some text files"; it is a set of sources, each with an origin, a collection date or snapshot, terms of use, and a processing history. Without that record you cannot answer basic questions later: is a benchmark inside my training data, may I redistribute this checkpoint, why does the model know this name?

A minimal per-source record:

| Field | Example | Why it matters |
|---|---|---|
| Source and URL | a public-domain book collection, a Hugging Face dataset ID | Lets someone else fetch the same thing |
| Revision / snapshot | dataset commit hash, crawl date | Moving datasets change under the same name |
| License or terms | CC-BY-4.0, "research only", unknown | Determines what you may do with derived artifacts |
| Collection method | crawl, API dump, manual export | Tells you what biases and noise to expect |
| Personal data notes | contains emails? names? | Drives filtering and release decisions |
| Processing applied | language filter, dedup, split hash | Makes your token counts reproducible |
| Size | documents, bytes, tokens (after your tokenizer) | Needed for budgets and scaling reasoning |

Licenses on aggregated datasets are often missing or wrong; a large audit of popular fine-tuning collections (Longpre et al., Data Provenance Initiative) found that dataset-hosting sites often omit licenses and frequently miscategorize them. Treat a dataset card's license field as a claim to check, not a fact. This course does not give legal advice: for a toy run, prefer sources whose terms are clear and record "unknown" honestly rather than guessing.

"Datasheets for Datasets" (Gebru et al.) is a good template for the questions to ask. In code this shows up as a small data card next to your P05 run, and the dataset revision field in your run manifest (see [L11](11-research.md)).

### Filtering: heuristic and model-based

Raw web text contains boilerplate, menus, spam, machine-generated pages, encoding garbage and near-empty documents. **Filtering** removes documents (or lines) that are unlikely to help. Two families:

**Heuristic filters** are hand-written rules over cheap statistics. Typical examples, in the spirit of the C4 and Gopher pipelines:

- document length in words outside a range;
- mean word length outside a range;
- fraction of lines ending in terminal punctuation too low;
- ratio of symbols (such as `#` or `...`) to words too high;
- fraction of duplicated lines or repeated n-grams too high;
- too few common function words ("the", "and", ...) for the claimed language;
- language-identification score below a threshold.

**Model-based filters** score each document with a learned model:

- *Perplexity filtering*: train a small n-gram model on a reference corpus (CCNet used Wikipedia), score every document's perplexity under it, and keep, bucket or drop documents by that score.
- *Classifier filtering*: train a classifier to distinguish "reference-like" text from random web text and keep high scorers. FineWeb-Edu, described in the FineWeb paper, trains a classifier on LLM-produced educational-value annotations.

Worked example (heuristic). A page has 40 lines; 6 end in `.`, `!` or `?`; 31 are menu items such as "Home", "Contact". The terminal-punctuation fraction is $6/40 = 0.15$. A rule requiring at least 0.3 drops it. A poem with no punctuation would be dropped by the same rule: every filter has false positives.

What to watch:

- Filters encode a notion of "quality" that can remove dialects, code, tables or whole topics. Inspect a random sample of *removed* documents, not only kept ones.
- A classifier trained on text resembling a benchmark can quietly pull benchmark-like (or benchmark) text into training. Note what the positives were.
- Filter on training data only after splitting rules are fixed, and apply the same rules to dev so the distributions match.

### Deduplication: exact and near-duplicate

Web corpora contain the same text many times: mirrors, quoted articles, templates, license boilerplate. Duplicates waste compute, overweight some documents, encourage memorization, and, most dangerously for evaluation, put copies of held-out text into training. Lee et al. found substantial duplication in widely used corpora and reported that deduplication reduced how often models emitted memorized training text.

**Exact deduplication.** Normalize each document (for example lowercase, collapse whitespace, strip a trailing signature), hash it with a cryptographic hash such as SHA-256, and keep one document per hash. Cheap and exact, but a single changed character defeats it.

**Near-duplicate deduplication.** Represent each document as a set of **shingles**, for example all overlapping 5-word sequences. Similarity between documents $A$ and $B$ is the **Jaccard similarity**

$$J(A,B) = \frac{|A \cap B|}{|A \cup B|}.$$

Tiny example: $A = \{a,b,c,d\}$, $B = \{b,c,d,e\}$. The intersection has 3 shingles, the union 5, so $J = 0.6$.

Comparing every pair of $N$ documents costs $O(N^2)$ set operations, which is impossible at scale. **MinHash** compresses each set into a short signature. Pick a random hash function $h$; the MinHash of a set is $\min_{s \in A} h(s)$. The key fact: for a random $h$,

$$\Pr\big[\min h(A) = \min h(B)\big] = J(A,B).$$

With $k$ independent hash functions, the fraction of matching signature entries estimates $J$, with standard error about $\sqrt{J(1-J)/k}$.

**Locality-sensitive hashing (LSH)** avoids comparing all signatures. Split the $k$ entries into $b$ bands of $r$ rows ($k = br$); two documents become candidates if *any* band matches exactly. The candidate probability for true similarity $s$ is

$$P(s) = 1 - (1 - s^r)^b.$$

Worked numbers with $k = 128$, $b = 16$, $r = 8$:

| True similarity $s$ | $P(\text{candidate})$ |
|---:|---:|
| 0.5 | about 0.06 |
| 0.7 | about 0.61 |
| 0.8 | about 0.95 |

The curve switches around $(1/b)^{1/r} \approx 0.71$. Choosing $b$ and $r$ chooses your threshold and your false-positive/false-negative tradeoff. Candidates are then usually verified with an exact Jaccard computation.

Two practical rules: deduplicate **across** your future splits (a near-duplicate pair split between train and test is leakage), and record how many documents each stage removed. A reference implementation of MinHash LSH exists in the `datasketch` library if you want to compare against your own reasoning; pin its version if you use it.

### Train/dev/test splits before tuning

- **Train**: the only data that produces gradients.
- **Dev** (validation): used for decisions — learning rate, early stopping, which run is better.
- **Test**: touched once, after decisions are frozen, to report the result.

Split **before** you tune anything, including filter thresholds you choose by looking at model results. Split by **document** (or by source, author or time when those leak information), never by packed chunk: if one article contributes chunks to both train and dev, dev loss measures memorization of that article.

A deterministic, reproducible split assigns each document by a hash of a stable ID: for example, take the SHA-256 of the document ID, interpret it as an integer, and send residues 0–1 (mod 100) to test, 2–3 to dev, the rest to train. The same document always lands in the same split, across reruns and corpus additions. Store a hash of each split's sorted ID list in the manifest.

Ways leakage gets back in after a clean split:

| Route | Example |
|---|---|
| Near-duplicates | The same news story syndicated on two sites, one in each split |
| Chunk-level splitting | Splitting after packing instead of before |
| Tuning on test | Picking the checkpoint with the best test loss |
| Preprocessing fitted on all data | Training the tokenizer or a quality classifier on dev/test text |
| External benchmarks | A test set that also appears in a crawled corpus ([L4](04-evaluation.md#contamination-and-leakage)) |

### Tokenizing a corpus

Tokenize once, offline, with a **fixed tokenizer revision**, and store the result. Typical layout:

- one flat array of token IDs per split, documents concatenated with an end-of-sequence (EOS) token between them;
- an offsets array giving where each document starts, so you can recover boundaries;
- a small metadata file: tokenizer hash, vocabulary size $V$, number of documents, number of tokens, bytes of raw text.

Choose the smallest integer type that holds $V$: `uint16` covers IDs up to 65,535, so a vocabulary of 32,000 fits; larger vocabularies need `uint32`. For example, 50 million tokens in `uint16` occupy 100 MB. A memory-mapped array lets the loader read slices without loading the whole file.

Also record **bytes per token** for each split (raw bytes ÷ tokens). You will need it to convert loss to bits per byte in [L4](04-evaluation.md#perplexity-and-bits-per-byte), and a sudden change between splits is a hint that one split was tokenized or cleaned differently.

The Hugging Face Datasets `map`/`filter` operations are a convenient way to do this while keeping a cache and fingerprint per step; see [Datasets processing](https://huggingface.co/docs/datasets/process).

### Batching and sequence packing

A training batch of token IDs has shape `[B, T]`. For next-token prediction you need $T+1$ consecutive tokens per row: inputs are positions $0..T-1$ and targets are positions $1..T$ (the shifted labels from [L2](02-transformer.md)).

Documents have variable length. Two ways to fill rows:

**Padding.** One document per row, truncated to $T$ and padded to $T$. Padding positions are excluded from the loss with a mask or ignore index. Simple, but wasteful.

**Packing.** Concatenate documents with EOS between them into one stream and cut fixed windows. No padding, every position trains.

Worked example with $T = 512$ and four documents of 300, 700, 150 and 900 tokens:

| Strategy | Rows | Real tokens trained | Padding | Tokens discarded |
|---|---:|---:|---:|---:|
| One document per row, pad/truncate | 4 | $300+512+150+512 = 1474$ | $2048 - 1474 = 574$ (28%) | $188 + 388 = 576$ by truncation |
| Pack with EOS (adds 4 tokens) | 4 full rows (2048) | 2048 | 0 | 6 left over (carried to next batch) |

(EOS counted as part of each document in the packed row; $301+701+151+901 = 2054$.)

Packing raises a question: **should tokens attend across document boundaries?** With a plain causal mask, the first tokens of document 2 can attend to the end of document 1 in the same row.

| Option | What changes | Cost / consequence |
|---|---|---|
| Allow cross-document attention | Nothing; EOS is the only boundary signal | Simplest and common in pretraining; model must learn that EOS means "context reset" |
| Document (block-diagonal causal) mask | Attention mask becomes `[B, T, T]` with blocks per document, or per-token document IDs `[B, T]` | No contamination between documents; needs mask support in attention |
| Reset position IDs per document | Positions restart at 0 after each EOS | Matches inference (documents start at position 0); interacts with RoPE |

Krell et al. analyze packing "without cross-contamination" and show how masks and positions must change. Whichever you choose, write it in the manifest, because it changes what the loss measures.

Shape summary for one packed batch:

| Tensor | Shape | Notes |
|---|---|---|
| Window of IDs | `[B, T+1]` | Sliced from the token stream |
| Inputs | `[B, T]` | First $T$ |
| Targets | `[B, T]` | Last $T$ |
| Loss mask (optional) | `[B, T]` | 0 for padding or positions you choose not to train |
| Document IDs (optional) | `[B, T]` | For a document mask or position reset |
| Logits | `[B, T, V]` | From the model |

In code this shows up as the data-loading function that feeds your P04 training step and, later, the P05 run.

### The training step and loop

One **training step** does, in order:

| Phase | What happens | State touched |
|---|---|---|
| Fetch | Take the next `[B, T]` inputs and targets | Data position advances |
| Forward | Compute logits `[B, T, V]` | Activations stored for backward |
| Loss | Mean cross-entropy over the counted target tokens | Scalar |
| Backward | Gradients for every trainable parameter | `.grad` filled (added to existing values) |
| Clip | Rescale gradients if their global norm is too large | `.grad` scaled |
| Update | Optimizer uses gradients and its own state to change weights | Weights, optimizer moments, step count |
| Schedule | Set the learning rate for the next step | Scheduler step |
| Clear | Reset gradients so the next step starts from zero | `.grad` cleared |

The **loop** repeats the step and, on separate cadences, evaluates on fixed dev batches in evaluation mode without gradient tracking, logs metrics, and saves checkpoints. Log at least: step, tokens seen, train loss, dev loss, learning rate, gradient norm (before clipping), and throughput.

With **gradient accumulation** you run several micro-batches before one update, so the effective batch is larger than what fits in memory. The gradients add up, so each micro-batch's loss must be weighted so the sum equals the loss you would have computed on the combined batch. If micro-batches contain different numbers of counted tokens, dividing each by the number of micro-batches is *not* the same as a token-weighted mean; [L9](09-distributed.md) returns to this across devices.

### AdamW and its state

Adam keeps two running averages per parameter. For parameter $\theta$ with gradient $g_t$ at step $t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta_t\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\,\theta_{t-1}\right)$$

Symbols: $m$ is the first moment (momentum), $v$ the second moment (mean squared gradient), $\beta_1, \beta_2$ their decay rates, $\hat m, \hat v$ the bias-corrected versions (both start at 0, so early averages are too small), $\eta_t$ the learning rate at step $t$, $\epsilon$ a small constant, and $\lambda$ the weight-decay coefficient. **AdamW** (Loshchilov and Hutter) *decouples* weight decay: the $\lambda\theta$ term shrinks weights directly instead of being added to the gradient, where Adam's per-coordinate scaling would distort it. Implementations differ in exactly how $\eta$ multiplies $\lambda$: PyTorch multiplies it by the full learning rate, as written here, while the AdamW paper scales it only by the schedule multiplier. Check the pinned PyTorch docs.

Worked step: $\theta_0 = 0.5$, $g_1 = 0.2$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.01$, $\lambda = 0.1$, $\epsilon \approx 0$.

- $m_1 = 0.1 \times 0.2 = 0.02$; $\hat m_1 = 0.02 / 0.1 = 0.2$
- $v_1 = 0.001 \times 0.04 = 0.00004$; $\hat v_1 = 0.00004/0.001 = 0.04$; $\sqrt{\hat v_1} = 0.2$
- Adam part: $0.01 \times 0.2/0.2 = 0.01$; decay part: $0.01 \times 0.1 \times 0.5 = 0.0005$
- $\theta_1 = 0.5 - 0.01 - 0.0005 = 0.4895$

Notice the first update has size about $\eta$ **regardless of the gradient's scale**: $\hat m/\sqrt{\hat v} = g/|g| = \pm 1$. Every coordinate moves by the full learning rate at once. That is one reason to warm up the learning rate.

**State size.** AdamW stores $m$ and $v$ for every parameter, plus a step counter. A 30M-parameter model in fp32 has 120 MB of weights and another 240 MB of optimizer moments. Weight decay is commonly not applied to biases and normalization gains; that is a choice to record. In code this shows up as the optimizer's `state_dict()`, which you must save (see Checkpoints and restart).

### Gradient clipping

A rare bad batch or an unstable layer can produce a huge gradient and one destructive update. **Global-norm clipping** computes the norm of all gradients concatenated,

$$\|g\| = \sqrt{\sum_{\text{all parameters } p} \sum_i g_{p,i}^2},$$

and if $\|g\| > c$ multiplies every gradient by $c/\|g\|$. Direction is preserved; only the length shrinks. Pascanu et al. motivated this for recurrent networks; it is now standard for Transformers.

Example: two parameter tensors with gradients $[3, 4]$ and $[12]$. The global norm is $\sqrt{9+16+144} = 13$. With $c = 1$, both are multiplied by $1/13$, giving $[0.231, 0.308]$ and $[0.923]$.

In PyTorch, `torch.nn.utils.clip_grad_norm_` clips in place and **returns the total norm before clipping**. Log that value: a gradient norm that is clipped on almost every step means $c$ is too small or the learning rate too high; isolated spikes point at specific batches. Clip after backward and before the optimizer step (and after unscaling, if you use fp16 loss scaling).

### Learning-rate schedules: warmup then decay

A **schedule** sets $\eta_t$ each step. A common shape is linear **warmup** from near 0 to a peak $\eta_{\max}$ over $W$ steps, then **decay** to $\eta_{\min}$ by the final step $S$.

Linear warmup, for $t < W$: $\eta_t = \eta_{\max} \cdot t / W$.

Cosine decay, for $W \le t \le S$, with progress $p = (t - W)/(S - W)$:

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi p)\big).$$

Linear decay replaces the cosine term with $(1-p)$. Worked numbers with $\eta_{\max} = 6\times10^{-4}$, $\eta_{\min} = 6\times10^{-5}$, $W = 200$, $S = 5000$:

| Step | Phase | $\eta_t$ |
|---:|---|---:|
| 100 | warmup | $3.0\times10^{-4}$ |
| 200 | peak | $6.0\times10^{-4}$ |
| 2600 | cosine, $p=0.5$ | $3.3\times10^{-4}$ |
| 5000 | end | $6.0\times10^{-5}$ |

Why warm up: early gradients are large and Adam's moment estimates are based on very few samples, so full-size steps at step 1 can push the model into a bad region. Why decay: a smaller step at the end lets the model settle into a lower-loss region rather than bouncing around. The Chinchilla authors report that setting the cosine cycle much longer than the actual run gives worse models at the point you stop, so **the schedule length should match the training length**. Schedules that hold the rate constant and decay only at the end (warmup–stable–decay) are studied in "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" (Hägele et al.).

A **broken schedule** is easy to create: stepping the scheduler once per epoch when it was configured in steps (the run never leaves warmup), forgetting to step it at all, computing $S$ from the wrong batch size, or restarting without restoring it. Always log $\eta_t$ and plot it next to the loss.

### Seeds and the limits of determinism

A **seed** initializes a pseudo-random number generator. A training run uses several: Python's `random`, NumPy, PyTorch's CPU generator, each GPU's generator, and generators inside data loaders (shuffling, worker processes). Fixing them makes initialization, shuffling and dropout repeatable *on the same software and hardware*.

A seed does not guarantee identical results everywhere:

- Floating-point addition is not associative. In float32, $(10^8 + 1) - 10^8 = 0$ because $10^8 + 1$ rounds to $10^8$, while $(10^8 - 10^8) + 1 = 1$. Any change in summation order (different thread count, kernel, batch split or hardware) changes low bits, and training amplifies small differences.
- Some GPU kernels are nondeterministic by design (for example, ones using atomic additions). PyTorch can be asked to use deterministic algorithms and will raise an error where none exists; see the [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html), which also state that results are not guaranteed across releases or platforms.
- Different library versions may change defaults or kernels.

So: record seeds, versions and device; on CPU with fixed thread counts expect bitwise or near-bitwise repeatability; elsewhere define a tolerance. Seeds also matter for **experimental** reasons: run-to-run variation across seeds is the noise floor for comparisons in [L4](04-evaluation.md#uncertainty).

### Checkpoints and restart

A **checkpoint** is everything needed to continue the run as if it had never stopped. Weights alone are a model snapshot, not a training checkpoint.

| State | What it contains | If missing on restart |
|---|---|---|
| Model weights | Parameter and buffer tensors | You restart from scratch |
| Optimizer state | $m$, $v$ per parameter, step counts | Moments reset to 0 and bias correction restarts: every coordinate takes a full-$\eta$ step, often a loss spike |
| Scheduler state | Current step / last LR | Warmup reruns, or LR jumps to peak in the middle of decay |
| Global step and tokens seen | Counters | Logging, schedule and stopping are misaligned |
| RNG states | Python, NumPy, torch CPU, each CUDA device | Different dropout/shuffle; run is valid but not the same run |
| Data position | Epoch, shuffle seed/permutation, index or byte offset in the stream | Data repeated or skipped; "epoch 1" no longer means one pass |
| Mixed-precision scaler (fp16) | Current loss scale | Overflow or wasted steps while scale re-adapts |
| Config and identity | Model config, code commit, tokenizer and dataset hashes | You may load into a different model or data pipeline without noticing |
| Best/early-stopping metrics | Best dev loss so far, patience counter | Early stopping behaves differently |

Practical points:

- Save to a temporary file then rename, so a crash mid-write does not destroy the last good checkpoint; keep the last few.
- `state_dict()` for the model, optimizer and scheduler are the standard containers; see [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`, which loads only tensors, primitive types, dicts and explicitly allowlisted types. Non-tensor objects such as NumPy RNG state can then fail to load: allowlist them with `torch.serialization.add_safe_globals`, or pass `weights_only=False` for checkpoints you created yourself. Check the behavior of your pinned version.
- Create the optimizer *after* moving the model to its device and loading weights, then load the optimizer state, so moments live on the right device and refer to the right parameters.
- **Test resume**: compare an uninterrupted run with one that stops and resumes at the same step, using the same data order. Decide in advance what agreement you expect (bitwise on CPU, a tolerance elsewhere) and which divergence would indicate missing state.

### Overfitting a tiny batch

Before a real run, take one small fixed batch (a handful of short sequences), turn off dropout and weight decay, and train on it repeatedly. A correct model and loop should drive the loss close to zero, because a model with far more parameters than target tokens can memorize them.

Start-of-training sanity check: with random initialization and small output logits, predictions are near uniform, so the initial loss should be close to $\ln V$. For $V = 512$ that is about 6.24 nats; for $V = 1000$, about 6.91. A much larger initial loss suggests overconfident initialization (logits too large).

If the tiny batch will not overfit, the bug is almost always in the plumbing, not in model capacity:

- targets not shifted (the model is asked to copy the current token, which is trivially easy; loss collapses suspiciously fast) or shifted the wrong way;
- loss mask zeroing everything, or padding counted as targets;
- gradients never cleared, or never reaching parameters (a detached tensor, frozen parameters, a parameter missing from the optimizer);
- learning rate 0 because the scheduler starts at 0 and never steps.

A loss floor above zero can also be legitimate: if two identical contexts in the batch have different next tokens, no model can predict both. Andrej Karpathy's "A Recipe for Training Neural Networks" describes this habit and related checks.

### Reading loss curves

Plot train and dev loss against **tokens seen** (steps are misleading if batch size changes), with a log-scaled x-axis for long runs. Plot learning rate and gradient norm underneath on the same x-axis.

| Shape | Typical reading |
|---|---|
| Starts near $\ln V$, drops fast, then a plateau | Model learns token frequencies first (unigram entropy), then context use |
| Train and dev fall together, small gap | Healthy for a corpus seen about once |
| Train keeps falling, dev flattens then rises | Overfitting: repeated data, too small a corpus or too many epochs |
| Dev *lower* than train for long periods | Dropout active only in training, dev easier than train, or leakage |
| Staircase drops at epoch boundaries | Data repetition; the model recognizes text it has seen |
| Isolated spikes that recover | Bad batches or LR too high; check gradient norm at the same step |
| Sudden rise to NaN or a flat high value | Divergence: LR, missing clipping, fp16 overflow, or a corrupted batch |
| Spike exactly at a restart | Missing optimizer or scheduler state |
| Loss stalls early, LR curve tiny | Schedule bug (still in warmup, or wrong total steps) |

Train loss is noisy because each batch differs; smooth it for reading but keep the raw log. Dev loss should be computed on the **same fixed** dev batches every time so points are comparable.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Dev loss suspiciously low, close to train | Chunk-level split, near-duplicates across splits | Recompute split by document ID; MinHash dev against train |
| Initial loss far above $\ln V$ | Output logits too large at init | Print logit standard deviation at step 0 |
| Loss collapses to near zero in a few steps on real data | Targets equal inputs (no shift) | Print the first few input and target IDs side by side |
| Tiny batch will not overfit | Mask, cleared/missing gradients, LR 0 | Print LR, gradient norm, and count of non-masked targets |
| Loss spike right after resume | Optimizer or scheduler state not restored | Compare LR and optimizer step count before stop and after resume |
| LR stays tiny for the whole run | Scheduler stepped per epoch or never | Plot the logged LR |
| Gradient norm always above clip threshold | Clip too low or LR too high | Log pre-clip norm; try a short LR sweep |
| Resumed run sees data again | Data position not saved | Log document IDs or offsets of the first batch after resume |
| Two runs with the same seed differ | Nondeterministic kernels, thread count, versions | Repeat on CPU with fixed threads; enable deterministic algorithms |
| Token count differs between reruns | Tokenizer or filter revision changed | Compare tokenizer hash and per-stage document counts |

## Check yourself

1. For one source in your corpus, fill in the provenance table. Which field could you not answer, and what would you do about it before releasing a checkpoint?
2. Pick one heuristic filter. Find a document it removes that you would want to keep. What does that tell you about using it on a different domain?
3. With $b = 32$ bands of $r = 4$ rows, where is the LSH threshold roughly? Predict whether more or fewer non-duplicate pairs become candidates than with $b = 16$, $r = 8$, then compute $P(0.5)$ to check.
4. Why must deduplication run across train, dev and test together rather than within each split?
5. Predict, then run: shuffle the order of documents in your packed stream and re-measure dev loss on a fixed dev set. Should it change? What about switching between cross-document attention and a document mask?
6. Your packed rows put an EOS between documents. What is the model asked to predict at the position *after* an EOS, and is that a useful target?
7. Using the AdamW equations, explain why restarting with a fresh optimizer but restored weights causes a spike even at a low learning rate.
8. Predict, then run: resume a short run with weights and optimizer state but *not* the scheduler. Sketch the LR curve you expect before plotting it.
9. Predict, then run: set the gradient-clip threshold 10 times smaller. What happens to the logged pre-clip norm, the loss curve and the effective step size?
10. Your tiny-batch overfit plateaus at a loss of about 0.4 instead of near zero. List two causes that are bugs and one that is not.
11. Run the same short training twice with the same seed, then once with a different thread count. Which differences do you see, and at which step do the losses first differ?
12. A colleague reports a dev loss lower than train loss for the entire run. Name three checks you would perform, in order.

## Going deeper

| Source | Read | Why |
|---|---|---|
| [CS336](https://cs336.stanford.edu/) | Data and optimization lectures for the current year | Course backbone for this module |
| [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al.) | Question lists by section | Template for provenance records |
| [The Data Provenance Initiative](https://arxiv.org/abs/2310.16787) (Longpre et al.) | Introduction and licensing findings | Why license fields need verification |
| [The Pile](https://arxiv.org/abs/2101.00027) (Gao et al.) | Dataset composition and processing sections | Concrete multi-source corpus with documented processing |
| [Exploring the Limits of Transfer Learning (T5/C4)](https://arxiv.org/abs/1910.10683) (Raffel et al.) | The C4 corpus description | Classic heuristic web filters |
| [Gopher](https://arxiv.org/abs/2112.11446) (Rae et al.) | MassiveText data appendix | Quality and repetition heuristics |
| [CCNet](https://arxiv.org/abs/1911.00359) (Wenzek et al.) | Pipeline and perplexity filtering | Model-based filtering with a reference LM |
| [RefinedWeb](https://arxiv.org/abs/2306.01116) (Penedo et al.) | Filtering and deduplication pipeline | Web-only corpus with heavy dedup |
| [FineWeb](https://arxiv.org/abs/2406.17557) (Penedo et al.) | Pipeline ablations and FineWeb-Edu | Measured effect of each filtering/dedup step |
| [Dolma](https://arxiv.org/abs/2402.00159) (Soldaini et al.) | Data pipeline and documentation | Open corpus with released tooling (optional) |
| [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) (Lee et al.) | Methods and train–test overlap findings | Exact and near-dup dedup in practice |
| *On the Resemblance and Containment of Documents* (Broder, 1997) | Whole paper (short) | Original MinHash idea |
| [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027) (Krell et al.) | Packing formulation and attention-mask changes | Document boundaries in packed rows |
| [Adam](https://arxiv.org/abs/1412.6980) (Kingma and Ba) | Algorithm 1 and bias-correction section | The update rule |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov and Hutter) | Sections on decoupling | Why AdamW differs from Adam + L2 |
| [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | Algorithm box and defaults | What your pinned PyTorch actually computes |
| [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063) (Pascanu et al.) | §3.2 (Scaling down the gradients) | Origin of norm clipping |
| [clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | Return value and arguments | Logging the pre-clip norm |
| [SGDR](https://arxiv.org/abs/1608.03983) (Loshchilov and Hutter) | Cosine annealing definition | Origin of cosine schedules |
| [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) (Goyal et al.) | Warmup section | Why warmup helps large steps (optional) |
| [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) (Hoffmann et al.) | Appendix B (optimal cosine cycle length) and Figure A1 | Schedule length must match run length |
| [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392) (Hägele et al.) | Warmup–stable–decay sections | Alternative schedule (optional) |
| [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Whole page | What seeds and deterministic modes do |
| [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | General checkpoint section | Checkpoint containers |
| [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (Karpathy) | Whole post | Overfit-one-batch and other debugging habits |
| [datasketch](https://github.com/ekzhu/datasketch) | MinHash LSH docs | Library to cross-check your dedup reasoning (optional) |
