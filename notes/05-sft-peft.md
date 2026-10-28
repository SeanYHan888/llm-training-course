# L5 — Supervised fine-tuning and parameter-efficient adaptation

> **Milestone:** [L5](../course/ROADMAP.md#milestones) · **Builds:** P07 (one SFT example traced through template, collator, masks and loss) and P08 (bounded adapter run, reload and held-out comparison) in [projects](../course/PROJECTS.md) · **Prerequisites:** [L2](02-transformer.md) (shifted labels, causal masks), [L3](03-pretraining.md) (training loop, packing, checkpoints), [L4](04-evaluation.md) (held-out comparisons)

Up to now you have trained a model you built yourself. This module adapts an *existing* pretrained model so it follows a conversational format and a task. The mechanics are the same next-token loss you already know. The new work is mostly bookkeeping: knowing exactly which string the model sees, which tokens it is graded on, which parameters change, and which artifacts you need to reproduce the result. Most real fine-tuning bugs are bookkeeping bugs that still produce a smooth loss curve.

## What you should be able to do

- Load a model and tokenizer from the Hugging Face Hub at a pinned revision, and explain what a dataset row, a collator and a model forward each contribute to a batch.
- Render one conversation through its chat template to the exact string and token sequence, and name every special token in it.
- Distinguish the attention mask from the loss mask, build a `labels` tensor using the `-100` convention, and say which positions are graded after the internal label shift.
- Detect prompt-loss leakage and explain completion-only / assistant-only loss.
- Predict what packing and truncation do to attention, loss weighting and EOS learning.
- Write the SFT objective and its normalization choices.
- Explain full fine-tuning versus LoRA: the low-rank update, rank $r$, scaling $\alpha$, and which modules to target.
- Explain QLoRA (NF4 storage, double quantization, paged optimizers), including what memory it does **not** save.
- Save and reload an adapter together with the base revision and tokenizer, prove the reload is faithful, and compare base versus adapted models on held-out data.

## Core ideas

### The three contracts: tokenizer, dataset, model

A fine-tuning batch is produced by three components, each with its own contract.

| Component | Input | Output | Things it decides silently |
|---|---|---|---|
| Tokenizer (+ chat template) | text or a list of messages | token IDs | special tokens, BOS insertion, whitespace merges, padding side |
| Dataset `map` | one raw row | one processed row (lists of IDs, variable length) | which columns survive, truncation, filtering of empty examples |
| Collator | a list of processed rows | a batch of tensors `[B, T]` | padding length, `attention_mask`, `labels`, which tokens become `-100` |
| Model forward | `input_ids`, `attention_mask`, `labels` | `logits [B, T, V]`, scalar `loss` | the label shift, the loss reduction |

**Pin revisions.** A model name on the Hub is a moving pointer: the repository's weights, tokenizer files or chat template can change. Pin the commit hash for both the model and the tokenizer, and for the dataset:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
rev = "<40-char commit sha>"          # copy from the repo's commit history
tok = AutoTokenizer.from_pretrained("org/model", revision=rev)
model = AutoModelForCausalLM.from_pretrained("org/model", revision=rev)
# datasets.load_dataset(...) also accepts revision=
```

Record the hash in your run manifest. "Same model name" across two runs is not evidence of the same model.

**What a collator does.** After `map`, each row holds Python lists of different lengths. The collator turns a list of $B$ rows into rectangular tensors: it pads every row to a common length $T$ (the longest in the batch, or a fixed maximum), writes `attention_mask = 1` for real tokens and `0` for padding, and builds `labels` (usually a copy of `input_ids` with ignored positions replaced by `-100`). Transformers provides generic collators (see [data collators](https://huggingface.co/docs/transformers/main_classes/data_collator)); TRL's trainers ship their own. Read the one you use; it is where most masking decisions live.

**The model shifts labels for you.** For Transformers causal-LM classes, you pass `labels` aligned with `input_ids` (same positions), and the model internally compares logits at position $t$ with the label at position $t+1$. If you shift the labels yourself as well, you train the model to predict two tokens ahead. This is the same off-by-one you diagnosed in [L2](02-transformer.md); here it hides inside a library call.

| Tensor | Shape | Notes |
|---|---|---|
| `input_ids` | `[B, T]` | integer token IDs, padded |
| `attention_mask` | `[B, T]` | 1 = real token, 0 = padding |
| `labels` | `[B, T]` | token ID where graded, `-100` where ignored |
| `logits` | `[B, T, V]` | one distribution per position |
| shifted logits / labels inside the loss | `[B, T-1, V]` / `[B, T-1]` | last logit and first label are dropped |

In code this shows up as: the first thing to print in P07 is one collated batch, with `input_ids` decoded token by token next to `attention_mask` and `labels`.

### Chat templates and special tokens

An instruction-tuned model is trained on conversations serialized into a single string with role markers. A **chat template** is a Jinja program stored with the tokenizer that performs this serialization. The model only ever sees the resulting tokens, so the template is part of the model's input contract: use the template the model was trained with, and use the same one for training, evaluation and inference.

```python
messages = [
    {"role": "system", "content": "Answer in one word."},
    {"role": "user", "content": "Capital of Peru?"},
    {"role": "assistant", "content": "Lima."},
]
text = tok.apply_chat_template(messages, tokenize=False)
ids = tok.apply_chat_template(messages, tokenize=True)
```

For a template in the widely used ChatML style, `text` would look like the block below. This is an illustration of one convention, not your model's template; print your own.

```text
<|im_start|>system
Answer in one word.<|im_end|>
<|im_start|>user
Capital of Peru?<|im_end|>
<|im_start|>assistant
Lima.<|im_end|>
```

Things to check on your own rendering:

| Question | How to check |
|---|---|
| Are role markers single special tokens or split into pieces? | `tok.convert_ids_to_tokens(ids)`; a registered special token appears as one piece |
| Which token ends an assistant turn, and is it the tokenizer's `eos_token`? | compare with `tok.eos_token`, `tok.eos_token_id` and the model's generation config |
| Is BOS added twice? | many templates already emit BOS; tokenizing the rendered `text` again with default `add_special_tokens=True` can prepend a second one |
| What does `add_generation_prompt=True` add? | it appends the opening of an assistant turn so the model generates the reply; use it at inference, not on a training example that already contains the reply |
| Does the template inject a default system prompt or strip earlier reasoning? | some templates do; diff the rendered text against the messages |

If you add new special tokens, you must resize the embedding matrix and output head, and those rows are new parameters that must be trained and saved (see the adapter section).

### Attention mask versus loss mask

These are two different masks that answer two different questions.

- The **attention mask** answers: *which positions may other positions look at?* It is combined with the causal mask inside attention. Padding positions are masked so they do not influence real tokens.
- The **loss mask** answers: *which positions are graded?* In Transformers it is expressed through `labels`: any position whose label is `-100` contributes nothing. `-100` is the default `ignore_index` of `torch.nn.CrossEntropyLoss`, which is why the convention exists.

A token can be attended to but not graded (a prompt token), or, in unusual setups, graded but hidden from some other positions. Mixing up the two masks is a classic bug: a pad token with `attention_mask = 0` but a real label still contributes loss.

**Worked example.** One conversation, $B = 1$, $T = 10$. Token pieces are simplified; `[U]` and `[A]` stand for the user and assistant headers, `<eot>` for the end-of-turn token.

| position $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| token | `<bos>` | `[U]` | `Hi` | `?` | `<eot>` | `[A]` | `Hello` | `!` | `<eot>` | `<pad>` |
| `attention_mask` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| `labels` (assistant-only) | −100 | −100 | −100 | −100 | −100 | −100 | `Hello` | `!` | `<eot>` | −100 |

After the internal shift, the logits at position 5 (`[A]`) are graded against `Hello`, position 6 against `!`, and position 7 against `<eot>`. Three tokens incur loss. Two details matter:

1. The final `<eot>` of the assistant turn **is** graded. That is how the model learns to stop. If it is masked, the adapted model rambles into a new turn.
2. The header `[A]` is **not** graded: it is part of the prompt the model is given at inference time.

### Prompt-loss leakage and completion-only loss

If `labels` is simply a copy of `input_ids`, every prompt and system token is graded too. Call this **prompt-loss leakage**. Consequences:

- The model spends capacity learning to imitate user prompts and system boilerplate.
- Long prompts dominate the gradient. With a 300-token prompt and a 20-token reply, $300/320 \approx 94\%$ of the graded tokens are prompt tokens.
- Loss comparisons become meaningless: a run that grades prompts reports a different quantity from one that does not, and repeated system prompts are easy to predict, which lowers the number without improving replies.

**Completion-only loss** (for prompt/completion data) and **assistant-only loss** (for multi-turn conversations: grade every assistant turn, mask every user/system/tool turn) fix this. Both are just rules for writing `-100`. Building the mask by finding the assistant span in the token sequence is error-prone because tokenizing the prompt and the full conversation separately can merge differently at the boundary. Prefer a method that produces the mask from the same tokenization: Transformers can return an assistant-token mask from `apply_chat_template` (`return_assistant_tokens_mask=True`, which also requires `tokenize=True` and `return_dict=True`) when the template marks assistant spans with `{% generation %}` (version- and template-dependent; check that the mask is not all zeros).

> **Version note (observed in the TRL v1.16.0 docs, checked October 2026; re-check at your pinned version):** TRL's SFT trainer distinguishes language-modeling and prompt-completion datasets and exposes assistant-only loss and packing options; its default `loss_type` was `"nll"`, with `"chunked_nll"` kept as a deprecated alias, and both may differ from older tutorials. Do not assume any default. Print the collated `labels` for one batch and decode the graded positions.

The check that settles it in any version: decode `input_ids[labels != -100]` (after accounting for the shift) and read it. It should be exactly the replies plus their end tokens.

### Packing and truncation

**Truncation** cuts sequences longer than a maximum length $T_{max}$. Right truncation of a prompt/reply example removes the *end* of the reply, including its EOS, so the model is trained on replies that never stop. If the prompt alone exceeds $T_{max}$, every label becomes `-100` and the example contributes zero graded tokens; if your loss divides by the number of graded tokens in a micro-batch that contains only such examples, you get `0/0 = NaN`. Count graded tokens per example before training and filter or left-truncate the prompt deliberately.

**Packing** concatenates several short examples into one row of length $T_{max}$ to avoid wasting compute on padding. You met it in [L3](03-pretraining.md). In SFT it has two extra effects:

| Packing style | What attention sees | Positions | Risk |
|---|---|---|---|
| Naive concatenation | later examples attend to earlier unrelated ones | continue across examples | cross-example contamination; the model learns that conversations follow each other |
| Boundary-aware ("padding-free" with per-example position IDs or sequence lengths) | each example attends only within itself | restart at 0 per example | requires attention kernels that honor the boundaries; silently falls back to naive in some setups |

Packing also changes **loss weighting**. If loss is a mean over all graded tokens in the batch, a long reply gets more total weight than a short one, whether or not you pack; if it is a mean per sequence and then over sequences, every example counts equally. Packing makes "sequence" ambiguous, so state which normalization you use. The same question returns with gradient accumulation and multiple ranks in [L9](09-distributed.md).

### The SFT objective

Let a training example be a prompt $x$ and target reply $y = (y_1, \dots, y_n)$. The model with parameters $\theta$ defines $\pi_\theta(y_t \mid x, y_{<t})$. Let $m_t \in \{0, 1\}$ be the loss mask. Token-mean SFT over a batch $\mathcal{B}$ is

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{\sum_{(x,y)\in\mathcal{B}} \sum_t m_t} \sum_{(x,y)\in\mathcal{B}} \sum_t m_t \log \pi_\theta(y_t \mid x, y_{<t}).
$$

The sequence-mean alternative divides each example's sum by its own $\sum_t m_t$ and then averages over examples. Both are maximum likelihood on the masked tokens; they weight examples differently. This is the same cross-entropy as pretraining. SFT differs only in the data (curated demonstrations in a template) and the mask.

**Tiny numeric example.** Two examples in a batch. Example 1 has graded-token log-probs $(-0.2, -1.0)$; example 2 has $(-0.5, -0.1, -0.3, -0.6)$. Token mean: $-(-0.2-1.0-0.5-0.1-0.3-0.6)/6 = 2.7/6 = 0.45$. Sequence mean: $\tfrac12(1.2/2 + 1.5/4) = \tfrac12(0.6 + 0.375) = 0.4875$. Same predictions, different number. Report which one you use.

### Full fine-tuning versus LoRA

**Full fine-tuning** updates every weight. With AdamW in mixed precision, a common accounting is about 16 bytes per parameter before activations: 2 (bf16 weights) + 2 (bf16 gradients) + 4 (fp32 master weights) + 8 (two fp32 Adam moments). A 1B-parameter model therefore needs roughly 16 GB just for state. You derive budgets like this properly in [L8](08-efficiency.md).

**LoRA** (low-rank adaptation) freezes the pretrained weight $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ and learns a low-rank correction:

$$
h = W_0 x + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times d_{in}},\; B \in \mathbb{R}^{d_{out} \times r},\; r \ll \min(d_{in}, d_{out}).
$$

- $r$ is the **rank**: the update $\Delta W = \tfrac{\alpha}{r} BA$ has rank at most $r$.
- $\alpha$ is a **scaling** hyperparameter; the factor $\alpha/r$ keeps the update's scale from changing too much when you change $r$. (A variant, rsLoRA, uses $\alpha/\sqrt{r}$.)
- $B$ is initialized to zero and $A$ randomly, so $\Delta W = 0$ at step 0 and the adapted model starts exactly at the base model. That is a free test: before any training step, adapted logits must equal base logits.
- Gradients flow to $A$ and $B$ only. Optimizer state exists only for them.

**Parameter count.** For a square projection with $d_{in} = d_{out} = 2048$ and $r = 16$: LoRA adds $r(d_{in} + d_{out}) = 16 \cdot 4096 = 65{,}536$ parameters versus $2048^2 = 4{,}194{,}304$ in $W_0$, about 1.6%.

**Forward example.** $d_{in} = 3$, $d_{out} = 2$, $r = 1$, $\alpha = 2$ (so $\alpha/r = 2$):

$$
W_0 = \begin{pmatrix}1 & 0 & 2\\ 0 & 1 & -1\end{pmatrix},\;
A = \begin{pmatrix}1 & 1 & 0\end{pmatrix},\;
B = \begin{pmatrix}0.5\\ -1\end{pmatrix},\;
x = \begin{pmatrix}2\\ -1\\ 3\end{pmatrix}.
$$

$W_0 x = (8, -4)$. $Ax = 1$. $BAx = (0.5, -1)$, scaled by 2 gives $(1, -2)$. Output $h = (9, -6)$. With $B = 0$ (initialization), $h = (8, -4)$, the base output.

| Tensor | Shape |
|---|---|
| input activations $x$ | `[B, T, d_in]` |
| $A$ | `[r, d_in]` |
| $Ax$ (low-rank bottleneck) | `[B, T, r]` |
| $B$ | `[d_out, r]` |
| update $\tfrac{\alpha}{r}BAx$ and base $W_0x$ | `[B, T, d_out]` |

**Merging.** After training you can fold the update into the weight, $W = W_0 + \tfrac{\alpha}{r}BA$, which removes inference overhead but produces a full-size checkpoint. Keeping the adapter separate lets you swap adapters on one base.

**Which modules.** In a decoder block the candidates are the attention projections ($W_Q, W_K, W_V, W_O$) and the MLP projections (gate, up, down). The original LoRA experiments focused on attention projections; the QLoRA authors reported that adapting all linear layers was needed to match full fine-tuning in their setting. Embeddings and the output head are usually left frozen unless you added tokens, in which case those rows must be trained and saved in full. Module names differ between architectures; list them with `model.named_modules()` before writing a target list.

**What LoRA does not save.** Activations still have to be stored for the backward pass, because gradients must flow *through* frozen layers to reach adapters in earlier layers. At long sequence lengths activations, not weights, can dominate. Gradient checkpointing (recomputation, [L8](08-efficiency.md)) is the tool for that.

### QLoRA: 4-bit base, higher-precision adapters

QLoRA stores the frozen base weights in 4 bits, dequantizes them on the fly to a compute dtype (for example bf16) for each matrix multiply, and trains LoRA adapters in higher precision on top. Three pieces:

**NF4 (4-bit NormalFloat).** Weights are split into small blocks (64 values in the paper). Each block is scaled by its absolute maximum into $[-1, 1]$, and each value is replaced by the nearest of 16 code levels. The NF4 levels are placed at quantiles of a standard normal distribution, on the assumption that pretrained weights are roughly normally distributed, so levels are dense near zero where most weights live. Toy illustration with a made-up 5-level code $\{-1, -0.4, 0, 0.4, 1\}$ (not the real NF4 table): the block $(0.12, -0.30, 0.05, 0.60)$ has absmax $0.60$; normalized it is $(0.2, -0.5, 0.083, 1.0)$; snapped it is $(0, -0.4, 0, 1)$; dequantized it is $(0, -0.24, 0, 0.60)$. The error on each weight is the price of storage.

**Double quantization.** Each block also needs its scale constant. At one fp32 constant per 64 weights that is $32/64 = 0.5$ extra bits per parameter. Quantizing those constants to 8 bits, in second-level blocks of 256 with their own fp32 constants, costs $8/64 + 32/(64 \cdot 256) \approx 0.127$ bits per parameter.

**Paged optimizers.** Optimizer state for the adapters is placed in memory that can be paged between GPU and CPU (NVIDIA unified memory), so a transient spike, such as one unusually long batch, does not crash the run with an out-of-memory error. It handles spikes; it does not shrink steady-state memory.

**Memory accounting, 1B-parameter base (rough, before activations):**

| Item | bf16 LoRA | QLoRA |
|---|---|---|
| frozen base weights | $\approx 2$ GB | $\approx 0.5$ GB + $\approx 0.016$ GB constants |
| base gradients / optimizer state | none | none |
| adapters, their gradients and AdamW state (say 10M params) | $\approx 0.16$ GB | $\approx 0.16$ GB |
| activations | unchanged | unchanged |

What QLoRA does **not** save: activations, adapter optimizer state, attention memory at long $T$, and the time cost of dequantizing every forward and backward (expect it to be slower per step). It also introduces numerical issues: the base you train against is the quantized base, so merging an adapter into the full-precision base, or evaluating it on a differently quantized base, is a different model. Evaluate in the configuration you will deploy.

> **Version note (observed in the PEFT docs checked September 2026; re-check at your pinned version):** PEFT documents a preparation step for training on a quantized base and a shortcut to target all linear layers QLoRA-style. Quantized training requires a GPU backend (commonly bitsandbytes); keep the QLoRA criteria pending until suitable hardware exists, as the [roadmap](../course/ROADMAP.md#required-versus-optional-depth) says.

### Saving and reloading an adapter

A PEFT `save_pretrained` writes the adapter weights and an `adapter_config.json` (rank, $\alpha$, target modules, base model name). The adapter is useless without the exact base it was trained on: a different base revision has different $W_0$, and $\Delta W$ is a correction to that particular $W_0$. Save alongside every adapter:

- base model ID **and commit hash** (record it yourself even if your PEFT version also stores a revision field);
- the tokenizer, including the chat template, and any added tokens and resized embedding/output rows;
- quantization config if trained with QLoRA;
- training config and data revision.

A faithful-reload check is a learner experiment, not a library guarantee: run fixed inputs through the trained model, save, reload the base at the pinned revision plus the adapter in a fresh process, and compare logits within a dtype-appropriate tolerance. Also check that the reloaded adapted logits *differ* from the base logits; an adapter that silently failed to attach passes the first check trivially if you compare against the wrong thing.

### Held-out comparison of base versus adapted

Evaluation follows [L4](04-evaluation.md). For an adapter:

- **Same inputs, same template, same decoding.** Fixed held-out prompts, rendered by the same template, with fixed decoding (greedy, or sampling with fixed seeds and settings). The base model may not know your template; record that as part of the comparison.
- **Held-out loss on graded tokens only**, same mask and normalization for both models.
- **Task metric and samples.** A loss drop on the target format does not prove better answers. Read samples side by side and score them with a pre-declared rule.
- **Regression check.** Run a small set of unrelated prompts to see whether the adapter damaged general behavior (forgetting).
- **Leakage check.** Confirm the held-out prompts do not appear in training data, including near-duplicates.

## Common failures

| Symptom | Likely cause | How to check |
|---|---|---|
| Adapted model never stops, or starts a fake user turn | end-of-turn/EOS masked from loss, or truncated away; or pad token equals EOS and the collator masked every pad-ID position (Transformers' `DataCollatorForLanguageModeling` with `mlm=False` sets every label equal to `pad_token_id` to `-100`) | decode graded positions; count EOS among graded labels; compare `pad_token_id` with `eos_token_id` |
| Loss is suspiciously low from step 0 and drops fast | prompt-loss leakage on long repetitive system prompts | ratio of graded prompt tokens to reply tokens; decode `labels != -100` |
| Model learns to predict two tokens ahead / garbled outputs | labels shifted manually and again inside the model | compare a hand-computed loss on one example with the model's `loss` |
| Train/inference mismatch: good loss, bad chat | different template, missing generation prompt, or double BOS at inference | render the inference prompt with the same template and diff the token IDs |
| `NaN` loss on some steps | micro-batch with zero graded tokens | count graded tokens per micro-batch |
| Packed run better than unpacked by a surprising margin | cross-example attention, or different loss weighting | inspect attention boundaries and position IDs; recompute both runs with the same normalization |
| Adapted logits identical to base after training | adapter not attached, wrong target module names, or $B$ stayed zero (learning rate or frozen flag) | print trainable parameter count; check adapter weight norms |
| Reloaded adapter gives different outputs | different base revision, quantization or tokenizer | compare pinned hashes; rerun the logit-equality check |
| QLoRA run still out of memory | activations at long $T$, not weights | measure peak memory versus sequence length; try gradient checkpointing |

## Check yourself

1. Render one multi-turn conversation from your dataset through the model's template. Write down each special token, whether it is graded, and why.
2. Predict, then run: tokenize the rendered template string once with `add_special_tokens=True` and once with `False`. Which BOS behavior does your model's template expect?
3. For one collated batch, decode the graded positions after the shift. Does the set match "all assistant replies and their end tokens" exactly?
4. Predict, then run: copy `input_ids` into `labels` without masking. How does step-0 loss change, and why does that not mean the model is better?
5. Choose $T_{max}$ so that one example's reply is cut in half. What happens to that example's EOS and its contribution to the loss?
6. Explain, with your own two-example batch, when token-mean and sequence-mean SFT losses differ and which examples each favors.
7. Before training, verify that the LoRA-wrapped model's logits equal the base model's. Which initialization property makes this true?
8. For your model, compute the LoRA parameter count for your target modules at $r = 8$ and $r = 32$. Then measure peak memory at two sequence lengths. Which term grew?
9. Explain why QLoRA cannot reduce activation memory, using the backward pass through a frozen layer.
10. Save an adapter, reload it in a fresh process, and show logit equality with the trained model and inequality with the base. List every artifact you needed.
11. Design a base-versus-adapted comparison on 30 held-out prompts. Which result would convince you the adapter helped, and which would only show it learned the format?

## Going deeper

| Source | Read | Why |
|---|---|---|
| [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating) | template rendering, generation prompts, assistant masks | exact serialized input contract |
| [Transformers data collators](https://huggingface.co/docs/transformers/main_classes/data_collator) | causal-LM collator behavior | how padding and labels are written |
| [Datasets processing](https://huggingface.co/docs/datasets/process) | `map`, `filter`, column removal | per-row transforms and provenance |
| [TRL SFT trainer](https://huggingface.co/docs/trl/sft_trainer) | dataset formats, completion/assistant-only loss, packing | version-specific trainer behavior; read the pinned release |
| [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) | §4 method (4.1 low-rank-parametrized update matrices, 4.2 applying LoRA to Transformer); §7 understanding the low-rank updates | the update rule, initialization, rank and target-module choices |
| [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) | §2 background (block-wise quantization, memory of parameter-efficient fine-tuning); §3 QLoRA fine-tuning (NF4, double quantization, paged optimizers) | what 4-bit storage saves and costs |
| [PEFT docs](https://huggingface.co/docs/peft/index) and [quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) | LoRA config, save/load, quantized preparation | library mechanics at your pinned version |
| [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) | §3.5 Models (the supervised fine-tuning stage); Appendix C.1 for details | where SFT sits before preference learning ([L6](06-preferences.md)) |
| [Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1) | fine-tuning chapters (optional) | an alternative walk-through of the same APIs |
