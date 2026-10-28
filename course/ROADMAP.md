# Model-training roadmap

Version 0.4 — explicit linear attention/KDA route, revised October 7, 2026; initial design September 28. Milestone dates are chosen by the learner. Hour estimates include reading, small implementations, source tracing and assessment, but exclude optional full university assignments, long unattended compute and major prerequisite remediation. Shared work is counted once. See the [syllabus](SYLLABUS.md) for module boundaries and materials, and the [schedule](SCHEDULE.md) for cadence and forecasts.

## Milestones

| ID | Learning destination and required topics | Demonstration and unfamiliar variation | Entry | Hours |
|---|---|---|---|---|
| L0 | Establish a baseline: Python, tensor shapes, probability, calculus/chain rule, linear algebra, generalization and any relevant prior research work | Trace a small computation in learner code; use initial implementation attempts to map strengths and gaps | None | 8–16 |
| L1 | Learn a neural model: parameters, linear layers/MLPs, activations, logits, softmax/log-softmax, NLL/cross-entropy, autograd, minibatches, SGD/AdamW, learning-rate schedules, regularization | Learner-written small training loop with a numerical gradient check; diagnose absent/accumulated gradients and train/validation divergence | L0 gaps addressed | 20–35 |
| L2 | Build the language-model computation: tokenization/byte-level BPE, embeddings, causal attention, Q/K/V and heads, residuals, normalization, RoPE, gated MLPs, output head, shifted labels, generation and KV cache | Trace shapes and masks through a tiny decoder; validate tokenizer round trip and causality; diagnose an off-by-one label or mask bug | Relevant tensor/mask foundations for initial construction; L1 criteria before training/mastery | 25–40 |
| L3 | Pretrain and resume: corpus provenance, filtering, deduplication, train/dev/test split, batching/packing, AdamW, clipping, warmup/decay, seeds, checkpoints and restart | Train a tiny model with held-out loss; overfit a tiny batch; restart and explain what state is restored; investigate leakage or a broken schedule | L2 | 25–40 |
| L4 | Design valid experiments: baselines, token-normalized loss/perplexity, downstream metrics, uncertainty, paired comparisons, length/judge bias, contamination and scaling-law assumptions | Produce a small evaluation protocol and one controlled ablation; catch a misleading comparison or contaminated split | L3; habits start in L1 | 20–30 |
| L5 | Adapt an existing model: Transformers/Datasets contracts, model/tokenizer revisions, chat templates, attention versus loss masks, SFT, LoRA/QLoRA, packing and truncation | Trace one example into the loss, compare base and adapted checkpoints on held-out data, and save/reload an adapter with its base revision; catch prompt-loss leakage | L3–L4 | 20–35 |
| L6 | Understand preference learning: pairwise reward models, Bradley–Terry likelihood, KL-regularized policy objective, DPO derivation/assumptions, reference model, beta, sum versus mean log-probability and pair quality | Explain and independently implement a toy DPO objective, compare it with a pinned TRL source path, diagnose reversed pairs/masks, and trace an existing research claim to evidence | L5 | 20–30 |
| L7 | Learn online post-training: policy gradients, returns/advantages, baselines, importance ratios, PPO clipping/value model, GRPO groups, KL, sampling, reward design and verifiable rewards | Hand-trace one update; audit a verifier; run a bounded GRPO-style experiment when hardware permits; diagnose zero-variance groups, reward hacking or truncation | L6; probability diagnostic | 20–35 |
| L8 | Explain training efficiency: FLOPs, arithmetic intensity, memory hierarchy, weights/gradients/optimizer/activations, bf16/fp16, accumulation, recomputation, profiling, efficient attention and Triton orientation | Derive a memory budget with assumptions; compare predicted and measured bottlenecks; preserve correctness across one optimization | L3; may precede L5 if memory requires | 20–35 |
| L9 | Reason about distributed training: DDP, gradient synchronization and loss normalization, sharding/FSDP/ZeRO, tensor/pipeline/expert parallelism, communication, checkpointing and recovery | Compare a tiny multiprocess update with a single-process baseline; explain uneven-token weighting and restart failure; distinguish CPU demonstration from GPU scaling | L8 | 20–35 |
| L10 | Read modern frontier reports: linear attention, delta-rule memory and Kimi Delta Attention (KDA); dense versus MoE, active versus total parameters, long-context training, multimodal objectives, data/optimizer choices and RL systems | Implement tiny linear-attention and KDA recurrence comparisons; use them in a Kimi K3 claim ledger, identify missing ablations and what would be needed to reproduce a claim | L2/L3 first pass; L7–L9 final defense | 15–25 |
| L11 | Defend a reproducible training investigation | Deliver a bounded baseline/treatment comparison with frozen evaluation, resource log, failure analysis and independent explanation; connect the outcome back to a preference-learning or K3 question | L4–L10 relevant criteria | 20–35 |

The nominal sum is **233–391 focused core hours** before diagnostic credits and extra remediation. Divide by your chosen weekly pace for a duration: about **30–49 weeks** at 8 core hours/week, **24–40 weeks** at 10. These are forecasts, not agreed dates. Re-estimate from actual work; see the [schedule](SCHEDULE.md#duration-forecast).

## Route and early architecture interest

Main route: L0 → L1 → L2 → L3 → L4 → L5 → L6 → L7 → L11. L8/L9 branch from L3 and feed the advanced work. L10 is a recurring reading thread, with its final defense after the relevant foundations.

Build-first route: single-head attention → KV reuse/memory → multi-head attention → tiny decoder → training/generation, using the [workshop](../exercises/transformer/README.md). Weave L0/L1 foundations into the operations that need them. The route above expresses mastery dependencies, not a mandatory lecture order. Fixed-weight attention can start before the full L1 milestone; the training step requires its relevant gradient/update capability. Complete remaining tokenizer, RoPE, gated-MLP and L1 criteria through bounded practical work before declaring L2 mastery.

Selected L10 continuation: softmax attention/KV evidence → [linear attention → Kimi KDA → Kimi K3 claim investigation](../exercises/attention-alternatives/README.md). P14 specializes the existing mechanism investigation in nine estimated hours; the remaining L10 time covers broader report connections and final defense. Reuse earlier evidence once. This is future curriculum, with activation after readiness and the current-window review.

Prior research work (for example, a preference-learning paper) may satisfy parts of L4/L6/L11 when supported by an artifact, your explanation and a fresh variation. Do not infer skill gaps solely from low confidence or grant credit solely from authorship.

## Required versus optional depth

Required: a tiny model built from components, a reproducible training loop, small SFT and preference-learning experiments, reward/online-RL understanding with a bounded practical route, resource accounting, a distributed-correctness demonstration, and a defended research comparison.

Hardware-dependent: GPU timing, fused kernels, quantized training and realistic rollout throughput. Keep these criteria pending until the environment supports them; a CPU substitute can demonstrate the objective or communication semantics, but cannot demonstrate GPU performance. Multihost and frontier-scale training are optional.

Optional specializations: full FlashAttention kernel implementation; full CS336 assignments; Megatron-scale deployment; new MoE training recipes; VLM training; diffusion/audio/video training; distillation projects; continual learning; mechanistic interpretability. Capture them in idea intake rather than silently expanding the core.

## Course overlap

When a learner takes a related course (for example on agent systems or serving), share tensor/attention/inference evidence where criteria genuinely match. This course owns gradients, data, objectives, optimizer state and training evaluation; agent orchestration, production services and application reliability belong elsewhere. Count shared work once.
