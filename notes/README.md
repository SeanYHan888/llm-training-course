# Study notes

One note per module, written for this course. Each note explains its module's concepts with definitions, the math, small worked examples, shape tables, common failures and self-check questions. Each ends with a **Going deeper** table pointing to exact sections of primary sources.

The notes support the builds; they are not a reading assignment. When your code needs a concept, jump to that section, then go back to the code. The notes contain no exercise solutions, and the self-check questions have no answer key: answer them against your own code and discuss them in review.

## Modules

| Note | Covers | Supports |
|---|---|---|
| [L0 — Foundations](00-foundations.md) | Tensors, shapes and broadcasting; probability and log-space; chain rule; linear algebra for attention; generalization | Diagnostic, P01 |
| [L1 — Neural learning](01-neural-learning.md) | Logits, softmax, cross-entropy; autograd; gradient checks; SGD/AdamW; schedules; regularization | P01 |
| [L2 — Transformer language model](02-transformer.md) | BPE tokenization; embeddings; causal attention and heads; residuals and norms; RoPE; gated MLPs; shifted labels; sampling; KV cache | [Transformer workshop](../exercises/transformer/README.md), P02–P04 |
| [L3 — Pretraining](03-pretraining.md) | Data provenance, filtering, dedup; splits; packing; the training loop; clipping and schedules; checkpoints and resume | P05–P06 |
| [L4 — Evaluation](04-evaluation.md) | Loss, perplexity, bits-per-byte; downstream metrics; uncertainty; judge bias; contamination; scaling laws; ablations | P06 |
| [L5 — SFT and PEFT](05-sft-peft.md) | Chat templates; attention vs loss masks; SFT; LoRA; QLoRA; adapter save/reload | P07–P08 |
| [L6 — Preference learning](06-preferences.md) | Bradley–Terry reward models; KL-regularized objective; DPO derivation and assumptions | P09 |
| [L7 — Online RL](07-online-rl.md) | Policy gradients; advantages; PPO; GRPO; KL; verifiable rewards; reward hacking | P10–P11 |
| [L8 — Efficiency](08-efficiency.md) | FLOPs; arithmetic intensity; memory budgets; precision formats; accumulation; recomputation; profiling; FlashAttention | P12 |
| [L9 — Distributed training](09-distributed.md) | Collectives; DDP; loss normalization across ranks; ZeRO/FSDP; tensor, pipeline and expert parallelism; recovery | P13 |
| [L10 — Frontier architectures](10-frontier.md) | MoE; linear attention; delta-rule memory and KDA; hybrids; long context; reading a frontier report | [Attention alternatives](../exercises/attention-alternatives/README.md), P14 |
| [L11 — Research practice](11-research.md) | Bounded questions; baselines; frozen protocols; run manifests; claim ledgers; defending a result | P15a–P15c |
| [Inference and serving](inference-serving.md) (elective) | Prefill/decode; KV memory; PagedAttention; prefix reuse; batching; serving metrics | [Inference 01](../exercises/inference/README.md) |
| [Interpretability](interpretability.md) (elective) | Residual stream; features and superposition; SAEs; lenses; J-lens; causal interventions | [Interpretability 01](../exercises/interpretability/README.md) |

## Suggested order

Follow the build, not the numbering. The current route is L2 attention → KV cache ([inference](inference-serving.md)) → multi-head attention → tiny decoder → training. Pull in [L0](00-foundations.md) and [L1](01-neural-learning.md) sections as each operation needs them. After that the main line is L3 → L4 → L5 → L6 → L7 → L11. L8 and L9 branch off after L3, and L10 is read alongside the later modules. The [roadmap](../course/ROADMAP.md#route-and-early-architecture-interest) lists the prerequisites.

Library defaults and APIs change between releases. Where a note describes TRL, PEFT or Transformers behavior, check it against the version pinned for your lab.
