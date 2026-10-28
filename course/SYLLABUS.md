# Course syllabus and module boundaries

This is a personal course with graduate-level expectations for mathematical reasoning, implementation, experimental evaluation and defense. The [charter](CHARTER.md) defines authority and workload; the [roadmap](ROADMAP.md) defines the required outcomes, prerequisite graph and stable L0–L11 milestone IDs. Each row below describes the instruction supporting its corresponding milestone. A module can require several bounded projects.

## Modules

Each module has an in-repo [study note](../notes/README.md) that explains its concepts. Materials below are selected from the existing [source guide](../references/README.md). They are on-demand references for implementation and evidence review, not a preparation checklist. Select exact sections and revisions when the current task needs them; this table is not a fresh verification of external sources.

| Milestone | Capability developed through projects | On-demand references | Project evidence |
|---|---|---|---|
| L0 — Readiness | Diagnose tensor, probability, calculus and training-path understanding; credit demonstrated prior knowledge. Remediate identified gaps rather than assign a general math curriculum. | Selected PyTorch basics; Session 01 as an optional diagnostic reference; optionally one learner-selected prior research artifact | Initial code attempts and applicable part of P01 |
| L1 — Learning mechanism | Trace forward, objective, backward and optimizer update; numerical checks and optimization failures. Transformer architecture belongs to L2. | PyTorch basics/autograd; instructor-selected tiny mathematical examples | P01: learner-written training loop and gradient checks |
| L2 — Language-model computation | Tokenization and a tiny causal decoder, shapes, masks and generation. Advanced architecture comparison returns in L10. | CS336 architecture/basics selections; Transformer paper; selected HF LLM course sections | P02–P04: tokenizer, attention, decoder/one-batch checks |
| L3 — Pretraining | Data provenance, splits, training, checkpoint state and recovery. Distributed performance belongs to L8/L9. | CS336 data/optimization selections; Datasets processing; selected checkpoint source paths | P05; training/data portion of P06 |
| L4 — Experimental judgment | Evaluation protocols, contamination, uncertainty, ablation and scaling assumptions. Habits begin in L1; their formal defense happens here. | CS336 evaluation/scaling selections; compute-optimal training paper; evaluation-harness configuration | Evaluation portion of P06; frozen protocol and controlled comparison |
| L5 — Supervised adaptation | Trace templates, masks, SFT and adapters; compare held-out behavior and reload artifacts. Preference objectives belong to L6. | Chat-template and TRL SFT references; LoRA/QLoRA mechanism selections; pinned PEFT source when activating | P07–P08 |
| L6 — Preferences | Reward models, DPO derivation/assumptions, implementation and data quality. Online rollout learning belongs to L7. | DPO paper; pinned TRL preference-training path; learner-selected prior research evidence | P09 |
| L7 — Online learning | Policy gradients, PPO/GRPO, rewards, verifiers and bounded online experiments. Frontier rollout infrastructure is optional. | CS336 alignment/reasoning selections; instructor-selected primary derivation material; pinned TRL online-training path | P10–P11; hardware-dependent criteria retain their limits |
| L8 — Efficiency | Memory/FLOP accounting, profiling, precision and correctness-preserving optimization. Full custom-kernel implementation is optional. | CS336 systems selections; relevant PyTorch performance references selected at activation | P12; distinguish measured GPU results from CPU illustrations |
| L9 — Distribution | Synchronization, normalization, sharding, parallelism and restart correctness. Production multihost operation is optional. | CS336 distributed-training selections; pinned PyTorch/Accelerate/FSDP/DeepSpeed references as applicable | P13; equivalence and failure explanation |
| L10 — Frontier reports | Build tiny linear attention and Kimi KDA recurrences; compare memory and audit K3 claims, including MoE, long context and multimodal objectives. | [Attention alternatives workshop](../exercises/attention-alternatives/README.md); linear-attention paper, Kimi Linear v2 and Kimi K3 v2 | P14: recurrence checks and K3 claim ledger; final defense after relevant L7–L9 criteria |
| L11 — Research defense | Integrate learned methods into one bounded question, baseline/treatment and reproducible conclusion. Publication is a separate decision. | Relevant methods and evaluation references from the chosen question | P15a–P15c |

The [project catalog](PROJECTS.md) owns deliverables and effort estimates. Its projects contribute evidence; finishing a project does not automatically pass every associated milestone. Assessment also covers required criteria absent from that project's artifact.

## Workshop lesson contract

Before a work block, specify:

1. The current file/component, one observable deliverable and the milestone criteria it can demonstrate.
2. The smallest coding action, allowed scaffolding/assistance and time/compute limits. Identify prerequisites through the attempt and teach missing pieces in context.
3. A concrete check: tensor shapes, reference equivalence, a controlled input change, a loss trace or a failure to repair.
4. Any optional source excerpt/video that addresses a current gap. Paper-specific investigations consult the exact figure or claim needed for the artifact; no blanket pre-reading or preparation-question requirement.
5. The learner's tracker entry, available hours and the evidence to record. Credit matching earlier work and retain unfinished criteria.

Expand only the current build step into code TODOs. Later steps remain capability-level until the learner is ready. [Session 01](../exercises/diagnostic/README.md) is a retained reference, not the next mandatory session; resume from current evidence and [the Transformer workspace](../exercises/transformer/README.md).

## Early project route and credit

| Build | Learn while doing | Completion evidence / remaining scope |
|---|---|---|
| Single-head attention | Matrix products, Q/K/V, softmax, scaling, causal masking | Shapes, numerical reference and causality; shared inference/P03 evidence |
| KV reuse and memory | Prefill/decode, position tracking, cache layout and byte counts | Same-token cached/full-prefix comparison and memory measurements; replaces separate Lab A/B instruction |
| Multi-head attention | Head dimensions, reshape/transpose, concatenation and output projection | One-head equivalence and independent per-head reference; completes P03 criteria when demonstrated |
| Tiny decoder | Embeddings, residuals, normalization, MLP and logits | Causal forward path; initial P04 uses learned absolute positions and a simple MLP |
| Learn and generate | Shifted labels, cross-entropy, autograd, optimizer and token selection | One-batch overfit and generation; numerical gradients and other missing L1 criteria get targeted practical work |
| Complete L2 coverage | Byte-level BPE, RoPE, gated MLPs and remaining architecture checks | P02 plus separately bounded extensions; a minimal decoder alone does not pass all L2 |

This route can weave L0/L1 foundations into L2 construction. Resolve the prerequisite for the next operation rather than requiring the whole of L1 before a fixed-weight attention experiment. Whole-milestone mastery requirements remain in the roadmap. Further projects continue through training/resume, evaluation, adaptation, preference learning, RL, efficiency, distribution and research.

## Scope of student choices

Student ideas can change the example, dataset, project or elective investigation through [idea intake](IDEAS.md). Required capabilities and prerequisites remain visible. For example, a chatbot fine-tuning experiment may supply L5 evidence; its billing and authentication work belongs in another project. A relevant topic with missing prerequisites can have an early bounded overview and a later technical investigation.

## Selected exploration — internal representations

[Interpretability 01](../exercises/interpretability/README.md) is a two-hour conceptual companion to [Inference 01](../exercises/inference/README.md): neurons and distributed features, SAE latents, J-space/J-lens readouts, and causal-intervention limits. It connects L1/L2 concepts with L4 experimental judgment; full mechanistic interpretability remains optional. Exact tasks and dates belong to the learner's tracker.
