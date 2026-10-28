# Linear attention → Kimi KDA → Kimi K3

Selected October 7, 2026 for L10/P14. This is a future workshop brief, not an active assignment or completed learner artifact. Begin from the learner's causal-attention code (their private `work/transformer/attention.py`) and KV-cache evidence. Review that attempt before preparing the next component's interfaces and TODOs; the learner writes assessed implementations.

## Build sequence and checks

| Component | Small observable implementation goal | Evidence to inspect | Estimated hours |
|---|---|---|---:|
| Linear attention | With tiny fixed Q/K/V tensors, implement a causal feature-map reference and its recurrent equivalent, including normalization | Matching outputs for the same attention rule; future-token isolation; state shapes and bytes across sequence lengths | 3 |
| Kimi Delta Attention (KDA) | Implement one head's recurrent update with supplied normalized Q/K, V and gates; trace a short write/read sequence | A hand-checkable state trace, gate/write-strength variations and consistent state continuation across sequence segments | 3 |
| Kimi K3 report | Connect the implemented recurrence to one narrowly scoped claim in §2.1 and its supporting evidence | A learner-authored claim ledger with exact source locations and a proposed discriminating ablation | 1 |
| Setup, review and repair | Review current code, pin runtime, repair a consequential failure and perform a fresh variation | Correctness, understanding, transfer and limits under the course rubric | 2 |

Nine hours specialize the existing P14 budget inside L10, replacing its unspecified mechanism example. At three hours/week that is about three weeks; the learner picks the actual pace and dates at activation, when an execution owner is created. The broader K3 study below uses the rest of the existing L10 allocation and matching module work, counted once.

## Mechanisms to learn through the builds

For linear attention, use one explicit positive feature map consistently in the quadratic reference and recurrent path. Inspect both the matrix memory and normalization accumulator. Output equivalence is required between those two implementations, not with softmax attention. Compare against softmax to investigate behavior, and separate dependence on sequence length from dependence on feature width. The primary foundation is [Transformers are RNNs, ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html), especially §3.

For KDA, use [Kimi Linear v2, §2.2–3](https://arxiv.org/html/2510.26692v2): additive memory, delta-rule correction, then channel-wise retention. The tiny artifact accepts projected inputs and gates; it is a recurrence study, not a full production layer. Use [K3 v2, §2.1.1 equation (1)](https://arxiv.org/html/2607.24653v2#S2.SS1.SSS1) as a cross-check, aligning state orientation and update order before comparing. Distinguish the memory-retention gate from other gates in a complete layer. Full projections, convolution, output processing and chunkwise kernels are later scope.

Compare recurrent-state storage with the growing KV cache using actual tensor element counts and explicit dimensions. Separate decode state from training activations. Inspect the hybrid layout in [Kimi Linear §4](https://arxiv.org/html/2510.26692v2#S4) and [K3 §2.1](https://arxiv.org/html/2607.24653v2#S2.SS1): a fixed-size KDA state does not make every layer's cache constant-sized. CPU demonstrations support mechanism and storage reasoning; hardware speed claims require separate measurements.

## K3 report thread

Use [Kimi K3 v2](https://arxiv.org/html/2607.24653v2) alongside the corresponding artifact rather than reading the entire report before coding.

| Course point | Source route | Learner artifact or question |
|---|---|---|
| L2/L3 first pass and P14 | §2 architecture, focusing on §2.1 attention | Map the toy recurrence to the report; identify which architectural parts the toy omits |
| L3/L4 | §3 pretraining and scaling | Identify the comparison conditions and what separates a mechanism result from a training-recipe result |
| L5–L7 | §4 post-training | Trace one objective/data/rollout choice to the matching course implementation |
| L8/L9 | §5 infrastructure | Explain one systems claim with a resource model and specify measurements needed to test it |
| L4/L10 defense | §6 evaluation plus the relevant methods section | Audit controls and evidence; revise the original claim ledger |

In P14 investigate one attention claim deeply. The other rows are later revisits within existing modules; they do not require additional implementations of every K3 component.

Create the claim ledger when the project starts, with these fields: claim in the learner's words; paper/version and section/equation/table; author-reported evidence; baseline and comparison conditions; what the toy tests; what it cannot establish; missing control or ablation; proposed test; conclusion and remaining uncertainty. Do not fill in results before conducting the investigation.

## Assessment and activation

Use [the assessment rubric](../../course/ASSESSMENT.md). Inspect reference agreement, causality and state continuation; ask the learner to trace shapes and explain memory/update choices. Select an unfamiliar modification at review. Substantially coached work needs a fresh independent variation. These artifacts can demonstrate selected L10 and resource-reasoning criteria; broader report coverage and the L7–L9-dependent final defense remain pending.

Before activation, inspect the learner's tracker and prerequisite code, choose dates, record the nine-hour allocation and pin Python/PyTorch and any inspected code commit. Use tiny synthetic CPU tensors with bounded sequence lengths; setup and failed attempts consume the budget. No model weights, GPU allocation or long training run is needed. If the budget is insufficient, split the deliverable before assigning dates.

Primary references checked October 7, 2026: the ICML proceedings paper, Kimi Linear arXiv:2510.26692v2, Kimi K3 arXiv:2607.24653v2 and the [official Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear). The repository is an optional implementation route, not a pinned or installed dependency.
