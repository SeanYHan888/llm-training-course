# Bounded project options

These are proposed slices, not active tasks. Each produces a useful artifact and is sized to finish in about three weeks at the learner's chosen pace. Estimates are focused learner hours and are already included in milestone forecasts; divide by your weekly hours to plan dates. Revise scope if actual work exceeds the estimate.

The [schedule](SCHEDULE.md) budgets project work alongside reading and assessment. Use the hours the learner gives the project when choosing its window. Student ideas may substitute a deliverable through [idea intake](IDEAS.md) only when the same required criteria and assessment standard remain covered. A module may span several execution projects; a project can contribute to several milestones.

| Slice | Milestone | Deliverable | Initial hours |
|---|---|---|---|
| P01 | L0/L1 | Shape/gradient diagnostic and one small learner-written training loop | 6–9 |
| P02 | L2 | Tokenizer round-trip and segmentation investigation | 6–9 |
| P03 | L2 | Tiny causal-attention computation with shape/mask tests | 6–9 |
| P04 | L2/L3 | Integrate understood components into a tiny decoder and overfit one batch | 6–9 |
| P05 | L3 | Reproducible small-corpus run and checkpoint-resume comparison | 6–9 |
| P06 | L3/L4 | Data-quality or leakage audit with one controlled comparison | 6–9 |
| P07 | L5 | One SFT example traced through template, collator, masks and loss | 4–6 |
| P08 | L5 | Bounded adapter run, reload and held-out comparison | 6–9 |
| P09 | L6 | Toy DPO objective, pinned source comparison and prior-paper connection | 6–9 |
| P10 | L7 | Reward/verifier audit and hand-traced policy update | 4–6 |
| P11 | L7 | Small online-RL run with fixed rollout/token/time limits | 6–9 |
| P12 | L8 | Resource budget and one measured optimization | 6–9 |
| P13 | L9 | Tiny multiprocess equivalence and restart test | 6–9 |
| P14 | L10 | Linear attention → Kimi KDA recurrence → K3 mechanism/claim investigation | 6–9 |
| P15a | L11 | Capstone question, frozen protocol and baseline | 6–9 |
| P15b | L11 | One treatment/ablation and failure analysis | 6–9 |
| P15c | L11 | Reproduction package and oral defense | 6–9 |

The bounded deliverable may be an intermediate component with its own tests/report. It is not a promise to complete a full Stanford assignment in nine hours. Start with a component whose immediate prerequisites can be demonstrated through a small attempt; teach missing foundations in context before dependent operations. P11 requires a working generation backend; P12's GPU claims require suitable hardware. CPU demonstrations retain narrower labels.

At activation, record the project in the learner's tracker with outcome, deliverable, the learner's chosen start/target finish and weekly hours, acceptance checks and compute cap. Do not turn the whole months-long course into one dated execution project.

## Transformer implementation route

P03 and P04 form the [Transformer workshop](../exercises/transformer/README.md), two bounded stages:

- **P03: Single-head to multi-head attention** — estimated 8 focused hours. Begin with one head, then implement four heads at fixed model width, with causality, reference-equivalence and gradient checks.
- **P04: Tiny decoder-only Transformer** — estimated 9 focused hours after attention and training-loop readiness. Integrate embeddings, one decoder block, vocabulary projection, shifted loss and greedy generation; overfit one fixed batch.

At 3 hours/week each stage takes about three weeks. Single-head attention can start inside [Inference 01](../exercises/inference/README.md) Lab A; reuse that work and count it once. Pin the runtime before execution. Instructor scaffolding is not completed learner work.

## P14 — Linear attention, Kimi KDA and K3

Use the [workshop brief](../exercises/attention-alternatives/README.md) to specialize P14's existing attention-mechanism investigation. Budget nine focused hours: three for linear attention, three for KDA, one for a K3 claim ledger and two for setup, review and repair. The nine hours are included in L10's 15–25 hours; broader report study and final defense use its remaining allocation.

Entry evidence: working causal attention and KV-cache comparisons with explained shapes; use L2/L3 evidence for the report's training context. Begin with tiny fixed inputs and one head on CPU. At three hours/week the slice takes about three weeks; choose dates at activation after readiness. If prerequisites or repair exceed the budget, split the slice into separately bounded deliverables before activation. Full KDA layers, fused kernels and model training are outside this slice.
