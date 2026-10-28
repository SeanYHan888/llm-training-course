# Exercises

Hands-on work for the course. Each workshop states its goal, checks and time budget; code TODOs hold the implementation steps. You write the assessed implementations. Use the matching [study notes](../notes/README.md) when a concept blocks you.

| Workshop | Module | When | Notes to use |
|---|---|---|---|
| [Transformer: attention → tiny decoder](transformer/README.md) | L2 (P03/P04) | Start here | [L2](../notes/02-transformer.md), [L1](../notes/01-neural-learning.md) |
| [Inference 01: KV cache, paging, prefixes](inference/README.md) | Elective; L2/L4/L8 evidence | Alongside the Transformer build (Lab A reuses single-head attention) | [Inference and serving](../notes/inference-serving.md) |
| [Interpretability 01: features and J-lens](interpretability/README.md) | Elective; L1/L2/L4 | Within Inference 01 | [Interpretability](../notes/interpretability.md) |
| [Attention alternatives: linear attention → KDA → K3](attention-alternatives/README.md) | L10 (P14) | Future; activates after causal attention/KV evidence | [L10](../notes/10-frontier.md) |
| [Diagnostic session](diagnostic/README.md) | L0 | Optional entry check; reuse single prompts | [L0](../notes/00-foundations.md) |

## Your code stays private

Files under `exercises/` are the questions: TODOs, stubs that raise `NotImplementedError`, and checks you are asked to write. Never fill them in place. Copy a scaffold into `work/` and implement it there:

```bash
cp exercises/transformer/attention.py work/transformer/attention.py
```

`work/` is listed in `.gitignore`, so the course repository never publishes it. It is its own git repository; commit your work there and push it only to a **private** remote. Keep your progress tracker, decisions, compute access and other personal notes in `work/admin/` too. Instructors and agents follow the same rule: learner implementations, worked solutions, filled-in checks and personal details go in `work/`, never in published files.

Projects without a workshop yet (P01, P02, P05–P13, P15) are listed in the [project catalog](../course/PROJECTS.md). Their scaffolds are written when each one is activated.
