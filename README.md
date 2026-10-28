# Language Models: From Foundations to Training Research

A project-first course on how large language models are built, trained, adapted, evaluated and scaled. You implement small pieces of a model, run them, inspect the results and debug them. The repo's study notes explain each concept as you need it. Primary papers and docs are there for depth, not as prerequisites.

By the end you should be able to explain what a language model computes, train and adapt small models, design a valid experiment, reason about memory and distributed training, and judge what a frontier technical report does and does not establish.

## Where to start

1. **Build:** open the [Transformer workshop](exercises/transformer/README.md). Copy its scaffold [attention.py](exercises/transformer/attention.py) into your private `work/` folder and implement the TODOs there (see [Your code stays private](exercises/README.md#your-code-stays-private)).
2. **Learn while building:** use the [study notes](notes/README.md), one per module. For attention and the KV cache, start with [L2 — Transformer](notes/02-transformer.md).
3. **Track progress privately:** keep your status, decisions and evidence in `work/admin/` or your own tracker (see [Your code stays private](exercises/README.md#your-code-stays-private)).

## Course map

| Module | Study notes | Hands-on work |
|---|---|---|
| L0 Foundations | [00-foundations](notes/00-foundations.md) | [Diagnostic](exercises/diagnostic/README.md), P01 |
| L1 Neural learning | [01-neural-learning](notes/01-neural-learning.md) | P01 training loop and gradient checks |
| L2 Transformer LM | [02-transformer](notes/02-transformer.md) | [Transformer workshop](exercises/transformer/README.md), P02–P04 |
| L3 Pretraining | [03-pretraining](notes/03-pretraining.md) | P05–P06 |
| L4 Evaluation | [04-evaluation](notes/04-evaluation.md) | P06 |
| L5 SFT and PEFT | [05-sft-peft](notes/05-sft-peft.md) | P07–P08 |
| L6 Preference learning | [06-preferences](notes/06-preferences.md) | P09 |
| L7 Online RL | [07-online-rl](notes/07-online-rl.md) | P10–P11 |
| L8 Efficiency | [08-efficiency](notes/08-efficiency.md) | P12 |
| L9 Distributed training | [09-distributed](notes/09-distributed.md) | P13 |
| L10 Frontier architectures | [10-frontier](notes/10-frontier.md) | [Attention alternatives](exercises/attention-alternatives/README.md), P14 |
| L11 Research practice | [11-research](notes/11-research.md) | P15a–P15c capstone |
| Elective: inference | [inference-serving](notes/inference-serving.md) | [Inference 01 labs](exercises/inference/README.md) |
| Elective: interpretability | [interpretability](notes/interpretability.md) | [Interpretability 01](exercises/interpretability/README.md) |

Milestone outcomes and prerequisites are in the [roadmap](course/ROADMAP.md); project deliverables are in the [project catalog](course/PROJECTS.md).

## How this course works

- **Build first:** write a small component, run it, inspect results and debug. Learn the mathematics and mechanisms when your code needs them; ask for explanations at any point.
- **Learn through the artifact:** hints, scaffolds and the study notes support your implementation. Reading and preliminary quizzes are not the default lesson format.
- **Use the ecosystem:** Transformers, Datasets, PEFT and TRL, with source traces connecting abstractions to the same mechanism.
- **Defend the evidence:** predictions, held-out evaluation, debugging, controlled comparisons and honest limits. See the [assessment rubric](course/ASSESSMENT.md).

This course is adapted from Stanford CS336, the Kimi K3 report, TRL and broader training literature. It is not affiliated with Stanford.

## Repository layout

| Folder | Contains |
|---|---|
| [`notes/`](notes/README.md) | Self-contained study notes, one per module |
| [`exercises/`](exercises/README.md) | Workshops, labs and learner code |
| [`references/`](references/README.md) | Pinned primary sources, reading routes and ecosystem briefs |
| [`course/`](course/CHARTER.md) | How the course runs: charter, roadmap, syllabus, schedule, projects, assessment, idea intake, glossary, design research |
| [`operations/`](operations/COMPUTE.md) | Compute access and reproducibility plan |

| Document | Owns |
|---|---|
| [Charter](course/CHARTER.md) | Authority, essential scope, workload and change rules |
| [Roadmap](course/ROADMAP.md) | Milestone outcomes, prerequisites and core effort estimates |
| [Syllabus](course/SYLLABUS.md) | Module boundaries, material placement and lesson requirements |
| [Schedule](course/SCHEDULE.md) | Teaching cadence, planning method and duration scenarios |
| [Study notes](notes/README.md) | Concept explanations for each module |
| [References](references/README.md) | Source roles, provenance and revision-selection rules |
| [Projects](course/PROJECTS.md) | Bounded deliverable options |
| [Assessment](course/ASSESSMENT.md) | Mastery rubric, tests, defenses and reassessment |
| [Ideas](course/IDEAS.md) | Student idea placement, substitutions and scope proposals |
| [Glossary](course/GLOSSARY.md) | Shared terminology |
| Your private `work/` folder | Your code, progress tracker, decisions, dates, hours and evidence |

The instructor owns the essential milestones and their independent assessment. Student ideas receive a reasoned place through idea intake. Equivalent projects can replace exercises while keeping the required outcomes. The learner chooses their own weekly hours and dates; the [schedule](course/SCHEDULE.md) turns any pace into a forecast. Adding a required milestone or spending money needs an explicit trade-off and agreement.

## Environment

Builds run on CPU with tiny tensors; the attention scaffold has been run with Python 3.13 and PyTorch 2.9 on CPU. Pin your environment before each lab. GPU labs need a course-appropriate allocation and spending limits from the [compute plan](operations/COMPUTE.md).
