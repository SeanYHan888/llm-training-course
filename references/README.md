# Source guide

Checked September 28, 2026. This is a selective reading map, not a requirement to finish every resource. Teaching questions and experiments are our own curriculum design. Recheck moving docs and select exact sections before a lesson.

The [syllabus](../course/SYLLABUS.md) assigns reading purposes to each module and defines the lesson contract. Use references to answer a concrete implementation or evidence question. Offer exact sections and a short timebox when useful; reading is optional support by default. Source-based claim investigations still need the relevant original evidence, which can be inspected alongside the artifact. This guide owns source roles and provenance, while the syllabus owns instructional placement. A student-suggested source enters through [idea intake](../course/IDEAS.md); it does not automatically expand the required reading list.

The [study notes](../notes/README.md) are the in-repo explanation for every module; each ends with a "Going deeper" table of exact sections from the sources below. Papers stay as version-pinned links rather than local copies.

## Focused reading lists

- [Inference, latent features and J-space](inference-and-interpretability.md) — reference links and selected sections to support the current simulations and visual investigations.
- [Linear attention, Kimi KDA and K3](../exercises/attention-alternatives/README.md) — build sequence, checks and versioned primary references for P14.

## Course sources

| Source | Role | Read when / question |
|---|---|---|
| [CS336 Spring 2026](https://cs336.stanford.edu/) | Main training backbone | L2–L10: connect architecture, data, optimization and systems |
| [CS336 basics assignment](https://github.com/stanford-cs336/assignment1-basics) | Optional original assignment / component specification | L2/L3 after prerequisites; pin a course-compatible commit and inspect current rules |
| [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Foundation bridge | L0/L1: tensors → data → model → autograd → optimization → save/load |
| [PyTorch autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | Mechanism reference | L1: what graph is recorded, which tensor gets gradients, what backward does |
| [Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1) | Ecosystem bridge | L2/L5: model/tokenizer/dataset concepts; selected sections only |
| [Datasets processing](https://huggingface.co/docs/datasets/process) | Data API reference | L3/L5: inspect map/filter/split operations and retain provenance |
| [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating) | Input-contract reference | L5: inspect the exact serialized conversation and special tokens |
| [TRL documentation](https://huggingface.co/docs/trl/index) | Practical trainer/source route | L5–L7: see [ecosystem review](training-ecosystem.md) |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original architecture context | L2: distinguish original encoder–decoder design from our decoder-only model |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Scaling-law case | L4/L8: which compute/data assumptions support an allocation rule? |
| [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) | Efficient adaptation mechanisms | L5: distinguish trainable adapters, frozen base and quantized storage |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | Objective/assumptions | L6: derive the preference objective and connect it to actual tensors |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Evaluation tooling | L4/L11: inspect task config, prompting, scoring and raw results |
| [Transformers are RNNs, ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html) | Linear-attention foundation | P14: compare a causal feature-map reference with a recurrent state implementation; this is a different attention rule from softmax |
| [Kimi Linear v2](https://arxiv.org/html/2510.26692v2) | Kimi Delta Attention (KDA) mechanism | P14: §2.2 delta-rule foundations, §3 fine-grained gating, §4 hybrid architecture; optional kernel material later |
| [Kimi K3 v2](https://arxiv.org/html/2607.24653v2) | Recurring frontier-report case study | P14: §2.1 mechanism claim ledger; revisit §3 pretraining with L3/L4, §4 post-training with L5–L7, §5 systems with L8/L9, and §6 evaluation with L4/L10 |

The linear-attention proceedings page, Kimi Linear v2 and Kimi K3 v2 section routes above were checked October 7, 2026. Paper versions are selected references; runnable package/source revisions remain to be pinned at lab activation.

## CS336 integration

Use the current course's basics, systems, scaling, data, and alignment/reasoning strands as references. Our route inserts foundation remediation and evaluation practice, spreads work into bounded projects, and gives DPO more weight because it connects to your research. Original assignment rules restrict directly AI-solved work; use learner implementations and conceptual feedback. Stanford dates, compute access, grading infrastructure and credit do not transfer to this course. Source: [course content and policy](https://cs336.stanford.edu/).

## Source-reading method

Start from one question and one pinned call path. For L5: dataset row → template/tokenizer → collator/masks → forward/loss → backward/update → save/reload. For L6: pair → policy/reference log-probabilities → objective → gradients. For L7: prompt → rollout → reward → grouping/advantage → objective → update. Identify actual symbols in the installed source at lesson preparation; avoid memorizing unstable internal filenames.

## Pins at activation

Record source URL, access date, course year/paper version, Git commit, Python and package versions, model/tokenizer revision, dataset revision, backend and device. Keep a lockfile with each runnable project. The website's current docs are not a tested dependency matrix. No environment has been selected yet.
