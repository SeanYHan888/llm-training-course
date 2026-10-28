# Course design research

September 28, 2026. Primary-source review plus original curriculum recommendations. This is an initial design investigation, not an exhaustive literature review or hardware benchmark.

## What the supplied sources can and cannot do

[CS336](https://cs336.stanford.edu/) expects Python, deep-learning/PyTorch familiarity and mathematical readiness. It is implementation-heavy. Recommendation: use it as the technical backbone with a diagnostic bridge, rather than treating its first lecture as a beginner introduction. The official self-study advice starts correctness work on CPU before GPU runs.

The supplied [K3 report](https://arxiv.org/abs/2607.24653v2) is a frontier-model case study. Its reported system scope makes a complete reproduction unsuitable as the initial learning project. Recommendation: revisit a few claims after architecture, pretraining, post-training and systems study. See the separate [source investigation](../references/training-ecosystem.md) for verified content and access limits.

[TRL](https://huggingface.co/docs/trl/index) supplies post-training implementations. It does not replace learning tensors, architecture, data construction, basic optimization or experiment design. Recommendation: independently understand a small objective, then inspect a pinned trainer's actual data/loss/update path. Keep conceptual derivations separate from version-specific API recipes.

## Filling the missing foundations

The [PyTorch basics sequence](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) supports a short readiness bridge; [autograd documentation](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) supports gradient reasoning. Diagnose the chain rule, broadcasting, reductions and log-probabilities using tiny examples before assigning a whole remedial mathematics course.

Use [the Transformer paper](https://arxiv.org/abs/1706.03762) for architectural context, while learning the actual decoder variant we implement. Use [compute-optimal training](https://arxiv.org/abs/2203.15556) to practice reasoning about constrained experiments. Scaling relationships estimated in one regime are questions to test, not a universal recipe for toy runs.

Use a learner's earlier preference-learning work, if any, to connect [the objective](https://arxiv.org/abs/2305.18290) to implementation and evaluation. The motivating gap might lie in architecture, gradients, tooling abstractions or experiment interpretation; current evidence does not identify which. Diagnose before repeating material.

## Training research includes experimental discipline

Our course requires dataset provenance, splits before tuning, baselines, raw outputs, configuration/revision capture, resource accounting and failure reports. These are assessment choices, not claims that a particular library guarantees reproducibility. The [evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) is a tool to inspect, not a substitute for validating the metric and task setup.

## Scope decisions

Train one tiny model from scratch for mechanism understanding, and adapt an existing model for practical post-training. Reuse experiments across lessons where valid. Keep foundation, applied and research evidence distinct. A tiny model can test a mask or loss implementation; it cannot reproduce frontier-model quality or scaling conclusions.

Broad model literacy includes attention alternatives/MoE, long context, multimodality, distillation, quantization and serving interactions. Full implementation depth for every family would make the course unbounded. The roadmap separates mandatory literacy from optional specialization. Detailed lessons beyond the first session are prepared from learner evidence, not generated as a textbook in advance.

## Unresolved constraints

At the initial investigation, weekly capacity, available GPU/VRAM and permitted remote access, monthly spend, Python/PyTorch/math baseline, and any prior-work artifact for credit review were unresolved. Current policy is in the [charter](CHARTER.md); use it for planning. Hardware, baseline and artifact evidence still need resolution. Current provider prices and hardware compatibility must be checked when a concrete run is proposed. No provider or model purchase is recommended yet.
