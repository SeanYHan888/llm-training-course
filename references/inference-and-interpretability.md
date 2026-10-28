# Reading list — Inference, latent features and J-space

Compiled October 3, 2026 from the agreed [Inference 01](../exercises/inference/README.md) and [Interpretability 01](../exercises/interpretability/README.md) lessons. Sources were checked during course preparation on October 3. October 6 update: use these as on-demand references for the practical activities in the lesson guides. The earlier reading timeboxes below are optional reference budgets, not prerequisites or extra assignments; current activity budgets are in those guides.

These readings fit inside the existing ten-hour investigation. Times are estimates, not recorded study time. Your progress tracker owns the schedule, tasks and progress.

## 1. KV-cache foundations

**[How caching works — Hugging Face](https://huggingface.co/docs/transformers/en/cache_explanation)**

- Read: attention matrices, the cache explanation and attention-mask discussion.
- Time: 15 minutes within the memory-accounting session.
- Focus: which tensors can be reused when one token is appended, and which must be computed again?
- Takeaway to explain: prefill versus incremental decoding, with a small tensor-shape drawing.

## 2. PagedAttention and vLLM

**[Efficient Memory Management for Large Language Model Serving with PagedAttention — Kwon et al. (2023), v1](https://arxiv.org/abs/2309.06180v1)**

- Read: skim §§2–3 as needed; focus on §§4.1–4.3 and Figures 4–7.
- Time: 30 minutes reading plus 15 minutes for a short paper card and feedback.
- Focus: how do logical token blocks map to noncontiguous physical KV-cache storage?
- Paper card: the problem, your own block-mapping drawing, and one limitation or unresolved question.
- Optional: §6 evaluation, only if time remains. Report its speedups as author-reported results, not your measurements.

## 3. SGLang and shared-prefix reuse

**[SGLang: Efficient Execution of Structured Language Model Programs — Zheng et al., v2](https://arxiv.org/html/2312.07104v2)**

- Read: skim §2 for context, then focus on §3.
- Time: 30 minutes reading plus 15 minutes for a short paper card and feedback.
- Focus: how can several branching calls reuse a common token prefix?
- Paper card: the problem, your own prefix-tree drawing, and one limitation or unresolved question.
- Optional: §§6.1 and 6.3. Leave §§4–5 for later. Historical comparisons do not establish a current package ranking.

## 4. Neurons, distributed features and SAE latents

Start with **[Decomposing language models into understandable components — Anthropic](https://www.anthropic.com/research/decomposing-language-models-into-understandable-components)**, then read **[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning — Bricken et al. (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html)**.

- Read: the short overview, then the paper's introduction and **Problem Setup**, especially **Features as a Decomposition**. Skim one feature example.
- Time: 15 minutes overview + 25 minutes paper + 20 minutes drawing/explanation = 60 minutes.
- Focus: why is an individual neuron not necessarily an interpretable concept? How does a sparse autoencoder decompose an activation vector?
- Takeaway to explain: neuron activation versus distributed feature versus SAE latent. An SAE latent belongs to a separately trained analysis model; it is not automatically an original LLM neuron.
- Save: one sketch and one limitation of assigning a human-readable label to a feature.

## 5. J-space, the Jacobian lens and causal evidence

Start with **[A global workspace in language models — Anthropic (2026)](https://www.anthropic.com/research/global-workspace)**, then read **[Verbalizable Representations Form a Global Workspace in Language Models — Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html)**.

- Overview selection: **How we found the J-space** and **Claude thinks in its J-space** — 10 minutes.
- Paper selection: the introduction's **The Jacobian Lens and the J-space**, the multi-hop internal-reasoning example, and **Limitations** — 20 minutes. Defer full Methods derivations.
- Visualization: **[Neuronpedia J-lens demo](https://www.neuronpedia.org/jlens)** → **Multi-Hop Reasoning** — 20 minutes. Compare **Versus Logit-Lens** only if time remains.
- Reflection: 10 minutes to record your prediction, the observed readout, the published intervention/result and one limitation.
- Total: 60 minutes, including the demo and reflection.

Keep the distinction explicit: **J-space** names the studied representations; **J-lens** names the probing method. Seeing a word in a readout is weaker evidence than showing that a controlled intervention changes the answer. The lens is approximate and incomplete, not an exhaustive transcript of reasoning. Label published demo outputs as demo-provided evidence, not a locally reproduced experiment.

## Optional later — outside the current reading budget

| Material | Start with | Purpose |
|---|---|---|
| [Toy Models of Superposition — Elhage et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html) | **Background & Motivation** and **Demonstrating Superposition** | Geometric intuition for representing more features than dimensions |
| [FlashAttention — Dao et al. (2022)](https://arxiv.org/abs/2205.14135) | Abstract, introduction and algorithm overview | Distinguish attention memory-traffic optimization from KV-cache allocation/reuse |
| [Official Jacobian-lens implementation](https://github.com/anthropics/jacobian-lens) | README, then Methods alongside code after autograd/Jacobian foundations | Prepare a later bounded reproduction; pin revisions before execution |
| [SGLang request tutorial](https://docs.sglang.io/docs/basic_usage/send_request) and [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) | One request example and the prefix-cache explanation | Deferred package comparison; match moving documentation to the release used |

## How to use this list

Begin with the current code, hand simulation or visual example. Consult the relevant selection to explain a mechanism or verify a claim, then use it to improve the artifact and record one uncertainty. Keep notes and progress in the existing project owner. Reading completion alone does not establish mastery; the lesson guides contain the learner exercises and review criteria.
