# Interpretability 01 — Neurons, latent features and J-space

Prepared October 3, 2026; revised October 6 for visual investigation first. A **two-hour conceptual exploration** within [Inference 01](../inference/README.md), not an additional project or a full mechanistic-interpretability course. Your progress tracker owns dates, tasks, actual time and learner evidence.

## Question and boundaries

What can we learn about the representations used during inference, and what evidence would establish that a representation affects an answer?

Connect L1 vectors/activations and L2 token/layer computation to L4 controlled comparisons. The overview needs only a basic picture of layers and vectors; pause for a small drawing if these are unfamiliar. Implementing sparse autoencoders or computing Jacobians requires later linear-algebra/autograd readiness. No model download, GPU allocation, SAE training or J-lens reproduction is assigned.

Keep these objects distinct:

| Object | Meaning | What to inspect |
|---|---|---|
| Neuron activation | One unit's numerical value at a particular input position/layer | Which inputs activate it; whether several unrelated patterns do |
| Distributed feature | A pattern represented across activation coordinates | A direction or pattern across units, not necessarily a single neuron |
| SAE latent | A unit in a separately trained sparse autoencoder used to decompose model activations | Its activating examples and reconstruction limitations; labels remain hypotheses |
| J-lens readout | A token-associated score derived from a layer's activation using a Jacobian-based lens | How readouts vary across layers/positions and respond to interventions |

“Latent nodes” is informal: identify the object before interpreting the picture. A cache stores tensors for reuse; an interpretability lens studies information in activations. J-space is not another KV-cache algorithm. The J-lens is approximate and incomplete; a word readout is not an exhaustive transcript of reasoning or evidence of subjective experience.

## Two practical investigations

Keep 120 minutes total. Start from a visual example and use selected sources to answer questions that arise. Sources were checked October 3; refresh the demo at use time. This is inspection of published examples, not a new model or SAE implementation.

### Features — 60 minutes

1. Spend 20 minutes inspecting one feature example in [Towards Monosemanticity (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html). Compare its activating examples and record one repeated pattern plus an exception or uncertainty.
2. Spend up to 15 minutes on instructor explanation or the paper’s **Problem Setup / Features as a Decomposition**, supported by [the short overview](https://www.anthropic.com/research/decomposing-language-models-into-understandable-components), to distinguish neuron coordinates from SAE latents.
3. Spend 25 minutes drawing activation vector → sparse latent representation → reconstruction and revising the drawing against the example. Label the feature description as an interpretation, and include one reconstruction/labeling limitation. Ask for vector explanations as needed; no advance reading gate.

### J-lens — 60 minutes

1. Spend 25 minutes inspecting **Multi-Hop Reasoning** in the [Neuronpedia demo](https://www.neuronpedia.org/jlens). Inspect available layers/token positions and the published intervention result. Record the prompt, viewed position/layer, readout, baseline answer and changed output. If controls are unavailable, compare the published views; if the demo is unavailable, use the corresponding paper figure.
2. Spend up to 15 minutes consulting **The Jacobian Lens and the J-space**, the multi-hop example and **Limitations** in [the technical paper](https://transformer-circuits.pub/2026/workspace/index.html), or [the overview](https://www.anthropic.com/research/global-workspace), to explain what the readout measures.
3. Spend 20 minutes annotating what the example supports and what it leaves uncertain. Add one possible control and one alternative explanation. The instructor gives feedback on that artifact, rather than a separate questionnaire.

Label precomputed outputs **author-reported/demo-provided**; they are not locally measured results. A visible concept alone establishes less than a controlled intervention, and one example does not establish a universal reasoning mechanism. Record assumptions made before inspection if any, but do not invent a prediction afterward. The saved feature sketch and observation/limitation note retain the existing exploration deliverables.


## Optional reading ladder — outside this window

1. [Elhage et al., Toy Models of Superposition (2022)](https://transformer-circuits.pub/2022/toy_model/index.html): **Background & Motivation** and **Demonstrating Superposition** first. Use its geometric examples to understand how more features than dimensions can be represented; leave phase transitions and advanced geometry for later.
2. Return to the full **Problem Setup** and feature analyses in *Towards Monosemanticity* after the first vector/activation lessons.
3. Read the J-lens **Methods** and [official implementation](https://github.com/anthropics/jacobian-lens) after Jacobians/autograd and residual-stream tracing. Pin revisions and scope a separate small reproduction then.

If foundations need repair, retain one feature example and one J-lens example with just enough explanation to interpret them, and return remaining exploration time to core work. Record partial coverage honestly and revisit it at the inference review; do not expand the ten-hour budget or mark learner work complete during preparation.
