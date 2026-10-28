# Build a tiny Transformer

Build first, introduce concepts when the implementation needs them. The learner writes core behavior; the instructor supplies scoped scaffolds, explanations and reviews. Reading is optional support by default. Assess through running, tracing and modifying the learner's code; no separate preliminary quiz is required.

## Build sequence

| Capability | Observable result |
|---|---|
| Single-head causal attention — current step | Correct shapes, normalized masked weights and causality checks |
| Incremental KV reuse — inference Lab A | Cached and full-prefix outputs agree for fixed tokens and positions |
| Memory accounting — inference Lab B | Count actual cache elements/bytes, then generalize to hypothetical configurations |
| Multi-head attention — P03 continuation | Multiple heads agree with an independent per-head reference |
| Decoder block and vocabulary head — P04 | Embeddings, residuals, normalization and MLP produce causal logits |
| Tiny training loop and generation — P04 | One-batch loss decreases and tokens are generated autoregressively |

Expand only the current capability into implementation TODOs. The code TODOs in [attention.py](attention.py) own implementation detail. That file is the question: copy it to `work/transformer/attention.py` and implement there, so your answer never lands in the published course (see [Your code stays private](../README.md#your-code-stays-private)). Your progress tracker owns project tasks, dates and evidence. Reuse work across inference and P03/P04; count its time once.

Later continuation: [linear attention → Kimi Delta Attention (KDA) → Kimi K3](../attention-alternatives/README.md) is now an explicit L10/P14 build route. Start it after causal-attention/KV evidence and activation review; the current Transformer component remains the immediate workspace.

## First work block

Use up to 20 minutes of today's existing Lab A block to implement Step 1.1 and inspect Q/K/V shapes. Begin with the supplied random input and weight matrices: fixtures, not trained language representations. No tokenizer is needed yet. Explanations, experiments and debugging hints happen when useful.

## Environment

The scaffold has been run with Python 3.13 and PyTorch 2.9 on CPU; record your environment with each run. Keep setup within the existing 20-minute troubleshooting cap. CPU only; no model downloads or cluster jobs.

## Credit and remaining work

The complete build can replace separate instruction/exercises on attention, tensor shapes within that computation, decoder assembly, shifted loss and basic generation when demonstrated in code. KV caching and memory lessons can be integrated rather than repeated. This does not pre-credit sessions or all L1/L2 outcomes. Tokenizer construction, RoPE/gated MLP coverage, broader training/evaluation, serving and interpretability remain separate unless equivalent practical work is substituted.

The first component started inside the Inference 01 slice. Multi-head attention (P03) and the decoder (P04) follow as separate bounded projects, each sized to about three weeks at the learner's chosen pace; choose their dates in your tracker.
