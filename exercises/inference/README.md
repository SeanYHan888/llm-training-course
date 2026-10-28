# Inference 01 — KV cache, serving and internal representations

Prepared October 1, 2026; revised October 6, 2026 for build-first delivery. This is an introductory investigation, not completion of the full inference track. Your progress tracker owns dates, tasks, actual hours and evidence. This document owns reading selections and exercise prompts; it contains no completed learner solutions.

## Question and scope

How does a model turn a prompt into tokens, and how do inference engines avoid wasting computation and memory when serving several requests?

Follow the chain: token generation → prefill/decode → per-sequence KV cache → memory accounting → paged allocation → shared-prefix reuse, then inspect internal representations through a bounded J-space exploration. Link foundation evidence to L0/L2 and resource/measurement evidence to relevant L8/L4 criteria; serving exploration can also count toward a related serving or agent-systems course. This slice does not pass those whole milestones or replace P03/P04.

Budget: 10 focused hours, including reading, setup, debugging and review. Six hours are foundations/assessment; four are exploration: two for serving mechanisms and two for neurons, sparse features and J-space. Full engine installation and GPU benchmarking are a separate optional extension. Teacher supplies explanations during implementation, feedback after attempts and fresh practical variations for coached work. Use [the Transformer scaffold](../transformer/README.md) for the first component; the learner writes the implementation.

**Scope:** keep **KV cache → PagedAttention → SGLang** as the main thread. Complete one small coding investigation plus memory calculations; understand serving mechanisms through drawings and hand traces. A second simulator implementation, production source traces and a four-package comparison are out of scope for this slice.

**Time split:** 360 core minutes; 45 minutes each for PagedAttention and SGLang, 30 for Lab C, and 120 for [Interpretability 01](../interpretability/README.md). The 60-minute package survey (Lab D) is deferred.

| Depth in this window | Topics | Evidence |
|---|---|---|
| Practice directly | Generation correctness, per-sequence KV cache, memory accounting and basic local timing | Lab A/B code, checks and explanation |
| Explain with diagrams and small examples | Paging and shared-prefix reuse | Selected paper cards and Lab C hand trace |
| Explore through published visualizations | Neurons versus distributed features/SAE latents; J-space/J-lens; correlation versus intervention | One feature sketch and one demo observation/limitation note |
| Recognize purpose only; study later | FlashAttention, quantization, speculative decoding, structured output and distributed inference | No assigned reading, implementation or mastery requirement this week |

## Entry and adaptation

Start from your current code. Earlier setup does not establish current runtime compatibility. Select/pin a local environment before execution, with setup capped at 20 minutes inside Lab A.

Begin with the single-head scaffold, then add cached decoding. If matrix multiplication, softmax or masking is unfamiliar, work through a tiny tensor example in the active code with instructor explanation. Implement the smallest useful operation before extending it. No preliminary reading or quiz is required.

Keep the slice's review date and time budget. Use the existing repair block for unfinished correctness checks. If foundations need more time, reduce deeper exploration to one feature example and one J-lens example with targeted context; retain the two short serving drawings and record partial coverage honestly. Remaining work becomes a bounded follow-on, not assumed mastery.

## Practical activity and reference map

These activities replace reading-led delivery within the existing blocks. Sources retain their previously recorded versions; refresh moving APIs when executing. Consult only the figure or passage needed to build or verify the artifact.

| Activity | Work first | On-demand reference | Total block |
|---|---|---|---|
| Lab A | Single-head attention, then same-token cached/full-prefix comparison | [HF cache explanation](https://huggingface.co/docs/transformers/en/cache_explanation), attention/cache/masks | 90 min |
| Lab B | Inspect cache tensors, count elements/bytes, then generalize memory accounting | Same HF reference when a shape or reuse rule is unclear | 60 min |
| PagedAttention | Arrange token blocks into physical slots and draw a logical-to-physical map | [Kwon et al. v1](https://arxiv.org/abs/2309.06180v1), §§4.1–4.3 and Figures 4–7 as needed | 45 min |
| Lab C | Complete the block/prefix hand traces using the fixed requests below | Revisit the source-supported drawings | 30 min |
| SGLang | Draw a branching prefix tree and trace reusable versus new tokens | [Zheng et al. v2](https://arxiv.org/html/2312.07104v2), §3 as needed | 45 min |
| Features/J-lens | Inspect published visual examples, change the viewed input/layer when supported, and annotate findings | [Interpretability 01](../interpretability/README.md) | 120 min |

For each serving mechanism, save a compact card: problem, own drawing and one limitation, with the supporting source. This replaces a reading-completion requirement, not the evidence. Optional evaluation sections, engine source tracing and full simulator implementations remain outside the required scope. Paper speedups are author-reported, never local measurements.

## Lab A — Cached versus uncached attention

**Time:** 90 minutes initially, plus final-session repair/measurement time. **Needs:** existing CPU Python/PyTorch or NumPy environment; no weights, network model calls or GPU required. Cap environment troubleshooting at 20 minutes within the lab. If blocked, do the hand trace and keep numerical validation pending.

**Start building:** implement the single-head reference one operation at a time in the workshop scaffold. Inspect intermediate shapes, then append a token and compare which tensors change. Explain cache reuse using the observed behavior and causal mask.

Build a toy causal attention computation with fixed random projections: batch 1, hidden width 16, one head, vocabulary 32, maximum sequence 32. Use float32, a recorded seed, fixed token IDs, no dropout and no parameter updates. Include deterministic position information indexed by absolute position. A full Transformer and tokenizer are outside this lab.

1. Write an uncached reference that processes the complete prefix, returning its last-position attention output; optionally project that output to vocabulary logits.
2. Write a cached path: prefill once, then append only the new key/value and compute the new query against all available keys. Keep an explicit position counter.
3. Compare the two paths for the same predetermined token sequence at lengths 1, 4, 8 and 16. Start with `atol=1e-5, rtol=1e-4` for this small float32 case; record maximum absolute error and investigate failures rather than widening tolerance blindly.
4. Verify causality, cache length and fresh-request reset. Introduce one wrong mask or position offset, explain the failure, then repair it. This is practice; the assessment uses another variation.
5. If ready, use the vocabulary projection for at most eight greedy generation steps. Random weights produce no meaningful language; generated text quality is not an objective.

**Evidence:** learner code, environment versions, shapes, equality/error table and a short causal explanation. For the tiny case, correctness matters more than speed. Do not assume caching must be faster at this scale.

## Lab B — Memory calculator and controlled measurement

**Time:** 60 minutes including any needed reference lookup, plus 45 minutes in the final session for controlled timing. **Start from the artifact:** inspect each cache tensor’s shape, element count and element size; append tokens and observe the payload growth. Use this to derive the formula before applying the hypothetical configuration below.

Derive the ordinary dense-attention KV payload from the number of K/V elements. Use a hypothetical configuration: 12 layers, batch 2, sequence 1024, 4 KV heads, head dimension 64, two bytes per element. Calculate baseline memory, double sequence length, and change to one KV head. Report bytes and MiB. This is a hypothetical configuration, not a claim about a named model. Explain what is missing: weights, temporary tensors, allocator waste, metadata and runtime overhead.

For Lab A, compare measured tensor bytes with your formula. Time cached/uncached processing of the **same** predetermined continuation, separate prefill from subsequent steps, warm up, and repeat five times. Record median and range with device/backend and lengths; label these small local measurements. Do not report this as an engine benchmark. If generation/streaming is absent, call the measurements attention-step timings, not end-to-end time to first token.

**Evidence:** formula with assumptions, three calculations and one measured comparison. No GPU-performance conclusion follows from a CPU result.

## Lab C — Paging and prefixes by hand

**Time:** 30 minutes. **Needs:** paper or a text editor. No second coding project is required. This is a hand simulation, not an implementation of an engine or attention kernel.

Use token-ID requests `[1,2,3,4,5,6]`, `[1,2,3,4,7,8]`, `[1,2,9,10]`, and `[11,12,13]`; set block size to two tokens. Build the trace with token cards or a text table, then change one request and inspect what is reusable.

1. Spend 15 minutes drawing logical/physical block tables for the first two requests. Label used slots, allocated capacity and blocks released when a request finishes.
2. Spend 15 minutes drawing a shared-prefix tree for all four requests. Assume unlimited retained prefix capacity for this toy. Count reused and new prompt tokens in order, checking that they sum to total prompt tokens. Revisit this drawing during the SGLang practical block, using its selected figure/text to verify the mechanism.
3. **Deferred optional follow-up:** draw a toy timeline: request A needs one prefill tick and three decode ticks; B arrives at tick 2 and needs one prefill tick and one decode tick. Compare processing requests serially with a policy that can admit B between A's decode steps. State your tick/capacity assumptions; this is conceptual scheduling, not a latency prediction.

**Optional only:** translate one already-understood table into standard-library Python if the required work finishes early. Do not add trie implementation, eviction or reference counting to the completion criteria. A full allocator/prefix simulator is a later lab.

**Evidence:** block table and prefix counts; the scheduling sketch is deferred. As you explain them, distinguish per-request KV reuse, cross-request prefix reuse and whole-response caching. Never translate simulated reused tokens or timeline ticks into measured speedups.

## Lab D — Explore inference packages (deferred)

**Time:** zero assigned minutes in the revised window; preserve this 60-minute option for a later slice. **Later option:** a small documentation comparison, with no installation. Spend 20 minutes on SGLang's request example, 20 on vLLM's prefix-cache explanation and 20 drawing the connection to the papers. MLX LM and llama.cpp remain optional links below, not assigned package studies.

| Package | Explore | Primary starting point |
|---|---|---|
| SGLang | Request submission, runtime/scheduler/cache responsibilities; connect paper mechanisms to current documentation | [Request tutorial](https://docs.sglang.io/docs/basic_usage/send_request), [source](https://github.com/sgl-project/sglang) |
| vLLM | Request/engine/cache boundaries and current automatic prefix caching | [APC documentation](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/), [source](https://github.com/vllm-project/vllm) |
| MLX LM | Apple-silicon local inference and how its deployment context differs from a shared GPU server | [Official repository](https://github.com/ml-explore/mlx-lm) |
| llama.cpp | C/C++ inference, model-format/backend choices and local serving | [Official repository](https://github.com/ggml-org/llama.cpp) |

Source tracing is deferred by default. If there is spare time and you choose one narrow source boundary, select a tagged release and record its commit SHA first. Match documentation to that release and record unresolved mismatches. This is optional, not an additional completion requirement.

Create a two-row SGLang/vLLM comparison: what it does, how it relates to your cache drawings, an official evidence link, and one unanswered question. Current vLLM also supports cross-request prefix caching; do not infer that only SGLang reuses prefixes from the historical paper comparison. [vLLM APC](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) explains the feature and its prefill-focused benefit.

**Evidence:** a short comparison with documentation links, or an explicit deferral if this block was used for foundation repair. No speed ranking without controlled measurements.

## Optional engine experiment — follow-up, not required for this window

Use a verified supported host and one model/tokenizer revision supported by both SGLang and vLLM. The [compute plan](../../operations/COMPUTE.md) requires an authorized compute account, allocation/spending limits and bounded run authorization before GPU use. Reserve two additional focused hours only through a later scope decision; do not add them on top of this ten-hour plan.

Run engines sequentially in isolated, pinned environments. Fix model, dtype, token IDs/chat template, sampling settings, context/output limits and workload. Compare shared-prefix and unique-prefix requests, cache cold/warm, concurrency 1 and 4. Keep input lengths at most 512 tokens, output at most 32, and at most 16 requests per condition with a 30-minute total measured-run cap. Choose an appropriately small model only after checking memory, compatibility and license; this note authorizes no allocation or download.

Measure time to first token, completion latency, aggregate output tokens per wall second, peak memory and errors. Distinguish first-start compilation/model load from cold **prefix-cache** conditions. Warm execution before timing; clear prefix state using release-documented controls or record that it could not be controlled. SSE chunks need not equal single tokens: do not label chunk intervals as token latency without token-level evidence. Repeat conditions three times within the cap and report the small sample size. Neither engine is presumed faster.

## Review and completion standard

The final session reserves 45 minutes for repair, 45 for measurement and 45 for practical review. Run the learner artifact, inspect the results and diagrams, and make a fresh code/input change with a brief explanation of its effect. References and hints are allowed during learning; record assistance and use an independent practical variation to establish transfer. No standalone preliminary questionnaire is required.

The minimum deliverable is one small investigation bundle: Lab A correctness evidence, Lab B accounting/basic timing, Lab C hand traces and two short paper cards. The two-hour [Interpretability 01](../interpretability/README.md) adds a feature sketch and one published-demo observation/limitation note. Lab D and the scheduling sketch are deferred. No second simulator implementation, full engine source trace or advanced-topic mastery is required. Unfinished independent criteria remain pending/`revise`; project completion and milestone mastery are separate judgments. Distinguish measured, simulated, author-reported and untested statements.

After the evidence review, the suggested next slice is scheduling and measurement: continuous batching/chunked prefill and, when compute is ready, a controlled vLLM/SGLang run. Later proceed to GPU bottlenecks/FlashAttention, quantization, speculative decoding and distributed serving; structured output is a separate application-driven branch. MLX LM/llama.cpp exploration and production source tracing remain available follow-ups. This is an unscheduled topic map, not extra work for this week.
