# Assessment

Each milestone is `demonstrated` or `revise`. Assess the learner's understanding separately from whether an artifact runs. Evidence belongs in the learner's private progress tracker; link code/config revisions and logs here when artifacts exist.

| Dimension | Required evidence |
|---|---|
| Correctness | Reproducible inputs, an appropriate baseline, expected behavior and meaningful failure checks |
| Understanding | Student traces data, tensor shapes, objective and parameter update in their own words, including assumptions |
| Transfer | Prediction and diagnosis of an unfamiliar change, followed by a test and explanation |
| Experimental judgment | Controlled comparison, held-out separation, measured conditions, uncertainty and limits without inflated claims |

All dimensions applicable to a milestone must be demonstrated before passing it; a small component check need not demonstrate the entire milestone at once. Reading a paper, running a Trainer, finishing an AI-generated notebook or obtaining a lower training loss alone is insufficient. Credit earlier valid work; reassess only the unresolved gap. A well-investigated failed experiment can demonstrate understanding.

## Session procedure

Read prior evidence and current code, name the capability being built, and inspect the learner’s implementation and checks. Use a small modification or debugging case to observe transfer, with a brief explanation tied to the code. A standalone quiz is not the entry gate. Record assistance and results for the applicable dimensions; mark unobserved criteria pending. Teach gaps in context and revisit substantially coached work with a fresh variation later.

Useful variations: alter sequence length; insert padding; reverse a preference pair; detach a tensor; change accumulation steps; inject train/test overlap; restore weights without optimizer state; give every GRPO completion equal reward. The teacher selects the appropriate variation at assessment time rather than providing solutions in advance.

## Tests and milestone defenses

The accepted standard is independent mathematical reasoning, implementation, experimental evaluation and defense. The instructor selects practical variations and judges evidence against the rubric; completion is mastery-based rather than an average that can hide a missing dimension.

- **Entry evidence:** begin with existing code or a small implementation attempt. Use Session 01 prompts only if useful to a specific gap; preserve its earlier completion record.
- **Weekly practical check:** run and inspect a component, repair a failure or adapt it to an unfamiliar input. Explain the mechanism using that code. This consumes the existing review allocation.
- **Project review:** inspect the learner-authored artifact, meaningful checks, experimental conditions and explanation. Automated code tests support correctness; they do not establish conceptual mastery.
- **Milestone defense:** independently explain or derive the key mechanism, defend the artifact and its limits, and address an unfamiliar variation. Reuse valid prior/project evidence; do not require duplicate artifacts merely for assessment.
- **Cumulative review:** at each execution-window review, reuse or modify an earlier component to check a previously demonstrated capability when one exists. The capstone integrates the relevant prior criteria.

Tell the learner the criterion and allowed references/assistance before each assessment. Instruction and practice may include hints; independently assessed responses must originate with the learner. Record assistance and reassess a substantially coached response with a fresh variation. Do not prewrite learner submissions or provide the forthcoming assessment answer key.

A `revise` result identifies the specific missing dimension, a bounded remediation and a fresh reassessment. Preserve demonstrated criteria and prior credit. Teach the missing prerequisite through a smaller code example before the operation that requires it; independent components may continue. Apply the charter's agreement rule if remediation changes an agreed milestone date. The instructor cannot lower a criterion merely to meet a calendar target.

## Run evidence

Record code commit or file hash, Python/package lock, device/driver/backend, model and tokenizer revisions, dataset source/revision/license and split hashes, preprocessing/template/masks, seeds, precision, optimizer/schedule, effective batch and token counts, checkpoint path/state, commands, wall time/peak memory, evaluation config/raw outputs, assistance and limitations. A seed alone does not guarantee identical results across platforms.

Keep prompts/test data fixed for a comparison. Fit and debug on development data, then evaluate the held-out set. Use repeated seeds or uncertainty estimates when affordable; label single-run and small-sample results. Separate reward improvement, preference accuracy, likelihood and task success. For judge-based metrics, record judge revision, prompt and order/length checks.

## Capstone acceptance

A reader can reproduce the small experiment within the declared environment and budget; the learner can explain a full example from data to update; the evaluation supports the narrow conclusion; a novel failure is diagnosed. No frontier-quality or general superiority claim follows from a toy result. Publishing results is a separate decision.
