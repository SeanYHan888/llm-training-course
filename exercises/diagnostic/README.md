# Session 01 — What changes when a model learns?

A 75-minute diagnostic guide. The course now starts from the build-first workshop rather than this full questionnaire; reuse an individual prompt when it helps explain the current code. Goal: identify where your model understanding is solid and which bridge is needed. This is an attempt-first diagnostic, not a test you must pass to deserve the course.

## Before the session

Bring any evidence you choose. Confirm weekly capacity and hardware constraints. You may bring one figure, equation or short training-code path from a previous training or research project; no private research files need to be copied into this course.

## 0–15 minutes: draw the whole path

In your own words, describe what happens to the text “The cat sat” between a dataset row and one parameter update. Mark which parts are data, learned parameters, temporary tensors and optimizer state. Explain how generating the next token differs from training. It is fine to write “I don't know” at any step.

## 15–35 minutes: tensor and objective attempt

Use batch size 2, sequence length 4, vocabulary size 7 and hidden width 8. Predict the shapes of input IDs, embeddings and output logits. Sketch which tokens supply targets and which positions may attend to which other positions. Explain what padding should contribute to the loss.

For a three-class next-token distribution `[0.1, 0.7, 0.2]` with the third token as target, write the loss expression and predict which probability training should increase. Explain why increasing a probability is not the same operation as directly changing one entry in a stored table. Derive or explain a tiny chain-rule example if ready.

## 35–50 minutes: locate the update

Sketch the order of forward, loss, backward, optimizer update and gradient clearing. Predict what happens if gradients are detached, never cleared, or computed while all parameters are frozen. If a local PyTorch setup already exists, inspect a tiny learner-written example; installation troubleshooting should not consume the whole diagnostic.

## 50–65 minutes: connect to preference learning

For one prompt with preferred and rejected completions, identify the policy, reference and completion log-probabilities. Explain which tensors require gradients and which tokens belong to the scored completion. Separate “I know the derivation” from “I can trace this in our code.” No full derivation is required on the first day.

## 65–75 minutes: choose the next lesson

Record strengths, uncertainties, assistance and the smallest next experiment in your progress tracker. If gradients are unclear, start L1. If those are strong but attention shapes are unclear, move to L2. If both are demonstrated, use L3 rather than repeating introductory material. Explain any prerequisite detour.

The teacher prepares one targeted explanation after the attempt. No answer key or completed training implementation is supplied in this planning step.
