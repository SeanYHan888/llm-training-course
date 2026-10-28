# Compute and reproducibility plan

Most of the course runs on a laptop CPU with tiny tensors. GPU labs need hardware you are allowed to use, a budget and a bounded run plan. Keep your own access details (cluster name, account, login procedure, spending limits) in your private `work/admin/` folder, never in this repository.

## Routes

| Route | Suitable work | Limits to preserve |
|---|---|---|
| CPU / laptop | Tensor/autograd checks, tiny models, data inspection, loss calculations, small correctness tests | Not evidence of CUDA kernel behavior or GPU throughput; Apple MPS availability and operations require a smoke test |
| Authorized single GPU | Small pretraining, adapter SFT/DPO, constrained online RL, memory/profiling experiments | Choose model, context, batch and precision from actual VRAM measurements; no universal fit promise |
| Authorized multiple GPUs | DDP/sharding correctness, communication and throughput experiments | Requires measured device/interconnect/topology; simulation cannot establish scaling speedups |

Start with a tiny random model and synthetic or small clearly licensed data. Model/data selections stay provisional until licensing, hardware, tokenizer/chat template and task suitability are checked. Do not download large weights merely to prepare a lab.

Before using a shared cluster or a paid provider, confirm which account or allocation permits coursework, the partition and job-time rules, allocation limits and any charges. Run compute on allocated compute nodes according to the cluster's rules. Seeing a partition listed does not establish entitlement, free capacity or a tested software environment.

## Before a real run

1. Verify device count, memory, driver/runtime and allocated access; select a supported Python/PyTorch/Transformers/TRL/PEFT stack and save its lockfile.
2. Validate model/tokenizer/dataset revisions and use rights. Inspect a few serialized examples, labels and masks.
3. Derive memory components for the actual precision/optimizer/adapter setup; run a short correctness and memory smoke test.
4. Set maximum optimizer steps, input/output tokens, rollout count, wall time, disk use and approved cost. Estimate spend from measured throughput and current provider billing, including storage/idle charges where relevant.
5. Specify checkpoint cadence, resume state, logging and a stop condition; verify one save/load cycle.
6. Launch only the bounded authorized run. Stop or shrink it on non-finite loss, unexpected memory behavior, repeated generation failures or approaching budget.

Memory accounting includes weights, trainable gradients, optimizer states, activations, temporary buffers and any reference/reward/value/rollout model. Adapter parameter savings alone do not describe total memory. Online RL requires both generation and optimization capacity. Record measured peaks rather than promising that a named model size fits a given device. See [L8 — Efficiency](../notes/08-efficiency.md) for the memory model.

A CPU distributed-correctness exercise may precede GPU access. Hardware-specific criteria remain pending until measured. Pin package docs to the selected release; consult [the ecosystem brief](../references/training-ecosystem.md) before selecting trainers.
