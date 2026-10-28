# Course charter

The course runs on independent mastery assessment, explicit idea placement and a learner-chosen schedule. Detailed lesson selection and estimates remain subject to diagnostic evidence.

## Destination

Independently explain, implement, train, adapt, evaluate and debug small language models; read modern training reports with enough mathematical and systems understanding to identify what a result does and does not establish. Connect to the learner's prior work where it exists, without making any single method the only lens on training.

Core outcomes: Python/PyTorch and relevant mathematics; neural learning and optimization; tokenization and decoder architecture; pretraining data and training loops; evaluation and experimental design; SFT and parameter-efficient adaptation; preference optimization and reward learning; policy-gradient/RLVR foundations; resource accounting, profiling and distributed-training reasoning; technical-report criticism and reproducible research.

Advanced exposure includes MoE, attention alternatives, long context, multimodality, distillation and quantization. Full frontier-scale reproductions, every TRL algorithm, every CS336 assignment, and every model modality are optional specializations. The [roadmap](ROADMAP.md) specifies required depth.

## Workload

**The learner chooses the schedule.** They set how many hours per week to study, how to split the time between core work and exploration, and when to start and finish each project. The course supplies effort estimates (focused learner hours in the [roadmap](ROADMAP.md) and [project catalog](PROJECTS.md)), so a learner can turn any weekly pace into a forecast; the [schedule](SCHEDULE.md) shows how. The instructor plans lessons within the time the learner says is available and never assumes more.

Exploration is optional. An exploration may earn core credit when it demonstrates the same required capability; count its hours once. Account for material shared with other courses once.

Keep one primary module and one bounded course project active at a time. Size each project so it is independently finishable, about three weeks at the learner's pace by default (for example, at most nine hours of work at three hours/week). Split a project or reduce its deliverable rather than stretching it indefinitely. Assessment, revision and environment troubleshooting consume study time.

The roadmap's hour ranges are planning estimates for adapted selections, not measured learner velocity, full Stanford assignment estimates or GPU-hour promises. Reforecast from actual hours after the first completed project. GPU access and spending limits are covered by the [compute plan](../operations/COMPUTE.md).

## Teaching and change process

**Build-first delivery:** the learner implements a tiny Transformer and asks for explanations as gaps arise. Use active code, small experiments and debugging as the default lesson format. Standalone reading and pre-lesson questions are not a prerequisite gate. Assess understanding and transfer through learner-authored implementation, traces and unfamiliar code changes; substitute demonstrated project work for equivalent standalone lessons.

Start from the current artifact and one concrete implementation goal. Let the learner attempt it, inspect the output or failure, explain the missing mechanism, then make a small change to check transfer. Use predictions when they help debugging, not as a separate entry quiz. Offer a short source excerpt or video when requested or useful to unblock the build. Give worked explanations when useful, but preserve student authorship of assessed artifacts. For original Stanford assignments, follow their current collaboration/AI rules; keep solution material out of public publishing unless explicitly permitted.

The instructor owns the essential milestones, module boundaries, readings, examples, assessments and prerequisite-respecting sequence. The instructor may adjust readings and substitute equivalent exercises within the learner's chosen time, then report the change. Preserve valid prior learning credit and stable milestone IDs. Core outcomes remain required when a project or reading changes.

Dates and weekly hours belong to the learner; the instructor proposes, the learner decides. Adding a required milestone or spending money requires a concrete proposal and the student's agreement. The student may request additional milestones. Once a specific change is authorized, implement it without reopening the same decision.

A new topic request authorizes evaluating and recording its place in the course. It does not by itself make the topic a required milestone or assign it immediate study time. Use the [idea integration procedure](IDEAS.md) to integrate, substitute, defer, propose a milestone or route the idea elsewhere. Explain the placement and next opportunity.

For a new idea: record purpose, prerequisite, smallest useful experiment, hours, displaced work and revisit condition in [IDEAS](IDEAS.md). Infrastructure spending, external publication and long training runs need their own concrete authorization.

## Ownership

This folder owns curriculum and durable learning artifacts. The learner's private progress tracker (in `work/admin/` or a tool of their choice) owns live tasks, status, deadlines, hours and mastery evidence. The [assessment rubric](ASSESSMENT.md) owns the evidence standard. [Projects](PROJECTS.md) owns bounded options; [materials](../references/README.md) owns reading roles; the learner's private decision log records revisions they agree to. A saved course plan is not student completion evidence.

The [syllabus](SYLLABUS.md) maps module boundaries, material selections and project evidence to the [roadmap](ROADMAP.md), which remains authoritative for milestone outcomes and prerequisites. The [schedule](SCHEDULE.md) owns teaching cadence and forecast assumptions; dated commitments remain in the learner's tracker. The [glossary](GLOSSARY.md) defines terms. Original design research is historical rationale, not the current operating policy.
