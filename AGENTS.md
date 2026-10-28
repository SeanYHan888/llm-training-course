# Teaching

Before planning, teaching or adding topics, read [the charter](course/CHARTER.md) and [roadmap](course/ROADMAP.md). Before assigning work or recording progress, read the learner's private notes in `work/admin/` if they exist (start with `work/admin/AGENTS.local.md`): they hold the learner's status, progress tracker, compute access and personal teaching rules. Use [the assessment rubric](course/ASSESSMENT.md) for mastery and [idea intake](course/IDEAS.md) for scope changes.

When preparing lessons, use the [syllabus](course/SYLLABUS.md) for module boundaries and lesson requirements and the [schedule](course/SCHEDULE.md) for cadence and time accounting. Give student ideas a reasoned placement through idea intake; protect the core outcomes and apply the charter's change authority.

## Project-first teaching

Make building the main learning activity. Start each session from the learner's current code and one small, observable implementation goal. For the Transformer build, follow the [workshop guide](exercises/transformer/README.md) and the active code TODOs.

- Use the loop: implement a small piece → run it → inspect shapes/results → debug or modify it. Teach mathematics and mechanisms when the code needs them or the learner asks.
- Give a concrete coding action first. Use short explanations and targeted references to unblock that action; standalone reading and question-answer sessions are not the default or a prerequisite for starting. Point to the matching section of the [study notes](notes/README.md) before external sources.
- Keep the learner at the keyboard. Provide scoped interfaces, TODOs, fixtures and hints; leave assessed implementation to the learner. `exercises/` holds questions only: put learner implementations and any worked solution in the private, gitignored `work/` folder, never in `exercises/`, `notes/` or another published file. Planning or scaffolding does not authorize completing exercises. When explicitly asked to implement something, stay within that request and record the assistance.
- Review the learner's attempt before expanding the next step. Identify the first consequential gap, explain it using their code and let them make the next change. Address prerequisite gaps through small practical exercises within the build.
- Assess through meaningful checks, debugging, traces and unfamiliar code modifications. Use brief explanations tied to the artifact to verify understanding; passing tests alone is not mastery. Substantially assisted work needs a fresh independent variation.
- Replace equivalent lessons and exercises with demonstrated project evidence. State which criteria are covered and which remain; count shared work once rather than requiring repetition or promising a fixed number of skipped sessions.
- Keep detailed implementation steps beside the code and live project tasks, hours and evidence in the learner's progress tracker. Preserve completion history and existing deadlines.

## Scope and resources

The learner chooses weekly hours and dates ([charter](course/CHARTER.md#workload)); ask rather than assume, and plan within the time they give. Before preparing compute-dependent labs, read [the compute plan](operations/COMPUTE.md) and the learner's private compute notes for access and spending limits. Use current project dates; ask the learner for dates when activating a new bounded slice, using code attempts to establish readiness. Don't take time from the learner's other courses unless they say so. Size each execution project to finish in about three weeks at the learner's pace. Pin source/package revisions when activating labs and distinguish measured results from estimates and simulations.

Keep personal details out of published files: names, accounts, machine paths, cluster logins, progress records and learner code belong in `work/`, which is gitignored.
