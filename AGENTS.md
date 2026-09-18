# OAHS collaboration instructions

For work on the OAHS selected synchronization constructor, read these files
before editing code:

1. `HANDOFF.md` for the active milestone, current evidence, and exact next task.
2. `docs/designs/oahs-todo.md` for parked work and acceptance criteria.
3. The relevant design source: `docs/designs/oahs-analysis.md` or
   `docs/designs/oahs-selected-plan.md`.

Treat the repository as project memory and the conversation as the current
working set. Work on one coherent milestone at a time. At a milestone boundary,
update `HANDOFF.md` with the final behavior, validation, remaining uncertainty,
and next action so a fresh session can continue without the old discussion.

## Current engineering policy

- Improve synchronization-plan quality before optimizing construction time.
- Preserve distinct source publication boundaries and consumer deadlines.
- Do not enlarge source prefixes merely to reduce command counts.
- Do not add a GEMM-name or opcode-sequence recognizer. Qualifiers must describe
  general control, storage, occurrence, or target semantics.
- A selected transfer grants credit only after actual causal propagation.
- Region boundaries do not reset storage histories, event occupancy, or
  consumption knowledge.
- Keep the native ACC ordering exception access-scoped. It does not complete an
  M operation or release its operand storage.
- Avoid a general post-construction refinement/search pass. Prefer preventing
  unnecessary synchronization during construction.
- Final acceptance uses the unchanged independent causal and emitted-order
  checks. Command counts alone are not success criteria.

## Session discipline

- Verify the branch, HEAD, and working tree before changing code.
- Preserve useful experimental artifacts outside the source tree and link them
  from `HANDOFF.md`; do not make the conversation their only record.
- Run focused checks while iterating, then the validation listed in the handoff.
- Do not run sanitizer suites unless the user asks for them.
- Keep local builds and tests within the machine-wide two-worker limit.
