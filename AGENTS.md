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

## Generality is an acceptance requirement

- Example-driven work must implement a general mechanism derived from actual
  control, physical storage, occurrence, or documented target semantics.
  Avoiding kernel names or placing a recognizer in a shared helper is not enough.
- Before implementation, record the missing semantic fact, the shared analysis
  that will supply it, and the constructor query that will consume it. Resolve
  the abstraction before adding another independent recognizer. Extend existing
  analyses where appropriate and remove superseded paths when covered.
- Justify every new or changed admission gate in the design/change record:
  identify the obligation it protects, its evidence, whether it is necessary or
  only sufficient, and whether equivalent IR spelling or unrelated surrounding
  work changes its answer. Safe fallback alone does not justify a narrow gate.
- Apply documented hardware contracts across their supported scope. Do not add
  dtype or shape whitelists, or require per-dtype device qualification, when the
  contract does not require them. Preserve actual target/mode/coverage premises;
  compiler tests verify application of the contract, not its existence.
- Derive physical and control facts independently of event allocation and
  predicted optimization benefit. Scope certificates to affected lifetimes,
  occurrences and interfaces. Do not substitute whole-function, whole-engine or
  whole-cell uniformity for a local obligation without explicit justification.
- Each mechanism change must include relevant generality regressions through
  the real importer/constructor: equivalent expressions and storage spellings;
  unrelated operations, descriptor updates and control; composition with existing
  endpoint/lifetime mechanisms; and negatives where actual obligations change.
  Explain non-applicable categories. The motivating kernel passing is insufficient.
- Correctness and generality are separate acceptance checks. Preserve independent
  causal/emitted-order validation, ownership, matching, participation and rearming.
  Compare complete payload-order sets for quality claims; report event resources
  and compilation work separately. Unchanged corpus output is not evidence that
  a mechanism generalizes, and changed output is not automatically a regression.
- A temporary sufficient restriction must be labeled an unfinished experimental
  limitation, with exclusions and a replacement plan recorded. Do not silently
  substitute it for requested general implementation, enable it as a completed
  solution, or report it as general. Missing analysis is not a hardware limit.
- Before declaring a mechanism general or complete, obtain a separate generality
  review that tries equivalent representations, unrelated context and mechanism
  composition. Include the gate inventory and test evidence in the handoff to
  the reviewing agent/person. If that review has not happened, state that it is
  pending; do not silently mark acceptance complete.

## Session discipline

- Verify the branch, HEAD, and working tree before changing code.
- Preserve useful experimental artifacts outside the source tree and link them
  from `HANDOFF.md`; do not make the conversation their only record.
- Run focused checks while iterating, then the validation listed in the handoff.
- Do not run sanitizer suites unless the user asks for them.
- Keep local builds and tests within the machine-wide two-worker limit.
