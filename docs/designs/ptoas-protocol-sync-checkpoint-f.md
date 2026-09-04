# PTOAS ProtocolSync Checkpoint F

## Commit 12 scope

The first Checkpoint F milestone adds a generation-aware selected-world
interpreter. It evaluates logical synchronization effects before event
allocation or IR mutation and emits deterministic residual obligations for
effects that the selected protocols do not discharge.

Use `--protocol-sync-dump=residuals` with analysis, one-shot, or ReadyRelease
mode. Analysis mode evaluates an empty selected world and therefore exposes the
remaining obligations without changing IR. Emission modes adapt the complete
selected plan into logical effects and require interpretation to produce no
residuals before allocation proceeds.

## Selected-world contract

A selected world contains:

- whole-channel `OneShotPublish` or `ReadyRelease` claims;
- must-execute completion edges with same-iteration or proved loop-carried
  distance;
- separately qualified global-memory visibility edges;
- phase completions established before function exit.

It contains no concrete event IDs. The adapter checks channel identity,
generation identity, capacity, complete one-shot boundary paths, ReadyRelease
lane shape, unique phase ownership, and iteration distance. A malformed
selected world is an internal error and never enters legacy fallback.

## Interpreter contract

For each frozen storage generation, the interpreter tracks production,
availability to consumers, active consumers, reclamation before overwrite, and
logical ReadyRelease lane state. It also checks may-alias global effects,
opaque ordered actions, unresolved schedule facts, and terminal physical work.
Generic phase completion does not prove write-to-read global-memory visibility;
only a selected visibility edge can discharge that obligation. Same-pipeline
write ordering remains intrinsic. Mutually exclusive control arms are not
treated as simultaneous memory peers, loop-carried distance-one hazards are
checked explicitly, and exit obligations retain every feasible terminal arm.

Unresolved requirements become sparse records of these kinds:

```text
completion, reclamation, ssa-completion, ordered-memory,
acc-conflict, visibility, exit-completion, unknown-alias
```

Each record retains source and target phase when known, optional generation and
channel identity, control relation, iteration relation/distance, the explicit
carrier loop for recurring effects, and a stable explanation. Distance one is
used only when both endpoints execute directly and unconditionally in that
carrier; guarded and nested-loop recurrences remain unknown and therefore
fail closed. Physical SSA provenance is traced through pure operations,
`scf.if` results/yields, and `scf.for` results and iteration arguments.

Statistics report interpreter transitions, peak abstract states, total
residuals, per-kind counts, and interpretation time. ReadyRelease token
certificate work is reported separately. This non-forking interpreter has one
live abstract state; the peak-state counter is a state-space metric rather than
a generation or token count, and remains meaningful when later milestones add
path-sensitive state splitting. Work completed before an internal failure is
still included in the transition counters.

## Deliberate boundary

This milestone does not create direct repair candidates, mix protocol and
direct synchronization, select among competing plans, or reverse-delete
candidates. Existing one-shot and ReadyRelease emission remains limited to its
already certified complete-function subsets. The next Checkpoint F milestone
will plan targeted repairs over these residual records.
