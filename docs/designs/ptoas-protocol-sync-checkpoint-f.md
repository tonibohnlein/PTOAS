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
only a selected visibility edge can discharge that obligation. Only scalar
same-pipeline issue ordering is intrinsic; asynchronous same-pipeline WAW and
WAR effects require targeted completion. Mutually exclusive control arms are not
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

## Commit 13 direct repair

`--protocol-sync-direct-repair` evaluates the empty selected world, plans
physical recipes for every eligible residual, re-evaluates the exact selected
world, and mutates only after the result is complete. It requires explicit
`--pto-arch=a2` or `--pto-arch=a3`; both select the shared
`Npu2201A2A3` capability profile. It is mutually exclusive with every legacy
synchronization mode.

The strict direct subset accepts same-core, same-block, exactly-once intervals
with exact single-phase stages. Same-pipe intervals share a targeted barrier;
legal cross-pipe intervals share one directed set/wait frontier. Terminal
physical phases in the same physical section, or in the flat function, share
one `PIPE_ALL` exit drain. Interval stabbing is deterministic and runs in
`O(n log n)` per pipe/control group.

Event allocation is independent by physical core and directed pipe domain. It
uses only compiler IDs `0..5`, avoids imported reservations, and clears every
tentative ID if the whole plan cannot be allocated. A separate verifier
reconstructs coverage, frontier extrema, target legality, allocation, concrete
placement, event pairing, and exit-drain placement on staged IR. The outer pass
clones the complete module, so a failure in any function leaves every original
function unchanged. The low-level in-place helper is restricted to that
disposable staging module and does not promise local rollback. Internal planner
invariants are never eligible for legacy fallback.

Recurring intervals, cross-core edges, non-exact stages, unsupported event
directions, ordered-memory, ACC, visibility, unknown-alias, and other
protocol-shaped obligations fail closed. Whole-function legacy fallback
remains available and is explicitly attributed in statistics.

## Commit 14 mixed selection

`--protocol-sync-mixed` constructs complete alternatives rather than fixing a
protocol prefix. It evaluates direct-only, channel-scoped `OneShotPublish`,
ReadyRelease, and compatible combined-protocol worlds. Each alternative is
interpreted independently, receives a freshly derived residual direct plan,
is reverse-deleted, and is dry-run allocated before it can compete. The broad
Checkpoint-D adjacent-phase total-order plan is retained as a diagnostic
baseline but is not an optimizing mixed-mode candidate. ReadyRelease may
coexist with exactly-once physical phases outside its carrier loop; no
additional physical phase may execute in that loop.

One-shot candidates are atomic publication/acquisition recipes. Disjoint
storage roots share a candidate only when their exact physical source and
target frontiers match. The initial structural comparison is lexicographic:
generated event pairs, necessary targeted barriers, event pressure, static
selectable actions, then a deterministic protocol tie-break. Function and
physical-section `PIPE_ALL` exit drains are fixed correctness costs: they are
reported separately and never affect alternative selection.

Candidate-set search is deterministic and bounded rather than exponential. It
evaluates direct-only and every singleton protocol world first, then adds each
remaining protocol in stable ID order when the newly interpreted, repaired,
and allocated complete world improves the structural key. If no singleton is
complete, the complete compatible protocol set is evaluated as a recovery
seed. This is the documented V1 heuristic, not a claim of global set-cover
optimality. Crucially, every protocol-set change regenerates its residual
obligations and direct candidates; reverse deletion never tries to replace a
removed protocol with a stale direct candidate pool.

Within every alternative, the selector rebuilds and interprets the complete
world after removing each direct candidate, each whole `OneShotPublish`, and
the whole ReadyRelease protocol in reverse order. A deletion is retained only
when the resulting world is still complete. Protocol actions are never split.
The verifier reconstructs the chosen cost, reruns complete-world selection,
repeats essential-candidate checks, and requires deterministic deletion
accounting. Statistics distinguish attempted/feasible worlds, selected event
pairs and barriers, and fixed exit drains.

Allocation is combined across the selected protocol and direct recipes through
one event-generation interference graph. Imported reservations remove colors;
recurring ReadyRelease lanes interfere for their complete prime/body/drain
lifetime. Mutually exclusive arms may reuse an ID; coexecuting generations
conservatively interfere until an explicit target-consumption proof exists.
Allocation first preserves a feasible full-pool coloring, then applies bounded
deterministic DSATUR minimization. A minimization limit retains that feasible
coloring, while a feasibility or input limit is an internal analysis-limit
result and is never reported as event scarcity. Public tuning may only lower
the immutable input, recursion-depth, and search-work safety caps. Partial
graph/search statistics are retained on every allocation status. A failure
clears the whole tentative allocation. If an optional protocol cannot
allocate, selection is retried once without it; an incomplete or unallocatable
retry uses the attributed whole-function fallback policy. The abandoned
resource-infeasible plan is counted as a rejected attempt before retry.

Materialization uses the existing whole-module transaction. One-shot,
ReadyRelease, and direct actions carry disjoint ownership tags, each component
verifier accepts only its own tags, and a final mixed verifier rejects unknown
or multiply owned generated synchronization. The initial mixed surface still
rejects multiple ReadyRelease candidates, extra recurring physical phases,
the selected ReadyRelease family being accessed outside its carrier loop,
visibility protocols, ordered/ACC effects, cross-core repair, and unsupported
event directions. Unrelated outside-loop local families are not interpreted as
part of the ReadyRelease protocol.

## Concrete emitted-IR verification

Mixed emission now has a second semantic trust boundary after component
materialization. It rebuilds the structured schedule, stages, storage
timelines, and channels from the staged function, then derives completion,
reclamation, token, and exit effects from the concrete PTO synchronization
operations. It does not read selected candidate coverage, direct-repair
incidence, cached token certificates, planner event assignments, or
`pto.protocol_sync.*` diagnostic tags.

Static set/wait generations are paired by concrete direction, event ID,
control block, and lifetime. A second set before the matching wait is rejected;
coexecuting generations conservatively require distinct IDs until a target
consumption proof certifies safe reuse. Targeted barriers complete all eligible
earlier work on their named pipe before later physical work, while a `PIPE_ALL`
drain must occupy the concrete function- or physical-section-exit frontier.
The verifier independently recognizes the complete ReadyRelease
prime/body/drain recipe, without consuming adjacent unrelated static events,
checks static and dynamic lane selection, proves the zero/one/odd/even token
invariant for capacities one and two, and validates the resulting recurring
event generations against target reservations.

The reconstructed world is interpreted over the newly extracted memory and
storage facts, with concrete fixed synchronization treated as modeled supply.
No unresolved memory, SSA, generation, visibility, or exit obligation may
remain. Fresh extraction means a staged storage-operand change cannot be hidden
by retaining planner tags. Focused fault injection also covers an early set, a
late wait, a wrong direction, an overlapping same-ID generation, a missing
prime, and an inconsistent depth-two lane role. Verifier-only reanalysis is
accounted in `verifier_transitions` and does not inflate planner analysis or
residual statistics.

## Post-C.7 evidence and remaining F revision

The 2026-09-05 storage-track campaign found that F has the correct completeness
boundary but not yet one canonical obligation source. Exact atom projection and
raw-pair expansion are lossless for the current linear subset, while branch
joins, loop-instance generations, dynamic-range precision, fixed supplied
protocols, and proxy/resource effects remain incomplete or separate.

Before concrete emitted-IR verification landed, reanalyzing 199 old
synchronized PTO programs recognized 5,868 fixed protocol actions but imported
zero of them into the planning selected world. The concrete verifier now
reconstructs modeled supply from emitted synchronization, closing part of that
gap. The corpus audit must be rerun against this verifier before claiming actual
old-placement coverage; planning still needs to consume the canonical typed
obligations and fixed-supply facts proposed in the
[obligation-engine amendment](ptoas-protocol-sync-obligation-engine-amendment.md).
Protocol recognizers and direct repair should both discharge that same store;
they must not maintain independent hazard universes.
