# PTOCanonicalSync design

`PTOCanonicalSync` is an experimental, dependence-driven alternative to
`PTOInsertSync` and `PTOGraphSyncSolver`. It deliberately solves only the
fixed-schedule synchronization problem:

- the order of operations on every hardware pipe is fixed;
- local-memory addresses and multi-buffer layouts are fixed by memory
  planning; and
- the pass may insert synchronization, but may not reorder operations or
  change addresses.

The pass is selected with `ptoas --enable-canonical-sync`. The direct MLIR
entry points are `pto-canonical-sync` and
`pto-print-canonical-sync-plan`.

## Pipeline position

The driver runs the pass after `PlanMemory`, reserved-buffer resolution, and
identity-TMOV removal. It runs before `PTOResolveBufferSelect`, so planned
physical addresses are available while `pto.multi_tile_get` still exposes the
slot expression needed for rotating event IDs.

The pass is a separate implementation. It does not modify or call the legacy
`InsertSyncAnalysis`, `RemoveRedundantSync`, or `GraphSyncSolver` algorithms.
It reuses `PTOIRTranslator` only as the PTO operation/memory-effect adapter
that maps scheduled macro phases to `BaseMemInfo` physical ranges.

Private functions marked `pto.tileop.helper` or
`pto.ptodsl.subkernel_helper` are implementation bodies, not independently
scheduled kernels, so the pass does not analyze or mutate those bodies. Their
call sites remain schedulable compound operations whose pipe and precise
read/write contract come from the helper attributes.

## Analysis model

Every schedulable macro phase is a node with:

- its source operation and stable issue order;
- its concrete pipe;
- read and write effects; and
- physical address-space, range, root, and multi-buffer information.

The canonical dependence graph contains the following edges:

- fixed issue-order edges between operations on the same pipe;
- fixed issue-order edges for structured control precedence;
- forward SSA dependencies;
- forward physical-memory RAW, WAR, and WAW hazards;
- loop-carried SSA dependencies; and
- loop-carried physical-memory RAW, WAR, and WAW hazards.

Distinct local SSA roots alias when their planned physical intervals overlap.
Unknown local ranges conservatively alias. Distinct GM roots are also treated
as potentially aliasing because PTO pointer arguments do not carry a no-alias
contract. Statically disjoint multi-buffer slots do not alias within one
iteration; recurrence analysis compares the rotating physical storage across
iterations. For constant-step `scf.for`, the slot-affine comparison evaluates
the consumer expression at the recurrence distance, for example comparing
`producer(i)` with `consumer(i + 1)`.
Integer constants and accumulated offsets are range-checked during this
comparison. An expression that cannot be represented without signed overflow
is classified as unknown and therefore retains conservative synchronization.

The recurrence scan checks increasing distances through the physical slot
count and retains the first aliasing distance for each hazard kind. This also
captures reuse such as `consumer(i) -> producer(i + N)` for an `N`-slot ring;
checking adjacent iterations alone is not sufficient.

Operations in opposite arms of one `scf.if` are mutually exclusive. The
initial implementation supports structured `scf.for`, `scf.if`, `scf.while`,
and `pto.section.*` regions. It rejects unstructured CFGs, unknown region
operations, unknown calls, unmodeled read/write effects, and input that already
contains manual synchronization, manual async communication/prefetch state, or
`pto.tassign`. A rejected function is not modified.

## Completion-qualified reduction

Issue order and completion are different facts. In particular, an earlier
MTE instruction being issued before a later MTE instruction does not prove
that its memory effect has completed. Redundancy elimination therefore tracks
reachability in the product state `(node, has-completion)`.

A required edge `u -> v` can be removed only when another path from `u` to
`v` contains at least one completion edge. Completion edges are provided by a
retained set/wait pair, a same-pipe hardware completion guarantee, or a
barrier. A path containing only issue-order edges cannot discharge a memory
hazard.

Forward dependencies in one MLIR block are reduced on the single-iteration
graph. Distance-one recurrences in one block are reduced on a two-copy graph
representing consecutive iterations, with inner-loop synchronization fixed
before reducing an enclosing loop. Longer-distance and cross-region
requirements are retained: a completion in a conditional or zero-trip region
is not assumed to cover an outer path. This keeps the reduction deterministic,
polynomial, and control-flow sound.

## Synchronization plan

A retained same-pipe requirement becomes a pipe barrier unless the target
architecture provides the required same-pipe completion guarantee. A retained
cross-pipe requirement becomes one set/wait event:

- the set is placed after the producer;
- the wait is placed before the consumer;
- anchors are lifted across structured regions when only one endpoint is
  inside the region; and
- loop recurrences are primed before the loop, waited/set in the body, and
  drained after the loop when required.

For recurrences, the set is lifted to the producer region's terminator so a
conditional producer cannot leave the next iteration without a token. A
single-ID recurrence waits at loop entry. Multi-buffer recurrences keep the
wait next to the consumer and key it by slot; they use dynamic IDs only when
the producer slot expression dominates the lifted set.

Multi-buffer recurrences in counted `scf.for` loops reserve one event ID per
physical slot and emit dynamic set/wait operations selected by the producer
and consumer slot expressions. This optimization requires proof that those
slot expressions select the same lane at the recurrence distance; an unknown
relation falls back to one conservative static event. Other loop forms use one
conservative event.
Event IDs used internally by PTO synchronization macros are reserved in the
corresponding pipe-pair domain.

## Event feasibility boundary

Event IDs are independent per directed `(source pipe, target pipe)` domain.
Every event lane has one inclusive set-to-wait lifetime, and two lanes conflict
exactly when those intervals overlap. A multi-buffer event contributes one
identical interval per required ID. The conflict graph is therefore interval by
construction; the allocator colors the intervals directly in start order and
reuses the smallest available color after an earlier interval ends. This greedy
algorithm is optimal, so a failure against the configured budget proves that
the fixed synchronization plan is infeasible.

The hardware budget defaults to eight and is configurable for analysis with
`--canonical-sync-event-id-max` or the pass option `event-id-num-max`. Values
outside `[1, 8]` are rejected.

This failure point is the boundary for future scarcity handling. Possible
extensions include synchronization coalescing and costed serialization. They
are intentionally outside the canonical dependence construction.

## Inspection

Stable text output:

```bash
pto-test-opt input.pto \
  '-pto-print-canonical-sync-plan=format=text view=all event-id-num-max=8'
```

DOT output:

```bash
pto-test-opt input.pto \
  '-pto-print-canonical-sync-plan=format=dot view=dependencies' > sync.dot
dot -Tsvg sync.dot -o sync.svg
```

The printer performs analysis and allocation but does not mutate the input IR.
