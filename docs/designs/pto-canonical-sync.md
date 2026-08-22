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
The adapter has a dedicated `CANONICALSYNC` mode. Precise GM subranges and
CanonicalSync-specific operation mappings are not exposed to legacy
InsertSync, so enabling this implementation cannot silently change a legacy
InsertSync plan.

Private functions marked `pto.tileop.helper` or
`pto.ptodsl.subkernel_helper` are implementation bodies, not independently
scheduled kernels, so the pass does not analyze or mutate those bodies. Their
call sites remain schedulable compound operations whose pipe and precise
read/write contract come from the helper attributes.

An entry function whose body consists only of calls to private
`pto.kernel_kind` functions is a kernel-dispatch wrapper. CanonicalSync skips
that wrapper and analyzes each leaf kernel function independently. A wrapper
containing any other non-terminator operation is not exempted.

## Analysis model

Every schedulable macro phase is a node with:

- its source operation and stable issue order;
- its concrete pipe;
- read and write effects; and
- physical address-space, range, root, and multi-buffer information.

The memory-effect adapter applies the following operation contracts:

- scalar loads/stores and `pto.comm.tnotify`/`pto.comm.twait` are `PIPE_S`
  payload accesses;
- `pto.set_quant_vector` reads its scaling tile on `PIPE_FIX`;
- global-entry `pto.talloc`, `pto.tpush`, and `pto.tpop` are scalar-side TPipe
  ownership operations, while `pto.tfree` is additionally modeled as a release
  write so it cannot precede the last asynchronous payload use;
- `pto.set_validshape`/`pto.get_validshape` and lowered TPipe initialization
  update descriptors rather than payload bytes and are not memory nodes; and
- `pto.cmo.cacheinvalid` remains an explicit producer-placed memory-consistency
  boundary. CanonicalSync preserves it but does not use it to infer payload
  dependencies or to prove completion of another requirement.

Memory effects attached to scalar SSA operands are not physical-memory
effects. They therefore do not require a recovered `BaseMemInfo`; pointer,
tile, multi-tile, tensor-view, and partition-view operands do.

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

### GM range and no-alias contract

GM pointers and views retain their function-argument provenance. CanonicalSync
tracks constant `pto.addptr` and scalar-access offsets in bytes and the
constant shape, stride, offset, and size operands of `pto.make_tensor_view`
and `pto.partition_view`. Contiguous regions are represented exactly. Static
strided regions are represented as a bounded union of equal-sized intervals;
when that representation would be too large, a conservative bounding interval
is used. Dynamic, negative, or overflowing address arithmetic remains an
unknown range and therefore cannot prove two same-root accesses disjoint.

By default, distinct GM arguments are `MayAlias`. A producer may state a
pairwise no-alias contract with the function attribute:

```mlir
func.func @kernel(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                  %arg2: !pto.ptr<f32>)
    attributes {pto.noalias_pairs = array<i64: 0, 1, 0, 2>} {
  // ...
}
```

Each adjacent pair is an unordered pair of function-argument indices. The
contract promises that, for one invocation of the function, every GM byte
accessed through either argument and its provenance-preserving derivations is
disjoint from every GM byte accessed through the other argument and its
derivations. The contract is pairwise, not transitive: the example says
`arg0` does not alias `arg1` or `arg2`, but says nothing about the relation
between `arg1` and `arg2`.

`pto.addptr`, tensor/partition views, pointer casts, and an immediate
`pto.ptrtoint`/`pto.inttoptr` round trip preserve argument provenance. A raw
`pto.inttoptr` does not. It may alias every range in its address space even
when the function has `pto.noalias_pairs`.

CanonicalSync rejects an attribute with the wrong type, an odd number of
indices, an out-of-range index, a self-pair, a repeated unordered pair, or a
pair naming a non-GM-pointer/view argument. The attribute is a static compiler
contract and has no runtime check. A frontend must emit it only when the
promise is valid; otherwise generated synchronization may be insufficient.
Unannotated distinct GM arguments remain `MayAlias`.

The recurrence scan checks increasing distances through the physical slot
count and retains the first aliasing distance for each hazard kind. This also
captures reuse such as `consumer(i) -> producer(i + N)` for an `N`-slot ring;
checking adjacent iterations alone is not sufficient. A recurrence is removed
when exact constant `scf.for` bounds and a positive constant step prove that
the loop executes at most its iteration distance, because no source/target
instance of that recurrence can then occur. Dynamic bounds, unknown steps, and
other loop forms remain conservative.

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

Same-pipe issue order may precede a completion edge because completion of a
later operation on that pipe also completes the earlier operation. Cross-pipe
sequencer order cannot participate on either side of a completion proof:
issuing work on another pipe does not transfer the original operation's
completion obligation, and a wait stalls only its destination pipe rather than
the global sequencer.

Before the general coverage check, same-block requirements in one directed
pipe domain use an exact dominance rule. A completion from a later producer to
an earlier consumer covers a requirement from an earlier producer to a later
consumer because both sides are connected by guaranteed fixed pipe order. The
general reducer then searches one tagged adjacency graph and ignores removed
requirement edges per query; it does not rebuild the dense graph for every
candidate.

Forward dependencies are reduced on the single-iteration graph with a
requirement-specific structured execution scope. A proof may cross nested
regions, but only through operations guaranteed to execute whenever that
requirement's source and target execute. Consequently, a completion in one
`scf.if` arm or a potentially zero-trip loop body cannot discharge an outer
requirement. Distance-`d` recurrences use the same rule on a `d + 1`-copy graph
representing the relevant iteration occurrences. Fixed and retained forward
completion edges are replicated in each copy, while same-pipe order crosses
only adjacent copies. Control conditions are tagged by copy, so a branch fact
from iteration `i` is not reused in another iteration.
For forward dependencies, an `scf.for` body is guaranteed to execute when its
lower bound, upper bound, and positive step are compile-time constants and the
lower bound is strictly less than the upper bound. This fact does not remove
conditions from nested zero-trip loops or conditionals. Dynamic loops and
`scf.while` remain potentially zero-trip.

Inner-loop forward synchronization is fixed before reducing an enclosing
loop. Inner-loop recurrence synchronization is not summarized as ordinary
completion for an enclosing recurrence. Its prime, per-iteration handshake,
drain, or recurrence barrier provides pipe-specific completion only at precise
loop boundaries; an unconditional source-to-target edge in each enclosing-loop
copy would be stronger than the emitted protocol. Shorter-distance recurrence
protocols are likewise not summarized while reducing a longer-distance
recurrence.

Before reduction, CanonicalSync preserves an immutable copy of every feasible
forward and recurrence completion requirement. Statically impossible counted
loop recurrences are discarded first. The reduced dependency graph still
constructs the conservative baseline, but mechanism selection cannot erase an
original obligation merely because a provisional barrier covers it.

The final recurrence verifier replays the reduction on a `d + 1`-copy graph.
It substitutes completion edges certified from the final non-recurrence
barriers/events and from exact retained recurrence protocols. Recurrence event
identity includes source, target, loop, distance, width, slot expressions, and
anchors. This prevents a different token age or multi-buffer lane mapping from
standing in for the required recurrence.

A non-recurrence barrier contributes completion edges from operations on its
pipe that are guaranteed to issue before the barrier to operations guaranteed
to issue after it. A conditional or loop-local barrier is usable only when its
execution context is implied by the edge endpoints. A barrier in a statically
positive-trip `scf.for` body may cover an outer forward requirement; barriers
in dynamic or zero-trip loops may not.

## Synchronization plan

A retained same-pipe requirement initially becomes a pipe barrier unless the
target architecture provides the required same-pipe completion guarantee. A
retained cross-pipe requirement initially becomes one set/wait event:

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

### Physical-slot ownership analysis

CanonicalSync analyzes reusable local-memory ownership before selecting a
synchronization mechanism. An ownership cycle records:

- a producer and consumer pipe;
- each lane as a sorted bundle of exact physical `(space, address, size)`
  ranges;
- the producer and consumer nodes for every use;
- write-acquire, ready, read-acquire, and release anchors; and
- every mutually exclusive structured-control path through the cycle.

Lane identity is physical rather than SSA-based. This lets the analysis follow
address reuse introduced by memory planning while refusing unknown ranges or
overlapping lanes. Paths may use the same lanes in different orders, as in a
parity branch. Every path must account for every lane, and alternative
consumers are accepted only when both arms access the same physical bundle.

The first client recognizes L0 operand ownership from exact `LEFT` and `RIGHT`
bundles produced on `PIPE_MTE1` and consumed on `PIPE_M`. It supports any
number of disjoint lanes and one or more producer operations whose combined
writes exactly match a consumer bundle. The analysis result is independent of
protocol selection and is available through the plan printer's `ownership`
view. L1 and L0C ownership use the same representation but require separate
role-specific discovery before they are enabled.

An analyzed cycle is only a protocol candidate. CanonicalSync emits its ready
and release events only after token-protocol verification, exact recoloring,
and whole-plan completion coverage succeed.

### Barrier optimization

CanonicalSync considers non-recurrence barriers in deterministic order. For
each barrier, it evaluates the complete mixed barrier/event plan rather than an
isolated edge. Candidate mechanisms include original cross-pipe requirements
hidden by dependency reduction and bounded round-trip protocols. A round trip
can serialize an operation on the barrier's pipe before an existing incoming
ready event on another pipe, producing a completion-qualified path back to the
later same-pipe operation.

The bounded search uses at most three additional events and retains the best
eight partial bundles by uncovered reduced requirements, event lifetime, and a
stable signature. It restarts from the first remaining barrier after each
successful replacement so a newly introduced protocol can enable another
replacement. Every searched state is recolored exactly. A barrier is removed
only when the complete candidate fits the hardware event budget and a
whole-plan check covers every immutable forward and recurrence requirement.
Under event pressure, complete recurrence ownership protocols are pruned only
in an over-budget directed domain and only until exact coloring fits. Each
removal rechecks that the remaining initialize/wait/set/drain protocols cover
every immutable loop-carried requirement.
Forward events crossing an `scf.while` before/after boundary carry their final
drain in both code generation and event lifetime. The emitted plan is verified
again after event-scarcity repair.

This stage deliberately uses an unweighted structural objective: removing a
barrier is preferred once correctness and event feasibility are proven, and
candidate bundles of equal depth prefer shorter event lifetimes. A calibrated
latency model may later choose to retain a cheap barrier instead.

Recurrence barriers are intentionally excluded from this first replacement
step. Removing them requires a complete loop protocol, including initial
credits, per-iteration waits and sets, slot/parity mapping, and final drains.
Unknown or conditional structures retain the conservative barrier. Calibrated
latency-based selection also remains a separate follow-up.

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

When the fixed plan exceeds that budget, CanonicalSync runs a bounded,
deterministic beam search around a maximum-overlap event clique. The initial
scarcity transformation coalesces forward width-one events whose producers and
consumers are all in one block. A group with producer endpoints `p_i` and
consumer endpoints `c_i` becomes one stronger event from the latest `p_i` to
the earliest `c_i`. This can reduce overlap by deliberately serializing work.

Every candidate is recolored exactly and is accepted only when
completion-qualified reachability proves that its coalesced event covers every
event it replaces. Candidates are ordered by remaining color overflow and then
by the sum of the weighted critical paths in the affected blocks. The latency
graph keeps one vertex per CanonicalSync node. Fixed pipe order and event edges
are finish-to-start constraints, and a vertex has one scalar weight equal to
its saturated compute-weight plus transfer-weight sum. The initial uncalibrated
model assigns unit compute weight to non-transfer nodes and unit transfer
weight to MTE/FIX nodes; it therefore measures critical-path operation count,
not hardware cycles. A target model can replace those two components without
changing the graph or search.

The previous structural cost remains a deterministic tie-breaker: it is the
total block-local source-pipe distance moved later plus the total block-local
target-pipe distance moved earlier. Event count and a stable group signature
break further ties. The beam retains at most 16 states per depth.

Recurrences, dynamic-width events, forward drains, and events crossing
structured-region boundaries are not coalesced. If those events keep the plan
over budget, the pass fails without changing the IR. Calibrated target weights,
loop-frequency composition, and costed barriers remain future extensions;
scarcity repair does not alter the canonical dependence construction.

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
