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

For a trusted ABI that guarantees every distinct GM pointer or view argument
accesses disjoint storage, callers may opt into
`--canonical-sync-assume-distinct-gm-args-noalias` or the corresponding
`assume-distinct-gm-args-noalias=true` pass option. The mode is equivalent to
an implicit no-alias pair for every pair of distinct GM arguments. It does not
separate views or pointers derived from the same argument, and it does not
assign argument provenance to a raw `pto.inttoptr`. The option has no runtime
check and is unsafe unless the caller enforces the stated ABI contract. The
annotation-independent conservative requirement universe ignores both this
mode and explicit `pto.noalias_pairs`; only the active requirement set applies
them.

The recurrence scan checks increasing distances through the physical slot
count. The annotation-independent conservative requirement set records every
aliasing distance for each hazard kind. The active set records only the first
distance that remains after applying proven no-alias facts, and is therefore a
subset of the conservative set by construction. Recording the conservative
distance frontier before active filtering also captures reuse such as
`consumer(i) -> producer(i + N)` for an `N`-slot ring without allowing a
no-alias annotation to change the candidate universe. A recurrence is removed
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
parity branch. Round-trip paths must account for every lane, and alternative
consumers are accepted only when both arms access the same physical bundle.
An alternating-prefetch cycle instead records distinct consumed and produced
lanes plus its initial ready lane and initially free lanes.

The first client recognizes L0 operand ownership from exact `LEFT` and `RIGHT`
bundles produced on `PIPE_MTE1` and consumed on `PIPE_M`. It supports any
number of disjoint lanes within the supported multi-buffer representation and
one or more producer operations whose combined writes exactly match a consumer
bundle. The emitted event protocol still consumes one event ID per lane in
each of its ready and release domains. The default eight-ID hardware budget
therefore limits an event-backed ownership cycle to at most eight lanes when
the domain has no reserved IDs. The analysis result is independent of protocol
selection and is available through the plan printer's `ownership`
view. L0C ownership uses the same lifecycle for an exact accumulator slot:
`PIPE_M` signals readiness after the complete nested compute region,
`PIPE_FIX` waits before writeback and releases the slot afterward, and the next
outer iteration waits before reusing it. Prime and drain actions make the token
lifecycle complete for zero-trip and final iterations.

L1 ownership has two protocol forms. A regular `MAT` cycle groups exact slots
that are loaded by `PIPE_MTE2` and completely consumed by `PIPE_MTE1` in each
path. An alternating-prefetch cycle recognizes a two-arm parity loop with
unit step, `(iv rem 2) == 0` path selection, one initial load, and next-slot
loads guarded by `iv + 1 < upper_bound`. It verifies every exact slot access,
separates same-iteration load/use slots from parity-shifted prefetch slots, and
emits explicit token-state transitions for continuing and final iterations.
The initial credits and final drains are guarded by the loop's proven
non-empty condition, so a zero-trip loop neither produces nor consumes an
ownership token. Producers and consumers must execute directly in their
recognized parity or continuation region, and recursively nested accesses to
managed slots before the loop are rejected unless they form the unique initial
load. Unknown addresses, missing path accesses, non-alternating predicates,
indirectly guarded actions, and unguarded final prefetches are rejected rather
than approximated.

An analyzed cycle is only a protocol candidate. CanonicalSync emits its ready
and release events only after token-protocol verification, exact recoloring,
and whole-plan completion coverage succeed. Ownership candidates are evaluated
transactionally from the pre-ownership barrier/event plan. A candidate that
makes the complete optimized plan infeasible restores the last feasible plan
rather than turning an otherwise compilable kernel into an event-scarcity
failure.

Discovery also recognizes two hierarchical ownership roles:

- `MAT` slots written on `PIPE_MTE2` and read by `PIPE_MTE1`, where all reads
  of one slot may be nested in a common extraction loop; and
- an `ACC` slot written by `PIPE_M` and read by `PIPE_FIX`, where the complete
  producer sequence may be nested in a common compute loop.

L0C and L1 cycles are protocol candidates. Each cycle has a stable identity,
and any duplicate or physically overlapping cycles in the same or nested loop
scope are rejected before protocol selection.
Ownership protocol pairs are retained or removed by cycle identity rather than
by loop, so independent roles in one loop cannot be silently coupled.

Every synthesized ownership cycle contains one explicitly tagged `ready`
event and one `release` event. Verification binds both events back to the
discovered cycle and checks their complementary pipe directions, width, loop
scope, action lifecycle, token-state traces, and the exact complete set of
lane-local completion endpoints and recurrence distances before allocation or
emission. Protocol construction is shared by synthesis and verification so
the accepted action/completion graph cannot drift from the emitted one. For a
verified pair, completion coverage may derive the full round-trip order from
the last use of a physical lane in one iteration to the first use in the next.
This lets the pair discharge same-pipe recurrence barriers on either endpoint
pipe without treating an incomplete or malformed pair as a proof.

An alternating-prefetch pair uses a narrower path-sensitive proof. The
coverage verifier composes the initial ready completion with the first release
completion to summarize the preheader producer-to-first-reuse order across the
two event domains. It also uses the recognized `(iv rem 2) == 0` predicate and
unit-step loop to reject an ownership-managed recurrence whose endpoint
combination cannot execute at the stated iteration distance. The exemption is
limited to memory hazards for which both endpoint nodes and every aliasing
physical access belong to the verified cycle. Unrelated storage effects on the
same operations retain their original requirements and barriers. These facts
are available only while the complete ready/release pair is present and passes
ownership-pair verification. Coverage also revalidates that the recorded
regions are respectively the then and else arms of the recognized loop's exact
parity branch; path-vector order alone is not a proof.

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

Recurrence barriers are intentionally excluded from synthetic replacement
search. A verified ownership protocol may still remove one directly when its
complete initialize/body/drain lifecycle and, for alternating prefetch, exact
path-sensitive facts cover every associated recurrence requirement. Unknown or
conditional structures retain the conservative barrier. Calibrated
latency-based selection also remains a separate follow-up.

## Fine-grained barrier replacement direction

This section records the intended extension boundary. It is a development
direction, not a claim about mechanisms already emitted by the pass.

### Mechanism boundary

An event cannot directly replace a same-pipe requirement `P -> P`; event
protocols require distinct source and target pipes. A replacement must instead
prove completion through another pipe. For a forward requirement this may be a
round trip:

```text
P: A --set------------------ wait -- B
Q:      wait -- operation -- set
```

For physical-slot reuse across loop iterations, a valid replacement is a
complete ownership lifecycle:

```text
prime free(slot)
P waits-free -> produces -> sets-ready
Q waits-ready -> consumes -> sets-free
next iteration reuses slot
drain remaining tokens
```

CanonicalSync currently removes a barrier when existing completion edges cover
it, synthesizes bounded forward round trips, or substitutes a verified L0, L1,
or L0C ownership pair. The resulting plan is sound and event-feasible, but it
is not proven globally minimal and barrier removal alone is not a performance
objective.

A pre-extension device campaign at `9d583686` demonstrated the distinction. On
the historical GEMM, CanonicalSync emitted 83 set/wait pairs and 7 barriers,
while InsertSync emitted 45 pairs and 21 barriers; CanonicalSync was 51 percent
slower. This measurement predates the latest L1 and L0C protocol changes, but
it proves that fewer barriers can lose when they are replaced by too many
dynamically executed event operations or by waits at expensive anchors.

The follow-up campaign at `1ad29fb53` closed correctness and mechanism
validation for path-sensitive L1 recurrence coverage. It removed exactly two
inner-loop `PIPE_MTE2` barriers, but changed latency by only 11 of 109,028
profiled cycles and tied the parent on silicon. The final plan still executed
87 sets and 87 waits, compared with 47 pairs for InsertSync and 51 for the
manual kernel. Set and wait instructions accounted for 48 percent of the
profiled cube-core cycles.

The event-plan decomposition localizes most of this gap. Sixteen synthetic
forward barrier replacements each contain an atomic two-event round trip
through `PIPE_MTE1`, for 32 event objects in total. They serialize consecutive
`tmatmul` or `tmatmul.acc` operations on `PIPE_M` after the ownership events
have already been synthesized. Removing those objects without another
completion proof would be unsound: A2/A3 does not generally promote
same-`PIPE_M` issue order to completion order.

The focused A2/A3 device experiment used three byte-identical MMAD chains that
differed only in synchronization:

1. a `PIPE_M` barrier between dependent operations, as the correctness
   reference;
2. no barrier and no event, to test whether the operation pair has an
   intrinsic ordering guarantee; and
3. an `M -> MTE1` set after the first MMAD whose matching wait is delayed until
   after the second MMAD, isolating the source-marker effect.

The campaign at `1ad29fb53` found that the unsynchronized and set-marker arms
both overlapped dependent MMADs by about 11 ns in the op-simulator, while only
the barrier arm serialized them. All three arms were exact over 44,800 silicon
launches for each load-bearing chain length, with a co-scheduled missing
`M -> FIX` control failing in every accepted batch. The result supports
intrinsic ordering for dependent same-L0C accumulation MMADs on one A2/A3 cube
pipe and refutes an `M -> MTE1` set as the source-completion mechanism.

CanonicalSync models this result as a direct hazard discharge, not as a
`HardwareCompletion` graph edge. The distinction is necessary: dependent
MMAD ordering does not prove that an arbitrary later event observes MMAD
completion and must not participate in transitive completion reduction. The
rule is intentionally limited to `tmatmul` or `tmatmul.acc` followed by an
in-place `tmatmul.acc` on one exact, statically known L0C interval. Unknown
addresses, out-of-place accumulation, non-accumulating destinations, A5, and
all MTE hazards keep their existing synchronization.

On the historical GEMM, applying that narrow rule preserves all 4,184 original
completion requirements and all four ownership cycles, while reducing the
plan from 42 to 10 event objects. It removes the sixteen synthetic
`M -> MTE1 -> M` round-trip bundles, so the expected dynamic action count drops
from 87 to 55 set/wait pairs, close to the manual kernel's 51. Five static
barriers remain. In particular, the two surviving `PIPE_M` recurrence barriers
precede non-accumulating `tmatmul` initializations and are outside the measured
intrinsic-ordering contract.

The immediate gate is therefore the historical hand-synchronized GEMM. Before
adding another protocol family, the latest plan must be checked for:

- numerical correctness over prime, both parity paths, steady state, and drain;
- zero avoidable steady-state `PIPE_M` and `PIPE_MTE1` barriers;
- event action counts and anchors relative to the manual protocol; and
- silicon latency against InsertSync and the hand-synchronized kernel while the
  non-synchronization operation sequence remains identical.

### Candidate protocol families

The next useful families are ownership protocols over reusable local storage,
not ownership of GM arguments:

1. **UB input slots:** `PIPE_MTE2` produces a VEC slot, `PIPE_V` consumes it,
   and `PIPE_V` releases it before the next MTE2 overwrite.
2. **UB output slots:** `PIPE_V` produces a VEC slot, `PIPE_MTE3` consumes it
   for GM writeback, and `PIPE_MTE3` releases it before the next vector write.
3. **Fixpipe output slots:** where real kernels materialize a FIX result in VEC
   before an MTE3 store, use a `PIPE_FIX <-> PIPE_MTE3` round trip. Corpus
   evidence is required before adding this specialized recognizer.
4. **N-way L1 rotation:** generalize alternating prefetch from the exact
   two-arm parity form to proven `N`-lane rotation. This requires generalized
   path selection, initial token state, final-iteration behavior, and drains;
   changing only the remainder constant is insufficient.
5. **In-place UB pipelines:** a slot that moves through MTE2 load, vector
   compute, and MTE3 store is a three-party state machine
   `MTE2 -> V -> MTE3 -> MTE2`, not two independent round trips.

The phrase "GM pipeline ownership" should be avoided. GM alias analysis still
creates correctness requirements for external memory, but the rotating owned
resource in these pipelines is the local UB slot. On A2/A3, MTE2 is the normal
GM-to-local load pipe and MTE3 is the normal local-to-GM store pipe.

### Generic protocol synthesis and proof

New recurrence-barrier replacements should be produced by a physical-slot
protocol synthesizer rather than by admitting arbitrary cyclic edges to the
forward barrier beam search. For each candidate, the synthesizer must:

1. recover exact physical lanes and every participating read and write;
2. construct a per-lane state machine with explicit owners and transitions;
3. place path-sensitive prime, body, condition, and drain actions;
4. bind each transition to concrete operation endpoints and recurrence ages;
5. verify token conservation, balanced traces, legal anchors, and complete
   ready/release or multi-party handoffs;
6. verify every immutable completion requirement against the final mixed
   barrier/event plan;
7. color all event intervals exactly, including reserved event IDs; and
8. accept the candidate transactionally so failure restores the previous
   feasible plan.

Completion reachability alone is insufficient for a cyclic protocol. It can
show that an abstract edge would cover a hazard without proving that emitted
set/wait actions cannot consume a stale token or deadlock on the first or last
iteration. Two-party additions may reuse the current ownership-pair verifier.
A three-party UB cycle requires a joint protocol representation and verifier.

The final coverage checker already computes transitive completion reachability
through retained fixed, event, and barrier edges. A proposed "multi-stage
forwarding closure" is therefore not a missing correctness mechanism. Search
may still fail to generate a useful joint candidate, but that is candidate
generation incompleteness and must not be addressed by weakening coverage.

Likewise, L1 and L0 operand stages do not share one physical buffer. The MTE1
operation consumes `MAT` storage and produces distinct `LEFT` or `RIGHT`
storage. Their event domains remain distinct. Cross-stage reduction should use
the completion graph through the common MTE1 operation; it should not merge
the two ownership states or assume that one event ID can serve both domains.

### Performance acceptance

Until calibrated hardware-cycle weights are available, a replacement should
be ranked with conservative structural signals after correctness and event
feasibility:

1. fewer dynamically executed set/wait actions in the steady-state loop;
2. waits placed as close as possible to the hazardous physical-slot reuse;
3. shorter event lifetimes and less peak event-domain pressure;
4. fewer newly introduced directed event domains; and
5. no additional serialization of an unrelated producer or consumer pipe.

The existing unweighted rule that prefers every feasible barrier removal is
not an adequate final performance policy. A later calibrated model may compare
the expected barrier stall with event instruction and critical-path costs, but
latency calibration is deferred until the historical GEMM protocol shape is
understood.

Development is intentionally staged:

1. validate the intrinsic MMAD ordering reduction on the historical GEMM;
2. explain the remaining four-pair gap from emitted actions, event domains,
   and wait anchors before changing the recognizers;
3. match the hand-synchronized GEMM with the existing protocol families; and
4. only then mine the corpus and add UB or N-way protocols with focused device
   cases.

### Joint mechanism selection

Barrier and event construction must not determine which mechanisms are
available to the optimizer. CanonicalSync therefore keeps four distinct
internal objects while leaving the public `CanonicalSyncPlan` as the final
selected output:

1. `Rmax`, the immutable conservative completion requirements computed without
   applying `pto.noalias_pairs`;
2. `R`, the active immutable requirements after applying all proven no-alias
   facts, where `R` is a subset of `Rmax`;
3. a stable mechanism universe generated from `Rmax`, structured control flow,
   and exact physical-slot ownership; and
4. a selected mixed plan containing barrier candidates and complete event
   bundles.

Reduction may mark working dependencies inactive, but neither `Rmax` nor `R`
is erased. Candidate generation is completed before a provisional barrier or
event plan is optimized. In particular, local ownership discovery is already
independent of GM aliasing because it examines exact local `MAT`, `LEFT`,
`RIGHT`, and `ACC` slots. The selector must not make acceptance of such a local
protocol depend on a conservative GM event being present in the provisional
plan.

An event bundle is the atomic unit of selection. A standalone forward or
recurrence event forms a one-event bundle. A synthetic round trip is an
explicit bundle with stable member identities and the barrier or completion
requirement it is intended to replace. The ready and release events for one
nonzero `ownershipCycle` form one ownership bundle. Ownership bundles enter the
universe only after the ownership-pair verifier reconstructs and accepts the
complete expected lifecycle. Synthetic bundles require an equivalent
whole-bundle verifier; merely counting two events with one numeric bundle ID is
not sufficient. No search, scarcity repair, or redundancy pass may retain or
delete only one member of a multi-event bundle.

After selection, the builder retains the selected bundle objects, including
their stable candidate IDs, conflicts, and completion witnesses. The flat event
vector is only an emission projection. If scarcity repair coalesces eligible
standalone events, that projection is reconciled with the retained bundle set;
final feasibility verifies the retained bundles rather than reconstructing
anonymous candidates from event tags.

Individually valid ownership bundles may still be incompatible. The mechanism
universe therefore records a stable conflict relation for candidates that
overlap physical slots or otherwise compete for one lifecycle. A selected plan
may contain no conflicting pair. Discovery may conservatively omit ambiguous
overlapping cycles, as it does today, but general candidate generation must not
silently assume that independent protocol verification proves compatibility.

One centralized feasibility function validates every complete candidate plan:

- no selected candidates conflict;
- every standalone event and complete synthetic or ownership bundle verifies;
- every action and barrier uses a legal immutable structured-control anchor;
- exact interval coloring fits the available IDs in every directed pipe
  domain, after reserved IDs are removed; and
- completion-qualified reachability covers every active requirement in `R`.

After building the invariant mechanism universe, the planner runs the existing
deterministic CanonicalSync construction on `Rmax`. If that covered plan fits
the event budget, it becomes the initial incumbent and also covers every subset
`R`. If the conservative plan is over budget, the planner constructs the
ordinary covered direct barrier/event plan for `R` and uses it as an infeasible
bootstrap state. A bounded bootstrap exchanges complete mechanisms until
centralized feasibility finds an active-`R` incumbent; the existing scarcity
repair remains the later coalescing fallback. Failure to find one remains
fail-closed; the design does not claim a complete solution to event scarcity.
Once a feasible incumbent exists, it is retained outside
every bounded frontier or beam and can never be lost through pruning. This
establishes monotonicity whenever the conservative input compiles while still
allowing a more precise annotated input to compile when bounded active search
succeeds. An optimization for `R` may remove redundant conservative
mechanisms, but failure to improve returns the verified incumbent rather than
rejecting the kernel.

The bootstrap frontier contains only protocol-well-formed states that already
cover every active requirement; event-ID overflow is the sole permitted
infeasibility. A transition adds one complete universe bundle or barrier,
protects that addition for the transition, and removes only mechanisms whose
removal preserves complete coverage and protocol validity. Conflicting bundles
are exchanged atomically. Partial states are ranked by exact aggregate color
overflow, event-lane count, mechanism count, and stable candidate signature.
The search uses the same four-round, eight-state bounds as joint selection
(one round and one candidate per class above 1,024 requirements). The plan
printer reports `bootstrap=yes` only when this path establishes the first
centralized-feasible incumbent.

Selection starts from that incumbent. A transition exchanges complete event
bundles and barrier candidates drawn from the invariant mechanism universe; it
does not search upward from an empty plan. Every complete candidate is checked
by the centralized verifier and recolored exactly. Bounded search may rank
covered exchange states by event-domain pressure, but only a feasible complete
plan can replace the incumbent.

Before joint exchange search, a deterministic redundancy sweep tries removing
one complete bundle or barrier at a time. Each removal re-enters centralized
feasibility. Individual set or wait actions are never deletion candidates.
Scarcity coalescing remains a later transformation over standalone forward
bundles and must re-enter the same verifier.

Until calibrated hardware costs exist, ranking is explicitly structural rather
than performance-optimal. Action frequency is derived from each emitted
anchor's enclosing loop ancestry and protocol lifecycle, not only from
`CanonicalEventActionPhase`: a `Straight` action anchored in a loop is still
dynamically repeated. Candidate plans are compared by one total lexicographic
order:

1. maximize the number of verified ownership bundles that cover an active
   requirement or enable a barrier removal;
2. prefer the lexicographically smallest stable signatures of those useful
   ownership bundles, resolving incompatible alternatives deterministically;
3. minimize the symbolic set/wait profile from the deepest repeated anchor
   scope outward;
4. minimize the barrier profile from the deepest repeated anchor scope
   outward, so an inner-loop barrier is not equivalent to a one-time outer
   barrier;
5. minimize wait distance from the protected reuse, total event-interval span,
   peak exact color pressure, newly introduced directed domains, and total
   barrier count, in that order; and
6. prefer the lexicographically smallest complete candidate signature.

The comparator is unit-tested independently before it controls plan selection.
A calibrated latency model may later replace this ordering.

A focused monotonicity regression is the historical GEMM with all three GM
argument pairs declared no-alias. The unannotated input fits eight IDs, so the
annotated input must also fit, preserve the four non-conflicting local ownership
bundles, and remove only synchronization that centralized feasibility proves
redundant.

Implementation is split into reviewable stages:

1. preserve separate `Rmax` and active `R` before dependency reduction,
   including whether each deduplicated dependency has an active alias witness;
2. add stable internal barrier candidates, atomic event bundles, whole-bundle
   verification, and bundle conflicts without changing emitted plans;
3. generate the invariant mechanism universe from `Rmax` and physical-slot
   structure;
4. centralize candidate-plan feasibility, construct the deterministic `Rmax`
   seed or bounded active-`R` bootstrap, and retain any verified incumbent
   outside bounded search;
5. add whole-bundle redundancy deletion and tests for missing bundle halves,
   overlapping ownership candidates, no-alias monotonicity, reserved IDs,
   deterministic ties, and forced frontier truncation;
6. unit-test the total structural comparator, then add incumbent-centered joint
   exchange search and anchor-derived structural scoring; and
7. compare the historical GEMM action profile and silicon performance against
   InsertSync and the hand-synchronized kernel before adding more protocol
   families.

The initial selector implements stages 1-6 with at most eight event-bundle and
eight barrier additions per round for four improvement rounds. Functions with
more than 1,024 active requirements use one candidate of each class for one
round until pressure-hotspot ranking can bound large candidate universes more
selectively. Stage 7 remains a device evaluation requirement; the structural
order is not a claim of latency optimality.

### Selection eviction diagnostics

The plan printer has a text-only `selection` view. With no eviction options it
lists the stable IDs, protocol kind, directed event domains, static action-site
count, and event-lane count of every selected barrier or event bundle. A
proposed mechanism set can then be removed atomically with
`evict-barrier-ids` and `evict-event-bundle-ids`, for example:

```text
pto-test-opt kernel.pto -pto-plan-memory \
  '-pto-print-canonical-sync-plan=format=text view=selection \
   evict-event-bundle-ids=1533,1658'
```

This mode does not modify selection or emission. It reports the active
requirements covered exclusively by the evicted set, including endpoint
pipes, hazard kind, iteration distance, and owning recurrence scope. For every
uncovered requirement it lists each distinct, non-incumbent universe mechanism
that restores that requirement when added alone, together with the resulting
whole-plan event-color overflow. It also reports the exact interval color count
before and after eviction for every directed domain and the available hardware
IDs after reservations.

Candidate capability is a local affected-slice fact, not a claim that the
candidate alone restores the whole plan. Multiple listed candidates can
conflict or exceed the color budget when selected together. The subsequent
affected-slice exchange search must therefore evaluate complete atomic
combinations with centralized protocol, coverage, and exact-coloring checks.
Protocol-equivalent universe entries are deduplicated for diagnostics so their
stable first ID represents the whole equivalent class. Equivalence compares
the complete emitted action, completion, trace, token, scope, witness, and
semantic conflict protocols; endpoint equality alone is insufficient. A
replacement that conflicts with any retained mechanism is not listed as an
"added alone" candidate and is left for the later atomic exchange search.

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
Text diagnostics assign stable IDs in deterministic construction order to the
immutable completion requirements and to barriers. A barrier reports its
operation-level anchor, every schedulable node belonging to that operation,
its recurrence scope, and the requirement IDs with matching endpoints and
recurrence scope that caused the barrier to be constructed. An empty anchor
node list identifies a structural anchor. Scope definitions include the loop's
stable operation order and parent scope, and recurrence requirements name their
scope explicitly. These IDs describe provenance; a barrier may cover additional
requirements transitively, which is established by the final coverage verifier
rather than by the printed provenance list.
