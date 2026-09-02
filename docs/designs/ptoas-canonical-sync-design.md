# PTOAS canonical synchronization design

## Scope

Canonical synchronization is an opt-in A2/A3 synchronization mode. Its first
implementation establishes a hardware-grounded dependency and demand model and
then selects a covering subset of direct hardware mechanisms.

The design separates five questions:

1. Which dynamic memory effects can conflict?
2. Which completion or visibility order does the conflict require?
3. Which direct hardware mechanism can satisfy that demand?
4. Which other demands are implied by one direct mechanism?
5. Does the final staged IR satisfy an independently reconstructed model?

The immutable demand graph answers the first two questions. Target semantics,
coverage evaluation, event allocation, and the independent verifier answer the
remaining questions without changing the graph.

The mechanical baseline through `7418323a8` passed its A3 device gate on eight
Ascend 910B2 cards: 164,000 launches completed without mismatch, timeout,
runtime error, sentinel violation, or divergent repetition. The loop-alias and
region-summary delta through `ed58734ed` then passed 96,000 candidate launches
without changing the materialized PTO, generated C++, or device binary for
cases accepted by both revisions. A2 was not tested because the available
toolchain and cards expose only the combined A2/A3 target class. Selection and
every later optimization remain outside those device verdicts until separately
gated.

## Hardware basis

The A2/A3 target table follows the NPU 2201 synchronization documentation:

- A `SetFlag<P,Q>` is issued only after preceding reads and writes on source
  pipeline `P` complete.
- The corresponding `WaitFlag<P,Q>` blocks `Q`, consumes the event, and makes
  the source completion frontier available to later work on `Q`.
- A pipeline barrier completes preceding reads and writes on its named
  pipeline before later work on that pipeline proceeds.
- Hardware event combinations marked as unavailable are not synthesized.
- Compiler-assigned event IDs are limited to 0 through 5. IDs 6 and 7 are not
  used because the static-tensor programming contract reserves them from
  ordinary compiler allocation.

The target table is explicit rather than inferred from pipe membership.

| Core | Supported event directions |
| --- | --- |
| AIC | Scalar `S` completion is intrinsic; it has no AIC event direction |
| AIC | `M -> {MTE1,MTE2,FIX}` |
| AIC | `MTE1 -> {M,MTE2,MTE3,FIX}` |
| AIC | `MTE2 -> {M,MTE1,MTE3,FIX}` |
| AIC | `MTE3 -> {MTE1,MTE2,FIX}` |
| AIC | `FIX -> {M,MTE1,MTE2,MTE3}` |
| AIV | Every off-diagonal pair in `{S,V,MTE2,MTE3}` |

AIC pipeline barriers are supported for `M`, `MTE1`, `MTE2`, `MTE3`, and
`FIX`. AIV pipeline barriers are supported for `V`, `MTE2`, and `MTE3`.
Each physical core has a scalar unit. Sequential `S` work has the target's
documented intrinsic completion ordering; this does not create undocumented
AIC `S` event directions.

References:

- [SetFlag and WaitFlag semantics](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.html)
- [Pipeline barrier semantics](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.html)
- [AIC and AIV event combinations](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/intra_core_sync_overview.html)
- [Static-tensor event-ID restrictions](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html)
- [A2/A3 cross-core set semantics](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/ascendcopapi/atlasascendc_api_07_0273.html)
- [A2/A3 cross-core wait semantics](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0274.html)

## Immutable correctness model

The graph has stable IDs and separate record classes for:

- structured regions and their cardinality;
- physical execution phases;
- memory accesses and byte intervals;
- existing physical fence effects and their target-defined drained resources;
- synchronization demands and their witnesses.

Regions preserve functions, sequences, choices, loops, and physical sections.
Each phase records its physical core and pipeline, source order, branch guard,
loop path, and optional macro phase. Accesses retain the logical alias root,
post-allocation physical intervals where available, address space, access mode,
dynamic slot expression, and provenance.

Raw physical instructions are classified through the normalized
`VPTOSchedulingSemantics` contract. A physical or observably effectful
operation without a known scheduling classification is rejected. Logical tile
allocations are explicitly classified as structural operations: they establish
alias roots but do not execute on a device pipeline. Pure structural IR may be
ignored only when purity is independently established.

The normalized record also carries the concrete issue pipe when an operation
does not expose one through a PTO op interface. PyPTO's deferred-completion
runtime adapter is the only such external call currently admitted. Recognition
requires the exact private external declaration and exact wrapper ABI; a
same-named definition, non-private declaration, or different signature remains
unclassified. The generated wrapper defines the admitted adapter as synchronous
scalar code, and the graph retains conservative read/write GM effects for every
pointer operand. Other opaque calls remain unclassified and fail closed.

The graph is frozen before mechanisms are generated. Later stages cannot remove
or rewrite a demand.

### Alias and physical-storage rules

Alias facts are propagated through allocations, planned multi-buffer
addresses, constant and dynamic slot selection, pointer arithmetic, pointer
round trips, views, reshapes, bitcasts, and selections. Known physical byte
ranges prove disjointness or overlap. Unknown ranges conservatively alias.
Distinct GM roots are also conservative because kernel arguments can alias.
An unsupported memory type, absent address space, unscoped memory effect, or
unrecoverable provenance produces an unknown access. Unknown accesses are kept
in both the planner and verifier and may alias every compatible access; they
are never discarded as an analysis convenience.

The pass runs after memory planning and before multi-buffer selection is
resolved so that it can use both planned addresses and per-use slot identity.

### Demands

Overlapping accesses produce the usual `RAW`, `WAR`, and `WAW` completion
demands. An access marked ordered by normalized scheduling semantics produces
an ordered-memory demand against every may-alias access; this preserves
volatile and atomic ordering, including read/read ordering. Generic unordered
read/read pairs do not produce a demand.

SSA results produced by a physical phase also produce completion demands when
they reach a later physical consumer, including through pure SSA operations and
structured `scf.if` or `scf.for` results. Storage handles are alias provenance,
not asynchronous SSA payloads. Unsupported effectful provenance fails closed
instead of silently dropping an SSA dependency. Memory-like loop-carried block
arguments and results are bound to the union of their initialization and
yielded backedge facts. The binding is iterated to a fixed point, so a selected
or ping-ponged handle cannot lose a possible alias merely because it traveled
through an `scf.for` argument.

A memory-like result produced by a physical phase retains its SSA-completion
dependency through pure storage wrappers. If such a physical storage result is
carried on a loop backedge, the pass fails closed until recurrence-aware SSA
completion is modeled; it is not treated as ordinary alias-only provenance.

Overlapping ACC reads on different AIC pipelines do produce an explicit
hardware ACC read/read demand. This is not a language-level data dependence; it
represents the documented hardware scheduling restriction and remains visible
in graph dumps.

GM conflicts between scalar and non-scalar pipelines require visibility, not
only pipeline completion. MTE3 and MTE2 bypass the scalar data cache; the
documented `MTE3_MTE2` SetFlag/WaitFlag relation orders a same-core GM store/load
round trip as a completion demand. Every visibility demand records three
independent requirements: direction
(`scalar-to-nonscalar` or `nonscalar-to-scalar`), fence scope, and cache
maintenance. Scalar publication requires source cache maintenance before a GM
fence. A scalar read acquiring a non-scalar publication requires a GM fence
followed by target cache invalidation. Directions that do not read stale scalar
cache state record no cache-maintenance requirement. An event cannot satisfy
visibility by itself; missing scope, cache maintenance, or ordering fails
closed. Generated cache-maintenance/fence sequences are restricted to one
physical core. Cross-core visibility fails closed until the pass models
source-core publication, an ordered cross-core transfer, and target-core
acquisition as one protocol.

For a same-core loop-carried visibility demand, the carrying-loop latch is a
conservative physical publication cut. It executes after every possible source
in iteration `i` and before scalar issue reaches iteration `i+1`. The generated
sequence performs direction-specific DCache clean/invalidate around the GM
fence, and the target model must prove that the fence drains the non-scalar
source pipe. This is a repeated DCCI plus data-barrier protocol; no event is
credited with cache visibility. The physical latch must resolve to the same
AIC/AIV core as both endpoints because each core has an independent DataCache.
Verifier cache-maintenance facts are therefore core-qualified as well as
address-qualified.

An addressed single-cache-line CMO covers only an access proved to remain in
the line containing that exact address. The current graph does not encode GM
pointer alignment or target cache-line geometry, so it can make that proof only
for an exact one-byte access. Wider accesses require whole-GM cache maintenance
until alignment and line intervals become explicit hardware facts.

Each demand retains separate source and target guards. Loop-carried demands
record a relation for every common loop: same iteration, any positive distance,
or any iteration after an outer loop advances. A recurrence family is emitted
for every possible carrying loop. Opposite arms of a choice nested inside the
carrying loop are feasible on different iterations and therefore remain in the
demand graph. Choices outside the carrying loop remain mutually exclusive. A
physical phase that accesses the same storage on successive iterations produces
a self-recurrence demand; it is not removed merely because both endpoints have
the same static phase ID. A cross-pipeline forward dependence and its reverse
reuse recurrence can use a lifecycle-complete single-lane protocol when both
endpoints are direct operations in the same `scf.for` body and both event
directions are legal. The protocol primes the reverse release event before the
loop, consumes and restores it around every ready transfer, and drains it after
the loop. When recurrence endpoints occupy mutually exclusive arms of the same
loop, a conservative boundary protocol primes both ready and release lanes,
waits for both at the loop header, restores them at the latch, and drains them
after the loop. The demand's explicit carrying loop owns this boundary
protocol, so endpoints may be nested at different depths inside it. Shapes
whose endpoints are not both descendants of that carrying loop continue to
fail closed. A same-pipeline barrier may remain inside a loop.

Every issued phase also has an exit-completion demand. Function-core work is
drained by a tagged `PIPE_ALL` before return. Work in an explicit cube or vector
section is drained by a `PIPE_ALL` at the end of that physical section, so the
barrier executes on the core that issued the work.

## Direct mechanisms and coverage

Each demand is assigned one direct mechanism:

- documented intrinsic ordering;
- a targeted same-pipeline barrier;
- a legal directed set/wait event;
- a once-only mode-2 FFTS counter transfer between ordered cube and vector
  sections when the function already provides `pto.set_ffts` setup;
- a single-lane ready/release recurrence protocol with explicit priming and
  draining;
- a carrying-loop-latch DCache/fence visibility protocol;
- an existing visibility sequence with the required cache maintenance and
  fence scope;
- the mandatory exit barrier.

Equivalent physical cuts are interned and retain all originating demand IDs for
diagnostics. Their semantics come from explicit `before(operation)` and
`after(operation)` program points, not from the first demand that created the
cut. Event sets are placed after the source frontier and waits before the target
frontier. When an endpoint is nested, the pair can be lifted to a common
once-only block. The lifted set captures every source-resource phase that may
precede its physical point, including alternative branch arms and all relevant
phases of a macro. A barrier before a macro does not complete phases inside the
macro. Ordinary event lifecycles that repeat inside a loop use the supported
ready/release protocol or are rejected. Cross-core counters remain once-only;
guarded, loop-repeated, and positive-distance cross-core demands fail closed.
Their existing `pto.set_ffts` setup must be unconditional in the function entry
block and execute before both generated counter endpoints.

An existing GM/all fence is fixed baseline supply, represented once by its
physical operation rather than once per demand. Its completion role publishes
the prefixes of the queues drained by the target contract: all resources for a
vector kernel, and MTE2, MTE3, and FIX for a non-vector kernel. Its visibility
role remains separate and still requires the demand's scope, direction, and CMO
protocol. This models a `PIPE_ALL` that is already part of an explicit vector
fence; it does not introduce a selectable internal `PIPE_ALL` fallback.
An enclosing vector or cube physical section determines the fence's execution
core before the containing function kind, matching EmitC lowering.

Coverage is evaluated as a scoreboard, not as pairwise graph reachability. A
barrier publishes its completed source prefix. A set captures the source
prefix and completion facts already imported onto the source resource. Its
wait transfers those facts to the target.

Regions are summarized bottom-up. Sequences compose child boundary transfers;
choices retain guarded correlations and form a must-transfer only when both
arms establish the same resource transfer; loops tag facts and transfers that
require at least one iteration, preserving the zero-trip path. Parents consume
these child summaries rather than rescanning logical demand endpoints.

Each coverage world is checked against two non-summary implementations: the
flat fixed-point scoreboard and a bounded structured interpreter that
enumerates both choice arms and zero, one, and two loop iterations. The bounded
interpreter is a differential oracle, not a proof for arbitrary trip counts.
If its finite state cap is reached, that comparison is recorded as
inconclusive; reaching the diagnostic bound never rejects an otherwise valid
program. A disagreement with the flat scoreboard or with an exhaustive
bounded run is a compiler error and prevents mutation.

Coverage is computed once for each unique direct physical mechanism. A
singleton world contains fixed baseline supply plus that mechanism. Its
bottom-up regional result must agree with the flat scoreboard and, when
exhaustive, the bounded structured interpreter. The resulting demand set is a
static column in the set-cover instance; later solving never invokes a coverage
oracle again.

Intrinsic order, existing fixed fences, and the required return drain are
baseline supply. Every remaining singleton barrier or complete set/wait pair
has weight one. Deterministic greedy selection chooses the column with the
largest uncovered demand set, then reverse deletion removes a selected column
when the union of the remaining cached columns still covers the universe. The
selected mechanisms are the materialized plan. Non-singleton mechanism groups
and symbolic AND/OR coverage formulas are outside this implementation.

## Event allocation

Logical events are allocated only for selected mechanisms. Allocation uses IDs
0 through 5 and subtracts event IDs reserved by hidden macro protocols for the
same directed domain. Scalar/MLIR order between an earlier wait and a later set
is not a hardware lifetime proof: the source pipeline may re-set the flag before
the target pipeline consumes it. Therefore all coexecuting generations in one
directed domain receive distinct IDs. Only once-only generations in provably
mutually exclusive control arms may share an ID, with one narrowly certified
exception for recurring ready lanes described below.

Two non-boundary ready lanes from sequential top-level loop lifetimes may
share an ID when both already have the complete reverse ready/release protocol.
For an earlier loop with source pipeline `P` and target pipeline `Q`, the
protocol establishes:

```text
ReadyWait_earlier(Q)
  -> ReleaseSet_earlier(Q)
  -> ReleaseDrainWait_earlier(P)
  -> ReadySet_later(P)
```

The first edge is target-pipeline issue order, the second is the documented
reverse SetFlag/WaitFlag transfer, and the last edge is source-pipeline issue
order between two distinct top-level loop frontiers. Consequently the earlier
ready wait has consumed the ready event before the later ready set can execute.
This proof is derived from the documented SetFlag/WaitFlag semantics; it is not
a separately documented hardware protocol. It does not use the lexical
position of `ReadyWait_earlier(Q)` as an ordering proof for
`ReadySet_later(P)`.

Boundary ready protocols are excluded because their zero-trip behavior does
not provide the same lifecycle certificate. Frontiers whose common ordering
block repeats are also excluded; a recurrence loop may be nested only when its
complete loop lifetime lies beneath a distinct, once-only function-level
anchor. The allocator proves that sequential relation from the immutable
program model. After
materialization, the event verifier independently reconstructs both recurring
protocols from tagged operations, finds the earlier reverse release drain, and
requires that concrete drain wait to precede the later ready set. If either
proof is absent, the ready generations interfere and require distinct IDs.

When selected generations exhaust a directed domain, the allocator first tries
to coalesce compatible cuts. Members must have the same direction, control
guard, physical block, and recurrence owner. One set is placed at the latest
member source cut and one wait at the earliest member target cut, and the two
combined cuts must remain ordered. This captures every member source prefix and
guards every member target suffix. For a recurring group, the combined ready
pair retains the existing primed and drained loop release protocol.

If coalescing is insufficient, an ordinary strict alternating sequence in one
physical block and under one identical control guard may be serialized with a
forward ready event and a reverse release event. The reverse direction must be
legal in the target table. A release token is primed before the first source;
each source consumes the preceding release before setting ready, and each
target consumes ready before returning the release for the next source. This
creates a hardware happens-before proof from every wait consumption to the
next set of the same key. It is not lexical event reuse. The independent event
verifier reconstructs the complete chain, checks every role, direction, ID,
guard, once-only placement, and alternating order, and represents both keys as
occupied across the whole chain when checking interference.

Both repairs may reduce overlap, so they are used only after ordinary
allocation fails. Neither adds an internal `PIPE_ALL` barrier. Unsupported
reverse directions, incompatible cuts, and remaining scarcity still fail
closed.
Cross-core mode-2 FFTS counters use their documented ID range 0 through 10.
Coexecuting cross-core generations conservatively receive distinct numeric
counter IDs, independent of their pipe direction, and exhaustion is an error.

## Independent verification and atomic mutation

Materialization first clones the function. The clone receives tagged set/wait
pairs, mode-2 cross-core set/wait pairs, targeted barriers, and exit barriers.
Before memory dataflow, an event
generation verifier independently reconstructs each tagged set/wait generation
from the emitted IR. It checks balance, direction, ID legality, control path,
once-only lifetime, set-before-wait issue order, macro reservations, and
same-key generation interference. For a recurring mechanism it independently
requires the six prime/body/drain roles, reverse ready/release directions, and
their loop-boundary and body order. It rejects any same-key coexecuting pair
unless the generations are mutually exclusive or it reconstructs the complete
reverse-release proof for sequential non-boundary ready lanes described above;
the lexical position of a ready wait on the target pipeline is never treated as
proof that hardware consumed the event before a later source set. A separate
verifier then extracts physical effects again and runs structured dataflow over
the resulting IR.

Its state contains pending effects per physical resource, completion frontiers
imported by waits and barriers, visibility facts, live event tokens, loop-carried
effects, and exit completion. It checks hazards from memory effects directly,
including ACC read/read and loop recurrences; it does not consume the planner's
demand or coverage lists. Loop transfer reaches a fixed point over the finite
state formed from the function's effects, resources, cache actions, and event
generations. Seen-state detection replaces an arbitrary iteration limit and
rejects a non-fixed transfer cycle. Fence effects are kernel-specific: a vector
kernel's fence drains its resources, while a non-vector kernel fence drains
only MTE2, MTE3, and FIX as specified by the operation contract.

Extraction ends with a separate function-wide coverage walk. Every modeled
physical instruction, normalized schedulable operation, explicit sync
operation, and unclassified effectful operation must be accounted for. A
normalized operation that reports memory behavior must yield at least one
verifier effect. This makes omissions fail closed even if the extraction
dispatch accidentally overlooks a newly added operation class.

The original function body is replaced only after both the independent semantic
verification and the MLIR verifier succeed. Any failure leaves the original IR
unchanged.

The verifier is independent of planner demand and coverage construction, but
it intentionally shares normalized scheduling semantics, alias recovery,
physical-resource resolution, macro descriptions, and the target table. Those
components are part of the current trust boundary. Host differential checks and
device microtests are required to find common-mode mistakes in that shared
front end; verifier acceptance alone is not device evidence.

## Interface and failure contract

The compiler interface is:

```text
--enable-canonical-sync
--canonical-sync-analysis-only
--canonical-sync-dump
```

Analysis-only mode implies a dump and does not mutate the function. Dumps have
stable sections for target semantics, regions, phases, accesses, demands,
mechanisms, singleton coverage worlds, the selected plan, and verification
status.

Canonical synchronization is mutually exclusive with the existing InsertSync,
buffer-ID, and barrier-all modes. It rejects A5, `pto.tassign`, pre-existing
intra-core synchronization owned by another mode, the `mte3-to-s-event0` tail
policy, unknown explicit target/device profiles, illegal event directions,
unsafe control flow, unsupported visibility, and event-ID exhaustion.
