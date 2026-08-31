# PTOAS canonical synchronization design

## Scope

Canonical synchronization is an opt-in A2/A3 synchronization mode. Its first
implementation establishes a hardware-grounded dependency and demand model and
then emits a mechanical direct solution. It deliberately does not choose a
smaller subset of mechanisms yet.

The design separates five questions:

1. Which dynamic memory effects can conflict?
2. Which completion or visibility order does the conflict require?
3. Which direct hardware mechanism can satisfy that demand?
4. Which other demands are implied by one mechanism or a group of mechanisms?
5. Does the final staged IR satisfy an independently reconstructed model?

The immutable demand graph answers the first two questions. Target semantics,
coverage evaluation, event allocation, and the independent verifier answer the
remaining questions without changing the graph.

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
| AIC | `M -> {MTE1,MTE2,FIX}` |
| AIC | `MTE1 -> {M,MTE2,MTE3,FIX}` |
| AIC | `MTE2 -> {M,MTE1,MTE3,FIX}` |
| AIC | `MTE3 -> {MTE1,MTE2,FIX}` |
| AIC | `FIX -> {M,MTE1,MTE2,MTE3}` |
| AIV | Every off-diagonal pair in `{S,V,MTE2,MTE3}` |

AIC pipeline barriers are supported for `M`, `MTE1`, `MTE2`, `MTE3`, and
`FIX`. AIV pipeline barriers are supported for `V`, `MTE2`, and `MTE3`.
Sequential `S` work has the target's documented intrinsic completion ordering.

References:

- [SetFlag and WaitFlag semantics](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.html)
- [Pipeline barrier semantics](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.html)
- [AIC and AIV event combinations](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/intra_core_sync_overview.html)
- [Static-tensor event-ID restrictions](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html)

## Immutable correctness model

The graph has stable IDs and separate record classes for:

- structured regions and their cardinality;
- physical execution phases;
- memory accesses and byte intervals;
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
not asynchronous SSA payloads. Unsupported effectful provenance and loop-carried
block arguments fail closed instead of silently dropping an SSA dependency.

Overlapping ACC reads on different AIC pipelines do produce an explicit
hardware ACC read/read demand. This is not a language-level data dependence; it
represents the documented hardware scheduling restriction and remains visible
in graph dumps.

GM conflicts between scalar and non-scalar pipelines require visibility, not
only pipeline completion. Every visibility demand records three independent
requirements: direction (`scalar-to-nonscalar` or
`nonscalar-to-scalar`), fence scope, and cache maintenance. Scalar publication
requires source cache maintenance before a GM fence. A scalar read acquiring a
non-scalar publication requires a GM fence followed by target cache
invalidation. Directions that do not read stale scalar cache state record no
cache-maintenance requirement. An event cannot satisfy visibility by itself;
missing scope, cache maintenance, or ordering fails closed.

Loop-carried demands record iteration distance. A zero denotes the current
iteration, while an explicit positive-distance marker denotes a summarized
later iteration. A physical phase that accesses the same storage on successive
iterations produces a self-recurrence demand; it is not removed merely because
both endpoints have the same static phase ID. Cross-pipeline recurrence events
are rejected until a proven repeating protocol exists. A same-pipeline barrier
may remain inside a loop.

Every issued phase also has an exit-completion demand. All returns receive a
tagged `PIPE_ALL` barrier, including returns in structured control flow.

## Direct mechanisms and coverage

Each demand is assigned one direct mechanism:

- documented intrinsic ordering;
- a targeted same-pipeline barrier;
- a legal directed set/wait event;
- an existing visibility sequence with the required cache maintenance and
  fence scope;
- the mandatory exit barrier.

Equivalent physical cuts are interned. Event sets are placed after the source
frontier and waits before the target frontier. When an endpoint is nested, the
pair can be lifted to a common once-only block. Event lifecycles that repeat
inside a loop are rejected.

Coverage is evaluated as a scoreboard, not as pairwise graph reachability. A
barrier publishes its completed source prefix. A set captures the source
prefix and completion facts already imported onto the source resource. Its
wait transfers those facts to the target. Fixed-point evaluation therefore
records singleton coverage as well as coverage that exists only when a group
of mechanisms is present.

The initial materialized plan keeps every direct mechanism. Coverage worlds are
diagnostic input for a later selection policy; they do not redefine
correctness.

## Event allocation

Logical events are allocated after demands, mechanisms, and coverage are
fixed. Allocation uses IDs 0 through 5, subtracts event IDs reserved by hidden
macro protocols for the same directed domain, and allows reuse only when
lifetimes are ordered or guards are mutually exclusive. Exhaustion is an error;
it does not fall back to a global barrier.

## Independent verification and atomic mutation

Materialization first clones the function. The clone receives tagged set/wait
pairs, targeted barriers, and exit barriers. A separate verifier then extracts
physical effects again and runs structured dataflow over the resulting IR.

Its state contains pending effects per physical resource, completion frontiers
imported by waits and barriers, visibility facts, live event tokens, loop-carried
effects, and exit completion. It checks hazards from memory effects directly,
including ACC read/read and loop recurrences; it does not consume the planner's
demand or coverage lists.

Extraction ends with a separate function-wide coverage walk. Every modeled
physical instruction, normalized schedulable operation, explicit sync
operation, and unclassified effectful operation must be accounted for. A
normalized operation that reports memory behavior must yield at least one
verifier effect. This makes omissions fail closed even if the extraction
dispatch accidentally overlooks a newly added operation class.

The original function body is replaced only after both the independent semantic
verification and the MLIR verifier succeed. Any failure leaves the original IR
unchanged.

## Interface and failure contract

The compiler interface is:

```text
--enable-canonical-sync
--canonical-sync-analysis-only
--canonical-sync-dump
```

Analysis-only mode implies a dump and does not mutate the function. Dumps have
stable sections for target semantics, regions, phases, accesses, demands,
mechanisms, coverage worlds, the mechanical plan, and verification status.

Canonical synchronization is mutually exclusive with the existing InsertSync,
buffer-ID, and barrier-all modes. It rejects A5, `pto.tassign`, pre-existing
intra-core synchronization owned by another mode, the `mte3-to-s-event0` tail
policy, illegal event directions, unsafe control flow, unsupported visibility,
and event-ID exhaustion.
