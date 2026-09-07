# InsertSync R5 — queryable lifecycles and guarded frontier placement

Base repository: `tonibohnlein/PTOAS`, `codex/insertsync-revision-r1`.
Base source: `d7d93ad47ca380881319e7fdffeb8707b8283896`.
Prerequisite: the exact R4 archive retained in `r4-base/`, or the unchanged base
checkout. The installer accepts either and produces the same R5 source.

## 1. Audit of R4 against the requested sections

The referenced sections are **6: occurrence/access, lane-completion and storage
lifecycle interfaces**, and **7: separate GEMM readiness/reclamation frontiers**
from the design discussion. They are not sections 6 and 7 of the older full
ProtocolSync plan.

R4 was not a complete implementation of those sections. It connected a useful
fixed-point memory/event check to barrier deletion, but its public entry point
was a transformation, not an inspectable analysis. Its region states were not
generation/frontier queries, and it neither moved handoffs nor constructed their
first/last/empty participation. Those limitations are preserved in the original
R4 archive rather than retroactively relabeled.

R5 supplies concrete shared query implementations and a native consumer of them:

| Requested part | R5 implementation | Boundary that remains |
| --- | --- | --- |
| Access/occurrence relation | Typed access slices, guards with dynamic SSA scope, affine byte differences, exact finite-residue slot-map comparison, explicit unknown/budget outcomes, stable requirement/access identities | The native importer does not yet recover every symbolic view, mutable descriptor version, or dynamic-event selector. The generic slot relation API is tested directly; native integration currently uses translated ranges, the R4 qualified GM slice, and direct-IV modulo guards. |
| Lane completion | Public immutable snapshot, lane projections, guarded CFG copies, must-completion fixed point and completion-witness queries; source completion remains separate from event-action causality | Completion is conservative: a phase bit means all earlier executed occurrences of that site completed. It is not a separate exact version for every simultaneously outstanding occurrence. Unknown means no optimization, not a fabricated certificate. |
| Lifecycle frontiers | Per-atom generation/order epochs, producer, all readers, first/final readers per consumer lane, next overwrites, readerless alternatives and region exits | A returned frontier is a set, not permission to choose a single lexical last reader across independent lanes. Conservative writes do not become definite byte kills. General descriptor/queue lifecycle recovery is not claimed. |
| Section 7 placement | Producers and final readers propose earlier publication; first readers/next overwrites propose later acquisition; guarded first-use waits and last-use signals with explicit empty-loop counterparts | No new ready/free protocol generation, broad-handoff splitting, arbitrary conditional-last-use synthesis, or new allocator. Native benchmark and hardware effectiveness remain unmeasured here. |

The **shared interfaces now have executable implementations**, not placeholder
records. That is different from universal optimization coverage for all PTO IR.
Do not claim section 6 is supported for every operation/descriptor/control form,
or that section 7's full hand-tuned realization is finished.

## 2. Files and connection to the existing pass

The R4 files remain the base:

- `StorageFrontierDomain.h`: fixed-point memory/completion foundation;
- `StorageFrontierAccess.{h,cpp}`: translated access adapter and guarded GM query;
- `StorageFrontierAnalysis.{h,cpp}`: native importer and transactional refinement.

R5 adds:

- `StorageFrontierRelations.h`: `OccurrenceRelation`, `GuardFact`, `AccessSlice`,
  `SlotMap`, `RequirementWitness`, and relation queries;
- `StorageFrontierControl.h`: finite symbolic guard states and the shared
  `structuredLoopTransfer` construction;
- `StorageFrontierQueries.h`: lane projection, completion witnesses,
  generation-specific frontiers, and atomic barrier-group refinement.

`analyzeInsertSyncStorageFrontiers()` returns a read-only `StorageFrontierSnapshot`:
physical owner/context, native access and guard bindings, graph anchors, access
witnesses, per-atom lifecycle transfers, generations, and completion supply.
Clients must discard the snapshot after mutation. `Complete` describes successful
analysis in the supported domain; it is **not** a whole-kernel safety verdict.
Required coverage and event proof are separate queries.

Production still follows existing InsertSync translation, insertion, motion,
cleanup, allocation and emission. `--insert-sync-frontier-refinement` retains
barrier-only behavior; the new `--insert-sync-frontier-placement` requests
placement followed by barrier refinement. Both default off. Existing explicit-sync
bypass, report-mode coverage, WAW/overflow/slot/provenance corrections, and MMAD
configuration are preserved. No new correctness certificate is required merely
to compile an established operation.

The current refinement consumer remains post-emission. The snapshot API can now
be consumed earlier by the general planner, but this package does not claim to
have replaced its legacy `alreadySync` records or its complete insertion core.

## 3. Occurrences, guards and slot relations

An `OccurrenceRelation` identifies same-instance, known loop-distance, or ordered
but unknown relationships. A zero distance must name the actual common loop;
static operation identity alone does not establish the same execution.

Guard facts identify their SSA expression and dynamic definition scope. Two uses
of one immutable outer value can remain correlated inside a nested loop. A value
redefined by the next iteration is forgotten unless a supported recurrence
explicitly supplies its next value. EQ/NE exclusions are retained per guard state.

The native importer recognizes the actual `iv rem N`, with N in [1,16], for
supported nonnegative signed/index loops. It initializes the residue from the
lower bound and advances it by the real step modulo N. It does not replace `j % 2`
with a global inner-iteration counter. Inner invocation state is reset at the
corresponding outer transition, while invariant outer state survives.

`compareSlotMaps()` compares actual physical addresses and extents over a proven
finite residue domain. Unequal slot numbers with shifted physical mappings are
not automatically disjoint. Multi-dimensional comparisons require every relevant
occurrence displacement. This API does not guess a next reuse distance or an
unknown arithmetic no-wrap condition. It enumerates residues, not runtime trips.

The R4 runtime arithmetic guard for qualified contiguous GM stores remains.
Placement checks use only unconditional access proofs: an unmaterialized
arithmetic guard cannot justify moving an event. Barrier refinement may still
produce `if slow: original_barrier`; guarded and deleted sites remain separate
metrics.

## 4. Structured loop transfer

The shared core represents the exact control language

    empty | one | first middle* last

with a middle backedge, not a chosen unroll horizon. The native adapter uses the
same C++ `structuredLoopTransfer` template as the host boundary tests. Static
physical phase IDs are shared between the analysis copies. Emitted physical
operations and loop bodies are not duplicated.

A supported first predicate is `iv == lower`. A supported last predicate is
`upper - iv <= step`; lower is known nonnegative and step positive. On executing
iterations this avoids introducing an overflowing `iv + step` expression. The
native qualified form is signed index iteration; `unsignedCmp` is excluded.
Other predicates are conservatively partitioned rather than assumed to have a
particular truth sequence. Unknown correlation can prevent optimization.

At every physical issue, completion information for that static phase is
invalidated, including pending event snapshots. A signal for an earlier
occurrence cannot certify a newly issued one. Token occupancy and causal
consumption-before-rearm are checked separately on every represented path.

The state space is explicitly bounded. Caps include 256 physical phases, 2,048
raw graph nodes, 8,192 guard-product nodes, 128 guard variables, and 64 local event
keys in one physical core context. Exceeding a cap leaves the function unchanged;
it does not mean the kernel is invalid or that hardware event IDs are exhausted.

## 5. Storage generations and completion witnesses

A generation is an **ordering epoch** at a physical write, not a claim that a
conservative write definitely replaced every byte. Existing may-definition and
live-in state remains independent.

For each generation and physical atom, the analysis follows feasible graph paths
until the next write to that atom. It records all readers, first reads per lane,
and possible final reads per lane. For example, a V reader and an MTE3 reader both
remain in the release frontier; neither is dropped just because the other is
lexically later. A read/write phase consumes the old epoch and starts the next.
An empty branch preserves the incoming epoch and its outstanding accesses.

`completionBefore(source, target)` returns a checked must-completion fact and an
explanation cone of supporting synchronization. The cone is not a minimal path
or a shortest proof. Acceptance comes from the fixed point over all represented
paths, not the existence of one convenient path. The query's conservative claim
is explicit: all earlier executed occurrences of the source phase are complete.

The non-atomized all-access-pair check remains independent of sparse frontier
retirement. It shares the translated effect contract, not the selected sparse
requirements. Thus a lifecycle bug cannot silently drop an older reader and
become the sole basis for deletion or motion. It is not an independent device
model or proof that all legacy effect declarations are correct.

Intrinsic MMAD ordering still discharges only a qualified ACC dependency. It
never populates completion or event causality, and never releases L0 operands.

## 6. Placement that is actually implemented

Candidates are now drawn from the computed **producer/final-reader publication
frontiers** and **first-reader/next-overwrite acquisition frontiers**, on the
appropriate physical pipe. They are not generated from a function name or a
whole-loop "done" flag. Every proposal is checked against all memory requirements
and the complete event state, not only its motivating lifecycle.

### Earlier publication inside a block

Before:

    final read of A
    independent source-pipe work U
    set A_released

Candidate:

    final read of A
    set A_released
    independent source-pipe work U

The signal is not advanced before a still-required reader. Known initialization,
readiness, and release frontiers can all supply proposal anchors. No additional
physical key is allocated. The first rollout does not cross another publication,
barrier or arbitrary region through the simple within-block rule.

### Later acquisition inside a block

Before:

    wait A_ready
    independent destination work U
    consume A

Candidate:

    independent destination work U
    wait A_ready
    consume A

Another wait can be crossed as a trial, but the rebuilt requirement and token
proof must establish that doing so is valid. Publications, barriers, and regions
require separate handling.

### First-use acquisition inside a loop

For a once-per-invocation wait moved from before a loop:

    for i:
        independent work
        if i == lower: wait ready
        consumer
    independent suffix
    if loop_is_empty: wait ready
    original terminal drain

There remains exactly one dynamic wait per original execution. The zero-trip
wait is placed after the suffix only when the whole candidate permits it;
otherwise the earlier supported cleanup location is considered. An existing
terminal ALL remains the final synchronization boundary. No readiness token is
invented for an absent producer.

### Final-use publication inside a loop

For a once-per-invocation signal moved from after a loop:

    for i:
        reader
        if upper - i <= step: set released
        independent work
    if loop_is_empty: set released
    wait released
    overwrite

The original outside signal existed even on zero trips. The explicit empty
alternative preserves that participation. An empty loop with an unconditional
prefix write is not treated as though its write disappeared. The actual
completion proof must cover that path too.

This recovers earlier panel-release opportunities when the old plan sank a
signal to a loop boundary. It does not require waiting for unrelated matrix or
output work. The initial boundary adapter chooses direct body phase anchors;
general conditional-last-reader synthesis and arbitrary multilevel frontier
placement remain conservative.

## 7. Transaction and ordering contract

Only generated events passed by production InsertSync, or explicitly marked
standalone-test events, are eligible. User/fixed barriers are never removed.
PIPE_ALL and PIPE_M remain outside the new barrier-deletion candidates.

1. Establish baseline all-access coverage and token validity.
2. Clone into a temporary module preserving target/data-layout attributes.
3. Assign private identities to the original operations.
4. Compute lifecycle-guided proposals; materialize one complete trial on a clone.
5. Rebuild translated accesses and all analysis state from that trial.
6. Verify storage requirements, event consumption/rearm and MLIR.
7. Compare original non-sync identities, attributes, types and SSA operand
   identities. Newly added control is synchronization-only.
8. Retain a successful trial; otherwise discard it. Budget exhaustion discards
   the whole placement transaction. Never publish removal/motion counts for an
   uncommitted change.
9. Apply the R4 barrier refinement on the staged result, verify, and commit once.

No allocation or scarcity behavior changes. This is not a claim of global
latency optimality or a device-qualified maximum-overlap proof. Earlier
publications and later acquisitions remove particular unnecessary waiting; their
actual source-pipe scheduling and runtime impact still require target validation.
The exact existing flags and hardware key assignment are retained dynamically.

## 8. Static and dynamic accounting

First/last/empty lowering can turn one static event site into two mutually
exclusive sites for **one dynamic action**. Consequently, total static sets and
waits need not balance. Report their separate counts; do not report
`min(sets, waits)` as a pair count. On each replayed execution, sets and waits and
event-key participation still require checking; equal totals alone prove no
pairing or progress property.

`compare_benchmarks.py` now has three same-compiler arms:

- baseline (frontier flags off);
- barrier-only refinement;
- placement plus refinement.

MMAD, staged traversal, architecture, caller alias contract, and original input
hashes are held fixed. R4-only event sequence equality is still checked. R5 event
positions may change, so its comparison uses separate inventories, unchanged
PIPE_ALL count, and per-scenario dynamic set/wait count and payload replay checks.
These do not replace asynchronous verification or numerical device tests.

## 9. What remains and how to measure it

Section 6's common query interfaces are implemented for the documented domain;
**universal native semantic recovery is not complete**. Remaining work includes
arbitrary mutable descriptor versions, full helper/macro/queue phase summaries,
multiple physical contexts, native dynamic-event/slot-family import and more
precise multiple-outstanding-generation completion.

Section 7 is advanced from deletion into actual placement, but does not yet
create or split whole ready/release protocols, choose new event streams, or
replace the allocator. This package does not claim to have ported all older
CanonicalSync ownership construction or to match every hand-tuned fixture.

Native acceptance must run the unchanged frozen two-/three-buffer, four-use and
historical GEMM cases. First determine whether the new supported analysis reaches
each input, then compare exact sites and executed scenarios. Preserve all
unsupported reasons; do not reduce the denominator. Run full check-pto and the
original corpus. Qualify numerical/progress behavior and device timing separately
before making either frontier flag default.

## Sources and precedence

The attached assessment's sections 2 and 5 govern reuse: keep explicit
requirements, distinguish outstanding from completed work, and move complete
participation constructions without creating a new admission gate. The older
ProtocolSync ban on evolving InsertSync was superseded by the user's subsequent
explicit direction.

R4's exact source and evidence are preserved in `r4-base/insertsync_frontiers_r4.zip`.
The native source interfaces were checked at the base SHA above. Current official
SCF/Arith documentation was consulted for loop and integer semantics:

- https://mlir.llvm.org/docs/Dialects/SCFDialect/
- https://mlir.llvm.org/docs/Dialects/ArithOps/

No new GM visibility, cross-core synchronization, or MMAD hardware rule is
introduced by this patch. Host tests, native compiler validation and silicon
evidence remain distinct; see STATUS.json.
