# InsertSync R4 — connected storage and lane frontiers

Base: `d7d93ad47ca380881319e7fdffeb8707b8283896`,
`tonibohnlein/PTOAS:codex/insertsync-revision-r1`.
This is a local patch package, not a pushed commit.

## Scope and decision

This implements a first connected slice of sections 6 and 7 of the preceding
proposal. It is not the complete hand-tuned GEMM protocol generator.

The production InsertSync translator, general insertion, motion, redundancy
removal, allocator and emitter remain. The new opt-in refinement consumes their
actual physical accesses and emitted local events. It can delete a generated
named barrier when all underlying storage requirements remain supplied, even
when independent work on that pipe is still outstanding. Unlike the earlier
completed-prefix pruner, it does not require the entire source prefix to be done.

Unknown models, invalid/unproved baseline token transfers, unsupported shapes,
and budget limits leave the original function unchanged. They are not evidence
of a production bug and do not become new admission gates. Malformed authored
contracts and internal failure of staged MLIR verification remain hard errors.

## Mapping to requested components

| Requested structure | Implemented here | Not claimed |
| --- | --- | --- |
| Occurrence-aware accesses | Shared translated local ranges; exact contiguous full-tile UB store partitions; correlated equality guards; a supported affine loop relation with no-wrap qualification | General polyhedral access solving, dynamic slot-selector inversion, helper/queue effects |
| Lane completion | Source phase identities, acquired completion per physical pipe, event snapshots and a separate event-causality state, through a symbolic structured CFG | Intrinsic ACC order means full completion; a finished arbitrary-control-flow protocol solver |
| Storage lifecycle frontiers | Physical atom partition, may-definitions distinct from outstanding readers/writers, availability/reclamation/write-order requirements, per-region incoming/outgoing frontiers | Exact last dynamic use for arbitrary predicates; a universal single release point for every region |
| GEMM release relationships | All panel readers remain obligations; a real release can discharge panel/operand overwrite requirements; MMAD intrinsic exemptions stay requirement-specific | New L0 bundle/alternating-prefetch/outer-carry synthesis, splitting broad handoffs, recovering the measured device slowdown |

This is a barrier-refinement implementation, not a new operation-admission
system. It is intentionally located after existing code generation for the
first rollout. That bounds the changed behavior: no event construction,
renumbering, handoff motion, or resource allocation changes. The records/core
are separate from this adapter so they can subsequently participate in general
repair and precise boundary placement.

## 1. Occurrence-aware access refinement

Local physical effects come from `CompoundInstanceElement::{useVec,defVec}` and
`BaseMemInfo`. Different handles at the same address share atoms; distinct
address spaces do not. Unknown physical ranges stay unsupported by the optional
optimizer rather than being declared disjoint.

Global disjoint-argument facts use the existing `pto.gm_alias` contract and root
tracer. Relative offsets from different bases do not supply a new alias promise.

The first same-root GM refinement admits only a plain full-static-shape UB
`TStore` into a contiguous row-major partition, f16/f32, complete aligned rows,
with no forwarded or mutable descriptor. FIX, conversions, padding modes,
helper operations and unrecognized views retain their original treatment.

For the supported loop, lower bound is a nonnegative constant, step is one, and
there is one carrier loop (no guessed flattened index through nested loops).
Index additions and constant products recover a nonnegative affine byte range:

    [stride * iv + bias, stride * iv + bias + extent)

Two accesses with equal positive stride are disjoint across different
iterations when both lie wholly in their own iteration-sized block. Within one
iteration the byte intervals must be disjoint, be the same once-per-iteration
site, or lie under proved mutually exclusive equality/inequality guards. This
last rule distinguishes the two `iv % depth == slot` branches; it does not
assume arbitrary predicates agree in different iterations.

### Arithmetic safety and the runtime guard

MLIR integer/index arithmetic must not silently be interpreted as unbounded
mathematical arithmetic. All relevant intermediate additions and products are
tracked, even an expression later multiplied by zero. The initial adapter uses
a deliberately conservative signed-32-bit-safe range (also safe for 64-bit
index lowering), along with the original valid-pointer/allocation contract.

A constant upper bound within that safe range needs no runtime guard. For a
dynamic bound, only an immutable function argument is admitted. One shared
scalar predicate is computed at function entry:

    slow = upper_bound > proved_safe_bound

Every conditionally removable barrier remains at its original point, inside
`if slow`. On the fast path all proved removals apply; on the slow path the
original synchronization is retained. Multiple bound predicates are ORed.
There is no mixture of different assumptions across loop iterations.

These are **arithmetic-proof guards, not event-scarcity fallbacks**. The static
barrier sites still exist for dynamic bounds; their executed count can drop to
zero on the declared small benchmark scenarios. Report `removed` and `guarded`
separately. Added scalar comparisons/branches are synchronization control, not
changes to the original arithmetic computation or payload/storage schedule.
Their runtime overhead still needs measurement.

## 2. Lane completion without stale generation certificates

The CFG preserves choices and loop entry/backedge/bypass edges. It does not
unroll the body a fixed number of times. Source/target resources are the pipes
of one explicitly identified Cube or vector function. Mixed physical sections,
unknown pipelines, macro phases and dynamic event IDs remain outside this adapter.

For each static physical phase s, a completion bit means:

    ALL earlier executed occurrences of s have completed.

On issue(s), the bit is cleared everywhere, including pending event snapshots.
This prevents a token for an old occurrence from certifying a newly issued
occurrence of that same site. It is conservative when multiple generations
are outstanding; it adds no synchronization to make that conservatism true.

A signal captures preceding source-pipe effects and acquired completion. It
does not advance the source's completion knowledge merely because its instruction
was encountered by the compiler. A wait imports the matching snapshot. A named
barrier establishes completion of the relevant preceding source pipe. A join
intersects must facts. The backedge reaches a finite-lattice fixed point.

A separate state tracks event occupancy and causality of the latest wait on a
key. Rearming requires the source lane to know that the previous wait was
consumed, not just observe a lexically earlier wait. Event states must be valid
on all conservatively represented paths, with all local tokens consumed at exit.
Failure to prove this is `unchanged`, not a newly reported device deadlock.

MMAD's existing opt-in `discharges()` may exempt an exact ACC requirement from
explicit completion. It NEVER sets a completion bit or releases L0 operands.
The new optimization itself never deletes a PIPE_M barrier.

## 3. Storage-specific region/lifecycle interfaces

Physical local ranges are partitioned into atoms. Every atom carries:

- may-definitions (content provenance, not physical completion),
- outstanding writer frontier,
- outstanding reader frontier,
- whether an incoming generation may remain.

Reads require earlier writes. Writes require earlier writes and all relevant
readers, then replace the ordering frontier only after recording those
requirements. A conservative write does not kill the semantic may-definitions.
Sequences, choices, loops and empty regions compose these facts. Each region
exports its per-atom incoming/outgoing state, not an undifferentiated `Done(loop)`.

The native graph retains physical atom ranges and operation references. Existing
InsertSync trace debugging can print the phase/lane/atom/requirement mapping.

An independent non-atomized all-access-pair check is also required before and
after refinement. It shares the translated effect contract, but not the sparse
frontier construction. It checks ALL prior conflicting occurrences, which is
conservative but prevents a sparse frontier mistake from dropping an old reader.

## 4. Actual optimization and mutation boundary

1. Snapshot pre-existing barrier identities before ordinary SyncCodegen.
2. Import the resulting operations/events and existing translated effects.
3. Build occurrence facts, local atoms, lifecycle requirements and lane supply.
4. Establish baseline requirement coverage and event participation/rearm.
5. Try removing one owned named barrier at a time, recomputing both proofs on
   the combined candidate. Budget exhaustion discards ALL tentative removals.
6. Clone the function into a temporary module carrying the original target/data
   layout attributes. Delete/guard only accepted barriers. Preserve all original
   event commands, IDs, payload operations, views and allocation placements.
7. Verify MLIR before committing the cloned body. Never alter the source after
   a failed/unsupported analysis. After commit, the old SyncIR pointers are
   intentionally no longer dereferenced by subsequent passes.

User/fixed barriers and PIPE_ALL are never candidates. PIPE_M remains owned by
the existing dedicated target rule. No fallback can insert a broader barrier or
coalesce events. The optional test-only pass requires explicitly marked
`pto.insert_sync.frontier_candidate` barriers and does not change InsertSync's
production explicit-sync bypass.

## 5. What this can and cannot do for GEMM

A proved existing chain such as:

    old panel load -> all relevant MTE1 reads -> next panel overwrite

can make the corresponding same-pipe WAW barrier redundant. This does not say
that all later matrix work or output stores have finished. L1, LEFT/RIGHT and
ACC are separate physical storage domains; their requirements remain distinct.

The native all-pairs check may remain conservative on the frozen GEMM (for
example, from imprecise TEXTRACT footprints, guard correlations, or a token
recurrence it cannot establish). No output count is promised without running
it. The important change is that the analyses are connected to a production
transformation rather than emitted solely as diagnostics.

This patch cannot move a release back to the true final reader if the existing
planner sank it too late. It cannot split one broad handoff into independent
publications. Those require a subsequent pre-emission construction/motion patch.
Do not relabel this refinement as the full section-7 hand-tuned realization.

## 6. Validation and next gate

Host evidence, native compilation, corpus admission, asynchronous correctness,
and device timing are separate. `STATUS.json` records exactly what ran.

Native acceptance uses the original two-/three-buffer, four-use, and historical
GEMM inputs in the current repository's hash-pinned manifest. Compare the same
native compiler with this flag off/on, holding MMAD, alias and traversal choices
fixed. Do not regenerate, shrink, or substitute the inputs.

Check event command/ID sequence and original payload replay, then report pairs,
barriers per pipe, PIPE_ALL, and overflow-guarded sites separately. A smaller
static count is not required for runtime-guarded elimination; executed counts
on named scenarios are essential. The scalar replay is not asynchronous or
numerical device verification.

After that, target one remaining real release-placement gap. Move its complete
first-use/final-use/bypass/cleanup construction, using these records. Only then
integrate L0 bundles, hierarchical panels and alternating/outer-carry protocols.
No complete-protocol match must become a prerequisite for ordinary admission.

## Source basis

- User-supplied InsertSync/ProtocolSync assessment, sections 2 and 5: separate
  requirements/completion, preserve production integration, port complete
  boundary constructions rather than fragments.
- ProtocolSync `LocalMemoryAnalysis.h`, `CompletionSupply.h`, and
  `LaneFrontierAnalysis.h` at `5cd3cb24f562539e24a88d32cacf98c79842d29c`.
  The last is a read-only experiment, not a ready-made complete optimizer.
- InsertSync translator, memory model, emitter and MMAD side analysis at the
  pinned base above. Existing event/target contracts remain the source of
  legality; this patch introduces no new GM publication or cross-core rule.
- MLIR Arith dialect arithmetic contract: https://mlir.llvm.org/docs/Dialects/ArithOps/

The old proposal's prohibition on evolving InsertSync and its covering
selection objective were superseded by the subsequent explicit development
choice. They are not reintroduced by this package.
