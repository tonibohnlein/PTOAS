# InsertSync logical construction

This branch implements the approved planning-core rewrite from
`d08f18f435cea863c05ca676d26d6c7b9fec9aa7`. The old worktree and its interrupted
constructor are separate. Existing construction remains the default.

## Acceptance contract

The new path starts from unsynchronized physical PTO. It reuses qualified
physical effects, control, storage geometry, target rules and emission utilities,
but never obtains an ordinary InsertSync or lifecycle-constructor seed and never
depends on their finalizer. Requirements survive changes to their implementation.
Direct and recurring handoffs share one logical plan and one allocation interface.

The planned selector is `--insert-sync-planner=existing|logical|logical-or-existing`.
Strict `logical` must independently construct the four unchanged milestone
kernels. Hybrid fallback runs the current path on untouched input; its success
is compilation compatibility, not new-constructor coverage. Unsupported effects,
unproved ordering/participation, analysis limits and allocation failure remain
distinct from internal errors. No new frontend promises or annotations are added.

| Milestone | End-to-end gate |
| --- | --- |
| 1 | Native/reference exact-query parity and independent looping online softmax, including preloads, empty paths, selected-barrier feedback, legal participation and finite keys |
| 2 | Independent Q projection and QK; command reductions retain useful releases, and QK first-panel readiness excludes the independent second preload |
| 3 | Joint key sharing and independent historical GEMM with zero named barriers and zero PIPE_ALL; no fixed flag-count target |
| 4 | Consolidation, immutable-fact caching, strict/hybrid corpus results, and a device task pinned to the pushed implementation |

Small buildable commits are progress inside a milestone. Each receives architect,
compiler and algorithms/performance review. The next milestone starts only after
the current native gate is accepted. The first query-library commit is **not**
independent online-softmax construction.

## Exact native relation queries

The first implementation uses `LogicalSyncRelations` with the existing MLIR
Presburger library. Python/libisl remains a test-only differential reference.
The inspected LLVM source is `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`.
Campaigns record actual source/library hashes; the source revision alone does
not fingerprint a linked binary.

For relations A and B, `A ; B` follows A and then B. All operands must use the
same parameter binding and occurrence coordinate convention. Space compatibility
checks prevent dimension mismatches; the caller owns semantic identity.

| Query | Exact implementation and qualification |
| --- | --- |
| Composition | MLIR composition preserves joined integer coordinates as existential locals |
| Difference / containment | Convert the right operand to an equivalent division-local representation, then subtract; integer emptiness establishes containment |
| Latest source for each sink | `R \\ (BeforeSource ; R)`; every original sink must still have a selected source |
| Next target for each source | `R \\ (R ; BeforeTarget)`; every original source must still have a selected target |
| Advancing readiness | `H \\ (ThroughSource ; H ; BeforeTarget)` with exact executing-occurrence domains |
| Completion | Exact composition and union of established paths; early query success does not assert a fixed point |

These operations avoid `projectOut()` and the pinned union piecewise-affine
lexopt routines. `projectOut()` is not always integer-exact. The pinned union
lexopt code intersects an initially empty unbounded-domain summary, and its
piecewise-function union does not support division-bearing pieces. Relational
dominance avoids relying on either behavior. Extrema queries additionally check
domain/range preservation so that an unbounded family is not silently erased.

Possible conflicts may be overapproximated. Proved completion and safe key reuse
must be underapproximated or exact. Definite production must not exceed definitely
written bytes. Byte identity remains present during last-producer/next-overwrite
selection, and may-writes do not kill preceding definitions. These qualifications
must also reach native effect export; descriptor-full is not itself a must-write
contract.

An earlier acquisition supplies a later demand only on executions where the
matching production was acquired. The remaining executions need their own
guarded acquisition. Original loop invocation identities cannot be replaced by
lexical order or a guessed recurrence distance.

Results distinguish proved, not-established, unsupported and budget-exhausted.
The completion cache belongs to one immutable plan. A later event-reuse query
can continue an earlier partial search; changing a plan requires invalidation.
Query term accounting charges inputs, composition products and results. It is
not an interruptible budget inside MLIR normalization or an asserted wall-time
limit. The first implementation retains the existing eight-million-work default.

## Construction, realization and trust boundaries

Candidates specify publication/acquisition domains, matching, predicate
availability, absent-role paths and entry/recurrence/exit behavior before key
assignment. When a desired predicate is unavailable, try a legal later cut or a
complete unconditional-publication alternative; otherwise report unsupported.

Selected barriers contribute completion immediately. A removed barrier cannot
justify itself; exit drains additionally require retirement proof. Generated
actions have explicit ownership; authored and hidden actions remain fixed.
All streams enter one allocator from the start, initially allowing dedicated
keys. Later sharing must prove consumption before rearm and logical token
ownership. Acquisition keys follow the matching publication, not a presumed
equal loop counter. Ordinary allocation cannot add drains or broaden boundaries.

Command simplification must not add mandatory payload ordering. Any tradeoff
that adds waiting is separate and cannot pass an overlap-preservation gate.
Payload, ABI, allocations, views and original scalar computation stay fixed;
verified synchronization-only control is permitted and counted separately.

Native/reference comparison challenges the relational calculations. Fresh
emitted-IR reconstruction challenges actual effects, predicates, endpoints and
keys. Targeted operation tests and device runs challenge their shared semantic
assumptions. Agreement on a shared incorrect footprint is not independent proof.

Final reports distinguish the 19 named inputs, the 213-row native population and
the 9,754 generated variants. Each corpus shows strict construction and hybrid
compatibility separately, retaining pre-existing failures. Sets, waits, named
barriers by pipe, PIPE_ALL, executed commands, scalar overhead, key pressure,
changed blocking and compile work/time are never combined into one score.

Device tasks use remotely available exact commits and inputs, correctness before
host-wall timing, binary deduplication and balanced arm rotation. Historical
device results do not qualify new output. Known Conv2D, manual TopK and
FlashAttention exclusions remain explicit. Local acceptance and device
qualification are separate outcomes.

## Native occurrence import progress

`SyncOccurrences` imports caller-identified physical program points before any
synchronization is selected. It preserves original loop coordinates, guarded
execution domains, and lexicographic schedules while keeping physical IDs
independent of schedule order. Positive constant steps retain exact integer
congruences. Unsupported arithmetic, integer overflow, unavailable loop-local
facts, duplicate physical phases, and bounded-work exhaustion are refused.

Root-block scalar parameters identify actual SSA values. Their unmodeled
arithmetic correlations are conservatively forgotten; these parameters are not
new caller promises. Reconstruction must preserve their bindings. The existing
string-based reference exporter has not yet migrated to this shared result.

The focused native driver passed 15 cases: Boolean signedness and negation,
extreme integer bounds, rejected induction/coefficient overflow, equivalent
parity forms, nonunit steps, nested invocations, and reversed caller phase IDs.
It also checks all 1,936 ordered phase pairs of unchanged online softmax against
independently specified isl relations, including the zero-trip domain.
These results validate occurrence import; they do not establish effect
precision, handoff construction, key reuse, or milestone-one completion.
