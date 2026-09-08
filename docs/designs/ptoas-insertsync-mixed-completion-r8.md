# InsertSync R8 — restore completion lost at the protocol/residual boundary

## Evidence and review boundary

The uploaded `REGRESSION_SUMMARY.md` is the empirical basis. It measures
`c3a9004b47596365f6d2a8bf08cf77492b26d571`, with target, alias contract, staged
repair and MMAD held fixed. All 88 automatic PTO/C++ invocations passed, as did
11 A3 manual PTO invocations. The nine replay-enabled fixtures preserve the
payload trace and balanced dynamic set/wait totals. This is not a 213-row rerun,
a numerical/device result, or proof of correct event matching on silicon.

The optimization result is negative:

* Only the four controls select lifecycles. Every source-derived production
  fixture, including the historical GEMM, falls back.
* One-buffer gains one recurring MTE3 barrier: 0 -> 16 executions at 16 trips.
* Four-use gains six static MTE3 sites: 2 -> 8; executed barriers 16 -> 64.
* No pair count improves. Event positions change in the selected controls, but
  the report does not provide a measured overlap or runtime benefit.

This is stronger evidence than the earlier zero-change R4/R5 result. The R6
optional path must not be promoted on the strength of selection counts or host
protocol tests. The report does not demonstrate a race/deadlock; additional
waiting and absence of production selection are effectiveness problems.

Reviewed source: the R6 branch head above, its callback and residual insertion
path, and the attached R7 package. During this review R7 landed at `c720df6bad7db78b07e97c5499f9c435f80c330f`.
The incremental contexts were checked against its formatted callback, driver,
header and unchanged InsertSyncAnalysis stage hook. Its more precise use of
`allocation.failedIdentity` in candidate retry and its `func.constant` symbol
exclusion are outside (or explicitly preserved by) the edits. This is not a
complete re-review or native rebuild of R7. This R8 package is incremental over
the R7 API and preserves that work. It is not a
wholesale R7 replacement. No repository changes have been pushed.

## Finding 1: exact-member exemptions hide real full completion

R6/R7's `removeSuppliedDependencies()` first demands an exact, equal local slice
and then asks one lifecycle certificate whether it orders those member accesses.
`exactSlice()` explicitly excludes GM. This is a sensible storage fast path,
but it is not the whole completion supplied by the selected events.

For an output slot, the already-selected complete construction gives:

    Store(g) -> publish Free -> acquire Free before Compute(g+1)
             -> Compute(g+1) -> publish Ready -> acquire Ready before Store(g+1)

That chain completes the preceding **physical store operation**, including its
GM write. The local channel's member is only the UB read. Consequently the
exact-member callback cannot discharge the GM WAW, although this construction
already orders it. When the channel events are kept out of the legacy lists,
ordinary `alreadySync` reasoning cannot see the chain either.

The new same-pipe residual barrier is therefore a predictable consequence of the
integration. We checked this mechanism with the actual C++ completion core on
the one-buffer and four-use operation shapes. That is a source/model diagnosis,
not a new native reproduction of the exact emitted sites. The uploaded summary
has no per-barrier insertion witnesses or GEMM fallback reason.

For four-use, successive stores within one slot are ordered through successive
computations. Stores into different independent local output slots are not
necessarily ordered. A completion-only change must retain that distinction; it
must not manufacture GM disjointness or acquire the next slot unnecessarily.

## Finding 2: moving analysis earlier was necessary, but not sufficient

The branch now constructs complete protocols before residual insertion and
keeps their actions atomic. That is an improvement in representation. However,
only exact-member exemptions are passed to the residual planner. The combined
actual event check occurs after insertion/allocation, too late to prevent these
extra repairs. Patterns and residuals have not been optimizing the same plan.

R4/R5 remain useful testing and placement infrastructure. Their narrow importer,
all-prior-static-phase completion model and post-emission placement limited their
native effect. R7 broadens guarded roles and retry. It does not by itself fix this
missing full-completion interface. Do not add another pattern catalogue first.

## Finding 3: GEMM fallback is still undiagnosed by the attached report

The report says `fallback`, not which stage failed. Import restrictions, guarded
roles, resource fit and the conservative combined-event check are all possible
reasons in the implementation. We do not assert that R7 or this patch fixes the
GEMM selection failure without its native candidate diagnostics.

Known helpers/macros and many unmodeled physical operations remain outside the
optional constructor. Their fallback is not evidence of invalid original kernels.
The source-level fixes made earlier in the branch (WAW, checked ranges, slot and
provenance reasoning, MMAD's separate intrinsic rule) must not be reverted to
improve the new constructor's counts.

The R6 integration also excludes `func.constant`: cloning one function into a
new module loses referenced sibling symbols. The original supplied R7 source
predates that integration fix. R8 preserves it when present and supplies the
same exclusion when absent; it never overwrites the rest of the integration.

## R8 implementation

### 1. Read-only logical completion overlay before repair

`LifecycleCompletion.h` composes the selected protocols' prime, guarded body,
readerless and drain actions with the existing guarded physical control graph.
The graph and physical operations are unchanged. Virtual keys are logical stream
identities, not concrete hardware IDs, and no runtime counter is added.

The overlay contains **no generated named barriers, ALL, or MMAD completion
exemptions**. A candidate cannot prove its own redundancy, and intrinsic ACC
ordering does not become full M completion.

The existing R5 completion transfer is interpreted once per repair stage. At each
reachable target occurrence it must prove that all earlier occurrences of a
source phase have completed. Results are intersected across every represented
path/target occurrence. An unrepresented target creates no positive query.

Reissue invalidates a source-site completion bit and saved event snapshots, as
before. This remains conservative for several independent outstanding instances
of one static phase. R8 does not claim to solve general occurrence versioning.

### 2. Selected protocols plus a qualified residual subset

At the first/combined walk, the view contains selected protocols only. At the
start of the staged same-pipe walk it may also contain already-built forward
residual set/wait pairs. Native admission requires:

* one local SET_EVENT / WAIT_EVENT pair;
* unchanged physical endpoints in the same block and source before target;
* exact presence in the actual before/after lists;
* no carried relation, compensation, dynamic selector, or multi-ID stream.

Actual list order is retained. Residual waits precede newly materialized protocol
waits at a phase; protocol signals precede residual signals, matching current
code generation/materialization. The full virtual event transfer, including
consumption-before-rearm, must be proved. Otherwise the separately checked
protocol-only view remains available. Unknown is not a positive certificate.

This is an initial **mixed-plan query**, not a complete interpreter for every
legacy recurring, hoisted or widened residual. No broad source frontier is
created merely to increase coverage. There is no singleton-cover matrix/solver.

### 3. Use full completion for same-pipe memory repair

`InsertSyncAnalysis::Run()` refreshes the overlay at each stage boundary. The
existing callback keeps its exact-member fast path. For an otherwise residual
**same-pipe** RAW/WAR/WAW pair it can additionally use the full-completion query.

This includes GM effects of a selected local-buffer operation. It does not
assert GM nonaliasing, cross-pipe GM visibility, cross-core publication, or an
ACC intrinsic exception. Every unsupported/unproved relationship still goes
through existing dependency repair. `alreadySync` is not updated from the new
certificate, so it cannot escape its source/target query.

The effect is prevention of unnecessary barrier construction, rather than
post-hoc deletion of a barrier that other legacy shortcuts may already rely on.
No event-pair deletion is attempted in this patch.

### 4. Retain and recheck each new exemption

Each new full-completion exemption keeps the original source and target
operations and a representative source/target access. Duplicate physical-phase
pairs are indexed rather than linearly rescanned. The full completion claim
covers all access effects of that physical phase pair.

After residual motion/cleanup/allocation and actual protocol emission, the
existing concrete reimport and event proof run. Every retained full-completion
claim is rechecked against that emitted program, not the virtual proof object.
Missing original phase identities or malformed graphs remain internal errors.

The legacy rewrites are not themselves certified by this query. If their result
no longer establishes the strong completion fact, the optional realization is
classified as unproved and R7 retries from the original unsynchronized input,
restoring every omitted dependency. The candidate is never committed on the
basis of its stale preallocation proof. This is not a claimed device counterexample.

### 5. Diagnostics and work bounds

Committed output records:

    pto.insert_sync.lifecycle_completion_pairs
    pto.insert_sync.lifecycle_completion_witnesses
    pto.insert_sync.lifecycle_completion_work

`lifecycle_completion_pairs` counts omitted **memory access-pair repairs**, not
set/wait pairs or removed barrier instructions. Repeated traversal can visit a
pair more than once; `lifecycle_completion_witnesses` is the unique phase-pair
count. They are output diagnostics, not input annotations or semantic promises. The
existing R7 candidate/attempt diagnostics include the mixed-plan stage, residual
handoff import count, query reason, and emitted witness rechecks. Individual
witness diagnostics are capped at 64; all witnesses are still checked.

R8 adds no endpoint-motion or allocation policy. Ordinary residual construction,
motion and allocation still run on the changed dependency set; their final output
must be measured. Final completion rechecking is not a proof of global overlap
dominance over the previous compiler output.

The existing graph/key/phase and work budgets apply. A completion-query limit
leaves its extra exemptions unavailable, not an invalid kernel or proof of
scarcity. This patch does not allocate keys, move endpoints, weaken WAW analysis,
change the alias contract, or change the function-exit policy.

## What is NOT implemented here

* General preallocation deletion of arbitrary residual channels/barriers.
* Full logical import of moved, recurring and dynamic legacy handoffs.
* A global replacement for `alreadySync`, storage generations or the allocator.
* New first/last/lookahead protocol families beyond R7.
* General symbolic GM occurrence disjointness.
* Kernel-specific rules or new information in the input IR.

This is a narrower, testable first slice of the previously proposed mixed-plan
work. It addresses the observed composition regression before broadening the
optimization search. The generic query is wider than a buffer-membership test,
but the supported native residual subset is stated explicitly.

## Native acceptance and next steps

1. Integrate R7, retain its native fixes, then apply this incremental delta.
2. Rebuild the actual adapter. Run the existing R7 tests and new original-input
   one-buffer regression. Keep both combined and staged traversal covered.
3. Compare unchanged inputs in four arms: reference ordinary, reference lifecycle,
   candidate ordinary, candidate lifecycle. Hold architecture, MMAD and caller
   contract fixed. R6/R7 reference binaries must be explicitly identified.
4. Required first result: eliminate the R6-added one-buffer barrier and the six
   additional four-use MTE3 sites, or preserve an actionable reason why the query
   is not reaching those native decisions. No reduced fixture replaces those inputs.
5. Inspect GEMM's actual fallback stage and remaining barrier witnesses. Only then
   extend a needed guard/footprint/resource/composition case. Selection alone is
   not the success criterion, and `fallback` is not counted as optimization.
6. Run full check-pto and the original corpus; qualify numerical/progress/device
   performance separately. No native or device result is claimed in this package.

The independent host reference uses individual dynamic issue/consume identities
rather than the C++ static bitsets. Additional asynchronous tests include
intentionally aliased repeated GM stores within a slot, independent slots,
three-stage overlap, and missing-readiness/release mutations. They do not run
PTOAS or the original GEMM. See STATUS.json for actual runs.

## Sources

* User-provided REGRESSION_SUMMARY.md (copied unchanged into evidence/).
* c3a9004b: LifecycleSynthesis.cpp, InsertSyncAnalysis.cpp, SyncCommon.h,
  one_buffer.auto.pto, four_use.auto.pto.
* Supplied R7 archive and integrated R5 physical/control/completion semantics.
* Official SetFlag/WaitFlag contract: source-prefix completion and consumption:
  https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.html
  This is a development-preview page, not device qualification. No new target
  or GM-visibility guarantee is introduced.

The old ProtocolSync instructions to create a replacement pass are superseded by
the user's later direction to improve InsertSync and use existing IR information.
