# Repeated-guard endpoint audit: milestone 1

Starting revision: `4d1767ae22a97f739d329a1702dd91561a705a5f`.
The implementation plan is [repeated guard emission](frontier-repeated-guard-emission-plan.md).
This milestone adds diagnostics and an opt-in exhaustive audit; it does not change
mathematical analysis, guard recovery, command semantics or allocation.

## Gated Delta evidence

For `prepared/gated_delta_rule/input.pto`, MayNotAlias still certifies all 29
original regions. MayAlias still certifies 23 of 29. Certification output under
both policies and the retained-demand check are byte-identical to the pinned
starting executables. Failed endpoint attempts leave the original module intact.

The token loop at line 144 has 60 sites, span three and 1,172 retained symbolic
records. An isolated detached preparation of every record finds:

| Outcome | Records |
| --- | ---: |
| Endpoint recipe prepares | 101 |
| Publication contract unavailable | 4 |
| Consumer guard recovery unavailable | 1,067 |

The four publication failures are records 4, 11, 21 and 42: source site 2
(`tcvt`, line 151) to sites 7 (`tsetval`, line 160), 14 (`tsetval`, line 170),
16 (`tsetval`, line 173), and 21 (`tgetval`, line 178). All have displacement
zero. The first failure therefore is a SET/publication obligation, not a
consumer-recovery failure. Its current-visit branch condition is defined after
the source cut and depends on `tgetval`; it cannot be replayed there.

Consumer failures have displacement one (355), two (356), or three (356).
The dump records each original operation, phase pipe, failing cut, exact guard
DAG, integer recipe signature, requested evaluation coordinate and original
opaque SSA leaves. Definition dominance and visit displacement are separate
fields: a late current-visit value is not classified as a future visit.

A separate uniform proof using the existing monotone visit-presence certificate
proves 900 original guards identically false. Of these, 879 are currently
blocked: two publication failures and 877 consumer failures. This is evidence
for emission-only inactive recipes, not a change to the reducer. Even after
such a change, 192 blocked records would remain: two publication failures and
190 consumer failures (62/64/64 at displacements one/two/three).

## Verification and reproducibility

The source-pinned candidate build uses effective `-O2 -UNDEBUG -std=c++17`
flags across all 185 Frontier and test-tool object commands. Baseline executables
were preserved before rebuilding. Local receipts, exact source/binary hashes,
object commands, all obligation records and paper hashes are under
`.local/three-paths/repeated-guard/m1/`; the final manifest is `manifest.json`.
Large DAG dumps are local diagnostic artifacts, not committed test data.

Focused optimized lit checks pass (4/4): bounded endpoint audit, bounded boundary
presence, guarded publication, and bounded regional interfaces. The last covers
session/result ownership and retention after failed preparation. The audit
checks repeated failure counts, agreement with normal preparation and unchanged
original IR. Gated Delta exhaustive audit completes in 24.54 seconds with
82,248 KiB peak RSS; normal certification takes 0.80 seconds (MayNotAlias) and
0.10 seconds (MayAlias). Watchdog/memory limits are measurement safeguards.

The changed-code checker reports 10 files, zero errors and zero warnings;
`git diff --check` passes. No new compiler warnings were introduced. Full
EChecker/SecK analysis was not run. Applicable rules cover C++17, explicit error
paths, detached ownership, checked diagnostic inputs and scoped changes.

Section 8 was reread before freezing the milestone. Its current SHA-256 is
`c9a6a9ad72f94e78bf54740b3ee2c25dd577e269b5a10adb59f59dba8db2647d`,
which differs from the planning pin. The working-tree changes refine explicit
versus compact allocation schemas, direct-plan reuse queries and numerical
periodic allocation. They do not change this milestone's guard or detached
preparation contracts. The other three paper pins remain unchanged. Section 8
still requires available guards and retained exact mathematics after emission
failure. This milestone makes those obligations observable without implementing
stateful evaluation.

## Milestone 2: contextual evaluation preparation

`IterationPredicates::prepare` rewrites exact current-visit placeholders to
original SSA values or enclosing branch constants before collecting live inputs.
Past/future evaluations stay distinct. Both this path and the ordinary
contextual emitter use the shared internal `OriginalCutFacts` collector. No
public expression API, producer or reducer changes are needed. Preparation
returns the selected root explicitly; detached value/cofactor caches do not
stand in for an expression rewrite. Foreign invocation cuts are rejected.

The discovered `sync_iteration_predicate_context.pto` test drives an independent
fixture with opaque memory-derived predicates, lower bound 11, step three and
zero/one/three trips. It checks branch-masked future/past inputs, late current
inputs, genuinely live past/future failures, opposite arms, foreign cuts, fresh
retries and failure-to-success reuse of the same cut-local context. No load is
replayed and original IR remains byte-identical. Focused bounded preparation,
regional ownership and guarded publication checks also pass.

Gated Delta still has 1,071 blocked endpoint records: this change removes an
unnecessary recovery ordering in general, but does not recover past outcomes or
make a future-dependent SET guard available. Certification and retained-demand
checks remain byte-identical to the starting revision. Local optimized build,
test, audit and source pins are under `.local/three-paths/repeated-guard/m2/`.
Section 8 was reread; its current pin remains `c9a6a9ad...` above, and the other
paper pins are unchanged. Private state and recurring publication remain later
milestones; this change does not authorize their use implicitly.

## Milestone 3, first patch: uniform inactive endpoint recipes

Endpoint preparation now reuses the existing monotone-presence proof to select
false emission guards for uniformly inactive records. This proof is scoped to
the original window and rolls back temporary expressions; it does not borrow
allocation-domain facts. Every original record, family and matching coordinate
remains present. Producers and reducers are unchanged. Unknown proofs retain
the original emission obligation. Normal preparation proves each record only
when reached; it does not precompute every proof before a possible early live
failure. This keeps Delta retained-demand checks at 0.90 seconds instead of the
61.10 seconds observed with eager proof construction. The exhaustive audit is
a deliberately larger diagnostic workload.

The optimized Delta audit now reports 192 blocked records out of 1,172: two
publication-contract failures and 190 consumer failures. The 900 inactive
records include 879 formerly blocked records. All 272 remaining potentially
active records have registered predicate offsets no later than their consumer:
82 have lookback zero, 62 one, 64 two and 64 three. This offset audit concerns
registered evaluation placeholders; it does not establish arbitrary SSA
availability, global-query locality or executable history storage.

Seven focused lit checks pass, including inactive emission, contextual
predicates, endpoint audit, bounded provenance, first/tail allocation, regional
interfaces and guarded publication. Synthetic guards include a masked opaque
future predicate, independent presence alternatives, original record IDs,
fresh detached retries, both local and cross-pipe endpoints, and an unsigned
wrap counterexample at `UINT64_MAX` which must remain live. Uniform proof
checking is deliberately sufficient: an unknown proof never drops a record.
Delta certification under both alias policies and retained-demand output are
byte-identical to the pinned baseline. Local receipts are in
`.local/three-paths/repeated-guard/m3/`.

Section 8 was reread before this patch; its hash remains `c9a6a9ad...` above,
and the other three paper hashes are unchanged. This is the first patch within
milestone 3, not completion of consumer selectors or the emission plan. Finite
predicate history, the two recurring publication obligations, shared scalar
state materialization and final integration remain outstanding.

## Milestone 3, second patch: detached structured private state

An internal materializer now rebuilds a function on an owned module clone,
threading typed private scalar slots through existing `scf.for` and `scf.if`
control. Original iterated values and results retain their positions. Missing
else arms forward state, zero trips preserve initial values, and actions bind
to original before/after cuts through an explicit IR mapping. The result is
verified before being returned. Unsupported region operations, foreign cuts,
invalid state types and non-dominating state values return explicit failures.
Neither successful preparation nor failure attaches anything to original IR.

This is shared preparation infrastructure, not an enabled private-state target
capability or an attached emission plan. Callbacks are trusted realization
code; the verifier establishes typed, dominating SSA, not mathematical demand
equivalence. Cloning the containing module preserves symbol resolution but has
a module-size cost; it is not claimed to be constant or region-local work.

The independent interpreter checks original payload traces and iterated
results across zero/one/three trips, both branch outcomes, nested resets and
nonunit induction coordinates. Negative checks cover partial callback failure,
foreign cuts, unsupported regions, wrong types and a same-type non-dominating
value; a clean retry succeeds after each rejected clone. The focused state,
inactive-emission and guarded-publication lit checks pass (3/3). Optimized
build and runtime receipts are under
`.local/three-paths/repeated-guard/m3-state/`; failed intermediate build/test
receipts are superseded by `build-range`, `fixture-range` and `focused-final`.
Section 8 was reread; all four paper pins remain unchanged. Predicate-history
and prefix realizations, their integration, and Delta public logical recovery
remain outstanding.
