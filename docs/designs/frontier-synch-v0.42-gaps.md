# FrontierSynch comparison with draft 0.42

Reviewed 2026-09-24: implementation `3bbc58a67` on
`codex/handoff-foundation`; synchronization draft `267c435` (revision 0.42).
This is a source comparison and proposed acceptance inventory. No code was
changed or runtime validation performed for this comparison.

Subsequent interface correction: the extra FrontierSynch effect audit was removed,
and TAXPY's destination read was added to its shared effect definition. The effects
row and recommendation below reflect that correction.

## Draft baseline and scope

The paper checkout is `/home/toni/work/synchronization_draft`. References below
name files under its `paper/` directory and stable LaTeX labels so that section
renumbering does not change the comparison's meaning:

| Reference | Draft file and label |
| --- | --- |
| Original representation and requirements | `sections/03a_first_pass_overview.tex`, `sec:structural-representation` |
| Provenance, continuation, look-ahead, D1–D4 | `sections/03_storage_analysis_and_the_completion_frontier.tex`, `sec:lifecycle-pass`, `sec:typed-boundary-queries`, `sec:lookahead-interface`, `sec:occurrence-decoder` |
| Joint contexts | `sections/03b_joint_contexts.tex`, `sec:joint-contexts` |
| Fixed-use formation and merging | `appendices/09_joint_context_derivations.tex`, `sec:joint-formation`, `lem:joint-quotient` |
| Symbolic interval frontiers | Same appendix, `sec:joint-interval`, `lem:interval-frontiers` |
| Sufficient region boundaries | Same appendix, `sec:joint-envelope` |
| Descriptor identity and query cost | Same appendix, `sec:joint-variant-policy`, `sec:joint-query-cost` |
| Constructor integration | `sections/04d_joint_context_integration.tex`, `sec:joint-packet-integration` |

The diff from draft 0.41 (`dc5c309`) retains the base representation and D1–D4
contracts. It adds joint context formation, two restricted placement derivations,
and their proposed constructor integration. The marginal provenance equations
remain the conservative view obtained by forgetting joint cases. The draft
does not claim a general mixed-write/while qualifier, a bound for arbitrary
combined contexts, or a complete constructor.

## Implemented foundation and base gaps

The following facts predate the new 0.42 refinements. “Present” describes the
source implementation; it does not certify the complete target contract.

| Area | Source evidence | Gap and consequence |
| --- | --- | --- |
| Original effects and storage | [`importOriginalStructure`, `importStorage`](../../lib/PTO/Transforms/FrontierSynch/OriginalStructure.cpp); [`appendCanonicalStorage`](../../include/PTO/Transforms/FrontierSynch/StorageWitnesses.h) preserve shared translated effects, interval atoms, and conservative overlap witnesses. | Import assigns no definite full writes. Both algorithms rely on the shared `SyncInput` contract, without an additional FrontierSynch effect audit or instruction whitelist. Exact generation qualification needs separate full-write evidence. |
| Provenance and requirements | [`OriginalLifetimes::Impl::solve` and `buildRequirements`](../../lib/PTO/Transforms/FrontierSynch/OriginalLifetimes.cpp) compute four marginal flows, retain read/write incidences, and install per-requirement source subscriptions. | Without full-write evidence, older writer/reader histories survive writes. `noFullWriter` records possible incoming contents, but a qualified invocation-origin case is not constructed from that flag. |
| D1 fixed and alternative sources | [`OccurrenceQueries::fixedVisit`](../../lib/PTO/Transforms/FrontierSynch/OccurrenceQueries.cpp) requires acyclic source/target sites, source dominance in the represented graph, and no intervening cell access on the traversed continuation. | Guarded alternative-origin correspondence is absent. The restriction can also reject a source-to-reader query containing another read; it does not qualify a whole multi-reader episode. `alternativeGuardsQualified` remains false. |
| D2 physical recurrence | [`SyncSlotMapping::derive`](../../include/PTO/Transforms/FrontierSynch/SyncSlotMapping.h), `OccurrenceQueries::bank`, and [`ProgramAnalysis::interpretAt`](../../lib/PTO/Transforms/FrontierSynch/ProgramAnalysis.cpp) retain independent selectors and check disjoint banks and mandatory uses. | The interpreted periodic case requires source and target to be the same operation, one participating operation, and every incidence of both hazard roles there to use the relation. It supplies an intra-invocation predecessor distance only; general producer/reader matching, initial domains, successors, and re-entry remain open. The 256-state scalar budget yields Unknown, not an event-capacity limit. |
| D3 guarded readers | [`OriginalReadQueries::derive`, `sequence`, and `predicateAvailable`](../../lib/PTO/Transforms/FrontierSynch/OriginalReadQueries.h) compose shared guarded frontiers and qualify counted repetition with invariant participation. | Unknown footprints, overlapping writes, varying participation, and unqualified repetition remain unresolved. Requirement-level exactness additionally needs an episode and occurrence certificate. WAW boundaries and the first conflicting write for WAR remain Unknown. |
| D4 owners and continuation | `OccurrenceQueries::children` retains child membership/skip/repeat facts; `OriginalLifetimes::mayAfter` preserves may witnesses and withholds repeated negative proofs. | `decodeAt` chooses `commonOwner(source, target)` before a next-write support candidate is selected. No subsequent step extends that owner to include the declared reuse/continuation. There is no general child re-entry transport, and `OriginalContinuationQuery` carries no checked occurrence certificate. |
| Support qualification | `interpretAt` selects unique write-delimited candidates; `supportBetween` distinguishes `stablePhysicalInterval` from `generationEstablished`. | A conditional reload prevents the whole may interval from being stable; the query does not split it by guard. `qualifiedBoundary` also requires `FixedVisit`. Full-write evidence alone will therefore not complete reader-frontier qualification. |
| Executable endpoints and typed demands | `PhysicalOperation` records outer executable cuts; `endpointCandidates` exposes guards and cut flags. `ProgramAnalysis` indexes SSA prerequisites of conditions, bounds, and represented allocation addresses. | Internal translated-phase cuts lack a lowering contract. Dominance checks original value availability, not selected asynchronous completion. Typed prerequisites retain unresolved incoming/occurrence cases; they do not yet certify native availability. |

`ProgramAnalysis::complete()` reports successful preparation. It does not
establish exactness of every query or effect completeness. The normal
[`frontiersynch::run`](../../lib/PTO/Transforms/FrontierSynch/FrontierSynch.cpp)
indexes requirements, interprets one sample, and returns the explicit
construction-unimplemented error. A richer analysis must preserve that
distinction when a constructor is introduced.

## New 0.42 gaps

### Joint records and context formation

`InterpretedRequirement` currently holds shared marginal source/target histories,
alternative-origin may sets, a support request, and structural reader answers.
It has no guarded family connecting those fields under one original case.
`OriginalReadQueries` interns predicates and frontier expressions, but these
are not whole producer/read/reuse/continuation tuples.

The admitted first increment in `sec:joint-formation` requires exact full-cell
effects, fixed uses in acyclic structured control, immutable total original
conditions, and explicit incoming cases. It needs these additional mechanisms:

- Connect forward provenance and backward next-use answers under the same
  original condition; retain conditional-write intervals separately.
- Choose owners using source, use, next-write or invocation boundary, including
  the declared continuation and its boundary inclusion convention.
- Share whole records and continuation descriptors in a decision DAG. Folding
  identical children is sufficient; equal current origin sets alone are not.
- Follow dependencies of source/deadline placement, including unrelated engine
  work that changes a prefix and subscriptions shared with other requirements.
- Preserve original requirements and existing subscriptions when precision is
  lost; mark the affected exact placement unresolved.

The original tree still contains unrelated translated work, and current source
subscriptions already retain early producer/reader cuts. The missing mechanism
is their joint qualification and preservation through context formation, not
the wholesale absence of those operations or subscriptions.

### Symbolic interval participation

The current D3 loop rule checks that body nonemptiness is available before the
loop. It cannot derive the new case where one read per unit-step iteration
participates exactly for `L_i <= j < U_i` with invariant original bounds.
There is no extrema representation or query deriving
`ell = max(0, L_i)`, `h = min(N, U_i)`, nonempty `ell < h`, first `ell`, and
last `h - 1`.

Implementing this rule requires qualified arithmetic, endpoint availability,
and an empty-case guard that does not evaluate an unsafe `h - 1`. It must be
independent of numeric trip count. The draft does not justify applying it to
multiple varying read sites, arbitrary affine coefficients, or mixed writes.

### Sufficient boundaries and placement descriptors

`OriginalBoundaryResult` exposes Unknown/NoHit/Exact reader answers, and
`OriginalEndpointCandidate` names a before/after physical-operation cut.
Neither represents a sufficient region bracket with its entry/exit occurrence,
complete reader population, unchanged producer, enclosed original work, and
retained consumer deadline. Original control-region identities are present,
but no public descriptor qualifies their entry/exit cuts in an engine stream.

The draft's first recipe selects the first enclosing region on the finite
ancestor chain whose entry/exit execute once between the fixed producer and
overwrite and contain every relevant read. It may execute its endpoints on an
empty reader visit. Thus it must have a separate result kind from an exactly
participating last-reader frontier. Packet failure cannot restart the search
at successively broader regions.

The Phase A interface also lacks a normalized request identity with multiple
placement descriptors retaining the same obligations. Existing cell/kind
deduplication and exact subscription indices are useful foundations, but do
not implement this normalization across physical witness subdivisions and
descriptor variants. Actual binder selection, coverage, event rearming, and
ordering tests belong to Phase B. A sufficient shape alone cannot certify
class 0; the draft otherwise proposes an explicit class-3 repair descriptor.

### Query identity and cost

Current caches are scoped to one immutable `ProgramAnalysis` instance, which
implicitly fixes the original-program version. That is consistent with their
current lifetime; an explicit version field is not required merely to rename
the interface. Extending queries with cases or role maps must also extend
their identities to include the joint case, occurrence/guard interpretation,
selector, stopping interval, and boundary convention. Synchronization edits
must not turn an original summary into selected completion.

`OriginalLifetimes` currently allocates four site-by-origin bit matrices per
cell and materializes marginal relationships and subscriptions eagerly.
Reader expressions and several interpretation results are shared or lazy.
No joint context population, context-formation work, or descriptor population
is measured yet. The draft's optional separator backend is not implemented;
its constant-width query result applies only after the contextual projection
and decomposition are qualified and paid for. It is an implementation option,
not a prerequisite for the first joint-record increment or a general cost bound.

## Proposed acceptance cases

These are specification-derived expectations for future tests, not newly
executed results. Use exact-effect synthetic fixtures to isolate the new
analysis rules, and separate native translation fixtures to establish that
real inputs supply those effects. A synthetic `definiteWrite` flag does not
validate native coverage.

| Case | Required observation |
| --- | --- |
| `W0; A; u; if g then W1; B; W2`, writes on P and reads/unrelated `u` on Q | Under `g`, B uses W1 and A's return is due before W1; under `!g`, A and B use W0 before W2. Retain the post-A source before `u`, all writer hazards, and one correlated interpretation per applicable case. |
| Equal current origins, different future placement | Do not merge cases whose source/deadline or shared continuation answers differ. Include unrelated source-engine work so a cell-only slice cannot erase the distinction. |
| Qualified full write versus partial write/RMW | Kill only provenance justified by full coverage, after retaining incoming hazards. Preserve conservative older origins for partial/unknown writes and both modes for RMW. |
| One reader in `0 <= j < N`, guarded by `j < K`, with `K >= 0` | Nonempty iff `min(N,K) > 0`; first visit 0, last `min(N,K)-1`. Zero/one/large bounds must not require iteration enumeration; unsafe arithmetic or unavailable guards remain unresolved. |
| Fixed producer, optional/variable readers, fixed overwrite | Keep an unknown exact frontier when appropriate, while separately qualifying the unique enclosing region bracket. Include empty visits, unrelated enclosed work, and a reader outside the proposed bracket. |
| Child read followed by reuse outside the child | Select an owner containing the declared continuation; a child exit must not replace the next overwrite. Keep repeated entry/backedge/exit domains explicit. |
| Independent period-2/period-3 selectors and mixed incidences | Preserve independent physical records. A fixed or differently selected access must not inherit the periodic proof. Do not create a period-6 control history. |
| Guard known to analysis but unavailable at an endpoint | Preserve the correlated fact while refusing its executable frontier; do not emit a test of hidden history or assume an asynchronous value has completed. |
| Equivalent witness subdivisions and multiple descriptors | Retain every physical obligation and subscription without creating extra semantic request weight or residual progress units. |

Retain the existing optional-sibling, unavailable-predicate, reload, while-before,
noninjective-selector, and owner-escape coverage when introducing joint cases.
Check normalized query answers against independent expected relationships;
passing the current one-sample entry diagnostic is insufficient.

## Recommended implementation order and evidence boundary

1. Use the shared instruction-effect contract and establish full-write
   certificates where justified; retain conservative writes elsewhere.
2. Add the fixed-use joint family, continuation-aware owner selection, guarded
   source subscriptions, and conservative loss-of-precision behavior. Start
   with the conditional-write fixture and incoming/zero-reader cases.
3. Add the interval rule and the separate sufficient-boundary descriptor under
   their stated premises. Preserve the existing exact-frontier contract and
   original request identity.
4. Extend D1/D2/D4 correspondence and repeated continuation qualification before
   claiming combinations across re-entry. Add query/cost accounting with the
   new records, rather than inferring it from the unrefined syntax tree.
5. Have the constructor consume these records through one shared causal and
   packet interface, with actual source snapshots, endpoint words, matching,
   key reuse, and boundary/progress checks.

This is a proposed dependency order, not an additional paper theorem. The
draft's combined adequacy and general cost targets remain open even after
implementing the restricted rules. Generality, correctness of selected plans,
and synchronization quality need their own evidence.

Existing validation is recorded in [HANDOFF.md](../../HANDOFF.md). The focused
probe artifacts predate this comparison and do not establish 0.42 acceptance.
This update checks documentation consistency and source correspondence only.
