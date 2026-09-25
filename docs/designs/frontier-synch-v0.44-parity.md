# FrontierSynch Phase A parity checklist

## Current integration update — 2026-09-25

Downloads steps 7–12 have been merged sequentially and independently accepted
against draft v0.44 (`4905f8a`), including integration corrections. Step 13 source
preparation and scoped consequences are implemented locally and independently
accepted within the documented scope after final integrated validation. See [the integration record](frontier-synch-steps7-13-review.md) for the current evidence,
precise supported fragments and remaining qualifications. Older baseline tables
and submitted patch reports below remain historical; they do not override this
update. Step 14 and aggregate draft parity have not been accepted. The later
v0.45 manuscript is not the contract for these patches.


## Step 7 candidate delta — native validation and independent review pending

Read these updates alongside the historical rows below. Procedure implementation
is not acceptance; the [step-7 record](frontier-synch-step7-review.md) separates
executed portable checks from the added, unexecuted native tests.

| Entries | Candidate implementation and consumer | Evidence and remaining gate |
| --- | --- | --- |
| PA-040 | `queryFixedVisitSources` reads the hazard-specific OLD demand root. `OccurrenceQueries::fixedVisit` and `ProgramAnalysis::fixedVisitFor` use it instead of source dominance and a no-intervening-access scan. Pure reads do not kill producer correspondence; local fixed-use loop projections are admitted without inventing backedge transport. The cache includes hazard as well as the complete interval. | Portable multiple-reader/old-state/partial-write oracle passes; native hazard-cache, local-loop and legal-cut fixtures added. Native run and independent step-7 review pending; D2/D4 transport stays at steps 8/9. |
| PA-041 | `fixedSourcesAt` and `ProgramAnalysis::fixedSourcesFor` return deterministic guarded origins, incoming cases and retained effect witnesses. `Both` remains simultaneous obligations; it is not relabeled an exclusive writer choice after a partial overwrite. Obligation IDs and conservative marginals are unchanged. | Portable conditional replacement, no-producer and incompatible-arm checks pass. Native public-family and incoming-only fixtures added but unexecuted. An incoming record is not a synthesized source publication or completed interface. |
| PA-042 | Each origin has its applicability and separately qualified source/target conditions, using the common original-value service at legal immediate cuts. Source lexical facts do not make a target guard available. `InterpretedRequirement::fixedVisit` retains the full record and its predicate arena. | Portable matching-domain checks cover optional targets and repeated guard identities. Native late-guard and unavailable-phase-cut checks added but unexecuted. Completion prerequisites remain explicit; no selected credit is created. |

Steps 10–13 (structural frontiers, covering boundaries, descriptors and preparation)
and the final integrated review in step 14 are not completed by this increment.
## Step 8 implementation delta (acceptance pending)

The [D2 implementation/evidence record](frontier-synch-step8-review.md) updates
entries **PA-043–PA-046**: physical selection, role matching and entry/exit
domains. `OccurrenceQueries::bank/periodic`, `ProgramAnalysis::periodicUseFor` and
the compatibility `interpretAt` path now share that implementation. Original
obligations, conservative footprints and unresolved enclosing transport remain.
The record names native validation and independent review as outstanding gates;
a passing standalone finite oracle is not integrated parity acceptance.
## Step 9 candidate delta

The D4 composition and re-entry candidate is documented in
[the step 9 review report](frontier-synch-step9-review.md), including the
PA-047–PA-049 implementation map, portable semantic tests, native supplier gates
and pending native validation/independent review. This is not an acceptance record
and does not change the historical statuses below by implication.
## Step 10 candidate delta — pending acceptance

The [Step 10 note](frontier-synch-step10.md) records the common exact-boundary
service, interval-participation recipe, public consumers, executed core tests
and unexecuted native checks on base `4122dd1531fbdb2859bd7b27772dba56930be393`.
It does not upgrade any row to independently accepted or claim complete parity.
D1/D2/D4 integration, native effect premises and the final integrated gate remain
explicit review obligations.
## Step 12 delivery delta — awaiting integrated review

PA-065–PA-067 now have request-group records, a shared three-slot formation
procedure and a frozen public `ProgramAnalysis::requests()` consumer. Original
obligation IDs remain owned by step 6. The portable tests exercise independent
source/target answers, shared choices, duplicate slots, unavailable guards and
retained reference coverage; native semantic fixtures are supplied, not yet run.
PA-036–PA-039 retain their original obligation/witness meaning. PA-060–PA-064
are NOT completed by accepting supplied covering answers: their step-11 derivation
remains missing in this base. PA-068–PA-071 preparation/support closure remains
step 13. Status is **partial pending native integration and independent review**,
not an acceptance decision. See [the step-12 record](frontier-synch-step12-review.md).

## Current integration delta — 2026-09-24

Steps 2–6 are applied in the working tree. All five increments have scoped independent
acceptance. The [integrated review](frontier-synch-steps2-6-review.md)
records changes, current consumers, semantic regressions and validation limits.
The 80 stable entries below deliberately preserve the step-1 audit of the pinned
baseline; they are not a fresh inventory of the integrated tree. Apply this delta
and the review ledger when reading their old implementation/status columns.

| Affected entries | Integrated change | Remaining boundary |
| --- | --- | --- |
| PA-009–PA-014, PA-080 | Common original cuts, complete interval keys and snapshot identity; native cut/owner and finite-interval tests | General D4/recurrence qualifications remain later gates |
| PA-003–PA-008 | Shared full-write interface, exact-cell coverage and RMW correction; native positive/negative coverage fixtures | Unsupported geometry remains conservative, not rejected |
| PA-016–PA-031 | Shared factored builder, writer/reader entry histories, scoped projections and concrete oracle | Cross-reentry transport and general repeated-region summaries need D4 |
| PA-032–PA-035 | Common value identity, width-correct arithmetic and four directional availability outcomes | Broader symbolic arithmetic remains unresolved explicitly |
| PA-018–PA-020, PA-030, PA-036–PA-039 | Guarded immutable obligation IDs, lazy membership/enumeration and shared witnesses | Uniform cross-partition mapping, descriptors and constructor consumption remain later gates |

## Baseline and use of this checklist

Implementation inspected: **`0a38c8e9f4bd4852177c1dd6f6e0d0ee26b6ca16`**, branch
`codex/handoff-foundation`. This includes factored-core commit
**`9bba6552055d5386ff7242c58031b0412885a0d6`**. The previous inventory's
`6220bb6c` baseline and claim that no provenance/demand DAG exists are superseded.

Specification inspected: the supplied **draft v0.44**, *Storage-Driven
Synchronization with Reusable Events*, 83-page PDF. Section/page references below
use that PDF. Manuscript pins retained from the handoff are `4905f8a` (v0.44)
and `a13650a` (v0.43); the implementation plan identifies their operative Phase A
contract as unchanged. This audit does not independently compare those two
manuscript commits.

This is **step 1** of the fourteen-step plan. It changes documentation, not
analysis or instruction semantics. The checklist's coverage passed independent
review on 2026-09-24, as recorded in the ledger below. The factored core's
prior reviewer acceptance is retained as reported in the implementation plan,
not expanded into acceptance of conditional provenance integration.

Each `PA-nnn` entry records the specified behavior and assumptions, implementation
entry point, actual consumer, status, existing test evidence, and the remaining
input/integration prerequisite with its later gate. Keep these IDs when updating
status. Split a row if its components acquire different acceptance outcomes.

**Status is scoped to the behavior in the row.** `Implemented` means that the
procedure is present for the stated inputs, not that its native qualification,
all callers, or regression coverage have been accepted. `Partial` means that a
component exists but the specified service is incomplete. `Absent` means no
implementation of that procedure/interface was found in the audited Phase A
path. A named method that always returns `Unknown` on specified supported cases
is incomplete. Missing implementation cannot be reclassified as a draft-open
problem. An input outside a procedure's *draft* premises may legitimately retain
`Unknown` and its original obligations.

No compiler, lit, corpus, sanitizer, or device run was performed for this
source/documentation audit. Test definitions were inspected. Prior run claims
below are explicitly historical; a missing semantic assertion is not filled in
by a successful process exit. Step 14 must recheck the integrated system even
when individual components have been accepted.

## Specification coverage map

| Draft location | Phase A obligation covered here | Checklist families |
| --- | --- | --- |
| Sections 2.1–2.2, pp. 3–4; F.1–F.2, pp. 68–69 | Fixed original control, complete shared effects, physical identity, whole-operation and typed prerequisites, target/availability premises | PA-001–PA-010, PA-032–PA-035, PA-039 |
| Section 2.5, pp. 5–6; Section 3.2, p. 7; Table 1, p. 8 | Legal cuts, original observations, owner/occurrence/continuation identity, conservative requirements, original/selected separation | PA-009–PA-015, PA-018–PA-019, PA-036–PA-038 |
| Sections 3.1 and 3.5, pp. 6, 10–11; original-data portion of 3.10, p. 14 | Early source milestones, adjacency, timely subscriptions, finite support closure, scoped consequences | PA-068–PA-073 |
| Sections 3.3–3.4, pp. 8–10; I.1, pp. 78–80 | Marginal and factored provenance, incoming cases, shared conditions, physical succession, All and MayAfter | PA-004–PA-008, PA-011–PA-031, PA-051, PA-059, PA-079–PA-080 |
| Section 3.6, pp. 11–12; referenced analysis results 5.2, pp. 32–35 | D1, D2, D3, D4; common fixed-selector first/last queries; conservative and exact query distinctions | PA-020–PA-022, PA-040–PA-056 |
| Section 4.1, pp. 15–17; I.4, pp. 81–82 | Obligation/group/descriptor identities, original endpoint qualification, three prederived slots, original support records | PA-032–PA-039, PA-065–PA-071 |
| I.2, p. 80 | Qualified interval participation, safe nonempty extrema, spatial-selection normalization | PA-057–PA-059 |
| I.3, pp. 80–81 | Independent directional covering queries, sequence trimming, conditional/finite-loop cuts, extra scope and executions | PA-060–PA-064 |
| I.5–I.6, pp. 82–83; original-query portions of F and G | Complete query keys, sharing, restricted formation bound, separately charged output/qualification | PA-014, PA-031, PA-074, PA-078; reference/backend register below |
| Section 7.1–7.3, pp. 53–54 | Evidence limits, integrated examples, independent checking | PA-075–PA-077; review ledger below |

Sections 3.7–3.9 explicitly leave Phase A: causal transfer, selected coverage and
must-joins belong to Phase B. Section 3.10 mixes original slot/role descriptions
with selected snapshots and assertions; only the original side is in this
checklist. The same distinction applies to Section 4.1: Phase A supplies original
requirements, boundary/observation descriptions and support relations; it does
not certify a selected packet or create a conditional selected proof context.
The reference/backend register at the end prevents theorem examples and backend
options from silently becoming mandatory Phase A planners.

## Audited entry points and consumer names

Paths are relative to the repository. These aliases keep the checklist readable;
function names in each row identify the actual path through the source.

| Alias | Source |
| --- | --- |
| `SI` | [SyncInput.h](../../include/PTO/Transforms/InsertSync/SyncInput.h), [SyncInput.cpp](../../lib/PTO/Transforms/InsertSync/SyncInput.cpp): shared `PTOIRTranslator` input |
| `OS` | [OriginalStructure.h](../../include/PTO/Transforms/FrontierSynch/OriginalStructure.h), [OriginalStructure.cpp](../../lib/PTO/Transforms/FrontierSynch/OriginalStructure.cpp): control/effect import; calls shared scalar/address analysis |
| `SW` | [StorageWitnesses.h](../../include/PTO/Transforms/FrontierSynch/StorageWitnesses.h): `appendCanonicalStorage` |
| `SO` | [StorageOrigins.h](../../lib/PTO/Transforms/FrontierSynch/StorageOrigins.h): `origin_detail::collectOrigins` and origin propagation |
| `CG` | [Control.h](../../lib/PTO/Transforms/FrontierSynch/Control.h): original finite graph, reachability and backedge witnesses |
| `OL` | [OriginalLifetimes.h](../../include/PTO/Transforms/FrontierSynch/OriginalLifetimes.h), [OriginalLifetimes.cpp](../../lib/PTO/Transforms/FrontierSynch/OriginalLifetimes.cpp): marginals, requirements, intervals, subscriptions |
| `FP` | [FactoredProvenance.h](../../include/PTO/Transforms/FrontierSynch/FactoredProvenance.h): committed acyclic factored transfer core |
| `OQ` | [OccurrenceQueries.cpp](../../lib/PTO/Transforms/FrontierSynch/OccurrenceQueries.cpp): `fixedVisit`, `bank`, `children` |
| `RQ` | [OriginalReadQueries.h](../../lib/PTO/Transforms/FrontierSynch/OriginalReadQueries.h): structural reader/predicate DAGs and availability tests |
| `PA` | [ProgramAnalysis.h](../../include/PTO/Transforms/FrontierSynch/ProgramAnalysis.h), [ProgramAnalysis.cpp](../../lib/PTO/Transforms/FrontierSynch/ProgramAnalysis.cpp): public query boundary |
| `RUN` | [FrontierSynch.cpp](../../lib/PTO/Transforms/FrontierSynch/FrontierSynch.cpp): actual `frontiersynch::run` pass entry |
| `TEST` | [pto-frontier-analysis-test.cpp](../../tools/pto-test-opt/pto-frontier-analysis-test.cpp): `checkFactoredProvenance`, `inspectRequests`, `analyze`, `runFile` |

The current consumer chain is `SI -> OS -> PA(OL, OQ, RQ) -> RUN/TEST`.
`RUN` counts indexed storage/typed requirements, interprets **one sampled
requirement**, then reports that construction is not implemented. `TEST` calls
`interpretAt` on **every indexed marginal requirement**. Neither is a functioning
Phase B consumer. Methods exposed by `PA` but not used by these callers are
identified as query APIs, not claimed to drive a selected synchronization plan.
In particular, `interpretAt` attaches `OL::factored(cell)` but does not use its
`Demand` nodes to replace the marginal requirement universe.

## Evidence register

| Evidence ID | Existing test definition or report | What it does and does not establish |
| --- | --- | --- |
| `F1` | `TEST::checkFactoredProvenance`, command `pto-frontier-analysis-test --factored-self-test` | Manually constructs `W0; A; if(g) RMW-W1; B; if(h) R; W2`, supplies definite-write flags, evaluates four guard assignments, and compares demand sets with handwritten expectations. Checks `priorWriters[B]` and `nextWriters[A]`. This is core semantics, not native write qualification or an independent general concrete scanner. |
| `F2` | The 64-optional-reader case in the same function | Checks successful formation and `nodes.size() < 64 * 12`, without enumerating the 64 guards. One size assertion is not a measured asymptotic work bound or frontier-output test. |
| `F3` | The possible-write case in the same function | Checks retained writers `{W0, possible-W}` before the following read and retained readers before the final write. Does not test imported partial tile geometry. |
| `T` | [frontier_synch_taxpy_effects.pto](../../test/lit/pto/frontier_synch_taxpy_effects.pto) | Imported TAXPY fixture expects `RAW=2 WAR=0 WAW=1`, three interpreted requirements, `unknown-occurrence=0`, and unchanged IR. This checks both input reads/destination write and narrow fixed visits. It does not check full-overwrite coverage or execute both construction modes. |
| `C` | [Five development kernels and README](../../test/lit/pto/frontier_synch/README.md), using `TEST::inspectRequests/runFile` | Checks indexed source subscriptions and unchanged IR, prints hazard/reader/occurrence counts, and verifies materialized manual references separately. No expected exact first/last, All/MayAfter or recurring-use answers are asserted by the runner. `inspectRequests` does not evaluate factored demand contents. |
| `H2` | [HANDOFF](../../HANDOFF.md), historical Stage 2/3 importer/address probes | Reports independent periods, aliases, root propagation, structured control, unsupported CFG and unchanged IR. Probe sources/logs are referenced under the developer's local `handoff-builds` directories; not inspected or rerun for this audit. |
| `H3` | Historical Stage 3 and eight-input query probe in HANDOFF | Reports optional/late-guard readers, fixed visits, invariant/varying counted readers, owner escape, same-role period-three and mixed-incidence refusal. These are reports, not new committed assertion evidence. |
| `H4` | Historical Stage 4 probe in HANDOFF | Reports reader/reload support, older-origin retention, source indices and May/NoHit/Unknown continuation cases. Same evidence limitation as `H2/H3`. |
| `None` | No semantic assertion identified in the inspected tests | The row names the assertion required at its later gate. Source inspection or corpus exercise is not substituted for that test. |

The `F1` expression evaluator discards the `Incoming` sentinel when collecting
its compared demand set. It therefore provides **no** incoming-obligation or
incoming-reader-interface test. All `F1`–`F3` effects are synthetic records.

Historical corpus counts are 77 (TileLang GEMM), 35 (PyPTO GEMM), 96 (FA cube),
1,976 (FA vector), and 58 (vector add): all 2,242 occurrence interpretations were
reported `Unknown`. These are baseline observations, not acceptance targets.
`T` separately demonstrates that the current implementation is not universally
`Unknown` on straight-line input. The handoff's later report of existing-mode
C++ emission and the corpus README's earlier focused-run-only statement refer
to different validation checkpoints; neither was rerun here.

## 1. Original-program contract, effects and physical storage

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-001 **Implemented** | Sections 2.1, 3.1–3.2: preserve payload/control, engine assignments, allocation and original values; side tables are permitted. | `OS::importOriginalStructure` builds a private candidate; `PA` owns read-only records borrowing `SI` and IR; `RUN/TEST` consume them without inserting commands. | `C/T`: `runFile` compares printed IR before/after analysis. | Owners must outlive the analysis. Retain this regression through all changes; step 14. No selected-plan correctness follows. |
| PA-002 **Implemented** | Sections 2.1, 3.2: admitted structured sequence/choice/for/while semantics, including bypass and final while-before execution; a may graph may retain infeasible walks. | `OS::importControl -> CG::buildControlGraph -> OL/OQ`; For graph retains a bypass conservatively; While decision follows the before region and exits there. | `H2/H3` report control probes; `C` exercises nested control but not every transition assertion. | Single-block structured regions are required by import. Add committed zero-trip/final-false/unsupported-control assertions in steps 2/14; arbitrary CFG recognition is not asserted. |
| PA-003 **Implemented** | Sections 2.1–2.2, F.1: one shared instruction-effect input; preserve every translated phase and read/write incidence rather than a second FrontierSynch opcode registry. | `SI::build` calls `PTOIRTranslator`; `OS::importStorage` consumes `defVec/useVec`; `OL/RQ` read those imported accesses. | `T` tests the shared TAXPY effects; `C` exercises the adapter. | Correctness still assumes complete shared effects. Coverage qualification must be added to that semantic path, not a FrontierSynch whitelist; steps 3/14. |
| PA-004 **Implemented** | Sections 2.1, 3.2; I.1: separate memory domain, storage origin and interval; partition qualified overlaps at endpoints, including subviews. Geometry is not a content-generation proof. | `SO::collectOrigins`, `OS::setCoordinates/importStorage -> SW::appendCanonicalStorage -> OL/RQ`; roots and original effect pointers survive partitioning. | `H2` reports alias/root/partition probes; `C/T` exercise concrete buffers. | Exact coordinates require qualified offsets, size and domain. Add committed subrange/domain/overlapping-view assertions; steps 3/14. Target-specific domain-instance completeness remains PA-008. |
| PA-005 **Implemented** | Sections 3.2–3.4: retain uncertain may footprints without turning nontransitive may-alias into physical equality or definite coverage. | `OS` privately widens carried/unknown ranges; `SW` retains separate conservative cells and pairwise alias witnesses; `OL` generates conflicts, `RQ/OQ` withhold exact certificates. | `H2` reports the nontransitive overlap and partially unknown-address cases. | Shared alias conservatism is a premise. Commit expected answers for an unknown footprint overlapping two disjoint ones; steps 3/14. |
| PA-006 **Partial** | Section 3.4; I.1; Section 2.1 footnote 2: strong update only for a definite full overwrite of an exact cell. | `Access::definiteWrite`, `SW` propagation and `OL/FP` strong-update branches exist; **`OS::importStorage` supplies false for every write**. | `F1` supplies the flag manually; no imported positive full-overwrite assertion. | Shared effect coverage plus actual geometry/valid dimensions/padding/subviews must justify positive flags. Bounding allocation size is insufficient; step 3. |
| PA-007 **Partial** | Sections 2.1–2.2, 3.4: distinguish may-write, definite overwrite and reading old contents; RMW retains its old-state prerequisites. | `SI` exposes reads/writes; `OS` retains both; `FP::transfer` queries old state before replacing it. Imported coverage is missing. | `T` tests one real RMW signature; `F1/F3` test synthetic RMW/possible writes. | Audit the relevant shared signatures and uncertain/partial geometry; do not infer destination-read absence from an overwrite flag. Steps 3/4/14. |
| PA-008 **Partial** | Sections 2.2, 3.2, F.1–F.2: complete target signatures/implicit effects and physical domains; native guarantees differ from obligations, including typed/control availability. | `SI/OS` import shared effects and engine/domain information; `PA` adds some typed dependencies. There is no complete Phase A target/availability qualification record corresponding to the full draft contract. | `T` is one signature regression; `C` is not a target-wide audit. | Missing implicit effects must be resolved/conservative at the shared interface. Qualify multi-phase/whole-operation meaning and required native availability; steps 3/5/6/14. Do not claim a universal architecture profile. |
| PA-009 **Partial** | Sections 2.5, 3.2, 4.1: every positive endpoint denotes a legal original cut, including structured boundaries; analytical instruction phases need not be executable gaps. | `PhysicalOperation` marks only outer before/after cuts; `SourceMilestone` is operation plus side; `PA::endpointCandidates` retains executable flags. `CG`'s internal sites are not a public structural-cut interface. | `H3/H4` report multi-phase cut checks; `C` does not assert internal-gap refusal. | Represent if joins, loop entries/exits and interval cuts; retain internal-cut obstructions. Do not relabel the enclosing post-instruction cut as the sufficient internal milestone; step 2. |
| PA-010 **Partial** | Sections 3.1, 3.3, 3.10: retain distinct early source positions and the complete original engine work a prefix would enclose, not only the queried cell's access projection. | `OS` retains all phases/engines; `OL::buildRequirements` records per-source sufficient cuts and executable subscriptions; `PA::interpretAt` exposes them. No general original prefix-scope descriptor exists. | `C` tests direct subscriptions only; `H4` reports distinct source information. | Early `a; unrelated-u; b` and shared-engine copy/release assertions, including explicit extra scope for covering boundaries; steps 2/11/13/14. |

## 2. Common occurrence, owner, cut and continuation records

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-011 **Partial** | Sections 3.2, 3.4, I.1/I.5: a query identifies cell/selector, owner, occurrence interpretation, starting/stopping cuts, inclusion and original version. | `OriginalUseQuery`, `OriginalContinuationQuery`, `ReaderIntervalQuery` and `OriginalOccurrenceInterpretation` carry different subsets; `PA::interpretAt` combines handles rather than one qualified interpretation. | `H3/H4` report some owner/interval distinctions; no full-key fixture. | Unify the interpretation without guessing a dynamic occurrence from a static site; step 2. |
| PA-012 **Partial** | Section 3.6; I.1: owner encloses source, target **and declared continuation**; a child index reset does not start a fresh physical use. | `PA::decodeAt` calls `commonOwner` on source/target ancestor paths only; `OQ::children` reports lexical metadata. | `H3` reports child-interval distinctions; no source/target-equal, continuation-different assertion. | Include the declared horizon in owner selection and preserve incoming/bypass cases; steps 2/9. |
| PA-013 **Partial** | Section 3.4: distinguish before/after stopping-access inclusion, child exit versus owner overwrite, and positive-length recurrence to the same static cut. | `OL::mayAfter` uses `SourceMilestone::Side`; `OL::frontier` uses `includeStops`. Repeated identical static gaps return Unknown. | `H4` reports May/NoHit/Unknown tests. | Public structural cuts and occurrence-qualified starts/stops are missing. Test all inclusion combinations and both continuations; step 2. |
| PA-014 **Partial** | Sections 3.2, 3.6; F query contract; I.1/I.5: share only equal complete query interpretations; retain negative and Unknown outcomes; invalidate on original changes. | `PA/RQ` caches are owned by an immutable analysis instance and keyed by their current parameters; `OL::factored` caches per cell. No explicit original version or common occurrence/horizon key is exposed. | `H3` reports unresolved-negative behavior; no richer-key invalidation test. | Current instance-local caching is not itself evidence of stale answers. Steps 2/4/9 must extend keys with the new semantics or rebuild the analysis; step 14 tests successes, negatives and Unknown. |
| PA-015 **Implemented** | Sections 3.2–3.5 and Table 1: required relationships, may exclusions and source subscriptions supply no acquired completion, key reservation or selected receipt. | `PA/OL/OQ/RQ/FP` have no selected ledger; `RUN` stops before construction. `InterpretedRequirement` retains unknown components. | `C/T` assert `construction-not-run`; source inspection verifies the original-only data path. | Preserve the boundary when Phase B starts consuming these APIs. PendingCompletion is a Phase B query, not a missing All-minus-MayAfter calculation. |

## 3. Marginal provenance, succession and original obligations

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-016 **Partial** | Section 3.4; 5.2.1–5.2.3: preceding writers and pure readers since the last definite write, union over qualified original control; generate obligations before update. | `OL::Impl::solve(false)` and `buildRequirements -> PA::requirementsAt/lifecycleAt`; finite matrices and strong-update plumbing are present. | `T` checks a small imported hazard population; `H4` reports reload retention. | All native writes are non-definite. `solve` also retains the RMW's read bit, so do not claim literal equality to the pure-reader sparse generator without checking that distinction; steps 3/4/6. |
| PA-017 **Partial** | Sections 3.4, 5.2.6: backward next writers/readers before the next write with the same original boundary interpretation. | `OL::Impl::solve(true) -> lifecycleAt -> PA::interpretAt` uses next writers to propose support. | `H4` is a historical report; `F1` tests the separate factored next-writer result. | Marginal equations exist, but incoming/continuation qualification and native full-write precision are missing; steps 2/3/4/9. Test backward readers and RMW boundary conventions explicitly. |
| PA-018 **Partial** | Sections 3.2, 3.4, 3.6; I.1: explicit incoming writer/reader histories and no-local-producer paths, not an implicit fresh region entry. | `OL::computeUninitialized` retains `mayHaveNoPriorFullWrite`; `FP` creates one incoming-writer sentinel; `PA::interpretAt` exposes a boolean. Neither accepts a complete incoming provenance interface. | None: `F1` filters incoming sentinels out of the compared demands. | Supply incoming roots/readers and original interface obligations through child analysis; steps 4/6/7/9. A boolean cannot identify the missing producer or incoming readers. |
| PA-019 **Partial** | Sections 3.2, 3.4, 3.6: retain complete physical/typed requirements, independent readers, intraproducer hazards, outside uses and zero-reader WAW; no ordinary R/R hazard. | `OL::buildRequirements` generates marginal RAW/WAR/WAW per cell and retains both engine roles; `supportBetween` includes other-cell affected requirements. `PA` keeps typed records separately. | `T`, `F1/F3`, `H4`; corpus counts alone do not establish per-path completeness. | Guarded and incoming requirements must reach the public model without replacing independent obligations by one episode label; steps 4/6/10. |
| PA-020 **Partial** | Section 3.4; 5.2.4–5.2.5; I.1: separate membership, origin enumeration and witness extraction on qualified positive-length paths; do not eagerly build every witness. | `OL::originsAt` and `lifecycleAt` enumerate matrix rows; `buildRequirements` eagerly expands source-target pairs. `firstUse/lastUse` expose may frontiers, not the full MayOrigin/Origins/Witness contract. | None for a separated public demand/witness service. | Add lazy qualified membership/witness access and preserve same-site recurrence versus zero-length identity; steps 4/6/9/13. The split-site backend is an equivalent implementation option, not a second mandatory backend. |
| PA-021 **Partial** | Sections 3.4, 3.6 D4; 5.2.6: original marginal sequence/choice/repetition summaries act on incoming origins; mode restrictions remain explicit. | `OL::solve` answers whole-graph may flow by fixed point. `FP::transfer` is internal and acyclic. Neither is a public reusable child provenance transfer with incoming/mode parameters. | `H2/H4` report loop flow; no parametric summary assertion. | Supply supported child summary/application interfaces for steps 4/9. Do not mistake a computed exit row for a transfer on arbitrary input or demand an unnecessary second backend. |
| PA-022 **Partial** | Section 3.6; 5.2.6/5.2.8; I.1: nearest matching roles retain entry/exit/write boundaries, ambiguity and qualified positive-length recurrence. | `OL::frontier(first/lastUse)` traverses the original graph with starts/stops and read/write filters; `RQ::segment` finds lexical write-delimited candidates; `PA` exposes these may views. | `H3/H4` report may-frontier/support cases. | No common selector/occurrence-qualified nearest-use service with complete boundary markers is supplied. Steps 2/7/8/9/10 must derive the specified relations, not turn a may path into a constant distance. |

### Dedicated All and MayAfter service checks

These entries separate the existing typed services from the broader boundary
and interval integrations above. They are not independent completion analyses.

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-079 **Implemented** | Section 3.4, p. 10; F typed query contract: All contains every imported original read/write for the selected owner, cell and optional engine/mode filter. Only complete may exclusion proves no access. | `OL::all` scans reachable owner members; `PA::all` caches owner/cell results then filters mode/engine. `PA::boundary` also consumes this storage summary for retained may accesses; `RQ` does not use the common public All filter. | `H3/H4` describe query checks; no committed expected All set is asserted by `TEST`. | Conditional on complete `SI/OS` effects and the specified owner. Keep unknown footprints conservative. Steps 10/14 add exact expected sets and filtered/unfiltered equivalence; PA-051 tracks common boundary integration separately. |
| PA-080 **Partial** | Section 3.4, p. 10; F, p. 68: MayAfter traverses the qualified original continuation with an explicit stop/inclusion convention; loops use finite flow rather than one unrolling; NoHit, May and Unknown differ. | `OL::mayAfter -> PA::mayAfter` walks the original graph with a visited epoch, owner checks and stop sides. Positive witnesses return May; no-hit results across a backedge or repeated endpoint are withheld as Unknown. This is a public query API; neither `RUN` nor the committed corpus runner calls it. | `H4` reports May/NoHit/Unknown cases; no committed recurrence-qualified exclusion assertion was identified. | Steps 2/9 supply complete occurrence and structural-cut interpretation, then allow the specified supported exclusion cases. Step 14 tests child versus owner horizons, bypass/backedge paths, inclusion, and last-use with completion still pending; no exclusion supplies a receipt. |

## 4. Committed factored core and its integration boundary

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-023 **Implemented** | Section 3.3; I.1: shared Empty/Incoming/Access/Both/Choose/Demand expressions on an already qualified acyclic cell projection; no expansion of independent guard valuations during formation. | `FP::{append,both,choose,demand,transfer}` shares node IDs; `OL::factored -> PA::interpretAt.factoredUse`; `TEST` evaluates the core directly. | `F1/F2`. | This is the committed core, not absent work. Defining-value guards, parameterized entries and native integration remain PA-027–PA-031, steps 4/5. |
| PA-024 **Implemented** | Section 3.4; I.1/I.6: reads query incoming writers; writes/RMW form old writer/reader demands before definite replacement; later writes do not erase generated demands. | `FP::transfer` stores prior roots, adds RAW/WAW/WAR demand nodes, then replaces writers/readers on `fullWrite`. Roots remain shared; actual evaluator is `TEST`. | `F1` checks the conditional RMW's old-state hazards and final demands. | Positive imported full writes require PA-006; demand consumption requires PA-030/PA-036. Core acceptance is not native acceptance. |
| PA-025 **Implemented** | I.1: an unproved overwrite retains conservative older writer and reader histories. | `FP::transfer` uses `Both` for a possible write and leaves earlier readers; `OL` also keeps older may origins. | `F3` checks both retained writer and reader lists. | Qualify the physical weak/strong distinction in actual PTO and retain effect witnesses; steps 3/4. `complete=true` is formation status, not exact effect qualification. |
| PA-026 **Partial** | Sections 3.4, I.1: backward expressions use the same choices, with an explicit continuation and dual boundary convention. | `FP::transfer(..., true)` fills `nextWriters/nextReaders`, reversing sequences; `FactoredUseResult` exposes roots; `TEST` checks one next-writer query. | `F1` checks the conditional next writer after A, not the full backward/incoming service. | Backward entry is hardcoded empty at the whole projection exit. Parameterize the continuation, test readers/RMW and integrate consumers; steps 2/4/9. |
| PA-027 **Partial** | Sections 3.3, I.1: identical original defining values in the same occurrence scope select related origins, demands and boundaries consistently, even at separate tests. | `FP::Choose` uses `region.originalOwner`; `RQ` interns an OriginalBoolean atom by if-owner. Each individual if is shared across its own output fields. | `F1` has distinct synthetic owner IDs for its two independent guards. No same-SSA-value/multiple-if assertion. | Canonicalize original value plus occurrence scope across services, with compatible evaluation domains; steps 4/5. Do not enumerate valuations to recover this relation. |
| PA-028 **Partial** | I.1/I.6: transfers, composition and application share supplied incoming expressions; incoming readers and no-producer alternatives are explicit. | `FP` passes shared roots internally but starts with `{incoming-writer, empty-readers, empty-demands}`; no incoming DAG/state argument exists. | None for a nonempty incoming reader interface; `F1` starts with a local full write. | Implement parameterized original interfaces and child applications, preserving witness/guard identities; steps 4/9. An internal method named `transfer` is not this completed interface. |
| PA-029 **Partial** | Sections 3.4/3.6; I.1/I.6: restricted acyclic formation is reusable within the supported program analysis; repeated regions require qualified summaries rather than dynamic unrolling. | `FP` rejects **any For/While in `original.body`** before formation; `OL::factored(cell)` is lazy but always passes that entire body. `PA` attaches the incomplete result. | `C` contains loops but does not inspect the factored result; no admitted local-projection integration test. | Derive qualified projections/summary applications; steps 4/9/10. The fixed-use theorem does not solve arbitrary recurrence, but blanket whole-function rejection does not complete supported integration. |
| PA-030 **Partial** | Sections 3.2–3.5, 4.1; I.1: factored demand roots supply the public original obligation service, with conservative marginals available when qualification fails. | The pointer `InterpretedRequirement::factoredUse` is present. `OL::buildRequirements`, `PA::requirementsAt` and `interpretAt` still use marginal origins; no current caller derives applicable obligations from `Demand` nodes. | `F1` evaluates synthetic expressions directly; `C` interprets marginal IDs without evaluating those nodes. | Steps 4/6 connect roots to stable guarded IDs, lazy membership/witnesses, and public callers. Do not describe the core as either absent or fully integrated. |
| PA-031 **Partial** | Section 3.3; I.6: restricted `O(q+b)` formation on qualified acyclic exact-cell inputs; count query/expanded output, guard work and repeated-control formation separately. | `FP` allocates shared constant-arity nodes without valuation products, but scans the original syntax/effect incidences per requested cell and allocates per-operation arrays. No factored-work accounting is exposed. | `F2` supplies one node-count upper check. No independent scanner or scaling/work series. | Step 4 tests admitted executions against a separate concrete scan and records actual visited syntax/incidences, incoming DAG and node work. Step 14 separates all materialization/query costs. |

## 5. Original-value, arithmetic and endpoint qualification

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-032 **Partial** | Section 3.6 D2/D3; I.2: qualify original loop bounds, integer widths/signedness/overflow, finite address selection and safe conditional evaluation. | `OS::{importControl,deriveAddress}` call `SyncSlotMapping::AnalysisContext::domain`, `derive` and `evaluate`; coordinate addition checks exist. `RQ` consumes `qualifiedCounted`. | `H2/H3` report induction/carried spellings and periods. No combined endpoint-arithmetic contract assertion. | Consolidate the supported arithmetic/value qualifier used by D2, D3 and interval bounds; step 5. Existing address qualification is not proof that every generated predicate is total. |
| PA-033 **Partial** | Section 4.1, p. 16; F.2: distinguish Available, NeedsCompletion with named prerequisite, NotObservableHere and unresolved qualification at each endpoint. | `RQ::predicateAvailable/availableAt -> PA::guardAvailableAt` returns a dominance-based boolean; boundary records use booleans, not the four outcomes. | `H3` reports a late unavailable predicate; no four-outcome regression. | Step 5 separates definition availability from asynchronous completion and qualification failure. A value defined later cannot guard an earlier cut. |
| PA-034 **Partial** | Sections 2.2, 4.1; F.2: an earlier asynchronous producer may enable an endpoint only through an independently dischargeable prerequisite; no circular guard justification. | `PA`'s typed-producer traversal records source phases/incoming unknowns; `TypedOriginalRequirement::occurrenceQualified` remains false. `RUN` counts records but cannot discharge them. | None for a guard-enabling typed prerequisite or a circularity rejection. | Steps 5/6 qualify the original value, source occurrence and enabling deadline. Executing the enabling transfer is Phase B, not a new Phase A receipt. |
| PA-035 **Partial** | Sections 3.3, 4.1; I.1: retain the earlier defining value and its occurrence through branches/loops; do not reevaluate a changed carried scalar or speculate outside its domain. | `OS` retains original SSA sites and `TypedOriginalRequirement` holds `Value`; `FP/RQ` guard IDs remain owner-based and no common old-value/occurrence transport is supplied. | None for repeated same-value tests or changed carried values. | Steps 4/5/9 derive value/occurrence identity and totality. Test invariant, varying, late-defined and changed-carried conditions separately. |

## 6. Stable guarded obligations and construction-facing identity

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-036 **Partial** | Sections 3.2, 4.1; I.4: an immutable ID denotes a guarded physical/typed witness family, source-use relation, consumer, deadline, scope and continuation, independently of placement. | `OriginalRequirementId` is `{deadline,index}` for an eagerly generated marginal pair; `OriginalRequirement` retains hazard, cell, engines, whole-operation scope and effect pointers. `RUN/TEST` consume these IDs. | `C` checks direct subscription indices; `T` checks three requirements. | Step 6 adds guarded applicability, incoming cases and full occurrence/continuation identity without putting keys or placement alternatives into the ID. |
| PA-037 **Absent** | Section 4.1: request groups reference one or more immutable obligation IDs; grouping, discovery reasons and placements do not create residual weight. | `DecodedRequirement` and `InterpretedRequirement` describe one marginal record. No separate request-group model or grouping consumer exists. | None. | Steps 6/12 define groups and compatibility; test two descriptors/groups referencing one unchanged obligation. Required-versus-selected residual filtering belongs to Phase B. |
| PA-038 **Partial** | Sections 3.2, 4.1; Proposition 5.47: keep all effect witnesses, coalesce duplicate reasons and preserve semantic family identity under uniform witness subdivision. | `OL::effectsByRole/buildRequirements` coalesces cell/kind reasons while keeping shared source/target incidence lists. Partitioned cells still generate separate marginal IDs. | `H4` reports incidence/index checks; no uniform-subdivision identity assertion. | Step 6 retains family IDs behind expanded witnesses; step 12 deduplicates placements without duplicating obligations. Nonuniform geometry refinement may legitimately change relationships. |
| PA-039 **Partial** | Sections 2.2, 3.4, 4.1: physical RAW/WAR/WAW, typed/control/address and incoming prerequisites remain complete and separately scoped at original deadlines. | `PA::typedRequirementsAt/typedSubscriptionsAt` index branch/for/while inputs and translated allocation-address dependencies; `OL` indexes physical requirements. `RUN` counts typed records separately. | `T/C` do not assert typed records; none for complete typed/incoming obligation integration. | Steps 5/6 join these into the public obligation service; audit all required address/guard dependencies, not just root allocation addresses. Unknown source/occurrence stays explicit. |

## 7. D1: fixed visits and alternative origins

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-040 **Partial** | Section 3.6 D1, pp. 11–12: qualify a fixed visit through its relevant physical-origin relation; an intervening access invalidates it only when its effects require that conclusion. | `OQ::fixedVisit -> PA::fixedVisitFor/interpretAt` requires noncyclic distinct sites, source dominance, and rejects **any** intervening same-cell access. It has no hazard-specific parameter. | `T` asserts three narrow fixed visits; `H3` reports another positive. | Step 7 must support one writer with several readers. Intervening reads must not sever that writer's relation; optional branches not reaching the target must not cause arbitrary rejection. |
| PA-041 **Absent** | D1; I.1: exactly one applicable guarded origin on each participating path, with an explicit incoming case on a no-local-producer path. | `PA::alternativeSourcesFor` returns a may vector; `alternativeGuardsQualified` remains false. `FP` can represent a choice, but no D1 alternative-source decoder consumes it. | `F1` tests conditional core provenance only, not D1 alternatives. | Steps 4/6/7 connect guarded roots and incoming interfaces to correspondence; test conditional replacement, no-producer branch and incompatible alternatives. |
| PA-042 **Partial** | D1 and Section 4.1: pair source/target occurrence conditions, qualify them independently at their endpoints and normalize alternatives in original site order. | Current marginal ordering and bank lists are deterministic; `RQ` provides leaf predicates, but no matched alternative-source frontier with separately qualified endpoint conditions exists. | None for a paired guarded D1 record. | Steps 5/7 retain both endpoint conditions and the same original case; unknown one side must not erase the other's useful answer. |

## 8. D2: periodic physical selection and domains

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-043 **Partial** | Section 3.6 D2: derive an explicit finite permutation of disjoint physical banks; follow each cycle once and use its own length, independent of selector spelling. | `OS::deriveAddress -> OQ::bank` checks one address per represented residue, pairwise disjointness, qualified counted owner and mandatory uses. Distance is the address-vector length; fewer than two addresses are refused. | `H2/H3` report periods two/three and equivalent spellings. | Step 8 supplies the stated permutation/cycle procedure, including a one-bank cycle and separate cycle lengths where represented; do not certify from modulus alone. |
| PA-044 **Partial** | D2: `prev(i)=i-Bc`, `next(i)=i+Bc` with original entry/exit domains and initial/reused/successor/final cases. | `PeriodicSameRole` stores `period`, `firstOrdinalWithLocalPredecessor` and an incoming-case boolean; no complete successor map/domain or boundary interface is returned. | `H3` reports the period-three predecessor/first-three-use case. | Step 8 derives both maps and qualified domains with original arithmetic. Step 9 transports them; initial uses cannot be silently treated as fresh. |
| PA-045 **Partial** | D2 and Section 3.4: relate the relevant producer and preceding reader roles of the same physical bank, not only repetitions of one instruction. | `PA::interpretAt` admits periodic interpretation only for `source==target`, one shared bank candidate, one participating operation and matching hazard-role incidences. | `C` reports bank relations but all kernel occurrences Unknown; `H3` covers only the restricted same-role case. | Step 8 supports distinct write/read roles and rejects genuinely interfering accesses while preserving their demands. A physical bank candidate is not an occurrence pair. |
| PA-046 **Implemented** | D2: independent selectors stay independent; noninjective/overlapping physical maps retain may footprints but receive no permutation certificate. | `OS` stores per-memory/owner address relations; `OQ::separated/bank` withholds exactness for overlapping/multiple addresses; no joint LCM history is constructed. `PA` retains candidate indices. | `H2/H3` report independent/noninjective and mixed-incidence cases; `C` only counts relations. | Preserve this behavior while extending positive D2 in step 8; add committed assertions and charge the explicit bank population, not a huge implicitly encoded modulus. |

## 9. D4: composition, unfinished uses and re-entry

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-047 **Partial** | Section 3.6 D4: compose child first/last summaries in the original owner, retaining predicates, original entries/exits and origin identities. | `RQ::sequence/query/segment` compose reader DAGs; `OQ::children` reports child membership. `PA` uses lexical owner and support candidates rather than a complete child-use interface. | `H3/H4` report optional children and reloads. | Steps 2/9 connect these summaries to qualified incoming origins and continuations; test transparent child wrappers rather than just identical counts. |
| PA-048 **Absent** | D4: a boundary between complete physical uses transports an already qualified D1/D2 relation; it does not discover a new general recurrence. | `OQ::children` returns skip/repeat metadata but no predecessor/successor transport certificate. No public D4 complete-use transport consumer exists. | None for a transported qualified relation. | Step 9 represents and checks the imported child relation and boundary premise. This specified composition cannot be deferred as an open general matching problem. |
| PA-049 **Partial** | D4 and Section 3.2: a boundary inside a read episode retains its enclosing owner; empty children/re-entry do not reset physical history; a reset/reload is an actual relation change. | `OL` graph flow retains may histories and `supportBetween` flags reload/re-entry; `RQ` composes reads. `PA::commonOwner` ignores the declared continuation and no exact re-entry map is supplied. | `H3/H4` report may-level cases; PyPTO corpus retains selector resets but has no expected correspondence assertion. | Steps 2/9 test unfinished-use boundaries, zero-use children, local selector resets and genuine overwrite. General coordinate inference remains open only beyond qualified supplied relations. |

## 10. Exact structural frontiers and the interval procedure

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-050 **Implemented** | D3; Lemma 5.22: on an admitted exact read-only cell/engine projection, empty/leaf, sequence and original choice derive shared nonempty and first/last expressions. | `RQ::{derive,sequence,guarded,unite}` uses shared prefix/suffix predicates; `PA::decodeAt` obtains a segment's structural reader result. | `H3/H4` report optional-child frontiers. `F2` tests provenance growth, **not** this frontier procedure. | Commit expected guarded first/last/no-hit evaluations in step 10. This component's presence does not qualify an episode, availability or paired endpoints. |
| PA-051 **Partial** | Sections 3.4/3.6; Proposition 5.23: All is a complete may-access filter; use one fixed-selector structural procedure for readers and first conflicts, skipping only proved no-hit subregions. | `OL::all -> PA::all` provides cell/engine/read/write may queries. `RQ` has private write/reader summaries but is not a common All-filtered arbitrary-conflict selector; `PA::boundary` supports only part of that surface. | `H3/H4` reports; no committed All-filtered-versus-unfiltered equivalence test. | Step 10 connects All to the common selector with identical normalized outcomes; unknown effects must block exclusion. Existing All functionality must be retained, not reimplemented as completion state. |
| PA-052 **Partial** | D3: counted repetition selects first/last iterations only for invariant body participation and a qualified reader role; zero trips stay explicit. | `RQ::derive(For)` uses `qualifiedCounted`, body nonempty availability at the loop, LoopNonEmpty/HasPrevious/HasNext atoms; `PA` reads the structural result. | `H3` reports invariant versus varying readers. `C` does not assert those predicates. | Steps 5/10 establish total available original arithmetic/participation and public positive cases. General varying participation is not solved by this rule; the specified interval fragment is PA-057. |
| PA-053 **Partial** | Section 3.6; Section 3.3 directional independence: distinguish NoHit, exact frontier plus no-hit predicate, and Unknown with may facts; qualify first and last separately. | `RQ/PA` expose these statuses and shared predicates, but `RQ::available` tests **both** first and last and exports one availability boolean. `PA::boundary` uses that joint result. | `H3` reports a late guard; no assertion retaining an available first boundary when only the last is unavailable. | Steps 5/10 preserve independent positive answers and explicit conditional no-hit cases. A failed last guard must not erase readiness before the first read. |
| PA-054 **Partial** | Section 3.6: FirstConflict/LastRelevantUse are original selector boundaries, not the first currently unfulfilled causal requirement, and cover supported writer targets too. | `PA::firstConflict/lastRelevantUse -> boundary`; WAW and first-WAR cases return Unknown unconditionally. Read cases require episode/occurrence flags not set by marginal construction. | `C/T` do not assert these answers; no positive writer-frontier test. | Steps 7/9/10 supply exact supported conflict selectors/occurrences. Always Unknown for a specified supported case is a gap, not completed conservative parity. |
| PA-055 **Partial** | D3/D4: delimit a read episode by a definite producer and next conflicting write; retain internal hazards, independent reader engines and zero-reader WAW. | `RQ::segment` stops at any overlapping writing child; `OL::supportBetween` separates `stablePhysicalInterval` from `generationEstablished`; `PA::qualifiedBoundary` needs generation plus FixedVisit. | `H4` reports reload support; `F1` checks hazards outside this episode client. | Every imported generation proof is blocked by PA-006. Steps 3/7/9/10 must connect the same qualified interval, including other-reader and empty cases, without dropping outside demands. |
| PA-056 **Partial** | Section 3.6; Definition 5.18 and 5.2.10: justify exactly-once participation and access-before/after boundaries, with independently justified episode transitions. | `RQ` constructs restricted D3 expressions; no general episode endpoint-count/access-order monitor or complete public participation certificate is exposed. `participationDemands` only collects selected observation needs. | `H3` reports restricted D3 cases; no independent count/access-order oracle in `TEST`. | Step 10 verifies the supported derivation premises and concrete frontiers. The finite count/access-order monitor procedure remains absent; it cannot be credited merely from the D3 expressions or invent resets/occurrence matching. Scope its implementation or equivalent certificate explicitly at review. |
| PA-057 **Absent** | I.2, Lemma I.2: for one qualified unit-step read site with invariant lower/upper bounds and no intervening write, intersect bounds; return NoHit or first `l` and last `h-1`. | No max/min interval-participation derivation or consumer exists in `RQ/OQ/PA`; D3 only handles its invariant-participation case. | None. | Steps 5/10 implement this specified fragment. Test clipped/empty intervals, zero/one trips and a last reader before the final loop iteration. Not an open general-affine problem. |
| PA-058 **Absent** | I.2: derive `l=max(0,L...)`, `h=min(N,U...)` with qualified signedness/overflow and guarded `h-1`; description size depends on bounds, not numeric trips. | No interval extrema/empty-case endpoint record is produced. Existing counted/address helpers do not implement this procedure. | None. | Steps 5/10 test unavailable bounds, overflow and conditional evaluation without speculative empty-case subtraction; step 14 measures expression/output populations separately. |
| PA-059 **Partial** | Section 3.3; I.1/I.2: use physical subcell overlap to select source occurrences; where qualified, disjoint pieces `A[j]` versus `A[k]` normalize to a singleton interval. | `SW` partitions exact subranges and `OS` retains physical selector relations; no occurrence-bound normalization connects them to the interval procedure. | `H2` reports geometry probes; no original-coordinate singleton-interval assertion. | Steps 3/5/8/10 qualify address equality and `k+1` arithmetic. Geometry cannot create a fictitious producer for an untouched subrange. |

## 11. Independent directional covering boundaries

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-060 **Absent** | I.3: on a qualified finite SESE cut interval, source query trims a proved no-hit suffix, recurses through sequence, and returns an access-after, conditional-join or finite-loop-exit cut. | No source-cover query/result exists in `PA/OL/RQ`. A may last-use traversal and a reader frontier are not this procedure. | None. | Steps 2/11 implement the declared cut rule. Test `A; compute g; if(g) B; U`: source is after the conditional, before unrelated U. |
| PA-061 **Absent** | I.3: the independent target dual trims a no-hit prefix and returns the corresponding before/entry cut without moving original deadlines. | No target-cover query or result exists; `PA::firstConflict` is not a covering-target service. | None. | Steps 2/11 implement and test the target dual, including an exact source with covering target and the converse. |
| PA-062 **Absent** | I.3: record reference coverage, occurrence multiplicity, observations, extra source/target work, and executions on no-access paths. A boundary supplies no protocol bracket. | `OriginalBoundaryResult` has only exact-reader/may fields; no covering scope or no-access-execution contract. | None. | Step 11 carries these facts to descriptors; step 12 preserves them for later packet checks. An optional empty interval must not manufacture a reader or release. |
| PA-063 **Absent** | Sections 3.3/4.1; I.3–I.4: obtain exact and covering answers independently on both sides, including a prederived cover when an exact answer is known but later unbindable. | No covering alternative is represented; the existing exact reader availability is also coupled (PA-053). | None. | Steps 10/11/12 form alternatives before any key probe. Failed binding must not trigger a new ancestor/cut search. |
| PA-064 **Partial** | I.3: transparent sequence wrappers preserve the semantic cut; all-no-hit returns exclusion, unknown effects block trimming, unqualified repetition returns an obstruction. | Existing `RQ` has sequence/no-hit/Unknown behavior, but reader intervals are lexical slices and public cuts do not name joins/exits. No covering procedure consumes those facts. | `H3` covers reader summaries only; no directional wrapper-equivalence test. | Steps 2/11 test wrapped/unwrapped cut identity, optional emptiness, uncertain effects and unqualified loops. Arbitrary IR rewrites need not preserve stable IDs or a greedy plan. |

## 12. Request groups and the three descriptor slots

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-065 **Absent** | Section 4.1; I.4: immutable placement descriptors attach both original boundaries, applicable cases, occurrence pairing, extra scope and justified support/observation requirements to group/obligation IDs. | No distinct descriptor type/builder exists. `OriginalEndpointCandidate` is one guarded operation cut, not this paired descriptor. | None. | Steps 6/11/12 provide descriptors with unresolved components preserved; no keys, predicted receipts or selected proof facts enter Phase A. |
| PA-066 **Absent** | I.4: form at most three prescribed roots per group—preferred, source-cover, target-cover—and deduplicate structurally identical complete slots. | No descriptor repertoire generator or actual consumer exists. | None. | Step 12 combines preferred independent answers exactly as specified, not an arbitrary source-by-target product or a mandatory fourth both-cover candidate. |
| PA-067 **Absent** | Section 4.1; I.4: compose corresponding slots through the same original Choose, not all arm combinations; retain observation requirements and identical original obligation identities. | `FP` and `RQ` each have choices but no descriptor-level shared-choice composition or original-observation skeleton. | None. | Steps 5/12 test mixed exact/covering arms, hidden/unavailable guards, duplicate slots and unchanged IDs. A selected proof-context obstruction remains Phase B; Phase A cannot repair a lost causal correlation. |

## 13. Preparation, subscriptions and scoped consequences

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-068 **Implemented** | Sections 3.1/3.5: index existing original requirements by deadline/source and retain distinct engine directions and deadlines. | `OL::buildRequirements` fills `byDeadline/bySource`; `PA` builds direction and typed source/deadline indexes. `RUN/TEST` consume deadline records and `TEST` checks source indices. | `C/T`, plus `H4` for duplicate-incidence/index cases. | This is implemented for the **current marginal universe**, not the complete guarded/descriptor repertoire. Steps 6/13 extend it without collapsing reader engines or deadlines. |
| PA-069 **Partial** | Sections 3.5/3.10: subscribe sufficient original sources before traversal, keep executable cuts distinct from analytical phase cuts, and never silently use a later current prefix. | `OL::buildRequirements` stores `SourceSubscription::position` at `enclosingAfter` and `sufficientPosition` at the source phase; `PA::subscriptionsAt` exposes them. | `C` checks subscriptions by `request.source.operation`; it does not test the differing enclosing-after case. `H3/H4` report multi-phase checks. | Steps 2/13 retain conservative executable hooks plus the precise obstruction for unavailable internal cuts. Test both positions and the source reached earlier than its consumer. |
| PA-070 **Absent** | Section 3.5: traverse all shared requirement/descriptor references to subscribe every promised original cut and recurring role before construction; do not evaluate guard valuations. | Subscriptions are made from eager marginal pairs only. `FP` is formed lazily in `interpretAt`; no traversal of expression/descriptor source references prepares a complete hook set. | None for descriptor/expression preparation; `C` checks the smaller direct set. | Steps 6/12/13 add the complete preparation walk, including structural cuts and explicit unresolved conservative hooks. |
| PA-071 **Absent** | Sections 3.5/4.1: follow the finite declared readiness/release/next-use support links with a visited-role set; keep index shifts on relations, not new dynamic nodes. | `OriginalSupportRequest` is a producer/reuse candidate; `qualifySupport` scans an interval. `participationDemands` is not a recurring support graph/closure. | None for role closure or source-timeliness under recursion of support links. | Steps 8/9/12/13 provide qualified finite role records and closure. Reserve no events and grant no circular completion from this original graph. |
| PA-072 **Absent** | Section 3.5, pp. 10–11: two-link source opportunity requires the **same dynamic middle occurrence**, compatible guard, scope and deadline, using original demands; it grants no credit. | `requirementsFromTo` indexes engine directions and `alternativeSourcesFor` lists origins; neither implements a scoped two-link consequence query. | None. | Step 13 tests valid cross-cell chains and false branch/iteration joins. Retain all original obligations and earlier subscribed source cuts; no arbitrary current-engine-prefix candidate. |
| PA-073 **Partial** | Sections 3.2/3.5: preparation completeness is relative to the declared repertoire, includes all promised subscriptions, and retains obstructions rather than substituting unnamed sources. | `PA::complete` reflects existing storage preparation, not completion of D1–D4, descriptors or support closure. `FactoredUseResult::complete` means acyclic formation only. `RUN/TEST` accept those limited meanings. | `C` explicitly tolerates Unknown interpretations; no repertoire-completeness assertion. | Step 13 exposes/checks preparation status against the declared roots/roles. Step 14 uses a simulated traversal to verify that each promised hook was registered before its cut. |

## 14. Integrated evidence, determinism and cost

| ID / status | Draft behavior and assumptions | Implementation entry -> actual consumer | Existing semantic evidence | Remaining prerequisite / later gate |
| --- | --- | --- | --- | --- |
| PA-074 **Partial** | Sections 3.6, I.1/I.5: deterministic normalization and sharing preserve query meaning, statuses and original positions without valuation/frontier-subset enumeration. | `PA/OL` use stable marginal indices; `RQ` interns predicates/frontiers and caches requested leaf conditions. There is no integrated normalized obligation/group/descriptor interpretation. | `F2` checks one factored size bound; `H3` reports shared reader expressions. | Steps 2/4/6/10/12 test equal complete queries and semantically different horizons; step 14 checks all public statuses and stable identities, not diagnostic witness formatting. |
| PA-075 **Partial** | I.6; Section 7.2–7.3: independent concrete provenance/frontier checks on small admitted inputs, plus formation tests without large valuation enumeration. | `TEST::checkFactoredProvenance` evaluates actual core expressions against handwritten fixtures; no general independent concrete provenance/frontier scan is present in that runner. | `F1/F2/F3` are useful bounded evidence, not full oracle coverage. | Steps 4/10 add independent expected semantics including incoming, backward readers, repeated guards and applicable supported recurrence; step 14 integrates them through PA. |
| PA-076 **Partial** | Section 7.2 and the implementation plan's final gate: exercise specified useful relationships through the public interface on the five development kernels, classifying each remaining Unknown. | `TEST::inspectRequests/analyze` visits every marginal requirement and prints aggregate counts. `RUN` samples one; neither asserts the intended kernel frontiers/relations. | `C`; historical 2,242 Unknown occurrences. | Step 14 adds the per-kernel assertions below. A successful parser, a bank count or an analysis-complete flag is not parity. |
| PA-077 **Partial** | Sections 2–3 and validation contract: preserve original IR and the shared-input existing autosync path when common effects change. | The current adapter is read-only and shares `SI`; no algorithm code changes are made by step 1. `RUN` still intentionally refuses construction. | `C/T` assert unchanged IR; existing-mode compilation is a historical handoff report, not a new dual-mode test here. | Steps 3/14 rerun focused shared-effect regressions and both actual modes. Keep the expected FrontierSynch construction-not-implemented diagnostic separate from Phase A semantic success. |
| PA-078 **Partial** | Sections 3.2/3.6, 5.6; I.5/I.6; G.2–G.3: charge syntax/effect projection, expressions, explicit banks, origin/witness output, query contexts and subscriptions separately. | `OriginalLifetimeStats` counts matrices/flow/requirements/frontier work; `RQ` has private expression/work counters; `FP` exposes nodes but no integrated formation counters. `TEST` prints only aggregate analysis counts. | `F2` checks stored core size; no end-to-end cost separation. | Steps 4/8/10/13/14 expose actual populations, including per-cell full scans and materialized outputs. Do not infer whole-pass linearity, bounded contextual width or subcubic compilation from the restricted formation result. |

## Kernel assertions required at step 14

These are **planned acceptance assertions**, not current passing tests. Use the
original effects, not manual flags, to derive them; retain each kernel's other
physical and typed requirements.

Each exact relationship must satisfy the premises of a specified draft rule.
For unsupported recurrence or coordinate matching, assert an explicit `Unknown`
and identify the missing premise. Kernel-specific recognition is not an
acceptable substitute for a general derivation.

| Existing input | Required semantic checks |
| --- | --- |
| `vector_add.pto` | Distinct input and output storage lifetimes; prologue, guarded next load and final cases; supported bank-use predecessors/successors. A store's output-buffer lifetime does not itself extend a disjoint input's physical read. |
| `tilelang_gemm.pto` | Every relevant L1 piece read is retained; the useful L1 release source follows the last relevant extract rather than later computation on L0 copies. Keep separate A/B effects, guarded prefetch and operand-bank reuse. |
| `pypto_gemm.pto` | Mat/L0 selectors and child resets preserve the qualified owning physical-use sequence; a child index starting at zero does not prove fresh storage or a new incoming interface. |
| `expert_fa_cube.pto` | Resident Q's participating readers and clipped/shortened batch cases; independent QK/PV/operand storage obligations. Unsupported external producer matching remains an explicit invocation premise, not an invented local producer. |
| `expert_fa_vector.pto` | Early physical release of `io`, separate store-buffer reuse, and retained other workspace/accumulator requirements; later computation on disjoint storage does not move the original read frontier. |

FA's cube/vector files are separately analyzed cooperating bodies. This local
Phase A gate does not prove their cross-core notification protocol, remote
producer matching, launch configuration, numerical behavior or device speed.
The unsynchronized payload files must not be executed as standalone device tests.

## Reference procedures, backend choices and genuine draft-open work

An item in this register does **not** excuse a missing `PA-nnn` procedure.

| Draft item | Implementation/evidence and correct classification |
| --- | --- |
| I.1, Lemma I.1: finite continuation-congruent merging | No labelled-query-graph minimizer is supplied. The draft makes this a reference sufficient merge criterion; equality of complete shared query/continuation descriptors is the initial implementation route. Implement that equality (PA-014/PA-074); do not first build a global history product to minimize it. No test of the optional minimizer is claimed. |
| Section 5.2.5 split-site origin reduction; I.1 nearest matching query | No dedicated split-site origin backend is present in the audited path. An equivalent qualified original-graph query may implement the required answers (PA-020/PA-022). Positive-length correspondence, boundaries, membership and witness semantics remain required; the choice of graph backend is not an independent parity gate. |
| I.5, Lemma I.5: separator/treewidth backend | Not implemented, and optional. The bound assumes an already qualified contextual graph and its decomposition; it is not a bound for forming that graph or for the selected event graph. Complete keys and charged formation/output remain required in PA-014/PA-078. No backend performance evidence is claimed. |
| Section 5.2.10 finite count/access-order monitors | No general monitor is exposed here. This specified certificate procedure assumes a supplied qualified finite projection and justified episode transitions; it is not permission to discover resets or enumerate candidate protocols. Its absence is recorded in PA-056, separately from D3's existing constructive fragment; it is not an open research problem. |
| I.3, Lemma I.3 sufficient-boundary packet | The isolated readiness/return bracket is a proved example, not a Phase A rule installing a bracket after every cover query. Phase A must implement the two directional queries and record extra executions. Constructing/checking the actual packet belongs to Phase B; no C++ packet implementation is credited here. |
| Sections 3.6 D4 and I.1 general matching limits | **Supported transport of already qualified relations is specified and missing/partial** (PA-047–PA-049). Automatic exact coordinate maps for arbitrary mixed-write loops, noninjective selectors, enclosing evolution or resets beyond those supplied relations remain open. Preserve useful may relations and name the missing premise; do not call all re-entry an open problem. |
| I.2 arithmetic limits | Arbitrary affine/mixed-write/multiple-varying-reader analysis and a general predicate solver are not supplied. The explicit invariant-bound interval rule and its arithmetic/availability checks **are** supplied and remain required (PA-057–PA-059). |
| Sections 3.7–3.10, 4.1 and E6/F8 | Selected causal frontier/snapshots, matching, event rearming, observation-uniform binding, proof contexts, loop hypotheses and sealing are Phase B. Their original occurrence, observation and finite support inputs are Phase A obligations above. A precise original case cannot replace a missing selected conditional guarantee. |
| I.6 remaining target and Section 7 | General combined representation adequacy, economical interacting packet construction, unrestricted recurrence inference and a whole-pass complexity/optimality theorem are not established. Restricted procedures already specified above must still answer their admitted supported cases. |

## Fourteen-step gate map and review ledger

The four former broad gates are replaced by these separately reviewable gates.
Numbers are those of the implementation plan, not the earlier source-port stages.
Rows can have several prerequisites; the map names the principal review surface.

| Step | Increment / checklist surface | Current scoped review state |
| --- | --- | --- |
| 1 | Complete checklist, source/consumer/test mapping, baseline and evidence correction; all rows | **Independent coverage review accepted, 2026-09-24**, after clarifying the supported-rule scope of kernel assertions. Acceptance covers the inventory, not implementation parity. |
| 2 | Common records, legal cuts, owner and complete interval/cache interpretation: PA-009–PA-014, PA-080 | **Accepted at increment scope, 2026-09-24.** See integrated review; overlapping later gates remain open. |
| 3 | Shared full-cell coverage and read/write/alias qualification: PA-003–PA-008 | **Accepted at increment scope, 2026-09-24.** See integrated review; overlapping later gates remain open. |
| 4 | Factored service and qualified projections/entries/next-use: PA-016–PA-031 | **Accepted at increment scope, 2026-09-24.** See integrated review; overlapping later gates remain open. |
| 5 | Original-value/arithmetic/availability qualification: PA-032–PA-035 | **Accepted at increment scope, 2026-09-24.** See integrated review; overlapping later gates remain open. |
| 6 | Guarded physical/typed/incoming obligations and lazy queries: PA-018–PA-020, PA-030, PA-036–PA-039 | **Accepted at increment scope, 2026-09-24.** See integrated review; descriptors, cross-partition mapping and later gates remain open. |
| 7 | D1: PA-040–PA-042 | Narrow fixed visit present; specified generality pending. |
| 8 | D2: PA-043–PA-046 | Bank candidates and same-role predecessor subset present; specified cycle/role/domain service pending. |
| 9 | D4 and child provenance interfaces: PA-021–PA-022, PA-047–PA-049, PA-080 | Reader composition/may histories present; qualified transport pending. |
| 10 | Common exact boundary procedure, D3 and interval rule: PA-050–PA-059, PA-079 | Structural reader DAG present; integrated selectors, qualification and interval procedure pending. |
| 11 | Directional covering boundaries: PA-060–PA-064 | Pending. |
| 12 | Groups/three descriptor slots and shared choices: PA-037–PA-038, PA-065–PA-067 | Pending. |
| 13 | Complete adjacency, source/support preparation and consequences: PA-068–PA-073 | Direct marginal indexes/subscriptions present; complete declared repertoire pending. |
| 14 | Public-interface examples, independent semantic assertions, unchanged IR/shared path, accounting: PA-074–PA-078 and all earlier rows | Final integrated review pending; no aggregate acceptance from component approvals. |

For each later increment record its implementation commit, changed checklist IDs,
exact assertions/commands and results, independent reviewer decision, findings
fixed, and any explicitly named remainder with its later gate. A partial
acceptance must retain that remainder. An `Implemented` source row with only
historical or missing semantic tests is not automatically an accepted gate.

The independent step-1 reviewer checked the coverage map against draft
`4905f8a`, checked all behavior families against implementation `0a38c8e9f`,
and accepted the status/evidence distinctions and fourteen-step coverage.
The sole required correction was to make the supported-rule scope of the
kernel assertions explicit; that correction is included above. Local patch
applicability, whitespace, checklist structure and source links were checked.
No compiler or runtime tests were run for this documentation review.

Step-1 reviewer checklist: compare this inventory with every location in the
coverage map, confirm each row's consumer and evidence level, check that the
committed core is credited without overstating integration, and verify that
specified missing procedures are not hidden in the reference/open register.
Review the supported positive cases as well as conservative failure behavior.
At step 14 repeat the entire review against the then-current implementation.

## Step 13 integrated implementation delta

The current original-data consumer is `ProgramAnalysis::preparation()`, constructed
immediately after `OriginalRequests::build`. This section supersedes the step-1
baseline statuses for PA-068–PA-073 within the following declared scope.

| ID | Current implementation and consumer | Evidence / remaining gate |
| --- | --- | --- |
| PA-068 | Deadline families and factored physical cell/role source buckets; explicit typed-family source buckets. Membership remains in `OriginalObligations`; no engine maximum replaces families. | Native preparation test and corpus typed-adjacency assertions. |
| PA-069 | Frozen hooks retain sufficient analytical cut, executable original cut, associated references and obstruction. An enclosing cut does not become the exact phase cut. | Native early-source traversal and internal-phase fixture. |
| PA-070 | Iterative visited-node traversal of every declared descriptor root, both choice arms, source boundaries and typed completion prerequisites. Opaque source families contribute conservative incidence hooks. | Native frozen inventory and zero origin-enumeration assertion. |
| PA-071 | Finite declared support-role registry and visited-role closure; relations retain shifts. Invalid/missing qualifications stay explicit. | Native cycle and missing-role tests. Positive nonzero-shift qualification remains unsupported; retained links are not support credit. |
| PA-072 | Lazy two-link service for qualified invocation D1 relations. It retains three obligation IDs and conditional applicability, matches the same translated middle phase, and caches by the full interval. | Native cross-cell, conditional, opposite-branch, changed-interval and loop-visit tests. Recurring/D4 middle composition remains unresolved. |
| PA-073 | `formed()` and `repertoireComplete()` distinguish frozen preparation from discharged subscription premises; individual hooks retain obstructions. Stale versions invalidate public views. | Native stale snapshot and incomplete support/phase cases. Full integrated parity remains step 14. |

Counters expose descriptor nodes, references, access incidences, support roles and
links, consequence requests and cache computations. Origin output and witness
queries retain their separate original-obligation accounting. No whole-pass
linear-time or complete conditional-recurrence precision claim follows.
