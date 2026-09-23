# OAHS storage and fixed-plan analysis

The live [selected constructor](oahs-selected-plan.md) consumes the
[shared semantic extraction](oahs-shared-semantics.md) contract. Storage
provenance, causal completion, and diagnostic reports have distinct roles.

## Original storage and occurrence information

`closeStructuredSyncOrigins` closes storage origins through counted-loop,
while-loop, branch-result, and supported view forwarding. Loop-derived geometry
is widened to whole roots when exact ranges are unavailable. A finite delta
worklist propagates every discovered origin; no work cutoff removes obligations.
This hook runs on the handoff path and does not change legacy origin analysis.

`MemoryDependentAnalyzer::storageCoordinates()` qualifies nonempty,
nonoverflowing translated intervals. Known local addresses share physical
coordinates across SSA allocation roots. GM coordinates remain root-relative
under the production alias contract. Native import partitions qualified ranges
at their endpoints and retains their root provenance. Bounding intervals do not
prove whole-cell writes: native import does not infer `Access::definiteWrite`.

Unqualified footprints retain independent recurrence records. Repeated uses of
an identical footprint share its cell; distinct footprints receive conservative
overlap witnesses where required. Unknown overlap with two known disjoint ranges
does not merge those ranges. Distinct-footprint comparisons can be quadratic.
Unknown local addresses and unrepresentable geometry do not establish no-alias.

`StorageFrontierAnalysis` exposes possible previous/next writers and readers,
first/last-reader ambiguity, requirement reasons, and enclosing-loop provenance.
`lifecycleAt` includes continuation to future accesses and invocation exit;
`describeRequirement` separates physical overlap from qualified readiness/reuse.
A self-recurrence witness requires a positive-length control walk. A crossed
backedge alone proves neither an iteration distance nor event correspondence.

Definite writes may kill reference-succession origins. They do not complete old
writers, release readers, or acknowledge events. Child-region exits do not clear
outstanding accesses or protocol state.

## Causal checker used by the live pass

`CausalFrontier` owns the imported program and its immutable snapshots. It tracks
a shared causal relation over next-issue gates, earlier-finish aggregates, current
publications, and latest consumptions, plus per-cell access histories and possible
key occupancy. Primitive transfers require actual prefix and consumption paths.
Failed primitives leave the input snapshot unchanged and supply no credit.

Joins retain must-causal facts and possible occupancy. `checkCausalFrontier`
solves the original control graph, including zero-trip counted loops and mandatory
while-before execution. It checks supplied command words without constructing or
repairing a plan. The live native adapter uses this checker after reconstructing
the actual emitted commands against the original imported obligations.

## Residual diagnostic service

`Analysis.h::analyze(program, commands)` reports requirements at every represented
consumer. `complete` means supported input was analyzed; `verified()` additionally
requires no residual, protocol, resource, retirement, or diagnostic obligations.
`validateProgram` checks declarations and structure only. `verify` uses the
residual service; it is distinct from the live pass's causal-frontier checker.

The report retains producer/consumer phase identities, cells, read/write roles,
RAW/WAR/WAW and resource reasons, static control contexts, cut states, event facts,
and invocation obligations. Multiple readers remain separate prerequisites.
`captureStates=false` avoids copying snapshots without changing acceptance.

Unproved event endpoints receive no established completion credit. The diagnostic
interpreter revokes failing endpoint certificates and recomputes dependent states
until no more certificates are revoked. Invalid commands remain reported; they
are not silently removed. The number of revocations is finite, independent of
runtime trip counts. Full reports can contain quadratically many witnesses.

`analyzeHandoffSync(function, report)` returns the imported program, residual
report, and original MLIR phase mappings without changing the function. Success
means analysis completed, not that synthesis or device execution succeeded.
Original-operation pointers require the caller to preserve the input IR.

## Other retained diagnostic interfaces

- `ReplaySession` owns one original program and caches fixed-plan analysis.
  Changed regions restart from bottom; changed key layouts force cold replay.
  Invalid candidates do not replace the last complete checkpoint. Changed phase
  plans use full collecting analysis, not ordinary compact state.
- `PrefixQuery` owns an immutable program and command population. It reports
  backward original cuts and prospective prefix coverage. A prospective prefix
  neither reserves a key nor establishes an actual acquisition.
- `BundleQuery` replays a complete supplied candidate and reports obligation
  changes. Completed analysis alone is not acceptance. This is a diagnostic
  service, not an alternative constructor.
- `Phases.h` validates supplied final-block profiles and checks access-endpoint
  causality and order. These model-only checks do not enable native UnitFlag
  credit. The selected constructor refuses phase profiles without a matched
  adapter; `phaseNativeQualification()` remains disabled.

Snapshots and queries belong to their exact original program and command plan.
They cannot establish completion for a different plan or stronger hardware
contract. Pinned reference comparisons provide finite evidence under matched
contracts; they do not complete the general simulation proof or device
qualification. Real population accounting uses the
[evidence tools](../../tools/oahs-evidence/README.md).

## Shared physical-use and occurrence queries (2026-09-23)

`RequirementFrontiers` now owns the constructor's common physical-use view.
Each `(original site, physical cell)` record retains access roles, original
context, qualified release candidate and distinct successor/return deadlines.
Ordinary entry placement and recurring qualification consume these same records;
selected decisions retain the original relationships and candidate return
subscriptions. A return deadline is not an acquired receipt. Definite overwrite
still kills only succession provenance, never outstanding causal history.

`Control::correspondence` pairs all participating occurrences of two emitted
words by traversing original control with either no pending publication or its
actual source site. Repeated publication, unmatched acquisition and a pending
publication at exit disprove matching under the represented graph. All matched
site pairs are retained, including copies sharing an original command word.
Invalid queries and bounded-work exhaustion return unknown with no usable pairs.
The work bound is independent of event capacity; repeated queries reuse facts
about immutable control, never selected causal state.

Binding's interval check consumes that relation and inspects every occurrence of
an intervening event command. A canonical cut outside one interval cannot hide
another occurrence inside it. The existing straight-interval sufficient
certificate remains; this query alone does not generalize lifetime succession,
source coverage, neighbor rearming or endpoint motion. Same-word matching also
requires the binder's publication-before-acquisition ordering premise.

Tests include equivalent child words, missing participation, repeated publication,
exit balance, exhausted analysis, reload provenance, two readers with a common
next overwrite, unrelated storage and a hidden intervening key occurrence.
The new interface does not yet replace the single-owner recurrence grammar or
supply generation-scoped producer support; those are later refactor stages.


### Shared matching of endpoint sets

The occurrence query accepts publication and acquisition sets and canonicalizes
aliases and duplicates as one emitted word. Multiplicity and exact intra-word
order remain packet obligations. Recurring qualification, alternative-source
placement and loop-entry placement use this same matching query; their former
empty/full monitors and the unused lookahead graph copy are removed. Recurring
selection still requires distinct endpoint words. A coincident ordinary pair
requires publication before acquisition.

Matching checks every reachable terminal with a pending publication; it does
not prove termination or peer progress. A source identity is retained through
shared continuations, so work can grow with both control sites and alternative
sources. The independent 65,536-state query budget yields unknown without any
usable pairs. It is a sufficient implementation bound, not a semantic limit.
The exhaustive token oracle and joining-alternatives regression check matching,
repeated participation, source identities and explicitly charged query work.


### First publication boundary after an original payload

`Control::publicationAfter` replaces the recurring clients' single-owner mode
comparison and full observation-atom equality. It certifies position and
participation over original control. It does not certify storage identity,
generation coverage, acquired completion or legality of moving an existing
selected endpoint. Those remain separate consumer obligations.

| Premise | Protected obligation | Remaining representation limit |
| --- | --- | --- |
| Every reachable occurrence of the source word has a payload | No phantom publication source | Source words with payload-free occurrences return unknown. |
| First legal boundary on each path, crossing no payload | Preserve the earliest available publication position | Multiple output words return unknown; a later endpoint-set client may represent them. |
| Boundary word differs from source word | Do not confuse before-issue with after-issue or next-visit phases | Explicit phase/gap identities are needed before admitting a shared word. |
| Found pairs equal shared matching pairs | Reject bypasses, extra receipts and repeated sources | Original participation, not predicate spelling, supplies the certificate. |
| Bounded scan and immutable canonical-source cache | Account for exploration and avoid repeated work across cells | Exhaustion stays unknown and grants no selected credit. |

Nearest-boundary work is reported separately, including discarded attempts.
The query's cells-independent result may serve different physical uses; it does
not assert that every payload copy uses one physical cell. Native equivalent
bank-selector, scalar-expression, view and unrelated-state cases exercise the
query through construction and require both bank readiness/release directions
to remain selected without observation or recurring retry.
