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


### Physical-use succession and open boundaries

`StorageFrontierAnalysis::nearestUses` projects original control to the nearest
represented access to one cell. Starts are inclusive; explicit stop boundaries
take precedence over accesses. Forward and backward queries preserve original
occurrence identities. Reads, writes, may-writes and RMW all stop the traversal;
none is silently converted into a definite overwrite or causal receipt.

The result separately records physical accesses and open entry/exit/region
boundaries. `complete` describes finite marginal reachability, not termination,
event matching or a fully qualified generation family. Invalid inputs return
incomplete; recurring qualification explicitly refuses that result. A boundary
is an obligation to the surrounding interface, not an implicit drain.

The recurring qualifier consumes this shared analysis in place of its private
nearest-access traversal. Exact normalized queries are cached in the existing
immutable storage-analysis owner; work is reported separately. Tests retain
repeated-child and following-child uses, reloads, outside readers, unrelated
storage, explicit stops, and old producer provenance across may-write/RMW.

This establishes succession facts needed for generation/support construction.
It does not yet remove the enclosing reader-owner gate or certify an affected
producer interval. Such an interval must retain other outstanding storage and
exported obligations, including the X/Y fence-relocation counterexample; the
next same-cell writer is not by itself a sound closing boundary.

### Reader participation across physical write episodes

`RequirementFrontiers::readerParticipation` classifies a represented physical
reader from its nearest accesses in both directions on original control. It
starts on predecessor/successor edges, retains reloads and outside readers,
and records open entry/exit boundaries. Writer-only predecessors identify first
participation; reader-only predecessors identify continuation. Reader-only
successors continue the episode; writer/exit-only successors end participation.
Mixed participation and RMW boundaries remain unknown. A may-write stops physical
succession but does not kill older writer provenance or prove initialization.

The enclosing-cycle consumer replaces its child-owner/first-tail atom recognizer
with this query. Every reachable copy sharing an emitted word must agree on its
roles. Exact endpoint matching and final causal validation remain separate. The
current consumer still qualifies one writer pipe and one reader pipe per cell;
other physical facts are retained, not rewritten to satisfy that restriction.

Before an enclosing cycle is committed, the exact selected packet must pass
protocol analysis. Producer overwrite seeds also define a conservative repair
interface: a remaining producer-targeted obligation is unsupported when its
source may precede a seed and its consumer may follow a seed. Include other
cells, backedges and continuations. Unioned reachability loses path correlation
and may refuse unrelated paths; it never proves separation from that loss.
Support seeds survive relationship replacement and private-channel omission.
The work is two graph walks plus one operation-membership scan per participating
producer and a linear residual scan, charged as `producerSupportWork`. Canonical
seeds and reader words are processed once, including analytical copies.

This is a producer-fence-relocation certificate, not the full Stage 2 affected
interface or a publication-prefix theorem. Exported publication, consumer-side,
continuation and retirement ordering remain separate obligations for later
composition; independent validation still enforces safety and event legality.
The new mechanism does not claim no added payload order. Full ordering sets,
resources and compilation work must be reported independently on changed plans.

### Joint original endpoint demands

`RequirementFrontiers::endpoints(owner)` collects first-consumer, first-write and
final-reader requirements together from cross-engine physical relationships,
before native control refinement. Each record retains its original access,
relationship and enclosing owner. Discovery does not require a useful protocol
inside the child and unrelated carried scalars do not invalidate the records.

Materialization currently requests separation when nearest original uses mix
read-only and write-bearing boundaries, or an initial boundary with a preceding
write. An RMW is one write-bearing boundary for this query; its read obligation
remains in the physical facts and downstream generation checks. Independent
bank-residue demands remain separate. A false query means no supported
separation was requested, not proof that all occurrence distinctions are
irrelevant: different producing identities and unavailable facts remain open.

Classification is cached per access/cell/direction, independently of relationship
multiplicity. Native discovery work is reported separately from construction.
Original-loop domain proofs supply normalized observation arithmetic as described
below; domain refusal retains original control. Exclusive
resource-only requirements and terminal readers without a reuse relationship
are not newly covered by this index.

### Refinement and construction traversal

An original backedge label is not proof that an edge in a refined graph is a
natural loop. Construction first qualifies its natural-loop summaries. It then
classifies all remaining labelled edges together in a candidate scheduling
graph, including the qualified exit summaries. Edges crossing this graph's SCCs
are finite transitions and remain in the construction traversal. Remaining
cyclic interfaces require existing contextual replay from the start: causal
propagation uses every original edge and unfinalized payloads supply pending
effects, never desired completion. The reduced graph only schedules decisions.
Original SCCs and word spans still govern cache reuse.

The classification runs once and reports finite transitions, unsummarized cyclic
edges and graph-size work. The shared SCC helper sorts component members, so its
current bound includes up to O(V log V) sorting. Synthetic exit-summary edges may
conservatively classify additional cycles; they never justify deleting original
semantic edges or granting completion. The independent final checker is unchanged.

Native command anchors follow observation liveness, not individual site
reachability. An unreachable canonical representative retains its anchor when a
reachable copy shares its word. Observations with no reachable occurrence have
no emitted anchor. This prevents composed first-use refinement from exposing
phantom command positions while retaining actual shared-word emission.

### Original loop domains and endpoint participation

The invariant is that an observation names logical participating visits of the
original loop, not its raw induction value. Shared `SyncSlotMapping::LoopDomain`
proves nonnegative constant lower, positive constant step, signed index control,
and that every possible final executed increment is representable. Dialect
constant folding and memoized `InferIntRangeInterface` queries supply scalar
facts. The maximum upper bound determines the last possible active IV; the
proof uses that IV's actual progression, not an unnecessarily strong upper+step
bound. Unknown arguments keep full ranges. A failed proof retains original
control; it does not erase independently proved carried-slot facts.

For an active visit, k=(IV-lower)/step is exact and nonnegative. Residue and
previous-use guards use k. A p-th next visit exists iff upper-IV > p*step.
If p*step exceeds signed index range, that predicate is false (its negation is
true); no overflowing arithmetic is emitted. Normalized ordinals and remaining
values are reused per owner/insertion block. Reconstruction independently checks
the original subtraction/division operands, comparison and threshold, including
constant overflow results. No emitter-created tag supplies that proof.

Raw scalar branch conditions use the shared finite dependency relation. They
are not reinterpreted as predicates on k. The current decision representation
admits one matching residue whose period divides the selected owner period;
other relations keep the original branch. Distinct owners are not multiplied.
The original first-use conjunction importer remains separately restricted to
its proved literal-zero/unit-step predicates; this increment does not broaden
that transformation's theorem. Negative/dynamic lower bounds, dynamic steps,
and index contracts other than the existing signed 64-bit model remain unknown.

### Consumption joins and exact ordered emission

Refining participation can expose several concrete WAIT identities for one
consumed key. Event emptiness and the observer's consumption knowledge are the
semantic facts; one distinguished WAIT ID is not required. If the existing
single-receipt repair has no candidate, construction may append a reverse
publication/acquisition followed by the required forward publication/acquisition
at the already selected publication/deadline gaps. It retains the current
straight-interval qualification and closed-key ownership restrictions.

Private causal execution at every participating source occurrence filters
bindings. It supplies no public credit. The complete ordered packet then passes
independent original-graph protocol, diagnostic and phase-resource checking.
Construction commits those same four endpoints and propagates them together.
Storage residuals outside the packet remain ordinary obligations. At a common
cut, the reverse transfer returns prior consumption before the forward use;
the forward receipt may establish reverse-key consumption for the next visit.
Checking or propagating the reverse half alone would miss this mechanism.
No cyclic-program exclusion or singleton-WAIT assumption supplies the proof.

These newly inserted reverse endpoints remain explicit retained support. They
are not yet represented by the existing single-WAIT dormant-helper mechanism,
and this increment does not claim complete helper-ownership unification.
A return at the selected gap can add ordering; exact protocol legality is not a
publication-prefix improvement certificate. Packet check count/site work is
reported separately from replay, command resources and payload ordering.

The shared `SyncCodegen` now distinguishes mergeable synchronization requests
from ordered command lists. OAHS selects `PreserveOrder`, retaining repeated
SET/WAIT signatures and adjacent barriers. Target-mandated lowering and explicit
compensation/deferred-tail behavior remain part of its input contract; the A3
adapter supplies active single-key commands without those auxiliary policies.
Independent native reconstruction still compares every actual ordered command
against the selected word. It is not weakened to accommodate deduplication.
