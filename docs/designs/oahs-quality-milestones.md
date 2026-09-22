# Synchronization quality milestones

Started 2026-09-22 at `b62b89de5`, following the comparison with research draft
0.35. The objective is to preserve pipeline parallelism under fixed payloads,
physical storage, original control and target event resources. Existing
InsertSync is a comparison, not an ordering specification.

Current work after M3 follows the dependency-ordered
[semantic correction sequence](oahs-semantic-corrections.md), based on the
2026-09-22 prefill audit. It refines M4/M5 prerequisites without reopening the
accepted M1–M3 scope. The bounded ACC contract correction is validated; independent
physical-use relations, generation/support intervals, endpoint composition and
binding generality remain active work. Device improvement remains unmeasured.

## Milestones and acceptance

1. **Accepted mechanism checkpoint: restricted publication-prefix certificate.** Establish a restricted
   certificate at actual command-word positions, including outward publications
   and neighboring key uses. Preserve the early release through later edits, or
   invalidate the certificate. Start from the current KDA release/acknowledgment
   witness; use hc_pre/RMSNorm for independent storage-lifetime checks and GEMM
   as a performance-qualified nonregression. A supplied protocol alone is not a
   constructor improvement. Require a linked production regression, independent
   complete payload-order comparisons, unchanged causal/reconstruction checks,
   and the larger prepared-kernel corpus. Record remaining native misses rather
   than promoting an experiment whose ordering gate fails.
2. **Complete: common lifecycle and occurrence view.** Consolidate physical identity, producing/reading
   occurrences, last physical users, next overwrite, legal gaps, participation
   and possible return support. Add each portion with a real construction
   consumer. Records prescribe obligations, not private event pairs, and do not
   become a second completion authority.
3. **Complete in bounded opt-in scope: rearming at its actual deadline.** Keep storage acquisition deadlines
   separate from the next applicable publication of a reused event key. Select
   actual required returns before unnecessary private acknowledgments. Retain
   unresolved rearming obligations without granting anticipated receipt credit.
4. **Complete lifetime protocols before residual repair and physical binding.**
   Use lifecycle and support records to select coupled readiness/release
   obligations, sharing an already required return where qualified. Establish
   actual completion and consumption paths before repairing residual hazards.
   Certify interaction with remaining repairs; investigate scoped producer
   support instead of weakening the existing safeguard. Avoid unnecessary
   channels before binding, replacing omission trials incrementally with direct
   certificates. Retain explicit recurrence support and cold final validation.
5. **Open lifetimes and scoped ownership.** Generalize qualified successive
   physical uses across skipped children and re-entry. Reuse physical keys only
   with certified ownership lifetimes, matching and actual consumption paths.
   Include zero-use children, independent readers and generation-boundary
   negatives. Lexical nonoverlap alone is insufficient.

## Research refinement: carrying GEMM mechanisms into prefill

Added after milestone 2 was committed as `55e706384`. Draft and implementation
jointly develop the algorithm; neither is the authority against which the other
must simply conform. The other agent's review through `edcdcecee` is useful
historical input. Its unfinished cross-word/common-interface status is superseded
by milestones 1 and 2; this does not imply universal composition or a complete
implementation of the draft's contextual ordering condition.

**Open question:** the user reports large isolated-GEMM gains that did not carry
into prefill and identifies Qwen and DeepSeek prefill kernels in pypto-lib as the
target population. Select one representative matrix-heavy prefill kernel from
each family; a user-supplied failing campaign is not required to start. Pin
source/binary versions, shapes and execution context before attributing the gap.
The expectation to investigate is that applicable GEMM mechanisms should carry
into these kernels, not that every prefill kernel must show a speedup. Missing
generality is a hypothesis, not an established cause. The archived
[MAT campaign](oahs-mat-device-final-results.md) reports projection gains of
4.4–16.2% against existing; distinguish those results from the reported prefill
comparison. That campaign's restored-fence controls retain the gains, so fence
removal alone is not an established explanation.

Start with one matched GEMM/prefill physical-lifetime witness. Follow:

1. Physical bank identity and successive participating uses: did analysis retain
   the relevant relation through nesting, guards and independent recurrences?
2. Candidate readiness/release boundaries: was the coupled protocol available,
   and if declined, was the reason participation, producer support or capacity?
3. Actual transfers and residual repair: did a required return carry reader
   completion, prior-writer completion and readiness consumption? Did another
   repair introduce new prerequisites or broaden a publication prefix?
4. Emitted ordering and execution: did the complete plan preserve the intended
   overlap, and did that overlap affect the measured execution path? Check event
   overhead, resource limits and coupled scheduling before blaming qualification.

The initial deliverable is one classified obstruction with a minimized,
structural regression, or evidence that the intended protocol already exists
and runtime attribution is needed. It is not a new broad profiling campaign or
a kernel-name recognizer. Reuse existing plans and records first. Device work
requires a concrete candidate or unresolved timing question and authentic
execution context. This diagnostic supplies a witness for the relevant
milestone without reopening completed M2 or making M3 depend on a prefill speedup.

Keep the implementation sequence, with these explicit research obligations:

| Mechanism | Implementation/proof question | Placement in the plan |
| --- | --- | --- |
| Deadline-driven rearming and shared returns | When can an actual required return discharge several consumption obligations before every applicable key reuse, without moving its endpoints to manufacture coverage? | M3: explicit obligations before helpers; preserve storage deadlines and certify fallback. |
| Complete readiness/release selection | How does analysis justify selecting a coupled protocol whose actual returns also cover prior-writer requirements before ordinary repair? | M4: generalize the qualified GEMM mechanism; a proof for a supplied cycle alone is insufficient. |
| Interaction with residual repair | Which scoped support conditions prevent a remaining repair from moving behind independent producer work and adding order? | M4: retain the current complete-producer safeguard until a narrower rule is proved and tested. |
| Independent physical-use relations | Can unrelated carried state or independent periods reject an otherwise useful bank relation? | Targeted analysis prerequisite when witnessed; reuse the existing address-slice/independent-slot work item, not a larger joint period cap. |
| Participation, joined consumption and dormant ownership | When do alternative receipts establish consumption, and which ownership promises survive helper removal and later restoration? | M3 must preserve these existing semantics; M5 generalizes ownership across skipped children and re-entry. |
| Contextual placement, including relays | Does a narrower destination prefix impose new prerequisites on intermediate work or outward publications? | Retain two-sided checks and M1/M2 proof boundaries; extend only for a concrete unsupported placement. |

For each rule, record: existing draft idea and premises; implementation discovery;
proposed algorithm/state amendment; exact proof obligation (with theorem reference
only after checking it); positive/negative construction witnesses; full-order and
native/device evidence. Do not infer a new theorem from finite tests or infer
practical value from a theorem alone. GEMM motivates and validates mechanisms;
there is no decision here to promote it into a paper case study or rewrite the
paper's evaluation. Amend the common algorithm first where evidence supports it.

M3 remains bounded: ordinary acyclic corridors, then qualified branches/shared
words; explicit forward-generation/consumption obligations, actual-return
selection and rechecking at exact republication gaps. Unsupported recurrence
retains the existing closed exchange. Test stale return generations, skipped
returns, earlier reuse, multiple readers, scarce keys and later endpoint edits.
Acceptance requires linked ordering improvements in constructor witnesses and
native/corpus nonregressions, not a particular kernel speedup. M4 and M5 remain
separate work rather than prerequisites for finishing M3.

## M3 initial implementation and construction-cost gate

M3 is complete at `e5fe6147a` for ordinary acyclic and qualified shared/branch
receipts. Unresolved obligations retain actual consumption identities; required
returns or balanced fallback helpers establish rearming at applicable deadlines.
Conditional reuse, stale returns, earlier deadlines, shared returns and later
edits have acceptance witnesses. The new native conditional witness removes
ordering without additions on all four traces. All 27 portable suites, both
native suites, 19 compatibility inputs, both 88-case corpus runs and six targeted
plans pass. Existing corpus plans and default replay work counts are unchanged.
The policy remains opt-in; general recurrence and ownership remain M4/M5 work.
See `HANDOFF.md` for the pinned evidence and the
[selected-plan account](oahs-selected-plan.md#milestone-3-explicit-deferred-rearming-initial-scope).

The user also prioritized compiler cost after observing slow handoff compilation.
A profile identified unnecessary original-requirement classification when the
actual cross-engine residual was empty; provider grouping now skips it. A
separate attention profile identified state-join allocation/copy cost; worklists
now consume the join's exact change signal. Keep plan-quality and independent
acceptance gates unchanged. Compare compiler cost against pinned equivalent
plans, distinguishing native import, preparation, construction replay and final
checking. Prefer deterministic work-count regressions to flaky timing limits.
New M3 obligations must reuse existing snapshots/indexes rather than introducing
one full-program solve per obligation. Further cyclic replay reduction remains
separate, measured work; the current fixes do not eliminate that cost.

## Quality and validation contract

Distinguish required-order closure, actual emitted-command order, and the order
of a checked alternative under identical contracts. An extra relation is an
opportunity, not proof that the primitive vocabulary can avoid it. Compare full
relation sets; fewer relations or commands alone do not establish dominance.
Finite traces are regression evidence, not induction over arbitrary loops.

During construction, certify only a bounded placement/edit with complete causal
interfaces; do not add all-pairs graph search or a general final refinement pass.
An incomplete plan is not the comparison baseline for a no-added-order claim.
All original physical requirements and independent acceptance checks stay intact.

Run focused portable and native regressions first, then the prepared corpus and
the archived compatibility inputs. Pin source, input and driver identities and
record changed plans separately from failures. Use disk-backed artifacts outside
the source tree and obey the aggregate two-worker limit. Device tasks follow a
specific qualified candidate or unresolved timing-attribution question; preserve
authentic coupled execution and compare existing, pinned OAHS and candidate.

## Shared foundations before further kernel optimization

The user clarified the implementation order on 2026-09-22: develop the shared
building blocks across the milestones before pursuing another KDA-specific
optimization. Kernel witnesses motivate and test the mechanisms; qualifiers
must continue to describe physical storage, control, occurrence and target
semantics. The subsequent acceptance clarification separates mechanism readiness
from kernel performance improvement. Milestone 1's restricted certificate scope
is accepted on its recorded construction, ordering and corpus evidence; shared
occurrence correspondence starts milestone 2. Broader certificates and default
activation remain separately qualified extensions.

Build these interfaces in dependency order, with executable consumers and
focused positive and negative tests for each slice:

| Building block | Existing owner and next extension | Consumers |
| --- | --- | --- |
| Physical lifecycle facts | Enrich `RequirementFrontiers` using `StorageFrontierAnalysis`: physical cell and access, source occurrence, candidate release gap, next conflicting-use deadline and participation. Keep distinct readers and deadlines. | Ordinary placement, recurring qualification and binding |
| Occurrence correspondence | Share `Control`'s original occurrences and canonical words. Expose qualified correspondence for source, consumer and successive physical uses; an unknown answer preserves separate frontiers. Refining control must preserve existing placement opportunities. | Cross-word certificates, recurring obligations and open lifetimes |
| Ordering certificate | A policy-independent query at exact command gaps compares payload observations and outward causal interfaces. Track which selected words its proof depends on and invalidate after unsupported edits. | Publication placement first; later endpoint grouping and movement |
| Rearming obligations | Consolidate existing pending-return records in selected construction state. Keep the storage deadline separate from each applicable selected key's next-publication deadline; record possible returns separately from acquired support. | Ordinary and recurring physical binding |
| Logical obligations and ownership | Qualify necessary transfers before assigning keys. Reserve only explicitly required recurrence support, then bind with actual consumption evidence across successive participating uses. | Recurring selection and open bank lifetimes |

The first three are the immediate foundation slice. Rearming and ownership
follow once they have real consumers of the shared lifecycle/occurrence view;
do not add unused placeholder records or implement all five policies at once.
Immutable analysis does not contain selected-event occupancy or declare a
future return acquired. The existing causal engine remains the authority for
actual completion and consumption knowledge.

Validate generality by varying structure: straight and branched paths, skipped
and repeated uses, shared analytical words, multiple reader engines, aliased
physical bytes and scarce/reused keys. Link each enabled decision to the
first-pass fact, its candidate boundaries, the actual selected endpoints and
key, and the residual after the real receipt. Unsupported cases must remain
explicit rather than acquiring inferred completion from a record. This follows
the draft's lifecycle and analysis-to-decision contracts in
`sections/07_literature_informed_mechanisms.tex`.

Keep analysis refinement and placement options independent. In particular,
enabling an ordering certificate must not implicitly enable final-read control
refinement. Compare each refinement separately: adding analysis structure can
change greedy construction even when the new placement mechanism never fires.
Default activation requires complete-plan comparisons and corpus acceptance;
availability of a building block alone is insufficient.

Use KDA, hc_pre, physical RMSNorm lifetimes, GEMM and the larger corpus as a
portfolio throughout this work. Mechanism acceptance requires explicit proof
scope, linked constructor tests, independent causal/emission checks, complete
order comparisons for changed native plans and corpus nonregressions. A KDA
improvement, fewer ordering relations or a measured speedup is not required
to finish milestone 1. Track performance improvements as a subsequent campaign
using the shared mechanisms; device timing follows a qualified native candidate.

## Milestone 2: shared occurrence foundation

The first consumer of shared occurrence correspondence is original-choice
placement. `Control::correspondence` returns the qualified pairs of analytical
source and receipt occurrences for canonical words. The result describes
original-control participation; it does not prescribe a key, declare a future
return acquired or replace storage history. The prior `balancedWords` query now
uses this relation too.

Construction checks source coverage, access freshness and physical-key evidence
at every paired occurrence. Choice qualification covers every copied arm and
resolves its original scope owner through canonical word identity. A private
trial appends through the canonical ledger rather than changing one analytical
copy. This preserves existing placement when observation analysis refines a
loop's final visit.

The real-refiner portable witness compares complete payload order on 22 traces
(1/2/4 visits and every branch sequence). All 26 portable suites pass. The native
analysis-only KDA witness adds zero and removes 276 finite payload relations
against the unrefined committed default; previously the refinement also added
16. Final-read placement remains disabled in this comparison. This is mechanism
and ordering evidence, not a timing result. See `HANDOFF.md` for the pinned native,
compatibility and corpus results.

Milestone 2 is complete. Exact-gap publication certification now proves every
qualified source/publication occurrence separately, compares its causal
interfaces, and retains the union of crossed canonical-word dependencies. A
later edit in either analytical copy invalidates the proof. Shared-word
constructor regressions compare complete order across both copies and branch
arms; negative cases cover later physical readers, unmatched copies, same-key
crossings and edits in a second copy.

`RequirementFrontiers` now provides the common physical-use and lazy lifecycle
view in `SelectedLifecycles.cpp`. Ordinary placement, recurring qualification and
binding consume it. Binding decisions retain original requirements, release
candidates, distinct storage deadlines and possible return deadlines. A
multi-reader regression preserves independent reader origins and checks these
links on an accepted construction. Possible returns remain analysis facts;
actual causal propagation is the only source of acquired credit.

The next milestone is rearming at the actual next applicable key-publication
deadline. It must use these facts to select required returns before introducing
private acknowledgments, without assuming their future consumption. Broader
recurring selection and open ownership remain milestones 4 and 5. See
`HANDOFF.md` for the final native and corpus acceptance evidence.

## Milestone 1 evidence and next action

The archived KDA path is M completion -> acknowledgment -> MTE1 release -> MTE2
load. The opt-in first-write/final-read plan previously removed 1,784 relations
and added 706; it is not accepted. Refresh this evidence at the starting commit.
RMSNorm's input storage also serves as reduction scratch, so conversion is not
necessarily its last physical read. hc_pre has an entry-prefix candidate on
distinct accumulator and input ranges. See `oahs-final-read-sources.md`,
`oahs-frontier-motion-context.md` and `oahs-sweep-followup.md`.

### First implementation: within one command word

`SelectedPublication.cpp` implements a default-on construction-time rule for
new ordinary completion publications. It uses the KDA witness's release-before-
unrelated-wait mechanism without recognizing a kernel or opcode sequence. It
does not yet improve the native KDA plan. This is mechanism evidence, not a
performance result.

At ordinary binding, the constructor selects one earlier gap in the same word.
It may cross acquisitions on the publishing engine and commands on other
engines, but stops at another publication or fence on the publishing engine,
any ALL barrier, a matching physical-key endpoint, or a protected prefix.
Keeping each outward source publication on its original side of the edit is
essential: a required payload check alone cannot certify its exported order.

The structural ordering argument is local. SET observes the current completion
prefix but does not gate subsequent launches. Moving it before a source WAIT
removes that WAIT's completion prerequisite from SET. No source SET/fence is
crossed, other engine orders are unchanged, and matching event endpoints do not
cross. Every retained boundary path in the new fragment therefore already
exists in the old fragment. This is a restricted instance of the draft's
boundary-path certificate, not a general graph-inclusion algorithm. Its
statement assumes an unchanged surrounding graph; later greedy choices still
require independent complete-plan comparisons.

Coverage and protocol legality are separate. For every reachable original
occurrence of the shared word, replay to the exact proposed gap must establish
the motivating storage completion, an empty physical event, and actual
consumption knowledge at its publisher. A private ledger copy then undergoes
full fixed-plan analysis: protocol/resource checking must succeed, and no new
missing storage or retirement obligation may appear. A refused certificate
leaves the live ledger unchanged. Accepted motion retains endpoint IDs and
increments the ledger version; selected replay still checks finalized queries.
There is one proposed gap, no alternative-plan search or final motion sweep.

The ledger records the source-engine endpoint IDs preceding the protected
publication. Later insertion and return restoration stay behind that boundary;
each selected replay, including a retry after restoring returns, checks that
no new source endpoint entered the recorded prefix. Deletion may shrink it.
This protects the prefix **inside that word only**. It does not yet certify
edits to incoming state from earlier words, recurring-role coalescing, movement
across payloads, or a general all-publications construction invariant.

The initial implementation deliberately spends two fixed-plan analyses per
considered motion. `prefix_checks` and `prefix_analysis_sites` expose that cost;
`prefix_publications` counts accepted motions. Replace those solves with a
shared local boundary summary only after the quality rule is established.
`--no-publication-prefixes` in the diagnostic driver isolates the new rule.

`selected_publication_test.cpp` exercises actual construction, including a reused
physical key, and compares complete issue/completion relation sets using the
independent graph oracle. It removes the unwanted compute-completion -> refill-
issue relation without adding any relation. Negatives retain required compute
completion, refuse an intervening outward publication, source fence, ALL fence
and same-key crossing, and check insertion/restoration after certification.
The earlier source-gap policy's comparison explicitly disables this independent
new rule, so it continues to measure that policy alone.

### Restricted cross-word foundation

The next local slice adds an opt-in `crossWordPrefixes` policy, exposed by the
diagnostic driver as `--cross-word-prefixes`. The ordinary within-word rule
remains default-on. The new option does not change native observation import;
final-read refinement remains separately selected by `--final-read-sources`.
`--no-publication-prefixes` disables both motion rules.

`RequirementFrontier::lifecycleRelease` retains the candidate gap after the
relationship's physical source use, alongside its existing physical cell,
access class, source occurrence, consumer deadline and participation metadata.
Unlike the older acyclic-only `publication` field, this gap may be inside a
loop. It is not a last-read, balanced-participation, completion or rearming
certificate. The constructor must establish those applicable facts separately.
This extends the existing analysis interface instead of creating another
storage-history authority or prescribing an event pair.

`certifyPublicationOrder` in `SelectedPublicationOrder.cpp` is independent of
the selection policy. At an exact gap, it compares the old and proposed
fragments on every admitted original path. Independent symbolic inputs name
the incoming A/T/S/D ports; distinct symbols name each crossed payload's issue
and completion. Commands propagate these dependencies. Each candidate payload
observation and every exit port must have a prerequisite set contained in the
original's. This includes outward event publications and consumption ports,
so crossing an existing source SET is possible when its exported completion
does not gain prerequisites. This is an observable prerequisite certificate
for the ordinary A/T/S/D command model. It is weaker than the draft's
all-labelled-terminal path certificate: it does not preserve every relation
between retained event representatives. Do not reuse it for arbitrary endpoint
grouping or interfaces with additional external attachments without extending
the proof and its representation.
The constructor's existing causal-frontier admission rejects unqualified typed
resource, visibility, private-event and final-block effects before this query;
the query consumes that admitted program and its unique physical-key population.

Admission requires qualified correspondence for every analytical occurrence of
the endpoint words, balanced participation, no crossed endpoint of the same physical event key,
and no crossed backedge. Paths remain separate. The query declines beyond
8,192 visited path steps or 64 completed paths. Unknown or unsupported cases
retain their original frontiers. Actual source-time coverage/rearming and the
private-plan protocol/residual check remain mandatory and separate.

The ledger preserves endpoint identity when moving a publication. It records
the exact endpoint sequences in the crossed interval after the move. At the
original publication word, a retained following endpoint identifies the end of
that interval; commands after this anchor remain in the continuation. If no
following anchor exists, the whole word is conservatively retained. Any later
insertion, deletion or reordering inside the interval, or loss of its anchor,
invalidates this certificate. Final helper omission also preserves certificates,
and finalization refuses a stale proof before invoking the unchanged checker.
This conservative dependency tracking is intentionally stricter than necessary.
Shared endpoint words are qualified by milestone 2. Re-certifying useful later
edits and choosing a key with the early boundary in view remain later work. There is no
general final-plan motion pass, automatic control refinement or native
improvement claim from this slice.

Decision records retain the binding-time placement; their stable endpoint IDs
resolve to the actual positions in the accepted ledger. This preserves both
the initial selection and the result of certification for diagnostics.

The linked constructor test crosses an original branch and an outward source
publication. Both branch traces strictly remove compute-completion to refill
ordering while preserving the outward publication's completion, with complete
independent payload-order inclusion. A later physical reader retains the late
release. Other tests cover same-key crossings, invalid gaps, a loop corridor,
private-copy isolation, stale-proof invalidation and unchanged continuations,
including a continuation inside the same command word. The constructor witness
runs with final helper omission both enabled and disabled.

The first opt-in corpus run exposed this exact-gap distinction in two MLP cast
modules: whole-word invalidation incorrectly included later continuation edits.
With the interval corrected, both native modules construct and reconstruct and
move an MTE2-to-V publication before an original conditional. Twenty independent
finite order comparisons cover first/nonfirst entry and 0, 1, 2, 3 and 5 loop
visits. They add and remove zero payload relations: an independently required
receipt still gates the consumer. This demonstrates native use of the building
block with ordering nonregression. Under the revised mechanism acceptance
criteria it contributes to milestone 1; it is not a native overlap improvement.

### Starting evidence before shared occurrence correspondence

At frozen `b62b89de5`, default KDA, dspark RMSNorm and hc_pre construct and
reconstruct. Both baseline and candidate now refuse the opt-in KDA
first-write/final-read experiment at cut 754: `split recurring receipt has no
certified return key`. The older 1,784-removed/706-added measurement above is
historical, not the current baseline. Do not relax key-rearming checks to
recover that experiment.

The separately checked final-read-only refinement succeeds on KDA at the
committed first slice, but removes 276 and adds 16 finite payload relations.
The same difference occurs when only the final-read observation refinement is
imported and publication motion is disabled. The added relations serialize
later-bank extracts before the first bank's compute. This exposes loss of an
existing placement opportunity under shared analytical words; it is not an
accepted tradeoff or evidence for enabling the refinement by default.

This motivated the shared occurrence correspondence slice now recorded above:
choice frontiers survive analysis refinement and the 16 added relations are
removed. Milestone 2 connects correspondence to exact-gap certificates for shared words
and lifecycle boundaries to placement, recurring qualification and binding.
Continue with actual rearming deadlines rather than adding a KDA-specific rule. Keep source-time consumption evidence mandatory;
the cut-754 combined experiment is a separate prerequisite if that policy is
used. Preserve the complete-order and corpus gates for each mechanism extension.
Track KDA overlap improvements separately from milestone 1 acceptance, with
hc_pre, physical RMSNorm lifetimes and GEMM remaining independent witnesses.

Artifacts, driver/input hashes, and final validation totals are recorded in
`HANDOFF.md` under the active quality work. Device timing remains pending an
improved native candidate; identical native plans establish no speedup claim.
