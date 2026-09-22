# Selected-plan construction: draft F1–F8 implementation

Specification: the synchronization draft v0.22, policies F1–F8.
See [shared semantic extraction](oahs-shared-semantics.md) and
[storage and fixed-plan analysis](oahs-analysis.md) for the input contracts.
`algorithm=handoff` uses this constructor. `algorithm=existing` remains the default
and comparison path; no additional pass mode or legacy fallback is introduced.

The ongoing quality work and the first restricted construction-time
publication-prefix certificate are specified in
[quality milestones](oahs-quality-milestones.md#first-implementation-within-one-command-word).
That certificate protects a new ordinary publication inside its command word;
the [opt-in cross-word foundation](oahs-quality-milestones.md#restricted-cross-word-foundation)
adds a reusable ordering query and explicit proof dependencies. Neither is yet
a global publication-prefix invariant. Shared lifecycle and occurrence
interfaces take priority over further single-kernel selection policies.

## Entry points and state

`constructSelectedPlan(program, fixedWords)` is a production C++ service. It
returns selected commands only after cold validation of the original program.
A failure retains diagnostic records but clears executable commands and the
certificate. `fixedWords` preserve their order and observation identity; reserved
keys remain unavailable. Unsupported typed effects are refused, not erased.

The three objects remain separate:

- `StorageFrontierAnalysis` owns immutable original storage/provenance queries.
- `CausalFrontier` owns the shared must-causal A/T/S/D state and event generations.
- `selected::Ledger` plus source, decision, checkpoint and update records own the
  selected construction. Stable endpoint IDs survive earlier word insertion.

A saved source does not publish anything. Its version and actual original cut
must agree with the current checkpoint map before it supplies a placement query.
The source's prefix is compared with the consumer's current occurrence record;
a later access cannot inherit an older receipt merely by sharing its class.

## Policy map

| Rule | Implementation |
| --- | --- |
| F1 | `SelectedControl.cpp`: original graph, SCC order and original-cut frames; `SelectedPlan.cpp::run` visits each task once. |
| F2 | `CausalFrontier::inspect`, `SelectedGroups.cpp::reasons/consume`: every effect including RMW, no hypothetical acquisition. |
| F3 | `SelectedGroups.cpp::groups/sourceGroup` and `CyclicFrontiers.cpp`: known readiness and reuse together, then remaining overlaps; compatible recurring cells share one source/target prefix. |
| F4 | `coverage/groups` and recurring-frontier merging: strict containment among required providers, then stable source ordering; `consume` rechecks the whole residual after real acquisitions. |
| F5 | `Control::after`, saved `SelectedSource` records and ledger placement preserve separate early source positions, including qualified straight corridors inside observed loops. |
| F6 | `consume`: cross-engine repairs first, named fence only for the remaining same-engine residual, then complete payload check. |
| F7 | `SelectedAllocation.cpp`: stable source-time key selection, complete forward/reverse interval certificates, nonrecursive consumption acknowledgment and one shortest eligible route. |
| F8 | `SelectedControl/SelectedReplay`: open choice/loop interfaces and actual fixed-point validation; `CyclicFrontiers.cpp`: qualified recurring relationship frontiers, participation checks and deterministic role allocation. |

All ordinary F7 binding routes share the ownership exclusion for closed roles,
recurring reservations and inactive restorable helpers. Certified borrowing is
explicit. Split reverse helpers additionally check their successor against the
complete proposed forward/return packet. A ledger-maintained publication index
answers qualified acyclic no-next-use queries without a continuation walk;
shared or cyclic cases retain the full structural check. See
[key binding consistency](oahs-key-binding-consistency.md).

Ordinary completion endpoint `request` IDs index `decisions`. Endpoints with
purpose `RecurringCompletion` index `channels`. Both kinds coexist in the same
ledger; `channels` contains qualified physical access-role requests. Neither table supplies completion to the checker.

## Qualified static FIFO relays

A lowering-qualified two-slot FIFO can expose one static slot for every send
and receive without changing control or adding guards. Its shared physical-use
view feeds ordinary storage analysis. For an indirect readiness requirement,
construction can separate the source publication, intermediate forwarding gap
and final receive deadline. Candidate keys use actual source-time consumption
credit and neighboring-use checks before placement ranking; exact staged words are protocol-checked
before commit, and their full analysis work is counted separately. Unsupported
cases retain ordinary binding. Ranking compares incidental receiver completion
and new middle-engine prerequisites separately, and protects earlier receipts
to the same receiver. Incomparable views keep the deterministic incumbent.
This remains a bounded heuristic; native payload-order comparisons are not a
general least-order theorem. See [implementation and evidence](oahs-static-fifo-relays.md)
and [the relay-selection correction](oahs-relay-selection.md).

## Independent physical-bank episodes

The guarded episode analysis also supports an independent cohort whose banks
have distinct reader frontiers but the same producer/reader engines. Each bank
retains its own early readiness and previous participating reader's release;
sibling-region boundaries do not consume its token. Selection happens before
key binding, and insufficient capacity declines before ledger modification.
This admission applies when the cohort has no existing common-reader group;
it does not append every recognized cell to an already composed protocol.

The original role, boundary, participation and balance proofs are unchanged.
It adds no observation modes, numerical loop expansion or omission trials.
Native non-unit-step loops can use this interface without refining their scalar
iteration count. Actual selected endpoints still establish every receipt.

## Look-ahead and publication frontiers

Known readiness/reuse and additional-overlap requirements retain separate
selection stages. Sharing a source engine and consumer does not automatically
combine their publication prefixes. A required overlap provider from another
source may compete during the Known stage when its current selected source
snapshot already covers a known prerequisite. This lets a necessary readiness
receipt discharge a reader-release requirement through an established
third-engine path. It does not extend the early prefix of a source that already
has a Known group, or grant credit before the receipt is selected and replayed.
Alternative publications intersect additional coverage from their actual saved
source checkpoints and exclude classes regenerated on any intervening path.
Loop-entry providers require source-time coverage, incoming-corridor freshness,
and no class regeneration inside the region. These claims are separate from the
requirements motivating the transfer. An overlap-only promotion with no Known
coverage declines before key selection or loop-entry candidate analysis.
After each actual acquisition, construction
recomputes the residual and provider groups; each edit must strictly reduce the
cross-engine residual before continuing.

`SelectedLookahead.h` indexes immutable original class issues within proven
corridors and strict-future payload reachability. Freshness queries use the same
half-open source-to-consumer interval as the original scan. Future reachability
includes backedges, so the last textual payload of a loop is not terminal.

When a straight-corridor source is unavailable, backward search can find an
acyclic frontier of alternative publication cuts. Each source must contain the
required completion in its actual selected snapshot. A separate empty/full
monitor checks participation from original entry to exit; shared observation
words and unqualified cyclic sources are excluded. One globally unused eligible
directional key is checked at every source position. The publications and common
acquisition are inserted together and replayed as one selected edit. The branch
regression checks absence of unrelated-load completion before consumer issue in
the independent graph oracle, as well as final safety.

A common-cut return acknowledgment can be omitted after the final
cross-engine acquisition of a terminal payload: no future payload, no remaining
cross-engine demand, one reachable occurrence of the word, and no later ledger
command other than retirement. Consumption knowledge at the publisher is not
invented.

For contextual region construction, closed-exchange helpers also retain a
pending rearming record. A newly selected necessary reverse transfer can
supersede a helper if its publication follows the original consumption and its
acquisition precedes every affected source payload, outward publication and
selected republication on the original paths (including backedges). Exit paths
need no rearming after their consumed token. The query considers every reachable
occurrence of a shared word. It retains helpers when these conditions are not
proved; it does not merge source prefixes or move either necessary endpoint.

Helpers remain a conservative construction fallback while the suffix is
unfinished. If a later selected publication introduces an earlier deadline,
that key's original helper is restored at its original prefix and marked
required. Actual occupancy and consumption checks remain authoritative. Each
helper/necessary-return pair is enumerated at most once: per-direction cursors
pair old helpers only with newly appended returns and new helpers with all
returns, retaining the original selection order. Unchanged populations require
no pair scan. The distinct pair population can still be quadratic, and each
continuation query and successful edit/replay has a separate cost.
Only successful discharges or required restorations replay the changed ledger.
This online discharge mechanism does not use changed-plan cold-check trials.
Separately, `finish()` performs bounded engine-pair helper-removal trials and
then a final certificate; the complete constructor therefore is trial-enhanced.
Count `helperCompositionTrials`, its site evaluations/time and final certification
separately from online pair visits, graph queries, restorations and net discharged
helpers. The ordinary
loop-hypothesis traversal retains its closed fallback: it does not yet export
these contextual token-generation interfaces. Physical reservations also remain
conservative, so this does not solve allocation under a one-key pool.

These acyclic queries do not themselves qualify loop-entry acquisitions or
previous-use bank correspondence across guarded or non-unit-step loops. Those need original
occurrence and participation certificates, including zero-trip and enclosing
continuations. Backward demand queries guide endpoint placement; the actual
completion and event-generation state continues forward through selected words.

## Alternating FIFO slot interface (native experiment)

The pinned unsplit vector tile FIFO exposes a two-slot GM envelope and separate
participating send/receive increments. Native admission proves complete
alternating episodes for one invocation-owned, otherwise unused backing root.
A hidden two-state quotient preserves original command observations; skipped
episodes do not advance it. Unsupported cases keep conservative pooled effects.
No cross-core receipt or TFREE operation supplies local completion credit.

The constructor selects a shared writer-to-reader publication before each
receive, its acquisition after the receive, and a reverse acknowledgment there.
For receive i, the previously acquired return covers writer i-2 of that same
slot. The current acknowledgment covers receive i before writer i. These are
consequences of the selected commands, checked from invocation entry, rather
than assumed future receipts. Local scratch obligations remain ordinary demands.

This conservative cycle has no general order-optimality claim. The partial
attention corpus preserves counts while improving checked receive placement.
It currently exposes substantial contextual-replay cost: see the handoff and
FIFO report before expanding its admission. It creates two directional roles,
not channels per cell, and never moves endpoints in a finished plan.

Distinct final-reader release publications retain separate channels. The former
later-publication/earlier-acquisition `joinedCycle` merge is removed: bank-phase
agreement and token balance alone do not establish order preservation. Common
identical publication frontiers can still share; broader movement requires a
separate ordering certificate. More event pairs can preserve more overlap.

## Selected updates

Updates reuse a predecessor-closed unchanged prefix. Each materialized endpoint
invalidates the saved
selected map from the component of its earliest changed word onward and
recomputes every finalized site from there to the consumer from bottom. It
rechecks affected payloads and both sides of each selected key use. No old
ledger-version fact seeds a changed traversal. A failed update stops construction;
there is no alternative-plan scoring, recoloring, retry search or legacy fallback.

Components before the earliest changed word are reused verbatim. The
construction order is a topological order of the strongly connected components,
so those components have unchanged equations and unchanged inputs and therefore
the same least solution. A cyclic component is reused only from a completed fixed
point, never from the hypothesis-seeded traversal of the active component, and
the active component is always recomputed. Invalidation includes every reachable
site sharing an edited observation word, even when its canonical site is later
in control order or unreachable. A reused endpoint aggregate must have all its
occurrences in the reused prefix; otherwise the boundary moves earlier.
Reused endpoint snapshots are copied
with their cuts. `replaySiteEvaluations` therefore counts only recomputed sites;
`selected_update_test.cpp` checks that each update evaluates at most the sites
from its earliest changed word to the consumer.

The canonical word of a site, the sites sharing it, and the component span of
each word are facts about the immutable program and control alone, so `Control`
builds them once. The boundary computation reads that index instead of rescanning
every site per query, and `Ledger` resolves a word through the same memo. This
work is deliberately outside `replaySiteEvaluations`; moving work out of a counter
is not an improvement, so it is reported as wall time.

`selected_update_test.cpp::ReplayTestAccess` compares the incremental and cold
replays of one edit on the complete semantic checkpoint at every cut, plus every
endpoint aggregate, and requires an inadmissible edit to be refused identically
with the same reason and cut. `compareEveryCut` sweeps every legal original cut
as the edit position on both replay paths.

The frontier's per-primitive closure is incremental for the same reason: the
retained relation is already transitively closed and a primitive adds three
fresh vertices with no edge back into it, so each old row gains only the fresh
vertices it reaches through its gate, prefix, or matched publication. The
differential bridges compare the resulting facts with the exact reference.

Acyclic forward advancement reuses an unchanged predecessor map only when its
ledger version is unchanged. `selectedUpdates`, `replaySiteEvaluations`, normal
`forwardSiteEvaluations`, finalized-query counts and changed cuts distinguish the
work. `recurringQualificationMicroseconds` measures recurring proposal qualification
inside this constructor, excluding native admission queries. Final helper
composition reports trial count, cold-check site visits and elapsed time
(including indexing and ledger copies); the final certificate has separate
visits/time. Both cold-check populations remain subsets of the historical
`invariantSiteEvaluations` aggregate and must not be added to it a second time.
`elapsedMicroseconds` includes portable model/control/storage preparation,
policy construction and final compact validation. `preparationMicroseconds` is a
subset; native import/emission and external reference checking are excluded.

## Structured construction and proof boundary

Choices keep their original successors and share one word per original
observation. A nonparticipating source is never awaited through a branch-local
publication: when an early correspondence is unavailable, the selected policy
uses the actual current-cut closed protocol, or refuses.

For unrefined reducible loops, construction provisionally retains body access
classes at the header and uses natural-loop exit summaries for surrounding
readers. It does not execute a failed future payload. These construction-only
summaries neither change the program graph nor establish a final certificate.
Completed loop components are checked on the **original** edges with a joined
fixed point from the actual incoming interface. Child exits are open; only the
invocation exit checks token balance and retirement. While-before is not treated
as an optional body. Irreducible/multiple-entry loop interfaces are refused.

Recurring common-cut channels keep distinct stable forward and acknowledgment
roles during a loop component. Compiler role reservations are released after the
component, but physical consumption/occupancy state is not reset. These
conservative protocols can require more keys and add more order than a qualified
readiness/release recurrence.

The recurrence qualifier projects each physical cell's access roles through a
qualified normalized region. It checks original first/reused/next/final cases,
write/read alternation, and unambiguous first/last reader boundaries on the finite
graph. Multiple same-engine readers, alternative control paths, unrelated effects,
surrounding accesses, and invocation retirement are retained. An ambiguous role
uses ordinary construction instead. Different cells with identical endpoint sets
share one physical prefix; independent loops do not require a product of their
observation vocabularies.

An enclosing physical-bank orbit can compose with an already qualified child
recurrence. Native import retains the exact bank selected by each original
residue and adds only that finite bank identity; it does not add enclosing
first/tail or elapsed/remaining modes. Existing child modes and effects remain
intact. Copies of a child interface export all corresponding entry and exit
boundaries, so guarded participation and re-entry are checked over the composed
original graph without treating a region boundary as a storage or event reset.

The occurrence reader distinguishes the child's counted residue from the
enclosing bank-only residue. The recurring qualifier then carries the previous
participating use of the same physical bank and its event generation at the bank
period. It still returns readiness/release obligations rather than commands.
Selection processes those obligations through the ordinary remaining-residual
and actual-credit rules; an enclosing interface is retained only when the
resulting graph admits a recurring physical access relationship. This is a
storage/control mechanism, not an operation-name or GEMM-pattern recognizer.

For observed loops that do not match the strict alternating-cell form, the
qualifier derives candidate recurring interfaces from the shared storage
succession relation. It first tries one word for a complete pipe direction and
then independently balanced operation pairs. A finite empty/full participation
monitor rejects branch bypasses, repeated publication or acquisition, and live
tokens at exit. The combined candidate words are replayed by the shared fixed-plan
analysis before construction; only missing completion may remain for ordinary
F1--F7. If that protocol certificate fails, the strict qualified interfaces are
restored unchanged.

Within an observed cyclic component, saved prefixes may also use a unique
forward predecessor corridor. Coverage is read from the actual saved causal
snapshot, and any intervening issue of the same access class invalidates the
candidate. This makes qualified continuations useful without treating static
body order as a generation certificate. Unobserved structured loops retain the
conservative common-cut path.

This qualifies **physical access ordering**, not produced-content identity. A
partial or may-overlap write is not promoted to `definiteWrite`. Whole-operation
completion can order conservative byte witnesses without proving a complete
value overwrite. The exact-slot order-quality claims still require the matched
exact-cell/full-write model used by the reference tests.

The qualifier returns requirements and original endpoint sets, not commands or
key numbers. The constructor reserves the lowest eligible keys in one global
ledger, respecting fixed words and reservations. It then processes surrounding
and remaining effects through ordinary F1–F7. There is no whole-program cyclic
shortcut. If an enclosing path can re-enter the initializer, the last local
return must also establish readiness-consumption knowledge at the publisher;
that return is an explicit rearm obligation, not an unconditional region drain.

While a qualified body is under construction, contextual replay solves the
selected commands over all original edges. Unfinished payload contributes its
actual pending effects and native ordering only: it does not insert the desired
memory-conflict edges. Event preconditions remain strict. Each edit checks all
previously finalized requirements at convergence. No stale completion or
provisional receipt escapes the selected ledger.

The same contextual path is required from the start when a command word or
original payload has multiple reachable analytical occurrences. A partially
visited hypothesis traversal cannot certify all copies. Ordinary shared-word
sources also require balanced participation and consumption-before-next-publication
support, independently of the experimental first-write option.

F7 can return consumption knowledge joined from alternative WAIT sites at an
existing publication position. This requires an empty forward key, actual
consumption knowledge at the reverse source, a usable reverse key and a checked
complete exchange across all occurrences. It does not select one branch as
representative. The initial implementation requires distinct publication and
consumer cuts; common-cut repair retains the closed-exchange path. See
[constructor compatibility](oahs-constructor-compatibility.md) for the exact
admission, counters and regressions.

Contextual replay reuses a predecessor-closed set of components from a successful
whole-original-graph contextual fixed point. Invalidation starts at every
reachable occurrence of every edited word. It closes under original control
successors and under all occurrences of each nonempty shared word reached by
invalidation. An unchanged alternative branch may therefore be kept even when
it follows the changed branch in the component ordering. Sequential successors
cannot be kept merely because their own command words are unchanged.

Every site applies the pending transfer regardless of whether its payload is
finalized, so unchanged words and a predecessor-closed unchanged region certify
the same incoming interface and the same least solution. Reused components keep
their complete cut states. The shared-word closure also ensures that every
endpoint aggregate belongs entirely to reused or recomputed occurrences; no
cached aggregate imports stale contributions from a changed branch. Changed
components restart from bottom with only actual reused-predecessor outputs as
boundary inputs. The solve still covers all original paths, and all finalized
requirements are checked afterward. Failed/partial caches do not qualify.

The original prefix-only rule remains the fallback when no whole contextual
fixed point is available. `SelectedOptions::siblingReplayReuse=false` (diagnostic
driver `--prefix-replay`) retains it for controlled comparisons. The ordinary
construction replay path is unchanged.

`SelectedOptions::traceReplay` (diagnostic driver `--trace-replay`) records
per-contextual-solve work and restart causes without changing the selected rule.
It includes all edited-word occurrence components, control-component edges,
unique/repeated evaluations, reused sites and finalized queries. See
[FIFO replay attribution](oahs-fifo-replay-attribution.md) for baseline costs and
the implemented sibling-reuse certificate. Dependency-walk sites, edges and word
occurrences are counted separately from expensive causal replay evaluations.

The cost that remains is the fixed point of the ACTIVE cyclic component, which is
re-solved per edit. Inside a strongly connected component every site is reachable
from every other, so invalidating a dependency cone recovers nothing there, and
seeding the previous solution is unsound because a command edit can both add
completion and reset event occupancy. Whole-pass cost must therefore still be
measured, and it scales with the edit count times the active component.

Final cold checking discharges entry, body, backedge and exit obligations from
the actual invocation state. `SelectedLoopInterface` exports the incoming and
outgoing states and original-observation clauses at the final ledger version.
These are derived proof snapshots, never extra assumptions supplied to the
checker. Fixed cell/key names are retained; an actual acquisition creates each
new consumption generation. Child exits preserve surrounding pending effects
and live events; only invocation exit requires global cleanup and retirement.

The must join deliberately loses some disjunctions. The update test contains a
safe pair of concrete branch continuations that the joined state refuses; exact
collection is a development reference, not a production fallback.

## Live native adapter

The selected native entry retains the original SCF graph, instruction anchors,
choices, zero-trip alternatives and backedges. Before construction it refines a
normalized leaf `scf.for` only when shared imported byte effects qualify a
physical ready/release role. The supported normalization uses index induction,
zero lower bound, unit step, no iter_args, signed comparisons, and compatible
original residue predicates. No operation-name table or full-write inference is
introduced. Other loops retain ordinary conservative control; there is no retry
with another constructor. The analysis-only report can still expose normalized
first/tail observations independently.

Nested slot mappings are imported conservatively first: an operation outside the
currently specialized orbit keeps the union of all physical bank effects. After
leaf recurrence qualification, a finite enclosing bank orbit may specialize
those effects and compose child boundaries as described above. Only the bank
period is expanded, reusing residue-zero sites; the complete product of nested
counted first/tail modes is not constructed. Malformed boundaries, duplicate
bank dimensions, may-write promotion, and candidates without a qualified
recurring access are declined.
The emitter groups identical complete ordered words only at the same original
anchor. It forms the exact union of their original predicates, absorbs redundant
clauses, and removes a predicate dimension only when all its values are present.
Predicate arithmetic is reused within a block when its definition dominates the
anchor. Read-back decodes the union and reconstructs the word for every original
observation; exact word comparison and cold verification remain mandatory.
Different words, payload anchors, and publication prefixes are not merged.

For unrefined counted loops with constant signed bounds proving at least one
iteration, native import also retains the original body entry. This includes
non-unit-step loops. A source prefix outside such a loop can be acquired once
at entry when its required access classes are invariant inside the loop, the
consumer cannot be bypassed on an exiting body path, no earlier payload on its
observing engine is crossed, and whole-graph endpoint participation is balanced.
The first phase also indexes the original word positions between entry and
that deadline. Admission checks their current selected commands: an observer
publication or ALL fence can transmit the newly acquired dependency to another
engine, even without an earlier observer payload, so such a crossing is refused.
The deadline's existing word is included; entry commands preceding the appended
acquisition and commands after the consumer do not block this placement.
Repeated surrounding entries reuse existing causal consumption knowledge first;
a return is proposed only for a reported missing rearm certificate on that key.
Contextual replay is an explicit construction property, separate from physical
key reservations. One-shot entry handoffs do not permanently reserve their
keys: later uses still need actual emptiness, consumption knowledge and a clear
endpoint interval. Repeated entry protocols retain their recurring reservations.
When no suitable saved prefix exists, a source-inactive enclosing region may
publish and acquire its incoming source prefix at entry. The source must issue
no payload or synchronization inside the region; every first observer path must
require the same completion, and the crossed-publication exclusion still applies.
This handles invariant incoming reader completion as well as writer readiness.
Native counted refinement preserves a positive constant-bound proof by excluding
only its impossible initial zero-trip alternative; it still permits termination
after each admitted positive number of visits. Unknown and zero-trip bounds keep
conservative placement. This is an invariant-entry mechanism, not general correspondence
between successive uses of alternating banks across guarded loop bodies.
The first phase builds each qualified region's issued access classes and unique
first-consumer frontier per observing engine once. Alternative first consumers
are admitted only when each needs the same required classes. A bypass or an
unrelated first payload leaves the entry deadline unqualified. The forward constructor queries
these immutable summaries, then separately checks the selected source receipt,
paired endpoint participation, and physical-key protocol. Preparation visits are
reported as `loopEntryPreparationSites`; they are not hidden in replay counts.

Recurring proposals allocate keys in temporary state and decline on insufficient
capacity before committing endpoints. A staged mandatory-protocol check now
rejects invalid proposals without committing their state. Exact-fit admission
also checks whether uncovered cross-engine demands lose their entire direct
key direction; such a cohort declines to ordinary construction. This remains
a conservative local admission test, not a complete future-allocation proof.
The proposal is materialized as canonical ordered endpoints before analysis,
including guarded publication-first ordering. Mandatory protocol, producer
support and resource checks inspect those exact words. An omission trial uses
the same materializer; an accepted trial retains its checked endpoint list.
Commitment only assigns channel IDs and appends that list, with no later
reordering. Existing ledger words retain their order before the new endpoints.
This is necessary because moving a return publication before an acquisition
can remove the consumption evidence that the return was supposed to carry.
See [admission results and limits](oahs-placement-experiments.md). A cheap
alternative-direction-path query identifies candidates for whole-channel
omission, but gives no completion credit. Certified replay must retain every
previously covered memory and retirement obligation and satisfy all remaining
event preconditions. A return carrying necessary consumption knowledge is
therefore retained even if its memory effect is redundant. Trial counts and
site evaluations are recorded separately. This finite selection can require
multiple full analyses; it does not make construction two linear scans.

Known and Overlap remain separate requirement stages, with the current-credit
provider exception described above. Look-ahead supplies original
placement and participation facts; actual transfers supply completion knowledge.
Native order quality and device latency still need calibration.

### Required output returns support other operand reuse

For an exact period-one storage cycle, `qualifyPipelineCycle` recognizes three
engine roles: producer write, middle in-place accesses, and final reader. The
outer writer/reader projection establishes occurrence and boundary participation;
the selected protocol contains the actual producer-to-middle and middle-to-reader
readiness hops plus the reader-to-producer reuse return. Both readiness hops
must correspond to original access requirements. Middle accesses must form one
straight corridor in the same qualified occurrence.

An independently loaded operand keeps its early readiness. Its private release
is not created when the primary producer precedes its write and the middle
publication follows its last reader in the same occurrence. The required final-
reader return then carries its completion and readiness-consumption evidence
before its next write/publication. This is structural selection before physical
key binding; only the actual selected words and replay grant causal credit.
Unrelated readers remain separate requirements. Unknown cells, guarded middle
routes, larger bank periods and unrelated existing owner protocols decline.

The post-RMSNorm instance retains the input/gamma deadlines and genuine vector
barriers. It uses four recurring channels rather than two, while avoiding the
ordinary duplicate transfers: emitted static pairs fall 16 to 10. It adds no
analysis sites and uses no omission trials. The qualifier performs cell/loop
scans, occurrence comparisons and finite balance checks; ordinary replay and
unrelated legacy trials retain their costs.

The GEMM, attention AIC and post-RMSNorm cases identify three distinct causes
of excess synchronization: lost physical-use correspondence, an existing
provider considered too late, and an incomplete selected cycle at the repair
deadline. Diagnose which fact is missing before extending construction. A
required receipt can cover several storage and key-consumption obligations;
that sharing does not require combining their early readiness publications.
The next complementary question is whether a necessary acquisition is placed
before its actual deadline and unnecessarily gates independent ingress work.

Measure redundant-transfer reduction separately from newly permitted pipeline
overlap. Post-RMSNorm reduces commands with unchanged checked payload ordering;
GEMM improved overlap by waiting for the preceding use of the same bank instead
of a different bank's recent computation. Preserve early publications and place
acquisitions at their real reuse deadlines. Counts and finite payload relation
sets do not fully represent hardware drain costs or predict latency; retain
negative overlap tests and qualify gains with device profiles.

### Carried scalar slots and physical bank correspondence

The shared `SyncSlotMapping` helper derives finite scalar orbits from original
normalized counted loops: constant in-range initial slots and nonnegative,
nonoverflowing `(slot + stride) % modulus` updates. Multiple slots use a common
qualified period. Constant/add/multiply/remainder/index-cast address expressions
are evaluated over that period, not over runtime trip counts. Unknown initial
values, unsupported arithmetic, narrowing/overflow and periods outside the
selected observation vocabulary keep conservative storage coverage.

Native import derives finite possible physical footprints for non-leaf loops
too. An outer A-bank pool can be disjoint from a B-bank pool without qualifying
which visit accesses each bank. Unknown expressions retain conservative aliases;
this does not form a Cartesian product of enclosing-loop periods.

Occurrence specialization admits a finite enclosing scalar-bank orbit without
forming a product with the child loop's first/middle/final modes. It copies only
the enclosing bank residue, specializes shared direct-allocation footprint
records, and exports copies of the child entry and exit boundaries. Invariant
child effects remain shared. Original anchors, addresses, payload instructions,
and runtime control remain unchanged.

The recurring qualifier composes this bank identity with the child's existing
producer/reader interface. It identifies the previous participating use of the
same physical bank across child invocations and carries the actual ready/release
token state through those boundaries. Empty child visits preserve the incoming
interface. Exact qualified cycles are selected directly and do not invoke
whole-plan channel-omission trials.

For two operands occupying disjoint pools but sharing one bank selector, the
constructor keeps their readiness publications separate. Each can therefore be
acquired at its own first reader. Their storage-release returns may share one
channel only when both cells have exact reverse cycles, complete guarded
occurrence correspondence, comparable frontiers, and a balanced merged token
stream. This prevents count reduction from delaying A readiness until B is
available.

A separate first-use qualifier recognizes conjunctions of original normalized
loop first-visit equalities. It splits only the entry prefix until the decision
or participating backedge and rejoins the existing graph afterward. The copied
prefix shares physical operations, observations, storage history, and event
state. Missing terms, disjunctions, unsupported bounds or steps, and intervening
while loops retain conservative control. The qualifier removes impossible
repeated initialization paths; it grants no completion credit and does not
weaken the access-scoped native accumulator rule.

The exact stripped Shenggan step4 payload is a native regression accompanied by
an independent concrete local-memory/event-order check. The selected plan keeps
separate early A/B readiness, shares the MAT release per physical bank episode,
and has no named local barriers. For one, two, and four output-tile entries it
executes 182, 360, and 716 event pairs plus one terminal ALL. The checker forbids
current-bank compute completion from gating different-bank preparation, child
compute from gating parent DMA, and B readiness from delaying the first A
reader. Its complete payload-order relation is a subset of the reconstructed
manual protocol on those traces. This is a plan-quality result; device latency
requires separate measurement.

Construction batches identical guarded local-fence decisions into one ledger
update and one emitted command word. On the exact payload, qualified recurring
cycles require no changed-plan omission analyses. This avoids the earlier
expensive refinement path while retaining final causal and native read-back
validation.

The original Qwen3 RMSNorm and post-RMSNorm kernels exercise this representation
in native regression coverage. Their unrelated reduction loops require
conservative body hypotheses without inferred first/tail correspondence.

The frontier now represents the existing synchronous-payload contract by putting
the next launch gate after that payload's completion. It does not turn SET into
a source-side barrier. Terminal ALL is allowed only at the original invocation
exit and records retirement without consuming notifications or granting interior
completion/rearming credit. Mixed retired/nonretired inputs retain both a may
continuation restriction and a must exit requirement.

`runHandoffSync`, used by `algorithm=handoff`, invokes `constructSelectedPlan`
and validates reconstructed commands with `checkCausalFrontier`. Both mutation
hooks share this exact implementation, including import, private-copy transaction,
SyncCodegen, and exact command/guard/payload read-back. The optional selected
report describes construction; LogicalResult also includes reconstruction.
Shared fixed-plan diagnostic services remain available independently of
construction.

Preserved queue, atomic-store and hard-collective operations continue through
lowering-owned shared extraction and unchanged original IR. No peer, UnitFlag,
implicit resource, cache visibility or intrinsic-drain completion is invented.
General authored/internal local-event models, selective phases, visibility and
exclusive-resource adapters remain explicit refusals where not represented.
Bounding geometry still does not imply a definite full write; native known
readiness remains conservatively classified under the existing importer.

`pto-oahs-selected-test` tests ordinary/branch/loop construction, retained A3
queue and hard-collective inputs, and transactional mutations through this hook.
Its `--construct INPUT` path writes synchronized IR only after success and reports
construction/reconstruction and work counters separately. It is a developer test
entry, not another production pass mode.

## Validation and remaining qualification

The accompanying portable tests construct from unsynchronized input. The exact
reference bridge checks 110 selected programs, including 100 deterministic
straight-line/choice/loop/nested inputs, four cyclic slot populations, and two
open/re-entered recurrence cases. Every
exported causal fact is checked against every reached exact state. Eight selected
examples are also checked by the unchanged paired-order collector; four
straight-line examples additionally use an independent full-history graph.
Cyclic endpoint-deletion tests cover 120 original-observation endpoint mutations.
The reference files remain byte-identical to the pinned repository versions.

Standalone tests build the production library. Native integration builds the
selected and overflow regression drivers. The native drivers are registered in
`check-pto`. Real-kernel replay is reproducible with:

```sh
python test/oahs/replay_prefill.py --build /path/to/native-build \
  --output /path/to/new/report-directory --timeout 60
```

Run with the Python version configured for the native build. This captures all
21 pinned prefill families (23 functions), preserves module-level failures and
timeouts, compares default/explicit `existing`, and lowers successful live handoff
output without a second synchronization insertion. Static inventories and host
timings are diagnostics, not device correctness or performance evidence. This
cohort does not enumerate all PyPTO/pypto-lib kernels.

The live constructor now integrates qualified region-local recurrence and
conservative F8 control handling. Arbitrary first-active predicates, general
affine slot inference, additional typed adapters, full population coverage, and
device qualification remain separate work. Report all refusals/timeouts and
whole-pass cost rather than treating this switch as production replacement
qualification.

## Retained calibration code

The small, unchanged `reference/vendor/v08/causal_interface.py` and
`order_interface.py` check actual constructed plans and exported facts. Their
historical directory name does not imply stale semantics: these modules match
the draft v0.19 counterparts. Unused JSON cases, stored ordinary schema and
certificate archives, and their unused reader/generator modules were removed.
The phase-reference campaign remains a separate model test; its constructor
cases explicitly require refusal while that adapter is unavailable.

The diagnostic all-residual interpreter remains because analysis-only native
reports use it. Its obsolete provisional/first-failure construction mode is
removed. `PrefixQuery`, `BundleQuery`, and `ReplaySession` still have diagnostic
and calibration callers; they are not live alternative constructors.


## Reusing unchanged construction prefixes

Ordinary construction retains its current hypothesis-seeded DAG boundary while
the ledger is unchanged. Advancing propagates only the just-finalized payload
and evaluates the next site, including its original header seed. Earlier
endpoint edits invalidate this cursor and use the existing replay rules. Saved
source snapshots are indexed by cut, avoiding an all-source scan on unchanged
advances. Cyclic fixed points, edits and the final cold check retain their costs.

Shared physical-address extraction also evaluates supported constant expressions
using the checked scalar evaluator. Unknown inputs, overflowing arithmetic and
narrowing loss retain conservative footprints; address certainty does not grant
definite-write or occurrence credit.

## First-consumer invariant input placement

`NativeFirstConsumer.h` qualifies a compact first-visit prefix of a proved
nonempty leaf loop. Only invariant inputs from a source pipe inactive in the
reader region are eligible, and admission requires later work on the source's
incoming corridor. The acquisition belongs at that input's first actual consumer,
not necessarily at the first operation on the observer. Positive non-unit steps
use the original `iv < lower + step` predicate with checked arithmetic.

`loopEntryFrontier` can use that cut from a later construction visit, with its
existing source-time coverage, invariant-class, participation and event checks.
Shared unconditional words cannot repeatedly consume a first-only publication.
Completed entry-protocol keys can be reused only with actual consumption credit
and the full protocol check, never merely because a lexical region ended.
See [qualification, measurements and limits](oahs-first-consumer-placement.md).

## Last-reader release within a child

A straight, nonempty reader child can expose a final-visit post-read source
before unrelated trailing work. Native qualification supplies an original
`LoopHasNext` observation; nearest physical-use roles and balanced participation
select the earlier source for the existing complete reader-region cycle.
The child exit remains its lifecycle boundary. Actual readiness/return commands
still establish completion and rearming under staged and final validation.
See [scope, ordering checks and cost](oahs-last-reader-placement.md).


## Placement-oriented experimental views (working tree after 8afb)

Default optional admission now checks a private protocol before reserving keys
and declines an exactly-fitting cohort when its staged residuals expose a
stranded ordinary direction. Experimental switches retain separate word-start
source coverage, defer qualified acyclic acknowledgments, admit class-invariant
inputs under disjoint producer work, and probe exact-equal-coverage bindings.
They share existing storage, control and selected-state views; first-pass facts
never grant completion. See [scope, validation and controls](oahs-placement-experiments.md).
The running MAT device snapshot is unchanged. Legacy generic frontier motion
remains default, with an independently testable no-motion control.

## Joint first/final reader prefix

The native first-consumer importer can collect final-reader endpoints on the
same original owner before refinement. Positive constant steps are retained
in the final-distance predicate, with checked arithmetic and a distinct
single-visit case. Only requested words gain predicates; conditional suffix
payloads and command identities remain shared. Final analytical continuations
remain separate to preserve exit/backedge correspondence.

The existing reader-region cycle selector uses this qualified frontier and
validates the exact staged readiness/return protocol. This moves down_proj's A0
release before the unrelated B0 wait while retaining B0's first-consumer receipt.
No receipt, completion or key credit comes from the new observation itself.
The [implementation report](oahs-joint-reader-prefix.md) records the corpus
ordering benefit, additional compiler work and pending device comparison.
