# Selected-plan construction: draft F1–F8 implementation

Specification: current synchronization draft v0.37, policies F1–F8.
The sections below include explicitly bounded implementation refinements.
See [shared semantic extraction](oahs-shared-semantics.md) and
[storage and fixed-plan analysis](oahs-analysis.md) for the input contracts.
`algorithm=handoff` uses this constructor. `algorithm=existing` remains the default
and comparison path; no additional pass mode or legacy fallback is introduced.

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
| F3 | `SelectedGroups.cpp::groups/sourceGroup` and `CyclicFrontiers.cpp`: known readiness and reuse together, then remaining overlaps; recurring cells share only identical endpoint frontiers. |
| F4 | `coverage/groups` and recurring-frontier merging: strict containment among required providers, then stable source ordering; `consume` rechecks the whole residual after real acquisitions. |
| F5 | `Control::after`, saved `SelectedSource` records and ledger placement preserve separate early source positions, including qualified straight corridors inside observed loops. |
| F6 | `consume`: cross-engine repairs first, named fence only for the remaining same-engine residual, then complete payload check. |
| F7 | `SelectedAllocation.cpp`: stable source-time key selection, complete forward/reverse interval certificates, nonrecursive consumption acknowledgment and one shortest eligible route. |
| F8 | `SelectedControl/SelectedReplay`: open choice/loop interfaces and actual fixed-point validation; `CyclicFrontiers.cpp`: qualified recurring relationship frontiers, participation checks and deterministic role allocation. |

Ordinary completion endpoint `request` IDs index `decisions`. Endpoints with
purpose `RecurringCompletion` index `channels`. Both kinds coexist in the same
ledger; `channels` contains qualified physical access-role requests. Neither table supplies completion to the checker.

## Stable packet gaps and exact commitment

`WordGap` identifies a canonical command word and its adjacent endpoint IDs,
including explicit word boundaries. `Ledger::preparePacket` resolves all such
gaps against one ledger revision and freezes the insertion order, physical
commands, provenance and references to earlier packet endpoints. Prepared fields
are private to the ledger. It grants no causal or resource credit.

Staging and commitment consume that same prepared value. The ledger rejects a
foreign owner, changed revision or changed endpoint population before any
mutation. Equal words after erase/restore do not revive a stale proof. Several
commands at one original gap retain their packet order, and canonical observation
aliases receive the same word. Tail packets avoid scanning/copying existing
words; middle insertions index and merge each affected word once.

Ordinary transfers, single-WAIT repairs, joined-consumption packets, common-cut
exchanges, alternative sources, loop-entry transfers and recurring proposals use
this materializer. Common-cut forward execution remains private while choosing
the reverse key; its complete exchange is then committed together. Loop-entry,
joined and recurring checks commit the exact prepared object they analyzed.
Existing eligibility, ownership, occurrence and causal checking remain mandatory;
ordinary transfers add no new full-program candidate solve.

This closes exact materialization, not all binding/placement work. Stable gaps
alone do not prove that a selected prefix is good or that a later edit in another
word preserves its exported ordering. Persistent publication and generation
support, neighboring-use qualification across discovery paths and further
return sharing remain the next increments. The normal-constructor repeated-join
test now requires the actual mechanism without a retry, including branch,
missing-receipt, reverse-only and unavailable-direction negatives.

## Requirement classification cost

`StorageFrontierAnalysis::classifyRequirement` supplies immutable reason flags
without constructing a provenance path. Incoming origin-bit membership proves a
positive-length path without an intervening definite overwrite. RMW sources use
the writer row. Canonical reuse classification needs only those original facts.

Readiness still requires a distinct, nonrepeated full writer that is the sole
incoming writer and dominates the consumer. The dominance condition is expressed
by one cell-local absence bit: starting at entry, propagate until a full write,
recording the incoming bit before stopping. Under the unique-full-writer premise,
every initialized path contributes its last full writer to the incoming row;
therefore absence of an uninitialized path is equivalent to writer dominance.
A branch that bypasses the sole possible writer remains uninitialized. Shared
original-control SCCs establish endpoint nonrecurrence. No selected credit is
created by either query.

Uniqueness is memoized per requested site/cell, absence reachability once per
requested cell, and SCC classification once per analysis. For graph size N/E,
cell writer population W_c and queried consumer population Q_c, the added work
is O(N log N + E + sum_c(N + E + Q_c W_c)), plus relationship lookups. The SCC
sort is inherited; this is not a near-linear bound for all storage analysis or
construction. In particular, retained dense origin matrices and selected replay
are separate costs. Ordinary classification no longer adds a graph walk for
every relationship. Detailed witness paths are extracted only on explicit request.
Counters separate classified relationships, setup sites, origin inspections and
witness requests/visits.

Grouping skips flags when no cross-engine residual exists and in the Overlap
stage. It compares all current providers using the existing policy, selects only
the first winner, then refreshes after actual receipt propagation. This removes
unused ranking without changing the provider policy.

## Look-ahead and publication frontiers

Known readiness/reuse and additional-overlap requirements retain separate
selection stages. Sharing a source engine and consumer does not automatically
combine their publication prefixes. After each actual acquisition, construction
recomputes the residual and provider groups; each edit must strictly reduce the
cross-engine residual before continuing.

`SelectedLookahead.h` indexes immutable original class issues within proven
corridors and strict-future payload reachability. Freshness queries use the same
half-open source-to-consumer interval as the original scan. Future reachability
includes backedges, so the last textual payload of a loop is not terminal.

When a straight-corridor source is unavailable, backward search can find an
acyclic frontier of alternative publication cuts. Each source must contain the
required completion in its actual selected snapshot. The common occurrence query
checks participation over original control; shared observation
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
There is no final-plan deletion sweep or trial cold-check population. Work
counters report pair visits, graph query visits, restorations and net discharged
helpers. The ordinary
loop-hypothesis traversal retains its closed fallback: it does not yet export
these contextual token-generation interfaces. Physical reservations also remain
conservative, so this does not solve allocation under a one-key pool.

These acyclic queries do not themselves qualify loop-entry acquisitions or
previous-use bank correspondence across guarded or non-unit-step loops. Those need original
occurrence and participation certificates, including zero-trip and enclosing
continuations. Backward demand queries guide endpoint placement; the actual
completion and event-generation state continues forward through selected words.

## Selected updates

Updates reuse a predecessor-closed unchanged prefix. Each materialized endpoint
invalidates the saved
selected map from the component of its earliest changed word onward and
recomputes every finalized site from there to the consumer from bottom. It
rechecks affected payloads and both sides of each selected key use. No old
ledger-version fact seeds a changed traversal. A failed update stops that attempt. If it contained an optional recurring
proposal, construction may retry once with a fresh ledger and no recurring
specialization. Both attempts start from the same original input and fixed words;
no endpoints, reservations or causal facts survive the discarded attempt.
`declinedRecurring` records its failure, cut and work. Total elapsed time includes
both attempts; the ordinary work counters describe the returned attempt. There
is no alternative-plan scoring, subset search, recoloring or legacy fallback.

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
work. `elapsedMicroseconds` includes portable model/control/storage preparation,
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

Contextual replay reuses the same predecessor-closed component prefix as the
acyclic path, under the same boundary rule: the earliest changed word is taken
over every original occurrence of the edited observation, and a shared nonempty
word is kept wholly on one side, so no reused endpoint aggregate holds a
contribution from a recomputed region. Two properties of this path make the
reuse admissible. Every site applies the pending transfer regardless of whether
its payload is finalized, so a component's equations depend only on its words
and its incoming interface. And every component is solved to its actual fixed
point over the original edges, never from a hypothesis-seeded traversal, so a
cyclic component before the boundary may also be kept. The recomputed region is
still solved over the whole remaining graph rather than truncated at the active
component, which is what keeps aggregates over shared words complete. Nothing is
seeded with old facts: an invalidated site restarts from bottom.

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
introduced. Other loops retain ordinary conservative control. Declining optional recurring
specialization uses the same constructor and unchanged independent checker. The analysis-only report can still expose normalized
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

Recurring proposals are checked before committing their endpoints. A cheap
alternative-direction-path query identifies candidates for whole-channel
omission, but gives no completion credit. Certified replay must retain every
previously covered memory and retirement obligation and satisfy all remaining
event preconditions. A return carrying necessary consumption knowledge is
therefore retained even if its memory effect is redundant. Trial counts and
site evaluations are recorded separately. This finite selection can require
multiple full analyses; it does not make construction two linear scans.

Known and Overlap remain separate selection stages. Look-ahead supplies original
placement and participation facts; actual transfers supply completion knowledge.
Native order quality and device latency still need calibration.

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

## Inherited-checkpoint hardening

First-use prefix refinement now requires selected decisions to be separated by
a tracked backedge before the next decision. The current transformation returns
to original control after a selected decision, so membership and single entry
alone do not establish its phase premise.

Recurring coalescing retains distinct publication and acquisition boundaries.
Identical endpoint sets can share a transfer; occurrence balance or a straight
corridor alone cannot justify moving boundaries. In particular, one bank's
release cannot be delayed to another reader and acquired before an earlier
refill merely to save a key. Broader motion needs a contextual ordering proof.

Local completion repair is decided at the current occurrence's deadline after
actual selected receipts. Analytical copies sharing an emitted word share its
fence, but unfinished distinct future words receive no speculative repairs.
This preserves the opportunity for intervening required transfers to satisfy
those later demands without introducing another local fence.

## Ordered proposal materialization

Recurring proposal trials and their commitment use `OrderedPacket` through the
same ledger append path. Packet sequence is preserved within each actual word;
fixed prefixes and analytical copies use the ledger's existing canonical-word
mapping. A private materialization leaves live endpoints, revision and change
tracking untouched. Accepted materialization retains logical request provenance.
This removes duplicate candidate/commit encoding; it does not itself prove
placement quality, resource availability or completion. The initial interface
appends at word ends. Exact arbitrary gaps and common multi-leg qualification
remain Stage 3/4 work.

Selected decisions also retain `SelectedLifecycleDemand` provenance from the
shared physical-use index. Its release candidates and return deadlines cannot
be used as completion or consumption credit. Actual command execution and the
unchanged independent checker remain authoritative.

### Persistent producer-support scope (Step K)

An admitted generation proposal retains an immutable support scope: for each
producer engine, the original consumer sites reachable after its overwrite seeds
and the read/write access classes of every operation that can precede those seeds.
All aliases participate. These are obligations, not completion facts or private
channel ownership; omission of a private channel does not remove the scope.

The existing contextual evaluator checks this scope after convergence, initially
and after every changed ledger. At every affected consumer, including unfinished
ones, no residual may belong to a protected source class. Finalized payload and
protocol checks remain separate. Actual causal propagation can discharge the
obligation; an endpoint identifier or the latest static origin cannot. In
particular a later compatible ACC issue cannot hide an older incompatible
history. A failed optional attempt is discarded by the existing fresh ordinary
retry, without retaining its scope, bindings or credit.

This is a sufficient producer-repair certificate. Unioning seed regions and
aggregate access classes can conservatively include a newer generation even if
the earlier origin is complete. The initial origin-sensitive admission check is
retained; first replay and subsequent replays use the same stronger class check.
It is not yet an exact generation-family interface or a certificate that later
edits preserve every publication's causal prefix. Those remain explicit work.

Scope preparation reuses the original reachability traversal and visits each
original operation's access list once per producer engine. Revalidation extends
the existing stabilized inspection pass; it adds no fixed-point solve or witness
search per relationship/edit. Work is charged separately in producerSupportWork.

### Enclosing bank facts and materialization (Step L)

Successful semantic bank refinement is admitted independently of whether a
private recurring protocol qualifies. The removed hasQualifiedRecurringAccesses
query copied the refined program and rebuilt analyses solely to ask that protocol
question. Boundary closure, original participation, effect validation and complete
child entry/body/exit correspondence remain requirements of refineBankOccurrences.
Physical relations grant no completion or event-allocation permission.

Native materialization has an independent limit of4096 added graph sites per
bank refinement, including copied headers: (period-1)*(bodySites+1). A refusal
retains the original control and independently derived physical-use relations,
with a distinct materialization-budget diagnostic. This is an added-site budget,
not a bound on all accumulated graph sites or child-interface metadata. The
existing restriction against multiplying a second enclosing bank dimension
remains explicit; it is not replaced by private-cycle qualification.

Varying effects are computed once per binding, including native ACC classes.
All residue variants for a used phase are derived together from immutable input.
An unchanged residue or an outside occurrence keeps its original phase record;
changed variants get separate records. Only a completely replaced phase can
reuse its ID for the first changed variant. This avoids residue zero corrupting
a later unchanged residue or an outside occurrence, while preserving the
validator's requirement that each physical phase appear in observed control.
Per-phase preparation is cached; work follows represented effects and copies,
without a fresh period-wide scan for every residue.

### Exact source-gap qualification (Step M)

An ordinary direct transfer may place its new publication before an incoming
wait in the same selected command word. The query resolves stable left/right
endpoint identities against the current ledger revision, executes that word's
actual prefix from the cached incoming causal state, and checks publication
legality and every required physical access at every matched occurrence.
Correspondence must be proved; source and receipt are acyclic and straight, and
no intervening issue may refresh a required class. Unknown retains tail placement.

The initial neighboring-use certificate requires a virgin physical key, including
dormant endpoint records. The ledger indexes all event uses by physical identity;
active interval checks and virgin-key queries share that index. Removing endpoints
does not erase ownership history. No new full-program solve is added per candidate.

The client scans one word suffix, stopping at source publications, source fences,
ALL, or a use of the selected key. It checks one earliest eligible gap; it does not
search all intermediate positions or change the existing key-selection policy. A
reused selected key can therefore retain tail placement even when another unused
key exists. The qualified revision-bound gap is consumed by the existing exact
packet materializer. Relay, recurring and structured-source clients retain their
existing placement until their additional premises are represented.

This is local qualification of a NEW publication under the current surrounding
ledger. It is not persistent absolute prefix protection or a guarantee that later
greedy decisions add no ordering. General protection requires an affected-interface
closure or withdrawal/requalification of the placement and dependent support. A
later mandatory repair cannot simply be rejected to preserve an optional prefix.

### Shared actual source coverage (Step N)

Motivating requirements select a provider's source boundaries. A separate query
intersects actual completion histories at its participating publication sites;
additional coverage never changes those already selected boundaries. Ordinary
sources retain the indexed source-to-deadline freshness check. Alternative
sources collect intervening issued classes during the existing source-discovery
walk, excluding its stopping source payloads. Loop-entry providers check every
reachable source-word occurrence, source-to-entry freshness and the original
region's issued classes. No prospective receipt supplies extra completion.

An independently required Overlap provider may compete during Known selection
when this actual coverage includes a due Known requirement. A source already
providing Known requirements retains its earlier prefix rather than combining
its demands with later Overlap work. Deterministic next-winner selection remains
unchanged; after a real receipt, construction refreshes the complete residual.

Every winning group and selected decision retain supporting requirements
separately from the motivating demand, including extra coverage of an already
Known provider. Earlier exact-gap placement must preserve these additional
requirements too. Failure retains the existing source word tail; it cannot move
the publication before the receipt which supplied its advertised coverage.
This is selection metadata, not an additional causal completion state.

Coverage work follows requested classes and represented source occurrences.
Alternative freshness adds no graph traversal per class; loop-entry queries reuse
original issued-class indexes and cached correspondence. Candidate populations
remain bounded by the existing engine grouping, and no extra full-plan solve is
introduced. The existing loop-entry packet qualification remains a separate cost.

The alternative crossed-class union and whole-loop issued-class test are
conservative sufficient freshness certificates, not exact generation intervals.
They may decline valid extra coverage; a refined replacement must preserve their
obligations rather than merely remove those tests.


## Exact restoration packets (O foundation)

Missing fact: a dormant endpoint retains its physical identity and provenance,
but the append-only packet representation could not include it in the same
transaction as a new transfer. Ledger preparation now resolves an ordered mixture
of original inactive IDs and fresh endpoints. A packet-local acknowledgment names
that ordered sequence, not an assumed contiguous range of new IDs. Candidate
commands and committed words use the same materialization. Existing restoreReturns
is a consumer; its subsequent selected replay remains the causal check.

The ledger indexes dormant physical identities independently of occupancy and
consumption knowledge. Erasure retains event-use membership; restoration changes
activity without duplicating that membership. This index is not yet allocation
permission. Default deferral remains disabled and shared owner closure is the
next increment.

Gate inventory: original cut/command/purpose/request/acknowledgment equality,
inactive distinct restored IDs, active original acknowledgment anchors, adjacent
stable gap neighbors, and ledger owner/revision equality are necessary integrity
premises for this API. They are independent of dtype, address spelling and unrelated
payload. Changed neighboring commands invalidate a prepared gap; a fresh query
may recover it. Cross-word relocation is intentionally outside restoration:
restoration preserves the original identity/word and does not certify motion.
No new optimization admission restriction is introduced.

Evidence includes mixed restored/new IDs and acknowledgments, stale and foreign
ledgers, malformed identity/provenance/cuts, inactive anchors, unrelated interleaved
commands and canonical-word aliases. Existing normal construction exercises
helper discharge and changed republication deadlines. Native effect/descriptor
and expression variants exercise the unchanged import boundary; this increment
adds no physical/control recognizer, so new dtype/expression cases cannot activate
its packet-integrity rules. General dormant-key pressure, multi-owner closure and
common binding remain unfinished and require their own construction witnesses.


## Shared dormant ownership qualification (P)

Missing semantic fact: an empty physical event can still be owned by an erased,
restorable helper. The existing ledger event-use index and helper provenance will
supply a complete owner closure; every constructor packet client will consume
that same closure. Ownership is independent of physical occurrence facts and
actual consumption knowledge. No new default deferral is enabled by this work.

A qualified packet retains exact prepared words and the restored helper receipts.
For every touched dormant identity, all inactive records must belong to complete
inactive helper pairs with live matching forward receipts. Original gaps are
resolved in a batch per touched word. Unaccounted/partial ownership is unknown;
no subset of owners is tried. The expanded packet is checked before commitment,
and restored receipts become required only after the same prepared words commit.
Ordinary unowned packets keep their existing local fast path. Exceptional pressure
tries deterministic eligible hardware keys, not alternative whole plans.

The independent fixed-plan analyzer is a precommit protocol/resource filter.
Its residual abstraction is not equivalent to the causal frontier, especially for
mixed ACC incidences; authoritative selected replay remains mandatory after the
edit, and any refusal rejects the construction attempt. This is not a new promise
of complete precommit causal qualification. Likewise restored early waits can add
order: report that separately from safety and count it as resource-driven closure,
not an order-preserving optimization. General publication-prefix protection remains
open. Whole construction success, ordinary-path discovery and complete ordering
comparisons are acceptance evidence separate from supplied-packet legality.


P gate inventory and evidence:

- Complete pair membership, original immutable endpoints and live acknowledged
  forward receipts are necessary ownership/provenance premises. Every dormant
  endpoint of a touched identity must be accounted for; partial/unknown owners
  refuse without changing ledger or support sets. Equivalent IR spelling and
  unrelated physical work do not change this query.
- Existing closed/recurring reservations remain owner-specific. The run loop
  initializes closedKeys from recurringKeys and inserts both for retained entry
  roles. This is a sufficient current reservation scope, not general lifetime
  coloring. Replacing it requires scoped ownership intervals, still unfinished.
- Gap identity/revision and original-word restoration are necessary for the
  admitted restoration operation. This API does not move existing endpoints
  across words. Helpers with a different original placement need their own
  preserved endpoint/support interface rather than silent relocation.
- Complete fixed-plan protocol/resource checking and finalized/protected residual
  filtering are shared by exceptional and structured packets. Ordinary unowned
  bindings keep their existing local certificates. The analyzer's residual
  filter is conservative and distinct from causal replay; a postcommit replay
  refusal rejects the whole attempt. This is not a guaranteed recovery policy
  for arbitrary deferred acknowledgments. Final retirement remains independently
  checked; unfinished payload/retirement obligations do not veto an intermediate
  packet merely because they are still pending.
- Physical keys are tried deterministically. Rejected candidate zero cannot hide
  candidate one; source positions are retained across binding attempts. Complete
  packet order can establish later-leg rearming. No owner subsets are searched.

Portable tests cover multiple owners, partial/unaccounted owners, unqualified
commit refusal, exact early single-consumption repair and later-key success after
atomic refusal. A public-constructor case spans a choice child, repeated entries
and parent continuation, with physical relabeling and unrelated payload variants.
The native post-RMSNorm fixture and identity-view, equivalent-address and unrelated
producer/descriptor/control variants all bind dormant ownership without retry,
then pass independent native reconstruction. These are effect/protocol tests;
there is no datatype admission rule to qualify separately on a device.

Twenty nonempty finite changed-plan comparisons preserve complete ordering sets;
twenty unsupported diagnostic rows remain unresolved. Event overhead and work
increase in some cases: RMSNorm pairs17→21, with unchanged tested ordering;
CSA compatibility pairs100→107 and84,022 ownership-check site evaluations. Some
ownership counters include the already-existing joined/loop-entry validation and
must not be added to those counters as independent work. Exact per-case records:
`../oahs-gemm-base-builds/refactor-step-p/`. Generality review accepts this shared
ownership prerequisite, not default deferral, arbitrary ownership borrowing or
persistent publication-prefix protection.


## Shared consumption obligations and actual rearming deadlines (Q design)

Missing fact: return substitution, key-pressure binding and replay recovery still
interpret separate helper-pair/direction tables. One record per tracked fallback, referencing its actual forward
consumption, owns the original fallback endpoints, forward physical key,
pinned-support status and the actual receipt used to justify substitution.
Direction, forward-key and helper-endpoint maps contain record IDs only. Ledger
activity remains authoritative; no record grants completion. A saved supporting
receipt and revision are historical evidence, never a reusable completion bit.

At a failed actual republication, indexed dormant obligations seed the same
complete ownership closure used by new packets. Closure includes every dormant
owner of the restored reverse identities, validates each once, batches gap
resolution and checks one exact packet before commit. It cannot depend on a
successful pre-restoration replay. Empty/unrelated key queries do no graph work.
Pins and counters change only after exact commitment; each restoration pins at
least one previously dormant obligation, bounding replay recovery. The old direct
restoreReturns mutation is replaced, not retained as a second repair mechanism.

Necessary premises: live matching consumption, complete original helper identity,
exact original gap, complete physical ownership and legal packet protocol. Existing
closed/recurring reservations and source-time certificates remain sufficient
restrictions with the replacement obligations stated in P. No datatype, allocation,
expression or kernel gate is introduced. The native ownership variants exercise
these representation/context invariants; changed-deadline and multiple-owner tests
exercise the new repair consumer. Actual acknowledgment deferral additionally
requires preserving fallback availability through later edits and reverse-key
pressure; record consolidation alone does not enable that policy.

Recovery may expose several missing consumptions from one selected edit. If the
complete staged packet still reports missing consumption, add the registered
dormant obligations of those actual failed identities and recheck the growing
packet. Commit only the successful complete closure. Each further solve requires
strict growth by a previously unstaged obligation; no owner subsets or alternate
placements are explored. Exhaustion of new support refuses without mutations; every kind of failure
must disappear before the complete packet can commit. Diagnostic forward keys
are expanded once across all rounds because selected ledger activity is unchanged. This exceptional monotone dependency closure is charged per
staged analysis; it is not a claim of constant replay cost per selected edit.


## Deferred owned common-cut returns (R design)

This bounded policy separates a storage receipt from its future event-reuse obligation.
At one original acyclic common-cut occurrence, first qualify the complete immediate
forward/reverse fallback against the current ledger. Retain its exact original
records as an owned, un-emitted fallback and propagate only the forward transfer.
A future payload alone is not a reason to emit the reverse pair. A failed
forward-only replay restores the already-qualified fallback before finalizing.
Recurring/shared occurrences retain their existing closed exchange; this is a
sufficient initial dynamic-generation boundary, not a general deferral theorem.

Every proposed use of either reserved identity closes the deferred owner through
the same packet binder. The fallback source remains the exact gap after the actual
forward consumption. The gap is an outstanding support use of its publisher's
knowledge: required-return substitution cannot erase another helper if that gap
is reached before its actual replacement receipt on any original path. The query
uses stable consumption anchors at every shared-word occurrence, including gaps
inside words and at their ends. A latent source grants no completion or consumption.

The bounded availability invariant combines a qualified original fallback,
matching-preserving additions with shared identity ownership, and latent-source
protection on the only later endpoint-erasure path. It does not justify arbitrary
endpoint motion. Restoring the original pair is a conservative realization at
an actual resource deadline; moving its acquisition later needs a separate exact
placement certificate. This increment does not claim general recurring deferral
or arbitrary late relocation. Event pressure may materialize the fallback even
when another realization could use fewer commands; record that limitation.

R gate inventory and evidence:
- Single reachable acyclic occurrence is a sufficient generation boundary; shared
  or cyclic cases keep the existing exchange. Refined acyclic roles can qualify.
  Replace this restriction only with an explicit recurring obligation interface.
- Complete initial fallback qualification, matching consumption and both-key owner
  closure protect recoverability. Unknown qualification retains the eager path;
  failed forward-only replay restores the exact fallback before acceptance.
- Latent publication support is protected by the existing substitution traversal
  at exact post-consumption gaps, including branch alternatives. No new whole-graph
  search is introduced. Live indexes remove owners when materialized.
- Native identity-view, equivalent-address and unrelated producer/descriptor/control
  variants activate deferral without retry. There is no datatype gate. Normal
  constructor tests cover stale receipts, skipped consumption and both key pressures.
  Branch-sensitive latent retention is supplied-ledger evidence, not demonstrated
  native selection (native counter zero).

All three reviews and bounded generality accept this scope.36 nonempty finite
complete-order comparisons are identical;10 unsupported rows remain unknown.
KDA static SETs312→390 and ownership sites103079→954169, and attention's6/6 restored
deferrals, remain unresolved policy/work costs. This is not a general performance
improvement. Evidence and separate command/key/work records are in refactor-step-r.
Current key-use closure does not exploit all actual returns or move the original
fallback WAIT; those are the next placement/discharge obligations after the pause.


## Step S invariant — typed original-use boundary queries

Missing fact: enclosing-reader placement consumes per-site marginal booleans and
repeats physical corridor walks. The shared StorageFrontierAnalysis service will
answer an explicit original owner/occurrence, inclusive/exclusive start/stop and
direction query. A compact summary retains read, full/partial write, RMW and open
boundary possibilities. It stops at ANY access, not only a definite overwrite.
NoHit means no represented physical access within that interval; it never means
completion, eventual exit or a drained token. Unsupported queries return Unknown.

Immutable per-cell/direction/stop projections share intermediate facts across
starts and original owner interpretations of the same graph; public requests
retain their full owner/occurrence/interval identity. Exact origin output remains
a separate lazy nearest-use query. Each finite role bit propagates monotonically;
there is no origin-pair path search or dynamic trip enumeration.

RequirementFrontiers projects the shared facts to typed first/final eligibility
at EXISTING original endpoint occurrences. The original observation remains at its
own anchor. Mixed participation retains may facts with Unknown; this increment
cannot synthesize/hoist a later predicate. An interval stop is an open obligation,
not a generation release. Enclosing qualification still checks every shared-word
alias and matching. This is the first query-equivalent consumer migration; the
full D3 guarded composition and residual-driven two-child construction follow.

### Step S bounded scope and cost

This increment is a query-equivalent nearest-role projection, not completion of
the guarded typed-query milestone. Explicit starts/stops define the horizon;
owner/occurrence annotations name caller identities but do not prove their
correspondence. Existing occurrence, shared-word, balance and producer-support
checks remain authoritative. `Exact` in the reader adapter means uniform
eligibility at an existing endpoint. A partial write is a nearest access, not a
proved generation kill. Mixed roles remain Unknown with may roles retained.

At most two empty-stop projections are demanded per cell by this consumer.
Each projection expands only demanded nearest-access corridors, caching fully
solved closures (including zero-role cycles). No provisional result is exported.
Each reached site/edge is expanded once per projection; monotone propagation adds
only seven possible bits per site. Sparse maps introduce logarithmic lookup cost. Distinct stop projections and inherited
all-cell loop scans remain separately charged work, not a near-linear whole-pass
claim. Source subscriptions and Q/R selected-state handling are unchanged.

## Step T design invariant — original structure and guarded read intervals

Missing fact: the original structured read interval is not retained independently
of residue-specialized operation effects. A tree that still names an overwritten
operation index cannot prove original participation. Capture a validated immutable
structure/effect view before refinement; preserve stable original owner/access IDs
and explicit correspondence. This is part of original-program analysis, never a
completion ledger. Missing structural provenance yields query Unknown without
rejecting an otherwise supported program or erasing physical facts.

The shared read-segment query names an original owner/sequence interval, physical
cell and reader engine. First, final and nonempty results describe that SAME
interval. Only overlapping writes (including partial writes) obstruct read-only
composition; unrelated writes remain independent. D3 uses shared original-predicate
expressions rather than enumerated guard products. Counted repetition additionally
needs invariant relevant participation and original arithmetic premises.

Executable endpoint qualification is separate: derived guards remain useful
facts when unavailable at an anchor. Existing observed words can consume them only
through checked implication/correspondence. New LoopNonEmpty endpoints need original
bounds dominance, exact predicate readback, and correlation with the original entry
decision in the graph seen by the independent checker. Sibling-loop bounds need
not have an ancestor loop; IV-dependent atoms still do. Never introduce an
independent nondeterministic guard that forgets that correlation. Retire a demanded
predicate interpretation after its affected interval rather than forming a global
product. Construction activates a prescribed packet only for an actual residual.

T implementation boundary: original structural expressions now drive the existing
`needsOccurrenceSeparation` read-role vocabulary request. Exact original facts
are not executable endpoint guards or a correspondence certificate. No selected
command, event allocation or completion fact enters this query. First-write and
unknown read cases retain the existing nearest-use discovery. Actual enclosing
placement still uses the S consumer until interval/occurrence qualification is
connected; this is not completion of the typed-placement milestone.

Current sufficient restrictions and replacements:

- A read interval containing an overlapping write is Unknown; next increment
  supplies generation-delimited intervals rather than weakening this condition.
- Counted-body nonemptiness must simplify to True. This proves invariant relevant
  participation for the bounded decoder; general guarded repetition needs an
  occurrence/availability proof. Both reading choice arms compose; a possibly
  empty arm inside a repeated owner does not imply a final-iteration reader.
- Per-cell read/write roles must remain present in every mapped analytical phase,
  with unchanged cell geometry and pipe. These conservative applicability checks
  are not exact occurrence matching. Variable relations retain their independent
  physical facts; D1/D2 correspondence must replace this sufficient restriction
  before using a variable-bank frontier for placement.
- Missing structure, duplicate relevant guard identities, ambiguous footprints,
  uncounted relevant participation and invalid handles remain Unknown/Invalid.
  Irrelevant no-hit subtrees need no guard identity. Synthetic periodic creation
  drops unsupported structural provenance; structured import reassigns/captures
  original owners, while checked refinement preserves the immutable snapshot.
- Exact counted readers request first/final vocabulary even when a later consumer
  proves a single visit. This is demand discovery, not a claim that every request
  needs a distinct emitted endpoint. No new placement or provider ranking policy.

Original identities are an explicit trusted import contract, like original effect
completeness. `hasOriginalIdentityMap` validates only index shape/range. Mutating
original control requires recapture, or dropping this optional provenance; an
observed graph cannot reconstruct original syntax merely from its current shape.

Cost: one shared immutable snapshot; sparse original/current incidence indexes;
one summary per demanded (original region, cell, reader); interned predicate and
frontier DAGs; cached complete interval answers. Counted-trip magnitude is not
iterated. `readCompositionParts` charges sequence/range scans separately; distinct
overlapping ranges still cost the sum of their lengths. No current production
client enumerates all ranges. A future range-heavy client requires shared range
composition rather than claiming that summary memoization removes that cost.

## Step U invariant — read segments and existing endpoint applicability

Missing fact: a whole enclosing owner contains producer writes, so its D3 result
cannot describe the read-only segment between them. Extend the same original
index with write-delimited sequence intervals and leaf-conditioned frontier
queries. A may/partial write is a delimiter, not a generation kill. Missing
producer or successor remains an explicit open boundary. This does not certify
that all readers of a physical generation have been covered.

The enclosing consumer asks whether its existing reader occurrence satisfies
the original first/final frontier condition. Three-valued implication uses exact
original observation atoms and lexical branch context; residues alone do not
prove first/final visits. Unknown never means NoHit. Paired owner interfaces and
all reachable word aliases must retain compatible interval roles. Final roles
are transported only through the existing checked publicationAfter relation.
Keep independent balance, producer-support, ownership and causal checks. This
step does not activate protocols or promise future acquired credit. The existing
bulk recurring installer is a separate subsequent migration.

U accepted scope: existing original observation/lexical predicates prove endpoint
implication; unavailable guards or an open incoming segment use S's shared
nearest-role fallback. Exact first/final hits additionally require actual graph
support: predecessors end at writes, and continuations end at writes or true
invocation exits. A write-containing child is not by itself a generation boundary.
Mixed typed/S proofs need equal roles; compare interval identities only when both
proofs are typed. Paired interface records are metadata, not generation matching.
The existing publication transport and full protocol checks remain authoritative.

The original query indexes parents and operation positions once, delimiters per
sequence/cell, and leaf conditions per shared frontier root. Predicate evaluation
is memoized per applicability query; owner interface memberships reference shared
records rather than copying each exit for each member/cell. Counters now include
original composition and endpoint applicability. Unknowns preserve obligations;
this checkpoint does not synthesize optional-child guards or activate protocols.

## Step V design invariant — residual-activated finite support

Missing fact: flat recurring channels erase which complete decoded lifetime needs
which support. Retain logical role identities and explicit family-to-role recipes;
sharing a role never joins the activating families. Index candidate families by
actual payload word/cell. Discovery reserves no key and grants no causal credit.

At the first indexed deadline, establish a cold contextual baseline of the current
ledger, then use its actual complete residual, including producer WAW/WAR. Select
one matching finite recipe in stable order. Stage its exact packet, all existing
ownership and new producer-support obligations privately. Reject without changing
ledger, reservations, support records or causal state. On acceptance commit the
same packet, propagate its actual credit and require strict complete-residual
decrease. No source snapshots are invented for future payloads.

The initial replacement consumes qualified slot/enclosing recipes. Standalone
relationship opportunities without explicit finite rearming support use ordinary
construction; the old population-level protocol trial and omission sweep are
removed rather than used to infer a selected recipe. This is an explicit migration
limitation: recover their useful support through common logical recipes, not by
restoring broader word replacement or completed-plan deletion. Temporary quality
changes are recorded; supported-input service and independent checking remain
mandatory. Preserve original recipes and exact endpoints, including GEMM banks.

New certificate state is private until commitment. The producer-support check
remains a sufficient interval-crossing safeguard, not full publication-prefix
certification. Complete ownership/neighbor checking uses the common owned packet
path. Key allocation initially chooses unused, unowned keys deterministically;
reuse across certified disjoint lifetimes belongs to the binding-policy increment.

V final scope: each family caches its affected scope; support closure follows
finite indexed family links and unions cached scopes. Current-ledger residuals
can overapproximate support needed after packet legs. Missing support declines,
never grants credit. Unhelpful proof results retain per-deadline improvement
information at the exact ledger version, so a later useful deadline is not
silently suppressed. A complete merged support rectangle is checked privately.

`PacketView` overlays only changed words, retaining exact endpoint order and
revision. The common contextual evaluator executes that view using the same
pending-issue, native-access history and join semantics as selected evaluation.
The event analyzer is an additional check, not an authoritative-credit substitute.
After exact commit, the accepted evaluation supplies the new selected state;
refresh source handles without a third complete replay. No future payload snapshot
is fabricated. Failed candidate caches are diagnostic/query state only.

This is a construction-policy migration, not a query-equivalent refactor.
Family-first selection, conservative support closure and unused-key allocation
remain sufficient unfinished restrictions. Supported RMW relationships need
explicit recipes before regaining the old trial-dependent protocols. The finite
RMSNorm ordering losses and global-evaluation cost are recorded in HANDOFF.
Draft v0.38 local probes require a later replacement for both full-graph private
checks; the present version does not claim that milestone or the new class-first
single-scan policy. Admission success alone does not establish preserved overlap.

## Step W invariant — executable original participation

Missing fact: D3 retains a sibling LoopNonEmpty predicate, but an existing reader
word cannot evaluate it or correlate it with the sibling's actual initial visit.
Extend original endpoint-demand queries and native observation refinement. A
qualified predicate names original lower/upper bounds with a positive-step domain;
its values must dominate every annotated endpoint. Ordinal predicates keep their
stronger active-owner/induction-variable premise. No scalar payload is speculated.

A demand-local Boolean quotient carries only this original participation fact
through its named read interval and correlates the actual child-entry decision.
Every feasible original execution retains its payload and endpoint occurrences;
each copied word names the exact predicate. Rejoining the continuation carries
all causal/event state normally. Re-entry resamples original bounds; no runtime
history counter and no completion credit. Existing readerBoundaries and finite
families are the actual consumers. Unavailable bounds or unsupported correlations
remain Unknown, retaining physical effects and ordinary construction.

Initial scope is one independently materialized live predicate per affected
interval; overlapping demands are an explicit unfinished precision limitation,
not a hardware requirement. Independent disjoint intervals do not form a global
product. Both-empty child families additionally require interval-nonempty
readiness/release support and are not implied by the mandatory-first-child case.
Use normal constructor tests, exact emitted predicate readback, negative polarity,
reload/outside-reader cases and repeated-entry/scaling evidence before acceptance.

W qualification inventory (sufficient representation certificates, not hardware
limits):

- Original positive-step domain and exact lower/upper values dominate the first
  relevant read frontier and all annotated positions. Unavailable values remain
  Unknown; neither scalar work nor a predicate is speculated.
- The sampled initial child decision has exact disjoint empty/nonempty successors.
  A single invocation contains the interval; internal entry predecessors and
  decision re-entry are refused. Parent re-entry samples the bounds again.
- All reachable occurrences of each affected original command word lie inside
  the interval. The unconditional entry word has one occurrence and no payload;
  cloned entry nodes have no word. This retains the early unconditional source
  without executing a command word twice or changing its participation.
- Only affected paired loop interfaces are remapped/refreshed. Unrelated legacy
  unpaired interfaces remain unchanged. Proved NoHit prefix regions are trimmed
  through the shared typed query; unrelated operations do not select the entry.
- Disjoint demands are applied in one batch (one program copy and metadata pass).
  Overlapping predicates, a relevant direct-payload entry without a separate
  source gap, and unsupported copied-owner correspondence remain Unknown. These
  require composed guarded-frontier and exact-gap representation, not more
  syntax/dtype recognition. Both-optional/no-reader families remain unfinished.
- The observation materialization budget bounds optional precision independently
  of event pools; original effects and ordinary construction remain available.

This is a precision extension. Immutable D3 demand discovery does not instantiate
storage-history relationships or protocol candidates. Requested frontier roots
share visited DAG traversal, but internal nodes do not cache transitive subject
sets. Interval geometry is reused across predicates. Batch preparation/output and
requested-root traversal have scaling evidence; arbitrary overlapping refusals
and existing occurrence-metadata refresh have no universal linear-cost claim.
The selected causal state, event ownership and authoritative checker are unchanged.

## Step X invariant — restored consumption waits at selected reuse deadlines

Missing fact: a dormant consumption-only helper records the consumed forward
publication but restoration currently binds both return endpoints immediately
after that old consumption. Later required forward-key reuse has an actual
selected republication deadline. Returning knowledge before that deadline does
not require gating the publisher at the old consumption point.

Extend the existing rearming obligation/common owned-packet path: retain the
return publication at its original consumption gap; qualify the return wait at
the exact gap preceding the proposed forward-key publication. The immutable
occurrence query must prove the same one-shot participating interval. The exact
complete staged packet must retain all active/dormant ownership, matching,
neighboring-use and rearming checks. A future return supplies no credit. Source
and target endpoints are committed exactly as checked.

This is a binding-policy/placement extension. It is not a proof that every later
publication prefix stays unchanged. It preserves existing independent publication
positions before this reuse deadline; existing causal/event checking still
handles the full packet. Unsupported occurrence or neighboring-use cases retain
the checked original closed restoration, recorded as a conservative repair.
Do not erase a mandatory rearming obligation when deadline placement is unknown.
The normal-constructor witness, complete payload-order comparison, intended
WAIT-before-republication case and guarded/multiple-owner negatives are required.

The selected restoration keeps one endpoint identity. `originalCut` preserves the
fallback word; `SelectedRestoration` records the consumed identity, helper pair,
actual republication identity, placed/fallback words, revision and refusal reason.
These are diagnostic construction records, never acquired completion. Private
packet views expose the relocated record; commitment installs those exact words.
Active publications, fixed endpoints and helper provenance cannot be relocated.

The current sufficient certificate requires one proposed forward republication,
one dormant owner on the reverse key, proved acyclic straight occurrence pairing,
and no earlier selected/proposed use of either key inside the interval. Exact
word gaps order uses; packet order only breaks ties at the same gap. Multiple
owners/deadlines and recurrence are unfinished composition limits, not hardware
restrictions. Their replacement needs a joint deadline/neighbor proof. No dtype,
opcode, allocation-spelling or producer-uniformity condition is introduced.

The whole dormant-owner closure is still authoritative. If local late placement
passes but its complete packet fails, qualify the same closure and frozen request
once at original gaps. Record the complete-check failure; do not try subsets,
other sources or helper networks. Loop-entry selection uses this same qualifier
and retains the final refusal diagnostics for its existing required-repair rule.
Raw recurring preparation currently uses unused keys; explicit `restoreReturns`
has no proposed republication and retains original placement.

Position indexes are local to one packet preparation and shared by every owner
query. Each touched original word is indexed once; key-use populations and
occurrence pairs remain charged explicitly. Full packet analysis and ordinary
selected causal replay remain separate costs. The fallback adds at most one
extra preparation/check for a failed late realization. This does not resolve the
open whole-graph replay cost or establish a whole-pass complexity bound.

The normal-constructor two-branch witness strictly removes complete finite
issue/completion relations versus the legal original-gap protocol. It keeps the
same communication population. Native tests retain imported effects/control and
restrict only the two relevant event pools to one key to force reuse; equivalent
views and unrelated arithmetic/control retain activation. Native reconstruction
also runs with the real target pools. This distinction prevents a scarce-profile
mechanism test from being presented as a device-performance result. Access-scoped
native ACC and descriptor contracts are unchanged, so their existing suites apply.

## Step Y invariant — preserve a frozen source milestone before key repair

Missing construction fact: a key chosen from the word-tail state may require an
incoming receipt which the selected physical source milestone does not require.
A different available key may realize that earlier gap. Choosing the key before
qualifying the gap can hide this independent progress.

Split the existing source-gap query into one immutable exact-position/occurrence
and selected-state coverage certificate, followed by key-specific ownership,
neighbor and source-time checks. Retain motivating and supporting requirements;
all actual occurrence prefixes must qualify. The ephemeral certificate is bound
to the selected ledger revision, source/observer and deadline. Scan physical keys
in stable order at that same prescribed gap, then retain existing checked repair
if no complete helper-free binding exists. Do not replay the word per key.

Existing virgin-key admission remains an explicitly sufficient neighboring-use
certificate for early positions, including dormant owners. This increment does
not generalize key reuse at arbitrary early gaps. Loop-entry candidates similarly
try complete bindings without restored helpers before their prescribed repair.
One frozen request/shape is the scope; F4 provider ranking and cross-request
recipe classes are unchanged. A source gap lacking the winner's extra coverage
must remain unavailable regardless of spare keys. Actual packets and subsequent
independent event/causal validation remain authoritative.

The source certificate stores actual prefix states only for this binding decision;
`FrontierState` copies share immutable state storage. Physical-key scans neither
reexecute command words nor repeat physical-history/freshness queries. Ledger
revision, current deadline and direction are checked on every key query. Refusing
an occupied/dormant key leaves the proved source facts available for another key.
The old key-first production queries are removed, with only tiny test adapters
composing the new interfaces.

The prescribed earlier gap remains after the last source-side non-acquisition
command, before intervening incoming waits. It does not cross a source publication
or barrier. Every matched occurrence is acyclic/straight; unknown correspondence
or missing motivating/supporting coverage retains ordinary checked placement.
Keys are helper-free only when unreserved, without dormant event uses or deferred
forward ownership. Early binding additionally requires no historical selected or
dormant use: a sufficient restriction awaiting a general neighboring-use proof.
No syntax, dtype, storage-cell uniformity or benefit gate is added.

Loop-entry probes freeze publication and acquisition positions. A first stable-key
pass admits only complete helper-free packets. A second pass retains the existing
restoration/required-acknowledgment policy. Actual failed-forward diagnostics are
stored as repair eligibility for the unchanged ledger, avoiding a second identical
forward probe. The memo grants no credit and retains no candidate completion
state. Full checking of each distinct prescribed repair is unchanged. This may
probe more distinct hardware-key alternatives; it is not an asymptotic speedup.

Normal-constructor tests compare all finite issue/completion relations against
the legal reused-key tail reference with one/two/four keys. A bypassed producer
followed by a repeated reader child checks higher helper-free key selection and
six nonempty execution paths. Native tests use the real target pools across a
storage view and unrelated control/arithmetic. Older supporting-coverage, stale
revision, shared-occurrence, barrier and ownership negatives remain active.

## Step Z invariant — one complete ordinary acknowledgment packet

Missing construction fact: a prospective consumption acknowledgment and the
forward transfer it enables are one proposed edit. The single-consumption path
currently commits/replays the reverse half first; dormant and joined paths already
qualify complete packets. A publication-support query cannot assess the full edit
while the first half is already selected.

Replace the split path with the existing common exact packet interface. Retain
the same forward/reverse key order, old consumption gap, selected source bound,
and consumer deadline. Existing source-time/neighbor checks remain cheap filters;
the complete packet and actual selected replay remain authoritative. Failure
before commitment leaves the ledger, reservations and credit unchanged. Do not
introduce source/shape search or select another provider. This is a construction
interface consolidation, not a claim of order improvement.

The normal witness Q:write U; P:write X; Q:read X; P:write Y; P:write Z;
R:read Z; Q:read Y with one P/Q key imports Q's U completion through the new
acknowledgment at Y's source. This remains an explicit unresolved resource/source
tradeoff: moving the source past R's publication can instead import Z into Y's
consumer. Persistent support must classify that change; atomicity alone does not
prove its preservation. The next increment must retain publication identities,
rooted contracts and dependencies through later ledger edits.

Closed ordinary repairs privately evaluate their reverse prefix to obtain the
continuation state the old selected intermediate update used. Exact current-word
offsets include those staged endpoints; the new return references the combined
packet's forward WAIT. This existing sufficient prefix-admission rule is not a
general feasibility result for mutually supporting six-endpoint cycles. The
private replay currently starts at the original entry and is counted separately;
recover unchanged components through the existing dependency certificate before
skipping that work. A failed complete deferral packet is not checked again.

Portable fixed-ledger tests exercise terminal and future-payload closed requests
through the public constructor, plus private six-endpoint ownership and rejection.
They do not claim native authored-event support. Native ordinary four-endpoint
activation uses an explicit one-key profile while retaining imported effects;
real-target reconstruction is checked separately. Existing dormant/joined closed
policy differences remain unchanged. Z consolidates the ordinary split mutation,
not every remaining closed-protocol selection rule.

## AA: admission-rooted publication source signatures (foundation)

Missing fact: an earlier selected SET can acquire a new prerequisite through an
inserted WAIT or through a changed SET feeding an unchanged WAIT. Stable gap and
packet atomicity do not certify preserved source independence.

`PublicationSupport` is the shared ordering-evidence service. It builds interned
symbolic issue gates, completion aggregates, original choices and exact selected
SET/WAIT identities. Payload completion retains its issue prerequisites. Each
selected publication occurrence keeps its immutable admission root. Complete
owned-packet preparation, including dormant restoration, obtains an affected
source-preservation result. Commitment reuses the exact prepared delta. Direct
fences, helper erasure and finalization refresh the same service before changes
are forgotten. Ordinary and recurring clients consume the common path.

This is a sufficient equality certificate only. Different expressions, unknown
recurrence and unsupported continuation yield Unknown, not a proven increase in
payload order. Unaffected unknown contracts remain unknown; a packet's Boolean
speaks only about affected existing sources. It does not certify own coverage,
newly gated payloads, full participation, event legality or a normal recipe class.
The current constructor records the outcome but retains its checked policy.
Persistent typed requirement/support records and common selection remain next.

Gates: acyclic original dependencies and matching selected event identities are
sufficient; native ACC never supplies operation completion. Original choice nodes
retain alternatives structurally, not as acquired conjunctions. A dirty cyclic
interface cannot become proved merely because Unknown compares equal to Unknown.
No dtype/shape/cell-count gate is introduced. Equivalent physical subdivision does
not affect signatures because they depend on original payload occurrences.

Cost: hardware-width interfaces are recomputed for affected sites until unchanged;
unchanged words/state remain shared. No payload-order closure, per-publication
walk or dynamic unrolling occurs. An early edit can still reach the whole suffix,
so this is not the completed local normal-probe contract. Interned expressions
from rejected probes are retained and charged to all distinct candidate work.
Identical repeated probes reuse nodes; committing a prepared delta does not rerun
it. The future selector must not blindly multiply suffix propagation by all
requests/classes. Counters report sites, commands, comparisons and new nodes.


### AB dependency: immutable finite recurring interface

Missing fact: `qualifiedCycle` and cell/direction membership do not prove the two
consumption-to-republication chains, or coverage of older outside accesses.
The shared original-control/physical-use service supplies a conditional two-role
certificate; recurring preparation consumes it before any selected update.

One fixed pair of logical readiness/release roles is interpreted over the shared
original CFG. A sparse per-cell operation index retains every original incidence
of each represented cell, including outside accesses and partial writes. Only
those operations enter a compact causal model; other payload is identity and
removes possible support. Original choice, entry, bypass, backedge and exit edges
remain unchanged. There are exactly two logical keys, independent of hardware
capacity; arbitrary support closures never form a dense virtual-key relation.
Opaque endpoint identities preserve each original (site, word offset), using one
synthetic legal boundary in the compact model. They do not change participation.
After convergence export only proved access-class coverage at relevant consumers
and exact role-word obligations. No temporary projected state becomes live credit.

Embedding requires unused distinct new keys and the exact relative role order in
the complete staged words. Every live use of an active shared-role key is checked,
including outside the listed words. Original causal paths survive these insertions;
added commands can provide additional credit but cannot justify omitted original
histories. The producer-repair guard requires every affected current residual to
be proved by the finite support closure. Unknown cases retain the explicitly
transitional full candidate checker. This does not yet replace its family-only
support discovery gate, classify normal placement, or unify selection.

The winner alone is committed and authoritatively replayed. A failed mandatory
update fails that attempt; the existing visible whole-attempt migration retry is
separate and is not locally certified success. Actual residual must be a subset
of the certified remainder after all maintenance. A priming capacity token alone
never proves completion of an older writer. Publication preservation remains AA's
independent certificate; cyclic Unknown is not converted into class 0.

Cost: one shared O(A) cell/access-incidence index; each immutable family currently scans
the original CFG to a finite fixed point, O((N+E+A_f) h_f), retaining graph-sized
temporary snapshots. This is not a contracted region executor. Histories have
only the represented physical classes, not sets of original producer identities.
The two-key causal matrix is fixed-size, but each key's static publisher-binding
alternatives can grow with control. The bound also includes that B_f population
and joined-state work; two keys do not make complete state hardware-constant.
The cache avoids repeated ledger-version solves; it does not remove the F(N+E)
term. Qualification records sparse role bindings and checks their real uses.
These sufficient restrictions (two roles, unused new keys, exact embedding,
complete producer support) remain explicit replacement obligations for common
binding/support; they are not hardware or datatype admission requirements.


### AC implementation contract — common normal realization selection

Missing decision: recurring eligibility still receives priority from a side-effecting
prelude. The next common record contains the frozen request, complete packet,
completion/consumption/F8 support references, guaranteed due coverage and placement
certificate. Pure ordinary source discovery and local-only recurring preparation
feed one normal-class competition before explicit transitional repairs. The latter
must not run candidate whole-program checks while enumerating normal requests.

Each recurring source/target retains its original physical milestone from the
existing qualifier. A normal source is at the first qualified legal boundary after
that use, before unrelated receipts; a normal target is the dependent physical use.
The complete support packet must preserve those gaps, including cross-word support.
Prime/drain endpoints lacking a physical milestone remain Unknown for this initial
normal proof; their event legality is independent and retained by AB. This is a
named sufficient placement limit, not rejection of the original physical facts.

Existing publications use AA equality or a scoped absence certificate over every
original occurrence, continuation and backedge, including Unknown contracts.
Absence says nothing about a NEW source. A reverse reachability summary of selected
publication sites is shared once per ledger version and charged separately; normal
queries do not solve alternative selected programs. Additional return/support
receipts cannot silently broaden the chosen source or change the checked shape.

Select known-readiness/reuse candidates within the first certified realization
class, then one stable strict-coverage dominance scan independent of discovery
order and uniform witness subdivision. Eligibility uses the complete actual due
residual, including only analysis-indexed same-engine returns. Apply actual credit
and all maintenance, then require strict residual decrease by inclusion. Unknown
normal cases retain explicitly transitional repair handling; this first migration
is not completion of classes 1–3, common key reuse or the complete refactor.

AC implementation evidence and boundaries:

- The complete due universe is frozen after original-graph baseline evaluation.
  Coalesced witnesses have identical complete original site/read/write/definite/
  native-class incidence signatures, cell exclusion/storage/domain, source role,
  consumer role and origin set. No footprint, target exception or participation
  is inferred from a shared producer alone. Partial physical coverage cannot
  count as discharge of the normalized obligation. Own coverage must be nonempty
  and included in guaranteed coverage. Known preference also uses complete
  normalized coverage, rather than one constituent access.
- Normal ordinary and recurring records compete at the same frozen version.
  Semantic source/target descriptors plus the COMPLETE support shape determine
  stable order; carried sources use their role/owner description, not a claimed
  dynamic position. A stable strict-containment scan follows class/Known priority.
  Sorting is O(G log G), separately from its O(G b) dominance scan and probe work.
  Exact identical complete role packets collect independently certified physical
  witnesses BEFORE probing. Sharing one role alone does not combine packets.
- Normal packets introduce no consumption-only helpers. Existing-publication
  support includes the complete prepared packet; a new source separately needs
  its physical milestone. Adopted publications must retain their leading position
  before incoming receipts. Unknown priming, draining, and cyclic preservation
  interfaces remain explicit transitional cases. Class-0 preference is not a
  global ordering-optimality or runtime claim.
- A key's active ordinary Completion endpoints can be adopted only as an exact
  subset of the SAME complete recurring role. The shared checker validates all
  existing and new occurrences, multiplicities, word order and consumption paths.
  Dormant/closed/deferred owners are excluded. Alternatively, ordinary exchanges
  may share a key when ALL old/new endpoint occurrences are control-disjoint in
  both directions, including backedges. Neither rule infers rearming from empty
  occupancy. These are bounded common-binding bridges, not general key borrowing.
- Normal probes never call the candidate analyzer or contextual evaluator. Only
  the winner gets actual causal propagation. Complete physical residual inclusion
  and strict normalized decrease are checked after maintenance. Existing helper
  discharge only removes certified endpoints; it cannot silently add a stronger
  packet. Failure of mandatory selected evaluation remains failure.
- Support closures share records for mutually reachable roots; exact complete
  role shapes are grouped before binding. Immutable physical milestone checks
  are indexed/cached per role. Deadline/cell indexes replace root-by-requirement
  products. Publication continuation summaries are shared by ledger version;
  alternative-key reachability is shared by gap qualification. Their visits,
  original incidence preparation, candidate populations and selected choices are
  separately observable. AA affected-suffix propagation and AB per-family CFG
  fixed points remain charged costs, not claimed near-linear whole-pass bounds.

The normal-constructor witnesses include one through four banks, ordinary versus
recurring competition, later adoption, and 1/8/32 uniform physical subdivisions.
Pure policy tests distinguish class preference from greater coverage and reject
support-only progress. Source/key negatives cover tail-only credit, missing own
milestones, duplicate/outside uses, receipt-before-adopted-publication, dormant
ownership and original-control re-entry. Two changed straight-line plans compare
complete nonempty payload-order sets with their former forwarding/separate-source
protocols. Joint WAR/WAW return selection is covered; a same-engine-ONLY normal
constructor witness remains an explicit acceptance item for subsequent typed
support/return discovery. No internal probe test substitutes for that evidence.

Remaining AC-to-next obligations: classes 1-3 must become certified records;
family-only producer support must become typed ordinary support; cyclic publication
support needs an affected-interface certificate; arbitrary historical-key binding
is not established by the two bounded bridges above. No serializer, kernel
recognizer, helper-deletion sweep, or alternative-ledger search was added.


## AD design — fixed-boundary source-word acknowledgment

Missing capability: a consumed key whose publication engine lacks consumption
knowledge cannot yet compete as a complete fixed-boundary repair. The existing
joined acknowledgment already has its reverse pair at the new source word.
Extend the common ordinary realization probe with that prescribed prefix, using
the same exact gap/occurrence/freshness query and common packet commitment.

The local invariant is reverse SET -> reverse WAIT -> forward SET at the exact
source gap, followed by its matched forward WAIT at the unchanged deadline.
Interpret the source prefix on every actual source snapshot. Retain only the
original source's guaranteed physical coverage; helper-added completion is not
assumed at the target. Neither key has an unaccounted later/same-gap use. Reverse
keys are virgin; forward keys have complete helper-free ownership and no dormant
population. The source/key-neighbor query is shared with class0, so already known
consumption does not force a helper. Unknown occurrence or neighboring use retains
the existing checked path. No candidate whole-program solve is introduced.

This is a class1 policy/precision increment. Existing selected publications must
still satisfy AA preservation; unsupported source contamination is not waived as
repair. Noncontiguous acknowledgments, dormant restoration, class2/class3 and
recurring repair remain transitional. The fixed forward/reverse key scan has no
Cartesian backtracking. Key-independent future reach is shared across candidates;
source primitive execution and key-use visits are charged. Normal-constructor
selection, cold checks, missing support/branch/unavailable-direction negatives,
stale gaps, later key uses, and publication contamination are required evidence.

The source-local repair applies only to joined consumption without one original
WAIT identity at every source occurrence. A unique WAIT already has an earlier
prescribed helper publication. Moving that endpoint to the new forward source
would be another policy, and the first AD campaign reproduced added ordering from
that mistake. Preserve the unique-WAIT recipe until its unchanged-interval proof
is available. This gate protects a source boundary; it is not an event-occupancy
or kernel-shape restriction. Helper-free direct reuse remains available in both
cases when the shared neighbor and actual-rearming certificates pass.

AD final evidence qualifies the claim above: source and host checks pass, but
`hc_head_reduce` adds 122 and removes 20 finite payload-order relations in each
of two bindings. The unique-WAIT exclusion does not recover this case; the final
candidate is byte-identical to the first attempted candidate there. A later
input refill is gated by preceding vector work. Determine the first changed
complete realization and whether source-local joined support or class0 historical
key choice causes that edge. Keep this quality loss open while general support
and binding migrate; do not add an example-specific veto or describe AD as
order-preserving.

## AE design — typed ordinary producer support

A recurring x packet may remove a producer fence that also protected an
unrelated same-engine z access. The current family-only link treats a missing
z recipe as a refusal. Retain z as a typed obligation at its actual consumer
occurrence and deadline, with its original physical access class and producer
seed interval. A same-engine barrier at the end of each designated producer
seed word is one prescribed ordinary discharge, when it precedes the seed
payload and every participating z deadline after all outstanding z accesses.
It may add old-z -> new-x order and is therefore a repair, not class0.

The proof asks which original accesses can reach each qualified deadline
without crossing its designated pre-payload producer fence. A backward query
collects all source-pipe access classes in one traversal, includes the payload
at the fence site because it executes after the fence, and retains prior visits
through backedges. Absence of the required class proves its older original
accesses have completed on every path. The original-use result is cached by
producer pipe, exact fence interface and deadline; packet embedding is checked
separately. No definite-write shortcut is used. All missing typed obligations in the complete selected
closure must be proved together; repeated physical witnesses do not add fences.
The barrier and recurring endpoints form one exact ordered packet, with
publication preservation and shared-word occurrence agreement. Selected causal
replay remains authoritative and persistently checks the affected producer
scope after later edits. Unknown paths or unsupported cross-engine residuals
retain the existing refusal. The former broad producer-scope guard remains until
all of its obligations have scoped replacements.


## AF design — selected-return source corridor

A previously selected reverse receipt can rearm an ordinary forward key after
its original source milestone. The ordinary request keeps its original source,
complete due coverage and consumer deadline. Class0 and class1 realizations are
probed first. Only if neither is available does the class2 query inspect actual
selected reverse receipts between that milestone and deadline. No anticipated
return and no unselected candidate ledger supplies credit.

The query follows current selected-word order, not endpoint creation order.
For each reached completion receipt it retains a hardware-sized summary of
forward keys publishable before any occurrence. The joined causal state after
that receipt must prove a previously unproved key publishable. Closed, recurring,
dormant and deferred-owned keys are excluded before fixing a corridor. An earlier
reverse receipt with no such rearming effect is irrelevant and does not hide a
later one. At the first relevant receipt, the constructor computes one exact
source-gap qualification and checks eligible physical keys in stable order,
reusing those facts. The same ordinary coverage, occurrence, key-neighbor,
complete-packet and persistent-publication checks remain mandatory. Selection
commits exactly the privately checked packet, then authoritative propagation
must reduce the complete residual.

The decision retains the original source milestone, exact selected gap neighbors,
and actual supporting receipt. It records prefix enlargement even when the cut
number is unchanged: a publication after a WAIT can import completion that a
publication before it did not. The early-publication counter therefore counts
only a genuinely earlier gap. This is a class2 policy extension, not an
order-preservation certificate. A selected return can broaden the new
publication, and complete payload-order comparisons remain required for quality
claims.

The current recipe covers matched straight ordinary corridors and helper-free
bindings. At its first applicable return, unknown source/gap or packet support
conservatively leaves the ordinary repair path; it does not search later source
positions or alternative ledgers. The key-transition summary is an eligibility
filter, not acquired credit or proof of concrete necessity. Native recurring
return transport, non-straight correspondence, persistent cross-cell support,
class3 placement and general historical-key binding remain open. The AD
`hc_head_reduce` added-order counterexample is not recovered by AF.

The normal-constructor fixture selects the return-supported class2 packet and
checks its exported source/receipt record and cold causal validity. A second
fixture puts an unrelated reverse receipt first; it cannot masquerade as
consumption support or hide the actual return. Erasing the return refuses the
corridor. Existing early-publication hardening confirms that a late class2
placement is not reported as an early source. Further conditional-credit,
intervening-conflict and shared-word mutations remain acceptance work for the
broader class2/general-binding interface. Scanned receipt and word populations
are counted; one request can still inspect an interval of selected receipts.
