# Selected-plan construction: draft F1–F8 implementation

Specification: the synchronization draft v0.22, policies F1–F8.
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
