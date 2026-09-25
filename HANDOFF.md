# FrontierSynch status

## Current integration update — 2026-09-25

Downloads steps 7–12 have been merged sequentially and independently accepted
against draft v0.44 (`4905f8a`), including integration corrections. Step 13 source
preparation and scoped consequences are implemented locally and independently
accepted within the documented scope after final integrated validation. See [the integration record](docs/designs/frontier-synch-steps7-13-review.md) for the current evidence,
precise supported fragments and remaining qualifications. Older baseline tables
and submitted patch reports below remain historical; they do not override this
update. Step 14 and aggregate draft parity have not been accepted. The later
v0.45 manuscript is not the contract for these patches.


## Step 7 candidate: conflict-specific D1

The step-7 patch adds guarded fixed-visit source queries over the shared
old-state provenance, separate endpoint qualifications and an explicit incoming
case. `ProgramAnalysis::fixedSourcesFor` consumes stable obligation families;
`interpretAt` retains the complete conditional result. The portable oracle and
ASan/UBSan checks pass. Native tests are added but have not been executed in the
patch environment; independent acceptance is still pending. See the separate
[step-7 record](docs/designs/frontier-synch-step7-review.md) and the new delta in
the parity checklist. The following integration record predates this candidate.
Step 10 review candidate on `4122dd1531fbdb2859bd7b27772dba56930be393`:
[exact first/last boundaries and interval participation](docs/designs/frontier-synch-step10.md).
Standalone production-core checks passed; native integration validation and
independent acceptance remain pending. This does not mark Step 10 accepted or
supersede the remaining Phase A gates.
## Step 12 delivery on 4122dd1531fb — awaiting native and independent review

The request/descriptor layer now forms the three fixed slots before construction,
keeps original obligation identities, composes shared alternatives without a
Cartesian product, and exposes a frozen `ProgramAnalysis::requests()` view.
See [scope, evidence and remaining gates](docs/designs/frontier-synch-step12-review.md).
The portable slot core is tested. Native integration tests are supplied but have
not been run in the delivery environment. Steps 7–11 still supply missing native
occurrence/boundary answers; step 13 supplies source/support preparation closure.
This is not independent acceptance or complete Phase A parity.

Branch: `codex/handoff-foundation`. FrontierSynch is selected with
`pto-insert-sync{algorithm=frontier-synch}`; `algorithm=existing` remains
available.

## Step 8 patch — D2 local periodic correspondence

The [step-8 record](docs/designs/frontier-synch-step8-review.md) describes the
physical-permutation proof, distinct producer/reader roles, both directional
domains and explicit initial/final/bypass cases. It also separates executed
standalone checks from native tests awaiting a build and independent review.
No step-8 independent acceptance, D4 re-entry transport, or complete Phase A
parity is claimed by this patch. The original obligation universe is unchanged.

## Current integration update — 2026-09-24

Downloads patches for steps 2–6 have been integrated on the baseline below.
All five increments passed scoped independent review and integrated validation. The [integrated review](docs/designs/frontier-synch-steps2-6-review.md)
records merge corrections, evidence and the remaining gates. Changes are
uncommitted. No full Phase A parity or working constructor is claimed.

The current implementation includes original cuts and snapshot keys, qualified
shared full-write semantics, scoped factored histories, original-value/endpoint
qualification, and guarded obligation families with lazy queries. Steps 7–14
remain. The numbered inventory and detailed source-status prose below describe
the pinned pre-integration baseline; use the integration record and parity
ledger for subsequent changes.

## Pinned baseline before steps 2–6

Status: Phase A has an original-program query boundary and a committed acyclic
factored provenance core, but does not yet meet every specified Phase A
procedure/interface in draft v0.44. Its operative analysis contract is inherited
from v0.43. Conditional provenance integration, full-cell overwrite
qualification, complete D1–D4 service, directional covering boundaries, the
interval rule, and full look-ahead preparation remain partial or absent.
Construction still reports an explicit failure. The seven original foundation
commits introduced FrontierSynch under its current name; later increments do
not turn that foundation into complete Phase A parity.

Base: upstream `master`, `f5eff3ee249697f6157088f649c6434fcc9d7c5b`.
Donor: `371fdb344d2783b92d6c39424c507b2ce082e08c`.
Implementation inspected for the current checklist:
`0a38c8e9f4bd4852177c1dd6f6e0d0ee26b6ca16` on `codex/handoff-foundation`.
Factored-core commit: `9bba6552055d5386ff7242c58031b0412885a0d6`.
Earlier query-boundary review: `3bbc58a67` (historical).
Draft inspected: supplied revision 0.44 PDF, 83 pages. Manuscript pins retained
from this handoff are `4905f8a` (v0.44) and `a13650a` (v0.43); their operative
Phase A equivalence is recorded in the implementation plan, not independently
rechecked between manuscript commits by this documentation update.

The [v0.44 parity checklist](docs/designs/frontier-synch-v0.44-parity.md) replaces
the earlier four broad gates. It records each specified behavior's assumptions,
implementation entry and actual consumer, status, test evidence, real-input
blocker and later gate. The previous inventory's `6220bb6c` baseline and claim
that there is no provenance/demand DAG are obsolete. Historical v0.42/source-port
notes below are retained as history, not the current completion specification.

## Committed factored core: present, not fully integrated

`FactoredProvenance.h` implements shared original writer/reader and RAW/WAR/WAW
demand expressions over an acyclic cell projection. Reads query their incoming
writers; writes and read-modify-writes generate old-state requirements before
updating provenance. Definite writes replace the state; possible writes retain
older origins/readers. Backward next-use roots and translated-effect incidence
lists are retained. `OriginalLifetimes::factored` owns the lazy per-cell result,
and `ProgramAnalysis::interpretAt` exposes it as `factoredUse`.

The core's prior reviewer acceptance is reported by the implementation plan and
is retained at that scope. The committed `--factored-self-test` checks a guarded
RMW/optional-reader fixture, a possible write, a backward next-writer query, and
a 64-optional-reader node-count bound. Its effects are synthetic and its small
expected demand sets are handwritten; it is not a native full-overwrite test,
an incoming-reader test, or an independent general concrete scanner. Those test
definitions were inspected, not rerun for this documentation-only update.

Current limits must not be hidden by that acceptance. `requirementsAt` still
returns marginal source-target records; attaching the factored result does not
make guarded demands their public obligation model. The core rejects any
For/While in the supplied whole body, starts with one incoming-writer sentinel
and no incoming-reader interface, and keys choices by original if-owner rather
than a common defining-value/occurrence identity. Every imported write remains
`definiteWrite=false`. Steps 2–6 and the supported repeated-summary work extend
this core; they should not reimplement it as though it were absent.

## Current draft parity and implementation sequence

The parity target is every *specified* Phase A procedure and interface in the
original-program contract, Section 3, relevant Section 4.1 interfaces and
Appendix I, including inherited D1–D4. Existing storage partitioning, All,
MayAfter, typed prerequisites and source cuts are part of that target. Missing
specified behavior is an implementation gap, even when an API conservatively
returns Unknown. General symbolic problems explicitly left open by the draft
remain separate; so do selected-state and packet construction responsibilities.

| Step | Bounded increment |
| --- | --- |
| 1 | Establish the complete parity checklist and correct the committed baseline/evidence record. |
| 2 | Define common occurrence, legal cut, owner, continuation and complete query/cache records. |
| 3 | Supply qualified full-cell overwrite information through shared effects and storage geometry. |
| 4 | Complete and integrate factored provenance, incoming interfaces and supported projections. |
| 5 | Implement original-value, arithmetic and endpoint-availability qualification. |
| 6 | Make stable guarded physical/typed/incoming obligations the construction-facing model. |
| 7 | Complete conflict-specific D1 fixed visits and guarded alternative/incoming sources. |
| 8 | Complete D2 physical-permutation correspondence, distinct roles and both boundary domains. |
| 9 | Implement D4 composition and supported re-entry transport of qualified relations. |
| 10 | Complete common exact first/last queries, D3 and the restricted interval-participation rule. |
| 11 | Implement independent directional covering boundaries over original cut intervals. |
| 12 | Build request groups and the three specified prederived descriptor slots. |
| 13 | Complete subscriptions, finite support preparation and scoped two-link consequences. |
| 14 | Perform the integrated public-interface semantic/corpus, preservation and cost review. |

Step 1 passed independent coverage review on 2026-09-24 after clarifying that
kernel assertions require a specified rule's premises; unsupported matching
must remain explicit Unknown. This accepts the inventory, not implementation
parity. For each later step: implement the bounded change,
run focused semantic checks, obtain independent review against that gate, fix
findings and record the accepted commit separately. A partial acceptance names
its remaining work and later gate. The final reviewer rechecks the integrated
contract, not just the collection of prior component approvals.

The actual `frontiersynch::run` consumer currently counts indexed requirements,
interprets one sampled request and stops at the construction-not-implemented
diagnostic. The corpus runner interprets every indexed marginal requirement,
checks direct source subscriptions and unchanged IR, but does not assert the
intended exact kernel relationships or evaluate the factored demand contents.
Public query availability is not evidence that the constructor consumes it.

Historical corpus/mode runs reported that `existing` emitted C++ for all five
development kernels and `frontier-synch` reached its expected construction
failure. The Phase A runner reported all 2,242 kernel occurrence interpretations
as Unknown. These are retained baseline observations, not new runs or acceptance
targets. The separate TAXPY fixture expects three non-Unknown fixed visits, so
the implementation is not universally Unknown on straight-line input. No result
here establishes selected-plan correctness, device behavior or complete parity.

## Historical draft 0.42 comparison

This comparison predates the factored core and the v0.43 refinement. In
particular, its whole-query-record DAG wording below is historical, not an
instruction to enumerate joint reader sets or replace the current shared
Both/Choose representation. Current procedure status is in the v0.44 checklist.

[The source comparison and acceptance cases](docs/designs/frontier-synch-v0.42-gaps.md)
record the earlier gap inventory. The draft retains the original/selected-state
boundary and adds restricted derivations for these capabilities:

- Joint guarded records must connect producer, reader interval, next reuse,
  frontiers, and continuation. Current histories are marginal may sets;
  `interpretAt` combines their handles without proving their correlation.
- Context formation must preserve source/deadline distinctions, including
  unrelated engine work that changes a prefix. A shared predicate/frontier DAG
  does not yet provide the draft's shared decision DAG of whole query records.
- The unit-step interval rule must derive nonempty, first, and last participating
  visits symbolically. Current counted D3 accepts invariant participation;
  induction-dependent participation remains unresolved.
- A separate sufficient-boundary descriptor must retain region participation,
  enclosed work, and the original deadline. Current exact-frontier results and
  operation-cut candidates do not represent that descriptor.

Existing prerequisites remain: full-write coverage, guarded D1 alternatives,
qualified D2 entry/successor cases and D4 re-entry, continuation-aware owners,
and executable endpoint predicates. The draft's general mixed-write/while
qualification, combined adequacy/cost result, and complete packet integration
also remain research obligations; changing the baseline does not prove them.

## Earlier implemented checkpoint: coherent Phase A query boundary

The follow-up review of the five Phase A commits found five remaining items.
The current checkpoint closes the owner escape in `firstMayUse`/`lastMayUse`, retains
every source/target translated-effect witness when coalescing a physical demand,
and exposes guarded original endpoint candidates with their executable-cut
qualification. A multi-phase original instruction offers only its outer cuts
until lowering supplies an internal insertion contract; its first legal later
source is subscribed separately from the analytically sufficient phase cut.
Support intervals now distinguish an uninterrupted physical interval from a full-content generation.
These records grant no selected completion.
Both algorithms use the instruction effects supplied by the shared `SyncInput`.
The additional FrontierSynch import audit has been removed; there is no separate
instruction whitelist. Definite write coverage remains a separate proof obligation.
Phase A now also indexes original SSA producers of branch conditions, counted
loop bounds, while conditions, and translated payload addresses as typed
value-availability prerequisites. Pure scalar dependency chains are followed to translated phases, while incoming
values and unmodeled effectful producers remain explicit unresolved cases.
These requests do not claim native availability or selected completion.

`ProgramAnalysis` owns the original structure and its provenance, occurrence and
guarded-reader services. Requirements and source subscriptions exist before
traversal. `decodeAt` gives a cheap indexed request; `interpretAt` brings its
original-only occurrence, may-origin, reader-boundary and support requests into
one lazy construction-facing record. The record marks unsupported D1/D2/D4
correspondence Unknown instead of converting bank geometry or a structural
reader expression into an exact use episode. Each bank relation is attached to
its translated effect and the correct RAW/WAR/WAW role. Phase A performs no
event allocation or selected completion reasoning. `all`, an owner/interval-
qualified `mayAfter`, source/direction indexes and source subscriptions remain
queryable. The invariant
conditional-reader case is now handled by the counted D3 rule when its exact
original predicate is available before the loop; iteration-varying participation
remains Unknown.

The shared translated effect identity and direct physical-relation ID are
retained with each access. Physical evidence alone does not establish
source/target occurrence matching.

The revised draft separates original structure, requirement records, qualified
placement structure, and the look-ahead index from selected completion.
`ProgramAnalysis` follows that boundary. For one mandatory same-role use of an
exact disjoint-bank permutation, its occurrence interpretation now records the
periodic predecessor distance **within the counted owner**. Initial uses and
transport across child re-entry remain unresolved. Admission additionally
requires every source/target hazard-role incidence on the cell to follow that
selector; an overlapping fixed access cannot inherit its period. Source subscriptions carry
the stable index of their exact deadline requirement, and duplicate incidences
of the same physical cell/role no longer duplicate that demand.

The constructor-facing interpretation holds shared lifecycle, alternative-
origin and boundary may-set handles instead of duplicating these populations
for every relationship. A support inventory is expanded only by
`qualifySupport`; for RAW/WAR its candidate interval is scoped to a producing
write and next conflicting write. A unique-write candidate is a sufficient
restriction, not proof that a less regular lifetime lacks a valid protocol.

[Contract, draft limits and LLVM/MLIR replacement audit](docs/designs/frontier-synch-program-analysis.md).

## Historical Stage 4 baseline: original lifetimes and requirement subscriptions

`frontiersynch::run` imports original structure from the shared `SyncInput`, queries
the Stage 3 reader/occurrence facts, and now indexes original storage demands:

- Donor-style previous/next writer and reader provenance flows retain partial
  histories and the possibility of uninitialized storage.
- RAW/WAR/WAW requirements have stable original source-after and target-before
  positions. Source subscriptions are indexed before construction traversal.
- Original first/last-use may frontiers and affected producer-to-reuse intervals
  retain readers, reloads, bypass/re-entry paths, and other-cell requirements.
- Unknown full-write coverage, occurrence matching and executable endpoint
  guards remain unresolved; no analyzed demand grants selected completion.

The analysis is read-only. Records borrow original IR and instruction pointers;
those owners must outlive the result. Unsupported control reports failure and
leaves the caller's output unchanged. The `frontier-synch` mode still reports that
construction is not implemented after completing this analysis.

`algorithm=existing` retains the upstream InsertSync construction path and the
same shared instruction input. The new mode currently changes analysis only.

[Port sequence, donor crosswalk, and gate inventory](docs/designs/frontier-synch-first-pass-port.md).

## Scope and remaining limits

The tree is original control, not a selected occurrence refinement. Marginal
histories and support intervals remain may facts. The factored core supplements
them on its acyclic formation fragment; it does not supply general occurrence,
endpoint or protocol certificates. Step 3 now qualifies supported full writes through shared semantics; unknown
coverage retains conservative write histories. Source subscriptions
identify original positions, not selected source-time snapshots or acquired
completion. The shared translator remains responsible for effect completeness.

The next implementation increment after the accepted step-6 increment is **step 7
(D1)**. Keep the shared instruction interfaces and the accepted factored core;
do not add an instruction admission whitelist or kernel-specific recognizers.

Specific supported gaps include D1's rejection of intervening reads, D2's
same-operation/single-role restriction and missing successor domains, D4's lack
of qualified transport, the coupled first/last availability result, and missing
covering/interval/descriptor/support-closure procedures. The checklist assigns
each to a later gate rather than describing it as an open research problem.
The availability service has passed its step-5 scope; the consequence service
remains assigned to step 13. Broader symbolic qualification beyond the draft's supported
rules remains distinct. Actual completion, matching, event reuse and ordered
selected-word placement are Phase B responsibilities.

## Validation and evidence provenance

This step-1 update is a source/documentation audit. The implementation entry
points and committed test assertions were inspected, and the delivery includes
whitespace/applicability and checklist-structure checks. No compiler, probe,
corpus, lit, sanitizer or device run was performed for this update. Independent
step-1 coverage review accepted the corrected inventory on 2026-09-24; the
decision and required wording correction are recorded in the parity ledger.

### Historical implementation validation (not rerun for step 1)

The 0.42 comparison is a documentation and source inspection update. No compiler
build, probe, corpus, sanitizer, or device run was performed for it. The evidence
below belongs to the earlier implementation checkpoints and does not validate
the new draft refinements.

Current FrontierSynch importer and translator sources compile in a focused C++17 probe
with the project's warning flags. The probe links existing LLVM 19 and generated
PTO dialect dependencies from the donor build; this is not a fresh full compiler
build. The earlier full CMake configuration remains blocked by the upstream
Python-extension dependency when Python bindings are disabled.

Stage 2 port probes remain at
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage2/`.
Stage 3 probe source, input cases and build logs are retained at
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage3/`. Its build is
serial and links current importer/occurrence code with the shared translator and
existing generated LLVM/PTO dialect dependencies. No old constructor is linked.

The induction and carried-selector examples both retain separate periods 2 and
3 despite unrelated carried state. GM-alias and nested if/for/while inputs pass.
The probe also checks nontransitive overlap witnesses, cyclic root propagation,
unchanged source IR, and one tree occurrence per translated phase. The
partially unknown-address input retains the independent period-3 relation and
an unknown footprint. The multi-block CFG is explicitly refused as expected
(exit 6 in the probe). `results.log` and `source-manifest.txt` pin the outcomes
and analyzed sources. `git diff --check` passes.

The Stage 3 focused probe passes optional sibling readers, a late unavailable
predicate, exact fixed visit, nested child interval versus unknown whole-loop
participation, and independent noninjective/exact bank relations. The FrontierSynch
entry reports analysis counts and still refuses construction. Focused probe
source/build logs are retained at the Stage 3 artifact path.

The Stage 4 serial probe and inputs are at
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage4/`. The current
source build and importer probe pass two optional readers followed by reuse and
a reload between children. It checks all affected readers, RAW/WAR/WAW
subscriptions, retention of older possible writers after an unproved reload,
and the affected producer interval. The existing Stage 3 inputs still pass.
This is focused source-port evidence, not a complete linked compiler/corpus run.

For the current Phase A query increment, the serial focused probe was rebuilt
from clean objects and passes eight inputs, including invariant versus
iteration-varying conditional readers, independent bank selectors, two reader
children, and a reload. It checks source subscriptions, retained bank/effect
identity, lazy decoded requirements, May/NoHit/Unknown continuation outcomes,
one shared exact **structural** final-reader expression for two optional
children, while requirement-level frontier placement remains Unknown without
episode/occurrence proof. It also checks unresolved-negative behavior.
The current D2 extension checks the same-role period-three predecessor domain
and its first-three-use incoming case. A focused mixed-incidence mutation adds
a fixed write to the same cell and confirms the periodic interpretation is
withheld. Source subscriptions are checked against exact requirement indices.
`FrontierSynch.cpp` and the `PTOInsertSync.cpp` dispatch compile with the
project's LLVM 19 flags. The changed-code scanner passed for the Phase A
checkpoint, and `git diff --check` passes for the rename. No normal selected constructor exists yet, so this
does not establish synchronization-plan quality or full supported-input service.

No sanitizer, corpus, device or synchronization-quality campaign was run.
The current checkpoint does not establish complete target effects, full-write
coverage, or constructor service.

Three read-only reviews of this increment checked architecture/generality,
correctness, and avoidable asymptotic work. Their mixed-incidence and repeated
scan findings were corrected; they did not validate a complete Phase A or a
selected synchronization plan. Separate generality acceptance of the complete
first pass remains pending.

The Stage 4 scanner false positives were removed by naming compound
conditions; its changed-code scan reported zero findings. New source files
retain the repository license header.

The FrontierSynch layout was checked with a clean serial rebuild of the
eight-input Phase A probe and focused C++17 syntax checks of the mode dispatch.
All eight inputs passed. The changed-code scanner reports 60 pre-existing
unbraced-control findings in the ported sources;
no control bodies changed during naming. No constructor or device validation was
added for this structural checkpoint.

### Kernel corpus added after the 0.42 comparison

The [development corpus](test/lit/pto/frontier_synch/README.md) now contains
TileLang intrinsic GEMM, PyPTO manual-pipeline GEMM, expert Flash Attention
(separate cube/vector bodies), and pipelined vector add. Five lit cases run the
current Phase A query boundary, interpret every indexed storage requirement,
check source subscriptions and unchanged IR, and verify materialized manual
references. A fresh focused build passes all five; the full compiler build and
device gates remain open. This adds fixtures and a test runner, not algorithm
implementation. The preceding no-corpus statement describes the earlier
foundation checkpoint. See the corpus README for source pins, porting
differences and observed Unknown results. The shared TAXPY effect definition now
includes its destination read, with a focused dependency regression in
`test/lit/pto/frontier_synch_taxpy_effects.pto`. All six focused tests pass after
rebuilding the dialect implementation and regenerating its operation/interface
headers from this checkout.
