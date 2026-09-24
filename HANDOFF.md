# FrontierSynch status

Branch: `codex/handoff-foundation`. FrontierSynch is selected with
`pto-insert-sync{algorithm=frontier-synch}`; `algorithm=existing` remains
available.

Status: Phase A has an original-program query boundary, but is not complete
against the specified first-pass contract in draft v0.43. The current paper
revision is v0.44; its operative Phase A sections are unchanged from v0.43.
Factored conditional provenance, directional covering boundaries, the qualified
interval rule, and complete look-ahead preparation remain unimplemented.
Construction still reports an explicit failure.
The seven foundation commits introduce FrontierSynch under its current name.

Base: upstream `master`, `f5eff3ee249697f6157088f649c6434fcc9d7c5b`.
Donor: `371fdb344d2783b92d6c39424c507b2ce082e08c`.
Implementation reviewed: `3bbc58a67` on `codex/handoff-foundation`.
Draft checked: paper repository `4905f8a`, revision 0.44, on 2026-09-24.
Its Phase A contract is the v0.43 contract introduced at `a13650a`. The
implementation checkpoint below predates both revisions; its historical
v0.42 comparison remains below. The current comparison and four review gates
are in [the v0.44 parity inventory](docs/designs/frontier-synch-v0.44-parity.md).

## Current draft parity and implementation sequence

The parity target is every *specified* Phase A behavior in the current draft,
including retained physical provenance, All/MayAfter, typed prerequisites,
D1–D4, and the v0.43 additions. Open general symbolic matching and selected
construction are recorded as such, not silently claimed implemented.

1. Factored provenance and demands: the current four marginal bit matrices
   give conservative may relationships, but lose shared branch correlations.
   All imported writes still lack a definite full-cell proof. A shared guarded
   expression service and its real-effect qualification are required.
2. Directional boundaries: current exact/may reader frontiers and operation
   cuts are present; independent source and target covering results over
   original cut intervals are absent.
3. Exact occurrence/frontier rules: restricted fixed visit, same-role period,
   and invariant-reader rules exist. The specified guarded alternatives,
   supported D2/D4 transport, and symbolic interval rule are incomplete.
4. Look-ahead preparation: current requirements have deadline/source indexes
   and subscribed operation cuts. Stable obligation/group/descriptor
   separation, closure over all referenced sources, and the qualified two-link
   consequence service are absent.

Each increment needs focused semantic tests, corpus queries, and an independent
review against its full specified contract before the next increment. The
earlier occurrence-relation proposal in the paper repository is exploratory;
it is not a substitute for implementing already specified rules.

The five development kernels were run through the current actual pass modes:
`existing` compiled and emitted C++ for all five; `frontier-synch` reached the
expected construction-not-implemented diagnostic. The Phase A corpus runner
completed all five, but all 2,242 interpreted storage requirements reported
Unknown occurrence correspondence. These runs establish the baseline only;
they do not establish plan correctness, device behavior, or draft parity.

## Historical draft 0.42 comparison

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

## Implemented checkpoint: coherent Phase A query boundary

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

## Stage 4 baseline: original lifetimes and requirement subscriptions

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

The tree is original control, not a selected occurrence refinement. Stage 3's
restricted D1/D2/D4 certificates remain unchanged. Stage 4's marginal histories
and support intervals are complete may facts, not exact generation or protocol
certificates. All imported writes remain non-definite until an effect-coverage
proof is ported; a reload therefore retains older possible writers. Source
subscriptions identify original positions, not ordered command-word gaps or
source snapshots under selected waits. The shared translator remains responsible
for effect completeness.

Recommended next increment: qualify full-cell effects for an explicit initial
instruction fragment, then implement the draft's fixed-use acyclic joint
records with continuation-aware owners and guarded source subscriptions. Use
the conditional-write example in the gap inventory as the first comparison
fixture. Preserve unsupported effects and correlations as unresolved demands.
The symbolic interval and sufficient-boundary rules can then extend that
interface under their own premises. Extend repeated-use D2/D4 qualification
before composing those records across re-entry. This order is an implementation
recommendation, not a new restriction in the paper.

Control-value prerequisites still lack qualified native availability and
dynamic occurrence matching; return/compatibility precision remains open.
The constructor must evaluate actual completion, matching, event reuse, and
ordered command-word placement separately from these original facts.

## Validation

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
