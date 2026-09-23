# OAHS implementation TODO

This list tracks implementation work that is not ready for the committed
selected-plan design. Safety remains mandatory, but the acceptance criteria
also protect useful payload ordering, resource availability, and construction
cost. Do not reduce synchronization counts by enlarging publication prefixes or
advancing acquisition deadlines.

## Active refactor stages — 2026-09-23

Canonical sequence: [semantic corrections](oahs-semantic-corrections.md).
Donors supply mechanisms/counterexamples, not a chronological cherry-pick queue.

- [x] **Stage 0 host closure:** portable/native checks, classified CSA mixed
  ordering, pinned manifest, exact packet materialization/starvation and shared
  merge-sort effects. Peer progress/device claims remain outside this evidence.
- [ ] **Stage 1 (active):** shared physical-use/deadline records and bounded
  occurrence pairing are implemented and host-validated. Complete the remaining
  placement consumers and partial-knowledge inventory; retain copied children.
- [ ] **Stage 2 (active):** shared read episodes span children/reloads/outside
  readers; staged producer-repair crossing guard implemented and reviewed.
  Complete open/exported obligations and access/episode-scoped ACC replacement.
- [ ] **Stage 3 (active):** joint original endpoint demand collection and
  demand-driven role separation implemented; normalize positive-step participation
  and complete first-read/first-write/final-read endpoints with exact
  gaps, original participation and invalidated publication-prefix certificates.
- [ ] **Stage 4:** complete actual-return support, common ordered packets,
  deadline-specific binding/ownership and checked rearming.
- [ ] **Stage 5:** remove replaced paths, restore reference overlap/sharing,
  explain corpus tradeoffs, coupled device validation and generality review.

Implemented foundations: shared instruction/ACC ports, hardening `be14229f2`,
physical-use and child correspondence `53a8458a4`. Their existence does not mean
stages 1–4 are complete. Keep the existing bounded local retry visible; do not
add the old serializing constructor or count retries as mechanism successes.

Cost track: extract measured plan-equivalent reductions with their consumers;
new caches/region execution require evidence and invalidation tests. Two local
resource-intensive workers maximum in aggregate.

## Historical GEMM checkpoint and parked findings

The following source-checkpoint notes are historical. Event counts, old milestone
names and old validation do not define the current implementation or acceptance.

## Locally completed GEMM plan milestone

The working tree carries finite outer-bank identity through the nested MAT
reader region and relates each overwrite to the previous participating use of
the same physical bank. It composes that fact with the existing child
ready/release interfaces without expanding the full nested first/tail product.

Two operands in the same exact bank episode keep separate early readiness but
share one storage-release return. A narrow first-use qualifier recognizes the
original `outer_k == 0 && inner_k == 0` conjunction and splits only the entry
prefix needed to eliminate impossible repeated ACC initialization. It grants no
completion credit and declines incomplete or unsupported predicates.

On the exact Shenggan payload, the plan has 182/360/716 event pairs for one,
two, and four output-tile entries, zero named barriers, and one terminal ALL.
The independent concrete checker finds no ordering relation added relative to
compact MAT or the reconstructed manual plan. It removes 446/928/1,892
relations relative to compact MAT and 16/32/64 relative to manual. It retains
same-bank prefetch, separate early A readiness, memory/event/rearming coverage,
and native ACC checks.

Construction has 718 analytical sites, three selected updates, 21,846 replay
site evaluations, ten recurring channels, zero recurring omission trials, and
zero recurring-analysis site evaluations. The recorded exact run took about
0.46 seconds. This satisfies the current requirement to avoid an expensive
whole-plan refinement or candidate-search stage.

Portable and native tests cover guarded/repeated bank entry, two and three
banks, malformed correspondence, first-use conjunction negatives, and
reconstruction of the exact emitted words. Device performance remains to be
measured; event and barrier counts do not establish latency.

The six representative attention outputs were unchanged by the bank-occurrence
mechanism before first-use integration. Their guarded/non-scalar correspondence
remains separate work and must be rerun in the next corpus/device qualification.

## 1. Restrict recurring endpoint coalescing

The current recurring coalescing rule can combine comparable endpoint cuts
without proving that the complete selected command word preserves payload
ordering. A shared static cut is insufficient: an intervening publication can
relay newly acquired completion to another engine.

Implement a recognizable private coalescing case that requires:

- compatible dynamic occurrence and event generation;
- unchanged publication and acquisition participation;
- no intervening publication or other selected command that can relay newly
  acquired completion;
- every source operation crossed by a later publication is already required at
  the same consumer deadline;
- every ordering change claimed absent is checked against the complete selected
  word, including fixed and recurring endpoints.

Until this certificate exists, keep broader recurring endpoints separate. If a
resource fallback deliberately broadens ordering, classify and measure it
explicitly instead of treating it as ordinary F3 sharing.

Acceptance criteria:

- strengthen `sharedRecurringPrefixes` with an independent forbidden-order
  assertion;
- add an interleaved-publication negative regression based on
  `WAIT early; SET relay; WAIT late`;
- retain memory coverage, matching, consumption-before-republication, and the
  original endpoint participation;
- show that accepted coalescing adds no payload finish-to-launch relation on
  the admitted fixture family.

## 2. Make recurring specialization optional under key pressure

Qualification currently binds every proposed recurring channel to a physical
key before redundant proposals can be removed. An oversized optional proposal
population can therefore fail with `EventResource` even when ordinary
construction could produce a valid plan.

Add a pre-selection resource check. If the complete specialized population cannot
fit, decline it before modifying the ledger and run ordinary construction.
Failure of the ordinary route may still report its genuine resource or
participation error.

Acceptance criteria:

- a redundant specialized population that exceeds the directional pool falls
  back to ordinary construction;
- failed specialization leaves no selected endpoints, reservations, receipts,
  or source snapshots behind;
- if both specialized and ordinary construction fail, report the ordinary
  construction failure without a partial specialized plan;
- existing admitted recurring protocols and their selected words remain
  unchanged when the pool is sufficient.

Longer term, select necessary logical channels before physical key binding and
reuse keys between disjoint scopes only with actual token-consumption and
publisher-rearming evidence. Do not add a whole-plan search or expensive
post-construction refinement pass.

## 3. Redesign guarded attention bank handling

The preserved prototype derives guarded physical-bank ready/release interfaces
from original control. It demonstrates the missing correspondence and removes
the targeted previous-bank-compute to next-bank-fill ordering edge. It is not
ready to apply:

- partial attention changes 146 event pairs to 149 while reducing named
  barriers from 86 to 81;
- single-block attention changes 74 event pairs to 88 while reducing named
  barriers from 27 to 25;
- partial-attention construction rises from about 0.8 seconds to about 6
  seconds, dominated by 103 vector updates and roughly 216,650 replay-site
  evaluations;
- the exact Shenggan GEMM plan stays unchanged.

The next design must derive useful interfaces from remaining obligations rather
than eagerly installing a ready/release pair for every qualified bank. It must
also avoid re-solving unchanged selected regions after each endpoint edit.

The general bank-qualified occurrence milestone does not by itself change these
six plans. Their guarded/non-scalar correspondence is not represented as the
finite scalar orbit admitted by the current native interface, so construction
reports no recurring interfaces. The next mechanism should export the guarded
participating occurrence and physical-bank correspondence, then select only the
channels still required after actual selected completion and consumption.

Acceptance criteria:

- preserve the demonstrated absence of current-bank compute ordering before a
  disjoint next-bank fill;
- do not increase event pairs on the six attention modules without a measured,
  justified ordering or device benefit;
- retain guarded participation, zero-trip behavior, repeated entry, physical
  bank identity, ACC separation, token balance, and rearming;
- bring construction cost back near the committed baseline before broader
  corpus or device evaluation;
- rerun the six attention modules, the exact Shenggan GEMM, independent
  command-graph checks, native reconstruction, and C++ lowering.

Preserved experiment:

- Report: `attention-guarded-work/REPORT.md`
- Patch: `attention-guarded-work/changes.patch`
- Raw plans and checks: `attention-guarded-work/corpus/` and
  `attention-guarded-work/ordering-comparison.json`

## 4. Avoid repeated immutable structure construction

Several services rebuild equivalent original-program structure:

- `Control`, `StorageFrontierAnalysis`, fixed-plan `analyze()`, and prefix
  queries each construct a control graph;
- native loop admission calls `hasQualifiedRecurringAccesses()` for candidate
  refinements, rebuilding control and storage analyses before the selected
  constructor builds them again;
- recurring omission trials rebuild fixed-plan analyses for each candidate;
- selected-ledger updates reuse topology but can still re-evaluate complete
  contextual graphs.

Create one immutable analysis bundle per imported program containing canonical
control, storage succession, requirement frontiers, and observation indices.
Share it across qualification and selected construction. Keep selected causal
state and event generations versioned separately: reusing immutable topology
must never reuse a receipt or token fact after a ledger edit.

Replace recurring changed-plan trials with direct frontier/current-credit
certificates where practical. Retain one independent cold validation of the
final selected program.

Acceptance criteria:

- count and report immutable graph constructions separately from state replay;
- native loop admission and selected construction share one compatible
  original-program analysis, or cache it under an exact program identity;
- unchanged ledger advancement performs no control/storage reconstruction;
- an endpoint edit invalidates affected selected state without rebuilding
  unchanged original control or storage relations;
- recurring selection does not invoke a full fixed-plan analysis merely to
  rediscover a relationship already represented by a qualified requirement
  frontier;
- differential tests show identical final commands and causal certificates
  before and after structural sharing;
- GEMM and attention measurements report preparation, qualification, replay,
  and final-validation work separately.
