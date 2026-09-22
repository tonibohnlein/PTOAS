# OAHS implementation TODO

This list tracks implementation work that is not ready for the committed
selected-plan design. Safety remains mandatory, but the acceptance criteria
also protect useful payload ordering, resource availability, and construction
cost. Do not reduce synchronization counts by enlarging publication prefixes or
advancing acquisition deadlines.

## Current priority order — GEMM baseline restart, 2026-09-22

Use [the semantic correction sequence](oahs-semantic-corrections.md) as the
implementation order. The review covers a later branch; establish each missing
fact on this baseline before replacing its consumer.

1. **Implemented: harden and validate `37554ef9b`.** First-use separation,
   deadline-local fencing, position-preserving release sharing and atomic
   optional-cohort decline pass focused/native checks. Corpus: 88/88; inherited
   compatibility refusals: 2/19. Resource/replay regressions are in HANDOFF.
2. **Next: shared facts and occurrence interfaces.** Derive physical-use relations from relevant dependency slices **together
   with** stable child occurrence/endpoint correspondence. Preserve independent
   facts and conservative effects regardless of event capacity.
3. Extend generation/support intervals and compose first-consumer, first-write
   and final-reader requirements before control refinement. Preserve residual
   repair and access-scoped ACC obligations.
4. Consolidate exact ordered-packet construction/binding, then exploit actual
   required-return coverage on a native prefill witness. Port later mechanisms
   selectively; retire superseded paths when their obligations are covered.
5. Recover reference GEMM/MAT and targeted ordering gains, classify corpus
   tradeoffs, and run matched device validation. Update the draft's algorithm
   and proof obligations alongside established mechanisms.

The old GEMM source snapshot's notes below are historical. The later device
campaign reported parity with the manual kernel; current host validation does
not constitute a new device timing claim. Preserve that plan and the independent
checker while replacing recognizer restrictions. Temporary implementation
boundaries must remain explicit in the correction document.

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
