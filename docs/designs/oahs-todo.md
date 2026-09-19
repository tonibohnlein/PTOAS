# OAHS implementation TODO

This list tracks implementation work that is not ready for the committed
selected-plan design. Safety remains mandatory, but the acceptance criteria
also protect useful payload ordering, resource availability, and construction
cost. Do not reduce synchronization counts by enlarging publication prefixes or
advancing acquisition deadlines.

## Current priority order

1. Complete the pending guarded-attention device qualification. In local work,
   attribute the proposed post-RMSNorm and attention opportunities on current
   output before selecting the next compiler change (C0 below).
2. AIC probability-readiness support is implemented and host-validated locally
   (C1): partial attention 126 → 119 emitted pairs; single-block 79 → 78;
   no added ordering on the 126 checked module-44 paths; GEMM unchanged.
   Next inspect post-RMSNorm's store-return support using the same current-credit
   selection rule before adding another mechanism.
   Follow with qualified AIV receive/reuse placement (C2); keep non-unit-step
   first-use qualification as a separate bounded control-analysis experiment.
3. Extend guarded participating-use support (C3) and validate it against paired
   projections, merge/finalization, quantization, and RoPE/staging (C4--C7).
4. Resolve the recurring coalescing ordering certificate alongside any affected
   change; do not claim general order preservation before this is done.
5. Optimize construction runtime and repeated immutable structure building.

The [composition roadmap](oahs-composition-roadmap.md) reviews the supplied
agent studies, maps them to current code, and gives concrete tasks, negative
tests, evidence limits, and cost requirements for C0--C7. Its reference-model
counts are not measured improvements to current native plans.
Its rated shortlist incorporates the four newly downloaded source notes and
inventory. Their archived hashes/excerpts were checked; executable experiment
archives are still unavailable. The earlier direct/indirect QK-release duplication has now been reproduced
and ordinary duplicate selection corrected; see [the handoff](../../HANDOFF.md) and
[the local report](../../../aic-completion-work/REPORT.md).

Finish and measure a coherent plan milestone before moving to the next item.
Runtime work may still be performed when it blocks plan experimentation, but it
is not the current optimization target.

## Locally completed GEMM plan milestone

The committed implementation carries finite outer-bank identity through the nested MAT
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
reconstruction of the exact emitted words. Device qualification reports
229.306 us median at 299.7 TFLOPS and 0.921 MAC ratio, matching the reconstructed
manual protocol's performance while retaining all numerical checks.

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

Current implementation gap: `joinedCycle` still permits the older later-release
publication/earlier-acquisition merge. Requiring both `a.storageRelease` and
`b.storageRelease` fixed category admission, but did not supply the missing
ordering certificate. The identical-publication check in the guarded episode
builder is not a certificate for every later merge or crossed selected word.

Acceptance criteria:

- strengthen `sharedRecurringPrefixes` with an independent forbidden-order
  assertion;
- add an interleaved-publication negative regression based on
  `WAIT early; SET relay; WAIT late`;
- add the distinct B-reader to earlier A-refill negative: a merged release
  must not make A reuse depend on an otherwise unrelated later B reader;
- retain memory coverage, matching, consumption-before-republication, and the
  original endpoint participation;
- show that accepted coalescing adds no payload finish-to-launch relation on
  the admitted fixture family.

## 2. Optional recurring specialization: implemented, verify and extend

`Constructor::recurring` now declines an oversized specialized population before
mutating the ledger and lets ordinary construction run. That removes the
specialization's immediate `EventResource` failure. Physical proposal binding
still precedes legacy omission trials; logical necessity before physical binding
remains longer-term work.

Remaining validation should explicitly exercise the fallback, including a case
where ordinary construction succeeds. A test requiring fewer keys because
logical episodes were already composed does not by itself test failed
specialization cleanup. Ordinary construction may still report a genuine
resource or participation failure.

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

## 3. Guarded attention bank handling

The current milestone composes exact cells with one guarded reader episode
before physical key allocation. It keeps each early readiness publication and
shares only an identical release publication, using the earliest comparable
reuse acquisition. It is disabled when authored synchronization is present and
declines before ledger mutation when the specialized key population cannot fit.

On representative host plans, partial attention changes from 146 to 126 event
pairs while named barriers change from 86 to 87. The independent module-44
checker finds no added payload ordering and removes the targeted
previous-bank-compute to next-bank-fill edge. Single-block attention remains at
79 pairs and 26 named barriers. Construction, reconstruction, and lowering pass;
device performance remains unmeasured.

The next adoption decision depends on device evidence. Current-output analysis
can proceed meanwhile using C0--C3 in the composition roadmap. Include both AIC
and AIV: indirect QK release, output-return placement, tail-buffer reuse, and
guarded bank/queue correspondence have different proof requirements.

The earlier eager prototype remains preserved for comparison:

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

Further work must derive useful interfaces from remaining obligations rather
than eagerly installing a ready/release pair for every qualified bank. It must
also avoid re-solving unchanged selected regions after each endpoint edit.

The guarded episode mechanism now exports participating occurrence and exact
physical-cell correspondence for the admitted common-reader case. More general
guards and unequal deadlines still need selection from actual remaining
completion and consumption obligations.

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
