# OAHS implementation TODO

This list tracks implementation work that is not ready for the committed
selected-plan design. Safety remains mandatory, but the acceptance criteria
also protect useful payload ordering, resource availability, and construction
cost. Do not reduce synchronization counts by enlarging publication prefixes or
advancing acquisition deadlines.

## Authoritative next order: placement experiments on 8afb90f17

See [implemented mechanisms and evidence limits](oahs-placement-experiments.md)
and [isolated device tasks](../../test/benchmarks/placement/DEVICE_TASKS.md).
This status supersedes the historical campaign priorities below.

0. Keep the MAT device snapshot fixed. Down now reports 14.3% faster than
   existing, 36/36 correctness passes and no measurable MTE2-omission benefit
   versus the retained-fence control. Finish LM/family transfer and matched
   profiling; do not generalize the down result yet.
1. **Implemented locally:** staged optional-proposal rejection, conservative
   exact-fit admission, separate trial controls/counters and next-provider-only
   selection. Default corpus is unchanged. General scoped recurring-key reuse
   and complete future-allocation admission remain open.
2. **Ready to measure:** native source-gap witness with unchanged pair count,
   forbidden dependency removed. Extend beyond acyclic virgin-key word-start
   gaps only with exact coverage/participation/rearming certificates.
3. **Ready as a separate control:** no-motion grouping. GEMM adds commands with
   identical checked ordering. Legacy motion remains default; obtain a specific
   contextual certificate or accept the measured no-motion cost before changing
   the default policy.
4. **Ready to measure:** deferred acyclic acknowledgment, with both branch
   outcomes and a real-key-reuse regression. Optional acquisitions and recurring
   cases retain the conservative fallback.
5. **Ready for device correctness:** access-class invariant first-consumer
   fixture with disjoint producer work. Default OAHS refuses this input; do not
   claim a matched-default speedup. General guarded/cross-child qualification
   remains future work.
6. **Host experiment only:** equal-coverage probe finds actual attention pairs
   but no positively certified selection change. Keep disabled; obtain a
   discriminating positive case before broadening it or launching device timing.

FIFO remains 813,458 replay evaluations / 125 contextual updates in the sampled
AIV construction. New mandatory proposal work is separately charged. Reuse of
contextual propagation and immutable structure remains explicit unfinished cost
work; no compilation speedup is claimed.

## Latest priority: measure MAT reader-region cycles

Committed as `8afb90f17` on fe1, pending device measurement. See
[the mechanism and validation](oahs-mat-reader-cycles.md) and
[the four-arm device task](oahs-mat-release-device-task.md).

- Complete each qualified reader child's readiness/release cycle before fence
  selection. Preserve separate first consumers and physical return boundaries.
  Production passes 88/88 corpus construction/reconstruction; exactly eleven
  projection plans change. GEMM, attention and post-RMSNorm remain identical.
- Measure down first and LM as transfer, then projection family. Production
  removes MTE2 fences by actual cycle credit; retain the separate certified
  control with baseline MTE2 fences restored to distinguish the effects.
- Finish matched profiling reconciliation on the remote agent. The pinned
  MAT campaign closes the down gap and beats existing by 14.3%; the other
  projections remain pending. Preserve the separate fence-restoring control.
- Revisit second-child A1 readiness only if a concrete remaining dependency is
  found; the earlier broad readiness existed in both baseline and existing.

The subsequent placement milestone implements admission hardening and next-provider
selection, exposes no-motion and trial variants, and measures remaining FIFO
work separately. See the current status above; FIFO replay optimization and a
general moving-frontier certificate remain open.

## Pinned hand-written reference kernels

**Revised priority, 2026-09-20:** the ordinary PTO-ISA GEMM is too similar to
Shenggan to be the next device priority. Keep that prepared experiment parked.
The user requested [three independent tasks](../../test/benchmarks/compositional_references/README.md):
retained-A (CATLASS example 25), cross-tile preload (example 06), and full manual
attention (PTO-ISA). Source-complete archives and dispatch prompts are prepared;
source adoption, matched automatic plans and device measurements remain for the
task agents. Frozen compiler `495fb9cbd`; no production changes in task preparation.

- **Prepared; device pending:** manual PTO-ISA GEMM versus OAHS/existing, with
  original C++ and normalized PTO controls. See
  [benchmark protocol](../../test/benchmarks/manual_sync/README.md) and
  [device task](../../test/benchmarks/manual_sync/DEVICE_TASK.md).
  The upstream readiness-key rearming gap remains an explicit contract question;
  a separately labelled banked-key control passes local checks. Record timings
  and paired timelines before drawing a performance conclusion.
- **Next adoption:** CATLASS ping-pong with separate A/B lifetimes, then preload
  and retained-A. Preserve the reference's payload prefetch and target-specific
  instruction modes; account for translation costs with an original source arm.
- **Later adoption:** full manual attention, including queue/cross-core contracts
  and prologue/body/epilogue. No local-only projection is a full attention result.

The [manual synchronization survey](oahs-manual-sync-references.md) now pins
CATLASS Ascend C and PTO-ISA manual kernel/build sources. The manual PTO build
uses explicit events without the separate automode build flag; generated PTO
examples are not independent manual references. No external kernel was run.

Use CATLASS's separate first-consumer readiness and last-L1-copy release as the
first projection target. Next study retained-input and cross-child ownership;
then manual attention's separate ingress/output reuse and delayed QK/PV schedule.
Keep hardware unit flags, queue semantics and payload prefetch changes separate
from synchronization placement. Source locations and ranked experiments are in
the survey; 26 pinned files and hashes are in `../manual-sync-reference-work`.

## Historical targets: reference-driven projection placement

The device task is dispatched at `495fb9cbd`. Keep that arm fixed. The
[reference study](oahs-projection-reference-study.md) confirms two separate
opportunities on current output:

- **First-consumer MAT readiness:** source-local B publication and one receipt
  at the first B extraction improve down, gate/up, KV, q/out and LM reference
  plans without added checked ordering. Implement the general invariant-physical-
  generation / first-participating-consumer interface. Preserve non-unit-step
  original control, aliases, actual rearming and separate early A readiness.
  Implemented and host-validated; production down now passes the early-MAT
  target. Measure the isolated change with the new device task.
- **Complete MAT-cycle support before fences:** unchanged cold checking accepts
  omitting all current MTE2 fences in gate/up, KV, q/out and LM. Down fails at
  the B0 load. Attribute the missing partial-state support and down's incoming
  paths before changing construction; no production fence-deletion stage.
- **Second-child readiness:** A1's current entry receipt still observes B1.
  The early-B reference experiment leaves this separate deadline unchanged.
- **Reference acquisition:** use existing same-payload device baselines first;
  pin a suitable external pipelined PTO kernel for an additional manual/stripped-
  sync benchmark. Match hardware, dtype, layout and payload before comparing
  sync quality. No external BF16 projection performance match is established.

The portable reference regression tests the mechanism's ordering boundary;
37 native-source-derived finite comparisons support the plan target. Neither
proves device benefit or that the current constructor realizes the target.

## Historical campaign order (superseded by the order above)

1. Complete device attribution of the projection operand-bank fix. Qualified independent bank
   episodes now retain early readiness and the previous same-bank reader release
   across child entries, without graph expansion. Down_proj removes 315/632
   checked ordering relations for one/two tiles; gate/up, KV and out projection
   also improve, with no additions on checked paths. Counts increase; preliminary
   device feedback reports 21.5–23.4% lower latency versus before, with all 153
   correctness runs passing. Remaining slowdown versus existing is 25.7–52.7%;
   the regression is not resolved. See the
   [reported medians and corrections](oahs-projection-device-preliminary-20260920.md)
   and `../../../projection-overlap-work/REPORT.md`. Profiles/archive are pending.
   First-consumer placement is now implemented and locally validated across the
   projection family; dispatch [its isolated comparison](oahs-first-consumer-device-task.md).
   Preserve genuine fences while measuring this change. The second-child A1
   publication and remaining MAT-cycle support are separate subsequent targets.
2. Review corrections are implemented and host-validated: distinct release
   prefixes stay separate; alternative/entry providers expose additional current
   coverage; repeated-entry cycle tests vary lengths; final helper trials are
   separately counted. GEMM remains barrier-free at 200/394/782 pairs, with no
   added ordering and 14/30/62 removed relations against the 182-pair reference.
   Device-qualify this changed plan; earlier parity measurements belong to 182.
3. Retain the successful AIC and post-RMSNorm compositions and the retained
   FIFO receive-placement experiment. The FIFO plan improves checked placement
   but increases contextual replay from 26,206 to 813,458 visits. Its cost and
   full peer/device qualification remain unresolved. The pending device task
   stays pinned; do not silently substitute this working tree.
4. Extend guarded participating-use support and use paired projections,
   merge/finalization, quantization and RoPE as generalization tests only after
   attributing an actual remaining requirement on current output.
5. Optimize repeated immutable structure construction and replay. Keep those
   measurements separate from synchronization-plan quality.

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

## Device-verified GEMM reference (before current review hardening)

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

The current hardening removes `joinedCycle`: different final-reader publication
frontiers now remain separate. A linked independent B-reader/A-refill regression
and the native Shenggan quality check cover this restriction. Identical-prefix
sharing remains. A broader endpoint-motion certificate, including crossed
ordered words, is still separate future work; the guarded identical-publication
rule is not such a certificate.

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

## 2. Optional recurring specialization: partial fallback, extend admission

`Constructor::recurring` now declines an oversized specialized population before
mutating the ledger and lets ordinary construction run. That removes the
specialization's immediate `EventResource` failure. Physical proposal binding
still precedes legacy omission trials; logical necessity before physical binding
remains longer-term work.

Resource-allocation failure is only one rejection path. The source review also
finds an invalid initial omission-analysis candidate can break its trial loop
and still reach commitment. Mandatory protocol rejection must discard the
complete staged state. Separately, an exactly fitting, protocol-valid cohort
can reserve keys required by unrelated ordinary work. Add linked regressions
for both; rollback alone does not fix later starvation.

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
