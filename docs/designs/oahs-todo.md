# OAHS implementation TODO

This list tracks implementation work that is not ready for the committed
selected-plan design. Safety remains mandatory, but the acceptance criteria
also protect useful payload ordering, resource availability, and construction
cost. Do not reduce synchronization counts by enlarging publication prefixes or
advancing acquisition deadlines.

## Current priority: broad sweep follow-up

The instruction policy is now **use existing InsertSync's translation**. No
independent OAHS approval of instruction semantics is required. Keep the optional
semantic audit separate from compilation. Instruction-effect corrections belong
in the shared interfaces/translator; absence of a registered contribution is not
an OAHS error.

2026-09-22 exact-input result: **19/19 construct and reconstruct**. The six
constructor failures are fixed locally through general occurrence, participation
and consumption-return handling. See [diagnosis](oahs-constructor-compatibility.md).
The stricter instruction-admission gate remains removed; do not reintroduce it.

Review amendments implemented locally after `5e0a72772`: common dormant/active
ownership exclusion, split-helper successor qualification and an indexed
acyclic no-next-publication answer. The checked borrowing paths remain intact.
See [scope and validation](oahs-key-binding-consistency.md). General cyclic
continuation-query cost and KDA order inclusion remain open.

The authored-local-event boundary is closed at the public pass: both algorithms
share the existing explicit-event exclusion before dispatch. Authored functions
remain unchanged; their protocols are not thereby validated. Mixed protocol
import/validation remains separate future work. See
[the entry contract](oahs-shared-semantics.md#authored-local-synchronization).

The additional KDA/dspark RMSNorm check exposed ordinary dormant-key exhaustion.
The existing checked restoration path now covers ordinary owned keys too; both
modules pass default public insertion and reconstruction with identical outputs.
This restores admission, with no new device or ordering-quality claim.

Next acceptance work:

1. Run full model builds and device correctness for the newly admitted sweep
   kernels. Local evidence currently covers native construction/reconstruction.
2. Compare complete ordering for changed partial-attention AIV plans in modules
   44–47: each adds 15 static SET/WAIT pairs; replay grows 610,965 → 616,865.
   All 88 corpus modules pass, 84 remain byte-identical. Refined
   control now obtains full contextual states before selection; Qwen AIV needs
   850,615 replay evaluations. Preserve correctness while reducing that work.
3. Resume KDA's optional early-final-read quality gate (1784 relations removed,
   706 added), then RMSNorm/hc_pre prefix attribution and compressor serialization
   with its finite-25% kv golden limitation explicit.

See [exact artifacts and instruction-policy change](oahs-sweep-followup.md#shared-translator-policy--2026-09-22).
Runtime/lowering-model audits are separate work. No new timing claim follows
from admitting these inputs or completing their synchronization plans.

## Implemented locally: ordinary final-read source

`--final-read-sources` now uses an exactly-once final-only gap before the shared
anchor word, with source-time coverage, source-time key reuse and actual return
support. It preserves acknowledgment endpoints and protects the gap against
later insertion. See [scope and validation](oahs-final-read-sources.md).

Historical blocker (now resolved by neighboring-use checks): KDA's later retained role at cut 330 failed rearming. Distinguish
missing support from occurrence matching. Identify the actual key and its next
publication deadline; realize separate support there without widening the early
source. The full no-added-order gate remains mandatory before device timing.

Other corpus plans may change. Judge each changed plan by correctness and its
complete ordering difference; byte identity is a diagnostic, not the goal.

## Current gate: first-write KDA order inclusion

Cross-control key-1 rearming, inactive key-0 borrowing and shared-occurrence F7
matching now produce a full native KDA plan. Exact emission preserves repeated
same-word generations. Host correctness passes, but the committed-plan comparison
removes 1,512 relations and **adds 634** in the 1,727-payload case. Keep the
experiment disabled and withhold its device task. Next: identify the actual
return/export paths introducing M-completion → MTE2-load order and remove that
broadening during selection, preserving all consumption support. Do not replace
this requirement with smaller relation counts or a deletion search.

The first added path is now attributed to an M→MTE1 acknowledgment exported
through the MAT release. The linked supplied-plan witness keeps the acknowledgment
and releases before it on the final read (2/3/4 visits, strict order inclusion).
Next qualification: an exactly-once last-read source at that pre-acknowledgment
gap for ordinary construction. The current top-level, step-64, conditional-suffix
case is outside native last-reader admission. Relaxed online helper removal was
reverted after a later rearming failure; do not treat immediate replay as a
certificate for future binding. Full native ordering remains unchanged.

## Working milestone: shared-word next-publication support

The linked constructor regression reproduces the old missing-rearm failure.
A structural must-path query now checks every occurrence of the new consumption
against the next selected publication, including backedges and shared words.
It uses an existing actual reverse receipt or selects a qualified return after
the new consumption without moving the early source gap. The old experimental
full-trial repair has been removed. Missing return evidence, insufficient reverse
credit, repeated entries and complete order inclusion against late publication
are covered. All 25 portable suites and the native selected suite pass.

The disabled first-write KDA experiment gets past cut 428 but fails later at cut
720 with no reusable key/nonrecursive acknowledgment. Do not promote it or fold
it into the device sweep. Next work is a concrete physical-binding witness at
that later boundary, followed by a bounded certificate or conservative admission.
See [phase-support evidence](oahs-kda-phase-support.md). Default KDA output remains
identical to the committed choice-frontier plan. Corpus result is recorded in
`../kda-first-write-work/shared-rearm-corpus/summary.json`; the four changed
attention AIV plans motivated gating the repair behind the disabled first-write
option. Targeted reruns in `shared-rearm-gated-corpus/` restore all four baseline
plans. General shared-word promotion remains unqualified.

### Reservation reuse follow-up

Implemented an inactive-interval certificate for fully materialized recurring
keys, within the disabled first-write experiment. KDA now binds key 0 at cut 720
without a helper and advances to cut 724. At 724 that borrowed key is correctly
full; the remaining empty key 1 still needs a return after cut-400 consumption.
Next: qualify that cross-control return while preserving the early gap, then
require full native acceptance and no-added ordering before device timing.
Do not globally free recurring reservations. See the phase-support note for
ownership, interval, repeated-entry and missing-support regression coverage.

## Latest local milestone: KDA choice frontiers — 2026-09-21

This entry supersedes the older active-investigation/commit status below.
On the rebased branch, [original-choice consumer frontiers](oahs-choice-consumer-frontiers.md)
are implemented and host-tested. Ordinary construction preserves
A readiness before independent B extraction while retaining B's later receipt.
KDA removes 32 finite explicit payload relations, adds none, and passes native
reconstruction. All 25 portable suites and both native suites pass; all 88 paired
historical corpus modules pass with identical plans. Compiler work increases:
two staged checks (5,144 sites), replay 184,538→189,813. Device validation remains
needed for the changed KDA plan; current remote task status is not inferred.

Remaining KDA priorities:

1. Complete first-conflicting-write occurrence/rearming qualification without
   broadening early source gaps. Current opt-in experiment still fails KDA;
   keep it disabled and separate from the successful choice mechanism.
2. Identify complete MAT readiness/reader-return support across phases before
   local repair. Removing MTE2/M barrier groups currently fails validation.
3. Explain why MTE1 repairs remain despite completed-plan redundancy, then
   construct their support earlier if it reduces overhead efficiently. Their
   diagnostic deletion removes no payload ordering; it is not an overlap gain.
4. Measure the new KDA plan using matched coupled execution and the qualified
   device methodology. Existing-pass rearming concern is unresolved on hardware;
   do not weaken OAHS safety to match it or claim superiority from event counts.

## Current status at a glance — 2026-09-21

This summary supersedes older scheduling/status statements below. Those sections
retain historical context, not a live remote queue.

- **Latest local implementation:** relay correction and campaign status are
  committed as `91fc0e728`. The subsequent key-feasibility-before-ranking
  amendment is implemented and host-tested, still uncommitted. It fixes the
  admission witnesses and preserves all 88 corpus plans. No new device arm
  is needed for those unchanged native outputs.
- **Joint-reader down_proj device task:** completed; correctness passed all
  12 cases / 24 launches. Performance unresolved. Further timing requires a
  qualified measurement method, not more repetitions of the completed task.
- **Outstanding focused handoffs:** coupled attention FIFO/relay validation,
  KDA measured-case artifact/attribution work, and retained-input microkernels.
  No returned results for these are recorded locally; current remote job state
  is unknown. Their committed task files are linked in the sections below.
- **Next local investigation:** use the KDA source packet when it arrives to
  identify a concrete native miss. First-receipt support for second-leg key reuse
  and scoped resource ownership remain extensions requiring their own witnesses.
- **Larger backlog:** contextual frontier-motion certificates, broader exact
  source gaps/observation composition, scoped producer support/key ownership,
  and witness-driven lowering precision. These are not all active projects.
- **Closed campaigns:** MAT family, placement microkernels, conditional MTE2
  isolation, broad full sweep, and joint-reader correctness. Reference adoption
  (retained-A, cross-tile preload, manual attention) remains separate backlog;
  task files alone do not establish dispatch or execution.

## Current milestone: relay binding before ranking — host complete

[Binding admission](oahs-relay-binding.md) now filters each candidate using the
existing source-time/key-interval queries before comparing placement. Occupied
first/second preferred legs reproduce a refusal on the old core and now select
an available alternate. Empty-but-unknown consumption, intervening use, real
return-supported reuse, missing memory support and missing rearming are checked.
Only the winner receives a staged solve; unknown binding retains normal fallback.

All 24 portable suites, both native executables and twelve attention finite cases
pass. All 88 corpus modules / 97 functions are byte-identical to `91fc0e728`.
Key queries increase 823→835; replay, staged solves and selected plans stay fixed.
There is no new device candidate. The first receipt still does not prospectively
rearm the second leg during the read-only probe.

## Completed milestone: bounded relay-selection correction

[The return-query hoist and both linked review witnesses](oahs-return-query-relay-review.md)
are complete. At 32 sharing candidates, propagation drops 496→31 queries and
389,205→24,330 site visits without changing requests. All 24 portable suites,
native tests and 88-input/97-function corpus checks pass; every corpus plan is
byte-identical to `a0c1d761e`. No corpus case reaches the sharing query.

The [bounded selection correction](oahs-relay-selection.md) now selects the
five-pair / 30-relation route in the avoidable witness and handles the other
case as an incomparable tradeoff. Required/acquired receiver history, crossed
middle gates and actual endpoint prefixes inform separate set comparisons.
Later receipts cannot silently broaden earlier forwarding to the same receiver.
Twenty-one balanced deletion negatives fail memory coverage; useful early
forwarding remains admitted when middle work and outward publication already
have the necessary credit.

- Validation complete: all 24 portable suites, final focused relay tests, both
  native executables and twelve finite attention cases pass. All 88 corpus plans
  / 97 functions remain unchanged, including the existing attention benefit.
- Keep the incomparable case and general ordering-certificate boundary explicit.
  This is a bounded ranking policy, not a proof about all future endpoint edits.
- Positive key feasibility before ranking is now implemented above.
  First-receipt support for second-leg rearming remains a separate extension.
- Alternatively, use the KDA artifacts or a concrete retained-input lifetime to
  identify a useful native miss before broadening support/occurrence admission.
- No new device campaign is needed for these unchanged native plans. Existing
  FIFO and joint-reader device tasks retain their pinned candidates.

## Completed host milestone: joint first/final native reader endpoints

[Implementation and evidence](oahs-joint-reader-prefix.md) now realize the
[down_proj lifetime](oahs-blocked-down-last-reader.md). B0's first acquisition
and A0's final publication coexist with step 128 and conditional matrix suffixes.
The compiler emits the narrow release; the reconstructed staged protocol and
18 independent traces pass (214 full relations removed, none added, executed
sync/fence populations unchanged). The original implementation pause is
superseded by the user's explicit instruction to implement this opportunity.

- Host qualification complete: 88/88 inputs pass, 11 projection plans change;
  37 family paths remove 1,414 full relations and add none with equal executed
  event/fence populations. Compiler work increases and is recorded explicitly.
- The [short device task](../../test/benchmarks/joint_reader/DEVICE_TASK.md)
  is complete: [audited results](oahs-joint-reader-device-intake.md) pass all
  12 correctness cases / 24 launches. Performance remains unresolved because
  identical-binary controls drift by 9–14%. Qualify the measurement method
  before further timing; no broad sweep or extra repetitions are warranted.
- Keep guarded participation, arbitrary same-word endpoint composition, broader
  active-producer support and generic frontier-motion certificates open.
- Treat the larger analytical graph and replay work as measured costs. Shared
  suffix words do not remove the need to preserve final/continuation identity.

## Device follow-up: static FIFO slots and split relay

The dedicated [row48/49 coupled attention task](../../test/benchmarks/attention_relay/DEVICE_TASK.md)
is prepared. Baseline `d1bf07ee5`, candidate `21f95f9b7`; this is separate from
the newer projection guards and the older full sweep. Qualify the original
shared-slot runtime in an isolated adapter, then check correctness/progress and
run the short matched comparison. The task authorizes that harness work; the
working separate-ring runtime must remain intact. Dispatch is not locally
confirmed. Host order reductions are not device speedup evidence.

## Completed local milestone: required-return sharing

[Required-return sharing](oahs-required-return-sharing.md) selects a composed
reader-region return before private-key allocation, with separate readiness
and unchanged supporting endpoints. Actual staged words prove producer support,
matching and rearming. The reversed-deadline case declines rather than moving
a wait. Twenty finite comparisons, seven paired native variants, all portable
and native suites pass. Native witness: equal 602 relations, pairs 10→7, reverse
keys 2→1. All 88 paired corpus plans remain unchanged from `21f95f9b7`.

Return-sharing applicability work still needs a real retained-input/preload
witness or a specific refusal in the KDA source packet. The down_proj target
above concerns last-reader placement, not return sharing. Do not schedule another broad device
campaign for unchanged plans. Global producer support, multi-occurrence return
frontiers and ambiguous shared observations remain conservative boundaries.

## Device follow-up: KDA projection attribution

[Initial diagnosis](oahs-kda-projection-diagnosis.md): identical prepared inputs;
only main AIC/AIV binaries differ, pad/zero binaries match. Scheduler window
202.880 ->281.390us accounts for almost all whole-entry increase. This is not
per-pipe or per-task attribution. Orchestration hashes differ and semantic
comparison remains required. Await the small measured-case source packet in
[the remote task](../../test/benchmarks/kda_projection/DEVICE_TASK.md); then
reconstruct current output and identify exact extra physical dependencies.
Do not choose a fix from the timing ratio alone or substitute a different kernel.

## Device sweep triage: new workload evidence (2026-09-21)

The [audited full-sweep report](oahs-full-sweep-20260921-review.md) takes priority
for choosing the next real native diagnostic. It measures frozen `2cc458cbe`,
not the latest FIFO commit. All report hashes and 250 raw-sample summaries match.
Prioritize GLM KDA projection (237.389 -> 316.950 us), then DeepSeek decode
compressor ratio4 and RMSNorm. Obtain prepared plans and per-kernel mapping from
the device's full archive before attributing the extra runtime to a mechanism.
The local download contains results/manifests, not those code artifacts.

Authentic two-layer Qwen prefill now passes both arms/seeds; reuse its coupled
runtime for the pending attention task after exact fixture/ABI/queue matching.
Keep frozen device candidates unchanged. Do not infer attention48/49 timings
from the unrelated `qkpv_plan` rows28–31. Small isolated measurements include
large identical-binary artifacts; request short controlled follow-ups, not
another broad sweep. Classify the 23 handoff-codegen blocker records using full
stderr and prepared inputs separately from performance work.

## Review reconciliation at `21f95f9b7` (2026-09-21)

This section takes priority over older implementation-order recommendations
below. The two reviews inspected only through `d1bf07ee5`; their finite-model
results are proposals, not current native or device measurements.

**Already addressed:** exact proposal checking/commit ordering, unnecessary
legacy-prefix work and the conditional deferred-ack admission gap. The newer
`21f95f9b7` additionally supports checked reused-key word-start gaps (opt-in),
static FIFO cursor identity and separate relay deadlines (ordinary handoff).
The attention diagnosis now has changed native plans; do not repeat the older
claim that all current corpus plans are unchanged. Coupled device validation
of the two changed single-block AIC variants remains pending. Generic interior
source gaps, arbitrary slot recurrences and contextual frontier-motion proof
are not solved by that commit.

### Next local work, guided by an actual missed opportunity

1. **Opportunity diagnosis before broadening admission.** Use existing cell-use,
   owner/observation, source and decision views to report the physical generation,
   first/last frontier, next conflicting use and exact refusal. Distinguish no
   opportunity, observation/occurrence limit, missing support, unavailable key
   certificate and already-acquired completion. Start with retained-input test
   cases and available retained-A/preload references; do not add a second causal
   analysis or claim a native benefit from a model-only example.
2. **Required-return sharing: bounded implementation complete.** See the
   [implementation and evidence](oahs-required-return-sharing.md). The existing
   positive now selects sharing, while reversed deadlines retain private
   returns. Completion, readiness consumption, outward effects and real reuse
   are checked. Broaden only for a concrete missed lifetime; no subset-deletion
   search or relaxation of the producer-support safeguard.
3. **Joint first/last observations: bounded implementation completed above.**
   Original positive steps, single-visit coincidence and conditional suffixes
   are qualified together. B0 first acquisition and A0 final publication now
   coexist on real down_proj. Remaining extensions are guarded participating
   readers, shared first/final word anchors and active-producer support; require
   another concrete missed lifetime before relaxing those boundaries.
4. **Support-driven endpoint words and certified grouping.** Fixed guarded
   publication-first ordering remains conservative: a valid request-order
   proposal can be declined after canonicalization. Derive same-engine receipt
   prerequisites and early-release constraints for a bounded new endpoint batch;
   preserve authored/committed order and validate the exact deterministic word.
   Never infer a hardware edge from traversal order on different engines.
   Tie motion certificates to outward publications, consumption and neighboring
   key uses; invalidate on edits. Keep this distinct from mandatory safety checks.
5. **Lowering precision, witness first.** Descriptor-scoped ACC qualification,
   address-dependent recurrence slices and qualified existing drain credit remain
   open. A read-only scan of the available MAT-campaign prepared inputs found
   attention `set_validshape` only in AIV functions (rows44–49); these cannot
   trigger the function-local ACC veto in their separate AIC companions.
   `gemm_eltwise__0` likewise updates only AIV. LM head row6 updates the actual
   ACC `%6` before output, so it is not an unrelated-descriptor positive example.
   This scan does not establish reaching descriptor state or cover every corpus.
   Find a concrete cube qualification miss before prioritizing an ACC change.
   Static FIFO slots do not fix the general `SyncSlotMapping::derive()` loop-wide
   recurrence/LCM restriction. Keep independent address slices separate from
   event capacity; do not merely increase the period cap.
6. **Scoped producer support/resource ownership, later.** Function-wide written
   cells can reject unrelated prelude work, but retain `repairFreeProducers`
   until a support-interval certificate covers every repair that could move.
   Preserve the X/Y fence-relocation negative and actual event lifecycle state.

No new code, linked tests or device results are claimed by this review
reconciliation. Do not substitute local follow-ups into the dispatched device
candidate. Equal-coverage tuning stays behind a positive selection witness.

## Active continuation after `d1bf07ee5`

1. **Wait for the device agent** before extending the last-reader experiment.
   Do not restart its pinned campaign for this local work.
2. **Implemented locally, opt-in:** word-start publications can reuse a key
   with actual incoming consumption knowledge and strictly earlier selected
   uses on an acyclic straight corridor. The one-key constructor witness
   removes four full payload relations with no additions; missing/late credit
   and neighboring-use negatives are checked. See [scope and evidence](oahs-reused-source-gaps.md).
   General interior/recurring gaps and useful changed native cases remain open.
3. **Implemented: native static FIFO slots and split relay deadlines.**
   Both single-block AIC variants now select the narrow early receive path and
   forward the later slot's completion before unrelated relay work. The source,
   relay gap and final receipt are checked separately; actual selected paths
   still establish rearming. Native reconstruction and twelve finite cases
   pass, with no added checked order. See [implementation and limits](oahs-static-fifo-relays.md).
   **Next:** coupled AIC/AIV numerical and queue-progress qualification before
   performance claims. Preserve the manual reference's actual overload and
   contract; do not time isolated halves. General changing-slot observations,
   multiple FIFO handles, UnitFlag and arbitrary relay optimality remain open.
4. **Grouping certificate still open:** supplied before/between/after export
   tests distinguish order-preserving and broader cases. Qualify exact ordered
   words and neighboring keys, and invalidate on relevant later edits. No
   blanket same-cut export veto, safety-only certificate or new default policy.

## Authoritative next order: placement campaign complete

### Review amendments checked at `6b1b32f46`

The reviews of `8afb90f17` through `d6e9b5365` identified two source-level
issues present after the retained-input commit. Both are now fixed locally;
see [implementation and validation](oahs-review-amendments.md). This order takes
priority over the broader milestones below.

1. **Implemented: exact proposal materialization.** `recurring()` previously
   checked request order but committed guarded publication-first order.
   One canonical ordered endpoint population now supplies
   the words for mandatory checking, producer-support and resource admission, omission
   trials, and commitment. The `repairFreeProducers` check now sees the committed
   ordering. The linked reciprocal-rearming witness rejects atomically; positive
   and omission tests check exact accepted words and channel identity. The old
   implementation fails the new regression. The reviews do not
   demonstrate an unsafe native emission; final validation remains required.
2. **Implemented: skip unused prefix comparison in normal sibling replay.**
   `contextualReplay()` calls the legacy prefix calculation only for prefix-only
   reuse/fallback or explicit tracing. Comparison availability and actual query
   work are separately reported. Chained-word scaling and full cached/cold
   state/endpoint tests pass; the old behavior fails the regression. All 88
   corpus plans and causal evaluation counts remain unchanged. This records
   work eliminated, not a measured wall-time speedup; no device timing needed.
3. **Implemented locally: bound deferred acknowledgment admission.**
   The linked one-key conditional-reuse test reproduces refusal after deferral
   while the closed baseline succeeds. Admission now requires a straight
   continuation to exit using the existing control index; it retains the helper
   before future branching. All four choice combinations and missing-support
   negatives are checked. The option stays disabled by default. More precise
   future-use admission or qualified conditional repair remains open; this
   conservative fix can retain a helper even without a later key reuse.
4. **Implemented locally: last read inside a straight nonempty child.**
   A qualified final-visit observation exposes the post-read source before
   trailing work. The existing physical-use/control views prove no later read
   and exactly-once participation; the selected cycle provides actual credit.
   Portable and native witnesses remove ordering with no additions. Native
   admission currently requires constant nonnegative bounds and unit step;
   guarded, empty/unknown and already-refined children retain the fallback.
   See [scope, proof checks and construction cost](oahs-last-reader-placement.md).
   Combining with other occurrence vocabularies and non-unit steps remains
   open. Keep any device follow-up separate from the dispatched campaign.

The contextual frontier-motion certificate remains a separate open quality
task. Equal-coverage selection remains experimental until a test demonstrates
`bindingChoices > 0` with certified benefit. Neither is resolved by safety-only
proposal validation. MAT-cycle device improvements and unchanged-sibling replay
are completed mechanisms, not new work to repeat.

See [implemented mechanisms and evidence limits](oahs-placement-experiments.md)
and [isolated device tasks](../../test/benchmarks/placement/DEVICE_TASKS.md).
This status supersedes the historical campaign priorities below. See the
[placement campaign intake](oahs-placement-device-results.md): host claims
reproduced, OAHS correctness gates passed, latency differences unresolved.
The returned archives pass local audits. The [conditional MTE2 isolation](oahs-conditional-mte2-isolation.md)
is complete: existing fails, a one-barrier mutation repairs it, and OAHS passes.
Next preserve a local existing-pass conditional-WAW regression and trace its
construction decision before implementing a general repair. No more device
repetitions are needed for isolation. Correct coverage is 35 successful placement
constructions plus one expected refusal; MAT owns the 264-row matrix. Repair
source-manifest and effective-optimization probes before new bundles.

**Historical consolidated campaign plan (superseded by the current summary):**
[consolidated task](../../test/benchmarks/open_experiments/DEVICE_TASK.md).
The broad `2cc458cbe` sweep has since returned and was audited above. This table
preserves the remaining reference work; it is not a live queue or instruction
to rerun the sweep. New focused tasks use their own frozen source pins.

| Workstream | Status / next action |
| --- | --- |
| D1 broad pypto-lib/PyPTO kernels and models | Complete; audited report received. Follow up concrete misses, starting with KDA. |
| D2 retained-input microkernels | Two host-qualified cases; device correctness/timing outstanding. |
| D3 retained-A reference | No completed matched comparison received; qualify source adoption. |
| D4 cross-tile-preload reference | No completed matched comparison received; qualify source adoption. |
| D5 manual attention | UF=0/UF=1 correctness reported; resume matched UF=0 PTO transcription and current automatic arms. |
| D6 coupled Qwen attention | Gated on authentic coupled runtime; part of D1, not standalone halves. |
| D7 no-motion GEMM | Optional low-priority command-cost control; previous evidence was host-only. |

Completed MAT, placement microkernel and conditional-MTE2 isolation campaigns
are not reopened. Source-gap/deferred/class-invariance microtimings remain
unresolved; another small noisy sweep is not a priority. The latest proposal/
replay fixes preserve corpus plans and need no separate device arm.

0. **Complete:** MAT family device campaign. All five projections beat existing;
   GEMM parity holds. No resolved MTE2-omission benefit. See the
   [final results and audit](oahs-mat-device-final-results.md). Obtain the detached
   checksum matching the downloaded final archive. The placement campaign
   is also complete; retain both snapshots.
1. **Implemented locally:** staged optional-proposal rejection, conservative
   exact-fit admission, separate trial controls/counters and next-provider-only
   selection. Default corpus is unchanged. General scoped recurring-key reuse
   and complete future-allocation admission remain open.
2. **Host/device correctness reproduced; latency unresolved:** native source-gap witness with unchanged pair count,
   forbidden dependency removed. Extend beyond acyclic virgin-key word-start
   gaps only with exact coverage/participation/rearming certificates.
3. **Host-only as designed:** no-motion grouping. GEMM adds
   commands with identical checked ordering. No device timing was requested or run. A small isolated comparison is optional;
   microkernel timing does not establish that GEMM's extra commands are free.
   Legacy motion remains default pending a contextual certificate/cost decision.
4. **OAHS device correctness reproduced; latency unresolved:** deferred acyclic acknowledgment, with both branch
   outcomes and a real-key-reuse regression. Optional acquisitions and recurring
   cases retain the conservative fallback.
5. **Device correctness reproduced; latency unresolved:** access-class invariant first-consumer
   fixture with disjoint producer work. Default OAHS refuses this input; do not
   claim a matched-default speedup. General guarded/cross-child qualification
   remains future work.
6. **Host experiment only:** equal-coverage probe finds actual attention pairs
   but no positively certified selection change. Keep disabled; obtain a
   discriminating positive case before broadening it or launching device timing.

FIFO sibling reuse now reduces the sampled AIV from 813,458 to 608,848 replay
evaluations at the same 125 updates, with identical plans across 88 corpus
modules. The fresh paired sweep passes 176/176 runs: 12/97 function instances
reduce replay, 85 are unchanged, none increase. Single-block AIV, QKV, RMSNorm
and top-k also benefit. The added dependency walk is separately counted. Repeated fixed points
inside an edited component remain unfinished cost work. Three matched local host
rounds reduce median construction/reconstruction wall time from 24.568 to
18.304 seconds; this is not a device performance result.

## Local work after the placement campaign

Keep source/binary identities for the completed experiments fixed. Local
follow-ups are separate candidates, with no automatic substitution into that
campaign. FIFO replay attribution and sibling reuse are implemented; see the
[findings and replay certificate requirements](oahs-fifo-replay-attribution.md).
The next bounded quality task is a narrow contextual frontier-motion certificate.
Its [outward-publication witness](oahs-frontier-motion-context.md) now has complete
payload-order comparisons and real key-reuse negatives. Exact constructor
proposal/word-order reproduction remains open; the supplied plans alone do not
prove a native defect. A linked two-layout probe exports before both waits and
does not reproduce it. Avoid a blanket same-cut-export veto. The
retained-generation task now has a native witness and a bounded implementation:
see [reader-child composition](oahs-retained-reader-children.md). The [multi-input extension](oahs-retained-producer-cohort.md) now admits complete
producer cohorts in one straight writer corridor. Its staged support check declines
if any producer payload requirement remains; the uncovered X/Y counterexample
still declines. All 88 corpus plans remain unchanged. The next boundary is an
actual corpus miss outside that qualified scope, not merely allowing more cells.

### Deliverables and exit gates

| Milestone | Local deliverable | Acceptance before the next step |
| --- | --- | --- |
| **L1: replay attribution (implemented)** | Opt-in per-solve trace: changed words, active/restart component, shared-word widening, reused sites, unique/repeated visits, per-component work and elapsed time. Partial AIV plus AIC/projection/RMS/GEMM controls reproduced. | All five modules retain identical PTO; counts reconcile. 64 edits confined to one AIV branch spend 202,944 evaluations in its unmodified alternative loop. |
| **L2: sibling replay reuse (implemented)** | Reuse unchanged predecessor-closed components; invalidate entire shared nonempty words. Dependency-walk work is separately counted. | Complete cached/cold checkpoint and endpoint comparisons pass, including failure/recovery. 23/23 suites pass; 88/88 native plans are unchanged. AIV replay falls 25.2%; active-loop fixed-point work remains. |
| **L3: frontier-motion certificate** | A narrow moving-frontier rule with an outward-publication counterexample. | Prove contextual ordering, not only memory safety; independent finite oracle rejects the broadened plan. A separate no-motion device control would be needed to assess runtime command cost. |
| **L4: retained-generation placement (bounded implementation)** | Shared nearest-role view selects one cycle across children; native A3 witness and eight portable traces pass. Single-cell and qualified multi-input cohorts; actual producer support checked before commit. | Separate positive persistence from reload, premature release and independent-reader negatives. Record analysis fact → endpoint choice → ordering change. |
| **L5: lowering facts** | First a qualified intrinsic-drain witness; next descriptor-scoped valid shapes; then address-sliced slot relations. | Each is a separate candidate with exact lowering scope and positive/negative importer tests; no inferred event reset, whole-M completion or joint bank-mode product. |

Device dispatch is conditional: identical emitted plans from L2 need host
equivalence and compilation-cost measurements, not another device timing sweep.
Placement/semantic changes in L3–L5 need a discriminating local witness before
preparing source-complete device tasks. Keep the current device archive immutable.

1. **Completed first replay improvement; retain remaining cost work.** Certified
   unchanged-sibling reuse preserves incoming states and complete endpoint
   aggregates. The trace and prefix-only control remain available. Further work
   must address repeated propagation within an edited loop, with the same full
   cached/cold state comparison. Do not add a second causal analysis, truncate
   loop history or weaken checks. Keep quality work below ahead of a broader
   rewrite of the fixed-point solver.
2. **Certify useful frontier motion.** The generic default coalescer still lacks
   a general contextual no-added-order certificate. The no-motion device cost is
   still unreported; locally prove a narrow admitted case, including outward
   publications and event generations. A safety check alone is insufficient.
3. **Extend shared lifecycle views from a real native miss.** Carry an invariant
   generation across sibling/guarded regions while preserving its last-reader
   frontier and actual first participating consumer. First identify a current
   missing qualification; do not broaden all recognizers or eagerly allocate
   one channel per exact cell.
4. **Find a discriminating equal-coverage binding case.** Current attention
   probes change no provider choice. Require an actual helper-free alternative,
   exact equal coverage, source-time binding evidence and an ordering comparison
   before expanding the probe. A synthetic cardinality tie is insufficient.
5. **Revisit smaller generalization kernels on current output.** Qwen partial
   merge/finalization first, then shared-input sibling projections; quantization
   and RoPE are later. Recount and attribute current obligations before proposing
   changes: earlier model-level savings may already be implemented. Keep A5
   native admission and queue/unit-flag contracts separate from A3 evidence.

Longer-term resource work: certified recurring-key reuse across disjoint scopes,
using actual consumption/republication evidence. Exact-fit rejection is already
implemented and should not be presented as unfinished. Likewise the successful
MAT projection improvement needs no second implementation.

## Review intake: placement and lowering facts, checked at 9f30b9fd8

The supplied reviews inspect `fe1fc454` and `16564fa8`. Their proposed MAT
reader-cycle milestone has since shipped and passed the completed family device
campaign. Reproduce any remaining second-child A1 dependency on current output
before treating it as a defect or performance target. Exact-fit rejection and
optional-proposal rollback are also implemented; general key-lifetime reuse is
still open. These follow-ups do not alter the dispatched device snapshot or
replace the active FIFO replay investigation.

### Confirmed current information losses

- **Qualified intrinsic local drain:** `SyncProtocolModel::localDrainBefore`
  records the pinned hard-collective contract, but `Native.cpp` explicitly
  supplies no completion credit. First establish a current residual across an
  original hard collective. Import its local completion at that anchor through
  the shared semantics and checker, without emitting another ALL. Test that
  event occupancy/consumption identities survive and that neither peer progress
  nor arbitrary GM visibility is inferred. This is the smallest new semantic
  opportunity, conditional on a useful native witness and exact lowering scope.
- **Descriptor-scoped valid shapes:** native ACC qualification still has a
  function-wide `SetValidShapeOp` veto. Qualify reaching descriptor definitions
  and aliases for the actual matrix operands. An unrelated descriptor update
  should preserve the certificate; a relevant or unresolved update must retain
  ordinary requirements. Keep the result access-scoped, never whole-M completion.
- **Independent physical slot relations:** `SyncSlotMapping::derive()` still
  requires every carried argument to match, forms their LCM, and is called with
  a cap derived from event-pool size. Dynamic `multi_tile_get` still retains the
  address union. Start with an address/slot SSA slice and an unrelated carried
  value negative-control pair; then independent periods three/four. Preserve
  separate use relations and charge represented output size. Do not raise the
  cap or enumerate a larger joint orbit. Unify explicit multibuffer and direct
  address forms through shared origin/slot/view facts, checking physical aliases;
  event feasibility remains a later construction obligation.

### Generalizations and contract audits

- **Retained generations across children:** build on `StorageFrontiers`,
  `RequirementFrontiers` and existing source/occurrence queries. Expose references
  to the physical generation, first/last readers, useful source positions,
  deadlines and qualified continuations; do not create another completion state.
  Test one load/two children, an intervening reload, premature release after the
  first child, and independent reader engines. Readiness persists only for the
  qualified generation; actual selected receipts supply completion/rearming.
- **Attention phase composition:** keep semantic row, physical bank predecessor
  and queue generation distinct across prologue/body/epilogue and skipped uses.
  Require a current native miss before extending qualification. Queue/peer
  obligations and A5 target admission remain separate from local A3 results.
- **Scalar completion audit:** the A3 profile still marks all S payloads
  synchronous. Inventory admitted scalar operations and their exact lowering
  evidence before expanding admission. This is a contract audit, not an
  established wrong-code finding or a switch to disable casually.
- **Later contract work:** lowering-owned may-read/may-write/must-write extents,
  verified macro boundaries and import of existing UnitFlag protocols. Keep
  UnitFlag import distinct from selecting new payload modes; block permission
  does not release all M operands. Additional event directions/IDs require their
  own target qualification. No native benefit is established by these reviews.

For each item retain the diagnostic chain: shared analysis fact → construction
decision → actual endpoints → checked payload-order difference. This intake was
a current-source inspection; no new compiler build, experiment or hardware
contract verification was performed.

## Completed priority: measure MAT reader-region cycles

Committed as `8afb90f17` on fe1; final family device results now available.
The original measurement checklist below is historical, not pending work. See
[the mechanism and validation](oahs-mat-reader-cycles.md) and
[the four-arm device task](oahs-mat-release-device-task.md).

- Complete each qualified reader child's readiness/release cycle before fence
  selection. Preserve separate first consumers and physical return boundaries.
  Production passes 88/88 corpus construction/reconstruction; exactly eleven
  projection plans change. GEMM, attention and post-RMSNorm remain identical.
- Measure down first and LM as transfer, then projection family. Production
  removes MTE2 fences by actual cycle credit; retain the separate certified
  control with baseline MTE2 fences restored to distinguish the effects.
- Completed family timing closes the regression for all five projections.
  Matched profiles cover down, gate/up and KV; no matched LM/GEMM profiles were
  delivered. The retained-fence control preserves the measured improvements.
- Revisit second-child A1 readiness only if a concrete remaining dependency is
  found; the earlier broad readiness existed in both baseline and existing.

The subsequent placement milestone implements admission hardening and next-provider
selection, exposes no-motion and trial variants, and measures remaining FIFO
work separately. See the current status above: sibling replay reuse is complete;
repeated work inside edited loops and a general moving-frontier certificate remain open.

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

The additional mandatory-protocol rejection and exact-fit starvation cases are
now fixed and covered by linked regressions in `9f30b9fd8`: an invalid optional
candidate leaves no committed state, and a valid full-pool cohort can decline
when it strands uncovered ordinary requirements. The default 88-case corpus
remains byte-identical. Admission is conservative direct-vocabulary accounting,
not a complete guarantee that every later allocation choice succeeds.

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
