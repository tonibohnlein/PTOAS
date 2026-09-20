# OAHS current handoff

Updated: 2026-09-20

## Checkout

- Repository: `/home/toni/work/pypto3_sync_more/PTOAS-oahs-clean-m1`
- Branch: `codex/oahs-clean-m1`
- Placement/admission milestone base: `8afb90f17` (MAT cycles, based on `fe1fc454b`); use Git HEAD for its subsequent commit
- Current milestone: placement/admission hardening and opt-in experiments; MAT down_proj device result positive, family measurements pending
- Retained prior experiment: AIV receive placement and FIFO occurrence qualification

Verify the branch, HEAD, and working tree before continuing. A newer user commit supersedes this record.

## Current implementation: MAT reader-region cycles

Commit `8afb90f17` selects complete readiness/return cycles at each qualified
reader child's boundaries, preserving separate first consumers. This removes
the late second-child dependency on first-pair refill. See
[mechanism and evidence](docs/designs/oahs-mat-reader-cycles.md) and
[device task](docs/designs/oahs-mat-release-device-task.md).

22/22 portable suites and native tests pass. 88/88 supported corpus inputs
construct/reconstruct: eleven projections change, other 77 outputs are identical.
Across 37 paths, production removes 6,881 ordering relations with none added;
GEMM remains byte-identical and passes its 200/394/782 checker. Construction
replay decreases; graph growth is a compact first-consumer prefix.

Down-projection device feedback below now confirms a latency improvement;
family transfer is still pending. The complete protocol also proves the MTE2
fences unnecessary before insertion.
M/FIX/ALL fences remain. A separately certified placement control restores all
baseline MTE2 fences; it removes 6,173 relations with none added. Measure this
control alongside production, fe1 and existing. The down result is recorded below.
The existing dispatch package preserves the precommit working-tree snapshot.
The implementation is now committed as `8afb90f17`; keep that package and its
hashes fixed for the ongoing family measurements.

## Current local work: placement-oriented views and isolated experiments

See [implementation, scope and results](docs/designs/oahs-placement-experiments.md)
and [targeted device tasks](test/benchmarks/placement/DEVICE_TASKS.md).

- Default hardening stages mandatory optional-protocol checks and rejects an
  exactly-fitting cohort that strands uncovered ordinary demands. The exact
  base reproducer fails; the candidate falls back and succeeds. Invalid-proposal
  rollback is covered. Next-provider-only selection preserves the old tie rule.
- Experimental switches isolate frontier motion, word-start source gaps,
  deferred acyclic acknowledgments, class-invariant first consumers and the
  exact-equal-coverage binding probe. Mandatory checks remain enabled.
- 23/23 portable suites and native tests pass. Default hardening produces 88/88
  byte-identical corpus plans. A 144-case matrix plus 16 final representative
  default/probe checks pass. Default GEMM remains 200/394/782, no named barriers.
- Native source-gap witness: 7 pairs unchanged, two payload dependencies removed,
  none added. Deferred acknowledgment: 5 -> 4 pairs, three/two dependencies
  removed on false/true paths, none added. All runnable microkernel arms lower.
- Class-invariant fixture extends construction coverage: candidate succeeds,
  default refuses rearming. It needs device qualification, not a claimed
  before/after default speedup. Exact-coverage attention probes change no plan.
- No-motion GEMM has 330/652/1296 pairs with identical checked payload ordering.
  Legacy frontier motion remains default pending this isolated cost comparison;
  the general ordering certificate remains open.
- Trial configurations expose recurring omission separately from final helper
  pruning. Proposal checking has its own counters. FIFO contextual replay remains
  expensive and is NOT optimized by this work.

Shared views use the existing deadline-indexed requirement frontiers, native
cell/loop effects, original control and selected replay/key state. No additional
completion authority or eager per-cell channel allocation is introduced.

These placement/admission changes have no device result yet. The separate
MAT campaign now reports down_proj at 25.792 us versus 30.077 existing and
31.646 fe1, with 36/36 correctness passes. The retained-fence control is
25.704 us: MTE2 omission has no measurable benefit in this comparison. See
[measurements, attribution limits and pending work](docs/designs/oahs-mat-reader-cycles.md#device-feedback-down_proj-2026-09-20).
The family and matched profiles remain pending. Keep the running MAT task
and both packaged source snapshots fixed.

Placement task bundle: [source-complete archive](../device-handoffs/oahs-placement-experiments-8afb/oahs-placement-experiments-8afb.tar.gz),
SHA-256 `1ab2919c8887869da33741dcd42d936472d2035ef385b65f320d594dd74f4d51`;
566 package checksums and reconstruction of all 8,046 candidate source files
verified. That precommit snapshot remains immutable; this commit also records
subsequently received MAT device feedback in documentation.

Historical sections below describe
previous milestones and are not current status.

## Prior local diagnosis: remaining projection gap (2026-09-20)

HEAD is `fe1fc454bb80cd4810410bcc8bd9212f36c8b5d8`; the device candidate remains
unchanged. User-reported first-consumer results pass 117/117 correctness, improve
all projection members 16.7–22.9% versus 495, and preserve identical-binary GEMM
parity. Residual versus existing: down 5.6%, Q/out 7.8%, KV 8.2%, gate/up 9.2%,
LM 17.8%. Matched profiling reconciliation remains remote and pending.

The [new local report](../projection-gap-work/REPORT.md) identifies late
MAT release as the next concrete target: the next A0/B0 loads acquire an MTE1
prefix containing the unrelated second reader child. Existing releases the
first pair earlier. Second-child A1 readiness is broad in both plans, so it is
not the distinguishing dependency. Saved current plans were regenerated exactly.

Two constructed diagnostic return protocols pass the unchanged imported-program
checker with every fence retained. Across 21 paths they add no payload relations;
down/17 chunks loses 224 relations and LM/bounded one tile loses 224. Counts rise
343→354 and 363→370 respectively. These are local reference protocols, not an
implemented compiler change or a device performance result. General production
selection must preserve individual physical last-reader boundaries rather than
assume every pair in one child can share a release.

MTE2 fence omission remains separate: LM passes the all-path checker, down fails
at B0 load cut42 even with the split return. Finite traces alone do not authorize
that omission. Existing also fails the strict local event oracle on some down
tail/re-entry paths; comparisons exclude those paths rather than waive the check.
Artifacts: workspace `projection-gap-work/`, report, native JSON/CSV inventories,
causal witnesses, reference PTO, 63 finite checks and six negative mutations.

Next plan mechanism: compact last-participating-reader interfaces across child
loops, preserving actual readiness/return/rearming and tail byte overlap. Keep
fences in the first controlled implementation. The requested review-hardening
items remain pending: cohort starvation, common-frontier certificate, next-provider
selection, and FIFO contextual replay accounting. No production code changed in
this analysis. Historical sections below describe their original milestones.

## Current result: first-consumer placement

The native first-consumer mechanism is implemented in the working tree on top
of 495fb9cbd. See [the implementation/evidence report](docs/designs/oahs-first-consumer-placement.md)
and [the next device task](docs/designs/oahs-first-consumer-device-task.md).
It keeps an input's early publication, acquires it once at its actual first
consumer, and retains credit through later reader iterations. It qualifies only
small straight prefixes with an inactive producer pipe and a genuine early-source
opportunity. No fences are deleted or hardware assumptions changed.

Checked key reuse across sibling entry protocols is part of realizing this
placement: token emptiness and actual consumption knowledge are required, followed
by the existing all-path protocol trial. No reservation is freed just at lexical exit.

Local validation:

- 22/22 portable suites; native positive nonzero-lower/non-unit-step repeated-entry
  construction/reconstruction and five admission negatives.
- 87 original corpus modules plus Shenggan construct/reconstruct successfully.
  Exactly 11 projection modules change. All six attention modules and post-RMSNorm
  remain byte-identical; GEMM is identical except for a trailing newline.
- 37 paired projection paths, 511,187 checked local conflicts, identical payloads,
  unchanged fences, zero added payload relations.
- Down (17 chunks): 359→343 pairs and 184 relations removed for one tile;
  712→680 and 368 removed for two tiles. Gate/up/KV/Q/out: 418→398 per active
  phase, 230 removed. LM bounded tile: 393→363, 235 removed.
- GEMM remains 200/394/782, zero named barriers, one ALL, with all overlap checks.
- Down/KV lower to C++; the new down and sibling KV lit checks pass.

Artifacts: workspace sibling `first-consumer-work/`; `corpus.json`,
`family-ordering.json`, logs and generated plans. Eight separately supplied A5
reference inputs fail layout/typed import and are not part of the qualified A3
corpus. This work has no candidate device timing yet. The delivery package records the
exact candidate commit in `candidate-manifest.json`.
The existing remote campaign stays pinned at 495fb9cbd.

Remaining plan opportunities are the second child's A1 readiness and complete
MAT-cycle support before genuinely redundant local repairs. Do not combine a fence
change with the pending first-consumer placement measurement.

### Manual attention reference task (external)

User-reported original manual smoke and pipeline cases pass on device, including
three stable pipeline repetitions, at the upstream 1e-3 threshold. The earlier
failure was cross-case input/binary contamination. The agent recorded a kernel-
launch token fix and omission of an unavailable manual-mode build flag; the
AUTO-mode replacement is not equivalent. Matched-manual PTO transcription is
still pending. The remote agent reports that PTO lacks the EN_UNIT_FLAG pipe
attribute, so it is preparing a separately labelled UF_ENABLE=0 explicit-event
family. UF_ENABLE=1 remains a separate reference; the reported UF_ENABLE=0 smoke
pass does not yet qualify its pipeline case. Preserve payload/allocation,
QK_PRELOAD=4, queues and prologue/body/epilogue scheduling. Check every expected
input file size/read: the upstream harness can falsely pass when inputs are absent.
This is a separate task from the local projection placement.

## Preliminary device feedback, 2026-09-20

The user reports 153 correctness runs with zero failures, a 21.5–23.4% projection
latency reduction versus the preceding handoff, and GEMM parity on the hardened
200-pair plan. Projection latency still exceeds existing by 25.7–52.7%.
See [the recorded medians and arithmetic corrections](docs/designs/oahs-projection-device-preliminary-20260920.md).
The supplied "regression closed" column and GEMM noise-floor explanation need
correction. Profiles/raw archive are pending; results are not locally reproduced.
This earlier device report validates neither the new first-consumer placement nor
the three new reference adoptions. First-consumer readiness is now implemented locally and awaits its separate
device comparison; attribution must distinguish other remaining stalls.
Follow-up aggregate profiles support reduced projection concurrency and partial
recovery; candidate factors are 1.19–1.39 versus existing 1.58–2.08. Per-instruction
timelines are unavailable on the remote CANN build. Use static deadline witnesses,
resource-conflict counters and controlled placement comparisons; a factor near
one alone does not prove complete serialization. The report records counter-scope
and active-time arithmetic caveats. ResourceConflictRatio/MemoryL0 are pending.

## Latest reference-source audit

### Three distinct reference tasks prepared

The user identified the ordinary PTO-ISA GEMM benchmark as too close to the
already-solved Shenggan pattern. Keep its artifacts but deprioritize dispatch.
The new [task set](test/benchmarks/compositional_references/README.md) covers
CATLASS example 25 retained-A, example 06 cross-output-tile preload, and manual
PTO-ISA attention with delayed QK/PV and separate ingress/output lifetimes.

Each task has a self-contained source/harness archive under `../device-handoffs/`:
`oahs-retained-a-495fb9cbd.tar.gz`, `oahs-preload-495fb9cbd.tar.gz`, and
`oahs-attention-reference-495fb9cbd.tar.gz`. Each includes its own dispatch prompt,
shared comparison protocol, exact compiler/CATLASS/PTO-ISA archives and hashes.
These are adoption-and-benchmark tasks: matched plans/device results for these
three sources are not prepared yet. The task agents construct the source adapters
and matched arms before numerical validation/timing. Preserve native unit flags,
FP16 output and mixed-core contracts; explicitly label any alternate template.
No tasks were dispatched by this agent. Serialize measurements sharing a device.

### First matched benchmark prepared

`test/benchmarks/manual_sync/` now provides a reproducible first adoption of the
pinned manual PTO-ISA GEMM. Production stays at `495fb9cbd`. Full source control
instrumentation matches all 388,224 payload operations on 24 cores, including
column-major B and the 32768-byte LEFT-bank spacing. Source C++ and normalized
manual PTO are separate device arms to expose transcription/lowering costs.

The original fixed A/B readiness keys fail the ordinary causal rearming check.
An explicitly labelled `manual_banked_keys` control changes only key identity,
passes bounded local checks, and preserves event count and payload ordering.
For one K=6144 tile: manual/banked manual 270 pairs, OAHS 296, existing 245;
named barriers 0/0/0/97. OAHS adds no checked payload order versus banked manual.
All eight PTO arms lower to C++; native construction/reconstruction succeeds.
Four discriminating oracle tests pass. No device build/timing was run locally.

The self-contained bundle is `../device-handoffs/oahs-manual-reference-495fb9cbd.tar.gz`;
its `DEVICE_TASK.md` specifies five arms, three seeds, queued repeats, cached
FP64 references, and 180 timings per arm. Host artifacts are under
`../manual-sync-benchmark-work`. CATLASS and full attention benchmark adoption
remain pending; this result must not be attributed to those kernels.

User requested actual hand-written Ascend C / PTO-ISA synchronization patterns,
rather than generated PTO examples. The [pinned survey](docs/designs/oahs-manual-sync-references.md)
checks source and build provenance and ranks CATLASS ping-pong/preload/retained-A,
PTO-ISA manual GEMM, and two attention implementations. CATLASS is pinned at
`2b85ed307b281baa76d663f11a9c9aa228d56652`; PTO-ISA at
`c0d7148e95ef73bd12a73165fdce4b723a3b7e72`. Twenty-six selected files and SHA-256
manifests are under `../manual-sync-reference-work` (about 318 kB of source).

Manual PTO GEMM/attention builds do not enable the separate PTO automode flag;
explicit source events are visible. This is a source/build audit, not a claim
that device lowering inserts nothing. No external build or device run occurred.

Most useful confirmed pattern: separate operand readiness at first L1-to-L0 use,
separate L1 release after the last copy, and L0 release after its matrix reader.
Manual attention also separates ingress-storage reuse (V→MTE2) from output reuse
(MTE3→V at vector work), while CATLASS's delayed PV schedule keeps workspace
message and operand-bank identities separate. Unit flags and shape-dependent M
barriers require their own target contract. Production first-consumer work below
remains the immediate implementation task; no constructor change in this audit.

## Projection first-consumer reference study

Production HEAD is `495fb9cbda7649f15a7fc09d6b1d91ffb7737d54`; its device task
is already dispatched. Current follow-up edits are tests and documentation only.
The [reference study](docs/designs/oahs-projection-reference-study.md) records
exact down_proj acquisitions, external reference suitability, six reference PTO
plans, 37 paired paths and the remaining production mechanism.

Confirmed across down, gate/up, KV, q/out and LM: B readiness can be published
immediately after its actual MAT load and acquired once at its first extraction,
without waiting for later unrelated MAT loads. Reference plans pass 511,187
strict local conflict checks and event/rearming checks, with no added
finish-to-issue relation. Six modules parse/verify natively. **They are diagnostic
plans, not constructor output or device-validated improvements.** LM uses distinct
SSA views of the same physical bank; do not match only allocation names.

New regression: `oahs_first_consumer_reference` checks early source/first-consumer
placement, the safe-but-broad loop-entry alternative, missing readiness, repeated
single-token consumption, skips and varying episode lengths. The optional
`check_projection_trace.py --require-early-mat` accepts the reference down plan
and rejects current production. Existing lit acceptance is unchanged. The new CTest and the 18-path projection
checker pass. All six attention modules reconstruct byte-identically; current
GEMM one/two/four-tile checks pass at 200/394/782 pairs and zero named barriers.

A separate one-batch diagnostic removes all MTE2 fences from each completed
current plan. The unchanged cold checker accepts gate/up, KV, q/out and LM;
down rejects at B0 load cut 42. These results motivate preparing complete MAT
cycles before fence repair; they do not authorize blanket deletion. The early-B
reference retains every existing fence so the effects remain separable.

Next production task: represent the first participating consumer of an invariant
physical generation inside qualified non-unit-step loops. Keep the original
source prefix and consumer deadline, one-token participation and real rearming
proof. Merely relaxing `loopEntryFrontier()` moves B readiness before unrelated A
work. Avoid the rejected full counted/nested expansion. First reproduce the
reference locally with native all-path checks, then revalidate GEMM/attention and
package a separate device arm. The dispatched SHA remains fixed.

Artifacts: `../projection-reference-work/REPORT.md`, `down-acquisitions.csv`,
`reference-ordering.json`, `negative-results.json`, `dma-fence-diagnostic.json`,
per-module reference/current PTO and JSON. These local paths must be packaged
explicitly for any remote agent.

## Previous milestone: projection operand-bank overlap fix

The placement defect is now fixed in construction: qualified independent banks
retain early operand readiness and their own previous-use releases across
sibling/guarded child entries. The prior selector discarded these valid episodes
unless two cells shared their complete reader frontier. The new admission selects
an independent cohort with one producer/reader direction before physical binding;
it preserves existing common-reader compositions and capacity fallback.

Down_proj retains **166 sites** and adds no graph expansion or recurring omission
trials. Its 17-chunk one-/two-tile traces remove **315/632** finish-to-issue
relations with zero additions. Gate/up, KV, and out projection likewise remove
376 relations per active phase (756 for both gate/up phases), with zero additions
on the checked paths. Early readiness and same-bank refill assertions now reject
the old plan. Static event counts increase; this is an overlap improvement whose
device benefit is still unmeasured. Genuine BF16 M completion remains required.
LM head also removes 346/696 relations on bounded one-/two-output-tile paths;
these use output-loop start parameters, not physical core IDs. New remote results
put its **prior** handoff slowdown at 1.9463x, provisionally pending correctness.
The MTE1-heavy-work hypothesis is contradicted by neutral qkv_proj; all three
qkv_proj outputs remain byte-identical under this fix. Use paired timelines to
attribute stalls, and keep out-projection AIV as a neutral performance control.

Local construction is about 0.71 s for down, 0.69 s for gate/up, 1.47 s for KV,
and 0.18–0.19 s for q/out. Qualification is below 0.4 ms; contextual replay increases
(e.g. down 3,498 -> 23,430 visits). These costs are recorded separately. The
counted-loop expansion experiment was excluded because its child-exit drains
added parent-DMA dependencies; the retained mechanism needs no loop expansion.

Validation: 21/21 portable suites; native regression; production down_proj and
18 concrete memory/event/ordering traces; sixteen paired family traces;
87/87 corpus constructions/reconstructions (71 byte-identical); production
Shenggan FileCheck and one/two/four-tile checks. All six attention and both
post-RMSNorm plans remain byte-identical. Sixteen corpus outputs change, detailed
in `../projection-overlap-work/REPORT.md`; new device timings are not available.

GEMM is byte-identical to the review-hardened **200/394/782-pair**, zero-named-
barrier plan. The earlier device parity result belongs to **182 pairs**.

Next: device-qualify projection overlap and GEMM against pinned baselines. Remaining
MAT readiness can still observe later unrelated loads; inspect those deadlines
before claiming the whole projection regression is closed. Preserve the FIFO
experiment and its separate cost/peer qualification work. The next remote
campaign is specified in `docs/designs/oahs-projection-device-task.md`; its
dispatch package pins the commit and includes all required local inputs.
Full evidence and alternatives are in
`../projection-overlap-work/REPORT.md`.

## Retained milestone: projection attribution and review hardening

The supplied device campaign shows a 64–76% regression across down/q/kv/gate-up
projections, present before guarded/helper changes. Local regeneration of
`down_proj` selects **zero recurring channels**: the broad release merge is not
its cause. All 21 ordinary transfers use common cuts. Readiness for the first
matrix operation observes preparation of both operand roles; the next first-role
refill acquires completion of the later second matrix operation. Existing has
separate earlier publications. Concrete identical-payload comparisons reproduce
these extra edges (38 handoff-only versus 6 existing-only relations for one tile,
two K chunks). These are ordering witnesses, not device stall attribution.

The full decision/physical-deadline inventory, fence residuals, explicit graph
witnesses and their scope are in `../projection-review-work/REPORT.md`.
The small even-K single-tile comparisons pass local memory, balance and rearming
for both arms. Other existing-plan paths have unproved rearming in this local
model, recorded without adding edges or claiming a native wrong-code finding.

Review amendments now implemented:

- Distinct release publications no longer undergo the broad later-publication /
  earlier-acquisition merge. Identical-publication/common-reader sharing remains.
- Alternative source frontiers report all-path additional credit with explicit
  no-regeneration checks; loop-entry sources use source-time coverage and region
  invariance. Hop deletion, one-arm missing support, stale-reader and early-prefix
  tests retain the distinction between motivating requirements and actual credit.
- Impossible Known promotions decline before loop-entry candidate solves. The
  down_proj plan stays identical; loop-entry analysis visits fall 2,471 -> 0.
- Three-engine tests vary inner lengths independently over repeated entries,
  including empty intervening entries. Qualification time, selected replay,
  final helper trials/visits/time and final certificate work are separate metrics.

**GEMM changed:** 200/394/782 pairs for one/two/four tiles, zero named barriers,
one terminal ALL. Separate MAT releases replace the broad shared release. The
new native check forbids the last B reader from gating the next A refill. All
previous overlap/memory/rearming checks still pass. Relative to the saved
182-pair plan, no payload ordering is added; 14/30/62 relations are removed.
The 182-pair plan remains the device-verified reference. Performance of this
200-pair plan is unmeasured and must not inherit the earlier timing result.
GEMM still has 718 sites, three updates and 21,846 replay visits; recurring roles
rise from ten to twelve, with no omission trials.

Validation: 21/21 portable suites, native regression, 87/87 corpus constructions /
reconstructions (all 87 outputs byte-identical to the prior FIFO baseline),
production GEMM FileCheck and one/two/four-tile trace checks; repository hooks
and direct compliance scan report no findings. Run the scanner with
`OAT_MAX_WORKERS=2` (`-w 1` means incremental mode, not one worker). The standalone
Shenggan fixture is separate from those 87 corpus modules. Test logs, paired
plans, JSON/CSV inventories and reproduction scripts are in the report directory.

The independent-bank milestone above now addresses operand-role readiness and
previous-use release through these original loops. It uses invocation-owned
physical episodes, so positive-step counted-loop expansion is unnecessary.
A paired device pipe timeline is still needed to quantify the remaining stalls.
The FIFO experiment's cost and full device/peer qualification remain separate.

## Retained experiment: qualified FIFO receive placement

The native AIV importer now derives the two-slot GM use correspondence from the
pinned unsplit vector tile-entry contract. It requires one invocation-owned
handle, no other backing-root users/aliases, complete alternating pop/push
episodes and no product with another refined observation dimension. Whole-episode
skips preserve slot state. Both phases share original command words; no runtime
counter or guard is added. TFREE remains a no-op and both directions share GM.

Slot precision alone does not improve construction: the previous supporting
return is absent when the first reuse is repaired. The new path therefore
selects two logical return/acknowledgment directions before ordinary repair and
physical key binding. The actual preceding return covers the older same-slot
writer, permitting the current acquisition after TPOP. No future credit, final
wait-sinking pass or per-cell channel population is introduced.

Partial modules 44–47 remain 119 pairs / 87 named barriers / two terminal ALL
barriers, with ten AIV acquisitions delayed. Original payload/control text is
preserved exactly. Native construction/reconstruction and A3 lowering pass.
All 87 corpus modules pass; the other 83 outputs are byte-identical, including
post-RMSNorm and single-block attention. All 21 portable suites and the native
regression pass. The native plan passes 140 independent finite path comparisons:
478,125 conflict checks, no added payload order, unchanged counts. GEMM remains
182/360/716 pairs, zero named barriers and one terminal ALL.

**Cost limitation:** AIV sites 738 -> 1,423; the same 125 edits now use contextual
replay, raising replay visits 26,206 -> 813,458. Construction is about 18.5 seconds
in the isolated diagnostic (23–33 seconds in native runs; latest 29.6), versus a
recorded 0.40 seconds before. This is a retained working experiment, not a claim
that the low-cost objective is met. The cost is repeated selected-state replay,
not an unbounded slot/guard analysis. Preserve the successful mechanism while
addressing that distinction; do not expand another history product.

Evidence and reproduction: `../fifo-slot-work/REPORT.md`, `native-diagnostic/`,
`native-comparison.json`, `corpus/summary.json` and validation logs. Local order
comparisons do not include the full coupled peer/backpressure protocol or device
performance. The already-issued d6c573127 task remains historical; the new
projection/device task includes this retained experiment as a separate target.

## Committed milestone: post-RMSNorm output-return composition

Base: `d6c5731279229f369579ab1dd69634f3a49bffa7`. This milestone is recorded
in the commit containing this handoff. The previous attention device task
remains pinned to the base commit.

Regeneration confirmed that the provider-selection change alone leaves
post-RMSNorm unchanged: gamma's private return is preinstalled by recurring
qualification. Declining all recurring channels removes that return but adds an
MTE2 barrier, so that diagnostic was not adopted.

`CyclicFrontiers.cpp::qualifyPipelineCycle` now prepares a restricted three-engine
write / in-place-work / final-reader cycle before ordinary repair. It preserves
the early input and gamma readiness words. The required store-to-load return
carries gamma reader/writer completion and readiness consumption, so gamma does
not receive its own release channel. The same actual credit preserves the
normalization result established through initial scratch release.

Admission requires exact cells, period one and straight same-occurrence middle
work; it declines unrelated owner protocols and fixed/authored words. It does
not generalize guarded or unequal-bank sharing. Physical key assignment and
final causal/reconstruction checks remain unchanged. There is no graph
expansion or new omission trial; genuine vector barriers remain.

Modules 22/23: **16 -> 10 static pairs**, **10 named barriers unchanged**, one
terminal ALL. Two outer entries with 20 iterations in each phase execute
**360 -> 246 pairs**, with all barriers unchanged. Forty-eight finite paths /
1,017,762 conflict checks pass; payload finish-to-issue relations are identical.
The graph remains 175 sites; selected updates 37 -> 22, replay visits
21,292 -> 11,773, recurring channels 2 -> 4, zero omission trials.

Validation: 20/20 portable suites; support/deletion/independent-reader/retained-
value regressions; native regression; both RMS variants construct, reconstruct
and lower. All 87 corpus modules construct/reconstruct: 84 byte-identical,
two RMS improvements, one RMS sample changes only event-key numbering. All six
attention plans are identical. GEMM remains 182/360/716 pairs, zero named
barriers, one terminal ALL, with emission/FileCheck/trace checks passing.

Details, scope and reproduction: `../rms-completion-work/REPORT.md` and its
paired native plans, logs and ordering comparison. Device benefit is unmeasured.
Next: qualify this mechanism on device separately
from the already-dispatched d6 suite; then return to AIV placement (C2) once its
receive-slot and tail-storage contracts are established.

## Committed milestone: probability readiness supports QK release

Base is the committed guarded-episode/hardening milestone `a6bbe1ff8`.
The committed constructor change is in `SelectedGroups.cpp::groups`.
It lets an independently required Overlap provider compete during Known
selection when its current selected prefix already covers a Known requirement.
Existing Known-source prefixes remain separate from that source's later
Overlap requirements; actual selection/replay supplies completion.

The module-44 QK-to-PV overwrite previously selected direct M-to-MTE1 release
before considering the required probability-readiness receipt. The latter
already carries QK completion through the current selected M-to-FIX,
FIX-to-MTE2 and MTE2-to-MTE1 path. It now discharges the reuse requirement
without creating the duplicate return. No recurring population, graph expansion,
new refiner, queue contract or hardware assumption was introduced.

Emitted static results:

- Partial attention modules 44–47: AIC **69 → 64 pairs**, AIV **57 → 55**;
  total **126 → 119**, named barriers unchanged at 87, two terminal ALL barriers.
- Single-block modules 48–49: **79 → 78 pairs**, 26 named barriers unchanged,
  two terminal ALL barriers. Analytical endpoint counts exceed emitted counts because
  complete identical guarded words share their emission.
- Shenggan GEMM unchanged: **182/360/716 pairs**, zero named barriers and one
  terminal ALL for one/two/four tiles.

Validation: 20/20 standalone suites; extended focused three-engine regression
including missing-support negatives and 0/1/2/4 visits; native regression
driver; all six attention constructions/reconstructions and A3 level3 C++
lowering; GEMM production emission, FileCheck and independent trace checks.
Module-44 finite comparison checks 126 paths / 90,699 conflicts, valid memory,
balance and rearming, and **identical payload finish-to-issue relations**.
Its imported local-effects model does not replace device/cross-core qualification.

AIC module 44 keeps 407 sites / three recurring channels / zero recurring
trials. Selected updates fall 66 → 61, replay-site evaluations
111,095 → 102,810. Device performance of this change is unmeasured.

Artifacts and reproducible diagnostics:
`../aic-completion-work/REPORT.md`, `corpus-summary.json`, `compare.py`,
`ordering-comparison.json`, paired baseline/candidate logs and plans,
all six emitted/lowered attention plans, and test logs.

### Remaining roadmap

The reviewed studies and rated choices are in
[`docs/designs/oahs-composition-roadmap.md`](docs/designs/oahs-composition-roadmap.md).
Downloaded source notes were checked against 18 archived hashes, 329 attention
excerpt lines and the post-RMSNorm input hash; evidence is preserved in
`../composition-review-20260919/`. Their reference-model counts are not native
compiler measurements.

Post-RMSNorm attribution and the required additional cycle mechanism are
recorded above. Keep pending attention device qualification separate from this
unmeasured local change. AIV receive placement (C2) still
requires queue-slot/tail-storage qualification; it was not implemented here.
Guarded use correspondence (C3) and the broader shared-release ordering
certificate remain open. Read retained projection results before repeating work.

## Guarded attention milestone

The constructor now derives complete guarded producer/reader episodes for exact
canonical cells before assigning physical event keys. Cells with the same
guarded first-reader episode keep separate early readiness publications and may
share one identical reader-release publication. The shared release uses the
earliest comparable reuse acquisition. This is admitted only when ordinary
recurring qualification produced no requests and no authored/fixed
synchronization is present.

The implementation does not reserve one recurring release channel per exact
cell and does not use omission trials for the new guarded channels. If the
specialized channel population cannot fit the directional key pool, it declines
before mutating the ledger and ordinary demand-driven construction continues.

Review hardening in the same milestone:

- local fences are selected only when their actual occurrence deadline is
  reached; future analytical copies are no longer permanently repaired from an
  unfinished checkpoint;
- first-use descriptors reject a true arm that can reach another listed
  decision before a tracked backedge or exit;
- the older nested shared-release merge requires both operands to be classified
  as storage releases. Its broader ordering certificate remains open work.

For attention modules `pypto_lib__prefill_fwd__44` through `__47`, the generated
plans have 126 SET/WAIT pairs, 87 named barriers, two terminal ALL barriers,
three cube recurring channels, zero vector recurring channels, and zero
recurring omission trials/analysis sites. Modules `__48` and `__49` remain at
79 pairs, 26 named barriers, two terminal ALL barriers, and their six existing
periodic channels on each participating function. All six construct,
reconstruct, and lower successfully.

The independent module-44 check covers 126 paths and 90,699 conflict checks. It
reports zero added payload-order relations, 347 removed path-local relations,
and absence of the designated previous-bank-compute to next-bank-fill edge.
This is host evidence only; device latency is pending.

The exact Shenggan GEMM regression remains unchanged at 182/360/716 pairs for
one/two/four tiles, zero named barriers, and one terminal ALL. The full
standalone OAHS suite passes 20/20.

Device instructions and the reviewed source patch are outside the repository:

- `/home/toni/work/pypto3_sync_more/guarded-attention-work/DEVICE_TASK.md`
- `/home/toni/work/pypto3_sync_more/guarded-attention-work/guarded-episodes.patch`
- patch SHA-256: `8ed4a0169ef66102f50e79c37b7ad93dc85b43ea96f9d1cc73b9aa60c28454f9`

## Delivered local synchronization plan

The current implementation composes three direct constructor mechanisms:

1. A finite physical-bank occurrence interface carries the outer MAT bank residue across the nested reader region. It relates each overwrite to the previous participating use of the same bank without expanding a product of nested first/middle/final modes.
2. Two operands in one qualified physical-bank episode retain separate early readiness transfers but share one exact storage-release return. This preserves early A extraction and removes the redundant second MAT release channel.
3. A first-use prefix qualifier recognizes the original conjunction `outer_k == 0 && inner_k == 0`. It splits one entry prefix, shares the remaining graph, and removes the impossible repeated ACC-initialization paths. It supplies no completion credit and retains conservative behavior for incomplete conjunctions, disjunctions, and unsupported loop forms.

Exact qualified cycles bypass whole-plan omission trials. Local fences are now
decided at their actual occurrence deadline; the hardening removed eager
batching over future analytical copies. Canonicalization still shares commands
in a genuinely shared word. The GEMM construction figures below are the
recorded GEMM milestone measurements, not a new whole-corpus cost bound.

## Exact Shenggan result

Generated artifact:

- `/home/toni/work/pypto3_sync_more/bank-occurrence-work/shared-release-first-use.pto`
- SHA-256: `3f1e19bf600e137685e0fdd4a49b5ea8ec4e19f1bf53f375618190ea2dfab8f2`
- construction log: `/home/toni/work/pypto3_sync_more/bank-occurrence-work/shared-release-first-use.log`

For identical concrete payloads:

| Tiles | Event pairs | Named barriers | Terminal ALL | Added / removed vs compact MAT | Added / removed vs manual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 182 | 0 | 1 | 0 / 446 | 0 / 16 |
| 2 | 360 | 0 | 1 | 0 / 928 | 0 / 32 |
| 4 | 716 | 0 | 1 | 0 / 1,892 | 0 / 64 |

The manual protocol has 167 event pairs for one tile. The 15-pair difference preserves separate early A/B readiness; the independent command-graph comparison finds the generated plan to be an ordering subset of the manual plan on all checked traces. Event count alone is not a performance estimate.

The one-tile direction counts are:

```text
M -> MTE1      66
MTE1 -> MTE2   18
MTE2 -> MTE1   32
MTE1 -> M      64
FIX -> M        1
M -> FIX        1
```

The trace checker validates one, two, and four output-tile entries. It checks memory conflicts, native ACC ordering, event occupancy, consumption before republication, final balance, same-bank prefetch, separate early A readiness, and the absence of current-bank compute ordering before different-bank preparation or parent DMA. Its injected terminal drain remains a discriminating negative control.

## Construction cost

The exact run reports:

```text
sites                         718
selected updates                3
replay site evaluations     21,846
recurring channels             10
recurring trials                0
recurring analysis sites        0
construction time          ~0.46 s in the recorded run
peak RSS                    ~112 MB
```

The first-use qualifier adds 67 analysis sites to the 651-site bank graph, but does not create a nested mode product. The expensive recurring omission population is gone. Construction time remains suitable for corpus and device qualification; repeated immutable structure construction remains a later optimization target.

## Validation completed

- standalone OAHS CTest: 20/20;
- portable bank occurrence tests: guarded and repeated entry, two and three banks, non-identity bank mapping, malformed boundaries, and exact shared-release structure;
- portable first-use tests: 145 concrete traces, zero trip, repeated entry, malformed boundaries, and genuinely repeating initialization negatives;
- native selected driver, including first-use conjunction and negative variants;
- exact native construction and reconstruction;
- exact Shenggan lit test: frontend lowering, FileCheck, and independent trace checker;
- `git diff --check`.

Sanitizers and device execution were not run locally.

## GEMM device qualification

The current 182-pair, zero-named-barrier GEMM plan is device-qualified. Across
180 samples it measured 229.306 us median at 299.7 TFLOPS and 0.921 MAC ratio,
compared with 230.548 us for the reconstructed manual protocol, 358.570 us for
compact MAT, 391.834 us for the earlier handoff, and 534.956 us for existing.
All 20 correctness runs passed with identical error ratios across the five
arms. The measured result is parity with the manual protocol, with a small
0.54% median advantage in this campaign.

Archive: `/opt/pypto/oahs-gemm-16564fa8a.tar.gz`, SHA-256
`9f71dc1fe129da538fe16ecebbd73d12410185b8a9176f91e512286e2c9dd25d`.

## Remaining work

1. Device-qualify guarded attention episode composition. Modules 44--47 require
   authentic runtime scheduler state; modules 48--49 are the primary runnable
   cases.
2. Attribute current post-RMSNorm/AIC residuals and select one native completion
   support experiment; follow the composition roadmap for AIV and other families.
3. Restrict broader recurring endpoint coalescing with an ordering certificate.
   Optional specialization now declines cleanly under key pressure.
4. Reduce repeated immutable control/storage construction after plan quality is settled.
