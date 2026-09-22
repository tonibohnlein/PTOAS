# OAHS current handoff

Updated: 2026-09-22

## Checkout

- Repository: `/home/toni/work/pypto3_sync_more/PTOAS-oahs-clean-m1`
- Branch: `codex/oahs-clean-m1`
- Sibling-replay milestone base: `9f30b9fd8` (placement/admission), based on `8afb90f17` MAT cycles; use Git HEAD for this revision
- Current milestone: shared translation and six constructor compatibility repairs, host validation
- Retained prior experiment: AIV receive placement and FIFO occurrence qualification

Verify the branch, HEAD, and working tree before continuing. A newer user commit supersedes this record.

## Latest local amendment: authored-event exclusion

The public InsertSync pass now applies its existing explicit-local-event skip
before dispatching to either algorithm. Functions containing flags or
record/wait events remain unchanged, including on a second insertion run.
Unknown algorithm names still fail. This closes coexistence with unrepresented
authored event keys without changing ordinary instruction-effect admission.
It does not validate manual protocols or implement mixed fixed-command import.

The native suite passes, including new public-pass checks for all four event
operations, repeated uses, distinct IDs, automatic insertion and idempotence
under both algorithms. Logs: `../sweep-followup-work/authored-events-native.log`.
The focused local build script needed the existing `SlotAffineAnalysis` object
when linking the public legacy pass; no production source repair was needed.

The additional requested KDA/dspark RMSNorm check exposed ordinary dormant-key
exhaustion after `7f22b091f`. The existing restoration certificate now covers
ordinary owned keys as a last binding alternative. It checks restored helpers
plus the exact new pair, retains the publication gap, and pins the restored
helpers. Full protocol analysis certifies repeated use; the selected packet is
not subsequently enlarged with a private return. Ordinary unchecked binding
still excludes dormant ownership. See
[the follow-up](docs/designs/oahs-key-binding-consistency.md#exhaustion-with-dormant-ordinary-helpers).
All 25 portable suites pass, including split/common-cut and repeated-entry
restoration regressions. KDA AIC/AIV and exact dspark RMSNorm now construct and
reconstruct with default options. The KDA input retains the previously recorded
syntax-only `load_scalar` to `load` conversion. Logs and plans are in
`../sweep-followup-work/authored-events-kernels/`. This is synchronization
compilation validation, not device compilation/execution or a speedup claim.
Final public-pass output is byte-identical to the construction/reconstruction
driver for both modules; `summary.json` records input and output hashes. Both
native suites also pass (`authored-events-native-final.log` and
`authored-events-selected-final.log`). The full historical corpus was not rerun.

## Previous amendment: key binding consistency

After `5e0a72772`, ordinary binding now uses one ownership condition for dormant
helpers and reserved roles, including source gaps, final-read sources, relay
legs and acknowledgment allocation. Explicit checked reservation borrowing
remains available. Split returns check their next publication against the full
proposed forward/return packet before selecting a reverse key.

The ledger also indexes the latest publication component per key. Acyclic
no-next-use queries avoid rescanning the remaining graph; cyclic/shared cases
retain the complete query. See [implementation and regressions](docs/designs/oahs-key-binding-consistency.md).
All 25 portable suites and both native suites pass. The targeted native recheck
covers the 19 compatibility inputs and four partial-attention modules: **23/23
construct and reconstruct; 22 plans are byte-identical to `5e0a72772`**. Qwen
`topk_select` differs only in event-ID assignments (no payload, position or
command-count change). Results are in
`../sweep-followup-work/key-binding-native-cases.log`. The full 88-module corpus
was not repeated for this amendment.
This amendment does not enable the KDA experiments or claim device speedups.

## Sweep follow-up: compatibility first

See [exact reproductions, fixes and remaining gates](docs/designs/oahs-sweep-followup.md).

The instruction-model policy has been corrected: OAHS directly consumes
`PTOIRTranslator::Build()` nodes, as existing InsertSync does. `describeSemantics()`
is an optional audit, not an instruction-admission gate. Missing registrations
contribute no additional sync effects; the original instructions remain in IR.
Origin refresh follows the translator's fallback for incomplete optional metadata.
The extra A3 notification admission code has been removed. The shared `tdivs`
effect correction remains.

Exact sweep recheck: **19/19 construct and reconstruct**. All six later
constructor failures are fixed locally. See [diagnosis and mechanisms](docs/designs/oahs-constructor-compatibility.md).
Refined occurrences now use complete contextual states before endpoint selection;
ordinary shared words check participation and actual rearming; alternative
consumptions can return their knowledge at the existing publication gap.
Strict source-time key checks, cold checking and reconstruction remain active.
No instruction-specific rule or unlimited-event assumption was added.

Artifacts: `../sweep-followup-work/constructor-final-*` and the corresponding
`translation-policy-cases/` / `translation-policy-corpus/` outputs. Both native
suites and all 25 portable suites pass. The 88-module corpus constructs and
reconstructs: 84 plans are byte-identical, four partial-attention modules
(`prefill_fwd__44`–`__47`) change in their AIV functions. Each adds 15 static
SET/WAIT pairs, with unchanged barriers; AIV replay grows 610,965 → 616,865.
Their ordering and device-performance comparison remains open; this is a
compatibility repair, not a synchronization-quality improvement claim.
These are host construction/reconstruction results, not complete model builds
or device tests. The earlier 13/19 report predates these constructor repairs.

- `hc_head_reduce` now passes using two joined-consumption returns (two staged
  checks, 1,568 analysis-site evaluations), while retaining the earlier key-3
  reservation borrowing. The six errors were not proof of event-ID exhaustion.
- Qwen AIV now completes with 850,615 replay evaluations in the exact decode
  input. Full occurrence evaluation has a compile-time cost; reducing it while
  preserving these states is follow-up work, not a claimed speedup here.
- KDA first-write + final-read now constructs/reconstructs after protecting
  dormant helper ownership and checking the next publication when choosing a
  reverse key. Its quality gate still fails: 1784 relations removed, 706 added.
  Keep the experiment opt-in and preserve the early gap while diagnosing additions.
- Exact RMSNorm/hc_pre local order comparisons identify broader entry/return
  prefixes; scratch aliases matter. Compressor attribution remains component-level
  work with the finite-25% kv validation limit explicit.

Prior compatibility checkpoints and their corpus comparisons are historical;
see the dated sections of the sweep follow-up report. Current acceptance does
not require byte identity: changed plans need separate ordering and cost review.

## Latest implementation: final-read sources in ordinary construction

The opt-in `--final-read-sources` path now selects a final-only source before an
anchor's shared word. It uses the existing previous-reader frontier to require
an actual overwrite deadline, qualifies original final-visit control, checks
source-time coverage and key credit, and proves matching plus actual rearming
across repeated entries. The storage view is shared with first-write analysis.
The ledger protects the publication prefix by endpoint identity. Existing
acknowledgments are retained by this binding; no new deletion trial is added.

See [implementation and evidence](docs/designs/oahs-final-read-sources.md).
The linked constructor fixture covers single/repeated visits and virgin/reused
keys; 16 finite comparisons add no payload order. Native step-64 and single-visit
cases select the source and reconstruct. All 25 portable suites pass.

Full first-write KDA now constructs with the final-read source after the
reservation/neighboring-use amendments described above. Keep both experiments
opt-in: the latest local comparison removes 1784 relations but adds 706. The
next KDA task is to remove those added paths while preserving the early release
and actual rearming; no new device comparison is ready.

Corpus byte identity is not an acceptance requirement for the experiment.
Changed plans can be improvements: check safety, ordering sets, event resources
and construction cost. The initial sweep's three changed plans selected no final
source; follow-up qualification avoids refining inputs with no overwrite deadline
and avoids changing replay mode when no final gap exists. Final sweep results
are in the implementation note.

## Current result: cross-control rearming and complete KDA plan

The local first-write experiment now completes **construction and exact native
reconstruction**. It qualifies the return after consumption at cut 400 to the
early publication at cut 724, checks the return together with its actual forward
receipt, and matches shared-word consumption occurrences for later F7 repairs.
A native emission fix preserves distinct same-word event generations instead of
signature-deduplicating the checked commands. Existing-pass emission is unchanged.

**Quality gate failed:** against committed `7c48f4ab3`, the 1,727-payload finite
KDA case removes 1,512 relations but adds 634 (including M completion → MTE2
load). Keep the experiment disabled; do not send a new timing task. The goal is
now to remove these added paths, not merely achieve resource admission. Two
helper-pruning attribution variants did not resolve the issue and were reverted.

25/25 portable suites and native selected tests pass. All 88 default corpus
modules pass with byte-identical plans; default KDA also remains byte-identical. The phase-support note records the full outcome and artifact
paths, superseding the historical cut-720/724 stopping points below.
All work remains uncommitted; no device task for this experiment was issued.

## Added-path diagnosis (latest local work)

The first added path is M payload 12 -> M→MTE1 acknowledgment -> MAT release
MTE1→MTE2 -> DMA payload 14. A linked supplied-plan regression preserves every
acknowledgment but moves the final-reader release before it: 2/3/4 visits pass
causal and independent rearming checks with strict payload-order inclusion.
Deleting the acknowledgment instead fails rearming. This is a boundary witness,
not a new native plan; the 634-added gate remains open.

Source-corridor and relaxed online-helper diagnostic edits were reverted. The
first changed no native commands; the second lost a later recurring consumption
path at cut 330. Next: qualify the last participating read for ordinary source
selection at the pre-acknowledgment gap, including top-level step-64 children and
their conditional suffix. Do not remove helpers based only on immediate replay.
See the phase-support note's latest section and `release-ack-*` artifact logs.

## Latest commit and next investigation

`7c48f4ab3` commits the validated choice-frontier mechanism without the unfinished
first-write experiment. The isolated source passes all 25 portable suites and
the rebuilt native selected suite; its KDA output matches the previous checked
candidate byte-for-byte. The first-write working edits are restored separately.

[Phase support investigation](docs/designs/oahs-kda-phase-support.md) now records
an implemented linked shared-word repair. The old constructor fails the reduced
case; a structural query over every consumption occurrence now certifies an
existing later reverse receipt or inserts a source-time-qualified return after
the new consumption. The early forward publication stays unchanged. No full
trial solve is used by this repair. All 25 portable suites and the native selected
suite pass; bank-A/bank-B checks pass four paths / 16 deadlines, no added order.

Full KDA with the disabled first-write option passes the old cut-428 failure,
then fails resource admission at cut 720. Keep the experiment disabled. Default
KDA output is byte-identical to `7c48f4ab3`. The 88-module comparison initially exposed four changed attention AIV plans.
The repair is now gated by the disabled first-write option; targeted reruns of
all four restore byte-identical baseline plans. The other 84 had no query or
plan changes. Logs: `shared-rearm-corpus/` and `shared-rearm-gated-corpus/` under
`../kda-first-write-work/`. All 25 portable suites and native selected tests
pass again after the admission correction. These new repair changes are uncommitted and unpushed.

## Latest local reservation-reuse milestone

The disabled first-write experiment now passes cut 720 by borrowing recurring
MTE2→MTE1 key 0, retaining SET at word 418 (occurrence 720) and WAIT at word 444
(occurrence 746). Endpoints 340/341 are the only endpoints for this transfer;
no new acknowledgment is needed. Reservation ownership remains intact.

Admission checks actual source-time credit at every occurrence, complete owner
materialization, absence of owner/other key uses during the borrowed interval,
and an actual return before every next publication across repeated entries.
Lazy/fixed ownership remains excluded. Linked tests cover repeated visits,
incomplete ownership, overlap and missing return, with finite ordering checks.

Native KDA stops next at cut **724**, consumer **750**. Key 0 is correctly full
there because the cut-720 transfer has not yet been consumed. Key 1 is still
empty but lacks consumption knowledge from endpoint 313 at cut 400. Keep the
qualified cross-control key-1 repair as the next alternative; do not move either
early source gap or treat target-time emptiness as source credit.

All 25 portable suites pass. Native validation and default comparison are
recorded below in the phase-support note. Artifacts: `reservation-*` and
`kda.reservation*` under `../kda-first-write-work/`. No full first-write KDA plan
or device improvement is claimed. This work remains uncommitted.

## Active KDA goal and acceptance boundary (2026-09-21)

User priority: develop an efficient, general lifetime/occurrence mechanism and
look for overlap beyond existing InsertSync, rather than merely match its
latency or event population. Work on this rebased branch; do not import the
other branch's mixed-producer relaxation or global correspondence guard.

The uncommitted first-write experiment separates a first-only receipt gap from
ordinary shared post-access publication words. The bank-A/bank-B regression
preserves all payload ordering on four finite paths (16 bank deadlines), and
the native selected suite passes. Full KDA is **not accepted**: the latest
earlier experimental staged-rearming version failed consumption-before-publication
at cut 428. That version used 12 staged rearming checks visiting 152,103 sites. These extra full
analyses are diagnostic scaffolding, not the intended efficient final policy.

Next: unify occurrence matching and actual key-rearming support while preserving
early source gaps. Require a useful native plan with no added local payload
ordering, measure preparation/replay cost, and compare against current existing
InsertSync on identical payloads before asking for device timing.

A focused current-source existing-pass driver now generates
`../kda-first-write-work/kda.existing.current.pto`. Both arms use the same archived
KDA input with only eight obsolete scalar-load spellings migrated to current
syntax. The ordinary finite checker stops on an unproved M-to-MTE1 key-0 rearm
in existing; establish the applicable target/model contract before interpreting
that as a correctness defect or reporting a qualified order comparison.
Inspection now identifies the first existing-plan failure at generated line197:
a loop-tail MTE1 consumption is followed by a new M publication with the same
key, without a modeled return to M. Existing allocation checks nonoverlapping
SyncIR intervals; direct flag lowering adds no acknowledgment. This occurs before
the first queue push. The reduced boundary fails and a real reverse receipt
repairs it in the finite model. This is not a demonstrated device failure or
proof that the previously tested binary has the same sequence. Details:
`../kda-first-write-work/existing-rearm-investigation.md`.

Artifacts and logs: `../kda-first-write-work/` (build-existing.py,
existing-driver.cpp, kda.split-rearm.log, split-rearm-suite.log). No current
KDA speedup or device qualification is claimed.

## KDA choice-frontier implementation — host complete

[Local opportunities](test/benchmarks/kda_projection/LOCAL_OPPORTUNITIES.md)
and [design](docs/designs/oahs-choice-consumer-frontiers.md) supersede the earlier
diagnostic-only status. Ordinary construction now preserves an earlier source
prefix across a qualified original choice whose two first consumers both need
it. This is enabled by default, with `--no-choice-consumer-frontiers` for a
matched control. It does not enable the unfinished first-write experiment.

KDA construction/reconstruction passes: two boundaries improve, removing 32
finite explicit payload relations with none added. Payloads and guards remain
unchanged. Two staged checks cost 5,144 site evaluations; replay increases
184,538→189,813. All 25 portable suites and both native suites pass. All 88 paired
historical corpus modules construct/reconstruct with byte-identical plans.
Artifacts: `../kda-first-write-work/choice-corpus-final/` and
`kda.choice-final.{pto,log,order.json}`. No device result is claimed.

The separate MTE1-barrier deletion diagnostic verifies but removes zero payload
relations. It is an overhead lead, not demonstrated lost overlap, and is not
installed. MTE2/M group deletions fail. Next quality work should target complete
MAT cycles or qualified first conflicting writes with actual rearming support.
Do not copy the other KDA branch's mixed-producer relaxation/global guard.

## Rebase compatibility repair (2026-09-21)

The rebase at `9f2063c22` completed and contains `origin/main` at
`6d744afb5`. Native compilation exposed obsolete PTO `IntToPtrOp` and
`PtrToIntOp` references in `SyncOriginClosure.h`. Upstream now represents
both directions with `CastPtrOp`; qualifying the names with `mlir::LLVM`
would target different operations and is not the repair. Origin closure now
follows pointer–integer–pointer `castptr` round trips to the original pointer,
matching the shared translator. A native regression requires both entry and
backedge storage roots to survive a round trip of a loop-carried pointer.

Validation: rebuilt all 21 OAHS/InsertSync objects and linked both native test
executables against a dedicated current-source sync archive and rebuilt
PTOIR/PTOAnalysis libraries. Both suites exit 0, including the new regression.
This is focused native sync validation, not a completed full backend build.
Evidence: `../kda-first-write-work/rebase-{native,selected}-test.log`.

The unfinished first-write experiment is parked, not included in this repair:
`../kda-first-write-work/parked-first-write/` contains tracked/untracked file
manifests and saved contents. Resume it on this rebased branch after delivering
the compatibility fix; do not import the other branch's mixed-producer change
or global correspondence guard.

## Relay binding before ranking (2026-09-21)

Committed the preceding relay correction and campaign audit as `91fc0e728`.
The [binding amendment](docs/designs/oahs-relay-binding.md) now probes both
physical legs at their exact gaps before ranking. Previously the preferred M
route could fail binding and hide a usable MTE1 route. First- and second-leg
occupied-key regressions reproduce failed construction on the frozen old core;
both now select the alternate route with one staged solve. Tests also cover
empty-but-unknown consumption, intervening use, and a reused alternate key with
actual return credit. No hypothetical first-leg credit is granted to the second.

All 24 portable suites, both native executables and twelve attention graph
cases pass. All 88 corpus modules / 97 functions remain byte-identical to
`91fc0e728`. Replay is unchanged at 3,262,799; four staged relay solves still use
2,636 site evaluations. Key queries increase 823→835 (34→40 on each affected
AIC). Evidence and frozen old driver/core: `../relay-binding-work/`.

This is an admission improvement on discriminating fixtures, not a native
speedup or a claim that the feasible alternate dominates an unavailable route.
The new binding amendment remains uncommitted. No new device task is required.
Next: use the KDA packet for a concrete native miss when it arrives; otherwise
require a discriminating opportunity before extending second-leg receipt credit
or scoped resource ownership. Existing device comparisons retain their pins.

## Joint-reader device result received (2026-09-21)

The [audited down_proj campaign](docs/designs/oahs-joint-reader-device-intake.md)
compares `2a130aefe` with `a0c1d761e`: all 12 correctness cases / 24 launches
pass, with identical paired output hashes. Local archive verification and graph
replay reproduce 214 removed relations, none added and unchanged sync counts.
Performance is unresolved: candidate ratios 0.9085/0.9119 are accompanied by
identical-baseline control ratios 0.9123/0.8618 on devices 0/1. Device correctness
is complete; qualify the timing method before further performance repetitions. This does
not validate the separate coupled attention candidate. Raw intake lives in
`../joint-reader-device-work/`; no new device run was performed locally.

## Bounded relay-selection correction (2026-09-21)

On `bec7dfe37`, [relay selection](docs/designs/oahs-relay-selection.md) now
compares incidental receiver history and newly gated intermediate work/selected
endpoints separately. Current required or acquired completion is useful only
with occurrence freshness. Incomparable views keep the deterministic incumbent;
no relation-count score or new mode is introduced. A later receipt also cannot
be prepended into an earlier selected forwarding word to the same receiver when
it would broaden that receipt. Actual staged protocol/key checks remain intact.

Both linked witnesses are addressed: the avoidable case selects five pairs and
30 full relations instead of 34; the incomparable case keeps the late route
without claiming dominance over the old early route. Added tests distinguish
pending from acquired receiver credit and outward SETs before/after a supporting
WAIT. A pending-credit case selects one composed transfer, avoiding a separate
required transfer. Twenty-one balanced deletion negatives lose memory support.
This remains a bounded placement heuristic, not the general frontier-motion
or open-interface certificate.

All 24 portable suites, the expanded relay suite, both native test executables
and all twelve finite attention cases pass. All 88 corpus plans / 97 functions
remain byte-identical to `b2d8482fc`, preserving the existing attention and
joint-reader improvements. Aggregate replay stays 3,262,799; each affected AIC
retains two staged solves / 1,318 evaluations. Shared corridor scans reduce
relay-preparation site visits 1,434→1,020; history comparisons remain additional
work, so this is not a compiler wall-time claim.

Evidence and frozen baseline driver: `../relay-selection-work/`. No new native
plan or device speedup is claimed; the four device tasks keep their pinned arms.
Next local options: a discriminating physical-binding feasibility case before
ranking, or a real missed lifetime from the requested KDA artifacts. Second-leg
credit from the first receipt and general contextual motion remain open. Do not
expand the policy solely to improve synthetic command counts.

## Attention device handoff prepared (2026-09-21)

The missing [coupled FIFO/relay device task](test/benchmarks/attention_relay/DEVICE_TASK.md)
is now written. It pins baseline `d1bf07ee5` and candidate `21f95f9b7`, exact
row48/49 inputs and PTO-ISA `0c112d61`. The task includes investigating the
reported separate-ring/shared-slot mismatch and building/qualifying an isolated
adapter; the device agent need not ask for a separately supplied harness.
It preserves the working runtime and requires authentic coupled numerical and
queue-progress gates before two rotated timing rounds (20 samples per arm).
Available devices may run independent matched blocks. Status is **prepared**;
dispatch/current remote execution is not confirmed by this local change.

## Return-query hoist and relay reproductions (2026-09-21)

On committed base `a0c1d761e`, the repeated return-sharing propagation is now
computed lazily once per Y candidate, after the first supporter passes cheap
filters. No selected endpoint or relay policy changes. The linked scaling
fixture reaches actual candidate formation: 32 candidates use 31 queries and
24,330 visits instead of 496 queries and 389,205 visits. Selected requests remain
identical. See [implementation and evidence](docs/designs/oahs-return-query-relay-review.md).

Both review relay witnesses now reproduce through the linked ordinary
constructor, independently checked by `GraphOracle.h`:

- Independent middle work: two pairs and 15 relations in both plans, but four
  relations removed and four added. The orders are incomparable.
- Required receiver history: the constructor selects five pairs / 34 relations;
  a supplied alternative keeps five pairs and removes four relations, adding
  none. The extra reader completion was already required at the receiver.

Thirteen balanced complete-leg deletions fail memory coverage. These fixtures
provide a qualified portable slot view; they are not native FIFO import or
coupled-device runs. Keys are virgin, so nontrivial rearming remains a separate
boundary. The tests characterize the current heuristic deliberately; update
their selected-plan expectations when correcting it, preserving the independent
model comparisons.

All 24 portable suites and both native test executables pass. The positive
native sharing plan matches the frozen base byte-for-byte; its private/shared
control retains 602 relations and 10→7 pairs. All 88 corpus plans / 97 functions
pass construction/reconstruction and remain byte-identical to `a0c1d761e`.
Aggregate replay stays 3,262,799; no corpus case reaches a sharing query.
Evidence, frozen baseline library/driver and commands: `../return-query-relay-work/`.

The bounded relay-selection follow-up is now implemented above. Positive binding
before ranking and first-receipt support for second-key reuse remain separate
extensions. Preserve the native attention control and pinned device sources.

## Joint first/final reader implementation (2026-09-21)

The user authorized implementing the [diagnosed down_proj opportunity](docs/designs/oahs-blocked-down-last-reader.md).
[Joint reader qualification](docs/designs/oahs-joint-reader-prefix.md) now collects
B0 first-consumer and A0 final-reader requirements before changing the original
owner. It preserves step-128 final participation, the single-visit case and the
conditional suffix's original payload/command identities. Actual selected
readiness/return cycles and the unchanged native reconstruction checker validate
completion, matching, rearming and emitted guards. No new pass mode.

Down_proj's existing A0 return is emitted after the final A0 extraction and
before the unrelated B0 wait; the same qualifier also advances other admitted
child releases. Eighteen independent native-plan traces check 73,449 conflicts
per arm, remove 214 full payload relations, add none, and preserve every path's
executed SET/WAIT and fence counts. This is compiler output, superseding the
older supplied-plan diagnosis. No numerical/device speedup is claimed.

All 23 portable suites and native construction/reconstruction checks pass.
The final 88-input/97-function corpus check passes; eleven projection plans
change and 77 remain identical. All 37 projection-family paths preserve executed
sync/fence counts and remove 1,414 full payload relations with none added;
511,187 local conflicts are checked per arm. Six emitted-plan negatives reject
missing/premature/repeated support and wrong final distance. Direct C++ lowering
and the live-pass trace regressions pass. Final evidence is under
`../joint-reader-work/corpus-final/` and the report linked above.
The final first-prefix minimization leaves all candidate plans identical to
the initial joint implementation. Final
analysis sites increase 181 ->262; replay 13,563 ->16,950. This is added ordering
precision, not a compile-performance claim.

Artifacts: `../joint-reader-work/`; baseline executable is frozen at `2a130aefe`
plus the read-only diagnostic. The first/final frontend shares suffix words and
physical operations but retains separate analytical final-continuation sites
so exit/backedge correlation is not lost. First-prefix copying stops after the
last first-consumer and its following source anchor.

Committed by the user as `a0c1d761e`; device qualification remains the
[short matched down_proj device task](test/benchmarks/joint_reader/DEVICE_TASK.md).
It uses repository checkouts, the existing qualified harness, correctness first,
10 warmups per round and 20 timing samples per arm. Existing frozen device
campaigns remain independent. Guard cost may outweigh the extra overlap.

## Local milestone: composed required returns (2026-09-21)

[Required-return sharing](docs/designs/oahs-required-return-sharing.md) is now
implemented on `21f95f9b7`. A closed reader-region cohort can use X's required
return for Y when that existing receipt precedes Y's overwrite. Separate
readiness and X's endpoint positions remain intact; selection precedes key
allocation, and the exact staged protocol must prove completion and rearming.
The global producer-support safeguard remains in force. No deletion search is
added. `--no-reader-return-sharing` is a diagnostic ablation, not a pass mode.

Twenty independent finite comparisons preserve full payload-order sets. The
reversed-deadline mutation is safe but adds four relations and is declined by
selection. Missing readiness-consumption credit is rejected even after erasing
payload effects. Empty/varying children, repeated entries, reloads, independent
readers, outward publications and chained support are covered.

All 23 portable suites, the final focused cyclic suite, both native executables,
and the new native FileCheck/order regression pass. Seven native variant pairs
pass. The positive witness keeps all 602 payload relations while reducing
executed pairs 10→7 and reserved channels 4→3. Replay remains 324 evaluations;
one structural sharing query visits 63 sites.

The 176-run paired corpus passes: all 88 plans remain byte-identical to the
frozen `21f95f9b7` executable. Its aggregate replay and trial counts are unchanged,
and no input reaches a sharing propagation. Artifacts and exact commands:
`../return-sharing-work/`. No new corpus device campaign or speedup is claimed.
The already-running device work stays on its pinned source snapshot.

Next: obtain a real retained-input/preload case admitted by this support rule,
or use the pending KDA packet to identify a concrete missing qualification.
Joint first/last observation work is now implemented above. General moving-
frontier certification and scoped producer support are still open.

## Active diagnosis: GLM KDA projection

The [initial KDA attribution](docs/designs/oahs-kda-projection-diagnosis.md)
verifies matched prepared-input hashes and localizes changed binaries to the
main AIC/AIV pair; pad/zero binaries match. Saved spans place nearly all extra
~80us in the scheduler dispatch/execution window, not the orchestrator phase.
Actual orchestration sources have different hashes and must be inspected.
The local report bundle lacks prepared plans/source, so compiler attribution
requires the [small remote artifact packet](test/benchmarks/kda_projection/DEVICE_TASK.md).
The request exports preserved files without rebuilding; a short rotated timing
confirmation is separate. No compiler change is justified yet. Raw local
analysis is under `../kda-projection-work/`.

## Wide device sweep received

Audited the report-only `oahs-full-20260921-report.tar.gz`; see
[findings and evidence boundaries](docs/designs/oahs-full-sweep-20260921-review.md).
Frozen compiler `2cc458cbe`: 48 original-entry and 77 isolated configurations
(54 normalized kernel groups). All 164 manifest hashes and 250 arm summaries
recomputed from 25,000 raw samples agree. The strongest new lead is GLM KDA
projection, 237.389 -> 316.950 us, followed by decode compressor ratio4 and
DeepSeek RMSNorm. Attribution needs the prepared/kernel artifacts from the
separate full archive. Several dramatic isolated differences have identical
binaries and cannot be compiler effects.

Qwen's authentic two-layer coupled prefill passes both arms at seeds7/23. Its
runtime may support the pending FIFO task after exact input/contract matching;
this older measurement does not validate `21f95f9b7`. The latest TODO prioritizes
these concrete misses and classifying 23 handoff-codegen diagnostic records.
No new compiler changes or device runs were made during the report audit.

## Review reconciliation after the FIFO commit

The reviews through `d1bf07ee5` are reconciled against `21f95f9b7` in the
[top-priority TODO section](docs/designs/oahs-todo.md#review-reconciliation-at-21f95f9b7-2026-09-21).
At that review, source inspection confirmed the joint first/last observation, global
producer-support and fixed guarded word-order limits. The newer FIFO and
reused-key gap implementation supersedes the reviews' corresponding absences.
A read-only prepared-input scan finds attention shape updates in AIV only;
LM head updates its real accumulator. There is no demonstrated unrelated-
descriptor ACC opportunity in those examples. The diagnosis, required-return
sharing and bounded joint first/final implementation have since completed above;
the historical implementation pause no longer applies.
This reconciliation changes documentation only; the device candidate stays
`21f95f9b7`, and no new native or timing claim is made.

## Current local continuation: source gaps, attention and grouping

User asked to investigate items 2–4 while leaving the last-reader device work
(item 1) to the running device agent. Base: `d1bf07ee5`; no change to that
agent's frozen source or task.

- **Implemented, opt-in:** [reused-key word-start publication](docs/designs/oahs-reused-source-gaps.md).
  The source state must already prove emptiness and consumption, and every
  selected use of that key must precede the gap on a qualified straight corridor.
  The linked one-key witness failed before the fix and now removes four complete
  payload relations (67 -> 63), adding none. Five boundary cases reject missing,
  late and conflicting neighboring-use evidence. `--source-gaps` stays off by
  default; no general recurring-key allocation or arbitrary interior gap.
- **Implemented in ordinary construction:** [static FIFO slots and split relay endpoints](docs/designs/oahs-static-fifo-relays.md).
  The lowering-owned envelope now covers qualified ACC sends / MAT receives.
  Independent cursor analysis preserves the shared GM root while identifying
  each static slot. The first receive gets FIX -> MTE1 -> MTE2 from QK2's push;
  QK3's push reaches the later receive through an M publication before PV0.
  Exact source gaps and actual key credit govern binding; one staged ordered
  population supplies checking and commit. Unsupported cases retain ordinary
  handling. There is no new compiler option or global relay preference.
  Both single-block variants emit/reconstruct successfully and pass six finite
  independent cases each. One native-length outer entry removes 80 payload
  relations, adds none, and uses 103 -> 109 event pairs. Removing real support
  fails memory/rearming; a safe late-forwarding mutation adds ordering.
  The previous hand-assembled prototype is now superseded. Next is coupled
  AIC/AIV device qualification, not another attempt to move the bank release.
- **Grouping boundary:** nine supplied-plan comparisons now cover outward
  publication before/between/after two waits. Before/after preserve full order;
  between broadens it. [A production motion certificate remains open](docs/designs/oahs-frontier-motion-context.md),
  including exact word positions, physical binding and later-edit invalidation.
  The existing coalescer remains unchanged; finite tests are not its certificate.

Validation: 23/23 portable suites; focused placement rerun after the additional
three export-position cases; native selected diagnostics; 16 targeted native
module runs (22 function constructions/reconstructions), default/source-gap
arms across projections, RMSNorm, attention and GEMM. All native outputs are
byte-identical to saved current plans. This is not a full-corpus rerun or a new
device result. Artifacts: `../source-reuse-quality-work/`.

Native integration validation: 23/23 portable suites; both native diagnostic
executables including five ACC/MAT slot-admission cases; seven targeted corpus
modules. Only the two single-block AIC plans change; down, gate/up, LM head,
post-RMSNorm and partial attention retain identical plans. AIV companions are unchanged. GEMM is byte-identical and passes its
200/394/782-pair trace checks. The production-word probe and missing-support,
rearming, phase and ordering comparisons are reproducible through
`test/benchmarks/attention_relay/run.py`; raw records are in
`../attention-native-work/`. Shorter/varying inner lengths are checker-CFG
stress tests, not native constant-trip executions. There is no new device result.

Two staged relay checks per admitted AIC add 1,318 analysis site evaluations;
relay preparation records 1,434 site visits. These are charged separately from
selected replay and final helper trials. The original graph stays at 239 sites.
The relay coverage ranking is a heuristic, not a general no-added-order proof.

The hardware/IR review remains mostly backlog: local-drain credit, descriptor
scoping and general address-sliced slot analysis remain absent. This milestone
implements a narrow lowering-qualified static FIFO slot view through the
existing analyses. It does not establish a new peer-progress or GM-visibility
contract, enable UnitFlag, or change the running device agent's frozen campaign.

## Completed locally: release before trailing child work

The [last-reader placement](docs/designs/oahs-last-reader-placement.md) extends
the existing reader-region cycle with a separate physical publication frontier.
A compact final/nonfinal visit projection shares original payloads and ordinary
words. Cell-use succession and balanced participation select the post-read
source only when no later read remains and it executes once per child entry.
Actual selected readiness/return transfers still supply completion and rearming.

Native admission is restricted to straight nonempty leaf children inside a
parent loop, constant nonnegative bounds, unit step, invariant input and an
inactive producer engine. Already-refined, guarded, empty/unknown and non-unit
cases keep their previous handling. No new constructor option is introduced.

Validation: 23/23 portable suites, both native diagnostic executables, nine
native admission variants and four native/FileCheck fixtures pass. Seven
portable traces remove 52 full payload relations with none added. The native
witness removes four relations, retains five executed event pairs, and keeps
the genuine V barriers. Missing support and repeated publication mutations
are rejected. All 88 corpus modules (97 functions) construct/reconstruct with
byte-identical plans to `2cc458cbe`; GEMM and attention are unchanged.

The native witness adds three analytical sites and increases replay 89 -> 130;
this is a placement improvement with an explicit construction cost. It has no
device timing yet and does not change any existing corpus plan. Artifacts:
`../last-reader-work/`, including an importer-disabled child-exit control and
independent emitted-plan ordering comparison. Keep the remote campaign on its
dispatched revision; there is no reason to restart it for identical corpus plans.

Next: obtain a useful native workload with trailing reader work before extending
this qualifier to non-unit steps or combining it with first-consumer/bank
observations. The contextual frontier-motion certificate is still open. Any
targeted device follow-up should measure the new guard cost as well as overlap.

## Local continuation: conditional key reuse

The device agent is running the dispatched `2cc458cbe` task. These working-tree
changes are a separate local candidate; do not change its frozen source or arms.

The next linked regression reproduced the review's admission issue: with one
forward key, an optional producer, a common consumption and a later conditional
producer/consumer, the closed policy succeeds but deferred acknowledgment fails
with `no reusable key or nonrecursive consumption acknowledgment`.

The opt-in policy now requires `Control::straight(current, graph.exit)` before
deferring. This reuses the original-control corridor index to ensure the
continuation stays within the current repair vocabulary. It retains the closed
exchange before a future branch, including branches that might not reuse the
key. No conditional repair, hypothetical consumption credit, new analysis
state or default-policy change is introduced.

Tests cover both choices independently, final helper trials on/off, cold
checking and independent graph checks. Removing the reverse transfers is
rejected. Straight-line deferral and actual-key-reuse tests still pass.
All 23 portable suites pass, with the final expanded placement test rerun after
adding helper-pruning assertions. Both native diagnostic executables pass.
The existing native deferred-ack fixture constructs/reconstructs under default
and deferred options; both plans are byte-identical to the saved placement
microkernel plans. No full corpus rerun or new device measurement was performed
for this opt-in admission restriction. Validation artifacts:
`../deferred-branch-work/`.

The restricted last-reader follow-up is implemented above. The general
frontier-motion certificate remains open. More precise deferred-ack admission
is parked until an actual provider example justifies it.

## Completed locally: exact proposal ordering and replay accounting

The review intake checked at `6b1b32f46` confirms that recurring precommit
analysis uses request-order words while guarded commitment partitions
publications before acquisitions. This mismatch is now fixed: one materializer
supplies the exact canonical ordered endpoints for mandatory protocol,
producer-support/resource checks, omission trials, and commitment. The linked
reciprocal witness rejects before changing the ledger or reservations; ordinary
fallback succeeds. Accepted omission trials retain their exact checked endpoints.

Certified sibling replay now skips the old prefix comparison unless tracing is
requested. `prefix_queries` and `prefix_span_examinations` account for actual
queries; `sibling_comparison` labels whether the extra-sibling statistic is
measured. Full states/aggregates agree across untraced, traced, prefix-only and
cold replay. On partial attention, 24,827 AIC and 177,875 AIV legacy span
examinations are skipped, with identical PTO and causal evaluation counts.

All 23 portable suites, native diagnostics, two retained-input native/FileCheck
fixtures and serial hooks pass. All 88 corpus modules (97 functions) construct
and reconstruct; every plan is byte-identical to `6b1b32f46`. Selected update,
replay, recurring-trial and rejection counts are unchanged. Both new regressions
fail when linked against the old behaviors. See [results and limits](docs/designs/oahs-review-amendments.md).
Artifacts: `../review-fixes-work/`. No new device measurement is needed for
these identical plans; this milestone is included in the accompanying commit.

The conditional future-key-reuse follow-up is recorded above. See the
[ordered TODO amendments](docs/designs/oahs-todo.md#review-amendments-checked-at-6b1b32f46).

The former last-reader TODO is now implemented for the restricted scope above;
general guarded participation and composition with other refinements remain open.

## Current local follow-up: complete producer cohorts

[Multi-input retained-reader admission](docs/designs/oahs-retained-producer-cohort.md)
now replaces the single-producer-write-cell restriction with a narrow closed
cohort: one straight writer corridor, complete per-cell cycles, separate source
and reader boundaries, and no uncovered producer requirement in the actual
staged ledger. An unmet support premise declines without endpoint/reservation
changes; `rejected_support` is separately reported. No new mode or solve beyond
the existing mandatory proposal check.

All 23 portable suites and native diagnostics pass. Sixteen multi-input traces
remove 164 full payload relations with none added; the original X/Y fence-motion
counterexample still declines. The native A3 witness preserves five V barriers
and removes 16 relations while executed pairs grow 6 -> 10. All 88 corpus plans
remain byte-identical to the frozen retained-reader baseline. Two observable
microkernels are prepared for a separate small device task; no device result is
claimed. Artifacts: `../retained-multi-work/`. Committed in `6b1b32f46`.

## Current local follow-up: retained reader children

[Retained-generation composition](docs/designs/oahs-retained-reader-children.md)
now reuses the existing nearest-role/control view to identify the first and last
reader child of one physical generation. A native A3 fixture reproduces the old
miss: the new plan acquires once, releases after both children before unrelated
V work, and preserves all three V barriers. Eight portable traces remove 62 full
payload relations with no additions; reload, independent-reader, missing-support
and premature-release negatives pass. All 23 portable suites, native diagnostics
and the native pass/FileCheck test pass. A serial 88-module native sweep passes
construction/reconstruction with all plans byte-identical to the committed
sibling-replay baseline. Included in `6b1b32f46`.

A new negative limits admission: removing an X fence can move an uncovered Y
fence after the new X write. That initial extension required a single producer
write cell. The closed-cohort qualification above now admits a bounded multi-input
case without leaving the producer repair pending. Read-only GM inputs are allowed. Artifacts: `../retained-generation-work/`.

## Current local follow-up: contextual frontier-motion witness

The [common-frontier regression](docs/designs/oahs-frontier-motion-context.md)
now distinguishes a private common consumer from an outward publication between
its two readiness acquisitions. Six finite contexts (one/two/four episodes)
compare complete payload-order sets and check real return/rearming paths. Both
plans can be safe while the merged plan adds B-load -> unrelated R-reader order.
A linked two-layout qualifier probe does not reproduce the bad order: its
outward role precedes both readiness roles. All 23 portable suites pass.
This is supplied-protocol evidence, not a reproduced native constructor defect.
The next step is exact proposal/word-order qualification before enabling a
narrow moving-frontier certificate. Production placement is unchanged.

## Consolidated open device campaign

The user confirms the planned overnight sweep did not run. The new
[master device task](test/benchmarks/open_experiments/DEVICE_TASK.md) supersedes
older dispatch instructions. It covers broad corpus/model measurements, the two
retained-input cases, retained-A and cross-tile-preload references, resumed UF=0
manual attention, authentic coupled Qwen attention if available, and a low-priority
no-motion GEMM control. Completed MAT/placement/barrier-isolation campaigns stay
closed. First queue ready work on all free devices; do not wait for transcription.

The self-contained archive is `../device-handoffs/oahs-open-experiments.tar.gz`,
with a detached checksum. It supplies a current compiler tree including these two
review fixes, immutable older input/harness bundles, pinned reference sources,
evidence and dispatch instructions. The root compiler overrides stale compiler
pins in the enclosed tasks (except the retained-input historical baseline).
Use hashes to identify the snapshot, not its base commit alone. The task requires
10 warmups and 20 measured invocations per arm, a durable coordinator, all free
cards, progress/job records and an eight-hour cap including final packaging.
Preparation is not dispatch or evidence that a remote job started.

## Remote scheduling: all eight devices

**Repetition update:** retain 10 warmup launches, then use 20 measured launches
per arm total in two rotated rounds. No nested large batches or timing for every
correctness seed. Keep completed evidence; increase sampling only for a specific
unresolved decision. This overrides the old 100–180-sample task guidance.

Use all eight allocated remote devices for independent matched comparisons;
serialize timing/profiling only within each device. Keep each comparison's arms
on the same device and report results by device. See the
[scheduling addendum](docs/designs/oahs-eight-device-scheduling.md), which should
be sent alongside the immutable MAT/placement archives. Their contents and hashes
remain unchanged. Remote CPU/reference work uses the remote host's own capacity.

## Completed placement campaign and barrier isolation

See [placement intake](docs/designs/oahs-placement-device-results.md) and
[conditional MTE2 isolation](docs/designs/oahs-conditional-mte2-isolation.md).
The barrier-only follow-up completed on card0: 12 runs/48 launches, no timing.
Existing fails three taken-path seeds; adding only the conditional MTE2 barrier
repairs all three; OAHS default passes. False-path controls all pass. The archive
hash and all 32 manifest entries verify, as do exact one-line PTO/C++ mutations,
raw results, matching inputs and build/hash records. Artifact binaries are not
included in this smaller archive. Audit: `../mte2-isolation-review/`.

The earlier dedicated failure-probe and row22 correctness/hash logs are recovered.
The 264-row matrix belongs to MAT; placement has 36 attempts: 35 successes and
one expected default class-invariant refusal. No-motion GEMM was host-only by
design, with no device job or samples. Its status is closed, not an overdue run.
The original placement archive remains immutable; its local standalone report
has no appended erratum, but the isolation report records the correction.

No campaign jobs remain. Keep the existing failing arm excluded from timing.
The next separate correctness task is a local existing-pass conditional-WAW
regression and source diagnosis; OAHS already has the required barrier. No new
device sweep is needed for this isolation. Local retained-generation work above
continues independently.

## Completed sibling replay work

The placement/admission device campaign is complete. Keep its source
snapshot fixed. The [local TODO order](docs/designs/oahs-todo.md#local-work-after-the-placement-campaign)
now has concrete deliverables and exit gates. Local opt-in `--trace-replay`
attributes FIFO work without changing replay or emitted plans. See
[FIFO replay attribution](docs/designs/oahs-fifo-replay-attribution.md): 99.5% of
AIV replay evaluations occur in two alternative loops. Sixty-four edits confined
to the first branch still spend 202,944 evaluations in the unmodified sibling.
One edit touches both loops and remains distinguished. Certified reuse beyond a
topological prefix is now implemented: invalidation closes under original
successors and all occurrences of touched nonempty words, preserving exact
incoming state and complete endpoint aggregates. Partial AIV drops from 813,458
to 608,848 replay evaluations (25.2%), at the same 125 updates; AIC is unchanged.

A fresh paired corpus sweep passes all 176 module runs (88 inputs, two modes),
with byte-identical PTO between modes and against `9f30`. Across 97 function
instances, replay decreases in 12, is unchanged in 85, and increases in none.
Besides partial AIV, reductions include single-block AIV (22.5%), QKV (13.6%),
RMSNorm (8.1–10.2%) and top-k (24.1%). Total causal evaluations fall from
4,085,803 to 3,236,543; the added 529,295 site/550,344 edge/81,370 word-occurrence
visits are separately charged. See the attribution note for the complete table. The rebuilt portable suite passes 23/23 tests, including full cached/cold
state comparisons for alternative/sequential loops, shared words, real events,
failed edits and recovery. Native diagnostic tests pass. The dependency walk is
counted separately; `--prefix-replay` retains the old rule for comparisons.
Artifacts and serial reproducer: `/home/toni/work/pypto3_sync_more/sibling-replay-work/`.
Initial attribution remains in `fifo-replay-work/`. The completed device snapshot
is unchanged; identical selected plans need no new device timing for this change.
Three matched serial host rounds on partial attention give 24.568 s prefix-only
versus 18.304 s sibling-reuse median diagnostic wall time (paired reductions
19.8–25.5%). This measures local construction/reconstruction, not device latency.

The next local priorities are a contextual frontier-motion certificate and a
real corpus witness for the bounded multi-input retained-generation interface. Further replay work must target the
remaining fixed point within an edited loop; sibling reuse does not solve that.

## Completed MAT family device campaign

See [final results and local audit](docs/designs/oahs-mat-device-final-results.md).
All five projections beat existing: down14.3%, gate/up16.2%, KV14.3%, Q/out15.2%,
LM4.4%. Candidate/control differences are unresolved; restoring MTE2 fences
retains the gain. GEMM baseline/candidate timing binaries are identical.
The agent reports 189 final correctness passes; all 264 host rows pass.
Locally verified: 4,072 internal checksums, 52 library/build-record hashes and
raw timing medians/IQR comparisons. The final archive's embedded outer checksum
is stale; its detached remote checksum has not yet been supplied locally.

The completed campaign used seven devices because card0 had a lock problem.
Keep per-device matched comparisons and use all eight when available for the
new microkernel tasks. These results qualify MAT cycles, not the new opt-in
experiments. LM/GEMM matched profiles were not delivered; the remaining claimed
"residual gap" is not an identified regression against existing.

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

Final family device feedback above confirms the latency improvement across
all five measured projections. The complete protocol also proves the MTE2
fences unnecessary before insertion.
M/FIX/ALL fences remain. A separately certified placement control restores all
baseline MTE2 fences; it removes 6,173 relations with none added. Measure this
control alongside production, fe1 and existing. The down result is recorded below.
The existing dispatch package preserves the precommit working-tree snapshot.
The implementation is now committed as `8afb90f17`; keep that package and its
hashes fixed as the completed campaign reference.

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

The completed placement results are summarized above. The separate
MAT campaign now reports down_proj at 25.792 us versus 30.077 existing and
31.646 fe1, with 36/36 correctness passes. The retained-fence control is
25.704 us: MTE2 omission has no measurable benefit in this comparison. See
[measurements, attribution limits and pending work](docs/designs/oahs-mat-reader-cycles.md#device-feedback-down_proj-2026-09-20).
The final family results supersede that preliminary update; see the completed
campaign section above. Keep both packaged source snapshots fixed.

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
