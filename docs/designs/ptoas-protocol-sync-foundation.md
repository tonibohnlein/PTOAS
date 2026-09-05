# PTOAS ProtocolSync semantic foundation

## Decision

ProtocolSync is a new synchronization planner developed from PTOAS `main`, not
an extension of the mutable InsertSync planner and not a continuation of the
CanonicalSync demand-cover implementation. Its governing principle is:

> Protocol-first, obligation-complete.

The implementation starts from base
`75e4a224d45bb81b7101df97edd1e4a98c0e1b9d`. The frozen research reference is
`codex/canonical-sync-hardware-graph` at
`19674e06ff25be87badf0cb3595e014e443b558d`; it remains an oracle and source of
narrowly reusable contracts, not a branch to cherry-pick wholesale.

## Checkpoint A boundary

The first checkpoint is diagnostic only:

- shared operation summaries record physical phases, memory effects, storage
  provenance, slots, queue roles, fixed supply, and event reservations;
- `StructuredSyncIR` records immutable regions, phases, accesses, and program
  points with stable pre-order IDs; phase cores come from normalized physical
  sections, with function/pipeline fallback only when no section is present;
- the existing InsertSync translator remains the production implementation;
- a shadow adapter compares the new schedule with the legacy flattened
  structure, legacy core class/pipeline, definitions/uses, macro phases and
  completion, slot/depth facts, hidden reservations, and queue semantics;
- the analysis pass is disabled by default and never emits synchronization.

The pass runs after memory planning and reserved-buffer/identity-move cleanup,
but before `PTOResolveBufferSelect`. At this point planner-assigned physical
addresses and `pto.multi_tile_get` selector/depth information must coexist.

Use:

```text
--protocol-sync-analysis-only
--protocol-sync-one-shot
--protocol-sync-ready-release
--protocol-sync-direct-repair
--protocol-sync-mixed
--protocol-sync-dump=schedule|channels|lane-frontiers|storage-tracks|residuals|plan
--protocol-sync-statistics
```

The dump and statistics options require one ProtocolSync analysis or emission
mode. The `plan` dump is emission-only. Registering the pass or compiling
without these flags does not run a preparatory walk and does not change the
default pipeline.

Statistics are emitted as one JSON record per function. The schema includes
work counters and microsecond timers for all later stages from the outset;
storage-generation, channel, planning, allocation, materialization, and
verification fields remain zero while those stages are outside Checkpoint A.

During this shadow checkpoint, the semantic context imports storage provenance
from the legacy translator. ProtocolSync records are independent of legacy
types, and a later reviewed change will replace this transitional source with a
shared provenance provider before legacy InsertSync consumes the new frontend.

## Checkpoint A implementation map

- `ProtocolSync/SyncSemantics.h` and its implementation own shared operation,
  phase, access, slot, queue, completion, failure, and statistics records.
- `InsertSync/LegacySyncIRAdapter.h` owns the only boundary that exposes legacy
  SyncIR types; it constructs the transitional provenance context and performs
  shadow comparison.
- `ProtocolSync/StructuredSyncIR.h` and its implementation own construction,
  freeze-time validation, and deterministic schedule dumping.
- `ProtocolSync/PTOProtocolSync.cpp` owns the analysis-only pass and its JSON
  diagnostics.
- `Passes.td`, the PTOAS CLI, and the main lowering pipeline provide opt-in
  wiring at the audited pre-`PTOResolveBufferSelect` pass point.
- Focused lit tests cover IR/C++ output identity, physical interval plus slot
  preservation, structured control, queue depth, macro reservations, and
  deterministic rejection of an unsupported effectful operation.

## Research evidence retained

The consolidated [experiment and related-work evidence
ledger](ptoas-protocol-sync-evidence-ledger.md) records the C.5-C.7 hypotheses,
methods, measurements, claim boundaries, reproduction artifacts, and primary
literature comparisons. The per-experiment reports remain the detailed source
for lane-pattern and storage-track construction.

The four archived CanonicalSync campaigns remain external evidence:

- `d99520397`, SHA256
  `ab105c5729f0bf556a8b3b1bf72d8f41d5b3924ab986c943629ee9ec163e30f3`:
  opt-in registration must not affect the default pipeline; unsupported input
  must not crash or mutate.
- `0ec958f1e`, SHA256
  `ee99830218e5f7e5796524a29aec43963559828a1ad74208117b41ddd15c78cd`:
  collect stage/work counters from the first diagnostic implementation.
- `137e6282`, SHA256
  `aba7ba1bb735aa1b351d41b409c9193dbf40c2f304e1eb23396275ce11d2a9f9`:
  optional protocol failure must not reject an otherwise repairable function.
- `19674e06`, SHA256
  `d327cd32ebf132117b22594e0719b805f2fb80e03b5423a2411aae1e180d6218`:
  shared frontiers reduced static pairs on a tested A3 subset, but did not
  prove scarcity or performance.

The archives are not committed into PTOAS.

## Hardware and rollout limits

- Generated cross-core AIC/AIV synchronization is unsupported initially.
- Event-direction legality does not establish MTE3-to-MTE2 same-address GM
  publication; that capability remains fail-closed.
- A2 and A3 use one simulator-qualified NPU 2201 capability profile. A3 has
  additional silicon qualification; no A2-silicon result is claimed.
- Existing function/section drains are production policy and are not removed
  by the diagnostic pass.
- ReadyRelease was enabled only after purpose-built fixtures proved logical
  lane depth, selector evolution, reuse distance, and repeated A3 device
  execution.
- Future mutation must occur on a clone and commit only after independent
  semantic and MLIR verification.

## Checkpoint B result

The audited sequence is memory planning, `PTOResolveReservedBuffers`,
`PTORemoveIdentityTMov`, ProtocolSync analysis, legacy InsertSync, then
`PTOResolveBufferSelect`. No additional IR descriptor is required at the
ProtocolSync point: planned local byte intervals, allocation roots,
`alloc_multi_tile` counts, `multi_tile_get` selectors, queue handles/depths,
macro phases, and physical pipelines are simultaneously available.

`StructuredSyncIR` now assigns stable storage-family IDs by allocation root and
address space. Each family preserves its merged planned intervals, logical slot
count, visibility role, and physical/unknown-range state. Slot selectors retain
their original SSA value and are conservatively canonicalized as a constant or
`(iv + offset) mod N`; unsupported expressions remain unknown. The frozen form
uses the loop's logical iteration coordinate: `coefficient` includes the
`scf.for` step and `offset` includes its lower bound. The diagnostic distance
API proves the official depth-two schedule has different adjacent slots, the
same slot at distance two, and reuse distance two. It never reports a period
beyond the explicit search bound.

`protocol_sync_information_preservation.pto` covers one-buffer, double-buffer,
offset double-buffer, constant-slot, and nonlinear-unknown schedules. The
existing analysis-only fixture also proves queue identity and depth survive at
the same point. Protocol generation, `ReadyRelease<2>`, and event allocation
remain disabled.

## Checkpoint C result

`PipelineStageAnalysis` conservatively creates one immutable discovery stage
per exact physical phase, so stage diagnostics cannot merge across a resource,
control, publication, or generation boundary. Roles come from physical memory
effects: the official path is recovered as `copy-in → compute → copy-out`
without operation-name matching.

`StorageTimelineAnalysis` groups exact local access slices by storage family
and canonical slot expression. A physical-address-space sweep also catches
overlap between distinct allocation roots. Each admitted symbolic generation
records its producer and consumer stages, publication, acquisitions, guarded
final uses, and—when loop-carried—the first proven same-slot overwrite. An
interval sweep detects partial overlapping access classes. Unknown ranges or
slots, ordered and in-place accesses, unknown aliases, cross-root overlap,
multiple same-slice lifecycles per iteration, missing endpoints, incompatible
loop domains, and unproven reuse remain deterministic diagnostic rejections.

Strict channel analysis admits only physical local storage whose allocation
root proves capacity one or two, one producer, one consumer, no unresolved
semantic failure or branch/nested-loop control, and a single physical core.
Loop-carried channels additionally require the proven reuse distance to equal
capacity. The `channels` dump reports one-shot and ready/release channel shapes
plus every timeline/channel rejection. On admitted focused fixtures, a root-indexed
exact-storage query over the legacy translator's memory-dependency records and
loop structure confirms the ready RAW and recurring release WAR relations.

The official one-buffer fixture produces two one-shot channels. The official
double-buffer fixture produces two capacity-two ready/release channels with
publication, acquisition, final-use, next-overwrite, and reuse-distance-two
frontiers. Focused negative tests cover extra consumers, guarded final uses,
partial and cross-root overlaps, in-place access, repeated same-slice loop
lifecycles, opaque effects, intervening branches, constant-slot
underutilization, nonlinear selectors, and nested loops. Statistics now report
stage counts, attempted and admitted timelines/channels, stable rejection maps,
and their stage timings.

No synchronization, protocol candidate, event allocation, or IR mutation is
performed in Checkpoint C.

## Checkpoint C.5 lane/frontier experiment

The read-only experiment projects every physical phase onto an execution lane
identified by `(physical core, pipe)`. Each projected occurrence retains its
region, guard, loop domain, and before/after program points. Occurrences are
printed in stable structured discovery order, which is explicitly a partial
order across alternatives and iterations rather than an executable total
order. An execution lane is distinct from the logical ownership lanes in a
`ReadyRelease<N>` protocol: the latter select storage slots such as ping and
pong, while the former identify where operations execute.

For each admitted storage timeline, the experiment examines ready and release
demands separately while retaining their common generation and strict-channel
rejection. Ready demands are grouped by target lane and searched backward to
the earliest covered acquisition. Release demands are grouped by source lane
and extend from final use to the next overwrite. The initial structural
frontier rules admit only:

- an exact endpoint;
- the earliest or latest endpoint in one linear region and guard class; or
- the entry or exit of the first enclosing choice when the peer is outside
  that choice.

A rejected storage timeline is recorded as a residual experiment instead of
being silently dropped. The `lane-frontiers` dump and JSON counters therefore
measure same-lane versus cross-lane demands, linear coalescing, choice-boundary
hoisting, ready/release decomposition, and frontiers found for channels that
the strict Checkpoint C recognizer rejects.

These records are observations, not protocol or direct-repair candidates. They
are never selectable and do not establish event-direction legality, pipeline
completion, visibility, or lane completion order. In particular, same-lane
membership does not mean that synchronization is unnecessary.

The focused experiment currently shows:

- a Vector `PIPE_V → PIPE_V` chain has one structural same-lane frontier while
  legacy InsertSync emits a `PIPE_V` barrier at that point;
- two linear consumers on one target lane collapse to one earliest-acquisition
  frontier, matching legacy's single ready event pair;
- consumers in the two arms of one `scf.if` collapse to the choice entry,
  matching legacy's hoisted wait;
- a static same-buffer overwrite separates into a ready demand and a reverse
  release demand, matching legacy's two directed pairs; and
- depth-one and depth-two loop timelines expose paired ready/release frontiers
  with reuse distance one and two respectively.

A diagnostic sweep over the 25 functions in the lane, channel, and information
preservation fixtures produced 46 experiments: 35 structural frontiers, ten
timeline residuals, and one unresolved schedule. Thirteen found frontiers came
from strict-channel rejections: four `multiple-consumers`, four
`static-overwrite`, two `reuse-capacity-mismatch`, two `nested-loop`, and one
`unsupported-control-flow`. This is static fixture evidence only; it says that
lane/frontier structure survives those recognizer rejections, not that the
corresponding protocols are complete or target-legal.

The existing overlapping-access regression adds a second boundary. Its two
MTE2 loads share a pipe and need no legacy barrier, while its two Vector writes
share a pipe and retain a legacy `PIPE_V` barrier. Both are rejected before
timeline frontiers are available (`partial-overlap` or
`conflicting-physical-range`). Residual repair therefore cannot be reconstructed
from admitted channels, or even admitted timelines, alone. A later residual
experiment must retain raw access-pair endpoints and ask the target model for
the hazard-specific same-lane completion rule.

This supports lane-aware frontier discovery and demand-specific reasoning, but
it also fixes the boundary for the next design amendment: only a target query
may classify a found same-lane frontier as intrinsic ordering, a targeted
barrier, or unsupported. Recurring ready/release observations must remain one
atomic protocol during selection.

When Checkpoint E selects a `ReadyRelease<1>` plan, the lane-frontier dump also
reports a read-only differential result. It compares the selected producer and
consumer execution lanes plus publication, acquisition, final-use, and
next-overwrite points with the C.5 observations. This catches structural drift,
but it is not an independent hardware proof because both analyses consume the
same frozen schedule and storage timeline.

## Checkpoint C.6 lane-pattern experiment

The follow-up experiment retains raw physical-access endpoints across rejected
storage timelines and adds three diagnostic-only recognizers:
`SharedOneShotFrontier`, `SameLaneCompletionCut`, and
`ChoiceBalancedRoundTrip`. Every candidate records its old CanonicalSync
analogue, target-query result, Checkpoint E status, and estimated action cost.
No candidate is selectable.

On the frozen 394-row A3 corpus, all rows completed and the experiment recorded
18,151 endpoints, 66,839 hazardous access pairs, and 1,050 candidates. The
target model classified 470 raw pairs as intrinsically complete, 51,261 as
requiring a same-pipe barrier, and 15,108 as cross-lane/not-applicable. All 22
shared one-shot placements matched an old selected event at the operation-name
level. Of 1,020 same-lane cuts, 507 matched an old selected barrier; all 481
cuts in old-admitted rows matched. Eight balanced-choice candidates survived
inside four old fail-closed rows but remained rejected by Checkpoint E for
multiple consumers.

The detailed method, fixture outcomes, comparison limitation, and conclusions
are recorded in the [lane-pattern experiment
report](ptoas-protocol-sync-lane-pattern-experiment.md). The result
supports lanes as a structural discovery and differential layer, not as a
replacement for storage/iteration proofs or target-specific completion rules.
Stable/alternating L1 and accumulator protocols remain postponed until their
missing storage, core, and iteration facts are repaired.

## Checkpoint C.7 storage-track experiment

The next read-only layer partitions exact local byte ranges into atomic storage
tracks and combines them with execution lanes, control, and iteration order to
form storage-transition frontiers. Affine depth-two occurrences retain
conditional `slot(t)==0/1` membership, and partial overlaps become shared and
unshared atoms rather than an overlap equivalence class.

On the frozen 394-row corpus, 18,025 of 18,151 attempted accesses projected to
3,492 tracks. The experiment represented all 66,839 members of its exact,
linear raw-pair universe with 37,261 lifecycle, completion, or residual
transitions. The actual Checkpoint E planner admitted none of the corpus's 474
recurring transition records, while all four strict `ReadyRelease<1/2>` fixture
functions matched E's storage masks, physical slots, directions, and lifecycle
placements. Thus the storage/lane layer is a broader discovery substrate, not
a substitute for E's complete protocol proof.

The post-rebase audits found zero atom-mask mismatches across 1,276,033 exact
interval pair relations and zero raw-pair omissions, but they also exposed the
remaining correctness boundary: 12,622 non-linear frontier memberships lack a
path proof, branch joins lack must/may-reaching generations, dynamic subview
precision can be lost, and fixed supplied protocols are not imported into the
selected world. An independent narrow reconstructor found 177 strict lifecycle
shapes in 76 corpus functions, all still outside E's complete admission.

The exact semantics, comparison limitations, corpus breakdown, literature
analogues, and remaining completeness gaps are recorded in the [storage-track
experiment report](ptoas-protocol-sync-storage-track-experiment.md). The
cross-experiment claims and source-by-source literature assessment are indexed
in the [evidence ledger](ptoas-protocol-sync-evidence-ledger.md).
The resulting pass revision is specified in the
[obligation-engine amendment](ptoas-protocol-sync-obligation-engine-amendment.md):
canonical path-sensitive typed obligations become the correctness substrate,
while E and other patterns remain atomic proof/compression templates.

## Checkpoint D and E status

Checkpoint D implements the opt-in, atomic same-pipe and directed same-core
`OneShotPublish` subset with target legality and independent verification. Its
exact-device qualification remains a separate campaign.

Checkpoint E implements strict `ReadyRelease<1/2>` prime/body/drain protocols
described in `docs/designs/ptoas-protocol-sync-checkpoint-e.md`. The exact
depth-two revision still requires its purpose-built A3 device gate; neither
host admission nor prior CanonicalSync evidence is a hardware certificate.

Checkpoint F Commits 12--14 add the generation-aware selected-world
interpreter, strict targeted direct repair, and complete mixed protocol/direct
selection. Combined allocation, exact selected-world re-evaluation,
whole-candidate reverse deletion, independent verification, and module-atomic
materialization are described in
`docs/designs/ptoas-protocol-sync-checkpoint-f.md`.
