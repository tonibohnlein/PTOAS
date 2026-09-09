# OAHS extraction from the accepted M2 checkpoint

This worktree starts at upstream `7e2ec3e29420e297dcf5b3c59ba4d90841464821`.
The extraction source is the committed research checkpoint
`5d6766d69d3494bfcf621b65cfa64b3d1c959bd7` (M2 implementation `68a31b6c8`,
following M1 `416b1e8688f2269ed03070ee2428e521b0058f28`). The later unfinished
relation-derived guard edits are excluded.

## Dependency boundary

| Component | Treatment |
| --- | --- |
| LogicalSyncRelations, SyncOccurrences, SyncPayloadSnapshot | Same accepted code |
| LogicalSyncPlan | Same constructor; physical import and alias calls redirected to independent utilities |
| SyncPhysicalFacts | Extracted physical phase, descriptor, target and footprint qualification; no lifecycle graph |
| SyncGlobalOccurrences | Extracted qualified GM partition query; unresolved runtime guards remain conservative |
| SyncGMAlias, SyncEffectCoverage | Qualified argument-root contracts and independent translated-effect checks |
| PTOIRTranslator, MemoryDependentAnalyzer, SlotAffineAnalysis | Upstream implementations unchanged |
| Legacy/lifecycle planners, handoff refiner, frontier optimizers | Not imported |
| Refiner/revised/checkpoint outputs | Frozen, hash-checked test artifacts only |

The constructor receives unsynchronized PTO. It discovers immutable typed
requirements, selects occurrence-prefix advances, establishes complete guarded
participation, repairs remaining requirements using combined completion,
assigns all logical streams together, then emits and reconstructs on a clone.
It never asks the legacy planner or the old refiner to select its synchronization.

`SyncPhysicalFacts` validates the translator's memory-effect mapping and
single-phase target contracts. It does not compute the research importer's
expanded graph, descriptor full-write dataflow, physical partition, storage
frontiers or lifecycle candidates: the accepted logical constructor did not
consume those results. Phase identities come from the original operations;
`SyncOccurrences` establishes execution order independently.

The logical access query keeps unresolved overlap conservative. Different GM
roots need the qualified, explicitly selected argument-disjointness contract.
Known local physical intervals are compared with overflow checks. The separate
GM occurrence query handles the committed bounded, contiguous full-tile store
subset and returns false when a proof needs an unmaterialized runtime guard.
Neither query asserts definite production from an allocation footprint.

## Default behavior and supported scope

`--insert-sync-planner=existing` remains the default and takes the upstream
path before logical contract resolution or diagnostic changes. The original
zero-argument pass factory remains available. Both serial module and ordinary
function pipeline sites forward the same logical options.

`logical` requires successful independent construction. `logical-or-existing`
may invoke upstream insertion on the untouched input payload when semantics,
realization, work or allocation are unsupported. Internal inconsistencies are
hard failures. Reports distinguish strict construction from fallback.

Memory-valued `scf.for` carried arguments and all `scf.while` operations are
explicitly refused before translation: their research-only forwarding changes
were not necessary for M1/M2 and were not copied. Broader handle support needs
its own qualified translator integration. The accepted function-only lifetime
restriction remains; physical-section exit lowering and MMAD discharge are
still later milestones. No new target or operation admission is claimed.

Removing the expanded graph also removes its expansion limit from this path.
Independent operation visitation, nesting, physical phase/access and relation
work limits remain explicit. Actual limit exits report `analysis-limit`.

## Acceptance gate

The following checks passed, and the software architect, compiler expert and
algorithms/performance reviewer accepted the completed extraction:

- Native MLIR/libisl query parity and occurrence-import checks.
- Independent known-local requirement comparisons, including actual overlap,
  equivalent predicates, additional readers and conversion extents.
- Strict construction of unchanged online softmax, Q projection, QK and one
  buffer, including their original zero/nonzero execution scenarios.
- Exact checkpoint mechanism, payload, allocation, ABI, scalar replay and
  synchronization trace parity; no later acquired cross-lane completion prefix.
- QK first-panel readiness excluding the independent second preload.
- Structured empty/skipped-reader checks and emitted-IR corruption rejection.
- Explicit physical-import refusal tests and upstream-equivalent fallback.

The old M1/M2 reports remain historical evidence. `checkpoint/manifest.json`
identifies frozen input/output hashes and original campaigns. A new extraction
report records the clean binary and test results. Local scalar replay and
symbolic reconstruction are not device correctness or wall-time qualification.

`codex/oahs-upstream` is now the sole development line. The research branch
`codex/insertsync-logical-plan` is a frozen reference. Its unfinished guard
changes remain byte-for-byte in the original worktree and in a separately
hashed patch. Continue those changes and the remaining milestones on this
clean branch after the accepted extraction checkpoint; do not evolve a second
planner on the research branch.
