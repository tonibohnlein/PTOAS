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

## Production and experiment boundary

OAHS is the `logical` planning mode of the existing InsertSync pass. Its C++
constructor starts before upstream synchronization selection and allocation.
The production dependencies are the qualified physical/occurrence utilities,
MLIR Presburger queries, construction, realization and fresh reconstruction.
The old research refiner is not a production dependency.

Keep the experiment campaign separate from the eventual upstream merge:

- `test/experiments/insert_sync/logical_plan/` contains the Python/libisl
  reference, frozen planner outputs, historical reports and benchmark observers.
  Its relation driver and differential/mutation gate are included only with
  `BUILD_TESTING`, Python bindings and `PTOAS_OAHS_TESTS`. None of these files
  is imported by production compilation; libisl is a test runtime dependency.
- `tools/pto-test-opt/pto-logical-sync-test.cpp` and
  `pto-sync-occurrences-test.cpp` are native validation executables, not compiler
  dependencies. Their targets currently live alongside the upstream test tools;
  final merge packaging should decide whether to retain them under the normal
  test-build gate or extract the necessary cases into upstream's test facilities.
- The small `logical_sync::testing` observation/mutation entry point is currently
  compiled with the constructor. It has no CLI or environment activation and is
  called only by the native validation driver. Do not describe this shim as
  already compiled out of production. Its declaration/packaging can be isolated
  further when preparing the merge.
- Preserve focused correctness regressions and production reconstruction even
  if the research campaign, comparison artifacts and test-only adapter are
  excluded from the merge. Compiler correctness must not depend on retaining
  the experiment directory.

New exploratory scripts and durable local campaign outputs belong outside the
source worktree. Continuing guard work must not add a dependency from production
code to the reference prototype, benchmark data or expected synchronization counts.

## Default behavior and supported scope

`--insert-sync-planner=existing` remains the default and takes the upstream
path before logical contract resolution or diagnostic changes. The original
zero-argument pass factory remains available. Both serial module and ordinary
function pipeline sites forward the same logical options.

`logical` requires successful independent construction. `logical-or-existing`
may invoke upstream insertion on the untouched input payload when semantics,
realization, work or allocation are unsupported. Internal inconsistencies are
hard failures. Reports distinguish strict construction from fallback. Authored
static/dynamic flags and record/wait events are classified before selecting a
constructor: existing mode preserves them, hybrid mode reports `authored` and
preserves them, and strict mode refuses their unmodeled protocol. A standalone
barrier does not imply a complete authored protocol and retains the existing
barrier policy. Unsupported logical input alone never authorizes inserting
another protocol around authored events.

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
