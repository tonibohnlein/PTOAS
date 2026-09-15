# OAHS native foundation: initial shared path

## Scope and integration

`include/PTO/Transforms/OAHS/Plan.h` defines the first native C++ contract.
`lib/PTO/Transforms/OAHS/Plan.cpp` implements construction and read-only checking.
It is included in the production transforms library and selected with
`-pto-insert-sync=algorithm=handoff`; `existing` remains the default.
The standalone CMake test compiles the exact production source without requiring
the old branch or its planner libraries.

The production adapter now provides a PTO-to-PTO straight-line vertical slice.
It imports through `PTOIRTranslator` and `MemoryDependentAnalyzer`, emits through
`SyncCodegen`, detaches generated commands, compares the remaining generic IR
with the original imported payload, and verifies the emitted command stream
against the original obligations before atomically replacing the function body.

The shared representation retains Sequence, Choice, For, and While structure,
qualified physical witness ranges and provenance, separate byte/resource/
visibility fields, physical phase identity, reservations, and an explicit
invocation contract. `analyze()` validates that representation independently of
`construct()`. Structured byte/resource-cell obligations use a conservative
all-pipeline realization at actual consumers. Authored protocol composition,
macro import, visibility/resource realization, and complete population coverage
remain open and are reported as synthesis or analysis blockers.

## Selected reuse from c73c04fb3

The old `StructuredSyncCore.h` supplies the fixed-pipeline representation.
`StructuredSyncComposition.cpp::PrefixState` supplies immutable producer-prefix
receipts and transitive completion knowledge. The new core represents versions
by original operation ordinals instead of per-cell may-history masks because each
operation occurs once in this slice. Later issue never enlarges a saved receipt.

`StructuredSyncStorageEffects.h` and the byteEffects hardening establish the
separation between byte roles and stronger resource exclusion. Here each cell can
require read/read exclusion without making its readers into byte writers.

`StructuredSyncOpenProtocol.inc` supplies the event-state principles: publication
does not gate subsequent source issue, occupancy and consumption knowledge are
distinct, acknowledgment snapshots carry causal knowledge, and fresh consumption
invalidates stale-generation knowledge. The new verifier implements those rules
without the old constructor stack or conditional refinement metadata.

This is an adaptation of those semantics, not a wholesale source transplant.
Old operation-name registries, experimental target profiles, independent child
allocation and numbered-plan replacement are not dependencies.

## Data flow

1. The caller supplies complete single-phase effects on prepartitioned physical
   cells, fixed pipelines, and an explicit target capability contract.
2. Immutable original-access witnesses produce byte-completion and resource-
   exclusion demands. Each witness keeps its latest writer and all readers since
   that writer, avoiding a pairwise operation cross product while preserving
   multi-reader release obligations.
3. Forward per-observer completion frontiers discharge established requirements.
   A remaining cross-pipeline demand publishes just after its required producer
   and acquires immediately before the consumer. Its prefix can serve other cells.
4. Logical handoffs undergo one global allocation. The initial allocator uses
   distinct keys per direction first. At key scarcity it tries deterministic
   reuse and accepts it only when reconstructed verification proves causal
   consumption-before-rearm. A supported all-pipeline barrier handles remaining
   scarcity; no automatic reverse acknowledgment is emitted.
5. The verifier reconstructs actual commands against original operations. It
   ignores constructor demand/handoff annotations, checks hazards and safe rearm,
   requires consumed tokens, and enforces retirement only when requested by ABI.

Target tables are explicit input. No hardware key range, synchronous scalar rule,
DMA visibility premise or retirement requirement is guessed by the core.
Completeness defaults to false; the forthcoming lowering-owned importer must
establish it, not trust an arbitrary IR attribute supplied by a caller.

## Next native integration

Complete the lowering-owned declarations for the captured operation population.
`SinglePhaseSyncOpInterface` owns ordinary one-phase completeness and
`MacroSyncOpInterface` makes completeness of a lowering-owned macro model
explicit. The current declarations cover plain TLOAD, TADD, and the existing
macro families; macro import is deliberately still rejected until its phases,
private events, and boundary effects are all connected.

Extend the current structural transfer beyond its conservative byte-effect
realization to authored/internal events, explicit visibility and resource
effects. There is no second planner or legacy fallback behind
`algorithm=handoff`.

The coverage gate remains the whole pinned production-supported population under
shared assumptions. This core's restricted input is not counted as PTO coverage.

## Verification and commit boundaries

Production changes: the OAHS header/source, semantic declarations, translator
failure propagation, pass selection, and transforms-library registration.
Evaluation changes: this note and `test/oahs/`. Keep these in separate consecutive
commits; test the combined working tree before committing either portion.

```
cmake -S test/oahs -B ../oahs-m1-core-build
cmake --build ../oahs-m1-core-build --parallel 2
ctest --test-dir ../oahs-m1-core-build --output-on-failure --parallel 1
```

Tests cover cross-cell sharing, early readiness/release, transitive prefixes,
new-write invalidation, causal and unsafe key reuse, finite-key exhaustion,
resource exclusion, explicit retirement and incomplete declarations. Two ordering
tests reject edges from unrelated work to the consumer. An independent graph
oracle checks 200 seeded generated programs with distinct issue and completion.
These are native core tests, not MLIR integration or device qualification.
