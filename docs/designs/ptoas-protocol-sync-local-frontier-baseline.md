# ProtocolSync local frontier baseline

## Status

This is the first production slice of the agreed sparse-requirement / backward
frontier algorithm, not another selectable diagnostic recognizer. It feeds the
existing direct-only and mixed planners. The C.6/C.7 diagnostic implementations
and their frozen evidence remain separate and non-selectable.

The implementation replaces recognition-derived local obligations only when it
accounts for the entire vector-UB domain in one function. Supported accesses
have directly addressed `alloc_tile` operands, checked static footprints, one
physical operation phase, no loop instance, and one common block/guard. Different
logical allocations at equal or partially overlapping addresses share atoms.
Ordinary in-place UB accesses are read/write effects, not ACC conflicts.

This does **not** complete GenerationSSA, arbitrary branch participation, loop
repair, or the full current-main PyPTO/PyPTO-Lib acceptance goal. The historical
4/152 result is not an updated admission measurement. No device campaign or new
target qualification is claimed by this patch.

## Five cooperating mechanisms

1. `LocalAccessRegion.cpp` independently recovers allocation footprints from
   actual addressed IR, with checked size multiplication and address addition.
   It does not promote the legacy alias map's intervals to authoritative facts.
2. `LocalMemoryAnalysis.cpp` partitions footprints with an endpoint sweep and
   constructs per-atom may-definition/use records and sparse outstanding-access
   requirements. Requirement IDs and access/atom provenance do not depend on
   which protocol world is selected.
3. The selected-world interpreter propagates the completion supplied by
   validated concrete fixed synchronization and selected candidates before
   returning residual requirements. It retains a separate visibility graph;
   completion is not imported as GM publication.
4. Direct repair groups residual intervals by physical lane pair and guard.
   It chooses common consumer boundaries, then walks sparse source links
   backward to the latest legal producer frontier for each boundary. The
   existing complete-world selector and allocator remain in use. Protocol
   candidates stay atomic, optional alternatives.
5. Concrete verification reconstructs actual synchronization and checks local
   hazards with an independent exhaustive access-pair traversal. This check
   neither builds atoms nor reads the sparse requirements or planner tags.
   Direct-only materialization now invokes fresh concrete verification as well
   as its existing candidate/materialization checks, before staged IR commits.

The backward traversal is a placement heuristic over already discovered
requirements. It does not define the hazard universe. A same-lane issue order
is not treated as a blanket completion edge; existing target-qualified barriers
and directed events implement the current conservative repairs.

## Precision and generation invariants

`SyncRegionPrecision` distinguishes `Exact`, `Conservative`, and `Unknown`.
Currently, recovered tile-wide access footprints are **Conservative**. A precise
allocation base and size do not mean an instruction accesses every allocated
byte. The atom partition is exact for those supplied upper bounds, not a claim
of exact dynamic instruction footprints.

Unknown addresses, view/alias operands without an independent bounded recovery,
slot selectors, packed-stride layouts, and cross-block/loop domains do not gain
native admission through this analysis. Domain admission is atomic: one such
access prevents replacing that domain's existing timeline protection.

For each atom, the discovery frontier tracks a previous write and the readers
since it. A new write requires completion of that write and all outstanding
readers. Subsequent requirements can follow the new write because the earlier
effects remain connected through mandatory completion requirements. This is
not the assertion that a lexical overwrite completed previous work.

May-definition records retain the previous definition. A conservative write
does not kill that prior reaching definition. Full must/may generation flow
through Phi/LoopPhi nodes is subsequent work.

For the admitted linear order, sparse coverage follows a simple chain argument:
successive writes are ordered; each intervening reader follows the previous
write and precedes the next write. Every earlier-write/later-read, read/write,
or write/write pair therefore has a path through those links. Coalescing links
across atoms retains their membership. This proves coverage of the supplied
effect bounds, not completeness of operation semantics or validity of the
upstream memory placement. Synchronization cannot restore a value already
destroyed by an invalid placement.

Read/write effects belonging to one physical phase are combined before transfer,
preventing invented self-dependencies. Read/read overlap produces no ordinary
memory obligation. ACC/proxy resource ordering remains outside this UB baseline.

The physical domain is the function's vector-core UB instance. Allocation roots
are intentionally not separate alias domains. Distinct physical sections are
not declared disjoint merely because they reuse numeric addresses; this initial
implementation rejects their cross-block relation.

## Fixed supply and verification boundary

Planning imports only synchronization that the existing concrete interpreter
can reconstruct and validate. Failed import leaves opaque-effect rejection in
place. Events, supported pipe barriers, and terminal drains contribute their
existing modeled effects; queues and unmodeled macros do not acquire contracts
merely by appearing in IR.

Fixed event IDs are reserved conservatively when allocating direct or protocol
repairs. This is not improved sequential event-ID reuse. Event-generation
assignment still needs its existing interference checks, including in the
concrete verifier.

A selected optimized protocol must continue to validate its own complete
certificate. Native residual repair cannot mask a corrupted selected protocol.
Existing OneShot/ReadyRelease exclusions for fixed/generated protocol mixtures
are unchanged; the newly admitted fixed-supply examples use the direct world.

The independent local verifier shares operation-local footprint extraction, but
not atom construction, sparse discovery, or the planner's claimed coverage.
It tests every overlapping, hazardous pair in the supported straight-line
subset against concrete completion reachability and can return the first
uncovered access pair. It is not yet an independent general control/iteration
interpreter or a PTO hardware oracle.

## Reproduction and acceptance tests

Use the workspace's configured Python and LLVM tools, with at most two build or
test workers in aggregate:

```sh
cmake --build build --parallel 2
build/tools/pto-test-opt/pto-protocol-sync-local-memory-test
llvm-lit -j 2 -v --filter protocol_sync build/test/lit
llvm-lit -j 2 build/test/lit
.venv/bin/python .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py --repo . --base HEAD
```

The unit oracle enumerates all 243 read/write/read-write sequences of length
five over overlapping physical footprints. It reconstructs each access's byte
set independently and checks sparse-chain coverage against all-pair hazards.
This is an exhaustive **straight-line** oracle, not bounded loop evidence.
Additional recovery tests require dynamic addresses, negative addresses, size
multiplication overflow, and endpoint-addition overflow to remain `Unknown`.

`protocol_sync_local_reuse.pto` exercises partial physical reuse, an outstanding
reader before overwrite, and in-place UB computation through direct-only and
mixed emission, with legacy fallback disabled. It also checks independent
concrete verification and idempotent reprocessing of emitted synchronization.
`protocol_sync_fixed_supply.pto` checks partial fixed coverage, barrier supply,
and preservation of occupied event IDs when further repair is necessary.

The concrete mutation tests delete or move synchronization in newly admitted
local programs. Existing tests continue to cover protocol corruption, recurrence
rejection, event capacity, guards, target policy, and module-level rollback.
Changed residual diagnostics explicitly show canonical local IDs instead of
claiming that a recognition rejection is an alias or accumulator hazard.

The focused host run passes all 40 ProtocolSync tests. The full host suite
passes 1,897 tests, with one unsupported and zero failures (1,898 discovered).
The scoped changed-code checker reports 22 code files, zero errors and zero
warnings. The oracle reports:

```text
protocol-sync local byte and sparse-pair oracle: 243 programs pass
protocol-sync local concrete mutations: pass
protocol-sync local unknown and overflow bounds: pass
```

These are working-tree validation results, not a new frozen C.6/C.7 campaign.
The regression programs run direct and mixed emission on A2/A3, exercise both
GM contracts, and disable legacy fallback. No native frontend-corpus admission
percentage or device-performance change follows from these tests.

## Remaining implementation order

### Frozen-input native admission measurement

The A3 host campaign on `1f9b2a27f` plus this local-frontier working-tree patch
reran the 394 frozen differential inputs with legacy fallback disabled. Both
campaigns recorded stable source/compiler fingerprints; they are patch-frozen
development evidence, not clean-commit or fresh current-main measurements.

| Input population | Rows | May-alias admitted | Disjoint GM arguments admitted |
| --- | ---: | ---: | ---: |
| PyPTO-Lib | 346 | 16 | 23 |
| PyPTO | 48 | 14 | 14 |
| Total | 394 | 30 | 37 |

The PyPTO rows contain 24 distinct input hashes; the whole corpus contains 370.
Unique admitted inputs number 23 and 30 respectively. Both modes include six
zero-physical-work rows. All 30/37 admitted rows also generate C++ and pass a
fresh concrete-verifier invocation. This does not compile device binaries or
establish runtime correctness/performance. No crashes or failed campaign probes
occurred. In both modes, 106 rows fail extraction; another 258/251 fail mixed
selection. Exposed blocker counts overlap and are not predicted admission gains.

Examples retained in the per-row evidence:

- `k005`, `decode_layer_active_trim`: direct-only guarded copy. One MTE2-to-MTE3
  handoff covers both local availability and the safe-mode GM read-before-write
  requirement. Both contracts emit one event pair and one exit drain.
- `k106`, `build_bias`: two UB atoms cover 8 KiB and 32 KiB overlapping tile
  allocations; 16 canonical requirements replace recognition-derived local
  protection. Disjoint-argument mode selects five event pairs, ten targeted
  barriers and one exit drain. Safe mode retains an unsupported GM write-to-read
  publication requirement: local reclamation does not establish GM visibility.
- `k358`, `dyn_kernel_softmax_prepare`: five atoms and 16 local requirements;
  direct-only selection needs two event pairs, nine targeted barriers and one
  exit drain in safe mode. Disjoint arguments remove two GM store-order barriers.
- `k128`, `prefill_idx_qr_hadamard`: optional OneShot plus grouped lane frontiers
  uses three event pairs and one drain. This cube example does not exercise the
  new vector-UB atom engine; it illustrates existing mixed-world composition.
- `k010`, `exp_gate_up_act`: metadata extraction and loop/reuse boundaries remain
  unsupported. Fixing only `set_validshape` would not establish loop repair.

All five examples generate C++ with InsertSync. The three admitted safe-mode
examples also work with native direct-only emission; `build_bias` additionally
works direct-only under the disjoint-argument contract.

The [archive manifest](protocol-sync-evidence/native-local-1f9b2a27-worktree/archive.json)
and adjacent aggregate JSON retain commands, input/compiler/toolchain hashes,
per-row hashes and patch provenance. The disk-backed archive includes all raw
logs, emitted IR, original inputs, untracked source snapshots and the C++ /
concrete follow-up. Its local path is recorded in the manifest; no external
artifact upload is claimed. The follow-up `v2` supersedes a producer-label
accounting error in `v1`; actual compiler and verifier outcomes are unchanged.

### Structured continuation

1. Extend independently recovered access provenance to static views and
   conservative dynamic bounds, with independent byte-set tests and explicit
   analysis limits.
2. Add per-atom path joins and participation; retain alternative writers and
   outstanding readers rather than selecting a lexical last writer.
3. Add iteration-local and carried requirements together with recurring event
   consumption, entry/exit, and zero/arbitrary-trip proofs.
4. Broaden domains and fixed supply only with their target contracts. Preserve
   GM may-alias versus assumed-disjoint policy and visibility distinctions.
5. Adapt protocol certificates to canonical generation/obligation subsets and
   evaluate the current-main native acceptance population in both GM modes.

## Changed-code applicability

Required rules cover C++17 bounds, overflow, lifetime, explicit runtime failure,
braced control flow, scoped CMake additions, and compiler regression tests.
Formatting follows the repository's local formatter; no broad formatting or
warning-suppression changes are included. Release hardening, device runtime,
network/process input, and unrelated language rules are not exercised here.
The changed-code checker is a prefilter, not a security or hardware proof.
