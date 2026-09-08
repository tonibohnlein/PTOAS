# InsertSync shared requirements and residual placement

This change extends `--insert-sync-buffer-generations`. It keeps the existing
input contract and the production default. The motivating native decisions are
the MTE3 barriers between output stores in the unchanged buffering controls and
TopK, including the case where no complete lifecycle can be constructed.

## One retained requirement interface

`SyncRequirements` replaces the separate MMAD pair vector, slot requirement type
and lifecycle full-completion vector. Direct repairs and selected lifecycle
omissions retain source/target payload operations and access SSA values. Their
justifications remain distinct: slot or GM disjointness, qualified MMAD order,
exact lifecycle membership, and full physical completion.

An ordinary repair additionally retains its logical synchronization group.
Motion and event allocation preserve this group; code generation binds the
actual emitted barriers and events back to it. Adjacent-barrier deduplication
retains both bindings. Recoloring a key does not change action ownership.
Selected protocol events have separate explicit ownership, established when
they are emitted. Their ownership is no longer reconstructed by guessing from
the assigned event numbers. No diagnostic attribute is trusted as input proof.

The first migrated relationship is a store/store ordering requirement:

```text
translated source/target accesses and occurrence relationship
  -> unconditional disjointness, or retained direct repair group
  -> ordinary motion and finite event allocation
  -> emitted action bound to that original group
  -> shared storage/guard/completion refinement of owned residual barriers
  -> independent reconstruction of the concrete event plan
```

`--pto-insert-sync-debug=1` on `pto-test-opt` prints each residual candidate's
retained group, source/target operations, access values and carried status.
This makes a remaining barrier traceable after placement and allocation.

## Connecting GM regions to storage and completion

The existing GM footprint importer rejected any layout attribute, including
the ordinary ND attribute inserted by normal lowering. It now admits explicit
ND for qualified contiguous, full-tile UB stores, while retaining the extent,
alignment, scalar arithmetic and physical-layout checks. The TopK index store's
32-bit integer element type is also admitted for this same non-converting store
contract. Other layouts and unknown footprints remain conservative.

The same occurrence query now has two native consumers:

- Ordinary repair can omit an access pair when disjointness is unconditional;
  its retained witness is re-extracted and rechecked after emission. This does
  not require successful protocol construction.
- Residual placement can use a conditional overflow-safe partition proof. It
  computes one invariant condition and preserves the original barriers on the
  overflow-risk path. Small benchmark trip counts execute no such barriers.

The arithmetic bound is necessary. Dynamic trip arguments are not presumed to
fit merely because recorded device launches are small. Zero trips retain their
existing behavior. Constant bounded loops need no runtime guard. Overlapping
output partitions retain ordinary synchronization.

Previously, guarded residual refinement required replacing the entire function
body and was therefore disabled with selected protocols. The new path wraps
only the owned residual barriers in place. Payload and protocol identities,
guards, event endpoints and assigned IDs remain fixed. Reconstruction then
checks the combined emitted program. Generic residual event movement and joint
sharing of selected protocol keys are still separate, unfinished capabilities.

## Other fixture blockers

TopK's single-pipe sort, merge and non-macro gather phases now use the same
qualified structural import as construction, ordinary generation queries and
refinement. Existing translated effects must match the operation interface;
multi-phase macros are rejected. TopK still has overlapping allocation views
that do not qualify as complete exact-slot channels. Its GM partition facts
nevertheless reach ordinary residual placement and improve that output.

The other recorded gates require different semantic work:

| Fixture | Remaining optimizer boundary | Specific next relationship |
| --- | --- | --- |
| Triangular inverse | Asynchronous scalar values from `tgetval` flow into `taxpy`; the optional effect checker does not admit this prerequisite. | Scalar-result publication/consumption and the in-place destination update must be modeled together. Admitting an opcode alone is insufficient. |
| Conv2D | Helper calls, implicit configuration and macro phases are outside the transactional constructor. | A qualified helper summary must include configuration resources and physical completion, and retain symbol context in the clone. |
| FlashAttention | Queue initialization, allocations and peer-ready/free effects are outside the local model. | Queue ownership and fixed peer event resources must coexist with local effects. |
| GDN / KDA | FFTS and fixed mixed-core signaling are outside the local effect import. | Keep separate physical contexts and model fixed remote resource participation before optimizing local regions around them. |

These are restrictions of the optional analysis, not findings that production
InsertSync misses required synchronization. The latest recorded device campaign
already launched GDN and KDA with coordinated peers. Its remaining device
blockers are Conv2D's arm-independent launch fault and FlashAttention's
arm-independent ND-to-ZN TLOAD assertion; local analysis changes do not repair
those launch/layout problems.

## Validation and measurement

`insert_sync_shared_requirements.pto` checks the unchanged two-buffer input,
constant-bound and overlapping-partition variants, and the unchanged TopK
input. Existing tests retain MMAD negatives, slot wraparound, empty iterations,
protocol reconstruction and GEMM parity equivalence.

`performance/compare_boundaries.py` reuses the scalar replay to compare the
source-completion prefixes required at physical operations, independent of
event-ID spelling. It retains complete handoff observations and payload parity.
It does not model latency, certify event causality, or establish device speed.
Its negative control delays a publication past an independent load and detects
the additional required completion.

The [results and boundary comparison](../../test/experiments/insert_sync/performance/SHARED_REQUIREMENTS_RESULTS.md)
record the unchanged eleven-fixture campaign and the remaining GEMM differences.
