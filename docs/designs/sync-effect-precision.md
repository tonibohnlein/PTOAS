# Synchronization effect precision

`SyncStorageEffects` consumes the instruction phases and read/write records
owned by `SyncInput`. It consumes physical geometry from the shared translator and partitions known
ranges. Read/write modes and pipe assignments use the same interfaces as InsertSync;
this layer has no independent instruction-support whitelist. It does not generate demands or change the original program.

## Contract

Each effect retains its phase, original memory record and read/write mode.
The phase and its source operation identify the control context; collecting
effects does not imply that all phases execute, or that a loop executes once.
The input and source MLIR must remain unchanged while these borrowed records
are used. A failed build exposes no partial result.

| Precision | Meaning for an executed access | Permitted use |
|---|---|---|
| Exact | The listed byte set equals the accessed set. | Track exact dependencies; an exact write overwrites every listed cell. |
| UpperBound | The accessed bytes are contained in the listed set. | Rule out disjoint accesses; retain possible conflicts. Do not discard old writers or readers because of this write. |
| Unknown | No absolute physical byte bound is established. | Retain possible overlap under the shared alias policy. Different known spaces remain disjoint; an unknown space may alias any space. |

These statements assume valid accesses within the operand's storage, including
in-bounds dynamic view offsets and slot selectors. Overflowing address/extent
computations are rejected by extraction or retained as unknown geometry. General input validation remains an
IR/front-end responsibility.

Precision describes the **supplied records**. `allAccessesExact()` does not
certify that the operation-effect registry includes every implicit access, and
is vacuously true for an empty record list. A consumer claiming exact program
analysis must separately establish completeness of the supplied effects.

## Physical identity and views

Storage identity is `(memory space, physical byte address)`, not an SSA root.
Consequently, a large allocation followed by two smaller allocations over its
address range shares cells with both of them. Allocation reuse does not start
a new storage history.

Known local ranges come directly from `BaseMemInfo`: memory space, physical
addresses and allocation/segment extent. This includes blocked layouts, padded
storage and views already resolved by the translator. It does not reconstruct
allocations or reject a layout merely because it is not plain. Address-plus-size
arithmetic is checked. Root records retain slot order for periodic analysis;
access records retain any constant-slot selection. Dynamic slot choices remain
unions of possible ranges, never definite writes to every slot.

An allocation or view extent remains an upper bound on an instruction's byte
accesses. The existing optional constant `tgetval`/`tsetval` refinement supplies
exact element accesses only with certified contiguous coordinates. Its absence
does not prevent other instructions from supplying conservative effects.
Exact routes still require exact input; no full overwrite is inferred merely
from a write effect or a known allocation size.

### Runtime GM pointers and alias policy

Absolute GM addresses need not be known during compilation. Shared records retain
pointer roots and view provenance; `mayOverlap` delegates GM queries to the same
`MemoryDependentAnalyzer` owned by `SyncInput`. Neither root-relative offsets nor
the placeholder address zero become absolute storage-cell identities.

`--insert-sync-gm-alias=may-not-alias` preserves the existing default assumption:
distinct resolved GM roots denote disjoint storage. `--insert-sync-gm-alias=may-alias`
retains possible overlap across those roots. Both modes keep same-root overlap
checks and conservative handling of lost provenance. The pass option is
`pto-insert-sync{gm-alias=may-alias}`; direct clients select `GMAliasPolicy` when
constructing `SyncInput`. The shared-input test tool accepts `--gm-alias=...`
before its inspection option.

Local allocation reuse is independent of the GM policy: overlapping physical
ranges share storage even with different SSA roots. Distinct known memory spaces
are disjoint. `mayConflict` additionally discards read/read pairs; operation
interfaces already determine which spaces each phase reads or writes and which
pipe executes it. Same-pipe accesses can still conflict. Event-direction
availability does not justify removing a storage demand.

## Partition and cost

Resolved intervals are normalized per effect, then split at all endpoints in
each memory space. Only covered intervals become cells. The algorithm neither
enumerates bytes nor compares every pair of accesses. Unknown effects remain
explicit records even though they have no cell IDs; an empty cell list must
not be interpreted as an empty access.

For `R` supplied records, `S` generated intervals and `I` effect-to-cell
incidences, construction takes `O(R + S log(S+1) + I)` operations and
`O(R + S + I)` space, with machine-sized checked coordinates. Here `S` includes
expanded subview rows/columns and alternative slots. This is an output-sensitive
bound: `I` can be quadratic in the number of interval endpoints, and expansion
is not polynomial in the bit length of an arbitrary encoded slot count.

`mayOverlap(a,b)` uses sorted cell lists for known local ranges, taking linear
time in their combined length. GM delegates to the shared alias checker, whose
range comparison costs the product of its two address-list lengths; unknown
geometry follows the policy described above. A positive answer means
possible overlap, not a proved dynamic dependence.

## Next consumers: recognition and demand generation

The [route recognizers](sync-analysis-recognition.md) inspect these effects
and the original control structure before selecting an analysis. Recognition
does not itself generate demands.

For an explicit execution with exact effects, the lifetime scan tracks the
previous writer and latest reader per pipe for each cell. It processes an
occurrence's reads before its writes, avoids self-demands for read-modify-write
operations, and emits sparse RAW/WAR/WAW generators. Forward reduction then
computes the minimum demands.

Upper-bound writes must not replace the writer history as if they certainly
overwrote every candidate cell. The conservative route must retain possible
writers, or apply a separately justified approximation. Likewise, a static
walk through both arms of a conditional or one visit to a loop body is not an
executed trace: structured routes must retain guards and iteration identities.
The lifetime scan, these control rules, and demand reduction are the next
implementation milestone, not part of this geometry layer.

## Regression coverage

The `sync_storage_effects_*` lit fixtures exercise reused physical ranges across
SSA roots, disjoint intervals, subview holes, unresolved views and addresses,
partial scalar writes, exact scalar reads, constant/dynamic slot selection,
blocked layouts, runtime GM roots under both policies, and read/read filtering.
The shared-input driver also verifies that the source IR is unchanged and that
rebuilding the partition does not accumulate records.
