# Synchronization effect precision

`SyncStorageEffects` consumes the instruction phases and read/write records
owned by `SyncInput`. It resolves physical geometry and partitions known
ranges. It does not generate demands or change the original program.

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
| Unknown | No physical byte bound is established. | Retain possible aliasing with all accesses in the same memory space. An unknown space may alias any space. |

These statements assume valid accesses within the operand's storage, including
in-bounds dynamic view offsets and slot selectors. Overflowing address/extent
computations yield unknown geometry. General input validation remains an
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

The initial implementation resolves constant local allocations with plain,
byte-addressable tile layouts, immediate constant subviews, and physical slots
of multi-tile allocations. Row/column segments preserve holes in subviews.
Planned multi-buffer addresses take precedence over contiguous slot inference.
A constant slot selects one range; a dynamic slot retains the union of possible
slots. Nested or dynamic subviews retain their root's bound. Unsupported
layouts, unresolved roots, global pointer geometry and unknown addresses remain
unknown. No disjointness is inferred merely from different global pointer SSA
values.

A known allocation extent is not an exact instruction footprint. The first
exact instruction refinement covers constant `tgetval`/`tsetval` element
accesses with unambiguous contiguous coordinates. Other supplied tile accesses
remain upper bounds until an instruction-specific footprint rule proves more.
In particular, a tile write is not assumed to overwrite its entire allocation.

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

`mayOverlap(a,b)` uses sorted cell lists, taking linear time in their combined
length; unknown geometry answers conservatively. A positive answer means
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
partial scalar writes, exact scalar reads, and constant/dynamic slot selection.
The shared-input driver also verifies that the source IR is unchanged and that
rebuilding the partition does not accumulate records.
