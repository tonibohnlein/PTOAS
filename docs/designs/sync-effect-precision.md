# Synchronization effects and buffer geometry

`SyncInput` supplies instruction phases, pipes and read/write operands through
the same translator and operation interfaces as InsertSync. `SyncStorageEffects`
retains these records without adding instruction-specific footprint rules.
The source IR and borrowed input records must remain unchanged during analysis.

## What the representation establishes

`descriptorRegion` describes a buffer operand: its base address, valid extents,
element width and coordinate-to-byte mapping. It composes pointer offsets,
tensor partitions and local views; symbolic values remain SSA symbols instead
of being evaluated or unrolled. Local physical identity includes memory space
and address, so separate SSA allocation roots do not hide address reuse.
Straight-line valid-shape updates are tracked; unresolved control-dependent
metadata remains unknown.

A descriptor is not an accessed-byte set. The shared read/write interface does
not currently specify the accessed subregion or guarantee that all native
accesses stay inside the descriptor's valid extents. Therefore the current
importer retains descriptor geometry separately and marks access precision
`Unknown`. It does not infer a definite overwrite or materialize descriptor
bytes as accessed cells. No opcode-specific recovery is performed, including
for scalar reads and writes.

The access representation retains three precision values for a future generic
region contract:

| Precision | Meaning | Allowed use |
|---|---|---|
| Exact | The actual accessed byte set is known, possibly symbolically. | Exact dependence queries; definite writes where coverage is established. |
| UpperBound | A supplied region contains all accessed bytes. | Disjointness filtering, but no overwrite kills. |
| Unknown | No accessed region is established. | Retain possible overlap. |

An empty cell list for an unknown or symbolic effect does not mean no access.
Completeness of the upstream read/write records remains a separate premise;
`allAccessesExact()` does not audit missing implicit accesses.

## Filtering and alias policy

Different known memory spaces are disjoint; an unknown space may overlap any
space. Two reads require no storage ordering. These facts remain usable even
when byte geometry is unresolved. Pipe assignments come from the shared
instruction phases; same-pipe accesses can conflict.

For GM, `MayNotAlias` asserts independence of distinct resolved function-entry
pointer roots, matching the existing default. `MayAlias` retains possible
aliasing between them. Pointer offsets preserve provenance; selected, carried
or unresolved pointers do not acquire an independence assumption. Absolute GM
addresses need not be known. Same-root accesses remain potentially overlapping
without an actual-region contract. Local allocation reuse never uses the GM
root-independence assumption.

## Analysis status

The structural recognizers remain available and report missing access premises
instead of claiming exact applicability from buffer descriptors. Exact demand
generation needs a generic accessed-region contract; this cleanup does not
supply it. The existing InsertSync pass and its dependency analysis are unchanged.

The retained cell-partition utility splits supplied byte intervals at endpoints,
not at every byte. Its cost is `O(R + S log(S+1) + I)` for records `R`, intervals
`S` and effect-to-cell incidences `I`, excluding geometry recovery. The current
importer supplies no exact intervals. The generic symbolic mapping and overlap
helpers are representations and queries, not a lifetime or demand algorithm.
