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

A descriptor is not an accessed-byte set. The importer retains the original MLIR `MemoryEffectOpInterface` declarations
alongside the translator's read/write records. A matching default-resource
effect with `getEffectOnFullRegion()` set, no uninterpreted parameters, and one
resolved operand mapping supplies whole-region coverage. For this bridge,
that region is the operand's valid logical region mapped to physical bytes.
Such a declaration produces an exact symbolic access; a concrete map can also
be materialized and partitioned. Partial, absent or ambiguous declarations
remain `Unknown` and cannot establish definite overwrite. Macro phases require
their own declarations and cannot inherit whole-operation coverage.

Existing PTO effect definitions generally omit the full-region flag, so this
bridge does not upgrade them automatically. No opcode-specific recovery is
performed, including for scalar reads and writes. Arbitrary parameter attributes
are preserved but not interpreted as access geometry.

The access representation retains three precision values:

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
instead of claiming exact applicability from buffer descriptors. The generic
whole-region bridge consumes a declaration; it does not prove native instruction
coverage. Exact subregions still need a shared producer contract. The existing
InsertSync pass and its dependency analysis are unchanged.

The retained cell-partition utility splits supplied byte intervals at endpoints,
not at every byte. Its cost is `O(R + S log(S+1) + I)` for records `R`, intervals
`S` and effect-to-cell incidences `I`, excluding geometry recovery. The importer supplies exact intervals only for qualified full-region declarations. The generic symbolic mapping and overlap
helpers are representations and queries, not a lifetime or demand algorithm.

## Compact occurrence information

The descriptor carries enclosing `scf.for` induction variables, bounds and steps
in outer-to-inner order. The original `PhaseIndex` still owns branch/control and
value-availability queries. No loop is expanded to obtain an address.

Nonnegative counted-loop induction variables modulo a positive constant become
MLIR affine modulo expressions. Constant-divisor quotients use the same signedness
check. Addition/subtraction and constant multiplication are expanded only with
MLIR's no-signed-wrap contract. Unsupported expressions remain SSA symbols;
retaining them does not claim membership in an arithmetic fragment. Expression
construction memoizes SSA values and limits recursive expansion depth.

A selected buffer also retains its original selector and planner-assigned slot
address table, including nonuniform tables for which no affine address map is
currently constructed. This avoids replacing `addresses[k mod b]` by an
unqualified union when passing compact storage information to future backends.
The retained table is geometry, not an exact access or a dependence result.

`sync_compact_access_bridge.pto` checks nested reset selectors, non-contiguous
and irregular slot tables, wraparound handling and original-loop preservation.
Its contract test injects synthetic MLIR coverage declarations to exercise the
generic consumer; those declarations are not claims about the fixture operations'
native footprints.
