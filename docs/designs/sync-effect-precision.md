# Synchronization effects and buffer geometry

`SyncInput` supplies instruction phases, pipes and read/write operands through
the same translator and operation interfaces as InsertSync. `SyncStorageEffects`
retains these records without adding instruction-specific footprint rules.
The source IR and borrowed input records must remain unchanged during analysis.
Declarations on operations without a translated pipe phase are retained through
`SyncInput::effectsFor(Operation*)`; their classification is not inferred from
the presence of a read/write effect. The Step 0 corpus audit is documented in
`test/step0/README.md`.

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

Possible overlap uses the existing `MemoryDependentAnalyzer::MemAlias` query
on shared buffer records. Planned local addresses, view ranges and possible
slot addresses can establish disjointness even without exact instruction
coverage. Unknown local addresses remain conservative; distinct SSA roots alone
do not establish physical disjointness. Supplied exact access regions can refine
the result further. Neither a disjointness test nor overlapping allocation
ranges establish a definite overwrite.

The shared root/range check is used only when the buffer's SSA dependencies do
not cross region arguments or region-producing results. The existing translator
may retain only the initial root of a loop-carried buffer; that root cannot prove
disjointness for later iterations. A memoized dependency walk checks this
condition once per build. Structured-control aliases remain conservative unless
an independent descriptor-region proof establishes disjointness.

Different known memory spaces are disjoint; an unknown space may overlap any
space. Two reads require no storage ordering. Pipe assignments come from the
shared instruction phases; same-pipe accesses can conflict.

For GM, `MayNotAlias` asserts independence of distinct resolved function-entry
pointer roots, matching the existing default. `MayAlias` retains possible
aliasing between them. Pointer offsets preserve provenance; selected, carried
or unresolved pointers do not acquire an independence assumption. Absolute GM
addresses need not be known. Same-root accesses remain potentially overlapping
when the shared ranges cannot separate them. Local allocation reuse never uses
the GM root-independence assumption.

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

## Shared access selections

`PTOAccessRegion.h` supplies a versioned parameter on the existing MLIR memory
side-effect declaration. It preserves the effect operand and read/write mode,
so existing InsertSync continues to consume the same dependencies. No new PTO
syntax, Python operand, or lowering convention is introduced.

A selection consists of a shape operand, a choice of valid or capacity extents,
and an affine map from selection coordinates to coordinates of the affected
operand. Map symbols refer to scalar operands of the same operation. An identity
map describes a whole region; a translated map describes a subregion. The generic
consumer composes either with the existing view/layout/physical-address map.
It checks the schema, indices, dimensions and arithmetic before publishing an
exact region. Missing or incompatible declarations remain unresolved. Several
accesses of the same buffer/mode must agree; a known selection cannot hide an
additional unspecified access. Malformed contracts publish no partial result.

The first native producer is ordinary noncompact MAT-to-LEFT/RIGHT `textract`
with 16/32-bit elements and boxed, aligned shapes using 512-byte fractals.
Other fractal layouts are not certified: native block addressing need not agree
with their descriptor map. Its read selection uses the
destination capacity and extraction offsets; its write selection covers the
destination capacity. The declaration accounts for native 16-bit offset
conversion and A5 sub-block rounding. On A2/A3 the producer requires proven
16-element offset alignment, including scalar constant-multiple expressions.
Other transfer forms keep the existing unresolved read/write declaration. This
is a qualification on the operation's shared semantics, not an opcode list in
the synchronization consumer. Additional producers use the same contract.

Explicit views already have geometry in `PTOIRTranslator` (including segmented
local subviews) and `resolveBufferRegion`. Existing InsertSync maps each memory
effect's operand to that buffer record; it does not apply extraction row/column
operands to construct a read rectangle. `SyncStorageBounds` on the
minimum-demand-sync branch similarly contains conservative allocation bounds.
The earlier removed implementation supplied selection parameters as well as
extensive native recipes. This version reuses the selection/geometry separation,
without restoring its scalar, vector, padding, packed-type and compact-transfer
recipe collection.

The diagnostic dump shows `descriptor` and `access` separately so that a source
buffer of 32 by 64 elements is not confused with a selected 16 by 16 region.
When an exact selection is available, overlap checks use it instead of assuming
that its accessed bytes equal the cached allocation range. Symbolic selections
remain symbolic; this change does not normalize loop-carried bank recurrences.


## Full aligned GEMM accesses and bank formulas

The shared declarations also describe ordinary full aligned matrix loads,
matmul/accumulating matmul, and accumulator stores. Loads use matching rank-two
ND source views and full MAT tiles; matmuls use coherent M-by-K, K-by-N and
M-by-N operands; stores preserve the logical rectangle when converting f32 to
f16. The qualification checks native layouts, dimensions, valid extents and
stride encodings. Partial-valid shapes, optional transfer modifiers, unsupported
layouts, and a distinct accumulating input retain their prior declarations.
These are shared operation semantics; the synchronization consumer interprets
only the coordinate maps.

Scalar address normalization uses constants, counted-loop induction ranges,
checked affine arithmetic and value-preserving casts. It can derive the closed
form of an unconditional carried counter
`bank_next = (bank + increment) mod modulus`, with a constant canonical seed,
nonnegative increment, positive modulus and a proof that the update cannot
wrap. With loop lower bound L and positive constant step S, the value at the
start of the iteration is
`(seed + increment * ((iv - L) floordiv S)) mod modulus`.
The seed is reapplied at every loop entry; it is not carried across enclosing
iterations. Direct `iv mod modulus` expressions use the same representation.
No loop is unrolled and the original program is unchanged.

For unflagged machine arithmetic the normalizer requires a proven signed range;
index arithmetic must fit both supported 32- and 64-bit widths. Explicit
no-signed-wrap operations retain their contract. Unknown expressions remain
original SSA symbols. Conditional updates, noncanonical seeds, unproved casts,
and loop results (including zero-trip results) are not replaced by the carried
counter formula. Symbolic trip counts are supported for the counter itself;
other arithmetic may still require bounded ranges or overflow flags.

These maps are inputs for later analysis, not a certificate that an entire
nested GEMM satisfies a periodic or arithmetic route. In particular, nested
control and whole-region recognition remain separate from exact access coverage.
