# Shared modeled storage accesses

`SyncInput` supplies the same instruction phases, pipes and read/write operands
used by InsertSync. `SyncStorageEffects` combines those declarations with the
shared buffer, view, physical-range and alias analysis. Frontier adds no
instruction whitelist. The source IR and borrowed records remain unchanged
until analysis and endpoint preparation finish.

## Contract

The demand algorithms compute the minimum demands for the **supplied storage
model**. A narrower shared access selection refines that model. Without one,
the shared buffer enclosure is the modeled access. If geometry cannot be
resolved, the shared alias predicate retains the possible interaction.
Missing byte addresses do not reject a payload or choose another algorithm.

There is no Exact/UpperBound/Unknown precision enum. Each record retains:

- The read/write mode, memory space and original shared buffer record.
- Available access maps, including views, valid extents and symbolic offsets.
- Materialized ranges and cells when available, with `rangesMaterialized`
  distinguishing a known empty set from an unmaterialized set.
- Original slot selectors and planner address tables.

These are representation choices. An enclosing range can be materialized;
a precisely described access can remain symbolic. Neither representation is
an instruction-admission category. Completeness of the upstream read/write
records is a separate input requirement.

A complete selected union is published only when every declaration can be
interpreted. Otherwise the importer retains the shared buffer bound. Likewise,
an affine representation of a fallback union is published only if every range
fits its coordinate representation; the complete unsigned ranges remain usable.

## Alias relationships and reduction

Known local addresses identify storage across distinct SSA allocation roots.
Different known memory spaces are disjoint; an unknown space may overlap any
space. Read/read pairs impose no storage order. GM uses the existing configured
MayAlias/MayNotAlias policy, including pointer provenance and view offsets.
Unresolved provenance cannot establish independence.

Materialized cell accesses use the storage-lifetime scan. Interactions absent
from that partition enter as additional forward conflict obligations, and the
same reduction computes their minimum generating set. An unresolved write does
not kill the state of a known cell. Possible aliasing is not an equivalence
relation: if an unknown access may overlap two disjoint buffers, those buffers
remain disjoint from each other.

Finite guarded analysis attaches both endpoint presence predicates to each
additional obligation. Compact adapters retain known symbolic maps and add
occurrence-independent alias obligations beside geometric generators. A periodic
body needs only the nearest reference-ordered source occurrence for each such
site pair; native completion order covers earlier sources. Arithmetic analysis
restricts the corresponding source/target domains by reference order before
using its existing reduction.

Regional exports retain shared effect identities and guarded first/last access
selectors in addition to cell lifetimes. Composition carries unresolved
relationships across boundaries through those selectors. It requires the same
unchanged access model and alias context. A varying crossing predicate still
needs an adapter that can represent it; a static alias predicate cannot replace
an available iteration-dependent relation silently.

## Cost and diagnostics

Partitioning intervals costs `O(R + S log(S+1) + I)` for records, ranges and
cell incidences, excluding geometry recovery. The ordinary materialized case
retains the existing scan/reduction bound. For `r` occurrences requiring alias
queries among `n` occurrences, residual generation performs `O(r*n)` occurrence
comparisons, multiplied by the effect-pair query cost. It short-circuits once a
conflict is found and visits each affected occurrence pair once. Compact uniform
relations are charged by their static effect/site pairs, not by runtime trips.
Regional residual crossing comparisons are charged separately from the cell
summary scan. No linear bound is claimed for arbitrary unresolved aliasing.

Dumps report `intervals`, `symbolic`, or `unresolved`, and Step 0 counts
`materialized_accesses`, `symbolic_accesses`, and `unresolved_accesses`. These
report the available representation; all three feed the same demand analysis.

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
The retained table describes storage selection; it is not a dependence result.

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
exact region. Missing or incompatible declarations retain the shared buffer bound. Several
accesses of the same buffer/mode must agree; a known selection cannot hide an
additional unspecified access. Malformed contracts publish no partial result.

An optional `valid_shapes` condition lists operand-index/row/column triples.
The mapper must prove those current valid dimensions at the accessing operation
before accepting a selection. Expected dimensions never replace unknown ones.
An operation whose execution depends on several operands repeats all conditions
on every declaration, so a failed source condition cannot leave an exact write.
This uses the existing metadata resolver and preserves its control/backedge
restrictions.

Ordinary aligned ND load/store and aligned TADD/TADDS use these conditions.
They qualify layouts, DMA alignment and field widths, vector tail behavior,
and narrowed repeat counts. Padding and special modes retain conservative
declarations. Exactness is available for constant valid dimensions recovered
from operands even when the tile type spells them `?`.

Finite pointer-relative GM intervals retain their canonical entry-pointer base
in every cell and regional summary. A set of exact intervals is an exact common
partition only after its base relationships are established: one shared base,
or independent canonical bases under `MayNotAlias`. Mixed absolute/based GM
addresses and unresolved distinct bases under `MayAlias` remain obligations.
Local cells continue to identify storage by physical address across SSA roots.

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
