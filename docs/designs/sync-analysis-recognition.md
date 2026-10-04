# Analysis-route recognition

The first recognizers inspect original MLIR using `SyncInput`, `PhaseIndex`,
and `SyncStorageEffects`. They neither unfold loops nor construct demands.
The library API is `FrontierSynch/Recognition.h`; its diagnostic client is:

```sh
pto-sync-input-test --recognize input.pto
```

The client reports the function's explicit and finite-guarded candidacy and
inspects both rotating variants of every `scf.for` separately, including loops
inside rejected outer loops. Arithmetic recognition checks the whole function.
Diagnostics identify missing premises. It verifies that the original IR remains
unchanged. This is a developer test tool, not a production synchronization mode.

## Separate recognition from a ready analysis

| Recognition result | Meaning |
|---|---|
| `applicable` | The implemented input checks establish this recognizer's contract, relative to complete supplied effects and valid input execution. |
| `missing-premise` | The shape may qualify, but an effect, geometry, or arithmetic obligation remains unresolved. |
| `not-applicable` | The supplied syntax is outside the implemented class, or contradicts one of its structural premises. |

An applicable input is not Section 8's `Ready` result. The current client
separately reports `backend=unavailable`: these recognizers do not yet produce
generators, reduced demands, endpoints, boundary selectors or reachability
queries. Failed recognition neither proves physical infeasibility nor changes
the selected ordering. No compiler fallback is invoked by the diagnostic.

As with the precision layer, completeness of the shared effect producer is an
input contract. Unrepresented side-effecting operations are reported, but this
does not audit every registered operation for missing implicit effects.

## Explicit regions

The explicit recognizer accepts a single-block sequence without nested regions
or ambiguous multi-phase anchors, with assigned pipes and exact supplied cell
effects. It uses the original `PhaseIndex` sequence and never makes a compact
loop explicit. Upper-bound or unknown effects produce `inexact-footprint`.
The result describes one invocation of the block, not repetitions of an
enclosing loop.

## Rotating storage

The first loop recognizer accepts canonical `scf.for` loops with lower bound
zero, step one, no loop-carried SSA arguments, and a fixed body without nested
regions or ambiguous phase order. The upper bound may be a parameter, including
an enclosing invocation's parameter; no enumeration of its values is needed.
Within-loop conditionals require a different guarded route and are not silently
flattened. An enclosing loop containing an inner loop is rejected by this
recognizer even if the inner loop qualifies independently.

An access identifies a fixed physical allocation or a `multi_tile_get` family.
Constant slots and `iv rem slot_count` (signed or unsigned, with nonnegative
canonical induction) are recognized. Simple constant-stride/offset numerators
are normalized to `(stride * iv + offset) mod slot_count`, but currently carry
an `index-arithmetic` obligation: algebraic normalization alone does not prove
equivalence under machine overflow and signed remainder. Runtime moduli,
loop-carried selectors and unsupported expression forms are not accepted.

The recognizer records each fragment's family, normalized stride and offset,
slot count, and candidate refresh distance `slots / gcd(stride, slots)`.
All accesses to one family must have a common stride. Known physical slot
extents must be disjoint, both within and across families. Distinct SSA roots
with overlapping physical addresses are rejected by this initial family
recognizer, not treated as separate storage. Refining such roots into a common
family remains a possible extension.

Within-slot fragments must come from an exact shared access region. The earlier
`tgetval`/`tsetval` recovery shortcut has been removed. The shared input now
retains MLIR effect declarations and maps qualified full-region coverage to
descriptor geometry. Current PTO declarations omit that coverage, so those
examples still report `inexact-footprint`. Structural checks still report slot
expressions, families and refresh distances. See `sync-effect-precision.md` for
the coverage contract and preserved loop/slot information.

## Guarded regions

The finite-guarded recognizer traverses loop-free, single-block regions with
nested `scf.if` arms. It retains a shared predicate DAG: each node records its
parent conjunction, condition value and selected arm. Payload phases and
rotating accesses refer to those nodes. Missing arms are empty; arbitrary loops
and other region control are rejected. Supplied footprints must be exact.

Conditions may be computed within a finite region. Recognition records whether
all conditions are available at entry; a late condition does not invalidate the
finite guarded analysis class, but recognition does not establish that paired
synchronization endpoints can evaluate their guards. Predicate evaluation must
respect the parent path, particularly for values defined inside an arm.

The guarded rotating variant permits nested conditionals in the canonical loop
body only when their conditions are available before the loop. This sufficient
check establishes immutable participation across iterations. It does not try
to prove invariance of expressions defined inside the loop. The same physical
slot and footprint checks as the unguarded route apply. No retention circuits
or guarded endpoint code are generated yet.

## Arithmetic relations

`ArithmeticRecognition.h` checks a supplied bundle of exact primitive relations.
Every role must be present: context, occurrences, order, native order, reads,
writes and extra prerequisites. A present empty union denotes false, whereas a
missing role is an unmet input obligation. Completeness and equivalence to the
program, finite executions and native-order semantics remain producer contracts.
The checker does not infer these facts from arithmetic syntax.

Each primitive declares named coordinates and their roles, including parameters,
storage coordinates and auxiliaries. All coordinates count toward the configured
dimension limit. Every piece is an MLIR `IntegerSet` over quotient coordinates
with explicit residues under one common fixed period. The producer must already
have split modular constraints into these pieces. Raw division, remainder and
nonlinear products are rejected.

The caller supplies fixed limits on pipe count, dimension, period and coefficient
magnitude. These limits must not be inferred from the current kernel. All rows
are collected with checked integer arithmetic and normalized by their coefficient
GCD. Inequalities use floor division, including negative constants; an equality
with a nondivisible constant makes its piece empty. The normalized rows identify
difference constraints, integer octagons, or the broader bounded-coefficient
class. For example, `x+y<=N` has three coordinates and belongs to the broader
class when `N` is a parameter; `u=7v+3w+c` needs coefficient bound at least seven.

An accepted result retains normalized rows and indices into the supplied schema
and residue pieces. Failed recognition publishes no partial normalized bundle.
This is a representation check, not feasibility, projection, reduction or selector
synthesis. Integer coefficients are currently limited to signed 64-bit values;
overflow is reported rather than wrapped.

### Primitive extraction from kernel IR

`ArithmeticProgram.h` now connects the shared input to the arithmetic checker.
It accepts sequences and nested canonical `scf.for` loops (lower bound zero,
step one, no loop-carried SSA arguments). Upper bounds may be constants, function
entry index arguments, or enclosing induction variables. This includes rectangular
and triangular nests and restarted inner-loop buffer rotation. It analyzes the
whole nest without unfolding any trip count or repeatedly composing an inner
summary.

The producer exports a site table and relations for occurrence domains, strict
reference order, strict native reachability, physical byte reads/writes, parameter
context and an empty extra-prerequisite relation. Relation metadata identifies
source/target sites, their coordinate spans, start/completion event kinds and
address space. Order compares common enclosing iterations first and static
program positions afterward. Native relations include same-pipe start/start,
completion/completion and start/completion paths, including each payload's own
start-to-completion edge. They represent native closure, not just adjacent edges.

Fixed exact effects use the shared physical ranges, so overlapping SSA roots
refer to the same bytes. Dynamic slots require `iv rem slot_count`, actual
physical slot intervals, and exact supplied within-slot footprints. Slots need
not be evenly spaced, and distinct allocations need not be disjoint for this
arithmetic route. The slot count must divide the configured residue period.
All quotient substitutions use checked integer arithmetic.

This first producer supports **P in {1,2} and D<=8**, with at most 256 residue
tuples per conjunction. Larger producer configurations are rejected before
splitting. Pipe/dimension limits are checked before constructing site pairs.
The test client fixes `(k,D,P,C)=(8,8,2,8)`; the library takes an explicit
configuration. The broader supplied-bundle checker remains available separately.

Conditionals, noncanonical loops, computed bounds, unresolved footprints,
existing synchronization and additional SSA prerequisites are currently rejected.
In particular, a payload result consumed by another payload, loop control or a
return is not silently treated as free metadata. Rejected inputs export no
partial primitive/site/parameter bundle. Completeness of the shared operation
effect registry remains an input premise.

`pto-sync-input-test --arithmetic input.pto` emits structured test data for these
relations. `pto-arithmetic-recognition-test` exercises the generic checker and
its integer normalization. Neither tool generates demands or synchronization.

## Cost and scope

For a candidate with `N` immediate operations, `A` supplied access records, and
`S` physical slot intervals, recognition takes expected
`O(N + A + S log(S+1))` time and `O(A + S)` additional space. Family lookups
use hash maps; a sorted interval sweep checks disjointness. Expression matching
checks only the documented bounded-depth syntax. These costs exclude building
the shared input and precision partition, and charge the slot intervals
explicitly: this first implementation enumerates known physical slots. It does
not claim the paper's slot-count-independent extraction bound. The output has
at most one normalized fragment per supplied record; deduplication and
generator extraction are subsequent work.

Guarded traversal adds expected linear work in visited operations, phases and
access records, plus the rotating geometry checks when requested. For arithmetic,
let `L` count expression DAG nodes across rows, `D` the configured coordinate
limit, and `M` the input schema/residue metadata. Checking takes expected
`O(M + LD)` fixed-width arithmetic operations, plus coefficient GCD operations.
Expression collection shares nodes within each row. These bounds exclude
constructing the primitive relations and residue splitting.

For the IR producer, let `n` be the number of operations, `h` the nesting depth,
`s` the number of payload sites, and `a` the supplied access/range count. At fixed
supported `D,P`, collection and construction take expected `O(nh+s^2+a)` work,
with `O(s^2+a)` relation output; residue splitting has the explicit factor
`P^D` and polynomial factors in `D`. The quadratic term constructs reference and
native order between static sites, not conflicting dynamic access pairs. Costs
are independent of trip counts. These operation counts exclude shared input
recovery and the later symbolic reduction backend.

Counted-pattern and compositional recognizers remain unimplemented. Practical
coverage next needs producer-supplied coverage or exact subregions through the
shared interface, broader bounds/views and arithmetic conditions. The arithmetic reduction/selector backend and rotating
generator/quotient backend are also still separate work.

## Validation

The lit tests cover missing access-region premises, parameterized
two- and three-slot loops, zero and large trip counts, independent assessment
of nested loops, unknown geometry, shifted-index obligations, mismatched
strides, physical aliasing across SSA roots, runtime moduli, conditional bodies
and unsupported loop steps. Existing shared-input and precision tests remain
applicable.

New tests also cover nested then/else paths, absent arms, immutable versus
iteration-varying loop guards, late finite guards, all three arithmetic classes,
negative-bound normalization, impossible equalities, residue/schema mismatches,
configured bounds and checked coefficient overflow.

The arithmetic program fixture now verifies that missing access contracts and
rejected control/prerequisite cases export no partial bundle. Its former
occurrence-level comparison relied on the removed scalar footprint shortcut;
it is no longer evidence for exact IR extraction. The supplied-relation checker
continues to test the arithmetic classes and immediate rejection of oversized
producer configurations.

## Arithmetic physical identity

Arithmetic access relations use `(memory space, canonical base, byte)` as the
physical identity. For planned local storage the base tag is absent and the
byte is an absolute address. Distinct allocation SSA values therefore still
conflict when their assigned addresses overlap.

For GM accesses the shared view/pointer mapper retains a canonical function-entry
pointer and an exact relative byte map. The arithmetic producer accepts this
base only when it agrees with the shared memory record's root. Views, pointer
casts and `addptr` retain that identity and contribute their byte offsets through
the existing shared maps. A base tag is not an extra integer variable or a
numerical pointer address. Consumers must compare both identity tags before
comparing byte coordinates. The IR owns the borrowed base values, as it owns the
payload anchors.

One canonical GM base is supported under either existing alias policy. Distinct
GM bases are disjoint only under the supplied `MayNotAlias` policy; under
`MayAlias` the producer rejects multiple bases because their relative displacement
is unknown. Opaque selected/carried pointers and mixtures of based and absolute
GM addresses also reject without exporting partial primitives. This changes no
instruction footprint or shared alias policy and introduces no new local
allocation identity. The `--arithmetic` test dump exposes the base's argument
number (`-1` denotes an absolute address); it never prints an address value.

## Fixed local byte-map materialization

The shared range materializer recognizes separable sums of constant multiples
of a coordinate, its quotient and its remainder by one positive constant divisor
per coordinate. Constant extents split exactly into full digit rectangles and a
partial final rectangle. Within each rectangle, sorted strides merge byte
intervals only when they touch or overlap; negative strides are reflected, zero
strides add no bytes, and gaps remain explicit. The result is an exact union of
physical ranges, independent of instruction names and layout names.

Cost is charged to the parsed expression, generated rectangles, dimensions per
rectangle and emitted sparse slices, rather than to every byte or element of a
contiguous tile. Rectangles and aggregate sparse slices are each capped at
4,096; parsing also has explicit depth and work limits. Arithmetic is checked,
and failure preserves the caller's output. Unsupported forms use the previous
exact materializer when possible, otherwise retain their symbolic access map.
This does not guarantee concrete ranges for every layout or resolve symbolic
origins whose simplification requires loop-domain facts.
