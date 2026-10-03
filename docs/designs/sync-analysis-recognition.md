# Analysis-route recognition

The first recognizers inspect original MLIR using `SyncInput`, `PhaseIndex`,
and `SyncStorageEffects`. They neither unfold loops nor construct demands.
The library API is `FrontierSynch/Recognition.h`; its diagnostic client is:

```sh
pto-sync-input-test --recognize input.pto
```

The client reports the function's explicit-region candidacy and inspects every
`scf.for` separately, including loops inside rejected outer loops. Source
locations identify missing premises. It verifies that the original IR remains
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

The first exact within-slot fragments are constant scalar `tgetval`/`tsetval`
accesses to checked plain tile storage. This can prove an exact symbolic scalar
fragment even when the precision layer retained the union of all possible
slots for its dynamic access. Other tile operations retain an
`inexact-footprint` obligation. Unsupported views and conditional alias
alternatives remain explicit obligations; unknown physical geometry does not
prevent reporting an otherwise recognized slot expression.

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

Guarded rotating, arithmetic, counted-pattern and compositional recognizers
are not implemented here. The next useful steps are exact tile-operation
footprints and view normalization for representative kernels, then the rotating
generator and quotient backend. Arithmetic acceptance requires checking its
actual primitive relations; merely finding affine-looking operands is not
sufficient.

## Validation

The lit tests cover exact versus upper-bound explicit effects, parameterized
two- and three-slot loops, zero and large trip counts, independent assessment
of nested loops, unknown geometry, shifted-index obligations, mismatched
strides, physical aliasing across SSA roots, runtime moduli, conditional bodies
and unsupported loop steps. Existing shared-input and precision tests remain
applicable.
