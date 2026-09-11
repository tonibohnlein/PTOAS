# Structured OAHS S4: loop ordinals and initialization/steady-state domains

## Scope and base

This is a follow-up to `60f131ab29fb0db610aaca7a1a0ef75f191458ff` on
`codex/oahs-upstream`. It extends the existing `structured` engine; `existing`
remains the default. No pass, selector, target instruction, annotation, solver,
Presburger operation, work quota or legacy seed is added.

The mandatory native targets are all seven existing population inputs:
`one_buffer`, `two_buffer`, `three_buffer`, `four_use`, `online_softmax`,
`qk_matmul`, and `q_proj`. Original sources and their manifest hashes do not
change. Native acceptance and device performance are NOT implied by the
portable production-core tests shipped with this candidate.

## 1. Correct Boolean interpretation before occurrence discovery

The previous native periodic evaluator represented Boolean values by their
bits, 0/1, then compared those bits as signed int64 values. Signed one-bit 1
means -1, not +1. For example `cmpi slt %true, %false : i1` is true. Incorrectly
calling such a guard false could erase the executing payload from the model.

The native comparison switch now calls `ordinalCompare`. Boolean operations,
equality and unsigned comparisons retain 0/1. Signed ordering uses -bit. Other
admitted scalar values retain their existing signed/unsigned interpretation.
All ten predicates and all four Boolean pairs are tested against independent
expected values. Raw native fixtures compare a periodic predicate with its
signed-i1 equivalent and require actual dependencies and handoffs; successful
compilation of an empty imported model is not a passing test.

## 2. Analysis coordinates, not an IR-normalization pass

For the original signed-index loop

```
for i = L; i < U; i += S
```

S4 admits a constant `L >= 0`, constant `S > 0`, and the existing qualified
signed-64-bit index layout. The ordinal is `k`, with `i = L + S*k`. Mathematically:

```
T = 0                                  when U <= L
T = (U-L)/S + ((U-L)%S != 0)             when U > L
```

`LoopOrdinal` computes this count without `extent + step - 1`. Values,
distances and inverse lattice membership use checked arithmetic. The original
loop, raw induction variable, payload expressions and allocations remain in
place. No count expression or flattened iteration product is emitted merely
to make the analysis look normalized.

The body still has S1's periodic atoms and S2/S3's boundary interfaces. Its
numerical trip count means ordinal count, never raw upper-bound value. The
normalized wrapper-loop restriction from S3 remains; nonunit inner loops are
added here, not arbitrary dependent or nonunit wrapper nests. Unsigned loop
comparison mode is explicitly refused in both body and wrapper qualification.

## 3. Physical selection uses the original IV

For a raw selector `i mod M`, the effective ordinal selector is

```
(L + S*(offset+k)) mod M
```

with period `M / gcd(S,M)`. Offset is zero normally and one for the steady tail
of a split initialization loop. `OrdinalResidue` retains this exact phase and
stride. It uses Euclid/modular arithmetic, not enumeration proportional to M.

Examples:

* `i=2*k`, `i mod 2`: always slot 0, NOT alternating slots.
* `i=1+k`, `i mod 2`: starts at slot 1.
* `i=1+2*k`, `i mod 3`: the original three-slot permutation is preserved.

Equality/inequality predicates against a constant use the exact modular
preimage. Unit-stride threshold predicates preserve their wrap/threshold cuts.
General strided threshold permutations are refused rather than represented by
an incorrect interval. Explicit physical-slot tables still justify finite
multi-residue expansion; a large numerical modulus alone does not.

Unknown selector precision retains a conservative admitted footprint. Different
slot numbers are not a physical-disjointness proof. Physical address overlap,
root/argument alias contracts and typed ACC/visibility requirements are unchanged.

## 4. A finite initialization phase is not periodic

Q projection has `i=0,2,...,62` and a predicate derived from `i*128 == 0`.
The ordinal mapping alone cannot make that predicate periodic. S4 recognizes a
qualified affine equality/inequality whose unique zero is ordinal zero.

Every participating index add/subtract or constant multiplication is checked
for representability over the original loop domain before using its affine
meaning. Bounded Q projection permits this proof. A data-dependent expression,
wraparound identity, unsupported cast or general inequality is not extrapolated.

For a nonempty loop the analysis creates:

```
original prelude + virtual first iteration
    -> periodic tail starting at ordinal 1
    -> original epilogue
```

The first iteration is represented using S2's one-shot Prelude atoms. Common
body operations have an initial analysis occurrence and a recurring occurrence;
first-only and later-only branches have only the applicable occurrence. Static
payload operations are NOT cloned, peeled, moved or rewritten in the IR.

This preserves common loads/extracts and their order around the initializer.
It does not replace several readers by one, kill old reads when a new write is
seen, or add an intrinsic MMAD guarantee.

For a dynamic upper bound there are two complete, complementary cases:

* `U > L`: the virtual first iteration exists, followed by zero or more tail
  iterations;
* `U <= L`: no original body executes; only actual prelude and epilogue effects
  and their requirements remain.

The empty case does not execute a fictitious initializer. For a constant bound,
the impossible case is omitted. Existing wrapper/choice invariance requirements
ensure these case predicates cannot change between the invocations S3 summarizes.

## 5. Direct emission in raw coordinates

Let `B=L+S*offset` be the original IV corresponding to ordinal zero of the
analyzed body. Let d be an ordinal handoff distance and r its residue.

| Semantic predicate | Emitted comparison |
|---|---|
| Matching future occurrence exists | `U - i > S*d` |
| Matching previous occurrence exists | `i >= B + S*d` |
| First occurrence of atom r | `i == B + S*r` |
| Final occurrence of the slot/period | `U - i <= S*D` |
| Atom r exists somewhere in the body | `U > B + S*r` |
| Initial virtual occurrence | `i == L` |
| Steady virtual occurrence | `i != L` |

All scaled constants are checked before mutation. `U-i` is formed only inside
an executing original body. Multi-residue endpoint guards use the qualified
ordinal `(i-B)/S`; that division is emitted only below the steady-role guard,
where its numerator is nonnegative. No potentially overflowing `i + S*d` or
unconditionally evaluated `U-L-1` is required.

The original upper bound must dominate an early publication whenever that
publication needs it. The constructor does not delay publication to solve an
operand-availability problem.

Reconstruction recognizes these exact raw-coordinate forms, original loop
identities, scaling and lattice constraints. It also requires the actual
empty/nonempty and initial/steady guards. Removing a phase guard, changing a
divisor or changing a zero-trip predicate is not authorized by an emission tag.
The original payload/control snapshot and fresh physical retranslation remain.

## 6. Causal reuse between initialization and recurrence

Virtual initialization can make S2's old policy (distinct boundary keys,
reserved from all recurring streams) unnecessarily restrictive. The portable
Q-projection-shaped witness has four boundary MTE1->MTE2 streams and four body
streams: reserving the former leaves too few keys, even though a causal
continuation can reuse them.

`Model::allowBoundaryKeyReuse` is an internal realization policy, not a new
input attribute or pass option. Startup units enable it. Other units keep their
old assignment if it succeeds; on allocation refusal they may try this one
alternative checked policy. No new candidate endpoints are selected in order
to fit resources, and no full-completion drain or acknowledgement is added.

Boundary streams remain distinct from each other. A recurring candidate can
use a boundary key only if the actual causal graph establishes consumption
before every relevant following publication. The check uses S3's qualified
first/last episode interface over all of its exact parameter cells. Interior
recurring-to-recurring reuse remains checked by the existing recurring causal
closure; omitted interior episodes are NOT treated as proof of safe reuse.

The final verifier reconstructs actual actions, checks the complete command
population, keeps actual same-boundary order, and rechecks mixed transitions.
Publication fire is not made a source-lane blocking action. A rejected sharing
trial loses an optimization, not a hazard. Failure of the complete deterministic
assignment still does not prove hardware infeasibility.

## 7. Test and reporting changes

The native gate now names all seven population inputs and preserves their
source hashes. `four_use` is not omitted, but neither is its current larger
inventory frozen as a correctness theorem. Keys are reported by direction;
sets, waits, named barriers, terminal ALL and scalar overhead remain separate.

New native fixtures cover signed-i1 guards, shifted lower bounds, nonunit steps,
constant effective selectors, mask/remainder forms, explicit slot permutations,
empty/nonempty initialization and scaled first-iteration predicates. New
mutations target only synchronization-only generated subtrees, not original
payload control that a snapshot would trivially reject.

`benchmark_buffers.py` defaults to existing/structured across all seven cases,
with warmup and three serial paired rounds. Arm order rotates. Refusals,
validation errors, timeouts and missing rounds remain visible; they cannot
produce a passing speedup/ratio result. Optional logical comparisons and C++
generation from already synchronized PTO are separately labelled. The old <=2x
whole-compilation target is available as an explicit gate, not a claimed result.

Historical GEMM is registered by immutable commit/path/blob and its original
provenance. `recover_historical_gemm.py` performs read-only extraction from a
local Git object into a new archive directory, verifying the complete blob.
It neither fetches nor changes branches. The preparation environment did not
obtain its full bytes, so those bytes are NOT bundled in this patch. Historical
RUN lines remain outside automatic lit discovery. Current syntax adaptation,
structured admission and full GEMM remain later work, without changing the
pairwise noalias contract into an all-accesses-noalias assumption.

## 8. Evidence and limits

Portable tests compile the actual production core/helper and compare ordinal
arithmetic against independent enumeration/wide arithmetic, and event plans
against the existing independent finite asynchronous oracle. They exercise a
hand-transcribed Q-projection-shaped model; they do NOT execute its native
importer. New shared mathematical and physical-model trust boundaries remain
subject to native corruption tests and hardware qualification.

This candidate has not been built with the pinned PTOAS/MLIR toolchain in the
preparation environment. Native all-seven acceptance, scalar overhead, controlled
compile timings and device results must be run before accepting S4. No numerical
kernel speedup, reduction of four_use's extra pair, or historical GEMM support
is claimed. No new UnitFlag, intrinsic MMAD, visibility or retirement rule lands.

Primary semantic references:
* MLIR SCF `scf.for`: https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop
* MLIR arithmetic bitvector/comparison semantics: https://mlir.llvm.org/docs/Dialects/ArithOps/
* Pinned source base and S1-S3 contracts: `60f131ab29fb0db610aaca7a1a0ef75f191458ff`.
