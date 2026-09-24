# Phase A Step 5: original values and endpoint qualification

> **Integration update, 2026-09-24:** this increment has passed local native
> validation and independent review at its stated scope. See the
> [integrated review](frontier-synch-steps2-6-review.md) for fixes and evidence.
> The candidate/pending statements below record the original patch submission,
> not the current integration status. Later Phase A gates remain open.

**Candidate increment; native validation and independent acceptance pending.**

Base: `0a38c8e9f4bd4852177c1dd6f6e0d0ee26b6ca16`,
`codex/handoff-foundation`. The factored transfer core from `9bba6552055d5386ff7242c58031b0412885a0d6`
is retained. Its transfer equations are unchanged; guard identity is delegated
to the common service. This does not claim complete Step 4 provenance integration.
The patch does not include uncommitted or parallel implementations of Steps 1–4.

## Contract and consumers

Draft v0.44 §2.5, §3.2–3.6, §4.1 ("Availability is an independent obligation"),
§4.6 and Appendix I.1–I.2 are the source contract. Phase A reads the original
program; it does not add selected completion, synchronization, keys, scalar
counters, or replacement payload computations.

`OriginalValueQueries` is owned by `ProgramAnalysis` and shared with
`OccurrenceQueries`, `OriginalReadQueries`, and the factored provenance client
through `OriginalLifetimes`. Standalone clients retain their previous
constructors and own an instance of this same service when none is supplied.

| Interface | Meaning / actual client |
| --- | --- |
| `qualify(Value, OriginalValueCut)` | Original scalar value at its own before/after cut; typed-requirement collection |
| `qualify(OriginalValueReference, cut)` | Same query with explicit defining scope and visit offset; retained-value client |
| `atom` / `predicate` | Width-qualified original observations and shared predicate DAGs; reader boundary queries |
| `counted` | Original positive-step loop, including its final increment; reader and bank-occurrence qualification |
| `canonicalGuardOwner` / `identity` | Defining SSA value and occurrence scope; both provenance choices and reader predicates |
| `qualifyEnabled` | Checks explicit enabling guards against named completion prerequisites; endpoint/binder-facing API |
| `guardedLast` / `extremum` | Safe arithmetic recipes for I.2; these are available for the later interval client, not an implementation of Step 10 |
| `ProgramAnalysis::guardQualificationAt` | Complete directional result for a translated payload boundary |
| `OriginalBoundaryResult::endpointQualification` | The requested direction's result, independent of the opposite frontier |
| `OriginalEndpointCandidate::qualification` | Complete result retained alongside the compatibility booleans |

The four statuses are `Available`, `NeedsCompletion`, `NotObservableHere`, and
`Unresolved`. Compatibility booleans are true only for `Available`.
`NeedsCompletion` is not selected credit and is not a successful key-allocation
certificate. The receiving constructor must realize every listed prerequisite
at its recorded deadline, under its original applicability, before using the
value. Missing recipes or value correspondence do not erase known prerequisites.

A requirement contains its produced SSA value, original occurrence scope,
translated source phase, executable whole-instruction source cut, source
engine, exact scalar-use deadline, and original lexical conditions. Several
phases of one instruction remain distinct completion requirements, but no
internal phase gap is invented.

For `async -> cmp -> if`, the producer's completion deadline is **before the
original cmp**, not merely before the if. `typedRequirementsAt` now indexes
that earlier scalar deadline and the source subscription names that deadline.
This is an intentional change to the old typed-collector behavior. Invocation
scalar arguments no longer manufacture unresolved asynchronous producers.

## Original value identity

SSA dominance is necessary but not sufficient. An asynchronous producer that
dominates the endpoint still requires completion. A result defined only later,
or only in a non-dominating branch, is not rematerialized at an earlier cut.
After a structured operation means its actual join/exit; its results are not
available before that operation or inside its own defining region.

The registry canonicalizes an unchanged loop-carried argument to its actual
initial SSA value only when the original yield returns that same argument.
Changed carried values retain their own identity. Pure, total scalar carried
recurrences are checked without substituting their initial values.

There is also a constructive, deliberately scoped previous-value rule. If the
original loop yields precisely the requested value, its corresponding carried
argument represents that value on the next visit. A `-1` query can use it on an
original branch proving a noninitial visit (`iv != lb`, the matching strict
comparison, or the false arm of equality), with qualified counted arithmetic
and pure total carried computation. The answer records both the original
reference and the retained SSA representative. It does not reevaluate the old
expression with current arguments. Initial paths and larger/unsupported shifts
remain unqualified. Async carried histories need their own completion transport;
static site equality does not supply it.

The original object, configured index-width contract, and version are immutable
for a service lifetime. Value queries are cached by SSA identity, defining
occurrence scope, visit offset, cut anchor/side, and original-program version.
An IR/effect/control change requires a fresh `ProgramAnalysis`; synchronization
state is not a cache input. Guard-DAG visits are memoized without valuation
expansion. Materialized prerequisite/value lists have their own cost; this
patch does not claim the I.5 formation bound for arbitrary scalar queries.

## Initial arithmetic fragment

Scalar integer widths 1–64 are represented explicitly. `index` uses the existing
64-bit `SyncSlotMapping` contract; another index width is an unresolved target
qualification, not a silent host-size substitution. Original integer constants,
comparisons, Boolean bit operations, integer min/max, integer/index casts,
extensions, truncations, and add/subtract/multiply retain their original
bitvector semantics. A wrapped original value is not a proof of mathematical
no-overflow. `nsw` and `nuw` require separate signed/unsigned range proofs.
Division/remainder require a nonzero divisor and the signed overflow check.
The restricted conditional rule also recognizes exclusion of a particular
constant by an original comparison on the **same SSA value** at an enclosing
branch; it does not solve arbitrary path predicates.

Counted loops use a positive constant step. Constant bounds get an exact
final-increment check. Unit-step loops also admit dynamic bounds; other dynamic
bounds require the stated sufficient range check. Arbitrary variable-step or
symbolic recurrence proofs are not supplied by this increment.

For a qualified body visit, the predecessor/successor recipes use unsigned
original-width distances:

```
has_previous(k): ((bits(iv) - bits(lb)) mod 2^w) / step >= k
has_next(k):     (((bits(ub) - bits(iv)) mod 2^w) - 1) / step >= k
```

The second subtraction occurs only on a body visit with `iv < ub`. This avoids
speculatively evaluating `iv + k*step` on a final visit. The original loop's
own final update is checked separately. No visit counter is added.

`GuardedLast` is an actual conditional recipe: on `lower < upper`, compute
`upper - 1`; on the empty arm, do not evaluate the subtraction. It must not be
lowered as an eager subtract followed by `select`. Integer min/max recipes
support the later clipped-interval query, but this patch does not assert that
arbitrary supplied bounds describe an access's participation.

An inherited bare `LoopResidue` atom does not state whether it means an IV,
ordinal, or carried-selector residue. It therefore remains unresolved instead
of inventing that meaning. The actual original remainder/selector SSA value
is qualified through `qualify`, and bank queries use that original address
expression. Complete physical permutation and boundary domains remain Step 8.

## Integration boundaries and review gates

This is Step 5 only. It does not complete fixed-visit alternatives, general
async/while scalar transport, D2 boundary domains, D4 re-entry, guarded-demand
obligation IDs, interval participation, directional covers, descriptor slots,
or preparation closure. The available arithmetic recipes do not implement
those later procedures. Existing marginal physical requirements are retained.
No shared instruction-effect definition or `algorithm=existing` code changes.
The current construction-not-implemented boundary is unchanged.

The old blanket reader availability gate has been replaced by a directional
query. An unavailable last-reader guard does not hide an independently
qualified first-reader answer. A structurally exact endpoint may carry
`NeedsCompletion`; consumers must not equate `Exact` with immediately available
scalar state. The new status and complete prerequisites are retained publicly.

Before accepting this increment, the independent reviewer must check the
native service and integration against these gates:

1. Run `native.cpp`, build the changed production translation units, and inspect
   the typed deadline/source records through `ProgramAnalysis` on actual PTO.
2. Run the existing FrontierSynch/factored tests and five development kernels;
   check unchanged IR and both autosync modes. A successful analysis invocation
   is not a positive answer to an unresolved occurrence query.
3. Check the first/last directional distinction, shared-guard identity, incoming
   scalar histories, retained previous values, and noncircular prerequisite
   paths. Record remaining supporting-premise failures explicitly.
4. Record reviewer findings and acceptance separately. The tests below do not
   stand in for independent review or complete Phase A parity.

## Supplied tests and actual local evidence

`test/frontier-synch-step5/arithmetic.cpp` executes the production arithmetic
helpers against an independent widened-integer/concrete-loop oracle. It covers
signed and unsigned widths 2–8, every operand pair, finite 5-bit loops, empty
intervals, initial/final domains, and separate 64-bit overflow boundaries.
There are **1,981,828 enumerated checks**, plus the explicit 64-bit/invalid-width
assertions. The loop enumeration belongs only to the test oracle.

Those checks passed with Clang C++17 under AddressSanitizer and
UndefinedBehaviorSanitizer, and through the standalone CMake/CTest target.

`native.cpp` supplies MLIR semantic fixtures for all four outcomes, a guard
available before a loop, future-only and async conditions, an enabling-guard
cycle, varying/changed carried values, actual retained previous values,
shared guards, structured cuts, hidden internal phase cuts, conditional
arithmetic, zero trips, overflow, cache reuse, and IR nonmutation. Its phase
records are explicit semantic fixtures, not claims about importing real PTO.
**This test has not been compiled or run in the preparation environment.**

No LLVM/MLIR development installation or repository checkout was available in
that environment. Source was read through the GitHub connector at the pinned
commit. Patch application was checked on a fixture assembled from the retrieved
change contexts, not on a complete source checkout. The complete-checkout
`git apply --check`, native build/tests, kernel corpus, existing-path regression,
and independent review remain acceptance gates. No device results are claimed.
