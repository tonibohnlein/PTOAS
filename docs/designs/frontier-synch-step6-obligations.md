# Phase A step 6: guarded original obligations

Base: `0a38c8e9f4bd4852177c1dd6f6e0d0ee26b6ca16`, branch
`codex/handoff-foundation`. Specification: draft v0.44, Sections 3.2–3.4,
4.1 (obligation/request/placement identities), and Appendix I.1. The operative
Phase A specification is unchanged from v0.43.

**Integration status (2026-09-24): independently accepted at increment scope.**
Applied together with steps 2–5, with
interface corrections and additional regressions. The original bundle's tests
were run without LLVM; current local native validation and independent review
are recorded in [the integrated review](frontier-synch-steps2-6-review.md).
Complete Phase A parity remains a later gate.

## Public interface

`ProgramAnalysis::obligations()` returns a frozen `OriginalObligations` universe.
`obligationsAt(originalSite)` includes memory and typed/control families, including
control-only original sites with no translated payload phase. `atOperation(phase)`
is the separate phase-indexed memory view. Original site IDs and phase IDs must
not be interchanged.

`OriginalObligationFamilyId` names a shared demand-expression root, **not** one
progress unit for all its independent sources. `OriginalObligationId` is a
functional pair of that frozen family ID and a source origin (or its explicit
incoming case). Creating or querying this value allocates no source/target-pair
record. Thus independent readers have different obligation IDs even when their
source expressions are shared. Alternative writers have separately guarded IDs,
not contradictory unconditional demands. Request grouping cannot change them.

Both identities are stable within this immutable analysis, independent of
descriptor creation, query order, selected synchronization and legacy pair
materialization. A retained unique universe token rejects handles from another analysis, including
a new analysis allocated at the same address.
IDs are not serialized, cross-build identifiers. The original structure and
translated effects retain their existing lifetime/immutability contract.

The family's full key preserves the physical cell, RAW/WAR/WAW or typed kind,
source and consumer access roles, original consumer and phase, completion scope,
original owner, occurrence interpretation, original-program version, starting
and stopping boundaries, and stop inclusion. Typed families additionally retain
the original SSA value and prerequisite cause. The native adapter uses the accepted step-4 scoped factored projection for a
supported consumer, including loop-body projections with opaque nested repeats.
Unresolved projections retain the conservative original-control marginal view.
The key retains the step-2 original-program snapshot identity and revision;
invalidating that snapshot invalidates these queries. Incoming scoped histories
are explicit interface obligations, not proofs of cross-iteration transport.

`membership(family, source)` tests just the requested origin and returns its
functional obligation ID via `id()`, together with a shared applicability
predicate. `membership(obligationId)` is the corresponding lookup. Its statuses distinguish Invalid, Excluded, Guarded,
and Conservative. Guarded means membership in the **modeled original-use
relation**, not a proof of exact physical geometry, endpoint availability,
exactly-once matching, or selected completion. Partial valuations remain unknown.
The fixed-use adapter canonicalizes repeated tests of the same original Boolean
by its defining SSA value; `obligationGuardValue` recovers that value. Identity uses the step-5 common value service and includes occurrence scope.
Endpoint qualification is queried separately; identity alone does not grant
availability or completion. Missing native guard mappings use marginal fallback.

`origins(family)` explicitly enumerates represented members with their predicates
and a population-completeness flag. An unresolved marginal callback leaves that
flag false. Membership does not invoke enumeration.
`obligationWitness(obligationId)` (or the family/source overload) returns the membership condition, physical cell,
source/consumer operations and shared translated-effect incidence lists, or the
original typed prerequisite. The operation records retain their engines and
physical selectors. An incoming member has no invented source operation.
Witness extraction here is semantic/effect evidence, not extraction of a proved
feasible dynamic control walk or a causal receipt.

A conditional producer choice remains one shared relationship: for
`W0; if (g) W1; B; W2`, with qualified full writes, B's producer is W1 on g and
W0 otherwise. W2's update cannot erase B's previously formed demand. `Both`
retains conjunctive requirements: independently applicable readers are not
mutually exclusive and one reader's membership does not cover the others.
An obligation ID is covered only after **every translated effect witness and
applicable occurrence case of that source-use relation** is covered. A family
root is collectively covered only after all its applicable member IDs pass;
covering one independent reader does not cover another. Request/descriptor
references are not additional progress units. No constructor coverage or
residual-discharge implementation is added here.

## Implementation and compatibility

`FactoredUse.h` separates the existing immutable expression records from their
producer. `FactoredProvenance.h` retains its accepted transfer algorithm; its
read/write/choice equations are unchanged. `OriginalObligations.h` supplies the
STL-only universe, predicate sharing, membership, enumeration and witness APIs.
`OriginalObligationAdapter.h` connects actual lifetime demand nodes and native
typed prerequisites to it. The adapter queries cached scoped projections per target/cell role and reads
their stored demand handles. Native guard mapping is checked once per arena.
It does not first enumerate every source/target pair.

`OriginalLifetimes` adds direct marginal row membership, nonempty and role-effect
queries. Its existing pair and subscription builder is deferred until an
explicit compatibility query. Those two legacy populations are built together,
retaining their original indices. `supportBetween` also materializes that view
before accessing its affected requirements. Legacy direction/alternative-source
indexes are similarly lazy in ProgramAnalysis. The production FrontierSynch
entry reads the new family universe rather than forcing pair expansion just to
print a diagnostic. Construction still reports that it is not implemented.

The four marginal provenance matrices and their fixed points are unchanged;
this increment does not claim to remove their formation cost. Factored core
formation has its existing per-cell arrays and syntax traversal. Map interning
and membership caches have their own costs. Membership visits the relevant
shared source expression without valuation products. Explicit enumeration can
still be quadratic across many origins/queries. The counters distinguish
families, inspected origin nodes, requested memberships, enumerated output,
requested witnesses, and materialized legacy pairs. No general linear bound is
claimed for the whole native analysis.

## Preserved unresolved premises and later gates

Step 3 supplies shared semantic coverage and exact-cell geometry qualification
for supported full writes. Unknown coverage retains older origins. This is an
optional precision interface, not an instruction admission list.

Step 4 supplies writer and reader entry histories, guarded demand expressions,
and scoped projections. A nested repeated region remains opaque where its
transport is unproved. D1/D2/D4 matching and cross-entry transport still belong
to steps 7–9. Unknown membership never becomes a negative result.

Step 2 supplies original cuts and qualified interval keys. Step 5 supplies the
four endpoint-availability outcomes and typed prerequisites. These services do
not establish that an internal phase delimiter is an executable endpoint.
Typed obligation source identity retains the producing phase even when several
phases share one executable completion cut.

Uniform subdivision invariance across different input cell partitions still
requires the stipulated semantic-family mapping. IDs here remain stable for the
fixed imported partition; this is not a proof of cross-partition invariance.

Request-group and descriptor formation belongs to step 12. The complete shared
source/support subscription closure and consequence service belong to step 13.
The compatibility source list remains available on explicit query; it is not a
claim that preparation of every later descriptor/source role is complete.
None of these implementation gaps is reclassified as an open problem in the
paper. Complete Phase A parity remains the final integrated gate.

## Tests and review evidence

Run from the repository root:

```sh
test/standalone/run_frontier_obligations.sh
SANITIZE=1 test/standalone/run_frontier_obligations.sh
```

These execute two portable C++17 tests with warnings treated as errors. A third
integration test is registered separately in native lit:

1. Obligation-model tests: conditional producers, duplicate registrations and
   descriptor references, preserved demands, shared guards, incoming cases,
   weak writes/RMW, typed-only deadlines, differing interval/version keys,
   unresolved marginal callbacks, foreign IDs and malformed records. The small
   independent-reader check evaluates 256 valuations; a separate 10,000-reader
   shared-DAG test does not enumerate valuations or all source/target pairs.
2. The production factored transfer connected to the obligation model, compared
   with an independent concrete provenance scan on 12 small executions. Both
   definite and partial conditional RMW cases are included, as are repeated
   tests of one guard and duplicate effect incidences.
3. Production `OriginalLifetimes.cpp` and `Control.h`: direct membership,
   incoming versus replaced origins, invalid/unresolved queries, lazy pair and
   subscription construction, support queried before pairs, and cyclic
   marginal fallback.

The lifetime integration test uses actual MLIR and original-structure headers,
with parsed structured-control operations and explicit synthetic effect phases.
It is `pto-frontier-lifetime-test`; it does not claim target-effect import
coverage. The native analysis runner separately exercises actual PTO imports.

The existing `pto-frontier-analysis-test --factored-self-test` additionally runs
the production-transfer/obligation concrete check. Each native corpus run now
checks the public ProgramAnalysis universe before requesting its old pair view:
no eager pair materialization, all original/typed deadlines, immutable IDs,
complete modeled origin enumeration, shared incidence witnesses, original guard
values, and explicit incoming cases. Existing whole-module IR preservation
checks remain. Current native execution results are recorded in the integrated review.

Before accepting the increment, build the actual native targets, run the
self-test and the five development inputs, run both existing and FrontierSynch
pass paths (the latter still ends in its expected construction diagnostic), and
obtain an independent review of the family interpretation, conservative entry
cases and public-query completeness. No reviewer decision is fabricated by this
record. Follow with the integrated Phase A review after the later gates land.
