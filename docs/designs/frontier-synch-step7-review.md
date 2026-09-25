# Phase A step 7: conflict-specific D1 correspondence

**Integration update (2026-09-25): accepted at this component's scoped v0.44 gate.**
See [current integration evidence and corrections](frontier-synch-steps7-13-review.md).
Patch-delivery validation/status statements below describe the original submission.


## Scope and acceptance state

Target: draft v0.44 Section 3.6 D1 (pp. 11–12), using the original provenance
contract of Sections 3.2–3.4 / Appendix I.1 and the independent endpoint
qualification contract of Section 4.1. Base inspected:
`4122dd1531fbdb2859bd7b27772dba56930be393`, `codex/handoff-foundation`.

This is a candidate implementation, not an accepted increment. The portable
checks below were executed. The native tests and the final independent step-7
review still have to run. Neither this document nor a successful portable test
claims all Phase A parity, a completed constructor, or device correctness.

## Procedures and consumers

`FixedVisitSources.h::queryFixedVisitSources` consumes a target's immutable,
hazard-specific old-state `FactoredDemandView`. The production provenance
transfer remains the source of truth. A read does not end a producer's relation
to a later reader. A definite replacement selects a new writer under its
original conditions; a partial/possible replacement keeps simultaneous origins.
WAR queries preserve independent reader obligations.

Applicability is propagated down the shared source DAG in reverse topological
order. Paths to a shared node are combined before processing it. `Choose`
retains one original guard interpretation, whereas `Both` propagates simultaneous
requirements. No guard valuation, reader-subset product, or candidate plan is
enumerated. Output is normalized by original operation ID; the explicit incoming
entry is last. The original arena and previously generated demands are unchanged.

Each result retains its original frame, applicability, source witnesses and a
read-only predicate arena. Each origin has a full membership condition and
separate local source and target conditions. Local cofactoring uses only facts
entailed by that endpoint's original lexical activation. Consequently a guard
implicit at a branch-local producer is not silently treated as known at a
post-join consumer. There is no unrestricted Boolean minimizer or history counter.

`OccurrenceQueries::fixedSourcesAt` adapts these records to physical roles and
legal immediate source/target cuts. It calls `OriginalValueQueries` separately
at both endpoints, preserving Available, NeedsCompletion, NotObservableHere and
Unresolved with their prerequisites. An incoming entry is an interface record,
not a fabricated payload or executable publication. Fixed-role qualification
withholds varying physical selection and conservative overlap witnesses; it does
not infer a bank distance from a finite may-footprint.

`ProgramAnalysis::fixedSourcesFor(OriginalObligationFamilyId)` is the public
family-based consumer, including incoming-only families with no legacy local
pair. It validates the universe/version and occurrence interpretation without
creating or discharging IDs. `fixedVisitFor` is the compatibility pair adapter;
its cache now includes **hazard** as well as the complete original interval.
An RMW can have WAR and WAW roots under identical endpoint/cell/interval fields.
`interpretAt` retains the complete `FixedVisitCorrespondence`, not just its
status bit. ProgramAnalysis shares its existing lifetime/value services with D1.

### Meaning of positive and unresolved results

`FixedVisitSources::complete` means the represented original source relation is
available. It does not mean an incoming source is bound, effects are exact, or
endpoint predicates are executable. `exclusiveWriters` distinguishes an
alternative-source family from simultaneously retained weak-write histories.
`FixedVisitCorrespondence::exact` is conditional on its retained predicates and
endpoint prerequisites; it is not an unconditional edge or acquired completion.
Predicate IDs belong to `alternatives->sources.predicates`, not the reader or
obligation predicate arenas.

A local single visit of a supported ForBody/WhileBefore/WhileAfter projection
can supply D1 facts even when the whole function contains repetition. This does
not transport them across a backedge, an enclosing continuation or a restarted
child. D2 and D4 remain steps 8 and 9. Exact structural first/last queries,
directional covers, descriptor slots and preparation remain steps 10–13.
Original marginal obligations and typed/control prerequisites are retained on
all unresolved cases.

## Tests and actual validation

Portable production-core regression:
`test/standalone/frontier_d1_sources_test.cpp`, also registered as the CMake target
`pto-frontier-d1-core-test`. The executable uses the branch's FactoredUse and
OriginalProgramPoints headers, not a mock provenance implementation.

The independent scanner interprets original control on small fixtures and tracks
old writers/readers before each access. It checks multiple readers, conditional
replacement, no-producer paths, partial writes/RMW, nested/reused guards and
incompatible arms. It separately compares both endpoint execution domains,
including executions with a producer but no optional target. Its weak-write
expectation is conservative origin retention, not a byte-level hardware oracle.
A 256-optional-reader case checks shared expression growth without enumerating
its valuations. Original arena sizes are checked before and after queries.

Executed in the patch environment:

- GCC and Clang portable runs: **1,009,798 checks, 1,311 concrete executions**.
- Clang AddressSanitizer plus UndefinedBehaviorSanitizer: the same checks pass.
  `ASAN_OPTIONS=detect_leaks=0` was used; leak checking is not claimed.
- Release-style (`NDEBUG`, exceptions disabled) portable compilation/run.

The query's source-DAG traversals do not repeat per origin. Predicate import,
constant folding, endpoint cofactoring, ordered-container work and materialized
origin/condition output remain additional costs. This does not claim a global
linear bound for arbitrary original programs or all target queries.

Added but **not executed here**: `pto-frontier-d1-test`, using real MLIR control
and the production `ProgramAnalysis`/lifetime/value/occurrence interfaces with
explicit semantic effects. It asserts source/target qualification separately,
late guards, unavailable phase cuts, incoming-only public families, conditional
replacement, partial origins, incompatible arms, a WAR/WAW cache collision, local
loop visits, snapshot invalidation and unchanged original IR. The lit entry is
`test/lit/pto/frontier_synch_d1.pto`. These native fixtures are not PTO effect-import
or device validation. No shared instruction-effect implementation is changed.

## Reproduction and remaining review gate

```sh
bash test/standalone/run_frontier_d1_sources.sh
CXX=clang++ ASAN_OPTIONS=detect_leaks=0 SANITIZE=1 \
  bash test/standalone/run_frontier_d1_sources.sh

# Use the repository's configured LLVM/MLIR build directory.
cmake --build "$PTOAS_BUILD_DIR" --target \
  pto-frontier-d1-core-test pto-frontier-d1-test pto-frontier-analysis-test
PATH="$PTOAS_BUILD_DIR/tools/pto-test-opt:$PTOAS_BUILD_DIR/bin:$PATH" \
  llvm-lit -v test/lit/pto/frontier_synch_d1.pto
```

Before accepting step 7, run the native fixture and existing focused Phase A lit
suite, including the five development kernels and both actual pass modes. Review
PA-040–PA-042 against the draft's premises and inspect the retained predicates,
not just non-Unknown counts. Record any correction and independent acceptance
here. Step 14 must still recheck the integrated system after steps 8–13.
