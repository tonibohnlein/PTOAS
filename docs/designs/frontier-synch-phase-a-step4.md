# Phase A step 4: factored original-use service

> **Integration update, 2026-09-24:** this increment has passed local native
> validation and independent review at its stated scope. See the
> [integrated review](frontier-synch-steps2-6-review.md) for fixes and evidence.
> The candidate/pending statements below record the original patch submission,
> not the current integration status. Later Phase A gates remain open.

Patch base: `0a38c8e9f4bd4852177c1dd6f6e0d0ee26b6ca16`, including the
accepted transfer core in `9bba6552055d5386ff7242c58031b0412885a0d6`.
Specification: draft v0.44, Sections 3.3–3.4 and Appendix I.1/I.6; those
operative analysis sections are unchanged from v0.43.

**Review status: proposed increment, not independently accepted.** The portable
production transfer passes its concrete-reference checks. Native integration
fixtures are supplied but have not been compiled or run in the patch-authoring
environment. The full Phase A parity inventory is a separate step-1 deliverable.
This record does not certify complete Phase A or a working constructor.

## Implemented path

`SyncInput` → `OriginalStructure::Access` → `FactoredProvenance` projection →
`FactoredUseBuilder` → `OriginalLifetimes::factoredAt` →
`ProgramAnalysis::originalUsesAt` / `interpretAt(...).factoredDemand`.

The accepted transfer equations now live in the standard-C++ `FactoredUse.h`.
The native adapter still consumes the same imported effects. This factoring
lets an independent concrete scanner execute the actual production transfer
without an MLIR substitute or a second implementation of instruction semantics.
No shared instruction effect, storage allocation, payload, insertion cut,
selected command, key assignment, or existing-autosync rule is changed.

### Shared relationships and old-state demands

An append-only arena owns constant-arity `Both`, `Choose`, access, input, and
demand nodes. Each access has one retained metadata record containing all its
read/write incidence indices into the original translated effect vector.
Incoming expressions and composed source witnesses are shared by reference;
destroying an earlier result does not discard a witness still referenced by
its arena. The source IR and shared `SyncInput` still must outlive native use.

A read records RAW against the incoming writer expression. A write records WAW
against old writers and WAR against old readers before changing either state.
RMW also records RAW. A qualified definite write replaces writers and clears
forward readers; a possible/partial write retains old histories. The backward
transfer preserves the read half of RMW at the write boundary, as did the
accepted core. A later full write cannot mutate previously formed demands.

`FactoredDemandView` identifies the hazard, target metadata, old-state source
root, and the target's complete shared applicability expression. Consumers do
not accidentally interpret an arm-local demand as unconditional. Original
simultaneous requirements use `Both`; incompatible choices use `Choose`.
Nested applicability short-circuits a test which is undefined outside its arm.

### Guard identity and input interfaces

Native tests are interned by the defining `Value` identity and its defining
repeated scope, not by `scf.if` site. Several tests of the same original value
share a node. Distinct values remain distinct even when they have the same
spelling. A function/outer-scope value used inside a loop retains that defining
scope; a body-defined value identifies its repeated owner. Single-visit
interpretation, not a fabricated global iteration number, qualifies this key.
Endpoint availability, carried-value transport, and arithmetic are not inferred.

`FactoredUseInterface` supplies four roots in one shared arena: incoming
writers/readers and following writers/readers. The latter provide the backward
continuation. A default child interface contains explicit input parameters on
both sides. Invocation entry also retains parameters rather than silently
assuming fresh histories; a caller may explicitly supply empty roots when its
entry contract permits them. Default invocation following roots end at the
original function exit, not at an unspecified caller continuation.

The arena frame carries the immutable original instance, its version, cell,
owner, and invocation/for-body/while-before/while-after interpretation. Mismatched
frames, cells, and expression sorts are rejected. The integrated native service retains step 2's shared original-program version
token in every frame and checks its captured snapshot before serving cached
results. A version change requires a new analysis instance. Native tests reject
stale cached results and supplied arenas from an earlier version. Synchronization state is absent from these records.

### Repeated regions: retain useful facts without inventing a summary

`originalUsesAt(op, cell)` queries the invocation or the nearest enclosing
single body visit. While-before and while-after have separate scopes and
arenas; the before visit is not conditioned on execution of the after region.
Queries in one scope/cell share one cached projection. Explicit applications
use the supplied input roots and are never cached under default-input keys.

Inside a projection, an entire nested repetition remains a repetition. Its
complete original may effects can prove it irrelevant to this cell, in which
case the provenance transfer is identity. A read-only repeat preserves the
writer expression but introduces an unresolved reader interface. A writing
repeat retains conservative histories and its unresolved generated origins.
Both directions retain the required input/output interface and missing premise.
A following local full write can restore subsequent provenance precision,
without deleting earlier unresolved demands. No repeat is evaluated once and
then declared summarized.

`complete` means the fixed-use transfer has been formed, conditionally on its
explicit inputs; it is not a full-cell, boundary-availability, recurring-use,
or completion certificate. `FactoredDemandView::hasUnresolved()` and node flags
retain unresolved components independently of still-usable components. An
incoming parameter is not a completed access and is not silently empty.

This is the step-4 bridge to later summary procedures, not their implementation.
Matching actual producer/reader occurrences across backedges or child re-entry
still requires the supported interfaces of steps 8–9. The original marginal
requirements, translated witnesses, source subscriptions, and typed demands
remain unchanged and available while those premises are missing.

### Public consumers and deferred identities

`interpretAt` now obtains the target's actual scoped service and hazard view,
rather than an often-empty whole-function DAG. The corpus runner checks that
view and its target witnesses. `originalUsesAt` is independently queryable for
incoming-only requirements, even when the old marginal table has no local-source
edge to interpret. Native fixtures visit every imported access through this API.

A factored demand view does not certify that any particular conservative
marginal source is applicable. Guarded source membership and alternative-source
qualification still belong to the later obligation/D1 steps. The patch neither
removes those marginal requirements nor relabels their occurrences as exact.
Stable obligation IDs, grouping, descriptor generation, and complete prepared
source/support closure remain steps 6, 12, and 13.

## Tests and measured formation work

`pto-factored-provenance-test` compares the generated expression interpretation
with an independent scan of the selected original accesses, including forward
and backward histories and all generated hazards. On exact-cell fixtures it
also compares the closure with all-pairs conflicts. The suite includes:

- Conditional overwrite and RMW, a guard reused at multiple sites, different
  defining scopes, and an inner test undefined on an outer bypass.
- Incoming writers and readers, backward following inputs, reference-sharing
  across two acyclic pieces, witness lifetime, and mismatched input frames.
- Weak/partial writes and RMW; a separate two-byte scanner checks four concrete
  write footprints (empty, either byte, both bytes) against retained origins,
  readers, and conflict coverage.
- Cell-disjoint and touching repeats, preservation of writers through read-only
  repetition, later restoration of precision, separate parametric body visits,
  and malformed/unknown projections.
- Forty deterministic generated small programs under eight guard valuations,
  plus directed cases: **343 concrete executions** in total.

Large independent-reader formation performs no valuation or origin-set
materialization. These are production-constructor counters, not a timing claim:

| Optional readers | Syntax nodes | Input nodes | Added expression nodes | Constructor calls |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 3 | 14 | 43 |
| 8 | 27 | 10 | 91 | 239 |
| 64 | 195 | 66 | 707 | 1807 |
| 512 | 1,539 | 514 | 5,635 | 14,351 |
| 4,096 | 12,291 | 4,098 | 45,059 | 114,703 |

Each admitted syntax node is visited once per direction. Supplied arenas are
not copied. For this stress family added nodes equal `11*m + 3`, and counted
constructor calls equal `28*m + 15`. Constructor calls include append and
constant-folding attempts. The restricted claim is the Appendix I.6
constant-arity node-formation/storage bound `O(q+b)`, after projection and guard
qualification. It is not a worst-case bound for hash-table operations, native
projection preparation, all queried cells/scopes, symbolic recurrence, or all
materialized output. The native adapter exposes its own syntax/effect/guard
preparation counters separately. Small-test output materialization was counted
separately: 10,283 origin members and 25,281 demand members.

The portable suite passed with GCC 14.2 using ASan/UBSan, GCC 14.2 release,
and Clang 17 release. All configurations use C++17, `-Wall -Wextra -Werror`,
`-fno-exceptions`, and `-fno-rtti`. Release checks remain active under `NDEBUG`.

`FactoredProvenanceNativeTests.cpp` supplies real PTO import fixtures for shared
SSA guards/conditional writes, in-place RMW, for-body interfaces, and distinct
while-before/after inputs. These use `SyncInput`, the physical importer, and
`ProgramAnalysis`; they do not mutate coverage flags. They check original IR
preservation and shared translated witnesses. **These native tests have not
been run in the patch-authoring environment.** The original `--factored-self-test`
entry now runs them; its older synthetic core cases are covered by the larger
portable suite. The new lit test and both executable dependencies are registered.

## Reviewer gates still open

| Gate | Required action; not a claim of an open research problem |
| --- | --- |
| This increment | Compile and run the native self-test, run the existing regressions and five development kernels, independently review the full step-4 contract, and fix findings before recording acceptance. |
| Step 3 | Integrated qualified full-cell overwrite evidence is consumed here; its native coverage and RMW tests pass. Unsupported write geometry remains conservative. |
| Step 5 | Step 2 snapshot identity is integrated. Independently qualify original values at proposed endpoints. Original expression formation does not make a future value observable. |
| Steps 6/7 | Attach stable original obligation identities and guarded source-membership/D1 queries to these shared demand roots. Preserve incoming-only and conservative cases. |
| Steps 8/9 | Supply and apply the supported repeated-region succession/demand summaries and qualified re-entry transport. A parametric body visit alone is not that proof. |
| Steps 12/13/14 | Finish descriptor/preparation consumers and perform the final integrated parity review. Existing source subscriptions are preserved, not declared complete for the future repertoire. |

There is no independent reviewer acceptance in this record. A successful
portable test or an interface returning a parameter is not acceptance of an
unimplemented specified repeated rule.

## Reproduce

From a real checkout with this patch applied:

```sh
c++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror \
  -fno-exceptions -fno-rtti -Iinclude \
  tools/pto-test-opt/pto-factored-provenance-test.cpp \
  -o /tmp/pto-factored-provenance-test
/tmp/pto-factored-provenance-test

cmake --build build --target pto-frontier-analysis-test pto-factored-provenance-test
build/tools/pto-test-opt/pto-frontier-analysis-test --factored-self-test
cmake --build build --target check-pto
```

Use the checkout's configured build directory in place of `build`. Native
acceptance additionally requires the existing five-kernel corpus, with semantic
answers checked rather than only successful traversal. No Phase B or device
execution result is supplied by this increment.
