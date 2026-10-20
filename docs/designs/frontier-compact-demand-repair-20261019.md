# Compact regional Persistent GEMM: representation repair

This follows the instrumentation checkpoint, with Section 5 as its acceptance
contract. Allocation algorithms, palettes, expansion limits and synchronization
placement policies are unchanged. The user-owned Sections 5.4/5.5 design is not
edited. All physical claims below are relative to the supplied access/layout
model and `gm-alias=may-not-alias`; they do not qualify native instruction
footprints. No device compilation or execution is included.

## Failure and repair scope

The four-stage K loop was already recognized as a two-phase compact region
with 80 occurrence types, including its boundary slices. A failed rotating
subslice (`unsupported-slot-footprint`) caused a finite **kk-body** scan; it did
not require enumeration of K. The enclosing wave composition subsequently lost
that compact child:

1. The explicit output-store run rejected its discharged GM writer because the
   adapter admitted only globally read-only discharges. Its retained original
   maps already supported the existing cross-phase independence and disjoint-
   writing-visits check.
2. The joint repeated-storage check then required periodic/invariant geometry
   even for GM readers already certified conflict-free against the whole input.
3. The phase selector fallback rejected a local-memory query merely because an
   unrelated GM residual lacked selectors.

After those export failures the wave adapter enumerated K as a finite body:
449 potential sites, 1,317 generators and 379 reduced records per nonempty phase
at K=32. The arithmetic fallback was not evidence that compact mathematics was
inapplicable: it was reached after a missing regional capability.

In the one-/two-stage derivatives the constant kk loop had been unrolled. The
same phase algorithm rejected the now-flat K loop solely because it had no
nested loop. It consequently reached general arithmetic storage-selector
construction. Both derivatives previously hit the unchanged 120-second
campaign watchdog before allocation.

The repair remains inside the existing services:

- `Shared/PhasedRecognition.cpp::phasedChild`: admit flat finite bodies under
  the same period, control, descriptor, prerequisite and original-coordinate
  checks. Nested loops still require their additional bound checks.
- `Structured/NumericTemplateRegional.cpp::specializedExplicitRunRegional`:
  use the existing original-map GM discharge certificate, including disjoint
  repeated writing visits, rather than substituting the weaker local partition.
- `Shared/RepeatedStorageGeometry.cpp`: preserve whole-input conflict-free GM
  reader evidence when checking residual completeness and joint separation.
  The original access boundary remains present; this is not a selector proof.
- `Shared/RepeatedStorage.cpp` and `Shared/RepeatedPhases.cpp`: reject queries
  in an unclassified residual's own/may-alias domain, while allowing queries
  whose address space or base is proved disjoint. Projected residual selectors
  exclude persistent effects only when the caller already supplies those
  certified persistent selectors.

No new mathematical class, backend, period discovery, ID allocator or kernel
recognizer is introduced. Unknown exact selector answers remain unavailable.
The outer replay export and coalescer cost remain separate follow-ups.

## Independent contracts

`CompactRegionDemands.h` traverses actual retained sequence crossings, repeated
crossings, concrete retained edges and supported periodic/guarded records. It
maps occurrence coordinates back to original payload identities, evaluates
record guards, and deduplicates the resulting completion-to-start pairs. It
never reconstructs covers from reachability queries. An unsupported leaf or
unresolved predicate is a test failure, not an empty record inventory. The
numerical-body endpoint adapter exposes its existing internal crossings through
a weak observer; no new analysis is performed for inspection.

The Python oracle starts from the supplied NZ byte-layout formulas, enumerates
all original conflicting pairs and native start/completion edges, and computes
closure and nonnative covers independently. The same fixtures check concrete
access extraction, all four event-kind queries, first/last writers and boundary
readers, payload preservation, matching and consumption-before-republication.
This independently checks extraction relative to the stated layout; checking
commands against the same compiler-generated generators would not do so.

The zero-inner-body case deliberately retains two extra completion-to-start
prerequisites caused by the existing barriers. It remains a coverage regression,
not an order-equality case. The production guarantee remains
`realization_order=covers-selected-order`; equality on other finite instances is
not a uniform leastness proof.

## Reproduction

Use the source-pinned incremental build, LLVM 19 shared libraries, one process
at a time, and at most two aggregate intensive workers. Substantial artifacts
are under `.local/compact-demand-repair/`, not `/tmp`.

```bash
OPT=<build>/tools/pto-test-opt/pto-test-opt
PROBE=<build>/tools/pto-test-opt/pto-sync-input-test
INPUT=test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto
python test/lit/pto/Inputs/check_compact_persistent_contract.py \
  --optimizer "$OPT" --trace-tool "$PROBE" --input "$INPUT" \
  --output small-contracts --static-stages
python test/lit/pto/Inputs/profile_compact_persistent.py \
  --optimizer "$OPT" --input "$INPUT" --output scaling --repetitions 1 --timeout 25
python test/lit/pto/Inputs/check_compact_vector_baselines.py \
  --optimizer "$OPT" --trace-tool "$PROBE" \
  --prepared-root test/npu_validation/benchmarks/auto_sync/prepared \
  --output vector-baselines --cases elementwise_pipeline group_norm
# Added after the first generalization checkpoint:
python test/lit/pto/Inputs/check_compact_vector_baselines.py \
  --optimizer "$OPT" --trace-tool "$PROBE" \
  --prepared-root test/npu_validation/benchmarks/auto_sync/prepared \
  --output cumsum-baseline --cases gdn_chunk_cumsum
```

Production requests and their profiles are separate invocations. Their times
include parsing, selected mathematics, exports, endpoint preparation and output;
they must not be called minimum-demand generation times. No audit-mode timing
is substituted for production timing. The vector driver uses the original
prepared input for stage costs; cumsum semantics use two chunks of three
scalars while retaining its supplied tile/storage layout. This bounded oracle
supports only the listed prepared vector syntax and refuses unsupported forms.

Final results and source/binary pins are recorded in the companion JSON after
the consistency rebuild and fresh qualification. Earlier diagnostic runs in
this artifact directory are preliminary, not the final qualification receipt.

## Qualified results

The ordinary input retains the existing two-phase K analysis through the outer
four-phase composition. Its nonempty outer description has 332 occurrence
types. The largest scanned finite phase body has 14 sites; the retained K
interface has 80 types across its startup/core/tail slices. Increasing K through
4, 8, 16 and 32 does not increase those retained sizes. K=2 has a smaller clipped
boundary description. Constant kk-stage work is still scanned explicitly; this
is not a claim of compactness in every independent input dimension.

The one-/two-stage scaling inputs are byte-identical to the separately checked
unrolled semantic fixtures (hashes in JSON). Both now complete the ordinary
logical path and six-ID physical allocation. Their outer descriptions contain
116 and 188 types respectively; four stages use 332. These inputs formerly
reached the arithmetic-selector watchdog. Watchdog results are censored, so no
exact speedup ratio is claimed.

Qualification:

- Seven of seven focused lit tests pass, including sequence analysis, bridge
  work, repeated region/storage, finite allocation, allocation profiling and
  the new static-stage contract checks.
- The standalone repeated read-only storage suite passes. It includes same-base
  GM refusal, `MayAlias` cross-base refusal and disjoint local-domain success.
- Ten independent small semantic cases pass: actual retained covers, required
  closure, four event-kind regional queries, selectors, concrete supplied-layout
  accesses, executed matching and physical reuse. The eleventh outer-replay case
  remains explicitly unavailable at exact query export; it is not counted as
  passing.
- The zero-inner-body case still has two extra completion-to-start prerequisites
  from the established barriers. No barriers were removed to meet equality.
- Ordinary Persistent logical insertion, six-ID allocation and C++ source
  generation succeed. No device compiler or device execution was run.

The scalar sibling test had an outdated refusal expectation: existing scalar
hardware protection already permits that parameterized pair. Its replacement
checks no added storage crossings, no relational fallback, all four native
query combinations at trip counts 0/1/5, exact first/last writers and suffix
readers, and absence outside the access domain. This is a test correction, not
an additional recognition route. The older pinned build used while diagnosing
that expectation is not an immediate-HEAD conformance baseline.

The inserted ordinary logical function has 134 static SETs, 137 WAITs, two
barriers, and 3,383 arithmetic operations including original arithmetic. These
are static inventories, not counts of executed communications or a proof of
smallest control code. Compact retained mathematics does not automatically
minimize endpoint code. Stage launcher times and exclusive ledger aggregates
are in JSON. Producer-attempt timings include qualification, construction and
requested exports; they are not pure reducer times. Some finer extraction and
export costs are not separately timed. No broad speedup claim follows from this
single-run checkpoint.

## Generalization checkpoint

The original prepared inputs are used for costs and stage outcomes. The
independent layout model checks blocks 0/1 and subblocks 0/1. The cumsum semantic
derivative uses two chunks of three scalar values. Matching and physical reuse
are reported separately from logical coverage.

| Case | Checked active payloads | Generated commands | Expert commands | Generated result | Physical result |
|---|---:|---:|---:|---|---|
| elementwise_pipeline | 16 | 25 | 29 | Covers the modeled event order exactly | Six-ID success and causal reuse check |
| group_norm | 78 | 96 | 36 | Covers the model, with two extra C-to-I pairs | Missing source-tuple-arity certificate |
| gdn_chunk_cumsum, small | 19 | 7 | 9 | Covers the modeled event order exactly | Six-ID success and causal reuse check |

Under this same model, the elementwise expert adds six C-to-I pairs. GroupNorm's
expert leaves 306 required C-to-I pairs uncovered on each checked active
instance. The small cumsum expert leaves 16 uncovered and its repeated V-to-S
ID lacks a causal consumption-before-republication witness. These findings
prevent treating those expert plans as native-qualified golden references;
they are not device miscompilation claims. The generated logical plan's
matching remains checkable even when its physical certificate is unavailable.

The source/ledger trace identifies regional sequence composition plus concrete,
numerical/rotating slice and arithmetic exports, rather than one class for each
whole kernel. The retained inventory reports all available counts without
constructing another export. Callback-only slice adapters currently omit some
producer/reducer metadata: their sizes and costs are recorded, while their
missing generator counts remain null. GroupNorm also uses numerical phase
bodies and repeated boundaries; cumsum retains a five-type arithmetic child but
its current boundary representation has 2,048 storage cells. Neither case
justifies a new recognizer in this experiment.

## Remaining gaps and scope

| Priority | Classification | Finding / next action |
|---|---|---|
| Repaired | Implementation / interface defect | Flat phase admission and certified GM discharge no longer force whole-K expansion. |
| Next separate follow-up | Unsupported regional export | Outer replay lacks the exact query interface. Retain the refusal and investigate its original occurrence frame. |
| Separate allocation follow-up | Missing resource certificate | GroupNorm needs an allocation rule with the emitted source-tuple arity. Failure is not an exact six-ID lower bound. |
| Qualification follow-up | Unqualified access / native assumptions | Validate instruction footprints, readiness and expert protocols independently before making native claims. |
| Instrumentation follow-up | Missing diagnostic export | Expose private slice producer/reducer metadata without running recognition or exporting new mathematics. |
| Explicitly deferred | Section 7 implementation cost | Coalescer optimization; its algorithm was not changed. |
| Manuscript qualification | Claim scope | Compactness is in the certified description. Boundary atoms and static stages remain charged; minimum demands, realization-order equality and physical feasibility remain separate guarantees. |

This establishes one compact, finite-validated path connecting the existing
Section 5.4 phase construction to Section 5.5 regional composition. It does not
establish universal Section 5 coverage or realization-order equality.
