# Return-query reuse and linked relay-quality witnesses

Base: `a0c1d761e`, 2026-09-21. This amendment hoists one repeated structural
query and reproduces the two relay-quality review witnesses. Relay selection,
physical binding, target admission and defaults are unchanged.

## Return sharing: one propagation per candidate being replaced

For a fixed Y candidate, the role graph marks Y's release and every write on
its producer engine. Those inputs do not depend on supporter X. The qualifier
now computes `nearestRoles` lazily after the first X passes the cheap filters,
then reuses that vector for the remaining X candidates. It retains one
graph-sized temporary, not a cache per candidate. A successful sharing decision
still ends the inner loop. Support-cell updates do not change the graph,
producer-write roles or Y release.

`returnSharingQueries` counts actual propagations. Candidate ordering, endpoint
positions, supported cells, capacity admission and staged protocol checks are
unchanged. This is a construction-cost change, not a synchronization improvement.

The new `returnSharingScaling` fixture reaches production candidate formation
and `qualifyCyclicFrontiers`, including closed-cohort qualification. P writes C
cells; Q reads each in two child regions, in the same cell order. All proposed
sharing pairs decline. Successful early exits therefore cannot conceal the
repeated queries. Requests match the sharing-disabled control; the two small
cases also construct identical complete command words with trials disabled.

The same test source was linked separately against the frozen pre-change core
library and the modified library:

| Candidates | Graph sites | Old queries | Hoisted queries | Old site visits | Hoisted site visits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 30 | 1 | 1 | 60 | 60 |
| 4 | 54 | 6 | 3 | 663 | 334 |
| 8 | 102 | 28 | 7 | 5,817 | 1,458 |
| 16 | 198 | 120 | 15 | 48,045 | 6,010 |
| 32 | 390 | 496 | 31 | 389,205 | 24,330 |

These are linked work counters, not timings. The amendment removes repeated
propagation inside the supporter loop; it does not establish a complexity bound
for the whole qualifier. Existing positive/chained-return tests remain active.

## Both relay witnesses reproduce through ordinary construction

`selected_relay_test.cpp` invokes `constructSelectedPlan` with recurring omission
and final helper trials disabled. It does not supply the selected endpoints.
It cold-checks the output, flattens the straight original control path, then
checks memory requirements, matching, rearming and complete payload-order sets
using `GraphOracle.h` independently of the causal evaluator.

The fixtures supply an explicit two-slot view and four eligible directed pools:
FIX→M, FIX→MTE1, M→MTE2 and MTE1→MTE2. All selected keys are virgin. This is
a linked portable-constructor reproduction, not native FIFO import qualification
or device evidence. It does not exercise nontrivial key rearming or peer progress.

### Witness A: incidental completion versus newly gated middle work

```
M:     U reads independent storage
FIX:   W writes slot 0
MTE2:  Z accesses independent storage
MTE1:  V reads independent storage
MTE2:  R reads slot 0
```

The constructor selects FIX→MTE1→MTE2, forwarding before V. That removes U's
completion from R's receipt but makes V wait for W. The supplied late M relay
does the opposite. Both are safe two-pair plans with 15 full payload relations.
Comparing their sets removes four relations and adds four: they are incomparable.
The constructed plan's complete relation set equals the review's early-relay
model, rather than merely having the same count.

### Witness B: the extra completion is required at the receiver

```
M:     read y
FIX:   write slot 0
FIX:   write slot 1
MTE1:  independent work
MTE2:  read slot 0 and overwrite y
MTE2:  read slot 1
```

The constructor first retains M's required y return, sends slot 0 through an
early MTE1 relay and slot 1 through M at its later deadline. Thus the selected
plan exactly reproduces the review's payload order: five pairs, 34 relations.
Routing slot 0 through M at the first receive instead retains the existing y
return and uses five pairs but only 30 relations: four removed, none added.
M's read of y is useful required completion, not incidental serialization.

A supplied four-pair control also removes the duplicate y return and preserves
the same 30 relations. The constructor does not select this control. Removing
the duplicate is unnecessary to demonstrate the ordering improvement.

| Comparison | Pairs before→after | Relations before→after | Removed | Added |
| --- | ---: | ---: | ---: | ---: |
| A: constructed early relay → supplied late relay | 2→2 | 15→15 | 4 | 4 |
| B: constructed early relay → supplied required-reader relay | 5→5 | 34→30 | 4 | 0 |
| B: supplied required-reader relay → shared receipt control | 5→4 | 30→30 | 0 | 0 |

Thirteen complete-leg deletion negatives isolate missing memory support while
remaining balanced, acyclic and rearming-valid. These diagnostic tests record
the current policy's limitations deliberately; when correcting selection,
update the expected selected plan while retaining both comparison models and
their independent relation-set assertions.

## What the next bounded correction must establish

The linked results identify two separate missing comparisons:

1. Which history is already acquired or actually required at the destination's
   current deadline? It must not all be penalized as incidental completion.
2. Which payloads and outward publications would a newly early middle receipt
   gate? A narrower forwarded prefix does not prove that those new prerequisites
   are already required or established.

Do not replace the score with another relation-count score or simply subtract
the current residual and claim order preservation. Witness A requires an honest
incomparability/fallback decision; witness B demonstrates a strict improvement
available without undoing an earlier selected transfer. Future selection needs
source-occurrence and exact-gap premises as well as the surrounding interface.

Read-only positive binding checks before final ranking, and interpreting the
first receipt privately before checking second-leg reuse, remain separate
extensions. Neither is implemented here. Preserve the native attention ordering
control when designing the correction, and keep whole-graph comparisons in
tests rather than adding one for every candidate.

## Validation and reproduction

Validation results are recorded in `../return-query-relay-work/` relative to the
repository. `baseline-core.a` and `baseline-selected-test` are frozen at the base
commit. `scaling-old.log`, `scaling-new.log`, and `relay.log` hold the linked
counters and actual selected endpoints. The report-only option skips the new
query-count bound so the same fixture can execute against the old library; the
normal test always enforces the bound.

```
cmake -S test/oahs -B ../oahs-m1-core-build
cmake --build ../oahs-m1-core-build --parallel 2
ctest --test-dir ../oahs-m1-core-build --parallel 2 --output-on-failure
../oahs-m1-core-build/oahs-selected_cyclic-test --return-query-report
../oahs-m1-core-build/oahs-selected_relay-test

# After completing the core jobs, rebuild the native tools.
cmake --build ../oahs-m1-native-build --parallel 2 \
  --target pto-oahs-selected-test pto-oahs-native-test pto-test-opt
```

All 24 portable suites and both native test executables pass. The positive
native shared-return witness is byte-identical to the frozen base executable;
its sharing-disabled comparison still has 10→7 pairs and identical 602 full
payload relations. Logs and plans are in `native-positive/`.

The serial corpus run reconstructs all 88 inputs / 97 functions successfully;
all 88 plans match the archived `a0c1d761e` outputs byte-for-byte. This includes
the current attention relay and joint-reader projection plans. Aggregate replay
remains 3,262,799 evaluations. No corpus function reaches a return-sharing query,
so these unchanged corpus counters do not measure the hoist's benefit; the
linked scaling fixture does. `evaluate.py`, `corpus-final/results.json` and
`corpus-final/summary.json` preserve commands, input/output hashes and counters.

No device experiment or latency claim belongs to this amendment.
