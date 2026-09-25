# Phase A step 12: request groups and descriptor slots

**Integration update (2026-09-25): accepted at this component's scoped v0.44 gate.**
See [current integration evidence and corrections](frontier-synch-steps7-13-review.md).
Patch-delivery validation/status statements below describe the original submission.


Base: `4122dd1531fbdb2859bd7b27772dba56930be393` on
`codex/handoff-foundation`. Specification: supplied draft revision 0.44,
Section 4.1 (pp. 16–17), Appendix I.4 (pp. 81–82), and the step-12
acceptance criteria in the implementation plan.

**Delivery status: implementation and portable checks supplied; native
integration and independent acceptance are pending.** This is not a step-14
parity decision. The historical step-1 and steps-2–6 records are unchanged.

## Implemented boundary

`OriginalDescriptorSlots.h` is the production, MLIR-independent formation core.
It interns original boundary and fact records, retains shared `Both`/`Choose`
expressions, and forms corresponding slots without distributing independent
choices. All slots share their leaf's immutable obligation references:

| Slot | Source | Target |
| --- | --- | --- |
| Preferred | Qualified exact, otherwise qualified cover | Qualified exact, otherwise qualified cover |
| Source-cover | Declared covering answer | Preferred target |
| Target-cover | Preferred source | Declared covering answer |

An unavailable component remains explicit in the declared slot inventory.
`OriginalRequestGroup::descriptors` contains only fully described roots;
`declaredSlots` also retains unresolved records. Neither `NoHit` nor an
unavailable boundary deletes a leaf's original obligations. Empty original
expressions, rather than failed placement queries, eliminate an arm. Duplicate
root results share one entry and retain slot aliases in fixed slot order.
An empty root is not a probe candidate.

The records retain both original intervals and cuts, original-program versions,
applicability expressions, occurrence references, support-role links and shifts,
all four observation outcomes, completion prerequisites, execution multiplicity,
no-hit conditions, extra executions and extra original work. These are original
requirements and facts, not event bindings, selected proof contexts or acquired
completion. A fully described record still needs ordinary packet qualification,
including uniformity, matching, support, actual coverage and event reuse.

`OriginalRequests.h` and the private `OriginalRequestAdapter.h` connect this core
to the step-6 obligation model. Groups retain original family IDs; an individual
source reference exposes the existing functional `{family, origin}` ID rather
than allocating an obligation. An opaque marginal reference denotes its entire
unexpanded source relation, not one residual progress unit or a fictitious
incoming producer. Distinct source engines remain separate groups. Group leaves
retain their own support interpretations and alternative source milestones.

`ProgramAnalysis` forms requests after freezing the original obligation model.
Its optional formation-time boundary provider receives the existing original
analysis and the referenced obligation; it supplies all four directional answers
at once. It is not retained in the result. The resulting `requests()` view is
const and frozen before construction can start. Its group/reference queries
reject stale original snapshots. No binding failure can call a boundary provider
through this view or change its descriptor population.

The default provider adapts the existing immediate-cut and fixed-visit queries,
and uses the common original-value qualifier. The `frontier-synch` pass reads
and reports this inventory before its existing construction-not-implemented
failure. It does not allocate events, emit commands or change shared effects.

## Dependencies not concealed by this increment

The integrated default provider uses steps 7–11: hazard-specific D1 and
independent source/target covering queries over singleton role intervals. It
retains a positive D2 relation when available, but per-bank descriptor endpoint
domains remain unresolved. Opaque repeated-program families retain their
whole-prefix obligation and do not turn a singleton cover into family coverage.
Native typed/incoming endpoint matching also retains its missing premises.

Other occurrence providers may supply their qualified original relation through
`occurrenceQualified` and `occurrence`; these records do not implement D2/D4 or
certify the matching of eventual physical events. Qualified support links are
retained, but their derivation and subscription closure remain later work.

The default observation bridge checks nonconstant conditional expressions
conservatively: it requires the referenced values in both arms to be qualified
at the endpoint. It does not introduce a new conditional-replay solver. The full
condition DAG and native qualification results remain accessible; a displaced
endpoint whose qualification needs more precise conditional evaluation remains
unresolved. The formation algebra can consume independently qualified answers
without changing this default bridge.

Step 13 must prepare source/deadline adjacency, subscriptions, finite support
closure and consequences using these retained references. `requests().complete()`
means formation finished for the frozen original snapshot, not that every
boundary query succeeded or that source subscription closure is complete.

## Historical validation performed in the patch delivery environment

The portable test compiles the actual new production core against the unchanged
repository `OriginalProgramPoints.h` (blob
`9b7e4f47da118a284f3ec3d4923a2be2bb514352`). No MLIR/compiler stub is used.

Executed successfully:

- GCC 14.2, C++17, `-fno-exceptions -Wall -Wextra -Werror -pedantic -O2 -DNDEBUG`.
- Clang 17, the same release configuration.
- GCC address and undefined-behavior sanitizers. Leak detection was disabled
  (`ASAN_OPTIONS=detect_leaks=0`); leak checking is not claimed.

Each run reports **34,353 always-active assertion checks**. The tests include
mixed exact/covering directions; pointwise mixed-form choice arms; all 1,024
valuations of ten optional readers; unavailable and independently enabled
observations; unresolved arms; no-hit answers that retain obligations; support
links, shifts and extra work; duplicate slots; invalid directional inputs; and
refused post-freeze mutation. The count is assertions, not independent fixtures.

The 4,096-optional-reader formation test does not enumerate valuations. It forms
12,288 input nodes and 36,862 descriptor nodes with three roots, checking one
formation visit per input and an explicit linear node-population bound for this
fixture. This is not a bound on all native queries or compilation time.

Native `pto-frontier-request-test` is supplied and registered in CMake/lit. It
uses real MLIR control with supplied semantic phase effects and boundary answers,
checks obligation IDs, independent reader engines, frozen provider calls, stale
snapshots and IR preservation, and also exercises the default direct adapter.
**It has not been compiled or run here.** Neither have the full compiler,
native lit suite, development corpus, existing autosync mode or device tests.
The local environment has no LLVM/MLIR development toolchain or full checkout.
Existing-file edits were formed from pinned source contexts retrieved through
the GitHub connector; patch syntax/context checks are not a full-checkout build
or an integrated application test.

## Work and representation accounting

Slot formation visits each newly registered input once and creates at most three
result nodes for it. Structural interning uses ordered maps, not a constant-time
oracle. Variable-size endpoint, observation, support and witness records and
comparisons must be charged separately. The three-root bound is not a bound on
endpoint output or proof-context work.

The native adapter walks each requested factored source DAG with a local memo
and materializes references for its source leaves. It calls lazy membership for
those leaves; it does not enumerate guard valuations, call `origins()` during
formation, or build the legacy compatibility pair inventory. This materialized
source/family work can be output-sensitive and is not claimed linear in compact
program size. Marginal populations stay opaque until an explicit original
membership/enumeration query is requested.

## Reproduction and remaining acceptance gate

From a checkout with this patch applied:

```sh
bash test/standalone/run_frontier_descriptors.sh
CXX=clang++ bash test/standalone/run_frontier_descriptors.sh
ASAN_OPTIONS=detect_leaks=0 SANITIZE=1 bash test/standalone/run_frontier_descriptors.sh
cmake --build <build> --target pto-frontier-descriptor-core-test pto-frontier-request-test
llvm-lit -j1 <build>/test/lit/pto/frontier_synch_descriptors.pto
```

Use the checkout's configured lit test path when it differs from the example.
Before accepting step 12, compile/run the native test and existing focused Phase A
suite, inspect the default adapter's actual supported and unresolved answers,
and obtain independent review against Section 4.1/I.4. Integrate the later
boundary providers without moving formation behind a binding probe. No external
reviewer was run for this delivery; this document records scope and self-review,
not independent acceptance.
