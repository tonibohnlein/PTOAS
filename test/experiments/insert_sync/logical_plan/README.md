# Native logical-constructor acceptance

[PHYSICAL_ADDRESS_RESULTS.md](PHYSICAL_ADDRESS_RESULTS.md) records the shared
cast/address, aligned-stride, safe-fallback and identity-copy fixes and their
direct all-mode, actual-lowering and default-pipeline regression evidence.

The authored-event and retirement/endpoint checkpoints are accepted; see
[RETIREMENT_ENDPOINT_RESULTS.md](RETIREMENT_ENDPOINT_RESULTS.md). Computation
and reconstruction improvements are documented in
[SCALABILITY_RESULTS.md](SCALABILITY_RESULTS.md). Two-/three-buffer strict
acceptance and occurrence-dependent slot refinement remain the next gate.
[SLOT_WIP.md](SLOT_WIP.md) records the paused slot-refinement work, its first
native precision result and the outstanding performance/coverage gates.
[SLOT_QUALIFICATION_RESULTS.md](SLOT_QUALIFICATION_RESULTS.md) records the
expanded slot campaign and mandatory guard-binding/partial-byte-overlap tests.
[EXACT_GUARD_QUERY_RESULTS.md](EXACT_GUARD_QUERY_RESULTS.md) records the bounded
cache, atom-complement and target-index changes and their rebuilt acceptance.
The [exact-query cost review](../../../../docs/designs/oahs-exact-query-cost-review.md)
defines the compact recurrence algorithms and data-structure acceptance gates;
cache improvements alone do not complete those gates.
[GUARD_WIP.md](GUARD_WIP.md) preserves the earlier guard checkpoint and its
limitations; it is not a report for the current source.

Milestone one includes direct construction from unsynchronized PTO, exact
query parity, shared native requirement export, and fresh emitted reconstruction.
The query driver links the same C++ source used by the compiler; the Python test process compares it with libisl and
explicit expected integer relations. No Python or libisl enters compilation.

Build only this small target using the established LLVM/MLIR build:

```sh
cmake -S test/experiments/insert_sync/logical_plan -B "$RELATION_BUILD" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir"
cmake --build "$RELATION_BUILD" --parallel 2
python test/experiments/insert_sync/logical_plan/check_relations.py \
  --driver "$RELATION_BUILD/pto-logical-relations-test" --output "$NEW_RESULTS"
```

Use disk-backed paths and a fresh results directory. If libisl is outside the
dynamic loader's search paths, set `PTOAS_EVENT_MODEL_ISL_LIBRARY` explicitly.
The runner preserves every request, answer, stderr and driver hash. Record the
LLVM source revision and linked library hashes alongside the results.

The cases cover carried slots one through four, partial replacement, may-write
retention, conditional first acquisitions, nested reader boundaries, exact
integer existential constraints, wrong invocations, absent maxima, continued
completion queries, selected-barrier feedback and exhausted/incompatible queries.
They do not establish native footprint correctness or emitted event safety.

The native occurrence test parses actual MLIR and retains caller phase IDs:

```sh
cmake --build "$BUILD" --parallel 2 --target pto-sync-occurrences-test
python test/experiments/insert_sync/logical_plan/check_occurrences.py \
  --driver "$BUILD/tools/pto-test-opt/pto-sync-occurrences-test" \
  --output "$NEW_OCCURRENCE_RESULTS"
```

The driver disables MLIR multithreading and the Python runner invokes cases
serially. The 28 cases include narrow integer overflow refusal, correlated
addition/subtraction, division/remainder qualification at the original definition,
signed Boolean conditions, nested loops, nonunit steps, reversed phase identities, and all
1,936 phase-pair order relations in the unchanged looping online-softmax input.
This checks input occurrence semantics, independently of selected synchronization.

## Constructor and shared requirements

```sh
cmake --build "$BUILD" --parallel 2 --target PTOASCompiler pto-logical-sync-test
python test/experiments/insert_sync/logical_plan/check_requirements.py \
  --driver "$BUILD/tools/pto-test-opt/pto-logical-sync-test" \
  --output "$NEW_REQUIREMENT_RESULTS"
python test/experiments/insert_sync/logical_plan/check_constructor.py \
  --python-root "$BUILD/python" \
  --native-driver "$BUILD/tools/pto-test-opt/pto-logical-sync-test" \
  --output "$NEW_CONSTRUCTOR_RESULTS"
```

Use the Python version matching the build. The serial constructor runner
defaults to **one_buffer, online_softmax, Q projection and QK**, the
accepted M1/M2 population. It also checks skipped/empty structured execution, deliberate
emission corruption, and explicit zero-budget strict failure versus unchanged
hybrid fallback. `--case` selects other frozen inputs as strict probes; it does
not turn fallback into acceptance. The guard-generalization checks additionally
require independent two-/three-buffer construction, short/empty/repeated slot
participation, emitted C++ and guard-corruption rejection. Historical GEMM and
broader allocation remain later milestones.

The lowerer derives available bound differences and small parameter residues
from the selected execution relations. Every proposed guard must exactly cover
its requested integer domain. Difference arithmetic must fit on the full
insertion domain; signed remainder executes beneath a proved nonnegative bound.
This is a bounded vocabulary, not arbitrary affine predicate synthesis: residue
divisors must occur in the relation and lie in 2..16, and threshold proposal skips
at most eight proved-empty integer levels. Unknown cases retain explicit refusal.

Ordered disjunctions short-circuit both failed clauses and unsafe arithmetic.
Before allocation or emission, a shared 4096-operation estimate and depth-64 cap
bound the actual recursive expansion. The native `guard-growth` check rejects
32 two-literal clauses without emitting their exponential tree, while retaining
supported linear guards. These are resource bounds, not solver wall-time limits.
The test-only boundary-projection observer separates private synchronization
arithmetic from payload; values escaping the region remain payload.

Containment first tries bounded sufficient rational implication, including only
canonical total floor definitions for qualified witnesses. Tightened residue
membership remains an obligation. Bounded integer samples may refute containment
but never prove it. All other cases retain the existing exact integer difference.
The 183 native/reference query tests challenge those directions independently,
including sparse endpoint indexing, common-row subtraction, and cache invalidation.
`PTOAS_LOGICAL_TRACE=1` records slow primitive queries and guard preparation in
addition to the existing stage work counts; it does not change planning.

`--insert-sync-planner=logical` requests independent construction.
`logical-or-existing` discards unsupported candidates and runs existing
construction on the untouched body. The default remains `existing`. The
constructor's default work allowance is 384 million, adjustable with
`--insert-sync-logical-work-budget`; this is not a hard time/memory bound.

The requirement observer is test-only and synchronous. It exports every phase
and read/write access, including accesses that produced no requirement. The
reference independently enumerates conservative known physical local conflicts
and compares occurrence relations against the original fixtures' control. GM
and unknown geometry are explicitly excluded from that reference denominator;
the native constructor retains its qualified conservative obligations. Writes
are exported as may-writes. This does not claim precise last-producer analysis.

Fresh emitted reconstruction still checks the actual payload, guards, event
matching, boundaries, requirements and key reuse. The native mutation tests
remove a wait, widen a barrier guard, reorder independent loads, alter a
conversion rounding mode, add an allocation and swap wait keys without changing
set/wait counts; each must be rejected without changing the input.
Device correctness and runtime are separate, still outstanding qualifications.

The accepted local evidence and limitations are in [M1_RESULTS.md](M1_RESULTS.md).
The M2 native results and remaining costs are in [M2_RESULTS.md](M2_RESULTS.md).

## Clean upstream extraction

This directory includes frozen accepted checkpoint outputs under `checkpoint/`.
The constructor runner compiles upstream and logical arms from unchanged input;
revised/refiner comparisons read the hash-checked artifacts. The old refiner is
not imported or invoked. The default acceptance covers M1 and M2 together.

Run `check_qualification.py --driver "$BUILD/tools/pto-test-opt/pto-logical-sync-test" --output "$CAMPAIGN/qualification"`
for the extracted physical-import refusal checks. Use a fresh disk-backed
campaign directory. The original M1/M2 reports describe research-checkpoint
results; clean extraction results are recorded separately.

## Required focused development gate

With `BUILD_TESTING=ON`, Python bindings and `PTOAS_OAHS_TESTS=ON` (the test-build
default), run:

```sh
cmake --build "$BUILD" --parallel 2 --target check-oahs-focused
```

The target builds the native drivers and runs the registered `oahs_focused`
CTest serially. CI runs this target explicitly. The libisl runtime must be
available; an unavailable reference fails the gate rather than skipping it.
It covers native/reference relation parity, occurrence and arithmetic import,
known-local requirements, physical admission, observer challenges, one-buffer
and skipped/empty native construction, emitted corruption/rollback, budget
fallback and dynamic-only authored synchronization in all three planner modes.
The gate also requires the independent physical-candidate oracle, native fanout
scaling checks, and actual endpoint/barrier mutation tests. These tests protect
qualified fast paths as well as their conservative fallbacks.
Six small native slot cases must construct strictly, including actual partial
byte overlap and conservative unknown/out-of-range selectors. Generated-only
guard mutations must preserve original operands and command counts while
failing fresh reconstruction. Boundary-condition tests compare exact atom
complements with general integer subtraction and check cache identity, charged
warm lookups, sparse endpoint scaling and budgeted wildcard enumeration.
Fresh run directories and commands/timings are retained under the build tree's
`test-results/oahs/`. Production compilation has no Python/libisl dependency.

`check_constructor.py --focused` deliberately excludes the four-kernel and
multi-buffer milestone campaign. Passing the focused gate does not qualify
three-buffer support, the unfinished guard milestone, or device behavior. The
ordinary runner still requires those buffering effectiveness checks.

The first integration run passed in 41.45 seconds (same Release `-O1` assertion
build). Compiler, architecture and performance reviewers accepted the authored
admission and focused-gate change. This does not accept the WIP guard milestone.
The performance reviewer blocks avoidable all-pairs order materialization,
repeated full requirement scans and structural quadratic deduplication before
broader coverage is accepted. Dense, genuinely overlapping requirements may
still have quadratic output size; that is reported separately from avoidable
bookkeeping. Work allowances must not be raised to satisfy these gates.

## Retirement and endpoint normalization

Current construction uses the explicit [retirement contract](../../../../docs/designs/oahs-retirement.md)
and retains one plain terminal ALL. The frozen M1/M2 artifacts remain historical
comparison inputs; the acceptance runner preserves payload/ABI/allocation and
useful boundaries, bounds executed pairs and scalar work against them, and
records changed exit inventories rather than requiring obsolete exact counts.
[Step-2 results](RETIREMENT_ENDPOINT_RESULTS.md) report the four native cases.
The focused gate now includes actual complementary-endpoint coalescing, real
overlapping-write obligations, retirement and emitted-corruption checks.
