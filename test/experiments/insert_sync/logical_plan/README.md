# Native logical-constructor acceptance

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
serially. The 15 cases include narrow integer overflow refusal, signed Boolean
conditions, nested loops, nonunit steps, reversed phase identities, and all
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
not turn fallback into acceptance. Two-buffer release-domain lowering remains
unsupported at this milestone. The wider control ladder and historical GEMM remain outside this
accepted strict-constructor population.

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
