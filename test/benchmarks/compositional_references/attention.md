# Task 3: manual attention with delayed QK/PV and distinct buffer lifetimes

Read `COMMON.md`. This is a mixed-core attention benchmark, including the actual
queue and workspace protocol. It must not become two independently timed GEMMs
or a local-only softmax projection reported as full attention.

## Primary source and bounded scope

Start with the pinned PTO-ISA directory `kernels/manual/common/flash_atten/`:

- `fa_performance_kernel.cpp`, `pto_macro_matmul.hpp`,
  `pto_macro_fa_softmax.hpp`, and all referenced includes.
- `main.cpp`, `scripts/`, generated-case configuration, CMake and README.

This source has separate ingress/output ownership and a configurable QK preload
schedule. Preserve the generated case's actual `QK_PRELOAD`, FIFO depth,
consumption-sync period, masks, core counts and all runTFA launch arguments.
Do not enable intermediate-check mode for timed runs; it changes execution.
The upstream host launcher also warms cores and prefetches Q/K/V, potentially
using SDMA. Measure that scope consistently and separately from runTFA itself.

The pinned CATLASS `examples/23_flash_attention_infer/fai_kernel.cpp` is a
secondary schedule reference: `preLaunch=2` offsets PV from QK and rotates
workspace over three positions. Use it to inspect delayed-row/physical-bank
identity. Do not require a second complete CATLASS attention port before
delivering the primary PTO-ISA result. Report CATLASS measurements only if a
separately qualified matched adoption is actually completed.

## Cases

Use the source's generated-case builder and validate every compile-time limit.
Begin with HEAD=128 and the README's admitted cases:

- Smoke: S0=128, S1=1024, CUBE_S0=128, TILE_S1=128.
- Pipeline: S0=2048, S1=2048, CUBE_S0=128, TILE_S1=512.
- Add one generated case large enough to exercise the source's full-core
  communication branch, if the pipeline case does not. The source changes
  communication behavior based on block-row count; report which branch ran.
- Select supported causal and noncausal variants, and logical sequence lengths
  that exercise prologue, steady-state and final drain. Test the shortest legal
  run with no steady state and a masked boundary. Do not invent unsupported
  nondivisible shapes; use the source's actual legality checks.

Keep the initial timing matrix to two representative qualified shapes plus one
mask variant. Broader shape tuning is outside this task.

## Preserve three separate identities

Trace and report:

1. The semantic QK/PV row or tile being computed.
2. The physical L1/L0/UB bank's previous participating reader/writer.
3. The queue message/workspace generation being produced, consumed and freed.

These can be out of phase. A PV row does not necessarily overwrite the bank last
used by that same row's QK. Preserve the original lag, prologue, skipped work,
re-entry and epilogue; do not introduce runtime history counters or flatten away
the mechanism being tested.

Expand macro effects enough to identify reads, writes, temporaries and aliases.
Cross-core queue/free/collective/visibility primitives stay fixed unless their
exact imported contract is already supported. Delineate that boundary explicitly
in the synchronization-removal manifest. Do not synthesize completion by treating
a macro call as an opaque synchronous operation.

## Questions for the plans and timelines

- **Cube:** Does a necessary probability-readiness path already carry QK operand
  completion before PV preparation, or is a direct release independently needed?
  Identify actual source occurrence, acquisition and readiness-key consumption.
- **Vector ingress:** Which prior reader protects the receive slot? Is the next
  receive delayed by output work that touches different physical storage?
- **Vector output:** Which push/store still reads probability data, and where
  is its first real conflicting overwrite? Acquire its completion by that
  deadline; do not delete a necessary return merely because ingress is disjoint.
- **Delayed schedule:** Is bank preparation gated by an unrelated later QK/PV,
  rather than the previous participating use of that same bank?
- **Full system:** Does the queue request/reply protocol already enforce an
  order that a local-only graph appears able to remove? Evaluate the complete
  mixed-core timeline before claiming new hardware overlap.

Do not assume the Qwen corpus's exact tail alias exists in this different source.
Derive actual byte intervals from the chosen reference, and retain every real
probability-output/scratch overlap it contains.

## Tests and result

Require full numerical attention outputs against an independent reference with
a predeclared tolerance that accounts for the approximate exp and casts. Include
three seeds, four queued invocations, guards, finite values and all outputs.
An upstream comparison is useful but does not replace the independent reference.

Pair the positive cases with missing-readiness/release and missing queue-return
negatives in the test oracle, not unsafe timed device arms. Keep the required
in-place vector fences. Add an artificial early output-return wait to check that
the ordering/timeline test detects unnecessary ingress blocking when the source
actually permits independence.

Deliver original/matched-manual/OAHS/existing end-to-end attention comparisons,
separate AIC/AIV counts and timelines, message matching and local rearming
evidence, and the exact first additional blocking edge or unsupported contract.
If a complete matched import is unavailable, provide the original baseline plus
the minimal failing import and honestly labelled local experiments. Do not
replace the benchmark with a count-only report or claim those are full-kernel
automatic synchronization results.
