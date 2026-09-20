# Device task: manual PTO-ISA GEMM versus OAHS and existing InsertSync

## Objective and scope

Benchmark a pinned hand-written kernel on identical payloads. Determine whether
OAHS recovers its overlap and whether extra event pairs are cheap, as they were
in Shenggan. This is a new campaign; do not alter the already dispatched
projection campaign at `495fb9cbd` or substitute its measurements here.

The attachment is self-contained: prepared PTO/C++ arms, generator, original
PTO-ISA archive and headers, compiler source archive, source/control probe,
local evidence, and remote build/correctness/timing scripts. No workstation
paths are required. Paths below are relative to the extracted package root.

Pins:

- Compiler: `495fb9cbda7649f15a7fc09d6b1d91ffb7737d54`.
- Manual PTO-ISA: `c0d7148e95ef73bd12a73165fdce4b723a3b7e72`.
- Target: the same A3 device/toolchain used for the projection/GEMM campaigns.

Read `tools/benchmarks/manual_sync/README.md` first, especially the upstream
rearming finding and distinction between original C++ and normalized PTO.

## 1. Verify, extract, and build

1. Verify the outer archive hash supplied with dispatch and `SHA256SUMS` after
   extraction. Inspect `manifest.json` for source revisions.
2. Extract `sources/pto-isa.tar.gz` into a disk-backed work directory. Use those
   exact headers for all arms. The source archive also contains the manual
   CMake/README and instruction implementations needed to investigate the
   rearming contract. Do not use the CPU probe headers for device compilation.
3. Use `plans/smoke` and `plans/reference` as the supplied lowered arms. They
   already contain synchronization. **Do not run InsertSync on them again.**
   Reproduction from `input.pto` is supported by `prepare.py`; use the matching
   compiler build or build the attached compiler source against the available
   LLVM/MLIR. Verify generated PTO against the package before timing.
4. Generate a JSON array of explicit device flags from the validated target
   build configuration. If deriving it from
   `ptodsl._runtime.native_build._kernel_compile_flags('cube','a3')`, call the
   function and record its source and any normalization. Select the pinned PTO
   include directory via `--pto-isa`, not a second hidden header installation.
   Require effective `-O2`, the correct Cube architecture, and manual PTO mode
   (`--cce-pto-enable`, **no `--cce-pto-auto-enable`**). Record CANN, driver,
   device model and compiler versions.
5. Build all ten shared libraries (two cases × five arms):

```sh
python3 tools/benchmarks/manual_sync/build_device.py \
  --plans plans --pto-isa WORK/pto-isa-c0d7148e95ef73bd12a73165fdce4b723a3b7e72 \
  --bisheng /PATH/TO/bisheng --flags WORK/device-flags.json \
  --output WORK/binaries
```

The flags path and toolchain path are remote environment choices. Audit each
saved `.cc1.log` for the actual optimization level and absence of a later
override. Resolve compatibility issues with a documented common build change;
do not silently change one arm's payload or enable an extra automatic pass.
The wrapper excludes only the original source's host launcher, retaining the
entire device body. It does not enable the `__COSTMODEL` device-header path.

## 2. Correctness before timing

Run the smoke case first. Then run the full reference case. Set `HOST_THREADS`
explicitly to a suitable count for this remote CPU/BLAS installation; use its
available resources without oversubscribing other jobs. Confirm NumPy uses a
working optimized BLAS. Do not substitute a scalar triple loop.

```sh
python3 tools/benchmarks/manual_sync/run_device.py \
  --plans plans --binaries WORK/binaries --case smoke \
  --host-threads HOST_THREADS --output WORK/smoke

python3 tools/benchmarks/manual_sync/run_device.py \
  --plans plans --binaries WORK/binaries --case reference \
  --host-threads HOST_THREADS --output WORK/reference
```

The script checks every output, three seeds, and four queued launches into
separate output slots with one final synchronization. It computes each FP64
oracle once per seed, shared across arms. Numerical bounds, output sentinels,
guards, finite values, input immutability, hashes and pairwise differences are
reported. Timing starts only after **all arms and all seeds** pass. Do not
loosen the numerical bound to qualify a result. Save progress if a queue limit
interrupts a job; avoid duplicate submissions and keep each queued job within
the broker's verified duration limit.

The full case is **M=N=K=6144, 24 cores**, half inputs, float output. B is DN/
column-major. Do not use the row-major Shenggan B layout. All 1,152 output
tiles must be produced. The smoke case is not a performance conclusion.

## 3. Keep the five meanings separate

- `original_cpp`: the actual upstream hand-written device body.
- `manual`: identical source payload/event trace represented in normalized PTO.
- `manual_banked_keys`: an explicit experimental amendment, readiness keys
  separated by MAT bank. Same command count/placement and payload graph.
- `handoff`: our OAHS pass.
- `existing`: existing InsertSync in the same compiler build.

First compare `original_cpp` with `manual` to quantify transcription/lowering
effects. The synchronization-only comparison is `manual`/`manual_banked_keys`
versus OAHS/existing on their common PTO payload. Report both comparisons.

The original manual lacks a causal consumption-before-republication proof for
its fixed readiness keys in our local ordinary-event model. Its amended arm
passes that check. Investigate the exact target's event semantics and emitted
instructions; report whether hidden stalls/synchronization supply a mechanism
outside that model. Successful finite numerical runs alone do not close the
contract question. If an original arm hangs or fails, record that outcome and
use a separately labelled follow-up campaign for the qualified arms; never
quietly remove the failing arm from a supposedly successful five-arm run.

## 4. Measure structure and latency

The harness gives 180 ACL-event samples/arm in six rotated rounds. Report median,
IQR, round medians, TFLOPS, and ratios to both manual controls and existing.
Archive the loaded `.so` hashes and verify them against `build.json`.

Take matched valid profiles/timelines of the five reference arms, separately
from timing. Inspect effective instructions and actual MTE2, MTE1, M and FIX
busy intervals, blocking waits, named drains, and next-bank overlap. Attribute
stalls to the protected bank/first consumer or key reuse. In particular:

1. Does OAHS preserve early A extraction before B-load completion?
2. Do next-bank loads/extracts overlap current-bank compute?
3. How much do existing's M and MTE2 drains cost?
4. Does readiness-key banking change the original manual's behavior?
5. Do differing terminal drains or C++ scalar control affect short cases?

Relation subsets and event counts are explanatory evidence, not latency
predictions. Do not infer a manual speedup from a lower count, nor attribute
source-C++/PTO differences entirely to synchronization.

## 5. Deliverables

- One concise result table separating numerical qualification, protocol-model
  status, compile results, timing and profiles.
- Exact commands/flags, compiler/target versions, generated sources, loaded
  binaries and hashes, profile validity checks and paired timelines.
- Source-normalization comparison and the readiness-rearming investigation.
- An archive with relative paths, internal SHA-256 manifest, and verified
  round-trip extraction. Include scripts/seeds to regenerate large host
  oracles; the multi-gigabyte oracle cache need not be archived.
- No device-speed claim for CATLASS or attention: those references have not yet
  received a matched-payload three-arm adoption. The next targets are CATLASS
  ping-pong, then preload/retained-A, then full-contract attention. Keep their
  tiling/layout, queue and target-specific instruction contracts explicit.
