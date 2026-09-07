# Device task: InsertSync correctness and wall-time regression on the hand-tuned ladder

## Dispatch status

`READY_WITH_LAUNCHABILITY_GATE` (2026-09-07).

The compiler revisions and all 22 paired PTO inputs are published. The device
agent must first establish a valid launch path and independent golden for each
fixture. A fixture that is only an extracted peer or tile remains a named
blocked row; do not turn it into a timing result by inventing an ABI, deleting
protocol operations, or running cooperating peers serially.

## Objective

Run the 11 checked-in benchmark pairs on A3 silicon and determine:

1. whether the hand-tuned and automatic arms are numerically correct and
   stable over repeated launches;
2. whether any arm hangs, times out, writes out of bounds, leaves output
   unwritten, or produces nondeterministic output;
3. device kernel elapsed time and synchronized host launch wall time for every
   correctness-closed arm;
4. the runtime effect of original InsertSync, revised combined analysis,
   revised staged analysis, and the optional completed-barrier pruner relative
   to the hand-tuned reference; and
5. the relationship between runtime and the executed synchronization profile,
   keeping set/wait pairs, named-pipe barriers, and `PIPE_ALL` separate.

The main performance question is whether the revised InsertSync reduces costly
whole-pipe serialization. Do not add synchronization mechanisms into one
total or use that total as a performance score.

## Device-host resource and isolation policy

- The coordinator machine's two-worker limit does **not** apply to this
  separate device machine. Follow the device host's standing resource policy
  and record the actual build, compile, and test worker counts.
- Build each PTOAS revision once. Compile every kernel/arm once per static
  shape and reuse the retained binary.
- Run only one measured workload at a time on a card. Other assigned cards may
  execute independent correctness cases when the device service permits it.
- Use private disk-backed source, build, generated-code, binary, cache, input,
  output, and result directories.
- Apply finite timeouts to every compiler invocation and device launch. After
  a device timeout, collect the runtime log and restore card health before the
  next launch.
- Keep every source checkout tracked-clean. Do not modify, commit, push,
  cherry-pick, amend, or rebase in campaign checkouts.

## Immutable PTOAS revisions

Repository:

```text
https://github.com/tonibohnlein/PTOAS.git
```

Original production InsertSync:

```text
SHA:      7e2ec3e29420e297dcf5b3c59ba4d90841464821
subject:  chore: retain GitHub mirror workflow
branch:   main                         # provenance only
```

Revised InsertSync and benchmark population:

```text
SHA:      c456adc12a6e04b5bef7caa5ffab4987de81557e
subject:  Separate synchronization mechanisms in benchmark analysis
branch:   codex/insertsync-revision-r1 # provenance only
parent chain:
  a7df3d925eaba7a146278f0a6ddaeb0444670002
  60db1026a1de8f1bd0154f9bdcb21e0f6d944a7b
  df89ee02ca1cafad1ac8a75f6d9c55b354ae4d84
  5cc1d946525f630879a57dba3fe97a848f09ce11
base:
  7e2ec3e29420e297dcf5b3c59ba4d90841464821
```

Fetch and build detached checkouts at the two exact SHAs. Do not replace them
with moving branch tips. Record `git rev-parse HEAD`, submodule SHAs, clean
status, and the SHA-256 of the loaded `libPTOASCompiler.so` before and after
the campaign.

## Frozen benchmark population

Take the benchmark files from the revised SHA:

```text
test/experiments/insert_sync/performance/
```

Require these manifest hashes before compiling:

```text
a5427e8b7eb8a8d8e838f6cdb0dadc8e83448ba091707da3789c813b95cba09f  manifest.json
e751aa463fd6dc63d352ee9331dfbeaae692057f19786039deaa276e88e1f5a5  kernel-pairs-manifest.json
```

Verify every per-input SHA-256 recorded by those manifests. The fixed
denominator is 11 pairs and 22 PTO files:

1. `one_buffer`
2. `two_buffer`
3. `three_buffer`
4. `four_use`
5. `historical_gemm`
6. `topk_128`
7. `conv2d_interior`
8. `flash_attention_cube`
9. `triangular_inverse_16`
10. `gdn_wy`
11. `kda_wy`

The four controls are source-generated diagnostic fixtures. The other seven
are pinned source-derived kernels. Preserve the reference and support-file
hashes in `kernel-pairs-manifest.json` as well.

The exact historical GEMM PTO hashes are:

```text
3db5d353475ee920857d16e5c3533ab23e0c6d2c38e22b57235c0e85a9a6a380  inputs/step3_swizzle.pto
ccd3abfb22eb6f81d87f6bdb72dfba655a456b54512f2b6def9557766922dae1  inputs/step4_manual_pipelining.pto
```

Its source and runtime-wrapper provenance is `huawei-csl/pto-dsl` commit
`36cd417e230a84a80870f564de27539ccc7c4a75`, directory
`examples/aot/matmul_optimization_guide`. Recover `caller.cpp`,
`run_matmul.py`, `bench_matmul.py`, and `compile.sh` from that exact commit.

TopK, Conv2D, and FlashAttention references come from PTO-ISA commit
`1216c55831fcc4ed2f096e4ca582ff75a633fbb6`. Triangular inverse, GDN, and KDA
come from `huawei-csl/pto-kernels` commit
`e118ec71ed0f170111d4f3380a7a93550c8c22ed`. The manifest identifies and
hashes every exact source file.

## Five comparison arms

Use the manual PTO member only for `hand_tuned`. Every automatic arm must
consume the byte-identical `.auto.pto` member. Compile with the original PTO
level recorded in the manifest and `--pto-arch=a3`.

### `hand_tuned`

Compiler: revised SHA `c456adc12`.

Compile the `.manual.pto` input with no automatic synchronization pass:

```text
<normal lowering and emission options only>
```

Prove from the pre-sync and post-sync PTO that all checked-in manual
`set_flag`, `wait_flag`, named barriers, fixed FIFO operations, and fixed
cross-core operations remain present and no second local sync plan was added.

### `original_insert_sync`

Compiler: original SHA `7e2ec3e29`.

```text
--enable-insert-sync
```

Use the original pass's legacy GM behavior. The fixture contract requires
distinct GM pointer arguments to use disjoint device allocations. Record each
allocation base and extent.

### `revised_combined`

Compiler: revised SHA `c456adc12`.

```text
--enable-insert-sync
--insert-sync-gm-alias=assume-disjoint-arguments
--insert-sync-effect-coverage=report
--insert-sync-audit=report
```

### `revised_staged`

Compiler: revised SHA `c456adc12`.

```text
--enable-insert-sync
--insert-sync-gm-alias=assume-disjoint-arguments
--insert-sync-effect-coverage=report
--insert-sync-audit=report
--insert-sync-defer-same-pipe
```

### `revised_staged_pruned`

Compiler: revised SHA `c456adc12`.

```text
--enable-insert-sync
--insert-sync-gm-alias=assume-disjoint-arguments
--insert-sync-effect-coverage=report
--insert-sync-audit=report
--insert-sync-defer-same-pipe
--insert-sync-prune-completed-barriers
```

The host regression observed zero removals from the pruner. Reproduce that
fact. If the pruned and unpruned outputs differ, retain the diff and treat it
as a result requiring full correctness and timing; do not assume the change is
safe because it removes an operation.

Do not enable CanonicalSync, ProtocolSync, or any experimental synchronization
branch in these five arms. Those implementations are outside this InsertSync
revision experiment and do not share a validated current benchmark contract.

## Phase 0: provenance and host generation

1. Record OS, CPU, compiler, CMake, Ninja, Python, LLVM, CANN, PTO-ISA,
   runtime, driver, firmware, board model, device IDs, and device health.
2. Build both immutable PTOAS revisions with the device host's approved
   parallelism. Run the complete required test target for each build and all
   discovered `insert_sync` lit tests; report discovered and passed counts.
3. From the revised checkout, run the checked-in host campaigns against both
   builds and preserve their output:

   ```bash
   python3 test/experiments/insert_sync/performance/compare_revisions.py \
     --original-python-root <original-build>/python \
     --revised-python-root <revised-build>/python \
     --original-source 7e2ec3e29420e297dcf5b3c59ba4d90841464821 \
     --revised-source c456adc12a6e04b5bef7caa5ffab4987de81557e \
     --arch a3 \
     --output <results>/host/original-vs-revised-a3

   python3 test/experiments/insert_sync/performance/run.py \
     --python-root <revised-build>/python \
     --arch a3 \
     --output <results>/host/manual-combined-staged-controls-gemm-a3

   python3 test/experiments/insert_sync/performance/run.py \
     --python-root <revised-build>/python \
     --manifest test/experiments/insert_sync/performance/kernel-pairs-manifest.json \
     --arch a3 \
     --output <results>/host/manual-combined-staged-kernels-a3
   ```

   The revision comparison reads both manifests itself. These scripts do not
   perform device execution. Use fresh output directories.
4. Generate post-sync PTO and C++ for all five arms. Preserve exact argv,
   stdout, stderr, exit code, elapsed compile time, input/output hashes, pass
   dumps, and audit diagnostics.
5. Require every emitted function to pass PTOAS verification and device C++
   compilation before launch. A failure remains a row in the denominator.
6. Normalize away only pass-owned local synchronization and its metadata.
   Require the same payload operations, types, constants, physical tile
   addresses, GM views, control flow, ABI, and launch geometry across the
   relevant manual/automatic pair. A material non-sync difference is
   `NON_SYNC_MISMATCH` and blocks timing for that comparison.

The checked-in host reference is
`test/experiments/insert_sync/performance/MANUAL_VS_FOLLOWUP_V2.md`. Reproduce
its static and loop-aware counts before reserving a device. A count drift is a
named result to explain; never refresh the reference silently.

## Phase 1: launchability census

Classify every fixture before running it:

```text
STANDALONE_EXECUTABLE
ADAPTER_VALIDATED
WHOLE_PROTOCOL_VALIDATED
NO_AUTHORITATIVE_HARNESS
NO_INDEPENDENT_GOLDEN
WHOLE_PROTOCOL_REQUIRED
ARCH_OR_DEVICE_BLOCKED
ARM_COMPILE_REJECTED
NON_SYNC_MISMATCH
```

Keep all 11 rows in `launchability.csv`. For each executable row record the
source of the ABI, launch geometry, buffer shapes/dtypes/extents, initialization
rules, golden, comparison tolerance, and wrapper hash.

Expected preparation by fixture:

- **One/two/three-buffer and four-use controls:** write one external wrapper
  shared by all arms. For each trip, the output is elementwise absolute value
  of the corresponding 16x16 FP16 input tile. `four_use` emits four identical
  absolute-value output tiles for each input tile. Do not edit the PTO bodies.
- **Historical GEMM:** use the exact pinned guide wrapper. The operation is
  `C = A x B^T`; function names differ between manual and automatic inputs, so
  bind each generated symbol explicitly through the wrapper macros. Preserve
  24-core launch geometry for the primary shape.
- **TopK:** validate the extracted `float32`, 128-column, four-rows-per-group
  ABI against the pinned source. Use the source algorithm as the independent
  score/index golden, including its ordering and tie behavior. Validate both
  score and index outputs.
- **Triangular inverse:** use the pinned `pto-kernels` test convention and
  golden for unit lower-triangular 16x16 float32 matrices. The wrapper supplies
  the core's contiguous matrix range. Check the complete 16x16 output.
- **Conv2D:** this fixture is one interior 3x3 output tile and contains helper
  calls. Build an external adapter around the retained
  `conv2d_helpers.h`/pinned source implementation, then compare the complete
  tile with an independent convolution reference. The adapter must not change
  the kernel body or its row-major A3 layout. If helper lowering or a valid ABI
  cannot be established, report `NO_AUTHORITATIVE_HARNESS`.
- **FlashAttention:** the checked-in function is only the cube-side QK/P/PV
  FIFO peer. Recover the exact matching vector peer and FIFO initialization
  from the pinned source. Keep that peer byte-identical across all five arms;
  only the checked-in cube function changes. Run the complete cooperative
  protocol and validate the final attention output. If the exact peer and
  golden cannot be recovered, report `WHOLE_PROTOCOL_REQUIRED`; never feed or
  drain its queues from the host as a performance substitute.
- **GDN and KDA:** launch the cube function and both vector stripes as
  cooperating peers with identical `%chunks`, `%tail_half`, FFTS setup, and
  workspace allocations. Never launch these functions serially. Use the exact
  pinned source wrappers/goldens. Time the complete peer group. A missing
  coordinated launcher is `WHOLE_PROTOCOL_REQUIRED`.

The expected starting point is that GEMM, the four controls, TopK, and
triangular inverse have self-contained PTO kernel bodies. Conv2D needs an
adapter; FlashAttention needs a missing peer; GDN/KDA need coordinated peer
launches. This is a planning expectation, not permission to mark a row blocked
without trying the pinned source recovery.

## Phase 2: device correctness and stability

Run correctness before timing. The hand-tuned result is a comparison arm, not
the golden. Every arm must independently match the same CPU/repository golden.

Use at least five deterministic seeds: `2026`, `2027`, `2028`, `2029`, and
`2030`. Include structured inputs that expose ordering or tail bugs in addition
to random inputs. Initialize unused output bytes to a sentinel and validate
that the required output is fully written while guards remain unchanged.

Required dynamic cases:

- controls: trips `0, 1, 2, 3, 5, 16`;
- GEMM: K-loop trips `1, 2, 3, 4` through K values `512, 1024, 1536, 2048`,
  plus `4096 x 4096 x 4096` and `2048 x 4096 x 4096`;
- TopK: groups `0, 1, 2, 3, 16`, including duplicate values and deterministic
  ties supported by the source contract;
- triangular inverse: matrices `0, 1, 2, 3, 16`;
- Conv2D, if admitted: panels `1, 2, 3, 32`;
- FlashAttention, if admitted: tiles `0, 1, 2, 3, 16`;
- GDN/KDA, if admitted: chunks `0, 1, 2, 3, 16`, each with
  `%tail_half=false` and `%tail_half=true` where the source domain permits it.

For every nonempty case/seed/arm perform at least 50 launches of the already
built binary. Reinitialize mutable or accumulating inputs before every launch.
Record:

```text
golden result
maximum absolute error
maximum relative error
exact integer/index mismatch count
NaN/Inf count
unwritten-output count
guard corruption
output SHA-256
device status
timeout status
```

Use the pinned repository's dtype and tolerance policy. Record the exact
tolerance; do not loosen it for an automatic arm. Require deterministic output
signatures for identical input. A wrong result, nondeterminism, hang, timeout,
runtime fault, guard write, or unwritten required output is
`CORRECTNESS_FAILURE` for that arm and excludes that arm from timing.

If all arms fail the same case in the same way, classify
`HARNESS_OR_DEVICE_BLOCKED` after preserving the evidence. Do not attribute
that outcome to InsertSync without an arm-specific difference.

## Phase 3: synchronization accounting

Report static sites and executed operations separately. For both, use these
columns:

```text
set operations
wait operations
balanced set/wait pairs
named-pipe barriers by PIPE_MTE2 / PIPE_MTE3 / PIPE_V / PIPE_MTE1 / PIPE_M / PIPE_FIX / other
PIPE_ALL barriers inside loops
PIPE_ALL barriers outside loops
fixed FIFO operations
fixed cross-core sync.set/sync.wait operations
```

A set/wait pair counts as one logical handshake only when its executions are
balanced for that concrete path. Also retain raw sets and waits so an imbalance
cannot be hidden. Event-ID count is a separate resource metric and is not the
pair count.

For GDN/KDA, executed totals cover one cube peer and both vector stripes. For
FlashAttention, they cover the complete admitted peer group, while identifying
which local sync operations belong to the cube function under test. Tag every
`PIPE_ALL` by function and loop depth.

Reproduce the current observation that revised automatic `PIPE_ALL` sites are
tagged `pto.auto_sync_tail_barrier` and lie outside loops. Any automatic body
`PIPE_ALL`, missing exit completion, unbalanced event path, or event ID outside
`[0,7]` is a structural finding even when the numerical sample passes.

## Phase 4: balanced wall-time measurement

Time only correctness-closed arms with structurally comparable payload work.
Keep one device quiet for each paired measurement and synchronize at the end
of every observed launch/batch.

Measure two distinct values:

1. **Device kernel elapsed time**, using the device runtime event/profiler
   clock around the kernel or complete cooperating peer group. This is the
   primary metric.
2. **Synchronized host wall time**, using a monotonic host clock from launch
   submission through device completion. This is secondary and must be labeled
   separately.

Compiler wall time is a host diagnostic and must not appear in either runtime
column.

For a kernel shorter than 30 microseconds, batch enough identical launches to
obtain at least 5 ms per observation, then divide by the exact launch count.
Reinitialize inputs as required outside the measured interval or account for
that time consistently across arms. Confirm the output after every timing
block.

Use at least 20 balanced blocks. Rotate all correctness-closed arms through a
deterministic Latin order so every arm appears in each ordinal position equally
often; for pairwise views this must contain balanced `ABBA` and `BAAB` order.
Warm every arm before sampling. Retain at least 50 positive per-launch or
per-batch observations per arm per block. Preserve every raw sample.

Primary timing scenarios:

- controls: 16 trips;
- GEMM: `M=N=K=4096`, 24 cores; retain `2048 x 4096 x 4096` as the secondary
  shape;
- TopK: 16 groups;
- triangular inverse: 16 matrices;
- Conv2D: 32 panels, if admitted;
- FlashAttention: 16 tiles with the complete peer protocol, if admitted;
- GDN/KDA: 16 full chunks and 16 chunks with a half tail, timing one complete
  cube-plus-two-vector workload.

For each case report the median and p10/p90 per arm, plus paired per-block
median differences and ratios for:

```text
original_insert_sync / hand_tuned
revised_combined / hand_tuned
revised_staged / hand_tuned
revised_staged_pruned / hand_tuned
revised_combined / original_insert_sync
revised_staged / revised_combined
revised_staged_pruned / revised_staged
```

Compute a seeded bootstrap 95% confidence interval over paired block
statistics. For GEMM also report TFLOP/s using `2*M*N*K / device_seconds`.
Do not compare absolute figures from an earlier machine as if they were paired
results.

## Phase 5: mechanism attribution

For every material timing difference, compare the retained generated code and
an in-core trace. Confirm that non-sync instruction counts and launch geometry
remain fixed, then attribute the change using:

- executed named barriers by pipe;
- executed body and exit `PIPE_ALL` barriers;
- executed balanced handshakes by directed pipe pair;
- wait/stall cycles by pipe;
- MTE2, MTE3, V, MTE1, M, FIX, and CUBE work/overlap as applicable; and
- complete peer-group span for FIFO/cross-core kernels.

The first specific investigation is GEMM: the host reference records three
additional revised `PIPE_MTE2` sites, executing 176 times on the 4096-cubed
core-0 path. Determine whether those barriers create measurable lost overlap.
Also determine whether the single automatic exit `PIPE_ALL` is visible in
short controls and whether it is amortized in large GEMM.

## Result classifications

Classify each case/arm independently:

```text
CORRECT_AND_TIMED
CORRECT_NOT_TIMED
CORRECTNESS_FAILURE
NONDETERMINISTIC
DEVICE_TIMEOUT
RUNTIME_FAULT
ARM_COMPILE_REJECTED
NON_SYNC_MISMATCH
NO_AUTHORITATIVE_HARNESS
NO_INDEPENDENT_GOLDEN
WHOLE_PROTOCOL_REQUIRED
ARCH_OR_DEVICE_BLOCKED
HARNESS_OR_DEVICE_BLOCKED
```

For every correctness-closed automatic arm also report one performance label
relative to `hand_tuned`:

```text
FASTER_THAN_HAND
WITHIN_2_PERCENT_OF_HAND
SLOWER_THAN_HAND
INCONCLUSIVE
```

Use the paired 95% confidence interval for the label. Report the numerical
ratio and interval even when it is inconclusive. Do not label a blocked or
incorrect arm as a performance result.

## Required deliverables

Return one disk-backed result directory and a compressed archive with a
SHA-256. It must contain:

```text
REPORT.md
manifest.json
provenance.json
launchability.csv
correctness.csv
sync_static.csv
sync_dynamic.csv
timing_raw.csv
timing_summary.csv
commands/
inputs/
goldens/
wrappers/
post_sync_pto/
generated_cpp/
binaries/
compiler_logs/
runtime_logs/
profiles/
analysis/
```

`REPORT.md` must lead with:

1. exact compiler/input/device provenance;
2. an 11-row launchability and correctness table that retains blocked rows;
3. device elapsed and synchronized host wall-time tables for eligible arms;
4. synchronization counts with pairs, named barriers, and `PIPE_ALL` in
   separate columns;
5. paired ratios and confidence intervals;
6. every failure or timeout with its artifact path; and
7. a direct answer to whether revised InsertSync improved, matched, or
   regressed against original InsertSync and the hand-tuned arm per kernel.

Archive exact scripts and wrapper changes as campaign artifacts. Do not commit
them into PTOAS from the device task. Finish with all source trees clean and
all assigned devices idle and healthy.
