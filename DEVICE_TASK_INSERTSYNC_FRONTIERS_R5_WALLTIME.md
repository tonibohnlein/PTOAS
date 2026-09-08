# Device task: InsertSync R5 frontier placement correctness and wall time

## Objective

Evaluate the R5 lifecycle/frontier-placement implementation on A3 silicon using
all eleven checked-in benchmark pairs. Establish numerical and progress
correctness first, then measure synchronized host wall time for every distinct,
correct executable.

The primary question is whether R5 moves generated `set_flag`/`wait_flag`
operations to useful lifecycle frontiers and changes wall time relative to the
same compiler with placement disabled. Keep set/wait actions, each named-pipe
barrier and `PIPE_ALL` separate. Never add them into a synchronization score.

This is an execution task. Finish adapters and coordinated launchers where
necessary; a repeated launchability census is not the deliverable.

## Immutable source

Repository:

```text
https://github.com/tonibohnlein/PTOAS.git
```

Candidate:

```text
SHA:     1509506a43a619fc0f888a49ead90aecce92b0fc
subject: Add lifecycle-guided frontier placement to InsertSync
branch:  codex/insertsync-revision-r1  # provenance only
```

R4 cross-revision control:

```text
SHA:     ca2647e71f0a069aa300c9ed9af3b498b35e741c
subject: Add storage frontier refinement to InsertSync
```

Use detached checkouts at these exact SHAs. Do not use a moving branch tip and
do not modify, commit, rebase or push from campaign checkouts. Record checkout,
submodule, compiler, Python runtime and `libPTOASCompiler.so` hashes before and
after the campaign. Follow the device machine's own resource policy and record
the actual build and execution concurrency.

## Frozen inputs

Take all inputs, adapters, reference sources and goldens from the candidate
commit. Verify these hashes before compilation:

```text
a5427e8b7eb8a8d8e838f6cdb0dadc8e83448ba091707da3789c813b95cba09f  test/experiments/insert_sync/performance/manifest.json
e6f04371ecb556c71dd44013c3aa6833e7d52f4f811336a1a3557f02ede020f3  test/experiments/insert_sync/performance/kernel-pairs-manifest.json
778f93bb4111e8d251e15d6ae403d7c5954537ea464de852bbf08cccd206f450  test/experiments/insert_sync/performance/device/reference-manifest.json
```

Verify every file hash inside those manifests. The fixed denominator is:

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

Keep all eleven rows in every launchability and result table, including failed
or blocked arms.

## Five candidate arms

Build the R5 compiler once. Use the manual PTO member only for `hand_tuned` and
the byte-identical automatic member for the other four arms. Use `--pto-arch=a3`,
the manifest's PTO level and this common automatic configuration:

```text
--enable-insert-sync
--insert-sync-gm-alias=assume-disjoint-arguments
--insert-sync-effect-coverage=report
--insert-sync-audit=report
```

Allocate distinct device buffers for distinct GM pointer arguments and record
their base addresses and extents. Keep staged traversal and completed-barrier
pruning disabled in all arms.

| Arm | Source | Additional options |
| --- | --- | --- |
| `hand_tuned` | manual member | no InsertSync |
| `insert_sync_default` | automatic member | none |
| `mmad_only` | automatic member | `--insert-sync-mmad-chains` |
| `r4_refinement` | automatic member | `--insert-sync-mmad-chains --insert-sync-frontier-refinement` |
| `r5_placement` | automatic member | `--insert-sync-mmad-chains --insert-sync-frontier-placement` |

`--insert-sync-frontier-placement` implies refinement; do not also pass the
refinement flag in that arm. Both frontier flags default off.

As a source-control check, also compile the automatic member with the exact R4
compiler and `--insert-sync-mmad-chains --insert-sync-frontier-refinement`.
Compare its post-pass PTO, generated C++ and binary with the R5
`r4_refinement` arm. If they are byte-identical after removing diagnostic-only
attributes, record the identity and do not create a redundant full timing arm.
If they differ materially, retain the diff and add the exact-R4 executable as a
separate correctness and timing arm.

## Host generation and synchronization accounting

Before device compilation, generate PTO and C++ for every fixture and arm.
Preserve exact argv, input/output hashes, stdout, stderr, exit status, compile
time, audit diagnostics and relevant pass dumps.

Run the checked-in local campaigns against the R5 runtime with fresh output
directories:

```bash
python3 test/experiments/insert_sync/performance/run.py \
  --python-root <r5-build>/python \
  --arch a3 --mmad-chains --frontier-placement \
  --output <results>/host/r5-placement-controls

python3 test/experiments/insert_sync/performance/run.py \
  --python-root <r5-build>/python \
  --manifest test/experiments/insert_sync/performance/kernel-pairs-manifest.json \
  --arch a3 --mmad-chains --frontier-placement \
  --output <results>/host/r5-placement-kernels
```

Reproduce the R4 reference in
`test/experiments/insert_sync/performance/FRONTIER_R4_COUNTS.md`. For every arm
record:

- static `set_flag` sites and static `wait_flag` sites separately;
- executed set/wait actions per event key and direction for every dynamic case;
- named barriers split into `PIPE_M`, `PIPE_V`, `PIPE_MTE1`, `PIPE_MTE2`,
  `PIPE_MTE3` and `PIPE_FIX`;
- `PIPE_ALL` separately;
- `frontier_signals_advanced`, `frontier_waits_delayed`,
  `frontier_boundary_handoffs`, `frontier_barriers_removed`,
  `frontier_barriers_guarded`, requirements, generations and work;
- exact synchronization placements, not only counts.

The local four-case R5 check observed zero moved events for two-buffer,
three-buffer, four-use and historical GEMM. Reproduce this rather than assuming
it. Determine whether any of the other seven fixtures enters a supported R5
placement domain. If `r4_refinement` and `r5_placement` emit identical C++ and
identical binaries for a fixture, that binary identity is the primary result;
run only a short labelled harness-order control instead of pretending duplicate
executions measure a compiler effect.

Any material non-synchronization difference between the paired manual and
automatic inputs, or between automatic arms, is `NON_SYNC_MISMATCH` and blocks
the affected performance comparison.

## Launch all eleven fixtures

Reuse and validate the support in
`test/experiments/insert_sync/performance/device`. Do not repeat the earlier
four launchability blockers without attempting the supplied adapters.

- Controls: use their shared wrapper and validate every produced FP16 tile.
- GEMM: retain the pinned guide ABI, `C = A x B^T`, explicit generated-symbol
  binding and 24-core geometry for the primary shape.
- TopK: validate both score and index output on every launch. The existing
  hand-tuned arm has an intermittent index defect; retain it as a correctness
  result and exclude its failing timing data. Do not silently repair it.
- Triangular inverse: validate the complete 16x16 FP32 output against the
  independent unit-lower-triangular inverse golden.
- Conv2D: use the supplied cube adapter, device-only helper guard, exact
  NC1HWC0/FRACTAL_Z strides and full interior-tile golden.
- FlashAttention: launch one generated cube peer and both supplied vector
  stripes as one coordinated operation with the three eight-slot FIFO rings.
  Keep the fixed vector peer identical across arms and validate final attention
  output. Do not feed or drain queues from the host.
- GDN/KDA: launch the cube function and both vector stripes together with the
  same count, tail flag, FFTS address and disjoint workspaces. Never launch the
  peers serially. Validate U and W, workspace rounding and untouched rows.

Compile and repair objective adapter/toolchain problems on the device host.
Retain every repair as a patch and apply it identically across arms. Do not add
local synchronization to generated functions or weaken a golden. Distinguish an
adapter failure shared by all arms from an arm-specific correctness failure.

## Correctness campaign

Use seeds `2026` through `2030`. Poison outputs and guards before every launch,
retain queue/workspace state across repeated launches, and validate every launch.
Run at least 50 launches per nonempty configuration.

Required dynamic coverage:

- controls: trips `0, 1, 2, 3, 5, 16`;
- GEMM: K-loop trips `1, 2, 3, 4` using K=`512, 1024, 1536, 2048`, plus
  `4096x4096x4096` and `2048x4096x4096`;
- TopK: groups `0, 1, 2, 3, 16`, including duplicate values and deterministic
  ties;
- triangular inverse: matrices `0, 1, 2, 3, 16`;
- Conv2D: panels `1, 2, 3, 32`;
- FlashAttention: tiles `0, 1, 2, 3, 4, 7, 8, 9, 16`;
- GDN/KDA: chunks `0, 1, 2, 3, 16`, with full and half-tail cases where valid.

For each launch record absolute/relative error, integer mismatches, NaN/Inf,
unwritten output, guard corruption, timeout/hang and repeated-launch stability.
An incorrect arm remains in the table as `CORRECTNESS_FAILURE` and supplies no
performance result.

## Wall-time measurement

The primary metric is synchronized host wall microseconds per launch. Start a
monotonic C++ clock immediately before submission and stop after stream
synchronization. Exclude allocation, initialization, copies, Python dispatch,
golden computation and validation. Record device-clock duration as a secondary
metric using the same samples.

Measure single-launch latency (`batch=1`). For kernels shorter than 20 us, also
measure separately labelled batched throughput with a batch duration of at least
5 ms; never merge the two metrics.

For each distinct correct executable use 20 balanced Latin-rotated blocks, four
complete rotations of the five arms, with 50 timing samples per arm per block.
If an arm is excluded or binary-identical, rebalance the remaining distinct
executables so every arm occupies each ordinal position equally. Run paired
arms on the same card without overlapping measured workloads. Record card,
clocks, temperature, power/utilization and background activity.

Bootstrap paired block ratios by resampling blocks. Report p10/p50/p90,
arithmetic sample counts, absolute host/device times and 95% confidence
intervals. Use 2% as the practical-difference threshold. A confidence interval
containing 1.0 does not establish a change, and a sub-percent effect whose sign
depends on host versus device clock is no result.

Primary ratios:

```text
r5_placement / r4_refinement
r5_placement / mmad_only
r5_placement / hand_tuned
r4_refinement / hand_tuned
mmad_only / insert_sync_default
```

Relate timing only to actual placement and executed synchronization changes.
Do not infer performance from a lower summed operation count.

## Deliverable

Create a new archive without overwriting prior campaigns. Include:

- `REPORT.md` with direct answers and limitations;
- launchability matrix for all eleven fixtures;
- complete correctness matrix and per-launch records;
- static and executed synchronization tables split by mechanism;
- R5 placement counters and exact placement diffs;
- raw host/device timing samples with arm/block/card/order labels;
- paired-bootstrap analysis and machine-readable summaries;
- all commands, logs, compiler fingerprints and environment inventory;
- input, post-pass PTO, generated C++, binaries and binary hashes;
- adapters, goldens and any repair patches;
- an internal manifest with SHA-256 and byte counts.

Verify the archive from a pristine extraction and write an external SHA-256
sidecar. Leave all cards healthy and idle and all campaign source trees clean.

Answer explicitly:

1. Did R5 move any generated event or remove/guard any named barrier? Where?
2. Did every moved plan pass all numerical, progress and repeated-launch checks?
3. Did R5 change synchronized host wall time by at least 2% on any distinct
   executable, with a paired 95% interval supporting that conclusion?
4. Did R5 narrow the GEMM or triangular-inverse gap to hand tuning?
5. Were all eleven fixtures launched as their complete operation, and what
   concrete blocker remains for any fixture that was not?
