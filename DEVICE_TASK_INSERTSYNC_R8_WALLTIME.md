# Device task: R8 InsertSync wall time

Measure synchronized host wall time on A3 for R8 versus R7, lifecycle synthesis
disabled, and hand-tuned code. Reuse the existing harnesses and finish a small
paired experiment. The primary question is whether fixing the added one-buffer
and four-use MTE3 barriers changes runtime, and whether lifecycle placement helps
when the synchronization counts match the baseline.

This brief supersedes the execution scope of the earlier R5 device task. Do not
repeat its large seed/shape campaign, full compiler test suite, corpus sweep,
profiling, or open-ended adapter development. Aim to deliver measurements within
45 minutes after compiler binaries are ready. Report compiler preparation time
separately and send the initial timings as soon as the controls and GEMM finish.

## Exact sources and arms

Repository: `https://github.com/tonibohnlein/PTOAS.git`.
The branch `codex/insertsync-revision-r1` is only a locator; use these exact pins:

| Name | Commit |
| --- | --- |
| R8 | `4c19cc1cba444f28ab9af2a022aae1e3c80079eb` |
| R7 | `c720df6bad7db78b07e97c5499f9c435f80c330f` |

Reuse matching builds; otherwise build only the compiler/runtime targets needed
to emit PTO and C++. Use the device host's resource policy and record explicit
build concurrency. Freeze compiler libraries and matching Python runtimes after
building; retain their hashes, toolkit/pto-isa versions, and device model.
Keep source checkouts clean and put generated artifacts in a fresh results tree.

Use R8's manual/automatic input pairs for both compilers, with the per-case PTO
level in the manifests and `--pto-arch=a3`. Common automatic flags are:

```text
--enable-insert-sync
--insert-sync-gm-alias=assume-disjoint-arguments
--insert-sync-effect-coverage=report
--insert-sync-defer-same-pipe
--insert-sync-mmad-chains
```

| Arm | Compiler | Input | Additional option |
| --- | --- | --- | --- |
| `hand_tuned` | R8 | manual | no automatic flags |
| `residual_baseline` | R8 | auto | none |
| `r7_lifecycle` | R7 | auto | `--insert-sync-lifecycle-synthesis` |
| `r8_lifecycle` | R8 | auto | `--insert-sync-lifecycle-synthesis` |

Keep frontier refinement, frontier placement, and completed-barrier pruning off.
`residual_baseline` is the current revised InsertSync with staged repair and
MMAD enabled. It is **not** the historical production InsertSync at `7e2ec3e`;
do not label it “original InsertSync.” No historical-compiler rebuild is needed.
Compile R7 once with lifecycle off as a source-control check; if its generated
program differs from R8's residual baseline, retain the diff and measure that
distinct baseline for the affected fixture so the R8 effect remains attributable.

## Frozen population and expected counts

All paths below are under `test/experiments/insert_sync/performance/` at R8:

```text
a5427e8b7eb8a8d8e838f6cdb0dadc8e83448ba091707da3789c813b95cba09f  manifest.json
e6f04371ecb556c71dd44013c3aa6833e7d52f4f811336a1a3557f02ede020f3  kernel-pairs-manifest.json
778f93bb4111e8d251e15d6ae403d7c5954537ea464de852bbf08cccd206f450  device/reference-manifest.json
```

Verify these manifests and their input/reference hashes. Allocate distinct GM
buffers as required by the common alias contract. Use the repaired FlashAttention
pair already in R8, without applying an older source repair a second time.
Use pto-isa `1216c55831fcc4ed2f096e4ca582ff75a633fbb6`, as in the prior task;
record any necessary compatibility change and apply it equally across arms.

Emit PTO and C++ for all eleven fixtures. Preserve exact argv and diagnostics.
The local R8 static reference is below; a pair means equal set and wait site
counts (e.g. 53 sets and 53 waits). Report both counts in the machine-readable data.

| Fixture | Hand pairs | R8 pairs | R8 named barriers | R8 PIPE_ALL | R8 selection |
| --- | ---: | ---: | --- | ---: | --- |
| one_buffer | 6 | 6 | none | 1 | applied |
| two_buffer | 12 | 12 | MTE3: 2 | 1 | applied |
| three_buffer | 18 | 18 | MTE3: 3 | 1 | applied |
| four_use | 24 | 24 | MTE3: 2 | 1 | applied |
| historical_gemm | 53 | 44 | FIX: 1, M: 4, MTE1: 2, MTE2: 3 | 1 | fallback |
| topk_128 | 10 | 15 | MTE3: 2, V: 14 | 1 | fallback |
| conv2d_interior | 24 | 0 | M: 8 | 1 | fallback |
| flash_attention_cube | 18 | 15 | none | 1 | fallback |
| triangular_inverse_16 | 278 | 277 | none | 1 | fallback |
| gdn_wy | 16 | 32 | MTE1: 1, V: 13 | 2 | fallback |
| kda_wy | 17 | 35 | MTE1: 1, MTE2: 2, MTE3: 1, V: 10 | 2 | fallback |

R8's inventory matches residual_baseline on every row. R7 lifecycle adds one
MTE3 site to one_buffer and has eight MTE3 sites in four_use instead of two.
The other R7 inventories match R8. Equal counts do not establish equal placement
or equal executables. Record lifecycle status and completion counters from PTO;
fallback is successful ordinary compilation, not successful specialization.

Compare generated C++ and executable device code before timing. Identical arms
with the same wrapper, ABI, flags, and launch configuration share one executable
and one timing series, with an explicit alias mapping. A shared series is not
an independent paired measurement; label its ratio `IDENTICAL_CODE`. When
identity cannot be established, time the arms separately. Do not deduplicate on
counts alone or remove executable differences by broad text normalization.

## Reuse harnesses; run the useful rows first

First reuse the latest validated device harness, including any fixes from the
R5 task. The older known archive is
`/opt/pypto/insertsync-revision-c456adc12-a3-device-final.tar.gz`:
reuse its controls, GEMM, TopK, and triangular-inverse wrappers when needed.
Keep the archive intact and record the hash of whichever harness is actually used.
Reuse existing compiled kernel binaries only when source, wrapper, toolchain,
compile flags, and launch configuration all match; collect fresh paired timings.

Run in this priority order:

1. One/two/three-buffer and four-use, each at 16 trips.
2. Historical GEMM, 4096 x 4096 x 4096 and 2048 x 4096 x 4096, 24 cores,
   the established `C = A x B^T` ABI and launch geometry.
3. Triangular inverse, 16 matrices; TopK, 16 groups.
4. Conv2D, 32 panels; FlashAttention, 16 tiles; GDN/KDA, 16 full chunks,
   using existing validated coordinated launchers where available.

For the last four fixtures, read `device/README.md` and reuse its adapters or
the newer validated equivalents. Attempt device compilation if no validated
binary is available. Allow at most ten minutes of adapter fixes in total, then
record remaining blockers and complete the timing report. Do not let one
unfinished launcher hold up the measured rows. This task does not require
finishing new mixed-kernel launcher implementations.

Keep all eleven rows in the report. Distinguish `MEASURED`, `IDENTICAL_CODE`,
`CORRECTNESS_FAILURE`, `BUILD_BLOCKED`, and `HARNESS_BLOCKED`; preserve the actual
error for a blocked row. Report shared blockers once with affected arms listed.
Mixed kernels require concurrent cube/vector peers and genuine queue protocols.
Do not substitute serial peers, host-fed queues, or added local synchronization.
Keep fixed peer/adapter synchronization identical across arms and account for it
separately. Full-group wall time includes those peers.

## Compact correctness gate

Reuse existing goldens, tolerances, guards, and output poisoning. Check each
correctness launch outside the timing interval; retain queue/workspace state
between launches. Allocate/initialize outside the measured region. Validate
scores and indices for TopK, full GEMM output, and final outputs of mixed groups.

For each distinct executable at each timed shape, run 50 checked launches with
seed 2026. Add five checked launches at counts 0, 1, and 3 for controls, TopK,
and triangular inverse. For GEMM add five launches at one and three K panels
(K=512 and 1536, with valid M/N tile multiples). For an available mixed adapter,
add count 1 and 3; use count 1 and 3 for Conv2D too, whose zero case is invalid.
This is a focused correctness check, not full synchronization qualification.

The original hand-tuned TopK has a known intermittent index defect. Retain it
as a correctness reference and exclude it from performance ratios even if this
short sample happens to pass. Do not silently repair the manual arm. Exclude
any other failing executable from timing and keep its failure evidence.
Use a 60-second timeout for a correctness invocation and 120 seconds per timing
invocation. After a hang, stop that lane and report it; do not keep retrying it.

## Wall-time experiment

Primary metric: **single-launch synchronized host wall time, microseconds**.
With the stream initially idle, start `std::chrono::steady_clock` immediately
before submission; stop after stream synchronization returns. Use batch=1.
Exclude Python dispatch, allocation, transfers, initialization, golden work,
and validation. The checked-in `device/host.cpp` already supplies this interval.
Keep device-clock samples only if the existing harness provides them; adding
device-clock instrumentation must not delay this host-wall experiment.

For each timed shape use 20 warm-up launches, then 12 balanced blocks with 25
single-launch samples per distinct correct arm per block. Rotate arm order
within blocks; keep paired arms on the same idle card, without other measured
work on that card. If the distinct arm count does not divide 12, round the block
count up to the next complete rotation. Reuse a persistent process/runtime where
the harness supports it; do not rebuild or redo correctness for every block.
Validate outputs after timing blocks outside the clock; invalidate any failing
arm's timing results. Record card identity and background activity.

Report host p10/p50/p90, paired ratios of block medians with 95% bootstrap CIs
(resample blocks, not individual launches), and sample counts:

```text
r8_lifecycle / r7_lifecycle
r8_lifecycle / residual_baseline
r8_lifecycle / hand_tuned       # only correct manual references
```

Lower ratios are faster. Treat a CI containing 1 as inconclusive and use 2% as
the practical-effect threshold. Report small estimates as small; do not keep
extending the experiment to make them significant. No batched-throughput or
profiling campaign is required. Report actual wall times even when there is no
measurable R8 effect. Do not attribute the full hand-versus-auto gap to R8:
the manual and automatic GEMM schedules already differ.

## Deliverable

Print one readable eleven-fixture table in the response: host median for each
arm, R8/R7 and R8/baseline ratios with CIs, and correctness/status. Keep the two
GEMM shapes as subrows. Show aliased arms explicitly and leave unavailable times
blank with a reason. Also print set/wait counts, named barriers by pipe, and
PIPE_ALL in a separate table; never combine them into a synchronization score.

Save `REPORT.md`, `timing_raw.csv`, `timing_summary.csv`, `correctness.json`,
`arm_identity.json`, source/toolchain fingerprints, commands, generated PTO/C++,
and any adapter patch in one fresh archive with a SHA-256 sidecar. Reuse earlier
evidence by pinned reference instead of repackaging compiler trees. State time
spent in compiler preparation, device compilation, correctness, and timing.
The report must answer whether R8 fixes a device slowdown on the controls and
whether it changes GEMM wall time, including an identical-code result if proven.
