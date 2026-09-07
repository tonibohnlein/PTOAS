# Device task: MMAD-chain InsertSync wall time and all eleven fixtures

Determine whether the opt-in MMAD-chain change improves **synchronized host
wall time**, without changing numerical correctness, and close the four
launchability gaps in the preceding A3 campaign. Execute the work; a new
launchability inventory by itself is not the deliverable.

## Source and comparison identity

Compiler base/candidate source is PTOAS commit
`9e061dcf91f3707dd7d5fc38ddf922186c841dce` on
`tonibohnlein/PTOAS`, branch `codex/insertsync-revision-r1`.
Its parent is `c456adc12a6e04b5bef7caa5ffab4987de81557e`.
The MMAD option defaults **off**: an ordinary rebuild does not activate it.
The published commit containing this task includes the benchmark adapters,
reference headers and count reports. Fetch `codex/insertsync-revision-r1` and
record the resolved commit SHA before building. No external patch is needed
when using that commit. Its compiler implementation is unchanged from
`9e061dcf`; the two revised arms use one binary with the MMAD flag off/on.
The portable `benchmark-device.patch` remains an alternative for a checkout
at the exact `9e061dcf` base; record its SHA-256 if that route is used.

Pin pto-isa to `1216c55831fcc4ed2f096e4ca582ff75a633fbb6` and the pto-kernels
references to `e118ec71ed0f170111d4f3380a7a93550c8c22ed`. Verify all input,
reference and support hashes in the two performance manifests and in
`device/reference-manifest.json` before compilation. Freeze compiler binaries
and Python runtimes; hash them before and after the campaign.

Use every case in `test/experiments/insert_sync/performance/manifest.json`
and `kernel-pairs-manifest.json`: one/two/three buffers, four-use producer,
GEMM, TopK, Conv2D, FlashAttention, triangular inverse, GDN and KDA.

Use exactly these four arms. Use the same input member and adapter across all
automatic arms. The frozen count population and the explicitly labelled FA
device repair are specified below:

| Arm | Compiler | Source member | Additional options |
| --- | --- | --- | --- |
| hand_tuned | 9e061dcf | manual | no InsertSync |
| original_insert_sync | 7e2ec3e29420e297dcf5b3c59ba4d90841464821 | auto | `--enable-insert-sync` |
| revised_mmad_off | 9e061dcf | auto | `--enable-insert-sync` |
| revised_mmad_on | same binary | auto | `--enable-insert-sync --insert-sync-mmad-chains` |

Use the original compiler's legacy default GM contract, as in the previous
experiment. Both revised arms explicitly use
`--insert-sync-gm-alias=assume-disjoint-arguments`. Allocate distinct buffers
and retain base/extent evidence. Use `--pto-arch=a3` and the per-case
`--pto-level` from the manifests. Emit both post-pass PTO and generated C++.
Keep `--insert-sync-defer-same-pipe` and pruning off. There are no additional
staged/pruned timing arms in this task.

The primary comparison is revised_mmad_on / revised_mmad_off. Also report
each automatic arm / hand_tuned and each revised arm / original_insert_sync.
Original InsertSync has not changed: reuse its existing verified compiler
build and count data when fingerprints match. Its device timings must be
collected in the same paired campaign as the revised arms. Existing original
kernel binaries may be reused only when input, generated code, adapter and
toolchain fingerprints all match this campaign.

Hand GEMM and automatic GEMM are different scheduled programs, as in the
earlier campaign; do not attribute their whole gap to this one option.

The count reference is `FOUR_ARM_COUNTS.md/.json`, assembled from existing
measurements without another local benchmark run. It uses the frozen input
files/manifests at `9e061dcf`, identical to the earlier experiment. Materialize
that snapshot into the campaign directory using `git show` or `git archive`
before applying benchmark-source changes; do not overwrite the checkout.

For ten fixtures, use those frozen inputs for the device comparison too.
FlashAttention requires the already prepared entry-aware `TFREE` repair to
return FIFO credits. Preserve its frozen count row (18 / 27 / 13 / 13 pairs),
and label the runnable source variant `flash_attention_cube_entry_free` in
all raw data. Apply that identical source repair to all four arms, retain
before/after hashes and the one-line diff, and record its actual emitted
counts separately. The revised repaired variant has 15 pairs, with MMAD off
or on. It must not silently replace the frozen FlashAttention count row.
This remains the FlashAttention fixture in the eleven-fixture device matrix,
with its source revision explicitly identified.

## Reuse the previous successful harnesses

The preceding campaign archive was at
`/opt/pypto/insertsync-revision-c456adc12-a3-device-final.tar.gz`.
The originally delivered checksum is
`a3c56567a7377a2a561d1f0ba5e0c01d0fa6e5c3bff2c16d06ffd87cf8733d8e`.
The device agent reported an updated host-wall-time version with checksum
`5686622e53f7c0422ca7fee12ab9f76bfc28b018ce361cf65fa4e4de1145e269`.
Only the original archive was available on the local workstation. Record
which one is actually present; preserve both if both exist. Never overwrite
an earlier campaign. The supplied user feedback corrects two tiny earlier
claims to “no established difference”; do not recover those claims from the
older report.

Reuse `wrappers/controls`, `wrappers/topk`, `wrappers/triinv`, and
`wrappers/gemm` from that archive. Preserve the validated operation/layout
contracts, but validate **every** correctness launch, not just a final dump.
Use the prior source-derived GEMM tolerances and full-output validation where
feasible; explicitly label any remaining sampled validation. Generate fresh
binaries from the new arms; do not time retained old binaries as candidates.

## Close the four missing launch contexts

New checked-in support is in `test/experiments/insert_sync/performance/device`.
Read its README and source. It provides a shared CMake build, mixed launcher,
NumPy goldens and an ACL runner without a `torch_npu` dependency. This support
is locally tested at the reference/host level, **not yet device-compiled**.
Compile it on A3, fix objective adapter/build errors and validate the resulting
operations before timing. Keep every repair as a patch and use identical
adapter code across arms. Do not weaken a golden to make an arm pass.

- **Conv2D:** include both `conv2d_helpers.h` and generated C++ only in the cube
  device pass. The original failure was file-scope Tile prototypes in the
  host pass, not missing helper implementations in the repository. This
  wrapper places definitions and calls in one translation unit. The golden
  handles the actual NC1HWC0/FRACTAL_Z strides and only the fixed 128x256
  interior output tile. Panels: 1, 2, 3, 32. Zero panels is outside the
  frozen kernel's contract because it stores an uninitialized accumulator.
- **FlashAttention:** launch the generated cube and **two** 8-row vector
  stripes together with the supplied fixed peer. Three eight-slot rings
  use exactly `(flag, slot bytes) = (0,1024), (2,512), (4,1024)`.
  Set the runtime FFTS control address on all peers. The recovered softmax/GU
  headers are pinned verbatim; the adapter matches the smaller extraction
  and normalizes the one-tile case too. The PTO pair's `TFREE` now includes
  its GlobalTensor entry: the old entry-less overload was a device no-op.
  Tiles: 0, 1, 2, 3, 4, 7, 8, 9, 16, covering notification boundaries and
  ring wrap. Check final attention output, not just FIFO contents.
- **GDN/KDA:** one mixed launch, one cube and its two associated vector
  subblocks. Pass the same chunk count/tail flag, runtime FFTS address and
  shared disjoint workspaces to all three peers; stripe is `get_subblockid()`.
  The GDN adapter drains its final flags 3/4 on both stripes; KDA already
  drains 12/13 in the extraction. Chunks: 0, 1, 2, 3, 16, both full and
  half-tail cases. Validate U and W against the supplied independent golden,
  including FP16 workspace rounding and untouched output rows.

Do not substitute serial peer launches, host-fed queues, prepublished credits,
or synthetic acknowledgements. Run the full coordinated group and time its
end-to-end launch. Report fixed peer/adapter synchronization separately from
compiler-inserted mechanisms. The FA vector peer is fixed benchmark support,
not a claim of reproducing the full original performance-tuned FA schedule.

If an automatic arm gives wrong results, retain it as `CORRECTNESS_FAILURE`;
it is launchable but cannot supply valid timing. Distinguish that from an
adapter defect shared by every arm. A blocked row must retain the exact command,
error, source/ABI evidence and attempted repair. Do not classify the previous
four rows as blocked again merely because the earlier harness lacked peers.

## Correctness and timing

Use seeds 2026..2030 and at least 50 repeated launches per configuration,
checking each launch. Preserve all the earlier successful small-kernel shape
sweeps; add the boundary sweeps above. Keep sentinels/canaries, check untouched
regions, and reject NaN/Inf. Include zeros, signed values, distinct/permuted
TopK keys, duplicate ties, and nontrivial unit-lower triangular matrices.
Keep queue/workspace state between repeated launches so stale-credit bugs
remain observable. Retain exact tolerances and the rationale per fixture.

The earlier hand-tuned TopK has a reproducible intermittent index defect.
Keep the original hand arm for correctness diagnostics, exclude failing data
from timing, and retain the per-launch mismatch. If a terminal drain or other
fix is investigated, call it `hand_tuned_topk_corrected`, preserve its diff and
measure it separately; do not silently replace the hand reference.

For GEMM retain the 24-core geometry and both substantial shapes:
4096x4096x4096 and 2048x4096x4096. Also cover K=512,1024,1536,2048 at M=N=4096
for correctness and loop-transition behavior. The main MMAD hypothesis is
**1,408 -> 176 executed PIPE_M barriers on the 4096-cubed core-0 path**, not
removal of PIPE_ALL. Verify the actual count before timing.

Primary metric: synchronized **host wall microseconds per launch**. Start a
monotonic clock immediately before submission and stop after stream completion.
Exclude allocation, initialization, copies, numerical checks and Python
reference work. Record device-clock time as a secondary metric when available,
but base the headline on host time. The new adapter's `host.cpp` places the
clock in C++, outside Python/ctypes overhead.

Measure single-launch latency (`batch=1`) and, separately, batched throughput
for very short kernels (batch duration at least 5 ms). Never merge these modes:
submission/synchronization amortization changes the metric. Warm up consistently,
use 20 balanced Latin-rotated blocks (five complete four-arm rotations) and
50 samples per arm/block. If a failing hand arm is excluded, use 21 blocks
(seven complete three-arm rotations) for that fixture;
keep paired arms on the same card, and avoid overlapping timing lanes on that
card. Record clocks, temperature, power/utilization and background activity.
Use an explicit remote build/launch worker count appropriate to the machine.

Bootstrap **paired block ratios**, resampling blocks rather than treating
launches as independent. Report ratios with 95% CIs, p10/p50/p90 and absolute
microseconds. Predeclare 2% as the practical-difference threshold, while still
reporting the intervals and small estimates. A CI containing 1 is not evidence
of a gain. Do not claim formal equivalence solely from nonsignificance. If
host/device tiny effects disagree, describe them as unestablished. Record
timeouts and rerun policy; retain abandoned partial runs separately.

## Local count reference and final deliverable

`FOUR_ARM_COUNTS.md/.json` contain the exact comparison requested. Each cell
below is **set/wait pairs / named-pipe barriers / PIPE_ALL**, never a summed
performance score:

| Fixture | Hand-tuned | Original InsertSync | Revised MMAD off | Revised MMAD on |
| --- | ---: | ---: | ---: | ---: |
| one_buffer | 6 / 0 / 1 | 6 / 0 / 1 | 6 / 0 / 1 | 6 / 0 / 1 |
| two_buffer | 12 / 0 / 1 | 12 / 2 / 1 | 12 / 2 / 1 | 12 / 2 / 1 |
| three_buffer | 18 / 0 / 1 | 18 / 3 / 1 | 18 / 3 / 1 | 18 / 3 / 1 |
| four_use | 24 / 0 / 1 | 24 / 2 / 1 | 24 / 2 / 1 | 24 / 2 / 1 |
| historical_gemm | 53 / 0 / 0 | 44 / 21 / 1 | 44 / 24 / 1 | 44 / 10 / 1 |
| topk_128 | 10 / 22 / 0 | 15 / 16 / 1 | 15 / 16 / 1 | 15 / 16 / 1 |
| conv2d_interior | 24 / 0 / 0 | 0 / 8 / 1 | 0 / 8 / 1 | 0 / 8 / 1 |
| flash_attention_cube | 18 / 0 / 1 | 27 / 0 / 1 | 13 / 0 / 1 | 13 / 0 / 1 |
| triangular_inverse_16 | 278 / 105 / 0 | 277 / 0 / 1 | 277 / 0 / 1 | 277 / 0 / 1 |
| gdn_wy | 16 / 6 / 2 | 32 / 14 / 2 | 32 / 14 / 2 | 32 / 14 / 2 |
| kda_wy | 17 / 7 / 2 | 35 / 12 / 2 | 35 / 14 / 2 | 35 / 14 / 2 |

Reproduce these frozen static counts during device preparation. Keep any
source repairs in separately labelled rows. Count each named pipe separately
in machine-readable results, including fixed queue/cross-core mechanisms and
adapter synchronization in separate columns.

The only MMAD-dependent static change is GEMM PIPE_M: 18 -> 4. Pairs stay 44;
exit PIPE_ALL stays 1. Other revised GEMM named sites stay FIX=1, MTE1=2,
MTE2=3. One-buffer / 16 trips is 66 pairs, zero named barriers, one PIPE_ALL.
The FA source repair is independent of MMAD and is described above.

Provide a fresh archive with report, launchability and correctness matrices,
per-launch correctness records, raw host/device samples, block labels, timing
summary/paired bootstrap analysis, all commands/logs, post-pass PTO/generated
C++, compiler fingerprints, binaries, adapter/golden source and repair patches.
Include an internal SHA-256 manifest and byte counts; verify a pristine
extraction and external archive checksum. Keep prior archives intact.

Answer explicitly: did MMAD on improve synchronized wall time, did any arm
lose correctness, are all eleven fixtures now launched as complete operations,
and which remaining manual/automatic gaps are supported by the measurements?
