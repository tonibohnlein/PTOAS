# Device task: shared-generation InsertSync at 862ab8111

Run a compact A3 correctness and synchronized host-wall-time comparison of
original InsertSync, the committed shared-generation revision, and hand-tuned
code. The main questions are whether the revised barrier-free GEMM recovers
manual performance, and whether removing executed MTE3 barriers/exit drains
helps the buffering controls and TopK.

Use the existing eleven-fixture ladder and validated launchers. Report GEMM and
control results first. Aim to finish measurements within 45 minutes after the
compiler binaries are ready; record preparation separately. This task excludes
the new Qwen kernels, full compiler/corpus test suites, profiling, and the local
uncommitted one-shot handoff-planning experiment.

## Exact source pins and arms

Repository: https://github.com/tonibohnlein/PTOAS.git

| Arm | PTOAS commit | Input | Synchronization options |
| --- | --- | --- | --- |
| original_insert_sync | `7e2ec3e29420e297dcf5b3c59ba4d90841464821` | automatic | `--enable-insert-sync` only |
| revised_generations | `862ab811124d2e67a7f2cde992461983163f6a7a` | automatic | flags below |
| hand_tuned | `862ab811124d2e67a7f2cde992461983163f6a7a` | manual | no automatic synchronization flags |

Common compiler options are `--pto-arch=a3` and the manifest's per-case
`--pto-level`. The revision's exact synchronization flags are:

```text
--enable-insert-sync
--insert-sync-gm-alias=assume-disjoint-arguments
--insert-sync-effect-coverage=report
--insert-sync-defer-same-pipe
--insert-sync-mmad-chains
--insert-sync-buffer-generations
```

Leave frontier placement/refinement and completed-barrier-pruning flags off;
generation mode already includes its qualified residual cleanup. Do not
substitute `--insert-sync-lifecycle-synthesis` for buffer generations. Original
uses its legacy GM behavior, with the SAME distinct-argument allocation
contract. Do not pass revision-only flags to the original compiler.

The R8 lifecycle-disabled arm is not original InsertSync. Do not substitute it.
There is no separate combined or experimental arm in this task.

Fetch exact commits into clean checkouts, or reuse verified matching builds.
The branch `codex/insertsync-revision-r1` is a locator only. Build only needed
compiler/runtime targets; choose explicit worker counts appropriate for the
device machine. Record source pins, matching Python/native library paths,
loaded-library hashes before/after, toolkit, device model and pto-isa revision.
Keep generated files outside the source trees.

## Frozen inputs and expected revision output

Use ALL inputs, references and manifests from revised commit `862ab8111`,
including when invoking the original compiler. Both automatic arms consume
byte-identical automatic inputs.

Paths are under `test/experiments/insert_sync/performance/`:

```text
a5427e8b7eb8a8d8e838f6cdb0dadc8e83448ba091707da3789c813b95cba09f  manifest.json
e6f04371ecb556c71dd44013c3aa6833e7d52f4f811336a1a3557f02ede020f3  kernel-pairs-manifest.json
778f93bb4111e8d251e15d6ae403d7c5954537ea464de852bbf08cccd206f450  device/reference-manifest.json
7729f69a0c2a8ef58058b1fee1e49bf5f2bd63d1e377bd10add8369467426f36  SHARED_REQUIREMENTS_RESULTS.json
```

Verify the manifests and all referenced source hashes. Emit PTO and C++ for all
arms; retain argv, diagnostics, and generated files. Verify the automatic pass
actually ran and compare payload/effects and allocation contracts across arms.
Any material payload/codegen difference must be reported when attributing timing.

| Fixture | Hand pairs | Revised pairs | Revised named barriers | Revised PIPE_ALL |
| --- | ---: | ---: | --- | ---: |
| one_buffer | 6 | 6 | none | 0 |
| two_buffer | 12 | 12 | MTE3=2 | 0 |
| three_buffer | 18 | 18 | MTE3=3 | 0 |
| four_use | 24 | 24 | MTE3=2 | 0 |
| historical_gemm | 53 | 56 | none | 0 |
| topk_128 | 10 | 15 | V=14, MTE3=2 | 1 |
| conv2d_interior | 24 | 0 | M=8 | 1 |
| flash_attention_cube | 18 | 15 | none | 1 |
| triangular_inverse_16 | 278 | 277 | none | 1 |
| gdn_wy | 16 | 32 | V=13, MTE1=1 | 2 |
| kda_wy | 17 | 35 | V=10, MTE1=1, MTE2=2, MTE3=1 | 2 |

Pairs mean equal total set/wait site inventories, not a cost score. Preserve
sets, waits, named barriers by pipe, and PIPE_ALL separately in machine data.
Record manual/original barrier inventories too, from their actual output.
Keep static sites separate from executed actions: at the timed 16-trip/group
bounds, revised MTE3 executes ZERO times in two/three-buffer, four-use and TopK.
The remaining sites protect the arithmetic-overflow fallback.

GEMM's 56 versus 53 is a static difference. The recorded nonempty launches
execute two more pairs overall than manual, not three per iteration. Eight
recorded traces match manual completion prefixes; neither fact predicts runtime.
An output with 44 pairs and named GEMM barriers is the old plan, not this arm.

## Reuse the latest device harness

Prefer the verified archive:

```text
/opt/pypto/insertsync-r8-4c19cc1c-a3-walltime.tar.gz
sha256 1afef7a708c21257520f1e30839f94e3c69c2f5d6f1ba068010204d8c8752ccd
```

It contains coordinated GDN/KDA launchers and the previously working controls,
GEMM, TopK and triangular-inverse paths. Its reported environment was Ascend
910B2, CANN 9.0.0, pto-isa
`1216c55831fcc4ed2f096e4ca582ff75a633fbb6`. Reuse that compatible toolchain;
record any necessary shared compatibility change. Keep prior archives intact.

For historical original binaries/wrappers, an earlier verified archive is:

```text
/opt/pypto/insertsync-revision-c456adc12-a3-device-final.tar.gz
sha256 5686622e53f7c0422ca7fee12ab9f76bfc28b018ce361cf65fa4e4de1145e269
```

Reusing a binary requires matching inputs, compiler, wrapper, toolchain, flags,
ABI and launch configuration. Collect fresh timing; do not import old medians.
The prior reports used different measurement intervals and baseline arms.

Fix the RECORDED rotation error in R8's mixed timing driver: a two-arm list was
indexed using `block % 4`, producing a 9:3 order imbalance. Rotate by the actual
distinct-arm count and assert equal occurrence of each arm in each position.
Perform this after binary deduplication.

Compare actual device kernel binary bytes with matching wrapper/launch contracts.
Identical arms share a timing series with `IDENTICAL_CODE` provenance. Equal
counts or normalized C++ are insufficient. Do not present aliases as independent
paired samples.

## Execution order and bounded correctness

1. GEMM: 4096×4096×4096 and 2048×4096×4096, 24 cores, established
   `C = A × B^T` layout and launcher.
2. One/two/three-buffer and four-use: 16 trips.
3. Triangular inverse: 16 matrices; TopK: 16 groups.
4. GDN and KDA: 16 full chunks, using coordinated peers.
5. Conv2D: 32 panels; FlashAttention: 16 tiles, if their shared blockers resolve.

For each distinct executable at each timed shape, run 50 checked launches with
seed 2026, poisoning outputs and checking guards before timing. Retain queue
state across launches as required by the actual protocol. Check outputs after
timing blocks outside the clock too.

Add five checked launches at valid counts 0, 1 and 3 for controls, TopK and
triangular inverse. For GEMM, check 128×256×512 and 128×256×1536 with the
established launch geometry, including cores with no assigned work. Do not
invent a zero-K numerical contract. For available mixed launchers and Conv2D,
check counts 1 and 3. Conv2D's zero-panel case is outside its contract.

Hand-tuned TopK has a known intermittent index defect: retain its correctness
result but exclude it from timing ratios even if this short run happens to pass.
Exclude every other failing executable from timing as well.

Conv2D previously built but faulted on launch with 507015 in every arm.
FlashAttention previously failed all-arm device compilation on the TLOAD
ND-to-ZN assertion. Try existing validated fixes first, with at most ten minutes
total for these two blockers. Then retain their rows with concrete failure
evidence and finish the measured rows. GDN/KDA are no longer presumed blocked.

Preserve fixed peer/adapter synchronization across arms and count it separately.
Launch cooperating cores together with genuine queue/FFTS protocols. Do not
insert host-produced credits, serial peers or local synchronization repairs.
Use 60-second correctness and 120-second timing invocation timeouts. On a hang,
stop the affected campaign invocation and report it rather than retrying blindly.

## Wall-time protocol

Primary metric: single-launch synchronized HOST wall time, microseconds.
Use a persistent process and C++ runtime where supported. With the stream idle,
start `steady_clock` immediately before submission and stop after stream
synchronization returns. Batch=1. Exclude Python dispatch, allocation, data
transfers, initialization, numerical reference work and comparisons.

Record launch method and timer boundaries. Do not mix amortized multi-launch
intervals with batch=1 or subtract an estimated launch overhead. Existing
device-clock measurements may be retained as a separately labelled secondary
metric; adding instrumentation must not delay this experiment.

For each shape: 20 warmups, then 12 balanced blocks with 25 samples per distinct
correct arm per block. Rotate the actual arm list; 12 blocks balances one, two
or three distinct arms. Keep paired arms on the same otherwise idle card.
Different fixtures may use separate available cards, but never overlap measured
arms on one card. Validate the position histogram before interpreting results.

Report host p10/p50/p90 and paired ratios of BLOCK medians with 95% bootstrap
confidence intervals, resampling whole blocks:

```text
revised_generations / original_insert_sync
revised_generations / hand_tuned
original_insert_sync / hand_tuned
```

Lower is faster. A CI including 1 is inconclusive; use 2% as the practical-effect
threshold and report small estimates as small. Keep the fixed sample budget;
do not extend runs to chase significance. Small controls may remain dominated
by launch overhead. If device-clock conclusions differ, report that explicitly
while retaining host wall time as the primary result.

## Deliverable

Print a readable timing table for all eleven fixtures, with both GEMM shapes:
arm medians, ratios/CIs, correctness and `MEASURED`, `IDENTICAL_CODE`,
`CORRECTNESS_FAILURE`, `BUILD_BLOCKED`, or `HARNESS_BLOCKED` status.
Print synchronization mechanisms in a separate table without combining types.

Answer directly: does the new 56-pair GEMM close the manual gap, and do the
control/TopK changes affect wall time? Report improvements and regressions
against genuine original InsertSync. Do not assign old R8 timing to new output.

Deliver a fresh archive named `insertsync-generations-862ab8111-a3-walltime.tar.gz`
with a verified SHA-256 sidecar. Include REPORT.md, raw and summary timing CSVs,
correctness, arm identity/order records, source/toolchain fingerprints, exact
commands, generated PTO/C++, static/executed synchronization counts, and any
adapter patch. Verify extraction/manifests. Preserve prior archives and source
trees; terminate only this campaign's processes. Report time spent on compiler
preparation, device compilation, correctness and timing.
