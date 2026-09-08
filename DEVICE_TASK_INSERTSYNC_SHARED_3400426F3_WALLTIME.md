# Device task: current shared-requirement InsertSync

Measure the committed current approach before another analysis expansion.
The primary new questions are whether QK's earlier readiness helps and whether
Q projection's reduced commands help. Preserve the qualified GEMM result,
recheck the GDN regression, and reproduce the automatic Conv2D failure.

This task is self-contained. All required compiler code, inputs, goldens and
reference manifests exist in the pinned repository below. The new local full
corpus reports and its external generated-input tree are not dependencies.
Do not run the full compiler corpus, a full test suite or a profiling campaign.

## Pins and reuse

Repository: https://github.com/tonibohnlein/PTOAS.git

| Arm | Exact compiler source |
| --- | --- |
| original | `7e2ec3e29420e297dcf5b3c59ba4d90841464821` |
| generations_862 | `862ab811124d2e67a7f2cde992461983163f6a7a` |
| current_shared | `3400426f393aac606c134663d5517218fe9a3dc6` |
| hand_tuned, where specified | current compiler, explicit manual input, autosync disabled |

The current commit was verified on remote branch `codex/insertsync-revision-r1`.
Fetch the exact object; the moving branch name is only a locator. Reuse verified
original and 862 builds. Build only required current compiler/runtime targets,
using an explicit worker count appropriate to the remote machine. Record
compiler preparation separately; keep generated artifacts outside source trees.

Reuse the latest verified device archive:

```text
/opt/pypto/insertsync-generations-862ab8111-a3-walltime.tar.gz
8507004abb1c23dee08c0b9c961f798b8c98288d85cb9be2c78e00afa66321bb
```

Its reused dependencies remain pinned to:

```text
/opt/pypto/insertsync-r8-4c19cc1c-a3-walltime.tar.gz
1afef7a708c21257520f1e30839f94e3c69c2f5d6f1ba068010204d8c8752ccd
/opt/pypto/insertsync-revision-c456adc12-a3-device-final.tar.gz
5686622e53f7c0422ca7fee12ab9f76bfc28b018ce361cf65fa4e4de1145e269
pto-isa 1216c55831fcc4ed2f096e4ca582ff75a633fbb6
```

The measured environment was Ascend 910B2, CANN 9.0.0. Prefer that compatible
toolchain. Verify archive hashes and source/build fingerprints; record device,
toolkit, Python ABI and loaded native libraries before and after. Preserve prior
archives. Reuse device binaries only with matching inputs, flags, toolchain,
wrapper ABI and launch contracts. Collect fresh timings.

## Compiler options and inputs

Use inputs and harness sources from current commit for every arm. Original gets
`--enable-insert-sync` only. Both revised arms get:

```text
--enable-insert-sync
--insert-sync-gm-alias=assume-disjoint-arguments
--insert-sync-effect-coverage=report
--insert-sync-defer-same-pipe
--insert-sync-mmad-chains
--insert-sync-buffer-generations
```

Use A3 and each manifest's original level. Keep other experimental/pruning/
frontier flags off. Maintain the same distinct-GM-argument allocation contract
as original's historical behavior. Do not substitute R8-with-lifecycles-off for
genuine original, or pass new flags to an old compiler that does not implement them.

All paths below are relative to `test/experiments/insert_sync/performance/`:

```text
a5427e8b7eb8a8d8e838f6cdb0dadc8e83448ba091707da3789c813b95cba09f  manifest.json
e6f04371ecb556c71dd44013c3aa6833e7d52f4f811336a1a3557f02ede020f3  kernel-pairs-manifest.json
cf08e32e10493462f2af68aa1584c1f1390427c6d0c40d4b9c2a6f3707982a8e  qwen-additions-manifest.json
778f93bb4111e8d251e15d6ae403d7c5954537ea464de852bbf08cccd206f450  device/reference-manifest.json
a818d45578628de0f92492b1309a694fedde1381ee925df37bc7a4840f427f7a  SHARED_PUBLICATIONS_RESULTS.json
```

Verify manifests and referenced input/golden hashes. Emit PTO and C++ for the
selected cases, retain argv/diagnostics, and verify actual pass participation.
Preserve payload, buffer addresses, views, helper/peer code and allocation
contracts across automatic arms. Report any mismatch before attributing timing.

## Cases, expected differences and correctness

Run the following mandatory cases in this priority order. Timed arms are the
three automatic arms, plus hand for historical GEMM. Hand GDN timing is optional.

1. **Qwen `qk_matmul`:**
   `test/samples/Qwen3DecodeA3/kernels/aic/qk_matmul.pto`.
   Time batch index 0, explicit SPMD pair index 0, block count 1, at context
   blocks 1 and 16. Use one physical Cube worker. Check contexts 0, 1, 2, 3 and
   16, including pair indices 0 and 3 (logical block counts 1 and 4 respectively,
   still executing one selected physical worker). Pair index 3 accesses groups 6/7 of the
   existing compact batch-zero buffers. Do not change tensor strides or IR.
   Empty execution must preserve output and retire its two preloads correctly.
2. **Qwen `q_proj`:**
   `test/samples/Qwen3DecodeA3/kernels/aic/q_proj.pto`.
   Time explicit SPMD output-block index 0, block count 1, one physical Cube
   worker. Check output-block indices 0, 1 and 31 with logical block counts
   1, 2 and 32 respectively and full backing allocations; execute one selected
   worker and verify the other output regions remain unchanged.
   This is the unchanged 16x8192 input and 8192x8192 weight problem, producing
   one 16x256 block. Its loop runs 32 iterations, processing two K tiles each.
3. **Historical GEMM:** existing manual/automatic inputs and validated 24-core
   launcher; time 4096 cubed and 2048x4096x4096. Check the established small
   128x256x512 and 128x256x1536 configurations, including idle cores.
4. **GDN:** 16 full chunks, using the coordinated cube-plus-two-vector launcher.
   Check counts 1, 3 and 16. Compare against both original and 862; preserve
   real queue/FFTS protocols and fixed peer synchronization.
5. **Conv2D, correctness only:** run manual and distinct automatic executables
   at panel counts 1 and 3 with the validated adapter. Manual passed and the
   original/862 shared automatic binary faulted in the previous campaign.
   Record current behavior, first failing call and relevant runtime log. Stop
   a failing executable after reproduction; never time it. Do not infer one
   arm's launch result from another arm's build result.

Qwen goldens are committed beside those kernels:
`qk_matmul_golden.py`, `q_proj_golden.py`, `qwen3_decode_golden_lib.py`, plus
`test/samples/validation_runtime.py`. Use their reference mathematics and exact
ABI interpretation. The existing scaffold is
`test/npu_validation/scripts/generate_testcase.py`; it provides these kernels'
pointer-allocation minima and scalar defaults. Keep the Qwen3DecodeA3 source
hierarchy when staging generated C++ to preserve sample-specific handling.
Generate a case with `--input <case.cpp> --testcase <kernel-name>
--output-root <artifact-directory> --run-mode npu --soc-version Ascend910B2`.
Copy the custom golden and its support imports into that artifact directory.
Reuse existing board validation glue where available; adapt only the shared
host timing wrapper to collect repeated samples with persistent allocations.

At each timed shape, check 50 launches per distinct executable; use five checks
at each additional valid correctness configuration. Use deterministic inputs,
check full expected outputs plus untouched regions and allocation guards, and
poison/reinitialize outside the clock consistently with the golden's initial
state. Check again after timing blocks. Record tolerances; do not loosen them
or change payloads to make an automatic arm pass. No zero-K GEMM contract is
invented. Do not launch cooperating peers serially or supply host-made credits.

Expected static sites, retained separately from execution:

| Case | Original sets/waits | 862 sets/waits | Current sets/waits | Named barriers | ALL original/862/current |
| --- | ---: | ---: | ---: | --- | ---: |
| QK | 21/21 | 21/21 | 22/21 | M=2 in each | 1/0/0 |
| Q projection | 19/19 | 45/45 | 39/39 | M=6 in each | 1/1/1 |
| GEMM | 44/44 | 56/56 | 56/56 | original M=18,MTE1=2,FIX=1; revised none | 1/0/0 |
| GDN | 32/32 | 32/32 | 32/32 | V=13,MTE1=1 in each | 2/2/2 |
| Conv2D auto | 0/0 | 0/0 | 0/0 | M=8 in each | 1/1/1 |

Manual GEMM is 53/53 without barriers; manual Conv2D is 24/24 without barriers.
QK's two alternative publication sites are mutually exclusive and share a
stream. At positive context count c, each arm executes 16*c+5 pairs; current
adds five scalar control operations per launch while its first Q0 extract no
longer acquires the independent Q1 preload. Q projection executes 389 / 779 /
651 pairs respectively; the current and 862 concrete payload completion
prefixes match in the recorded replay. Count all mechanisms by type, never as
one score. Preserve discrepancies rather than adjusting expected counts.

## GDN diagnosis and bounded work

The archived original/revised GDN C++ differs at exactly two sites: a V barrier
moves from before to after an MTE2-to-V wait, before TCVT and TEXP. First compare
the current emitted C++ and device binary with 862. If the 3% slowdown repeats,
an optional diagnostic variant may restore only those two barrier positions.
Keep payload and all other events unchanged; require the same correctness gate.
Label it `gdn_barrier_order_probe`, not an accepted compiler fix. If its raw
device binary equals original, alias it instead of timing another series.

Aim to finish within **30 minutes after compiler binaries are ready**, with at
most eight minutes total for new Qwen harness adaptation and Conv2D diagnosis.
Do the mandatory timing cases first. Remaining six Qwen additions may receive
correctness checks if their existing harnesses work within that budget; extra
timing is optional and cannot displace QK/Q projection/GDN. Retain unresolved
cases with concrete status. Do not reopen FlashAttention's existing all-arm
TLOAD build blocker in this task. The newly discovered external-corpus A8W8
and small-GEMM probes need a separate portable input/golden package.

## Timing and attribution

Primary metric: **clean, single-launch synchronized host wall time**, batch=1.
Use a C++ steady clock with the stream idle before the interval:

```text
start clock
submit one kernel (or one coordinated mixed-kernel group)
synchronize its stream
stop clock
```

No device-event recording belongs inside this primary interval. The prior
archive's GEMM/control timer included two event submissions: retain those older
results as instrumented wall time, and collect fresh baselines with the new
common timer. Exclude Python dispatch, initialization, allocation, transfers,
reference generation and checking. Do not subtract estimated overhead.

Collect device-event time in a **separate** fixed-size run for QK and Q
projection to help interpret launch-dominated results. Label clock domains and
sample populations separately. Do not substitute secondary results for host
wall time. Reuse persistent processes/contexts and buffers; at minimum keep
them alive for an entire block, never relaunch a process for each sample.

Deduplicate on actual device kernel binary bytes with matching wrapper/ABI and
launch contracts, before choosing arm order. Identical arms share one fresh
series with `IDENTICAL_CODE` provenance; equal counts or normalized C++ are
insufficient. In particular, current GEMM/GDN may alias 862 and should not be
presented as independent measurements if they do.

Use 20 warmups and 12 balanced blocks of 25 samples per distinct correct arm
and shape. Rotate by the actual distinct-arm count. Assert both complete
sample cardinality and flat position histograms after deduplication, not merely
balance among whichever samples survived. Use paired arms on the same idle
card; different fixtures may use separate cards without timing overlap on one
card. Apply 60-second correctness and 120-second timing-invocation timeouts;
stop and report an affected lane on a fault/hang, preserving other completed
results. Terminate only campaign-owned processes.

For each clock, report p10/p50/p90 and median paired block ratios with 95%
bootstrap CIs: 20,000 resamples of whole blocks, seed 2026. Ratios are current /
original, current / 862, and 862 / original; include current / hand for GEMM.
Keep both current/original and current/previous visible on Q projection. A CI
containing one is inconclusive; estimates below the 2% practical threshold are
small, even when their CI excludes one. Do not extend sampling to chase a
significance threshold. Do not call inconclusive results proof of equivalence.

## Deliverable

Print a readable wall-time table, separate synchronization table, correctness
and binary-identity results. Answer: did current preserve GEMM's manual-level
performance; did earlier QK readiness or Q-projection command removal help;
does the GDN regression reproduce and does the order probe explain it; does
current still fail Conv2D? Distinguish earlier archived findings from new runs.

Deliver `insertsync-shared-3400426f3-a3-walltime.tar.gz` with verified SHA-256
sidecar and internal manifest. Include raw samples, bootstrap script, complete
arm/block order, input/ABI/toolchain fingerprints, generated PTO/C++, correctness
logs, timer source and adapter/probe diffs. Verify extraction and byte counts;
preserve prior archives and report preparation, adaptation, correctness and
timing durations separately. A missing harness is a recorded blocker, never an
inferred all-arm failure or a reason to invent a numerical contract.
