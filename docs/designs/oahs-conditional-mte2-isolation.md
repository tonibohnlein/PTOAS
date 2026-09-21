# Conditional MTE2 overwrite: device isolation complete

## Result

The correctness-only `deferred_ack` experiment isolates one conditional
same-storage overwrite. Adding one MTE2 barrier to the existing synchronized
plan repairs all three failing seeds; OAHS already has this barrier.

| Arm | Active=1, seeds 1/2/3 | Active=0, seed 1 |
| --- | --- | --- |
| Existing InsertSync | FAIL / FAIL / FAIL | PASS |
| Existing + conditional MTE2 barrier | PASS / PASS / PASS | PASS |
| OAHS default | PASS / PASS / PASS | PASS |

Job `task_20260920_212311_12321018449`, card 0: 12 harness runs and 48 queued
launches. No timing or profiling was performed. No job remains running.

The static requirement is completion of the preceding write before the
conditional overwrite issues. Waiting for both loads before the later reader
is insufficient to establish ordering between those writes. The mutation and
controls support this WAW explanation and establish that this barrier is
sufficient for the tested kernel, site, seeds and device. This is not a compiler
implementation fix, a minimality result or evidence about other kernels.

## Local archive audit

Downloaded archive SHA-256:
`97aa8d60f6740582f8ff0fc924a0875f158239bd495436232af3f280cfdbb8de`.
All 32 manifest entries verify (33 regular files including the manifest).
The standalone report matches the archived report byte-for-byte.

The local audit verifies:

- Patched PTO equals original PTO plus exactly one conditional barrier line.
- Patched C++ equals original C++ plus exactly `pipe_barrier(PIPE_MTE2);`.
- The 12 raw run records reproduce the table; inputs match across arms.
- Loaded-binary hash records match all three build records and are distinct;
  wrapper and harness source hashes agree. The new archive contains records,
  not the `.so` bytes, so this is not a fresh binary-byte or runtime-load audit.
- Both actual cc1 command expansions end with effective `-O2`; compiler version
  banner lines containing `cc1` are excluded from command counting.

The device report additionally records byte-identical re-lowering of the
original archived C++, ruling out observed lowering drift in that comparison.
No archived executables or scripts were run locally.

Reproducer and detailed audit:
`/home/toni/work/pypto3_sync_more/mte2-isolation-review/audit.py` and `audit.json`.
The extracted archive and raw logs are alongside them.

## Recovered evidence and corrections

The archive recovers the earlier dedicated probe (12/12 existing taken-path
failures, all other groups passing) and row22's nine passing correctness runs
with identical per-seed output hashes across its three arms.

Campaign accounting is now:

| Campaign | Verified recovered host matrix |
| --- | --- |
| MAT reader-region | 264 successful construction/lowering rows: 88 cases × 3 arms |
| Placement experiments | 36 attempted constructions: 35 successful and 1 expected default-OAHS refusal for `class_invariant` |

The report's correction correctly separates the campaigns, but the accompanying
message's “36, all rc=0” is still inaccurate. The recovered 16-row matrix has
15 successes and that expected refusal; the 20-row matrix has 20 successes.
The placement campaign's four pinned-input reruns must not be described as its
own independent 88-module sweep. Our later local 88-module sweeps remain separate.

No-motion GEMM was intentionally host-only: 200/394/782 → 330/652/1296 pairs,
with zero added or removed checked ordering relations. It has no device binary,
timing samples or job. This is not an undelivered requested device measurement.

The standalone placement report currently in local Downloads still has no
ERRATUM appended. The correction is available in the isolation report and is
recorded here; the original placement archive remains immutable.

## Next action

Close the barrier-isolation device task; no further repetitions are needed for
this decision. Preserve the failing existing plan as a compiler regression and
trace the existing pass's conditional WAW decision before designing a general
repair. This is separate from OAHS placement improvements: OAHS needs no new
barrier for this case. Keep the failing arm out of timing comparisons.
