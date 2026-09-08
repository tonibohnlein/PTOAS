# R8 device results and local evidence review

Received and inspected on 2026-09-08. This campaign tests R8
`4c19cc1cba444f28ab9af2a022aae1e3c80079eb`, R7
`c720df6bad7db78b07e97c5499f9c435f80c330f`, a lifecycle-disabled residual
baseline, and hand-tuned inputs. The baseline includes staged insertion and
MMAD; it is not the historical original InsertSync arm. The new uncommitted
per-buffer generation implementation is **not included**.

Archive: `insertsync-r8-4c19cc1c-a3-walltime.tar.gz`.
SHA-256: `1afef7a708c21257520f1e30839f94e3c69c2f5d6f1ba068010204d8c8752ccd`.
Local verification matched this hash and all 620 internal manifest file
hashes/byte counts. Extracted evidence is retained under:

```text
/home/toni/work/pypto3_sync_more/insertsync-builds/device-feedback/insertsync-r8-4c19cc1c-a3-walltime
```

The archive reports Ascend 910B2, CANN 9.0.0, and pto-isa
`1216c55831fcc4ed2f096e4ca582ff75a633fbb6`. Measurements below are taken from
its CSVs/report, not a new local device run.

## Host wall time

Microseconds per synchronized launch, batch=1, global p50. Equal automatic
entries can be aliases sharing one timing series: the archive records exact
device-binary identity, not equality inferred from synchronization counts.

| Fixture / shape | Hand | Baseline | R7 lifecycle | R8 lifecycle |
| --- | ---: | ---: | ---: | ---: |
| One buffer / 16 | 80.1 | 73.7 | 77.1 | 80.1 |
| Two buffers / 16 | 68.9 | 66.1 | 67.6 | 67.6 |
| Three buffers / 16 | 63.3 | 67.2 | 68.5 | 68.5 |
| Four-use / 16 | 76.1 | 76.4 | 80.7 | 75.1 |
| GEMM / 4096³ | 519.2 | 617.5 | 617.5 | 617.5 |
| GEMM / 2048×4096×4096 | 292.6 | 340.1 | 340.1 | 340.1 |
| Triangular inverse / 16 | 145.7 | 148.9 | 148.9 | 148.9 |
| TopK / 16 | Excluded | 80.8 | 80.8 | 80.8 |
| GDN / 16 | 85.3 | 74.8 | 74.8 | 74.8 |
| KDA / 16 | 75.4 | 68.8 | 68.8 | 68.8 |

Ratios use paired block medians with 95% bootstrap intervals; they are not
division of the global p50 values above.

- Four-use R8/R7: **0.9030 [0.8588, 0.9871]**. R8 restores MTE3 barriers
  from eight sites to the baseline's two. R8/baseline is
  0.9752 [0.861, 1.060], which does not establish an improvement over baseline.
- GEMM's baseline, R7, and R8 have identical device code. Automatic/hand ratios
  are **1.1914 [1.162, 1.216]** and **1.1646 [1.151, 1.174]** for the two
  shapes. The gap predates R8.
- One-buffer R8 and hand are identical device code. Two/three-buffer R7 and R8
  are identical. TopK, triangular inverse, GDN, and KDA have identical automatic
  arms according to the archive's identity records.
- Reported GDN/hand is 0.8718 [0.847, 0.895]; KDA/hand is
  0.9180 [0.902, 0.940]. **The local review found an arm-order imbalance for
  these two rows; see below before treating these as confirmed gains.**

## Timing-method findings

The archive contains 7,200 samples from 24 distinct fixture/shape/arm series,
12 blocks × 25 samples each. Batch=1 control intervals are dominated by host
submission/synchronization overhead: roughly 63–80 µs wall time for roughly
5–13 µs device work. These wall-time numbers are not interchangeable with the
earlier campaign's amortized per-launch measurements. The supplied Python
runner also starts a subprocess for each arm/block; it does not maintain one
process across all blocks.

**Confirmed mixed-arm rotation defect:** `harness/time_mixed.py` rotates a
two-element `ARMS` list using `b % 4`. For both GDN and KDA, `timing_raw.csv`
confirms hand runs first in nine blocks and second in three, while the automatic
arm has the reverse distribution (225 versus 75 samples in each position).
This contradicts the report's fully balanced-order description. The reported
intervals do not by themselves account for that order confounding. A targeted
repeat should rotate with `b % len(ARMS)` and balance the two orders. No repeat
was run here. Four-use's four-arm ordering is balanced in the raw data, and
GEMM's automatic-code identity is unaffected by this finding.

The four-use host result should remain labeled a host-wall result: its raw
global device p50 changes from 13.14 to 12.87 µs, which is not the reported
9.7% paired host reduction. No device-clock significance test was performed in
this review, and the association with barrier removal alone does not isolate
the source of the full host effect.

## Correctness, coverage, and mechanisms

`correctness.json` contains 106 PASS and two NONDETERMINISTIC rows, both the
known hand-tuned TopK index defect. Manual TopK is excluded from timing.
Nine of eleven fixtures ran; GDN/KDA now use coordinated mixed-core launchers.
Conv2D builds but launch faults with 507015 in all arms. FlashAttention fails
device compilation in all arms, including hand-tuned, on the TLOAD ND-to-ZN
layout assertion. These shared blockers do not identify an R8 regression.
Their repairs were left outside the device task's ten-minute adapter allowance.

The archive reports exact reproduction of all 33 checkable R8 static reference
rows. Its `sync_static.csv` retains set and wait sites, named barriers by pipe,
and PIPE_ALL separately. For GEMM, all automatic arms still have 44 sets,
44 waits, MTE2=3, MTE1=2, M=4, FIX=1, and PIPE_ALL=1; hand has 53 sets,
53 waits, no named barriers, and no PIPE_ALL.

The subsequent [local generation results](BUFFER_GENERATION_RESULTS.md) remove
the five MTE2/MTE1 sites and emit 56 pairs. Those changes need a separate device
comparison; this R8 archive neither confirms nor refutes their runtime effect.
