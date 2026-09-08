# Device results for 862ab8111: verified interpretation

The 56-pair shared-generation GEMM is competitive with the hand-tuned kernel
on both measured shapes. Its runtime is about 19% and 17% lower than genuine
original InsertSync. Three additional static pairs are not a demonstrated
remaining performance problem.

Archive supplied by the device agent:
`insertsync-generations-862ab8111-a3-walltime.tar.gz`, SHA-256
`8507004abb1c23dee08c0b9c961f798b8c98288d85cb9be2c78e00afa66321bb`.
Local verification matched that hash and every one of the 462 manifest-listed
file hashes and byte counts. All 8,400 raw samples use batch=1; each of the 336
arm/block groups contains 25 samples. Independently recomputing the median
paired-block ratios and 20,000 block bootstrap resamples (seed 2026) reproduces
the reported GEMM and GDN estimates and confidence intervals.

## GEMM

Host microseconds are pooled p50 values. Ratios are medians of paired block
ratios, so they need not equal the ratio of the pooled p50 values.

| Shape | Hand | Original | Revised | Revised/original, 95% CI | Revised/hand, 95% CI |
| --- | ---: | ---: | ---: | --- | --- |
| 4096 cubed | 511.3 | 633.7 | 513.1 | 0.8102 [0.7926, 0.8332] | 1.0075 [0.9901, 1.0202] |
| 2048 x 4096 x 4096 | 293.1 | 350.1 | 291.8 | 0.8316 [0.8193, 0.8560] | 1.0000 [0.9737, 1.0116] |

The accurate conclusion is **matches hand-tuned performance within this
experiment's precision**. A CI containing one is not a formal proof of
equivalence: the first interval permits roughly 2% slower performance, and
this does not establish behavior on other shapes or devices. The earlier
20–24% manual advantage over original is no longer observed against revised.

These results qualify `862ab811124d2e67a7f2cde992461983163f6a7a` on Ascend
910B2 with the recorded toolchain. They are not new measurements of `3400426f3`.
Original is `7e2ec3e29420e297dcf5b3c59ba4d90841464821`. The automatic arms use
the same input; the archive notes that manual and automatic GEMM inputs are
different scheduled programs. The manual comparison establishes an achieved
performance reference, not an isolated causal estimate for each removed barrier.

## What else the data establish

- **GDN regression:** revised/original is 1.0304 [1.0022, 1.0458]. The only two
  textual changes in generated GDN C++ move a V barrier from before to after an
  MTE2-to-V wait, before `TCVT` and `TEXP` respectively. That is a concrete
  placement hypothesis; a focused repeat/ablation should establish causality.
  Equal static counts alone would not identify the cause.
- **Conv2D correctness:** manual passes at panel counts 1 and 3; the shared
  original/revised automatic binary faults. Earlier claims that every arm
  failed were incorrect. This is an automatic-arm correctness failure shared
  with production, not a newly introduced revision regression. Isolate helper
  effects, synchronization and adapter interaction before asserting a root cause.
- **Triangular inverse:** original and revised have identical device code and
  remain about 7% slower than manual in this campaign. The revision has not
  addressed that case.
- **TopK:** the paired estimate is about 1.9% faster, with an interval just
  excluding one. This is a small measured result under the task's 2% practical
  threshold, rather than an interval that includes one. Manual remains excluded
  because of its index defect.
- **Controls and KDA:** the intervals do not resolve a change. This does not
  establish that their synchronization differences are free or irrelevant.
- **FlashAttention:** still a build blocker across the tested arms, providing
  no device synchronization comparison.

## Two qualifications to the supplied report

First, the archived GEMM, control, TopK and triangular-inverse wrappers start
the host clock **before recording a device event**, then submit the kernel,
record another event and synchronize before stopping the clock. These are
instrumented host-wall intervals. The observations remain valid for that shared
measurement protocol, but the interval includes instrumentation beyond the
requested kernel submission and drain. This matters particularly for small
kernels. GDN/KDA use the separate mixed host launcher.

Second, the control explanation in the report is incorrect. Scalar replay of
the actual archived outputs at 16 trips gives, for each of two-buffer,
three-buffer and four-use:

| Arm | Executed MTE3 barriers | Executed PIPE_ALL |
| --- | ---: | ---: |
| Original | 16 | 1 |
| Revised | 0 | 0 |

The fact that revised fallback sites do not execute does not imply there was
no dynamic synchronization reduction. The experiment simply does not resolve
its wall-time benefit with this timer and sample population.

## Development decision

Measure the current committed approach now, before broadening its analyses.
Use Qwen QK/Q projection to isolate the latest boundary/command changes, GEMM
to preserve the demonstrated result, and GDN to test the two identified barrier
placements. Reproduce and reduce Conv2D as a correctness task; do not time its
failing executable or relabel it as an all-arm harness blocker.

Descriptor/operation summaries and graph-size work remain justified by the
corpus, but should follow concrete supported-effect contracts and bounded
acceptance cases. Command-efficient lowering should preserve the boundaries
that now deliver manual-level GEMM performance. Do not prioritize removing
three static GEMM pairs without evidence of a runtime cost.

The new task is
[DEVICE_TASK_INSERTSYNC_SHARED_3400426F3_WALLTIME.md](../../../../DEVICE_TASK_INSERTSYNC_SHARED_3400426F3_WALLTIME.md).
It uses a clean primary host timer, a separate secondary device timer, fresh
paired measurements and exact byte-based arm deduplication. Its required inputs
and goldens are already committed; it does not depend on local corpus artifacts.

Local extracted evidence and independent checks:
`insertsync-builds/campaign/device-generations-862ab8111-verified/`.
The accompanying JSON preserves the source ratios, local bootstrap verification
and control replay. The supplied archive remains unchanged.
