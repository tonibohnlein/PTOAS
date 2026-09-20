# MAT reader-region cycles: completed device campaign

Reviewed locally on 2026-09-20. This campaign qualifies the MAT source snapshot
corresponding to the mechanism committed in `8afb90f17`. It does not measure the
later opt-in placement experiments in `9f30b9fd8`.

## Results

Medians in microseconds, 180 samples per arm and six rotated rounds. The source
snapshot is pinned by SHA-256
`721966d5bc0075b74d2ec3b3d0ee2706f7cb77750dc5fb0db9e25f90cbe0ac35`.
Baseline is fe1; placement_control restores its MTE2 fences in the candidate.

| Kernel | Existing | fe1 baseline | Placement control | Candidate | Faster than existing |
| --- | ---: | ---: | ---: | ---: | ---: |
| down_proj | 30.077 | 31.646 | 25.704 | 25.792 | 14.3% |
| gate/up | 65.287 | 71.388 | 54.699 | 54.693 | 16.2% |
| KV | 33.572 | 36.318 | 28.771 | 28.783 | 14.3% |
| Q/out AIC | 33.578 | 36.029 | 28.316 | 28.471 | 15.2% |
| LM head | 434.800 | 512.463 | 415.616 | 415.592 | 4.4% |
| Shenggan GEMM | 531.014 | 228.747 | — | 229.182 | 56.8% |

All five projection candidate/baseline and candidate/existing IQR comparisons
are disjoint. Candidate/control IQRs overlap throughout. GEMM candidate/baseline
IQRs overlap and their timing libraries have the identical SHA-256
`1f8376e43eb653969d347439962500a10663152e723fb47193f641af15756e60`.
Its small timing difference is repeatability, not a new improvement. This
campaign contains no new manual arm; prior manual parity remains prior evidence.

The agent reports 189 final correctness runs passing: down36, gate/up48, KV36,
Q/out36, LM24 and GEMM9. Earlier missing-binary/nonexistent-arm harness failures
remain in the archive and are explicitly distinguished from final device gates.
All 264 host rows (88 inputs x three arms) construct and lower successfully.
Attention44–49 and post-RMSNorm22–23 plans and C++ are reported unchanged.

## What this establishes

The measured regression against existing is closed across the tested Qwen
projection family. The complete readiness/return protocol improves latency by
18.5–23.4% versus fe1 and 4.4–16.2% versus existing, while preserving GEMM parity.
It is implemented already; no second implementation of this mechanism is needed.

Restoring the MTE2 fences retains the gain. Their omission has **no resolved
latency benefit in this campaign**. This supports the shared protocol/control
placement improvement. It does not isolate endpoint placement from dynamic
event overhead: those populations also change versus fe1. Overlapping IQRs are
not a formal equivalence proof or evidence that fences are universally free.

Matched profiles cover down, gate/up and KV. Their per-row aggregate overlap
indicators rise from baseline 1.74/1.81/1.76 to candidate 2.18/2.40/2.28; the
controls match the candidates. This supports restored concurrency, not an
instruction-level stall attribution. Conflict-specific counters are empty.
LM and GEMM have no matched profile in this campaign.

The report's remaining "residual gap" hypotheses do not establish a remaining
regression against existing: the table now has none. Any further improvement
needs a concrete new dependency/resource witness. The new source-gap, deferred
acknowledgment and class-invariant tasks remain separate experiments.

## Local verification and provenance

Original archive: `/home/toni/Downloads/oahs-mat-release-device-campaign.tar.gz`.
Extracted evidence: `../../../mat-device-final-review/oahs-mat-release-device-campaign/`.
[Original report](../../../mat-device-final-review/oahs-mat-release-device-campaign/report/00-FINAL-REPORT.md),
[checksum audit](../../../mat-device-final-review/verification.json), and
[raw-sample/build-record audit](../../../mat-device-final-review/raw-summary-check.json).

- Gzip integrity/CRC passes.
- All **4,072** entries in the outer internal manifest verify. There are 4,074
  regular files including the two checksum manifests themselves.
- Medians and IQR overlap/disjointness recomputed from 4,140 primary timing
  samples reproduce the headline table.
- All **52** archived `.so` files match their adjacent build-record hash.
  Matching the file actually executed remotely remains the device agent's
  recorded provenance; local hashing is not a fresh device execution.
- All 264 host-result records report zero construction and lowering exit codes.

The downloaded archive's SHA-256 is
`2db32301ca158c2fedfe7223ccaf0544db78d6fb7b9c75f0a2adb0adc9ef0a93`.
The embedded report still lists an older outer hash (`dbafd65c…`) and 4,071
entries. Its internal manifest is consistent, but the detached checksum file
was not present locally during this review. Obtain that file to match the
final remote artifact; do not use the stale embedded outer hash.

## Scheduling and remaining tasks

The campaign used cards1–7. Card0 was unavailable because of a reported lock
problem. Parallel independent comparisons across devices were explicitly
authorized; this is not a violation of the revised scheduling instruction.
All arms of a comparison remain on the same device. A serial Q/out repeat
changes candidate/baseline by about 0.06%; the final table uses that repeat.
This is one interference control, not a proof for every concurrent workload.

Mark MAT family timing and the delivered profiling campaign complete. Next run
the isolated placement microkernel tasks, with per-device matched comparisons
and all available devices. Resolve/report the card0 scheduling problem rather
than treating the machine as a single-device resource. Preserve both existing
source archives and avoid re-running already completed identical comparisons.
