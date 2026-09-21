# Device task: MAT reader-region cycles

## Objective and immutable arms

Measure whether returning MAT storage at its reader-region boundary removes the
remaining projection serialization. Read `mat-release-manifest.json` and verify
`SHA256SUMS`. This package is self-contained; no workstation files are required.
The candidate is an explicitly labelled **working-tree source snapshot**, based
on `fe1fc454bb80cd4810410bcc8bd9212f36c8b5d8`. Its exact archive, tree hashes and
patch are supplied. Do not label the snapshot as the unchanged fe1 commit.

Use four arms on identical prepared payloads:

1. **existing:** existing InsertSync, from the baseline source.
2. **baseline:** OAHS at fe1fc454b, the measured first-consumer candidate.
3. **placement_control:** candidate emitted plan with precisely the baseline
   MTE2 fences restored at their original payload deadlines. All baseline named
   fence positions are retained; supplied PTO and generator are test artifacts.
4. **candidate:** the new production OAHS output, without those now-unneeded
   MTE2 fences. M/FIX fences and the terminal ALL remain.

Candidate vs baseline measures the complete mechanism. Placement-control vs
baseline measures the protocol/control placement with the same fence population.
Candidate vs placement-control isolates omission of those MTE2 fences. Neither
protocol comparison isolates event-count overhead: counts also change. Do not
attribute time from a pair-count difference alone.

The complete selected readiness/return cycles are installed before ordinary
repair. There is no fence-deletion pass or stronger same-pipe hardware assumption.
Candidate also qualifies the actual first consumers for the second child and
odd-tail inputs; count this control precision as part of the mechanism.

## Priority and execution

1. down_proj rows0/1: correctness, timing, matched profiling; report promptly.
2. LM head row6: transfer check, then gate/up2/3, KV4/5, Q/out7–10.
3. Shenggan: preserve 200/394/782 pairs, zero named barriers, terminal ALL.
4. Reconstruct all six attention rows44–49 and post-RMSNorm22/23. Their candidate
   host plans are byte-identical to fe1; report whether lowered/loaded binaries
   are also identical. Reuse existing qualification for unchanged controls only
   with exact input/source/binary provenance. No manual-attention transcription
   or new queue contract is part of this task.

Reuse the qualified remote harnesses from the first-consumer campaign, recording
source hashes and ABI/shape/scalar arguments. The package also includes earlier
harness sources for recovery. Avoid building new numerical references unless the
existing qualified oracle cannot serve these identical payloads. Generate and
verify each seed's reference once, reuse it across arms, and use cache-friendly
host computation. Use all eight remote devices, with independent complete
arm comparisons on different devices and timed/profiled exclusivity per device.
Keep all arms of each matched comparison on the same device. Follow the
[eight-device scheduling addendum](oahs-eight-device-scheduling.md); preserve
per-device results and keep the verified source archives fixed.

Keep prior campaigns pinned. Continue dependent stages without leaving the device
idle. Respect the broker's observed job time cap; split large jobs into resumable
units. Record queue/run/host-reference times separately. Do not duplicate jobs.

## Host acceptance before device work

- Build baseline from `sources/first-consumer.tar.gz`; build candidate from the
  source archive named in `mat-release-manifest.json`, or apply its exact binary
  patch to fe1 and verify every candidate-tree hash.
- Regenerate candidate plans from packaged `inputs/`. Compare with all candidate
  host plans. Native construction and independent reconstruction must pass.
- For placement_control, use the supplied PTO or run `tools/restore_fences.py`
  against baseline/candidate. Verify static original payload sequence, fence
  positions, and the recorded per-case payload-cut indices.
- Do not rerun InsertSync during lowering of already synchronized PTO. Record
  the no-second-insertion check and actual compile flags; require -O2 with no
  later override. Archive generated C++, build metadata and the .so actually
  loaded. Hash the loaded path, not an unused build product.
- Compare concrete executed payload instruction, physical effects, context and
  input hashes across arms. Equal MAC time is not an identity proof.
- Run the packaged 37-path ordering comparisons for candidate and control; they
  preserve payloads and add zero checked finish-to-issue relations vs baseline.
  Down's early-MAT and early-release assertions must pass. The baseline must
  fail the new release assertion. Memory/rearming checks remain unchanged.
- Use `sync-pins.json` for exact static populations; distinguish static SET/WAIT
  totals from dynamically matched pairs. Full local evidence is included.

## Numerical gate and timing

Correctness-gate every arm before its timing: three seeds and the established
boundary-block plus queued four-slot launch test (one final synchronize, all
slots checked). Preserve untouched-output, guard and nonfinite checks and the
existing justified numerical criterion. Do not loosen tolerances. An arm that
fails is unqualified and must not contribute a speedup claim.

For new follow-ups, use 10 untimed warmup launches and 20 measured launches/arm
in two rotated rounds, with identical launch parameters and device state.
Keep completed campaign measurements. The scheduling addendum supersedes the
original larger sample budget; do not add nested large batches. Report medians, IQRs, per-round medians and
ratios against baseline and existing. For LM head report full launch coverage;
local one-/two-tile traces are bounded diagnostics, not a full numerical run.

If baseline and candidate GEMM binaries are identical, overlapping timings
measure repeatability, not a speedup. Preserve the established payload-order and
rearming oracle alongside binary identity.

## Matched profiling

Finish the earlier fe1 profiling reconciliation using the SAME timing binary,
scalar arguments, block choices, warm-up and sample population with/without
profiling. Earlier correctness-main profiles and timing-main measurements were
not comparable. Record profiling perturbation per arm.

Then profile all four down arms and the transfer representatives under matched
conditions. Report per-row pipe-active/core ratios before aggregation, the
counter scope/window/core aggregation, and cold rows separately. Archive raw
ResourceConflictRatio and MemoryL0 results if available. Sum-of-pipe-active/core
is an aggregate overlap indicator, not an instruction timeline. Core-minus-MAC
is not stall time. Identify hypotheses as hypotheses; do not assign a specific
wait a measured stall cost without supporting evidence.

## Delivery

Archive exact source/patch/tree manifests, harnesses/oracles, prepared and emitted
PTO, payload identity checks, generated C++, actual loaded binaries, flags, input
hashes, correctness, raw timing/profile rows, summaries and explicit limitations.
Round-trip extract and verify all internal checksums and loaded-binary matches;
report outer SHA-256. A failure must retain its smallest reproducer and exact
stage. No compiler changes or arm substitution during this campaign.
