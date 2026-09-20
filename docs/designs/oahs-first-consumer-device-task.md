# Device task: first-consumer projection placement

## Objective and immutable arms

Measure whether earlier MAT readiness publication, acquired at the actual first
B extraction, restores more projection overlap. This task changes that placement;
it retains the genuine M/MTE2/FIX fences. Keep the earlier running campaign pinned.

The supplied archive is self-contained. Verify `SHA256SUMS` and read
`candidate-manifest.json`. The candidate is the exact committed revision recorded in `candidate-manifest.json`,
based on `495fb9cbda7649f15a7fc09d6b1d91ffb7737d54`. Its Git archive and tree
manifest are supplied and hashed. No local workstation paths are required. Use
that pinned revision even if the branch later advances.

Arms:

1. **candidate**: `sources/first-consumer.tar.gz`, OAHS/handoff.
2. **baseline**: pinned 495fb9cbd, OAHS/handoff (the previous campaign's current).
3. **existing**: that same pinned source, existing InsertSync.
4. **manual GEMM**: retained reference, for GEMM parity only.

The older `manifest.json` calls the baseline `current`; it is NOT this candidate.
Materialize it using `python3 tools/materialize_sources.py --out WORK/sources
--arms current --models`. Extract `sources/first-consumer.tar.gz` into a separate
candidate directory. The old broad campaign task is preserved as
`PRIOR_DEVICE_TASK.md`; this file defines the new scope.

Build with the device machine's appropriate explicit worker count and record
LLVM/MLIR, Python ABI, CANN, PTO headers and runtime pins. Generate synchronization
once from each original prepared input. Lower the synchronized result without
running InsertSync again. Verify single insertion and archive actual compiler
commands, including the effective optimized flags and both cc1 invocations.

## Order of work

1. Down projection (`prefill_fwd__0,1`): reuse the correctness-qualified harness,
   input configurations, seeds and derived numerical bound from the previous
   campaign. Measure all three arms before expanding scope.
2. Transfer: gate/up (`2,3`), KV (`4,5`), Q/out AIC (`7,8,9,10`) and LM head (`6`).
   Keep every ABI row, while deduplicating genuinely identical bodies/binaries
   for timing. Report precisely which rows each measured kernel covers.
3. GEMM: candidate, baseline, existing and the supplied manual reference. Check
   the committed trace oracle's 200/394/782 pairs, zero named fences and terminal
   ALL. Candidate and baseline PTO should match apart from trailing whitespace;
   compare emitted C++ and loaded binaries as well.
4. Preserve attention: regenerate/reconstruct all six attention modules (`44–49`)
   and compare with baseline. Their plans should be unchanged. Continue the
   separate coupled-attention qualification task independently; do not silently
   substitute these sources into it.

All required prepared inputs, older harness sources and manual GEMM are in the
archive. Reuse newer remote harnesses when their ABI/input/oracle contracts match;
record the exact harness revision and differences. Do not rewrite payload or
allocation to favor one synchronization arm.

## Correctness before timing

Gate each arm/configuration before timing. Use three fixed seeds plus the
previous queued multi-slot repeat with a single final synchronize and validation
of every slot. Preserve guard bytes, untouched-output coverage and nonfinite
checks. Archive worst error over the same derived bound; do not loosen tolerances.
Keep per-case inputs, golden outputs and binaries isolated to prevent cross-case
contamination. Input and executed payload/effect/context hashes must match arms.

Compute each input/oracle once per configuration and seed and reuse it across
arms. Use the optimized cache-friendly reference already established by the
remote campaign. Correctness stays outside latency windows. Size jobs below the
actual broker limit, preserve progress after each case and avoid duplicate chains.

## Measurement and mechanism

Use the prior six balanced rotated rounds and 180 measured samples per arm, with
warm-up and controlled device state. Report median, IQR, per-round medians and
candidate/baseline/existing ratios. Do not infer wins from small-kernel scatter.

Start profiling down; extend to a representative second projection and GEMM.
Record available MTE1, MTE2, MAC, FIX/scalar and core counters, with exact units,
core aggregation and measurement windows. The local hypothesis is removal of
later unrelated load completion before early B extraction. Compare that static
witness with measured latency and concurrency.

The CANN build previously exposed task-level counters rather than instruction
swimlanes. Do not promise a timeline it cannot produce. Report ResourceConflictRatio
and MemoryL0 if available. Sum-of-pipe-active/core-time is an aggregate overlap
indicator only when windows/categories align; core-minus-MAC is not stall time,
and a ratio near one alone does not prove no concurrency.

Local one-tile examples: down 359→343 pairs, gate/up single phase and KV/Q/out
418→398, LM bounded tile 393→363. All local fences are unchanged and no payload
ordering is added on 37 tested paths. These are structural expectations, not
performance claims. Complete counts and configurations are in
`first-consumer-evidence/family-ordering.json`.

## Deliverables

Archive source/toolchain/harness manifests, original and synchronized PTO, emitted
C++, every binary actually loaded with its hash, correctness rows, raw samples,
profiles, analysis scripts and failures. Verify all checksums after round-trip
extraction. Clearly distinguish measured results, static ordering evidence and
remaining hypotheses. Report down promptly, then the family and GEMM result.
