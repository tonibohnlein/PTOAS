# Common benchmark protocol

These are **reference-adoption and device-benchmark tasks**, not bundles of
already-generated candidate plans. Each task includes all pinned source trees,
the current compiler, and reusable harness examples. Construct the matched arms,
validate them, then measure them. No access to the originating workstation is
needed. No result from the earlier ordinary GEMM applies to these new kernels.

## Pins and contents

- PTOAS/OAHS and existing InsertSync: `495fb9cbda7649f15a7fc09d6b1d91ffb7737d54`.
- CATLASS: `2b85ed307b281baa76d663f11a9c9aa228d56652`.
- PTO-ISA: `c0d7148e95ef73bd12a73165fdce4b723a3b7e72`.
- Archives: `sources/ptoas.tar.gz`, `sources/catlass.tar.gz`, `sources/pto-isa.tar.gz`.
- `examples/benchmarks/manual_sync/` and `examples/oahs/` are reusable host/protocol and
  device harness examples from the ordinary-GEMM study. Their function names,
  layouts, dtypes, dimensions and event assumptions are **not** this task's ABI.
- `SOURCE_SURVEY.md` supplies navigation, not validation of these experiments.

Verify `SHA256SUMS` first. Extract into a disk-backed work directory. Use an A3
device matching the current projection campaign where possible; record actual
chip, CANN, driver and compiler versions. Confirm CATLASS's AtlasA2 template path
is supported on the selected device. Do not switch to Ascend950/A5 templates.

## Required comparisons

For each selected source configuration, retain:

1. **Original manual:** native upstream implementation, including its scheduler,
   data layouts, padding/packing, prefetch, instruction modes and final drains.
2. **Matched manual:** source-derived PTO carrying the same ordered payload and
   manual synchronization. This exposes transcription/lowering overhead.
3. **OAHS:** run `--pto-insert-sync=algorithm=handoff` once on that same PTO with
   only reconstructible local synchronization removed.
4. **Existing:** run `--pto-insert-sync=algorithm=existing` once on the identical
   unsynchronized PTO using the same compiler revision.

Archive the removal manifest. Preserve queue, cross-core, collective, visibility
and target-specific synchronization outside the pass's admitted contract. Do
not strip every synchronization instruction indiscriminately. Inspect implicit
scratch, DMA geometry, MMA accumulation modes, unit flags and output conversion.
An unsupported operation is an importer limitation, not permission to omit it.

CATLASS defaults use unit flags. Prefer retaining their exact admitted semantics
in every matched arm. If that cannot be represented, use a source-supported
explicit-event template instantiation as a **separately labelled** comparison
family, retaining the original tuned implementation as a reference. Never call
that alternate family an unchanged-original comparison. Do not silently disable
unit flags or K shuffling, change output dtype, broaden the ACC exemption, or
turn instruction completion into synchronous execution.

Instrument source control/template calls or otherwise establish an independent
trace correspondence. Compare payload opcode, issue sequence within each core,
physical effects, tile shape/layout, numerical parameters and original guards.
Retain cross-core/message identities separately. Use full invocation identity
where practical and report any bounded trace campaign's exact coverage.

The compiler is a frozen benchmark subject. Harnesses, source adapters and test
oracles are in scope. Do not repair its scheduling policy as part of measurement;
record refusal or a poor plan with the smallest discriminating example. Continue
other supported cases without waiting for confirmation. Keep complete-kernel,
alternate-template and reduced-mechanism results explicitly separate.

## Local/native and numerical gates

- Require parser/lowering, OAHS construction/reconstruction, and independent
  memory, event matching/rearming and final-drain checks for admitted mechanisms.
  Supply a deliberate missing-transfer negative and a safe extra-drain ordering
  negative. A checker must distinguish correctness from scheduling freedom.
- For each manual/automatic protocol disagreement, retain an exact witness.
  A successful finite device run does not prove an event-contract premise.
  Record empirical results and contract qualification separately.
- Use three fixed seeds, all output elements, sentinels/guards, finite-value
  checks, and four queued launches into separate output slots followed by one
  synchronization. Check every slot and unchanged inputs. Include declared tails
  and repeated-entry boundaries; do not reset event history inside a kernel.
- Establish the numerical tolerance before comparing arms. Include FP16 output
  rounding, padding/layout conversions, and attention's approximate exponential
  where applicable. Pairwise equality alone is insufficient. Retain upstream
  checks plus an independent reference; do not loosen a failed tolerance.
- Compute vectorized/blocked CPU references once per shape/seed and reuse them
  across arms and slots. Use an explicit host/BLAS worker count suited to the
  remote machine. Avoid oversubscribing concurrent campaigns and avoid scalar
  per-arm FP64 recomputation. Check the broker's real job-duration limit.

## Device measurements

Build matched arms with identical effective target flags and headers, normally
`-O2`; archive the actual cc1 commands and reject a later optimization override.
Disable downstream automatic synchronization for manual and already-synchronized
arms. Source Ascend C and PTO may require distinct frontends: record that
difference and use original-versus-matched-manual to expose its cost.

After all compared arms pass correctness, use 180 samples per arm in six
balanced rotated rounds. Warm up consistently. Use all eight allocated devices
for independent comparisons. Serialize timed/profiled jobs only on the same
device; each matched comparison must run all of its arms on that device.
Record device identity and retain per-device results rather than pooling raw
latencies. Follow the [scheduling addendum](../../../docs/designs/oahs-eight-device-scheduling.md).
Record kernel-only latency and separately end-to-end cost where wrappers launch
padding, warm-up, SDMA prefetch or helper kernels. Never include such work in
only one arm's timing scope.

Report median/IQR, per-round medians, ratios to matched manual and existing, and
an appropriate work metric. Small-kernel scatter is not a speedup. Capture valid
paired instruction/pipe timelines separately from timing; attribute blocking
acquisitions to the physical generation and first consumer they protect.
Count dynamic SET/WAIT, named barriers and terminal drains separately, but judge
pipeline overlap from actual waits, pipe idle gaps and profiles. A relation
subset does not predict latency or capture every cost of a same-pipe drain.

Record compiler preparation/qualification, selected updates/replay, final helper
trials/checks, wall time and peak RSS separately where available. Do not charge
test-only trace enumeration or oracle generation to compiler complexity.

## Deliverables

Return a concise comparison table, one explanation of the dominant overlap gap,
and a reproducible archive containing source/transcription diffs, manifests,
all compiler commands, prepared/selected PTO, emitted C++, input generators,
numerical results, raw timing samples and profiles. Verify hashes of binaries
actually loaded against build records, and internal checksums after round-trip
extraction. Include sufficient scripts to regenerate large oracle caches rather
than archiving multi-gigabyte caches unnecessarily.

For unavailable matched cases, report the precise unsupported contract and a
reproducer, alongside any measured original baseline. Do not present original
timing alone as an OAHS-versus-existing benchmark or a local projection as a
complete native kernel.
