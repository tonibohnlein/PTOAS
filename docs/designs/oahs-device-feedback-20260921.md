# Device feedback, 2026-09-21

This is a feedback-only milestone, not a compiler change. Measurements remain
attached to their frozen revisions; none measures the subsequently rebased
`bec7dfe37` checkout. All device work used TaskQueue.

## Controlled KDA projection

Compiler: `2cc458cbe1e5aff6f77ffc6fea62b13686f38a94`, existing InsertSync versus
default OAHS. Complete original-runtime projection entry, physical device 0,
job `task_20260921_121700_199949532682`. Numerical checks pass for two seeds and
two consecutive launches per arm (eight launches).

| Arm | Median us | Q1–Q3 us | Ratio to existing |
| --- | ---: | --- | ---: |
| Existing | 241.810 | 239.244–242.615 | 1.0000 |
| OAHS | 318.160 | 317.089–319.450 | 1.3157 |
| Identical existing control | 238.9995 | 237.070–240.094 | 0.9884 |

Ten initial warmups per label, then three alternating matched rounds with four
individual samples per label per round: 74 launches including correctness,
warmups and controls. Timing scope is device-runtime wall time for the complete
entry, excluding host submission but including runtime overhead. Per-round OAHS
ratios are 1.3227, 1.3103 and 1.3209. This supports a regression in this run,
not instruction-level attribution. Warmup warning checks pass, but are not a
proof of stationarity. Scheduler-window medians are 205.460 and 283.260 us.

[Raw records](device-feedback-20260921/kda-records.json),
[summary including small-kernel controls](device-feedback-20260921/method-summary.json),
and [static investigation](oahs-kda-pipeline-investigation.md) are committed.
The small split-kernel control remains unresolved: warmup checks fail in both
attempts. Its small differences are not synchronization speedups.

## Joint-reader down_proj

Baseline `2a130aefe613d93b9cdfaa47955c63510483b5c4`; candidate
`a0c1d761e7624b42a0f494853cc70eecaabe9c44`. Candidate tracked source is clean.
Baseline build includes a read-only local `--explain` diagnostic patch, retained
in the external archive; production construction is unchanged.

Both arms use input SHA-256
`8ce546e96c28c0235cebb0e0d604e0b03d0a0bb84afabab16bc7e872c2db6fcb`.
PTO-ISA `5a4f74cbf627d4aac2e0ce10d5e0d8b118343265`, CANN 9.0.0,
LLVM/MLIR 19.1.7 ABI0, GCC 15; identical warning-only native build adjustment
`-Wno-error=restrict`. Device compiler expansion confirms effective `-O2`.
No second synchronization pass is applied.

Host gates: 18 paths, 73,449 conflict checks per arm, 214 full payload-order
relations removed, zero added, unchanged executed event/fence counts. Both
construct and reconstruct. This short task did not rerun the full corpus.

Correctness: 24/24 launches pass, seeds 7/23, blocks 0/19 and four pipelined
slots (0, 7, 13, 19). Original M=128, K=4352, N=5120, 256-column blocks and
17 K chunks. Original double-precision reference bound is
`8 * FLT_EPSILON * sum(abs(a[k]*b[k]))`; worst error/bound is 0.06029.
Paired output hashes match; no nonfinite results, guard failures, input
mutations or writes outside selected columns. Residuals reset before each
timed invocation; goldens are reused.

| Physical device | Arm | Median us | Q1–Q3 us | Ratio to baseline |
| --- | --- | ---: | --- | ---: |
| 0 | Baseline | 50.260 | 45.235–54.550 | 1.0000 |
| 0 | Candidate | 45.660 | 42.820–52.100 | 0.9085 |
| 0 | Identical baseline control | 45.850 | 41.900–45.965 | 0.9123 |
| 1 | Baseline | 49.490 | 44.435–56.015 | 1.0000 |
| 1 | Candidate | 45.130 | 43.990–46.060 | 0.9119 |
| 1 | Identical baseline control | 42.650 | 33.965–45.150 | 0.8618 |

Two rotated rounds: baseline/candidate/control then control/candidate/baseline.
Each label gets ten warmups and ten measured single launches per round.
120 launches per card, 240 total plus 24 separate correctness launches.
No trimming or hidden batches. The control loads the exact baseline binary.
Single-launch ACL event intervals are susceptible to host enqueue delay:
**latency benefit is unresolved**, because identical-binary changes match or
exceed the apparent candidate improvement. Do not pool cards or add more
repetitions under the same biased method. Next: qualify a device-duration or
state-correct batching method with a passing identical-binary control.

| Purpose | TaskQueue job | Device |
| --- | --- | --- |
| Seed 7 correctness | `task_20260921_131517_21568454197` | 0 |
| Seed 23 correctness | `task_20260921_131520_21570036370` | 1 |
| First timing | `task_20260921_131607_215979015305` | 0 |
| Replication | `task_20260921_131724_216310827888` | 1 |

[Device 0 raw records](device-feedback-20260921/joint-device0.json) and
[device 1 raw records](device-feedback-20260921/joint-device1.json) retain all
samples and hashes. Only the server root prefix is replaced with
`<campaign-root>/`; commands are historical records, not standalone recipes.
Full sources, commands, binaries and correctness evidence remain in external
`oahs-joint-reader-20260921.tar.gz` (2,754,531 bytes). Its verified detached
SHA-256 is `9efd678ca7b71e6616747422d5d37f03cd2e259509a7ee38c5c23fb2090a9d96`.
The archive and dependencies are deliberately not committed.

## Coupled attention FIFO/relay qualification

Baseline `d1bf07ee55025d27bc66f508a3fa46727af7ffd0`; candidate
`21f95f9b759aa2446a514a402e15e4f12b201206`. Unchanged row48/row49 input hashes:

- row48: `2127cef1e8080fa8972ba4b952167d7c4e164265da592b66cdaaac144485faac`.
- row49: `7ca19d0b9d6d506c9bbe32c177390fe9afd00ec7ec3c50039354af4aa83cd7c2`.

Host regression passes: two split relays per AIC; 103→109 event pairs;
80 full payload-order relations removed, zero added per native-length entry.
Missing-support mutations fail, late forwarding broadens order, AIV companions
are unchanged. Twenty-three portable suites passed in that campaign. Five
selected controls have byte-identical plans/lowered C++; this is not a claim
of device binary identity or new numerical control measurements.

### Contract distinction

Prepared row48 initializes both peers from argument 7, direction mask 3,
8,192-byte slots, two slots, two local slots, flag base 0, no-split true and
split 0. AIC uses ACC push/MAT pop; AIV uses VEC pop/push. One IR base alone
does not prove aliasing; lowering establishes the physical representation.

Pinned PTO-ISA `0c112d61f41342bd0867ce1080c29f1590d72484`, `TPush.hpp`:
producer/consumer start at tileIndex=0, entryOffset=0; constructor lines
499–501 preserve those offsets. Address calculation is
`GM_SLOT_BUFFER + (tileIndex % SLOT_NUM) * SLOT_SIZE + entryOffset`
(plus the applicable sub-AIV term). C2V and V2C reuse physical slots 0/8192.
Working-framework PTO-ISA `3b4faf67aebb3e0d41be7952c56908b3adba7a8f`,
constructor lines 485–501, offsets the bidirectional cube consumer/vector
producer by `SlotNum * SlotSize = 16384`: separate directional rings.
This proves a representation difference, not scheduler incompatibility with
the older headers.

Isolated runtime `22385d2b0c08b8f697fe8feede488c87cf83ef84` does not build
against pinned headers: missing `WarmupSdmaControlPathForAiv` and
`kSdmaMaxChannelGroups` in SDMA sources. That is API incompatibility, not a
queue-progress result. Runtime `443dfa36aba8dbc34860d1f23d4c1cacbff8941a`
builds with the pinned contract. CANN 9.0.0 and `-O3` are shared between arms.
The isolated adapter keeps original payload wrappers, workspace addressing,
24 mixed blocks, synchronized start and one AIC/two AIV topology. Two
`TFILLPAD<InPlace>` calls become `TFILLPAD_INPLACE` with the same underlying
`MAP_INSTR_IMPL`; older orchestration type names are used. Queue calls and
prepared contracts are not changed. Inputs already contain upstream RoPE.

### Actual execution and blockers

| Arm/input | Job / physical device | Result |
| --- | --- | --- |
| Baseline row48 | `task_20260921_120217_198153622284` / 0 | Completes; 491,520/655,360 output mismatches, active NaNs |
| Same row48 binary replay | `task_20260921_120353_1983645725` / 0 | Same 75% mismatch; tensors preserved |
| Baseline row49 | `task_20260921_120221_19817542855` / 1 | Prelaunch `halMemCtl` error 42 in control-register discovery |
| Candidate both rows | Not executed | Baseline qualification gate failed |

Row48 CPU reference is finite; compiler/FIFO/harness attribution remains open.
Row49 older runtime passes logical IDs to HAL; later upstream `9a30e6dd`
introduces `acl_to_hal_device_id`. This mapping boundary is distinct from
deadlock or numerical failure. Neither candidate timing nor profiling is
qualified. No isolated-half measurements substitute for this task.

Next: isolate row48's first nonfinite output under the preserved peer protocol,
and qualify logical-to-HAL mapping independently for row49. Then complete the
three-seed repeated-launch baseline gate before candidate execution. Full
failure tensors, adapter and logs are preserved externally under
`oahs-relay-20260921/`; this feedback is not a self-contained failure reproducer.
