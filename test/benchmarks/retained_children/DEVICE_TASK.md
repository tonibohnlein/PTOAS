# Targeted device task: retained inputs across reader children

Run the attached `oahs-retained-producer-cohort` bundle separately from the
already dispatched overnight corpus campaign. No compiler changes, experimental
flags, or substitution into that campaign. Local results are ordering evidence,
not a prediction of latency improvement. Budget: 45 minutes after environment
setup; return precise blockers and partial evidence rather than enlarging scope.

## Arms and inputs

- **baseline:** default handoff from the frozen retained-reader snapshot based
  on d6e9b5365, before multi-input producer support.
- **candidate:** default handoff from the bundled new source snapshot.

Both source archives, exact prepared inputs, generated plans, emitted C++, source
manifests, builder, wrapper templates, host harness, and local evidence are
included. Nothing requires access to the author's local workspace. LLVM/CANN,
PTO-ISA headers and the established A3 device environment are prerequisites.
The source archives include the ptodsl compiler-flag helper. These are uncommitted
snapshots: use hashes, not HEAD alone, to identify them.

| Case | Parent tiles | Iterations per reader child | FP32 elements per input/output | Exact output per element |
| --- | ---: | ---: | ---: | --- |
| retained_2_1 | 2 | 1 | 16,384 | abs(a) + a + abs(b) + b |
| retained_4_2 | 4 | 2 | 32,768 | abs(a) + 2a + abs(b) + 2b |

Each parent uses different contiguous GM input and output tiles. The kernel loads
X and Y, reads each in two separate children, and combines/stores the results.
Local addresses and payload/control order must remain fixed. One A3 vector block;
ABI is three pointers `(a, b, out)`. The compiled wrapper exports the case sizes,
so the harness does not infer them from a filename or accept mismatched CLI sizes.

## Host gates

1. Verify `SHA256SUMS` and `manifest.json`. Extract `sources/baseline.tar.gz` and
   `sources/candidate.tar.gz` separately. Verify source files/modes with
   `python tools/verify_package.py . --source-dir EXTRACTED --source-arm ARM`.
2. Build both snapshots using the remote machine's established compiler setup
   and an explicit worker count suitable for that host. Candidate targets are
   `pto-test-opt`, `pto-oahs-selected-test`, and `PTOASPythonPackage`. Run candidate
   portable suites and native diagnostics. Local workstation worker limits do
   not apply to this separate remote machine.
3. For each arm/input regenerate with `pto-oahs-selected-test --construct INPUT`.
   Require construction/reconstruction, byte-identical pinned PTO, and successful
   lowering with the configured Python and `BUILD/tools/ptoas/ptoas
   --pto-level=level3 --pto-arch=a3 PLAN -o kernel.cpp`. Do not synchronize again.
   Require C++ pin identity, or stop and explain the toolchain discrepancy.
4. Run candidate `test/benchmarks/retained_children/check.py BASELINE_PTO CANDIDATE_PTO`.
   Required full local payload ordering: smaller case 12 removed/0 added;
   larger 48 removed/0 added. Memory, matching and rearming checks must pass.
   Record genuine V barriers and actual executed pairs (10 -> 14; 20 -> 26).
5. Build the supplied `cases/CASE/ARM/wrapper.cpp` and `harness/main.cpp` using
   `tools/build_device_arm.py --kind vector --arch a3 --cpp CPP --wrapper WRAPPER
   --main harness/main.cpp --out FRESH_BUILD`, with the needed `--python-path`
   entries for the built compiler and its `ptodsl` source package. The builder
   requires a real `-v` cc1 expansion with `-O2` last and records loaded-library
   hashes. The new harness has not been compiled against CANN locally; device
   build and wrapper qualification are mandatory, not preclaimed results.

## Device correctness and modest timing

Schedule the two matched case blocks on available devices without interrupting
other jobs. Each block keeps both arms on one physical device; remaining device
capacity stays available for the large corpus campaign. Honor device locks and
record physical-to-process device mapping (the harness uses logical device 0).

Run **both arms** with `run correct 7` and `run correct 23`. Each command queues
four independent slots before one synchronization. The bounded integer FP32
inputs vary across parent tiles and slots; compare their printed hashes across
arms. Every output element, guard, input preservation, and nonfinite check must
pass before timing either arm. No slow host matrix oracle is needed.

Run two timing rounds in rotated order: baseline/candidate, then
candidate/baseline, using `run time 7`. Each invocation first checks the actual
timing data, then performs **10 warmups and 10 measured single launches** with
ACL device events, and checks the final output again. Total: 20 samples per arm,
20 warmups across its two fresh processes, plus the explicitly logged correctness
launches. No hidden inner launch batches or automatic repetition escalation.
Exclude all timing if that invocation's final check fails.

If a difference appears material, one additional same-device matched block with
the baseline binary under two labels may assess scatter. Do not claim small
speedups from a noisy microkernel. No profiling is required for this task.
The comparison changes the whole selected protocol, including event population;
it does not isolate placement from event/fence instruction overhead.

## Return

Provide a compact per-case table with correctness, median/IQR, round medians,
candidate/baseline ratio and an explicit resolved/unresolved conclusion. Archive
raw outputs, source/PTO/C++/wrapper/loaded-library hashes, build flags, device and
job IDs, commands, ordering reports, failures, and checksums. State whether either
case is blocked. Keep host qualification/replay costs separate from device time.
Do not rerun the 88 unchanged corpus plans for device timing or claim a model
speedup from these synthetic witnesses. Deliver the report/archive/checksum when
finished without waiting for another instruction.
