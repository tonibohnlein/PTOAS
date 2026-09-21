# Short device task: joint first/final reader placement in down_proj

Measure the default handoff change that preserves B0's first-consumer receipt
and releases A0 after its final extraction, before the unrelated B0 wait.
The general qualifier also advances other admitted child releases. Compare
whole compiler outputs; this is not an A0-only endpoint mutation.

## Source and host gates

Use Git checkouts, not another compiler archive:

- Baseline: `2a130aefe` (resolve and record its full commit).
- Candidate: the commit containing this task and
  `docs/designs/oahs-joint-reader-prefix.md` (record full commit and clean tree).

Build both with the established LLVM/CANN/PTO-ISA versions. Keep the runtime,
optimization flags, launch configuration and original payload identical.
Regenerate `test/lit/pto/oahs_projection_down.pto` with each checkout's
`pto-oahs-selected-test --construct INPUT`. Require construction AND emitted
reconstruction success. Lower those selected plans directly; do not insert
synchronization a second time. Record prepared-input, selected-PTO, C++, build
and actual loaded-library hashes. Verify the real compiler expansion with
`-v` and the effective `-O2` setting; exit status of an unsupported `-###`
probe is not evidence.

Run from the candidate checkout:

```sh
python3 test/oahs/check_joint_reader_trace.py candidate.pto \
  --baseline baseline.pto --output joint-ordering.json
```

Require 18 passing traces, 73,449 local conflict checks per arm, 214 removed
complete payload relations, zero additions and identical executed SET/WAIT
and fence populations. Confirm the final predicate is `256 - iv <= 128`,
B0's first-consumer guard remains, and the A0 publication precedes the next
B0 reuse wait. Host evidence is described in the implementation document.
Stop and diagnose changed pins/results rather than silently measuring another
plan. The emitted guard/instruction cost is part of the candidate being tested.

## Harness and correctness

Reuse the already qualified MAT/down_proj harness from the completed MAT
campaign. Preserve its exact ABI, input/output sizes, golden computation and
guard checks; do not substitute an isolated synthetic loop. Input manifests
must match the actual runtime files before each run. Require no read errors,
no nonfinite outputs, intact guard bands and output/input preservation checks.

Run both arms with seeds 7 and 23 on block 0, block 19 and the established
four-slot pipelined configuration. Timing starts only after both arms pass.
Record numerical differences and tolerances, not just exit status.

## Timing: small matched experiment

Use any of the eight available devices after checking occupancy and locks;
leave other jobs intact. Distribute independent correctness cases across free
devices. Keep both timing arms in one matched block on the SAME device. Other
available devices may run a complete matched replication; never compare
arms timed on different devices as one ratio.

Use the same timing executable/ABI, block 0 and actual kernel timing scope
for both arms. Prefer ACL device events. Run **two rotated rounds**:

1. baseline, candidate;
2. candidate, baseline.

For each arm in each round use **10 warmup launches and 10 measured single
launches**: 20 samples per arm total. Log actual invocation counts. Disable
old defaults such as 100 samples or hidden 1,500-launch batches. Validate the
timing inputs before and outputs after timing. Exclude an arm if correctness
fails. A different profiling executable or cold correctness launch cannot
supply the timing result.

If the difference is unclear, report it as unresolved. At most one additional
complete matched block on another available device is useful; report its ratio
separately. No profiling, full-family sweep or model campaign is required.

## Return

Provide baseline/candidate median, IQR, per-round medians, ratio and correctness
in one table. Include raw samples, actual warmup/launch counts, device/job IDs,
commands, code/library hashes and a verified report/archive/checksum. Distinguish
compiler work from device latency. A static ordering improvement is established;
whether its guard cost pays off on this device remains the experiment.
