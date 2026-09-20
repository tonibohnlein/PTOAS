# Scheduling addendum: use all eight remote devices

User instruction, 2026-09-20. Apply this to the active MAT campaign and the
queued placement/reference tasks. This changes scheduling only: keep existing
source snapshots, binaries, inputs, correctness gates and measurement arms.
Send this addendum alongside immutable archived task instructions; do not
rewrite an archive that a running campaign has already verified.

## Work queue

- Discover the eight allocated devices and maintain a worker/queue for each.
  Run independent eligible experiments concurrently across all eight. When a
  worker finishes, it takes the next ready item; do not wait for a global stage
  or one long LM reference job before scheduling unrelated ready work.
- A timing work item contains a kernel/configuration and **all compared arms**,
  with their balanced rotated rounds on the same device. Do not put baseline
  on one device and candidate on another and call the difference a speedup.
- Keep one timed or profiled job active per device. Different devices can time
  or profile concurrently. Check for interference from shared host submission,
  power or other resources; report any detected effect. Exclusivity is per
  device, not across the machine.
- If more capacity is available than distinct kernels, run complete matched
  comparison blocks on additional devices as replication. Keep results
  stratified by device, and compare within-device arm ratios before combining
  results. Do not pool raw latencies across devices as interchangeable samples.
- Correctness-gate each runnable arm on the device/configuration where it will
  be measured. Reuse verified inputs and CPU references across devices/arms;
  preserve output isolation and exact loaded-binary hashes.
- Give every job isolated input/output/profile directories. Record physical
  device ID, assignment, binary hashes, launch configuration and round order.
  Use broker-supported affinity/reservation so job-device assignments are real,
  with no two orchestrators independently submitting the same pending work.

## Suggested initial distribution

Continue already running jobs without discarding completed evidence. Fill the
remaining devices from this queue; assignments are illustrative, not static:

| Worker | Ready work |
| --- | --- |
| 1 | LM head, all four MAT arms |
| 2 | Gate/up, all four MAT arms |
| 3 | KV, all four MAT arms |
| 4 | Q/out, all four MAT arms |
| 5 | Shenggan parity/control comparison |
| 6 | Matched four-arm down profiling, with its timing control |
| 7 | Source-gap microkernel, then deferred-acknowledgment cases |
| 8 | Class-invariant correctness/timing, then ready RMSNorm/reference work |

New placement work becomes eligible after its own host/build/correctness gates.
If those artifacts are not ready, assign a complete replica or another ready
campaign item instead. Equal-coverage selection currently has no changed binary
and requires no device timing. Do not manufacture work just to occupy devices.

## Host work and reporting

Build each source/configuration once and share immutable artifacts. Compute
each numerical reference once per input/shape/seed, with a cache-friendly
implementation, then reuse it across arms, slots and devices. Set explicit
build/BLAS/reference worker counts appropriate for the remote host's CPU and
memory capacity; account for their aggregate use across all eight queues.

Report the eight-device queue and the current stage of each job, along with
build, reference, queue, correctness, timing and profiling durations. Explain
idle devices through actual prerequisites or unavailable allocation. Preserve
per-device results, then report agreement or disagreement across devices.
