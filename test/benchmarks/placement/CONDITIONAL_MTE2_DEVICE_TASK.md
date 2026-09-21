# Quick device task: isolate the conditional MTE2 overwrite failure

**Status: completed.** The one-barrier mutation repairs the tested existing-pass
failure; OAHS passes. See [audited results](../../../docs/designs/oahs-conditional-mte2-isolation.md).
The instructions below are retained for reproduction; do not redispatch by default.


**Correctness only. No compiler rebuild, broad kernel sweep, timing or profiling.**
Test whether adding one MTE2 barrier repairs the existing InsertSync arm of the
archived `deferred_ack` microkernel. This is a diagnostic plan mutation, not a
compiler fix or a new production candidate.

## Inputs already on the remote machine

Use `/opt/pypto/oahs-placement-experiments-device-campaign.tar.gz`:

```text
SHA-256 5b07c6bb795eacee98fee49520838961a609747da753e7015b9eb78fcd4269a9
```

Verify that hash and the extracted `SHA256SUMS`. The root directory is
`oahs-placement-experiments-device-campaign/`. No files from the local workstation
are needed. Use these archived files:

- `host/t4/deferred_ack/existing/plan.pto` and `kernel.cpp`
- `host/t4/deferred_ack/default/plan.pto` and `kernel.cpp`
- `binaries/deferred_ack_existing/` and `binaries/deferred_ack_default/`
- `harness/wrap_deferred_ack.cpp`, `main_deferred_ack.cpp`, `micro_common.h`
- `tools/build_device_arm.py` (the corrected `-v` version)

Keep originals immutable. Work in a fresh disk-backed directory. Reuse the
campaign's qualified lowerer, interpreter, CANN setup and build configuration;
recover their locations from the prior work/build records. Do not execute the
archived shell scripts blindly: some contain old scratch-directory paths.

## Three arms

1. **existing:** archived failing arm, unchanged.
2. **existing_plus_mte2:** copy its synchronized PTO and insert exactly one
   `pto.barrier <PIPE_MTE2>` inside the `scf.if`, immediately before the tload
   which conditionally overwrites x.
3. **oahs_default:** archived passing control, unchanged. The deferred-ack option
   is not needed as a fourth arm for this question.

The mutation is:

```mlir
scf.if %17 {
  pto.barrier <PIPE_MTE2>    // the only added instruction
  pto.tload ins(%1 : !pto.partition_tensor_view<32x256xf32>) outs(%12 : !pto.tile_buf<vec, 32x256xf32>)
}
```

Assert that the conditional load site matches exactly once and that the original
existing body has no barrier there. Save the exact one-line diff. Keep every
payload, footprint, guard and other synchronization endpoint unchanged.

Lower the edited, already synchronized plan once, without another InsertSync
pass. Reuse the configured Python interpreter (the campaign used 3.10). Build
only the new arm with the unchanged wrapper/main and A3 flags. Require an actual
device cc1 expansion whose last optimization flag is `-O2`. Verify hashes of
all libraries actually loaded and their generated C++/build records. Preserve
the harness's `base+GUARD` pointers and its exact FP32 oracle.

## Small run budget

Choose an available device through the broker; never queue on an occupied device
or rebuild a pending job's inputs. Run the complete three-arm comparison on the
same device. Other devices remain available for independent work; do not create
replicas just to fill them.

For **each of the three arms**:

```text
./run pipelined 1 1
./run pipelined 2 1
./run pipelined 3 1
./run pipelined 1 0
```

Set library search paths to that arm's directory, as in the original campaign.
This is **12 harness runs total / 48 kernel launches**: each run enqueues four
slots and performs one final synchronize. The first three runs exercise the
failing true path on three seeds; the last is the false-path control. There is
no timing stage, so no timing warmup or measured repetition loop is needed.

Capture the complete output and exit status of every run. Preserve exact input
hashes across arms for each seed, sentinel/guard checks, all four slot outputs,
input immutability and finite-value checks. Do not loosen the oracle. Do not
replace real failing runs with a passing retry.

## Verdict and deliverable

- Existing fails / patched existing passes / OAHS passes: the one-barrier repair
  supports the WAW-order explanation. It is not yet a compiler implementation
  fix or a proof for other kernels.
- Existing fails and patched existing fails: the barrier alone is insufficient;
  retain the diagnostics and stop this task.
- Existing no longer fails: report failure not reproduced under this run set;
  do not silently launch a large repetition campaign.
- A control fails: investigate the harness/environment before attributing the
  patched result to synchronization.

Return a short per-arm/seed/active PASS/FAIL table, first mismatches, device/job
IDs, exact plan diff, generated C++, build/loaded-binary hashes and all raw logs.
Package these small artifacts with checksums. No latency number is requested.

Also recover, without rerunning them, the prior dedicated 12-run probe output,
row22 correctness/output-hash log, full 264-row host matrix and the status of
no-motion GEMM device timing: these were absent from the returned archive.
Missing historical logs must not delay this small correctness experiment.
