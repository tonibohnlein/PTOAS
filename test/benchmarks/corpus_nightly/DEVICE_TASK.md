# Overnight pypto-lib performance corpus

## Objective and time budget

Compare **default OAHS handoff versus existing InsertSync** across a broad set
of real pypto-lib kernels and runnable model entry points. Maximize distinct
qualified workloads, not repetition count. Work autonomously for up to **8 hours
from task start**, reserving the last 20 minutes for the report/archive. Finish
earlier if the eligible queue is exhausted. Deliver partial coverage with precise
blockers; do not spend the night repairing one model integration.

This is a new, two-arm campaign. No source-gap/deferred-ack/class-invariant flags,
no trial ablations, no no-motion variant, and no compiler-mechanism changes.
The source is an explicitly hashed working-tree snapshot based on d6e9b5365,
including the bounded retained-reader-child change. It is not unchanged d6e9.
Its 88 archived A3 handoff plans match that committed baseline byte-for-byte.
Performance on those inputs measures current default OAHS, not a new speedup from
the retained-reader change. Never substitute an older four-arm campaign candidate.

## Package and preparation

The accompanying archive is self-contained for project sources, prepared inputs,
plan pins and reusable harness sources. LLVM/MLIR, CANN, PTO-ISA, Python build
packages and device drivers remain environment prerequisites. No workstation path
is required. Start in a new disk-backed campaign directory.

1. Verify outer SHA-256, extract, then run `python3 tools/verify_package.py .`.
   Read `manifest.json`, `inventory.csv`, and `capture/cases.json`.
2. Extract `sources/compiler.tar.gz`; verify it with
   `python3 tools/verify_package.py . --source-dir /path/to/extracted/compiler`.
   This verifier checks the shipped list schema, hashes and modes explicitly.
3. Build one compiler snapshot using the remote machine's compatible LLVM/MLIR
   and configured Python. Reuse a build only after verifying its source identity.
   Targets: `pto-test-opt`, `pto-oahs-selected-test`, `PTOASPythonPackage`.
   `ptoas` is the generated Python entry point, not necessarily a Ninja target.
   Reuse environmental CMake configuration, record it and all tool versions.
   Run the portable OAHS suites and native diagnostic executable once.
4. `sources/pypto.tar.gz`, `pypto-lib.tar.gz`, `runtime.tar.gz` contain the pinned
   framework, library/models and runtime. Pins are in the manifest. The runtime
   archive supplies the pypto runtime submodule; preserve that relationship.
   These archives have no enclosing root: extract each into its own directory,
   with runtime under `pypto/runtime`.
   Use their compatible PTO-ISA dependency; do not silently upgrade source pins.
5. Reuse the supplied MAT harness sources where applicable. Their old timing
   defaults are deliberately **not** the campaign configuration: see timing below.
   `tools/build_device_arm.py` uses the corrected real `-v`/cc1 `-O2` verification.
   Build only once per unique generated C++/ABI/options tuple. It needs the
   configured `ptodsl._runtime.native_build` module and CANN paths.

For captured inputs, construct and lower both arms with:

```text
pto-oahs-selected-test --construct INPUT > handoff.pto
pto-test-opt --mlir-disable-threading --pto-insert-sync=algorithm=existing INPUT -o existing.pto
ptoas --pto-level=level3 --pto-arch=a3 handoff.pto -o handoff.cpp
ptoas --pto-level=level3 --pto-arch=a3 existing.pto -o existing.cpp
```

**Do not run synchronization insertion again when lowering either plan.**
`tools/prepare_plans.py` runs this paired preparation with timeouts, hashes,
per-function reconstruction results and exact handoff pin checks. Run `--help`.
Use separate output directories for parallel subsets; ready cases can enter the
device queue before every case has compiled. Capture logs for refusals too.

## Coverage and priority

`inventory.csv` contains **96 archived modules: 63 pypto-lib A3, 24 PyPTO A3,
1 Shenggan control, and 8 A5 research inputs**. These are module counts, not 96
independent benchmarks. Model-generated rows, aliases, and coupled functions must
remain identified. The existing library source includes broader entry points for
additional discovery. The 88 A3 plan pins are host evidence, not 88 device gates.

| Priority | Workload | Scope |
| --- | --- | --- |
| 1 | Established controls | down, gate/up, KV, Q/out AIC, LM head; post-RMSNorm, out AIV, x_gamma, QKV, RMSNorm test; one Shenggan control |
| 2 | Other pypto-lib kernels | GEMM/matmul, GEMM+eltwise, layer norm, RMSNorm, softmax, RoPE, top-k, scalar-add; use original golden runners where available |
| 2 | Remaining Qwen prefill functions | partial merge seed/update, attention finalization, final RMSNorm, reciprocal, SiLU, casts/seeds, RoPE/cache, embedding and last-token extraction |
| 3 | Runnable model entry points | Qwen prefill and decode, and other A3-compatible model/layer harnesses discovered in the pinned tree; report complete-program measurements separately |
| 4 | Additional diversity | A3 PyPTO FFN activations, fused linear/bias, dynamic valid shapes, reductions and supported mixed-core examples |

Aim for **20 or more distinct correctness-qualified kernel families**, with one
representative original workload per family before adding shape variants. This is
an objective, not permission to fabricate coverage. Attempt every captured
pypto-lib A3 row at host level; account for every row as timed, duplicate,
correctness-failed, unsupported, harness-blocked or time-budget-deferred.
Add a second shape/branch for important boundaries only after family breadth.

Use the original model/golden invocation, allocations and ABI where practical.
Small runnable model configurations are welcome when supported by the original
source; publish exact shapes and changed invocation parameters. Do not rewrite
model computations, storage schedules, guards, target or queue protocol to force
admission. Model component timings are not end-to-end model timings, and their
sum is not a model latency or throughput estimate.

The Qwen captured attention halves require their **authentic coupled AIC/AIV
runtime**. Do not launch them independently or borrow the unrelated manual
attention harness. If the real model runner supplies that contract, test the
coupled program; otherwise record the missing contract and move on. Preserve
UnitFlag, queues, GM visibility, peers and collective participation.

The archived DeepSeek A5 inputs are inventory only on the qualified A3 OAHS
profile. Do not relabel them A3. Admit any additional DeepSeek case only if its
actual source/target/runtime passes the existing qualification; record refusals.
Draft scripts and distributed models needing unavailable peers are not overnight
integration projects. Cap first-time harness/ABI investigation at **20 minutes
per family**; save the minimal blocker and advance the queue.

## Model/frontend integration

PyPTO's normal `--enable-insert-sync` route defaults to existing. It does **not**
automatically select handoff. For complete model runs, provide an explicit,
recorded compiler adapter at the prepared-input stage: prepare once using the
same memory planner, construct the chosen arm for every generated module, then
lower without reinserting synchronization. The paired arms must have identical
prepared payload/control/allocations, runtime graph, launch parameters and inputs.
Archive the adapter and log each generated-module mapping and selected plan hash.

Do not silently fall back to existing for a failed handoff module. Such a model
is unsupported for the all-handoff comparison; individual qualified components
can still be reported separately. If a frontend cache is reused, separate its
arm namespaces and prove which binaries are loaded. Do not time an old cached
existing model under a handoff label.

## Correctness gate and reference cost

- Each timed arm/configuration must first pass the original independent golden
  check with identical inputs and tolerance across arms. Cross-arm equality alone
  is not sufficient. Keep legitimate model-specific tolerance; never relax it to
  rescue one arm. Existing InsertSync is a comparator, not a correctness oracle.
- Start with two deterministic nontrivial seeds. Include relevant true/false,
  tail and short-loop paths for newly covered conditional kernels. One queued
  repeated-entry check (e.g. four slots) suffices initially; do not multiply
  every seed by every launch configuration. Log the actual number of launches.
- Include guards, input immutability, sentinel/finite-output and real file-read
  checks where applicable. Verify all expected file sizes and hashes before runs;
  a missing input must fail rather than compare zeros with zeros.
- For atomic/in-place/cache-updating models, reset state or use isolated slots
  between repetitions. Keep resets outside the timed interval. Correctness must
  qualify the state and launch configuration actually timed.
- Compute/cache each golden once per source/shape/seed and share it across arms.
  Cache-friendly blocked/BLAS references are preferable to repeated scalar FP64
  loops; preserve the established tolerance and record any reference adaptation.
  Use the shared cache-directory argument supported by the supplied
  `oracle_cache.h` harnesses; inspect each main for its exact argument position. Estimate reference
  cost before starting LM head or full-model data generation. Stream/checkpoint
  large data; do not regenerate huge weights and golden outputs for every arm.
- If either arm fails, quarantine that paired row, preserve the failing log and
  continue other cases. No latency ratio for a failed comparator. The conditional
  `deferred_ack` existing failure is already isolated: do not rerun that microtest
  or patch existing here. Compiler fixes belong to a separate task.

## Timing: small budget, precise scope

Use **10 untimed warmup invocations, then 20 measured invocations per arm total**.
Split measurements into two rounds of ten: existing→handoff, then
handoff→existing. Warm each arm before its first measurements; keep process and
state handling matched. Additional warmup must be separately counted if needed.
No 100/180-sample campaigns, no hidden 50/1500-launch inner loops, no timing for
every correctness seed. Correctness launches are separately counted.

Use ACL device events for isolated kernels with one invocation per sample. For
short kernels, report unresolved timing if resolution/noise dominates. If modest
batching is necessary, **20 remains the total measured-invocation budget**:
for example, four event samples of five launches, with legal state handling.
Do not report four samples as twenty independent observations. Report raw sample
count, inner batch count and actual launches. No automatic repeat to obtain a win.

The supplied legacy `timing_*.cpp` files default to large batches; override their
arguments explicitly (B=10, PER=1 per round; W=10 before a fresh process) or make a clearly
recorded common harness adaptation to both arms. Some old harnesses use constant
timing data: qualify those exact data or adapt both arms to shared validated data.
They also hardcode logical device 0: use verified visibility mapping under the
broker or parameterize the device consistently; log physical and logical IDs.

For a complete model, measure wall time from invocation to full completion after
compilation/allocation/reference generation; label this **end-to-end invocation**.
Retain device-event timing separately if the multi-stream runtime provides a
valid whole-program bracket. Never mix host launch+sync, device kernel time,
profiled operator time and end-to-end model time in one latency column.

Record median, quartiles, round medians, both raw sample sets, handoff/existing
ratio and absolute microseconds saved/lost. Twenty samples are exploratory;
small apparent differences and conflicting round signs are unresolved. Do not
call overlapping IQRs proof of equality or disjoint IQRs universal significance.
Use one small identical-binary label control for timer calibration, not replicas
on every device. Deduplicate exact same-binary/launch cases and mark their ratio
as structurally identical (no separately measured speedup).

## Eight-device unattended scheduling

Discover which of the eight allocated devices are free. Use **all available
cards for independent ready cases**, one comparison pair on one device. Keep one
timing or profiling job per device; never compare arms across devices. Do not
occupy unrelated jobs or assume card0 is broken. Respect broker locks and record
actual assignment. Do not run a simultaneous full-model multi-device workload
on cards already assigned to kernel timing.

Use one persistent coordinator and a durable queue: prepared → built →
correctness-passed → timed → archived. On completion, each worker takes the next
ready family; no global wait for LM head, model integration or a profiler. Avoid
duplicate submissions and binaries changing underneath queued jobs. Save job IDs,
commands and state atomically after each item; resume completed work by hashes.

Set explicit host build/reference worker counts from this remote host's CPU and
memory capacity. Build once, share immutable artifacts, and bound aggregate host
parallelism so reference generation does not starve device submission. This
remote scheduling policy supersedes workstation-only resource notes in archived
project documentation. No full
reference computation is repeated merely because another device became free.

Broker jobs should be checkpointed chunks of at most **40 minutes**; do not trust
`--max-time 0` as unlimited. Bound device hangs and host stages, record timeouts,
and continue unrelated cases. Keep the next stage queued before a worker exits.
Report progress about hourly and at the first meaningful family result; finish
without waiting for a reply. Never keep a card busy solely to fill utilization.

Optional profiles: only after breadth, at most three largest stable regressions
or unexplained wins. Use the same binary/shape/launch/state as timing, report
profiling overhead separately, and do not claim per-instruction timelines from
aggregate counters. Profiles must not delay the final archive.

## Required output

Deliver `oahs-corpus-nightly-REPORT.md`, `oahs-corpus-nightly.tar.gz` and its detached
SHA-256 under `/opt/pypto/` (or the host's durable equivalent). Verify extraction
and every internal checksum before declaring completion. Include:

1. **Kernel table:** source/model, family, exact shape/configuration, covered
   manifest rows, device, correctness, existing/handoff medians and quartiles,
   ratio, absolute delta, sample/launch counts, scope, binary hashes and status.
2. **Separate model table:** full entry point, model configuration, runtime graph
   identity, all generated kernel arm mappings, correctness, steady invocation
   latency and precise timing scope. Empty with explicit blockers is honest.
3. Coverage denominators: source modules, unique functions, unique workload/shape
   configurations, deduplicated binaries, successful constructions, expected
   refusals, unexpected failures, correctness passes/failures and paired timings.
   Never reuse an earlier campaign's 264/88 counts as new evidence.
4. Raw samples, input/golden hashes, plans/C++, compiler and loaded binary hashes,
   build configs/cc1 logs, all correctness and timeout logs, durable queue/job
   records, harness/adapter diffs, and independent reference definitions.
5. Compile wall time/RSS and OAHS qualification/replay/trial counters separately
   from device latency. List the worst regressions and useful next structural
   investigations; synchronization counts alone do not explain performance.
6. Top-level provenance manifest for every artifact. Reconcile all report counts
   from machine-readable records. Distinguish unmeasured aliases and host-only
   cases from device-qualified results. Include blockers and partial coverage.

No automatic compiler patches, PRs, performance claims from failed arms, or
another full sweep to chase tiny differences. The deliverable is a broad,
trustworthy measurement table and concrete next targets.
