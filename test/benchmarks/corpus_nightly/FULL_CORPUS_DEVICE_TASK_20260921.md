# Full PyPTO and pypto-lib synchronization timing sweep

> Historical task for the completed `7c48f4ab3` sweep. Preserve this frozen
> compiler pin when reproducing that campaign; this is not a request to time
> the newer experimental working tree. Results and limitations are recorded in
> [the sweep follow-up](../../../docs/designs/oahs-sweep-followup.md).

## Objective and pinned compiler

Compare existing InsertSync with OAHS handoff across the full runnable **PyPTO
and pypto-lib** kernel/model corpus, including cases from earlier campaigns.
This is a new measurement campaign using the protocol below.

- Repository: https://github.com/tonibohnlein/PTOAS
- Branch: `codex/oahs-clean-m1`
- **Pinned compiler commit: `7c48f4ab35ae118d115f6ce85f0a4f3d5b6492e6`**
- Commit title: `Preserve early readiness across conditional consumers`
- Compare `algorithm=existing` and `algorithm=handoff` from that same commit.
- Use default OAHS options. Do not include the uncommitted first-write/rearming
  experiment, which still fails KDA validation.

Checkout the repository directly. No large source archive is needed. Verify the
exact commit and clean checkout before building; do not silently follow a moving
branch head. Any necessary harness adaptation must be recorded separately.

## 1. Freeze the experiment and inventory coverage

Record exact compiler, PyPTO, pypto-lib, framework/runtime, PTO-ISA and toolchain
revisions. Record target architecture, build flags and synchronization options.

Inventory the full runnable corpus in **both PyPTO and pypto-lib**, covering
kernels and models, with shapes/configurations and cases from earlier campaigns.
Do not silently restrict the sweep to the earlier 88-module host corpus or to
pypto-lib alone. List unsupported, failed, duplicate and skipped cases explicitly,
including missing authentic harnesses. Separate planned coverage from executed
coverage. Identify the measured KDA projection and other previously significant
regressions so they can be compared using their original configurations.

For each comparison, use the same prepared payload IR, inputs, shapes, layouts,
launch configuration and compiler flags. Only synchronization selection may
differ. Hash prepared IR, generated code and loaded binaries. Preserve payload
identity evidence. Complete compilation before submitting timing jobs; never
rebuild binaries underneath queued jobs.

## 2. Correctness gates before timing

Check both arms against the same reference with identical inputs and documented
tolerances. Include representative seeds and relevant tail, empty, conditional
and reuse cases. Record runtime errors, timeouts and guard-region failures where
supported.

A failing arm is excluded from performance comparisons; retain its evidence and
report the case as incorrect rather than assigning it a timing advantage.
Previously passing cases still require correctness checks at this pinned revision.

For coupled AIC/AIV kernels, use an authentic qualified coupled harness.
**Do not time isolated halves as a substitute.** Do not silently change queue
layouts, peer protocols, runtime contracts or payload schedules to make a case
runnable. If runtime/compiler contracts disagree, report the exact mismatch and
keep that case blocked pending a separately qualified harness.

## 3. Timing protocol: 100 runs, first 50 discarded

For **each case, arm and measured block**:

1. Prepare inputs, compile, load and allocate outside the timed region.
2. Execute **100 runs**.
3. Discard runs **1–50 as warmup**.
4. Retain **all 50 measurements from runs 51–100**.

A run is one complete intended invocation. Preserve raw measurements for all
100 runs and mark warmups explicitly. Do not discard outliers selectively or
select the fastest subset. If an interrupted/invalid block must be repeated,
retain its status and explanation; do not silently combine fragments.

Timing requirements:

- Prefer validated device-event timing of the complete intended invocation.
- State exactly what the timer includes and its units. Keep host
  launch-plus-synchronize measurements separate from device latency.
- For models, distinguish end-to-end model latency from individual kernel
  latency. Summed kernel times are not measured model latency.
- Ensure completion before reading the result; do not measure asynchronous
  submission alone. For multiple streams/engines, cover all required work.
- Reset mutable state consistently outside the timed region. Each arm must
  execute equivalent work on equivalent input state, including warmups.
- Establish timer resolution and report it. Do not claim differences below its
  useful precision. Any batching needed for resolution must be explicit and
  identically applied; report batch size and both batch/per-invocation units.
- Run correctness checking outside the timed interval.

## 4. Parallel scheduling across devices

Use **all available, unoccupied devices, up to eight**. Check occupancy and
acquire the normal device lock before dispatch. Do not assume a device is free
from an old note or card number.

Use one timing worker per available device, with no competing timing jobs on
the same card. Distribute independent cases across devices. Keep both arms of
**every matched comparison on the same device**; never form a ratio from arms
run on different devices.

Balance arm order across cases. Recheck substantial wins, regressions and
suspicious results with a second matched block in reversed arm order. Each block
again uses 100 runs with its first 50 discarded. Use spare devices for complete
matched replications, recording per-device results separately before summarizing
reproducibility. Avoid unbounded extra repetitions when the result remains noisy.

## 5. Qualify the measurement method first

Before the full sweep, time an **identical binary under two labels** on every
participating device using the same protocol. Choose representative durations
and execution modes, including short kernels where timing overhead matters.

Use these controls to expose label/order bias, drift and inadequate timing
resolution. If controls show material apparent differences, investigate before
interpreting similarly sized arm differences. A large sample count does not
repair systematic measurement bias.

Record device model/ID, occupancy, timing API and available clock/thermal
information. Preserve raw controls alongside the campaign data.

## 6. Report per-case results and aggregate coverage

Provide the complete table, including exclusions and failures:

| Case / shape | Device | Correctness | Existing median | OAHS median | OAHS / existing | Existing IQR | OAHS IQR | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |

- Lower ratios favor OAHS. Label all latency units.
- Calculate median and IQR from the 50 retained runs in each block.
- Show replication results separately, including sign disagreements.
- IQR overlap is descriptive, not a proof of equality or difference.
- Classify results as reproducible improvement, reproducible regression,
  unresolved, incorrect, unsupported or blocked. State the evidence supporting
  the classification rather than treating every numerical difference as real.
- Summarize planned/executed/correct/timed coverage for each repository and
  workload family. Report a geometric mean over valid paired case ratios with
  explicit inclusion rules and weighting; do not treat device replicas as
  independent extra corpus cases.
- Keep model-level results separate from kernel aggregates so many small kernels
  do not obscure end-to-end behavior.
- Do not attribute latency changes to fence counts or endpoint placement without
  a matched control isolating that effect. Static ordering and command counts
  are diagnostics, not a latency measurement.

## 7. Deliverables and reproducibility

Deliver:

1. Markdown report containing the complete performance/coverage table.
2. Machine-readable case manifest and summary CSV/JSON.
3. Raw samples including case, shape, arm, device, block, run index, warmup flag,
   timing method, units, binary hash and relevant input identity.
4. Correctness logs, reference/tolerance definitions and failures.
5. Exact commands, source/toolchain pins, build records and loaded-binary hashes.
6. Prepared inputs and generated artifacts for changed, failing and
   performance-significant cases, or immutable retrieval references for large
   artifacts. Retain enough evidence to reproduce every reported comparison.
7. Archive with a detached SHA-256 checksum; verify extraction and manifest
   integrity before handing it back.

Avoid packaging an entire checkout or build tree when pinned repository access
suffices. Keep immutable original campaign artifacts distinct from corrections;
mark any erratum clearly.

Finish by listing any remaining jobs, releasing device locks, and stating which
cases remain blocked. Do not report the campaign as complete while work is
silently queued or missing from the coverage inventory.
