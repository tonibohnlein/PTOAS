# Device task: coupled attention FIFO-slot and relay validation

Status: prepared; this file does not confirm dispatch or a running device job.

Validate the default handoff change in `21f95f9b7` on the original single-block
attention inputs `pypto_lib__prefill_fwd__48` and `__49`. Determine whether its
narrower FIFO receipts preserve numerical correctness and queue progress, and
whether the additional overlap improves coupled execution time.

This task includes investigating the reported runtime mismatch and building
and qualifying an isolated coupled harness. The device agent may do that work
without requesting a separately supplied harness. Preserve the working runtime
and existing campaigns; use separate checkouts/builds for pinned dependencies.
Do not patch the compiler or change the original payload/queue protocol to make
this experiment run. This is separate from the manual-attention transcription,
the older full sweep and the newer down_proj joint-reader experiment.

## 1. Freeze the comparison and recover the original inputs

Use Git checkouts of `tonibohnlein/PTOAS`; no compiler-source archive is needed.

| Arm | Exact commit | Policy |
| --- | --- | --- |
| Baseline | `d1bf07ee55025d27bc66f508a3fa46727af7ffd0` | Default handoff before static FIFO slots / split relay |
| Candidate | `21f95f9b759aa2446a514a402e15e4f12b201206` | Default handoff with the complete slot/relay change |

Use the same LLVM, CANN, compiler flags and runtime for both arms. Pin PTO-ISA
to `0c112d61f41342bd0867ce1080c29f1590d72484`, the lowering contract used by
the importer. Record dependency revisions, compiler versions and clean source
states. Later compiler commits are not substitutions for these arms.

Locate the original prepared modules in the preserved corpus/bundles on the
device machine. Match their contents, not merely their row numbers:

| Input | Prepared PTO SHA-256 |
| --- | --- |
| `pypto_lib__prefill_fwd__48` | `2127cef1e8080fa8972ba4b952167d7c4e164265da592b66cdaaac144485faac` |
| `pypto_lib__prefill_fwd__49` | `7ca19d0b9d6d506c9bbe32c177390fe9afd00ec7ec3c50039354af4aa83cd7c2` |

Recover the matching original AIC/AIV functions, argument mapping, shapes,
workspace layout, scalar values and launch configuration from those artifacts
and their generating source. Rows28–31 (`qkpv_plan`) and unrelated manual
attention are not substitutes. If an exact input is missing, return that
specific missing artifact and its expected hash rather than regenerating an
unmatched workload under the same label.

## 2. Reproduce the host result before device execution

Build both native drivers and construct each input with
`pto-oahs-selected-test --construct INPUT`. Require successful construction and
emitted reconstruction for every function. Preserve the original payloads,
control, queue operations and peer obligations. The AIV companion must remain
identical between arms. Lower the selected plans directly, without running
synchronization insertion a second time.

From the candidate checkout, run the committed host regression with an
already-built native Makefiles build:

```sh
python3 test/benchmarks/attention_relay/run.py \
  --build PATH_TO_CANDIDATE_NATIVE_BUILD \
  --out PATH_TO_RESULTS/host \
  PATH_TO_ROW48/prepared.pto PATH_TO_ROW49/prepared.pto
```

It checks the input and original-graph pins, constructs the candidate through
production code, and compares against captured pre-change words. Also retain
the independently generated baseline PTO from the baseline checkout; diagnose
any mismatch with the documented baseline rather than silently accepting new
pins. Each input must pass all six finite cases and missing-support/late-relay
mutations. The key native-length expectations are:

| Trace | Executed pairs baseline→candidate | Relations removed | Added |
| --- | ---: | ---: | ---: |
| Empty | 2→2 | 0 | 0 |
| One outer entry, inner length 3 | 103→109 | 80 | 0 |
| Two entries, inner lengths 3,3 | 204→216 | 160 | 0 |

Shorter/varying inner lengths in the host suite are graph stress tests; do not
change the native fixed inner length of three to reproduce them on hardware.
The early bank release was already narrow. The changed dependency is the FIFO
receipt: PV0 needs the preceding local slot-0 FIX writer QK2, while the later
slot-1 receipt retains QK3 support. Removing the latter is not an optimization.

## 3. Resolve the runtime contract and qualify a coupled harness

The reported obstacle is concrete: the working framework runtime uses separate
C2V/V2C rings, while these pinned inputs/lowering use shared GM slots. Begin
with the authentic coupled Qwen runtime recovered in the full sweep, but use
it only if its exact ABI and queue semantics match these inputs. The older
two-layer Qwen run alone is not qualification for this experiment.

Inspect the original initialization, pinned `TPush`/`TPop` implementation and
both peers. Document, with source locations and resolved runtime arguments:

- The shared GM allocation and per-core/entry offsets. Both directions in
  these inputs refer to the same root: two slots of 8192 bytes, entry offset
  zero within the queue. QK sends occupy 8192 bytes; probability receives read
  4096 bytes. Preserve actual allocation size/alignment and all other aliases.
- Independent send/receive cursor initialization and advancement on each
  participating operation; the original prologue/body/epilogue message order.
- Peer/core pairing, flags, backpressure, queue initialization and reset between
  invocations, and the supported collective/GM-visibility contract.
- Exact unsplit A3 overloads, disabled UnitFlag modes, launch geometry and
  output ownership for the coupled AIC/AIV computation.

If the working runtime differs, locate the matching historical adapter or
implement a small isolated launcher/adapter reproducing that original contract.
Test the queue address/message mapping for both peers before loading the
candidate. Do not make the model fit by allocating separate directional rings,
aliasing two independently managed rings, inserting extra device barriers, or
changing queue depths, payload order or consumption cadence. Such changes
would test a different contract. Do not downgrade the working installation.

Use an initial 90-minute budget for locating/building/qualifying the adapter.
If the required contract cannot be established, return a minimal mismatch
reproduction, exact source/ABI differences and the smallest remaining work.
Report host-qualified/device-blocked and retain the useful artifacts. A local
event checker is not proof of peer progress, and timing isolated AIC/AIV halves
does not satisfy this task.

## 4. Correctness gate

For each exact input, run both arms coupled using seeds 7 and 23. Use the
original supported scalar/shape configuration that exercises native-length
steady state and its prologue/epilogue. Add one legitimate boundary configuration
from the original dispatch domain if available; state precisely what it covers.
Do not invent an empty/short case that the original ABI cannot express.

Use an independent numerical reference with documented dtype-appropriate
tolerances. Check finite outputs, complete output coverage, guard bands, input
preservation where required, and queue completion. Missing input files must fail.
Apply a bounded timeout to every launch and retain timeout/progress diagnostics.
Verify that repeated invocations reset state according to the original runtime.
Both arms must pass before timing; a failing arm contributes no latency claim.

Record the actual loaded-library hashes, including both AIC/AIV pieces, prepared
input, selected PTO, generated C++, effective build command and runtime source.
Use the real compiler's `-v` expansion to verify effective `-O2`; an unsupported
`-###` probe returning zero is not evidence. Both arms use identical numerical
inputs, launch geometry, optimization flags and harness code.

## 5. Small matched performance comparison

Use all available devices among the eight allocated cards for independent work,
after checking current occupancy and locks. Leave unrelated jobs intact. Keep
every baseline/candidate timing block on one device; never form a ratio across
devices. Independent row48/row49 cases can run on different cards. Choose and
record an explicit build-worker limit appropriate to the remote machine.

Measure the complete coupled attention invocation. Prefer device events that
actually span both participating halves and their completion. If the runtime
only exposes a launch-plus-synchronize or scheduler window, report that scope
explicitly; do not call it kernel-only time or add separately timed half-kernels.
Reset mutable state outside the measured interval identically for both arms.

For each input, use two rotated rounds: baseline/candidate, then
candidate/baseline. Each arm gets **10 warmup launches and 10 measured single
launches per round: 20 measured samples per arm total**. Disable inherited
100-sample defaults and hidden large launch batches. Validate outputs after
timing and log actual invocation counts.

Report median, IQR, per-round medians and candidate/baseline ratio. For a small
or unstable difference, report unresolved. If useful, run at most one complete
matched replication on another free device, reporting its ratio separately.
A short identical-binary two-label control can check suspicious label effects;
record that both labels load the same hash. No broad sweep or automatic
hundreds-of-runs escalation. Profiling is optional follow-up only after a
repeatable difference, and must remain separate from unprofiled latency.

The experiment measures the combined slot-precision/relay change, including
its event-count and FIX-fence changes. It does not isolate placement alone or
prove the general relay heuristic preserves ordering on other inputs.

## 6. Deliverables and completion

Return one table with a row per input/device block:

| Input | Runtime contract | Correctness | Timing scope | Baseline median/IQR | Candidate median/IQR | Ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |

Include exact commits, host regression summaries, the harness contract and
source, input/plan/code/library hashes, commands, device/job IDs, numerical
errors/tolerances, warmup/sample counts and raw timing samples. Distinguish
prepared, host-qualified, device-correct, timed and blocked states.

Write `oahs-attention-relay-REPORT.md`, a focused artifact archive and its
detached SHA-256 under `/opt/pypto/` (or report the actual persistent directory).
Verify archive contents and checksum. Use Git references for unchanged source
trees; include the newly qualified adapter and scripts, not a huge compiler or
framework archive. Retain large runtime inputs remotely with hashes/paths and
reproduction instructions. Finish with a clear completion/blocker report and
confirm no experiment jobs remain running. Do not claim dispatch merely from
the existence of this handoff.
