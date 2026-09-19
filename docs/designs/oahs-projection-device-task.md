# Device task: projection overlap, separate GEMM releases, and AIV placement

## Objective and inputs

Qualify the candidate and measure whether retaining each operand bank's early
readiness and previous-use release repairs the Qwen projection regressions.
Check the review-hardened GEMM for continued manual parity. Then test the retained
AIV receive-placement mechanism with the complete coupled attention protocol.
Command counts alone are not the objective: measure latency and pipeline overlap.

The dispatch archive supplies `manifest.json`, original prepared inputs,
compiler/model/runtime source snapshots, capture metadata, reference harnesses,
manual GEMM, and host evidence. No file on the author's workstation is required.
The exact candidate is `compiler_pins.current` in the manifest: the commit
containing this task, also pinned in `DISPATCH.txt`. Never follow a moving branch
without checking that SHA. This task supersedes the older packaged device task.

Verify `SHA256SUMS`, then materialize sources into a fresh campaign directory:

```sh
python3 tools/materialize_sources.py --out WORK/sources \
  --arms current before gemm_reference --models
```

Use the device machine's matching LLVM/MLIR 19, CANN, PTO ISA headers and Python
ABI. Record all toolchain pins. Adapt harness plumbing and paths as needed;
preserve payloads, allocation, numerical contracts and compiler algorithms.
Continue independent cases when one is blocked and record the exact blocker.

## Arms and attribution

| Arm | Definition |
|---|---|
| current | Exact candidate SHA in the manifest, `algorithm=handoff` |
| before | `8cc0d58927c1b5acb032415ead339e2cc8d9fcab`, handoff |
| existing | Candidate compiler, `algorithm=existing` |
| gemm_reference | `16564fa8ae7282631f20ce112c10bab0cf69ed36`, Shenggan only |
| manual | Packaged original Shenggan manual plan, identical payload |

The candidate includes projection episode selection, removal of the broad
release merge, structured-source coverage, accounting, and AIV slot qualification.
Thus current versus before is a combined revision comparison, not a universal
single-variable experiment. Keep the mechanisms separate in conclusions.
The prior device campaign's `current` was d6c573127; do not silently relabel its
results as either new arm. Its plans/harnesses may be reused as documented inputs.

Use three freshly built main arms. Reuse qualified harness code from the recent
remote campaign where available, verifying its ABI and input hashes. Do not repeat
pre_guarded/guarded sweeps unless a new discrepancy needs them. Include manual and
gemm_reference for Shenggan. Historical timing is context, never a fresh sample.

## Priority and coverage

All numbered prefill IDs mean `pypto_lib__prefill_fwd__N` in the manifest.

1. **Projection regression:** down 0-1 first, then gate/up 2-3, KV 4-5,
   q/out AIC 7-10 and LM head 6. Exercise gate-only/both/up-only ownership,
   both KV output arms, and full original launch distributions. LM head's local
   bounded trace parameters are not actual device core IDs; run the full kernel.
2. **GEMM:** exact `shenggan_step4`, original 2048 x 4096 x 4096 problem,
   24-core launch and all 256 output tiles. Compare current/before/reference/manual
   and existing. Use committed overlap checks as well as numerical qualification.
3. **Changed vector plans and controls:** out AIV 20-21, x_gamma 42-43,
   rms_norm_test, and all qkv_proj variants. The latter remain unchanged locally
   and are a useful Cube control; high MTE1 workload alone did not predict a
   regression. Retain RMSNorm, TopK and the small GEMM as neutral controls.
4. **Composition cases:** post-RMSNorm 22-23, partial attention 44-47 and
   single-block attention 48-49. Test AIC and AIV together with authentic queue,
   scheduler and backpressure behavior. Do not benchmark the two halves as a
   substitute for their coupled launch. Include empty/full/tail participation
   and repeated invocations where the original runtime supports them.

Regenerate/reconstruct the entire 87-module captured A3 corpus and Shenggan on
host. Account for every manifest row. Reuse the prior campaign's qualification
of untouched families with explicit provenance; a full new device sweep or new
A5 harness campaign must not delay these priorities. A5 remains a separate target
and hardware contract. Merge/finalize and RoPE cases with unresolved scheduler or
numerical contracts remain explicitly blocked, not fabricated into runnable cases.

## Host gates and expected plans

Build Release `pto-test-opt`, `pto-oahs-selected-test`, the alias-overflow test,
and normal ptoas/Python targets. Run focused native regressions and standalone
OAHS suites. Use explicit parallelism suitable for the remote machine.

`tools/host_matrix.py` accepts a JSON config mapping current and before to
`driver`, `opt` and `ptoas` argv, all pointing into the remote builds:

```sh
python3 tools/host_matrix.py --config WORK/host-tools.json --out WORK/host \
  --arms existing before current
```

Lower already synchronized input with `--pto-level=level3 --pto-arch=a3`.
Do not rerun InsertSync during lowering. Verify single insertion in generated C++.

Representative current static SET and WAIT populations, each counted separately:

| Input | Before | Current |
|---|---:|---:|
| down 0 | 33 | 56 |
| gate/up 2 | 44 | 72 |
| KV 4 | 44 | 72 |
| q/out AIC 7,9 | 22 | 39 |
| LM head 6 | 22 | 39 |
| partial attention 44-47 | 119 | 119 |
| single-block attention 48-49 | 78 | 78 |
| post-RMSNorm 22-23 | 10 | 10 |

Projection graph sizes are unchanged; the increased event population serves
separate deadlines. BF16 matrix completion barriers remain under the existing
contract. Do not remove them to obtain favorable numbers.

Current Shenggan executes **200/394/782 pairs** on 1/2/4 tile traces, **zero
named barriers and one terminal ALL**. Before/reference use 182/360/716;
manual uses 167/329/653. The prior manual-parity result belongs to the 182-pair
arm, not automatically to current. Run the candidate's committed
`test/oahs/check_carried_slot_trace.py` and `check_projection_trace.py` on its
plans. The former additionally forbids the preceding B reader gating A refill.
Use each older revision's checker for its own plan; failure of a new quality
assertion on an old plan is not itself a correctness failure.

Partial attention retains 87 static named barriers plus two terminal ALL;
single-block retains 26 plus two. Partial AIV moves ten output-return acquisitions
after receives without deleting the tail-storage completion requirement.
Native FIFO qualification is restricted to the unsplit private two-slot contract.
Its local proof does not establish coupled peer progress or device correctness.

Record qualification time, graph sites, selected updates/replay, recurring
omission trials, helper-composition cold trials, final certificate and total
compile time separately. Partial AIV replay is known to grow substantially;
report it without confusing it with device latency or redoing expensive sweeps.

## Correctness before timing

Use the original mathematical contract and real launch/capture metadata. For
each timed arm require three deterministic seeds, all output elements checked,
guards/sentinels intact, no unexpected nonfinite results, plus four queued
launches into separate output slots with one final synchronize and all slots
validated. Record raw error, bound, worst error/bound and element populations.
Preserve the existing backward-error bound for Shenggan.

Reuse the identical per-seed host oracle across arms after verifying payload/input
identity. Make FP64 reference computation cache-friendly and use validated
optimized/multithreaded routines where appropriate; preserve the bound and log
oracle versions. Cache generated inputs instead of regenerating huge random
tensors per arm. Validate harness improvements against the existing reference.
Do not relax tolerances to rescue an arm or infer correctness from equal profiles.

The previous LM timings were provisional because timing preceded correctness
qualification. Verify the final status of those runs; discard failed/unqualified
timings. This campaign must enforce the gate before submitting timing jobs.
Approximate-exp partial-merge failures under an inherited, underived tolerance
remain unqualified unless a justified original-contract bound is established.
Do not invent plan/block-table values for scheduler-dependent attention or RoPE.

## Build provenance and measurement

Call `ptodsl._runtime.native_build._kernel_compile_flags(kind, 'a3')` for the
correct core kind. Preserve the full common device flags. Record ordered compile
argv and actual cc1 expansion: effective **-O2 with no later override** is
required in every compared arm. Freeze toolchains, wrappers and launch settings.
Hash original input, selected PTO, generated C++, and the .so actually loaded.
Verify exact original payload/effect/control identity; equal MAC time is not
an identity proof. Preserve queue/cross-core contracts separately.

After gates, collect 180 samples per qualified arm in six balanced rotated
rounds. Report median, IQR, round medians and ratios against both before and
existing. Use fresh runs of all compared arms; note device clock/concurrency.
Collect valid nonempty profiles for representative down, gate/up, KV, q/out,
LM, Shenggan and qkv control. Add coupled attention profiles if qualified.

For down first, produce paired MTE1/M/MTE2 timelines if available. Attribute
blocking waits and fences to the actual bank/deadline, and check whether next-bank
preparation overlaps current-bank compute. Report whether residual MAT readiness
still gates early extracts on unrelated later loads. This is the remaining known
limitation; local ordering improvement does not guarantee the entire regression
is repaired. Extend the timeline diagnosis to LM if its regression persists.

Active-time ratios are not stall attribution. Core time minus MAC active time
is not a measured stall total. Completion-to-issue relation subsets cannot see
every drain's performance cost. Event count and occupancy alone are insufficient.
The previously observed tiny-kernel scatter can exceed 19%; use identical-binary
controls and do not label noise as improvement.

Submit each remote job once, checkpoint per case/arm, and keep jobs below the
broker's observed one-hour hard cap with margin. Do not assume `--max-time 0`
disables it. Schedule the priority results before optional breadth; reuse
validated expensive oracles. Report blockers promptly and continue independent work.

## Deliverables

Return an incremental projection/GEMM summary as soon as those comparisons finish,
then a complete matrix of qualified/failed/blocked/unchanged cases. Include raw
correctness, samples, profiles, timeline attribution, compile metrics, exact pins,
commands, hashes and loaded-binary identities. Distinguish measured performance
from local-order evidence. State remaining regression and qualification gaps.

Package all reproduction inputs and harness changes into a return archive with
internal SHA256SUMS and an outer SHA-256. Round-trip extract and verify every
checksum and measured .so. Regenerate at least down and Shenggan and compare
plans to those actually measured. No author-workstation paths may be required.
