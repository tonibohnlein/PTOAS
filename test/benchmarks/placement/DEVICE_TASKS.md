# Placement experiments: independent device tasks

Use the self-contained `oahs-placement-experiments-8afb` bundle. The candidate
is an **uncommitted source snapshot based on 8afb90f17**, identified by the
archive and file hashes in `manifest.json`. Do not substitute it into the
already dispatched MAT campaign. These tasks share sources but have independent
arms. Do not combine experimental flags unless a task explicitly asks for it.

## Common preparation and acceptance

1. Verify `SHA256SUMS`. Extract `sources/baseline.tar.gz` and
   `sources/candidate.tar.gz` into separate compiler trees. The baseline is exact
   8afb; the candidate includes these experiments. `sources/candidate.patch`
   plus baseline reconstructs the candidate, including new files. Tree manifests
   identify every source file. Build with the existing qualified LLVM/CANN setup;
   LLVM and CANN installations are environmental prerequisites, not bundled.
2. Candidate targets: `pto-oahs-selected-test`, `pto-test-opt`, `ptoas`.
   Run the portable CTest suites and native selected tests. The driver uses
   production import, construction, emission and reconstruction. Experimental
   switches are on this driver, not new ordinary `ptoas` CLI options.
3. Generate a plan with `pto-oahs-selected-test --construct INPUT FLAGS > plan.pto`.
   Existing comparator: `pto-test-opt --mlir-disable-threading
   --pto-insert-sync=algorithm=existing INPUT > plan.pto`.
   Lower synchronized plans exactly once with the configured Python/ptoas:
   `python3.12 BUILD/tools/ptoas/ptoas --pto-level=level3 --pto-arch=a3 plan.pto -o kernel.cpp`.
   Do not run another InsertSync pass. Verify payload/guard/footprint identity,
   and compare against bundled PTO, C++ and synchronization pins.
4. Device compile A3 at `-O2`, no later override. Record compiler flags, exact
   generated C++, build metadata and the hash of the binary actually loaded.
   Retain the explicit-event contract; do not enable automatic synchronization
   in a downstream compiler or change queue/unit-flag semantics.
5. Correctness-gate EVERY runnable arm before timing. Three seeds and a queued
   four-slot repeat, one final synchronize, validate every slot. Independently
   hash inputs and initialize output/guards with sentinels; check untouched,
   guard and nonfinite outputs. A failed or unsupported arm is a recorded
   failure, not a timing result. Reuse each seed's oracle across all arms.
6. Timing: six balanced rotated rounds, 180 samples per arm, same binary/launch
   shape/scalars/warm-up. Serialize timed/profiled use of the shared device.
   Report medians, IQRs, round medians and identical-binary controls. These tiny
   witnesses may be dominated by noise; report an unresolved difference as such.
   Use repeat sampling of unchanged launches, not rewritten kernel loops.
7. Profile only a reproducible timing difference, using the SAME timing binary,
   arguments and warm-up. Report profiler perturbation; aggregate overlap is
   not instruction-level attribution. Host qualification/replay/trial cost is
   separate from kernel latency. Respect the broker's measured time cap and
   keep resumable jobs queued without duplicating submissions.

Remote machine concurrency should follow its own resource capacity and shared
device policy. No workstation path is required. Recovery runtime and qualified
GEMM references are in `support/`; input capture/provenance is also supplied.

## Task 2 — admission robustness and trial cost

First run `oahs-selected_placement-test` (CTest name `oahs_selected_placement`).
Reproduce the exact-fit failure on baseline using `evidence/baseline-probe.cpp`
and the build command in `evidence/baseline-probe-command.txt`. Candidate must
reject the optional cohort, then construct normally. Its invalid-proposal test
must show no committed endpoint/reservation/version leakage. These are host
admission results; they need no invented device benchmark.

Repeat host construction with fresh processes for default, no-recurring-trials,
no-helper-trials and both-disabled on:

- `inputs/shenggan_step4.pto`
- `inputs/pypto_lib__prefill_fwd__0.pto`
- `inputs/pypto_lib__prefill_fwd__22.pto`
- `inputs/pypto_lib__prefill_fwd__44.pto`

Record preparation, qualification, staged proposal check, selected replay,
structured entry solves, recurring omission trials, final helper trials and
final certificate counters separately. Record wall time/RSS independently.
Default hardening must reproduce the pinned baseline plans. Hash-identical
loaded binaries need no duplicate performance campaign.

Device priority: post-RMSNorm row22, default versus no-helper-trials versus
existing, using the established exact ABI/oracle. Both-disabled is a separate
construction configuration; deduplicate its timing if the binary is identical.
Partial attention row44 trial-cost comparison is host-first. Device timing is
eligible only with authentic coupled scheduler state, peer protocol and the
already qualified harness. Do not invent standalone state or transpose this
plan onto the unrelated manual-attention kernel. Report a blocked runtime
contract precisely and deliver the host result.

## Task 3 — exact source gap; separate no-motion control

**3A, priority:** `inputs/micro/source_gap.pto`.
Arms: candidate default, candidate `--source-gaps`, existing InsertSync.
Local evidence: 7 -> 7 dynamic pairs, two removed finish-to-issue relations,
none added. The old MTE3 read of y stops gating the MTE2 overwrite of x.
Run `test/benchmarks/placement/check.py evidence/micro` from the candidate tree
against the package evidence directory, and rerun it against regenerated plans.
The linked five-operation and outward-publication tests must also pass.

**3B, separately labelled:** Shenggan default versus `--no-frontier-motion`.
Default retains 200/394/782 pairs; no-motion has 330/652/1296, zero named
barriers, one terminal ALL and identical checked payload ordering. Use
`tools/compare_no_motion_gemm.py --repo CANDIDATE_SOURCE
--default DEFAULT_PTO --candidate NO_MOTION_PTO`; it keeps memory, event,
rearming, ACC and forbidden-overlap checks. Do not overwrite the default
committed count oracle. This measures the overhead of the conservative grouping
control, not a claimed parallelism improvement. Use the established Shenggan
numerical harness and retain existing/manual only as qualified references.

## Task 4 — acknowledgment at the real republication deadline

`inputs/micro/deferred_ack.pto`; arms default, `--defer-acyclic-acks`, existing.
Run both `active=0` and `active=1`. Input x is initialized before the conditional
load, so the false arm is numerically defined. Local results: 5 -> 4 pairs,
three/two dependencies removed respectively, none added. The later-key-reuse
portable negative must pass; this is not permission to delete acknowledgments
from recurring or optional-acquisition protocols.

## Task 5 — class-invariant readiness with disjoint producer work

`inputs/micro/class_invariant.pto`; runnable arms `--class-invariant-inputs`
and existing InsertSync. Record default OAHS's expected rearming refusal; do
not time its empty failed output. This is initially **construction coverage
and device correctness**, with existing as a numerical/performance comparator.
It is not a before/after speedup over default OAHS.

The original leaf loop has lower=3, upper=9, step=2. MTE2 performs disjoint work
inside it, while the two inputs remain invariant. Verify one acquisition at the
qualified first consumer, retained genuine V/MTE2 fences and repeat-entry
behavior. Run native admission negatives for a regenerated input, guarded
consumer and empty/unknown bounds. Do not silently replace the input or weaken
the ordinary-core checker if existing's same-pipe assumptions differ; device
correctness is a separate mandatory gate.

## Microkernel ABI and numerical oracle (tasks 3A, 4, 5)

All are A3 vector kernels with one block and dense FP32 matrices of shape
32 x 256 (8,192 elements). Allocate disjoint GM input/output buffers and guards.
Original function signatures in the prepared PTO define pointer order:

| Kernel | Arguments | Exact expected outputs |
| --- | --- | --- |
| source_gap | a,b,o0,o1,o2,o3 | a,abs(b),abs(b),a |
| deferred_ack | a,b,o0,o1,o2,o3,active:i32 | abs(a),abs(active?a:b),b,active?a:b |
| class_invariant | a,b,o0,o1 | abs(a),abs(b) |

Use finite nonzero signed FP32 inputs with distinct a/b patterns and varied
seeds. Loads/stores/abs have exact expected results; no FP64 matrix reference
is needed. The local allocations occupy 163,840 bytes; preserve their addresses,
extents and operation sequence. Generate a minimal launch wrapper around the
supplied C++; qualify its vector launch mode and argument order. The wrapper
must not add device synchronization inside the kernel. Record wrapper source.

The local oracle checks declared local byte overlaps and ordinary consuming
event semantics. It does not claim GM numerical correctness or validate the
device compiler. Preserve that boundary in the report.

## Task 6 — equal coverage: host evidence first, no timing arm yet

Run default versus `--equal-coverage-binding` on row44. Expected AIV diagnostic:
two equal-coverage encounters, four probes, zero choices; output identical.
Retain default selection on Unknown. No device timing is warranted for this
unchanged plan. The next milestone is a real positive helper-free alternative
with exactly equal coverage, not merely equal cardinality. Deliver that host
witness and its ordering/cost comparison before proposing a new timing arm.

## Delivery

For each task archive source/config hashes, host logs, emitted PTO/C++, actual
loaded binaries, wrapper/oracle sources, input hashes, all correctness rows,
raw timing/profile samples, summaries and unsupported cases. Keep baseline,
candidate, existing and experiment labels exact. Round-trip extract and verify
the archive and report its SHA-256. Do not change compiler mechanisms in the
measurement task; report a minimal reproducer if a gate fails.
