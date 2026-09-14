# OAHS terminal retirement checkpoint — 2026-09-14

This is compiler/host evidence on `7584d9dc5` plus uncommitted changes, not a
clean-revision or device result. The default remains `existing`. The developed
planner is `composition` with precision enabled.

## Why GEMM had seven extra pairs

The 61/61 candidate emitted an exit return acknowledgment for each of its seven
selected storage lifetimes. Those pairs conveyed consumption of the final
release token back to its publisher for the next invocation. Native emission
already required an explicit final ALL drain, but open-protocol reconstruction
did not use that drain between invocations. This was a missing retirement
premise, not a missing MMAD accumulator-ordering fact.

| Direction | Before | After | Removed purpose |
| --- | ---: | ---: | --- |
| MTE2 → MTE1 | 11 | 7 | Four L1 exit acknowledgments |
| MTE1 → M | 18 | 16 | Two operand exit acknowledgments |
| M → FIX | 2 | 1 | One ACC exit acknowledgment |
| MTE1 → MTE2 | 10 | 10 | None |
| M → MTE1 | 18 | 18 | None |
| FIX → M | 2 | 2 | None |
| Total SET/WAIT pairs | 61 | 54 | Seven terminal return pairs |

The final release WAIT remains for every lifetime. Retirement never consumes a
live token. It takes effect only after the entire invocation, only when every
event is empty, and only under the independently checked final ALL premise.
It preserves GM write/visibility history. Internal lifetime exits still need
explicit return acknowledgments. Authored synchronization after a section's
ALL receives no whole-function retirement credit; explicit acknowledgments can
still establish that authored plan.

The qualified native GEMM now has **54 SET, 54 WAIT, zero named body barriers,
and one terminal ALL**. It selects seven lifetimes. The independently imported
54-pair prototype has the same directional population. Five recorded scenarios
(empty grid, one/two/three panels, distributed tiles) have zero differing
acquired producer-prefix observations at corresponding payload operations.
These finite observations are not a universal performance-equivalence proof.

The original manual source's recorded 53-pair plan is a historical reference.
The prototype documents a readiness-key split needed for its strict reuse
proof; its 54-pair construction is the independently checkable comparison used
here. No claim of device speedup or native qualification of the original manual
numbering follows from matching the prototype.

## Hardware and input premises

The barrier-free result uses `a2a3-mmad-acc-v1`, the recorded scalar ABI bounds
and multiples, and the qualified disjoint-argument alias contract. Ownership is
`none`. The MMAD fact discharges only matching M→M ACC updates. Operand release,
M→FIX readiness, FIX→M reuse, output-store ordering and event consumption remain
explicit obligations checked from the emitted commands.

No MTE3→MTE2 GM publication capability is enabled. Its separate
[device qualification task](gm-publication-device-task.md) remains OPEN.

Under the explicit UnitFlag ownership profile, removing ACC obligations now
keeps remaining operand lifetimes in residual construction. Previously that
removal switched them back to pair deletion. The paired fixture with ownership
credit uses seven handoffs plus one named barrier, versus eight handoffs
without credit; required operand-release/GM-completion mutations still fail.

## Verification and artifacts

Artifacts are in the sibling workspace directory `oahs-coverage-work/`:

* `composition-retirement-qualified/summary.json`: final native checks, original
  input/contract hashes, binary identities, qualified GEMM, independent
  re-import, five boundary comparisons, and qualified composition PTO/C++.
  The compiler pipeline checks its event population against the native result.
  GEMM mutations
  delete each of 108 event endpoints, remove/move the terminal drain, and move
  cleanup after the section drain: 111 negative controls.
* Core test: **2,465,872 assertions**, including absent/misplaced retirement,
  deleted lifetime endpoints, and overlapping versus disjoint GM across
  repeated invocations. Retirement cannot publish the overlapping GM write.
* `coverage-retirement-r13/summary.json`: 19 focused provenance/geometry cases.
  The six witness-classifier tests and nine UnitFlag native cases also pass.
* `campaign-retirement-final/summary.json`: frozen 363 inputs, 336 distinct
  physical replays, **253 admissions** (79 PTOAS, 7 PyPTO, 167 pypto-lib), no
  gains or losses against the fresh baseline. All 86 publication refusals remain.
* `benchmarks-retirement-r13/summary.json`: eight benchmarks, three timed samples
  and one warmup per arm, serial compiler invocations, static/executed events,
  scalar guards and synchronized C++ emission. All `existing` and `demands`
  samples compile. The separate `structured` arm refuses historical GEMM with
  “payload guard is outside the ordinal periodic/initialization fragment”, so
  the complete three-arm report is explicitly `incomplete-or-failed`.

The corpus and benchmark runs precede the final authored-verification-only
suffix correction. Their recorded binary hashes remain their identities; they
are not relabeled as executions of the final binary. The final native suite
tests that correction. It does not change the production constructor.

All eight composition/existing median compilation-time ratios are recorded as
telemetry, without a 2× gate: one buffer 1.029, two buffers 1.024, three buffers
1.029, four-use 1.001, softmax 1.026, QK 1.029, Q projection 1.041, GEMM 1.028.
This benchmark population uses the conservative hardware profile and original
inputs; its GEMM arm is distinct from the ABI-qualified barrier-free result.

Only changed InsertSync/core-test objects and their dependent local drivers and
compiler DSO were rebuilt, with at most two aggregate workers. Architect,
performance, and correctness reviews led to explicit retirement accounting and
the post-section cleanup/GM tests. Device execution was not performed.

## Remaining original-plan work

Coverage contracts and the publication refusals remain unresolved. Full typed
per-event obligation reports, broader guarded/deferred protocol participation,
per-family recovery from optional failures, indexed-slot access selections and
the shared checked ordinal provider remain unfinished. General command and
ordering quality, including softmax/QK, remains separate from the recovered
GEMM result. Replacement qualification and device performance remain open.
