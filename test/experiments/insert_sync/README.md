# InsertSync native revision campaign

The compiler objective is memory/effect correctness while preserving pipeline
overlap. Compilation, independent local checking, event-lifetime checking, and
device performance are separate results. A smaller event count is not a
performance certificate.

## Frozen R1 comparison

The imported patch is commit `5cc1d946525f630879a57dba3fe97a848f09ce11`, based
on fork main `7e2ec3e29420e297dcf5b3c59ba4d90841464821`.

Use `compare_native.py` with the original frozen TSV manifest and input root.
Run `--arm main`, `--arm combined`, and `--arm staged` into different output
directories. Each arm records both PTO IR and C++ compilation, source hashes,
the **native library** hash, complete commands, diagnostics, and pre/post-pass
IR. Explicit-sync bypass detection examines the pre-pass IR, not source comments.
An absent analysis marker is not counted as successful automatic insertion.

Use `--compare MAIN COMBINED STAGED --output DIFFS` to retain a complete
per-kernel IR diff. Differences include placement and control changes, not only
counts. An IR difference alone establishes neither a race nor a speedup.

The first-per-static-seed population has **213 rows**, including repeated input
hashes. It remains distinct from the structured driver selection and the full
generated collection. Never remove pre-pass failures from its denominator.

Initial A3 native result: main, R1 combined, and R1 staged each compile **208/213**
to PTO IR and C++. All 208 show InsertSync analysis and allocation execution;
none is an explicit-sync bypass. The same five inputs fail before InsertSync
because `tci` requires an explicit temporary when level-3 skips memory planning.
The four supplied R1 regression fixtures pass their A2/A3 native/FileCheck and
C++ emission checks. Main passes all 131 existing tests whose RUN lines invoke
InsertSync. R1 combined and staged each pass 134 of 135 tests. Their sole
failure is the old `issue667_tload_overlap_no_mte2_barrier` assertion: its row
slices overlap in columns 64–127, so the restored WAW check correctly adds an
MTE2 barrier. The renamed regression requires that barrier; the separate
disjoint-view regression still passes without one.
The corrected test was then rerun successfully against the frozen R1 native
runtime in both modes. This is an expectation correction, not suppression of a
candidate failure.

Per-kernel diffs and the original compiler outputs are retained in the local
campaign artifact tree `insertsync-builds/campaign/213/r1-differences`, alongside
the `main`, `combined`, and `staged` arm records. Relative to main, combined
changes 106 emitted PTO files; staged changes are recorded per row. SSA naming
and printing differences must not be interpreted as additional ordering.

Native SHA-256 fingerprints for this measurement:

- Main: `0fa0a43290cb0a43f48755ded11fe9879323698b86ea8aa63cf127bb3844b7b3`
- R1: `cb97d6facd91aa1a60a2e284230a5e9a1b5016368a377ff0fa92e304c9de5107`

After building main, its staged Python/native runtime and results were frozen.
R1 was built incrementally in the same validation worktree at its exact commit,
reusing unchanged objects. This is not a comparison against the ProtocolSync
branch or against an old installed executable.

The lit parameter `-D insert_sync_staged=1` adds staged traversal to CLI
invocations. It uses a separate execution directory, allowing one serial run
per mode without temporary-file collisions. Two earlier runs shared a directory
and are invalid evidence; use only the subsequent isolated runs.

## Contract-aware dependency and checking work (subsequent patch)

The next patch introduces:

- `--insert-sync-gm-alias=may-alias|assume-disjoint-arguments` and the equivalent
  `pto.gm_alias` function attribute. Without either, different arguments may
  alias. An argument-disjoint promise is usable only after complete root tracing
  through supported forwarding. It is **not** a GM publication/cache guarantee.
- A physical slot partition check before unequal selectors can discard a hazard
  or select recurring event streams. Different mappings with equal capacity do
  not satisfy this condition.
- A translator coverage check: payload effects must appear in its read/write
  sets; missing helper/resource summaries are explicit errors. Descriptor
  mutation remains separate from payload completion. Conservative allocation
  bounds are not shrunk using dynamic valid-shape values.
- `--insert-sync-audit=off|report|strict` and the standalone
  `pto-test-opt --pto-audit-insert-sync` checker. Strict mode accepts only the
  explicitly supported **local** checker domain. Report mode records the verdict
  without turning unsupported verification into a pass.

The initial checker uses fresh emitted-IR effects and a completion/event
scoreboard, not the mutable InsertSync dependency graph. It checks ordinary
A2/A3 vector UB, static flags, exhaustive result-free choices and literal finite
loops within explicit budgets. Dynamic loops require an inductive extension;
cube/proxy/communication and GM publication remain unsupported by this checker.
Their exclusion is not evidence of correctness or failure. Conservative-footprint
overlap without supply is reported as uncovered, not as a device-race witness.

The supported-domain mutations remove waits, remove a signal from one branch,
and remove terminal completion. Further event-rearm and structured-loop gates
must be established before calling the full independent checker complete.

## Remaining acceptance work

Finish native validation of the contract/checker patch, then replace boundary
motion using actual corpus witnesses. Preserve publications after their required
source and acquisitions at executing consumers, with explicit bypass/token
handling. Finally harden deletion/allocation with completion and consumption-
before-rearm proofs. Additional serialization belongs only to attributed finite-
resource recovery. Staged traversal remains opt-in until its regression gate is
resolved. No device performance or full synchronization correctness is claimed
by the compilation campaign above.
