# ProtocolSync hardware-grounded continuation

## Scope and evidence boundary

This milestone follows `27d2c2c4e` and includes the bounded, diagnostic-only
compositional-memory follow-up. It starts the approved five-step
continuation with target-contract corrections and a bounded recovery of the
hardware-graph concrete scoreboard. It does not enable ordinary loop repair,
cross-region repair, new L1/ACC protocols, or new event-ID reuse.

Compiler tests establish agreement with the declared model. No new CA-model or
device campaign, native corpus census, or performance measurement is claimed.

## 1. Target-contract corrections

- A named pipe barrier establishes completion only on its own pipeline. It
  cannot discharge an unrelated cross-pipe hazard simply because that consumer
  appears later in the IR. A barrier without a later same-pipe operation is
  still a valid fixed operation, but receives neither cross-pipe nor function-
  exit completion credit. Terminal drain obligations remain separate.
- Consecutive MTE2 loads are not intrinsically ordered when their destination
  regions overlap. The lane diagnostic now requests a named-pipe barrier.
  InsertSync's exemption is not imported as a hardware guarantee.

Primary evidence:

- [A2/A3 PipeBarrier contract](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0271.html): synchronization within the named pipeline.
- [DataCopy constraints](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0103.html): overlapping copy destinations require MTE2 or MTE3 synchronization.
- [Static-tensor synchronization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html): forward readiness and reverse buffer-reuse handoffs are distinct.

The frozen C.6/C.7 target-discharge and candidate-grouping aggregates predate
this correction. Preserve them as historical evidence; do not relabel those
numbers as results from the corrected target. Storage overlap/projection facts
do not depend on this target policy, but grouped transition identities may
change when a previously residual pair enters a completion cut.

## 2. Independent concrete scoreboard

`ConcreteLocalScoreboard.cpp` recovers the per-resource interpretation from
hardware-graph revision `19674e06f`, rather than its selection architecture:

1. Retain issued accesses with independent physical intervals, including
   outstanding readers and superseded writes.
2. A set snapshots its source pipeline's issued accesses and learned completion
   facts into a token identified by direction and event ID.
3. A wait consumes that token and transfers its facts only to the destination.
4. A named barrier adds its own issued accesses to its own completion knowledge.
5. Before a conflicting access, require knowledge of every relevant prior
   RAW/WAR/WAW access, returning the first uncovered access-ID pair on failure.

The implementation does not read selected completion edges, planner candidates,
atoms, timeline admission, or sparse requirement chains. It shares schedule
extraction and operation-local conservative range recovery; those are explicit
remaining common trust boundaries. An independent byte-set oracle already
tests the latter on the supported footprint subset.

This is an additional rejection gate, not a replacement completeness proof.
It checks ordinary non-recurring vector UB effects inside a region-free block.
Blocks containing structured regions, recurring or macro phases, unknown
footprints, and nonlocal effects retain the existing verification gates.
The scoreboard alone does not certify a complete function or general event-ID
reuse. Allocation/reservation checks and strict recurring verification remain
mandatory. No body `PIPE_ALL` completion credit is added.

The reduced tests cover same-pipe and cross-pipe completion, overlapping loads,
WAR reuse, early sets, late waits, duplicate sets, missing actions, unconsumed
tokens, and transitive event payload forwarding. The empty-intermediate-lane
relay is tested only as a scoreboard property: current F extraction still
requires physical endpoints for each direct event pair.

For every negative hazard case, the test also constructs an intentionally
overstrong all-forward completion graph. The graph-based local coverage check
accepts that fabricated supply while the concrete scoreboard rejects the actual
program and returns an access-pair witness. This pins the independent trust
boundary rather than merely comparing two graph traversals.

The existing `protocol_sync_fixed_supply.pto` now requires an explicit V/MTE3
handoff after the producer; a V barrier plus an unrelated MTE2/MTE3 event cannot
publish that producer's UB result. The historical shared-MTE2-barrier diagnostic
fixture now agrees with `canonical_sync_shared_pipe_barrier.pto` at `19674e06f`.

### Review corrections

Independent algorithm and compiler reviews identified two publication blockers:

- Named barriers had acquired unqualified exit-completion credit. That credit
  is removed. Full-verifier tests reject `tload; barrier MTE2; return` without
  a terminal drain and accept the restored drain. Native A2/A3 direct repair
  must retain or insert `PIPE_ALL` at exit. Scoreboard-only success does not
  certify exit completion.
- Expanded may-definition snapshots could exhaust their budget and abort
  production interpretation. Expansion is now opt-in, with explicit
  `NotRequested`, `Complete`, and `LimitExceeded` status. Production retains
  the sparse predecessor/outstanding-access transfer; cross-region diagnostics
  do not run implicitly. On expansion exhaustion, complete production
  requirements are preserved and unsupported boundaries remain protected.
  A 1,600-write regression crosses the old snapshot limit, and a zero-entry
  branch budget checks that partial diagnostic output cannot escape.

This correction bounds the optional experiment, not all compiler resource use.
Expanded histories still have quadratic worst-case size inside that budget;
persistent generation state remains future work, not a claim of this patch.

## 3. Ordinary loops and choices: next semantic checkpoint

Continue [structured repair](ptoas-protocol-sync-structured-repair.md) with
explicit dynamic phase instances. Share canonical obligations across entry,
body, backedge, exit and zero-trip paths; retain outstanding readers. Extend
the concrete scoreboard with the corresponding inductive state/token invariant
in the same patch as recurring admission. A bounded trace is only a test oracle.
Then add feasible-path joins and balanced participation. Do not relax strict E
or erase rejected-timeline protection merely to improve admission counts.

## 4. Reduced GEMM differential fixtures: planned

Use the official [A2/A3 GEMM kernel](https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp)
as a source of small lifecycles: MTE2/MTE1 L1 panel readiness and final-extraction
release; MTE1/M L0 operand readiness and reuse; M/FIX accumulator writeback.
Preserve slot identities, multiple consumers, stage periods, prime and drain.
Compare concrete obligation coverage, not operation-name placement matches.
Require zero/one/odd/even trips, partial overlap and missing-release mutations.
L1 and accumulator examples remain diagnostic until their target domains are
qualified; their presence is not permission to enable those protocols.

## 5. Optional ownership optimizations: postponed

Recover candidate structures and regression fixtures from covering-performance
revision `5eab65e9`: bundled L0 operands, hierarchical consumer groups, stable
and alternating L1 prefetch, outer-loop carries, and composite ownership.
Reconstruct them as indivisible certificates over canonical requirements.
Do not import its broad barrier effects, body-wide fallback candidates, or
unqualified target rules. Direct baseline admission must not depend on whether
these optimizations match or allocate.

## Validation

Targeted build: `PTOASCompiler`, `pto-test-opt`, and the six ProtocolSync unit
drivers, using the existing LLVM 19.1.7 / Python 3.12.13 CMake tree and
`--parallel 2`. No LLVM rebuild or full PTOAS target build was performed.

```text
cmake --build build --parallel 2 --target PTOASCompiler pto-test-opt \
  pto-protocol-sync-one-shot-test pto-protocol-sync-ready-release-test \
  pto-protocol-sync-direct-repair-test pto-protocol-sync-mixed-test \
  pto-protocol-sync-local-memory-test pto-protocol-sync-scoreboard-test

PATH="$PWD/.venv/bin:$PATH" .venv/bin/python "$LLVM_BUILD_DIR/bin/llvm-lit" \
  -v -j 2 build/test/lit --filter protocol_sync \
  -o build/protocol-sync-hardware-grounding-lit-results.json

.venv/bin/python .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py \
  --repo . --base HEAD
```

`LLVM_BUILD_DIR` denotes the validated LLVM build selected by this checkout.
The test command must put the workspace venv on `PATH`: the CLI wrapper uses
`env python3`. An initial run without that PATH selected an incompatible system
Python and failed before compiler execution; no rebuild was needed to fix it.

Pre-review results: all 41 focused ProtocolSync lit tests pass (1,899 total discovered,
1,858 excluded). This includes 36 initial scoreboard cases across A2/A3 and the
existing 243-program byte/hazard oracle and 162-row / 324-path branch oracle.
The changed-code checker reports 14 code files, zero errors and zero warnings.
`git diff --check` passes. Formatter: clang-format 21.1.8 with repository style.
After the final named-condition cleanup, the targeted rebuild and the three
scoreboard/fixed-supply/local-reuse lit tests also pass; the final smoke JSON is
`build/protocol-sync-hardware-grounding-final-smoke.json`.

Post-review validation (2026-09-05): the same targeted build succeeds. The four
focused regression tests pass in 6.48 seconds, followed by all 41 ProtocolSync
lit tests in 40.24 seconds, both with two workers. These include the expanded
40-case A2/A3 scoreboard, missing-drain native repair, 243 linear oracle rows,
162 branch rows / 324 paths, and the 1,600-write and branch-budget regressions.
Both independent reviewers accept the corrected scope; their static reviews
were followed by this successful build/test gate. The checker remains at 14
code files, zero errors and zero warnings, and `git diff --check` passes.

The focused and suite commands above use these additional arguments/results:

```text
--filter 'protocol_sync_(scoreboard_unit|local_memory_unit|fixed_supply|local_reuse)'
-o build/protocol-sync-review-regressions.json

--filter protocol_sync
-o build/protocol-sync-review-lit-results.json
```

The earlier 41-test run alone did not exercise the missing-drain or analysis-
limit regressions; the post-review results supersede it for this source state.

The JSON result is a local build artifact, not a frozen corpus evidence archive.
The full host suite, corpus rebaseline, device/CA tests, and performance runs
were deliberately not repeated. Required C++17 bounds, ownership, braces and
explicit-error rules were reviewed; release hardening and target-device/static
analyzers are outside this host-only checkpoint. The changed-code check is a
prefilter, not proof of full compliance or hardware correctness.
