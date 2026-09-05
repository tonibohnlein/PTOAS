# ProtocolSync completion-supply continuation

## Scope and implementation order

This continuation starts at `5fcb8cf279084587163c6f0c48306f6ab75275ea`,
after native prefix/ordinary-loop/suffix integration. The approved order is:

1. Close the independent execution oracle's function-exit gap.
2. Introduce explicit completion-supply summaries, reproducing straight-line
   interpretation without changing placement.
3. Deliver standalone choices and joins end to end: guarded memory state,
   participation, balanced repair, and independent path tests.
4. Compose choices with ordinary loops, keeping the complete serial alternative.
5. Add selective loop repair and its semantic verifier together.
6. Experiment with bounded relay/resynchronization replacements and
   distance-aware optimization.

The initial patch implements steps 1 and the bounded first slice of 2.
The subsequent [balanced structured continuation](ptoas-protocol-sync-balanced-structured-frontiers.md)
adds native bounded steps 3 and 4. Steps 5–6 remain future work. The supply
summary itself is still straight-line-only; it introduces no hardware claims
or placement changes. Existing straight-line and serial-loop alternatives
remain available; optional protocols still compete as complete alternatives.

## Execution-oracle exit semantics

The bounded asynchronous execution oracle now distinguishes three operations:

- A named barrier waits for preceding effects on its own pipe only.
- `PIPE_ALL` records every lane's preceding command prefix and the preceding
  phase occurrences. No lane can cross that boundary until all recorded commands
  have executed and all recorded phases have completed. Lanes first encountered
  after the boundary have an empty incoming prefix.
- Function return is checked after the preceding command prefixes have executed.
  Outstanding phase completion is a failure, not an implicit wait. Unconsumed
  event tokens remain failures; a global drain does not consume them.

This is a test interpretation of the existing terminal completion contract, not
authorization to emit body-wide barriers. The oracle can interpret internal
drains in adversarial test inputs; production placement policy remains unchanged.
It still uses fixed fixture byte ranges and shared operation extraction, not an
independent general-purpose PTO hardware model. Return is modeled after command
issue on all lanes, not as a separately modeled scalar dispatch pipeline.

The fifteen new cases cover empty return, pending load at return, correct global
drain, premature global drain, wrong-pipe drain, sufficient named drain,
work on a newly encountered lane after a drain, missing post-work drain,
unconsumed token despite a global drain, consumed event completion, and repeated
named drains in a loop. Four additional combinations use two disjoint active
pipes: a global drain or both named drains suffice, but either named drain
alone does not. Each runs for trip counts 0, 1, 2, 3, 4, 7, 8, and 11.
The prior cycle and boundary execution tests now use these stronger exit rules.

Passing the oracle without a `PIPE_ALL` is possible when other synchronization
already guarantees completion. That does not remove the production policy's
mandatory terminal drain. The oracle checks semantic retirement, whereas the
production canonical-shape checker also checks the declared emission policy.

## Explicit logical completion supply

`SyncCompletionSupply` is a read-only result over a frozen schedule and a
selected world. Each phase-before frontier records its program point, region,
physical core, receiving lane, and a set of completed source phase identities.
The set is not an issue counter or necessarily a contiguous source prefix.

The first supported domain is one unguarded, non-recurring block of ordinary
vector-core phases. Only explicit forward, same-iteration completion effects
enter the transfer. For a selected completion from A to B:

```text
completed-before(B) includes A
completed-before(B) includes completed-before(A)
```

There is no implicit propagation from lexical succession or shared pipe identity.
There are no self-completion facts. Visibility, exit completion, and protocol
selection records do not create additional edges. In particular this summary
cannot decide what a hardware signal publishes merely from its location.
That remains the selected-mechanism/concrete-extraction contract.

Input completion indices are retained. A backward witness query returns a
source-to-target path of those indices; no path means no represented guarantee.
The source and target contexts are recovered from the immutable schedule.
This is the first interface for supply queries, not yet a general region
transfer function, path-sensitive join certificate, or backward placement search.

The builder validates endpoints, forward order, control, iteration, and duplicate
edges. It does not certify the input mechanisms' hardware effects. Callers must
already have validated their logical selected effects or reconstructed the
actual synchronization. The independent concrete scoreboard does not consume
this summary, preserving a separate verification implementation.

The selected-world interpreter dual-runs the new answers against its established
completion graph for canonical local requirements in this subset. Disagreement
fails closed. Canonical requirements, selected candidates, event IDs, and
placement are not changed or deleted by this migration gate.

## Bounds and validation contract

The initial representation deliberately favors an auditable transfer over a new
compression heuristic. It stores per-phase bitsets, with a default limit of
1,048,576 phase-pair membership slots. The multiplication bound is checked by
division before allocation. Unsupported domains and exhausted budgets return
distinct statuses and no partial frontiers; the old interpreter remains in
charge. This limit is not an admission limit. Witnesses use stored direct edges,
not a quadratic predecessor table.

The independent logical oracle enumerates all 1,024 forward graphs on five
phases, computes Floyd closure, and compares all 25 source/target queries per
graph. Every supplied witness is checked against the input edge chain. Reversed
input edge order, empty supply, repeated pipes, invalid endpoints, self/reverse
edges, unknown guards, invalid iteration relations, duplicate edges, exact budget
boundaries, and visibility/exit separation are covered.

Those graphs are logical inputs, not 1,024 legal physical event plans. This
experiment tests the representation and transfer, not event allocation or PTO
hardware semantics. Existing concrete tests continue to check physical programs.

## Next acceptance gates

Standalone choices must preserve incoming generations under conservative partial
writes, outstanding readers on both arms, empty alternatives, and conditional
source/target participation. A common acquisition requires a guarantee on every
feasible path, expressed over guarded generation interfaces rather than a
lexically selected writer. Balanced branch-local repair is the initial placement
baseline. Do not admit a choice merely by setting a whole-choice ordering flag.

Only after independent path tests and native fallback-disabled cases in both GM
modes pass should summaries compose with loops. Recurring composition needs an
inductive event/state invariant; bounded unrolling is only a differential oracle.
The serial loop alternative stays available throughout.

Selective loop repair must ship with a verifier for actual obligation coverage
and event lifetimes. The current serialized-shape checker remains a fast path,
not the acceptance oracle for layouts that intentionally remove cycle actions.
Later relay experiments must measure added ordering separately from action count
and event pressure. No performance benefit follows from fewer events alone.

## Validation record (initial steps 1 and 2)

Baseline `5fcb8cf279084587163c6f0c48306f6ab75275ea` plus this working-tree
patch; existing LLVM/MLIR 19.1.7 and workspace Python 3.12.13. Only incremental
PTOAS targets were built, with at most two resource-intensive workers globally.

```bash
cmake --build build --parallel 2 --target \
  pto-protocol-sync-loop-memory-test pto-protocol-sync-scoreboard-test \
  PTOASCompiler pto-test-opt pto-protocol-sync-local-memory-test \
  pto-protocol-sync-one-shot-test pto-protocol-sync-ready-release-test \
  pto-protocol-sync-direct-repair-test pto-protocol-sync-mixed-test
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python \
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit \
  -v -j 2 build/test/lit --filter protocol_sync \
  -o build/protocol-sync-completion-supply-results.json
```

The build succeeded and the ProtocolSync selection passed **44/44** in 40.35
seconds. After adding the final four two-active-pipe exit cases, only
`pto-protocol-sync-loop-memory-test` was rebuilt. The final focused selection
passed **2/2** in 1.65 seconds:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python \
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit \
  -v -j 2 build/test/lit \
  --filter 'protocol_sync_(loop_memory_unit|scoreboard_unit)' \
  -o build/protocol-sync-completion-supply-focused-results.json
.venv/bin/python \
  .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py \
  --repo . --base HEAD
git diff --check
```

The checker reports nine code/build files, zero errors and zero warnings;
whitespace checking passes. An initial SmallVector copy incompatibility was
fixed using explicit iterator assignment before the successful build. No
compiler warnings were introduced. The regex prefilter initially misread
compound braced conditions; named boolean predicates make those conditions
explicit without suppressing any rules.

SHA-256 comparisons against ten retained pre-change artifacts were identical:
the five local-reuse IR outputs (A2/A3 direct/mixed and repeat emission), and
the five loop-boundary outputs (A2 IR, both A3 GM-mode IR outputs, and both
A3 GM-mode C++ outputs). These are targeted emission checks, not a corpus
rebaseline. Both native boundary GM modes still run with fallback disabled.

The fifteen exit cases comprise 120 case/trip executions. The 1,024 logical
supply graphs comprise 25,600 coverage queries, each with a checked witness
or checked absence. Existing local-memory, cycle, boundary, concrete-scoreboard,
protocol, and mutation tests remain passing. Result JSON lives in the ignored
build tree. No full-system suite, native-corpus campaign, device/CA-model test,
performance qualification, or external static analyzer was run for this slice.
