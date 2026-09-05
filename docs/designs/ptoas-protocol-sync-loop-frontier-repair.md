# ProtocolSync acknowledged loop-frontier repair

## Status

Step 2B (`db6041516`) introduced isolated-loop repair. The boundary continuation
adds a native `loop-frontier` complete-world alternative to F under
`--protocol-sync-mixed --protocol-sync-fallback=fail`. It composes straight-line
prefix/suffix work with an unconditional loop, including zero-trip flow. The
isolated low-level API remains available and is not a full-function certificate.
The new native acceptance fixture is separate from historical corpus results;
no corpus-wide admission or device-performance improvement is claimed.

The builder consumes the bounded local-memory analysis gate, then conservatively
orders every adjacent physical phase, including phases with disjoint storage.
It does not match a logical storage channel or infer storage capacity. This is
the initial fully serialized implementation alternative; sparse backward motion,
candidate coalescing and protocol optimization remain later work.

## Declared subset and interfaces

- One unconditional positive-constant-step `scf.for`, no iter_args or nested
  regions, and ordinary vector-core phases with bounded local UB footprints.
- Physical phases may be in the straight-line prefix, loop body or suffix.
  Choices, sections, dynamic/subview/modulo storage, macros and queues reject.
- Planning rejects pre-existing synchronization and hidden reservations. Explicit
  caller-supplied event reservations are honored; event scarcity publishes no
  partial recipe. Fixed-supply composition remains an integration requirement.
- Events and barriers must be permitted by the existing A2/A3 target contract.
  No new target ordering or visibility rule is introduced.

`buildLoopFrontierRepairPlan` returns Ready, Unsupported or ResourceInfeasible
separately from internal failure. `materializeLoopFrontierRepair` mutates only
caller-owned disposable staging IR. The caller must discard it on failure and
must not commit it merely because local verification succeeds.

`verifyConcreteLoopFrontierRepair` requires a freshly extracted frozen schedule
for the current staged IR. It reconstructs lexical phase/event/barrier order
without consulting the plan, requirement IDs or diagnostic attributes. Unknown
fixed synchronization, extra actions, malformed keys and unsupported topology
reject. The verifier intentionally recognizes a conservative canonical repair
shape; rejecting another layout is not a claim that the layout is unsafe.

## Handoffs and induction

For a three-pipe load/compute/store loop, the recipe is:

```text
prime: set MTE3 -> MTE2 (backedge credit)
loop:
    wait MTE3 -> MTE2
    load
    set MTE2 -> V; wait MTE2 -> V
    compute
    set V -> MTE3; wait V -> MTE3
    store
    set MTE3 -> MTE2
drain: wait MTE3 -> MTE2
```

Each adjacent equal-pipe handoff uses a targeted barrier instead of an event.
If the last and first phases share a pipe, a barrier before the first phase
orders the backedge, and a same-pipe barrier after the loop closes the local
boundary; no credit is primed. The one-phase case therefore retains its
same-static-phase recurrence. No body `PIPE_ALL` operation is emitted.

Under the existing event and barrier completion contract:

1. Each handoff completes its source frontier before its target phase issues.
2. The composed chain orders all phases within an iteration; the closing
   handoff orders the final phase before the first phase of the next iteration.
3. Every forward event's consuming wait occurs before its producer can reach
   the next iteration's set. The closing event is consumed at body entry before
   being rearmed at body exit. Distinct static handoffs use distinct keys within
   each directed event domain; this milestone keeps allocation conservative.
4. The prime permits the first iteration without asserting any live-in data
   production. With zero trips, the drain consumes only the prime. With one or
   more trips, it consumes the final closing token.

These local transition facts give an inductive consumption-before-rearm
argument for arbitrary trips, not merely a bounded unrolling result. The cycle
also establishes all local RAW/WAR/WAW completion and reclamation edges by
ordering every phase occurrence. A capacity-one control handoff says nothing
about the number of physical storage slots.

This proof does not establish GM publication, cross-core visibility, ownership
effects, mandatory function exit completion, or the completeness of shared
operation summaries. Those facts remain separate full-function obligations.

## Independent execution tests

The existing occurrence test now also builds and emits 27 three-phase R/W/RW
topologies, including overlapping buffers and repeated/equal pipes. For each,
an independent test oracle expands actual IR for trips 0, 1, 2, 3, 4, 7, 8 and
11: 216 topology/trip combinations.

The oracle constructs per-pipe FIFO command streams and explores all enabled
issue and asynchronous-completion interleavings within its explicit state
budget. Issuing an operation does not mark it completed. Events and named
barriers wait for preceding source-pipe effects; ordinary same-pipe issue alone
supplies no completion proof. It independently checks fixture byte-overlap
hazards, event underflow/deadlock, live-key rearming and leaked tokens. Budget
exhaustion fails the test, rather than reporting a passing proof.

Additional tests delete every emitted action; move signals before producers
and waits after consumers; mismatch event IDs; guard a signal; and duplicate a
set to expose a live-key rearm in the execution oracle. Tests also cover a
single-phase recurrence, repeated directed domains, non-contiguous available
IDs, exhausted reservations, and rejection of prefix physical effects.

The two oracles have separate jobs: the earlier 648 traces validate sparse
requirement phase-order coverage, while the new execution oracle tests actual
emitted instruction lifetimes. Neither substitutes for device qualification or
validates all canonical record provenance fields.

## Remaining integration gates

The boundary continuation implements canonical-record consistency, outer
handoffs, atomic F selection, conservative event allocation, concrete cycle
reconstruction and the mandatory exit drain. Full residual interpretation still
checks GM visibility, ordered effects and any nonlocal timeline. Unsupported
fixed supply remains a rejection of this alternative, not ignored semantics.

Remaining work is participation for choices, nested-region composition, wider
provenance recovery, integration with fixed supply, and sparse backward motion
instead of a fully serialized chain. Event keys remain distinct per static
handoff in each directed domain. Scarcity cannot publish a partial loop recipe.

## Boundary certificate

The last body pipe is the entry gateway; the first body pipe is the drained
exit gateway. The prefix's final phase hands off to the entry gateway **before
the prime**. The closing wait after the loop hands off to the first suffix
phase. Other adjacent outer phases use directed events or named barriers.

On a zero-trip execution, entry acquisition precedes the prime, the drain
consumes that prime, and the suffix handoff follows the drain. Thus the prefix
still orders the suffix without an unconditional wait for a skipped body set.
With equal first/last body pipes, named barriers replace the closing event;
the entry and exit gateways remain the same pipe. The arbitrary-trip cycle
invariant above composes with these non-recurring boundary handoffs.

F retains the canonical requirement records, including atoms, access IDs and
iteration relations, and compares them against fresh analysis during plan
verification. A compact logical `orderedLoop` certificate supplies completion
for the supported occurrence relations; it does not populate visibility supply.
Only that complete certificate enables canonical loop requirements to replace
local timeline rejection records. Unknown local footprints still reject.

Concrete verification reconstructs every phase, boundary handoff, loop action,
event key and final exit barrier from freshly extracted IR, without reading
planner tags or the selected recipe. It then runs the residual interpreter with
the reconstructed completion certificate. The straight-line-only scoreboard is
not used to interpret loops; the conservative total-order induction establishes
local completion, with the independent bounded execution oracle as a test.

The alternative competes with existing complete worlds, including when optional
protocol recognition is disabled in the mixed planner. It is not yet integrated
into the separate legacy `--protocol-sync-direct-repair` entry point. Action and
event-pressure costs are structural counts, not performance predictions.

## Step 2B review disposition and generality

Independent algorithm and compiler-engineering reviews accepted Step 2B
with no blocking findings for its isolated-loop, local-completion scope. This
is a length-N serialized phase cycle, not yet sparse backward frontier search:
requirements currently gate eligibility, while the repair orders every adjacent
phase. It should remain a conservative alternative as compositional region
analysis and more selective placement develop.

Those reviews requested a native acceptance case with a prefix write, an
ordinary loop with unrelated work, and suffix consumption or reuse. It must
compile with fallback disabled in both GM modes, with canonical entry,
backedge, exit and zero-trip obligations checked by F. F must also check fixed
and hidden event reservations; the local concrete checker has no reservation
context. The boundary continuation implements the native case and canonical
provenance/mask mutations, and rejects fixed/hidden supply for this alternative.
Separate algorithm and compiler-engineering reviews accepted the native boundary
continuation with no blocking findings. Both were read-only code reviews; their
acceptance does not imply independent test reruns or new hardware qualification.

The reviews retain these nonblocking follow-ups:

- Preserve resource-infeasible versus unsupported alternative diagnostics.
- Validate complete-world accounting/optimality alongside semantic consistency.
- Keep the logical certificate builder restricted to checked callers; consider
  an explicitly checked certificate type before broadening its use.
- Add an all-lanes drain and function-return rule to the execution oracle. It
  currently treats `PIPE_ALL` as a separate lane; final exit completion is
  checked by production reconstruction, not independently by that oracle.
- Compose fixed supply and implement sparse placement only with corresponding
  proof and verification extensions. The current exclusions remain intentional.

The current deletion and motion tests mainly establish sensitivity of the
canonical-shape checker. They do not establish that every rejected mutation
is semantically unsafe: this conservative cycle can contain redundant actions.
The duplicate-set test additionally has an independent execution witness.
The boundary continuation adds semantic witnesses for unsafe entry/backedge
deletion, premature signals, late waits and missing drains. The execution oracle uses
fixed 512-byte fixture ranges and shared operation extraction; it is not a
general storage-provenance oracle or hardware qualification.

## Step 2B validation record

Validated on baseline `78a8c09f59a87d90a1a3fa861318d497cd64776f` plus this
working-tree milestone, with the existing LLVM/MLIR 19.1.7 toolchain:

```bash
cmake --build build --parallel 2 --target \
  pto-protocol-sync-loop-memory-test PTOASCompiler pto-test-opt \
  pto-protocol-sync-local-memory-test pto-protocol-sync-scoreboard-test \
  pto-protocol-sync-one-shot-test pto-protocol-sync-ready-release-test \
  pto-protocol-sync-direct-repair-test pto-protocol-sync-mixed-test
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python \
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit \
  -v -j 2 build/test/lit --filter protocol_sync \
  -o build/protocol-sync-loop-repair-lit-results.json
.venv/bin/python \
  .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py \
  --repo . --base HEAD
git diff --check
```

The targeted build succeeded. The focused loop-memory, local-memory and
scoreboard selection passed 3/3 tests; the full ProtocolSync selection passed
42/42 in 35.54 seconds. The changed-code checker reported seven code files,
zero errors and zero warnings; whitespace validation passed. No full-system,
device, timing or native-corpus campaign was run for this local-repair slice.

## Native boundary continuation validation

Baseline `db6041516420c27b4fe77d87fed2c90925921786` plus the boundary
working-tree changes, LLVM/MLIR 19.1.7, workspace Python 3.12.13. Builds and
tests were limited to two workers in aggregate; no LLVM or full-system rebuild
was performed. Relevant compiler and unit targets were built incrementally.

The final ProtocolSync selection passed **44/44** in 32.58 seconds:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python \
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit \
  -v -j 2 build/test/lit --filter protocol_sync \
  -o build/protocol-sync-loop-boundaries-suite-final.json
```

The dedicated boundary regression checks two functions: a three-phase loop
with unrelated scratch work and suffix physical-address reuse, and a
single-phase same-pipe loop. Both compile through native mixed selection with
fallback disabled in both GM modes, and both emitted programs pass fresh
concrete verification. A2 IR emission also passes. The A3 C++ emission smoke
checks use the same input and flags without `--emit-pto-ir`; C++ text emission
is not a device compiler or device execution result.

The multi-phase fixture has ten canonical local requirements, four distinct
event keys and conservative directed-domain pressure two. The single-phase
fixture has four requirements, two event keys and pressure one. These are
fixture-specific structural counts, not corpus or performance measurements.
Safe GM mode succeeds here because the relevant GM reads precede writes;
completion ordering discharges possible WAR/WAW overlap. A separate same-GM
prefix-store/body-load fixture remains rejected in **both** modes: completion
does not supply unqualified publication, and argument disjointness cannot
remove overlap on the same argument.

The loop unit additionally checks both GM contracts with partial local overlap
for trips 0, 1, 2, 3, 4, 7, 8 and 11. Its independent asynchronous execution
oracle witnesses unsafe entry-wait deletion (including zero trips), backedge
wait deletion, premature signal, late wait and missing closing drain. Plan
verification rejects changed atom masks, source access IDs, iteration distances
and missing boundary edges. The previous 27-topology/216-execution cycle tests
and 81-program/648-trace requirement tests remain passing.

One existing mixed-selection assertion was updated from four/two to five/three
attempted/feasible worlds: the extra serial-loop alternative is feasible, but
the previous combined protocol still wins. No safety or allocation assertion
was removed. Changed-code checking passed for fourteen code files with zero
errors/warnings; `git diff --check` passed. Local result JSON files live in the
ignored build tree, not a committed campaign archive. No corpus rerun,
device/CA-model campaign or performance measurement was performed.
