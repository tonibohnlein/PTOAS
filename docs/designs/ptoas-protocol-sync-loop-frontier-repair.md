# ProtocolSync acknowledged loop-frontier repair

## Status

Step 2B, following the occurrence analysis at `78a8c09f5`, adds a low-level
repair builder, disposable-clone materialization, and an independent concrete
cycle verifier. It is **not connected to F selection or native pass emission**.
This is executable repair machinery, but not yet a complete function certificate.
No native corpus admission or device-performance improvement is claimed.

The builder consumes the bounded local-memory analysis gate, then conservatively
orders every adjacent physical phase, including phases with disjoint storage.
It does not match a logical storage channel or infer storage capacity. This is
the initial fully serialized implementation alternative; sparse backward motion,
candidate coalescing and protocol optimization remain later work.

## Declared subset and interfaces

- One unconditional positive-constant-step `scf.for`, no iter_args or nested
  regions, and ordinary vector-core phases with bounded local UB footprints.
- Every physical phase is in the loop body. Prefix/suffix physical effects,
  choices, sections, dynamic/subview/modulo storage, macros and queues reject.
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

1. Add canonical-record consistency and provenance mutations before selecting
   repairs by obligation ID or mask.
2. Compose prefix/suffix requirements with loop entry, exit and zero-trip flow.
3. Represent this indivisible cycle as a complete-world alternative in F,
   including allocation, costs, fixed supply and mandatory exit drains.
4. Extend F's concrete-world reconstruction to consume the local certificate
   while retaining all nonlocal, visibility and unsupported-effect obligations.
5. Run native compilation with fallback disabled in both GM modes. Then extend
   participation to choices and nested-region composition.

The public pass remains fail closed until those gates are implemented together;
the low-level local verifier is not used as a shortcut around F.

## Review disposition and generality

Independent algorithm and compiler-engineering reviews accepted this milestone
with no blocking findings for its isolated-loop, local-completion scope. This
is a length-N serialized phase cycle, not yet sparse backward frontier search:
requirements currently gate eligibility, while the repair orders every adjacent
phase. It should remain a conservative alternative as compositional region
analysis and more selective placement develop.

Before native integration, the next acceptance case is a prefix write, an
ordinary loop with unrelated work, and suffix consumption or reuse. It must
compile with fallback disabled in both GM modes, with canonical entry,
backedge, exit and zero-trip obligations checked by F. F must also check fixed
and hidden event reservations; the local concrete checker has no reservation
context. Canonical provenance and mask mutations remain an integration gate.

The current deletion and motion tests mainly establish sensitivity of the
canonical-shape checker. They do not establish that every rejected mutation
is semantically unsafe: this conservative cycle can contain redundant actions.
The duplicate-set test additionally has an independent execution witness.
Integration should add semantic witnesses for unsafe backedge deletion,
premature signals, late waits and missing drains. The execution oracle uses
fixed 512-byte fixture ranges and shared operation extraction; it is not a
general storage-provenance oracle or hardware qualification.

## Validation record

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
