# Balanced choices and ordinary-loop composition

## Scope

This continuation over `5fcb8cf279084587163c6f0c48306f6ab75275ea` implements
steps 3 and 4 of the [revised order](ptoas-protocol-sync-completion-supply.md):
standalone choices/joins, then choices inside an ordinary loop. It retains the
existing straight-line and serial-loop alternatives. It is a conservative native
completion baseline, not selective loop-frontier optimization.

The bounded language admits sequences, result-free `scf.if` (including empty
arms), and at most one unconditional top-level `scf.for` with constant positive
step and no carried SSA arguments. Choice guards have depth at most four;
recursive region traversal is bounded at sixteen. The complete function must
contain at least one choice, at most 128 ordinary vector-core phases and 1,024
accesses. Supported phase pipes are V, MTE2 and MTE3, with qualified PhaseEnd
completion and bounded UB allocation footprints.

Nested loops, conditional loop execution, result-bearing choices, macros,
queues, pre-existing synchronization and hidden reservations do not enter this
alternative. Unknown local footprints reject its atomic local proof. Nonlocal
effects still go through F's residual interpreter. In particular neither GM
alias mode permits unqualified same-address MTE3-to-MTE2 publication.

## Requirements: guarded outstanding histories

Physical atoms are constructed once across the function's shared storage
domain, not separately inside branch blocks. For each atom, structured transfer
keeps possible writing definitions and all outstanding read/write histories.
Each arm receives the same incoming state; its outgoing state joins by union,
including the unchanged state of an empty alternative.

Allocation extents are conservative footprints. A possibly partial write must
not kill the incoming generation or outstanding readers. This baseline therefore
keeps all relevant history and exposes only may-definitions: must-definitions
are empty and possible live-in state remains present. These records are not
full GenerationSSA or sparse retirement. Region summaries expose incoming and
outgoing history snapshots; they are not optimized boundary transfer functions.

Every overlapping write/read, read/write or write/write pair receives canonical
access endpoints, atom membership, precision and occurrence provenance. Pairs
within one phase occurrence and ordinary read/read pairs are excluded. Canonical
requirements remain independent of selection and coverage.

Participation distinguishes must-coexecute, same feasible path, mutually
exclusive, source-may-be-absent, target-may-be-absent and unknown. Opposing arms
of one choice are exclusive only in the same iteration. They may both execute
in different loop iterations.

The loop transfer visits a symbolic body twice, retaining first-pass histories
as carried and unioning the incoming bypass state. Body/body carried requirements
are `LoopCarriedAny`: **every positive distance**, not an inferred distance one.
Keeping all histories makes this a conservative closure over static access
identities, including writes skipped for arbitrarily many iterations. This is
why two transfer passes suffice for this overapproximation; finite execution
tests alone are not the proof. Prefix/body and body/suffix requirements retain
entry and exit relations. Prefix/suffix same-iteration requirements survive
unchanged and also apply on the zero-trip bypass.

Bound exhaustion discards the partial structured result. No access is marked
covered by successful analysis alone; F requires a complete checked repair.

## Supply and repair: one balanced completion interface

The baseline chooses PIPE_V as a completion interface. Every non-V phase P is
wrapped, entirely inside its original block and guard, in:

```text
set V -> P
wait V -> P
phase on P
set P -> V
wait P -> V
```

Each V phase is followed by a target-qualified V barrier. There are no body
PIPE_ALL repairs. The mandatory function-exit drain remains separately emitted.

The dynamic package invariant is: prior physical work is complete at the V
interface on entry; the phase has completed at that interface on exit; neither
direction retains an unconsumed token. A package's first publication can carry
completion or initial control permission; it does not assert that live-in
storage has been produced.

The reverse acquisition is before the next forward publication on V. The
forward acquisition must have been consumed before P publishes its return.
Consequently sequential packages can reuse one key per directed event domain
without rearming a live generation. Empty arms preserve the interface. Both
arms establish the same semantic interface regardless of which operation IDs
executed. Sequences and arbitrary positive trip counts compose by induction;
zero trips compose prefix directly with suffix. There is no loop-wide prime
credit because each dynamic package balances within its own execution.

This deliberately serializes every phase, even disjoint accesses. It is not a
claim that storage has capacity one or that a lane's lexical issue order implies
completion. It is the complete conservative alternative against which later
selective repairs can be checked.

`SyncCompletionSupply` v1 still declines guarded and recurring programs. This
integration uses a checked list of acknowledged phase interfaces, **not** a
generalized completion-summary implementation. The list is a logical certificate
whose validity comes from the complete recipe, never a whole-choice order flag.

## Whole-world and concrete verification

F compares the structured alternative with existing complete worlds. Its plan
verifier reconstructs the expected phase packages and canonical requirements,
checking atom provenance, participation, event keys and cost. Only complete
worlds reach staged emission. Static cost counts every branch's instructions;
it is not a path-weighted runtime or performance estimate.

After materialization, fresh extraction reconstructs actual adjacent set/wait
pairs and named barriers on the phase's own block. It verifies directions,
matching target-legal IDs, final drain, and that every fixed action belongs to
exactly one checked package. It does not consult planner tags or a selected
recipe. Only then does the completion certificate discharge canonical local and
supported nonlocal requirements. Visibility receives no such certificate.

This verifier recognizes a conservative repair language, not all safe layouts.
A deleted redundant action can fail its shape check without creating a race.
Therefore mutation tests separately run an asynchronous execution oracle and
require an actual unsafe-state witness; oracle budget exhaustion is distinct.

## Validation and interpretation

The unit population has six shapes in both GM modes: one-arm vector writes,
one-arm stores/readers, and two-arm reader/writer choices, standalone and inside
a loop, with partial physical overlap and prefix/suffix effects. Independent
concrete path expansion checks every raw overlapping local hazard against the
canonical store. The asynchronous oracle independently explores issued versus
completed phases and consuming event tokens for each selected path.

Trip counts are 0, 1, 2, 3, 4, 7, 8 and 11. All path masks through three
iterations are enumerated; larger trips include skipped, alternating and
all-taken paths. Actual choices may vary each iteration in these adversarial
tests even though a fixture's scalar condition is invariant: this is a safe
overapproximation, not a general scalar-condition interpreter.

Mutations delete entry/return waits, move a branch return acquisition to a common
frontier, or move it onto the opposing arm. Canonical atom/certificate mutations
are also rejected. Separate same-GM publication negatives test both planning
and fresh verification, in both alias modes, standalone and in a loop.

The native lit fixture compiles a standalone choice and a loop choice with
fallback disabled, reparses emitted IR for fresh verification in both GM modes,
checks A2/A3 output, and checks generated C++ on A3. These are compiler and
bounded model tests, not a device campaign or current PyPTO corpus census.

No throughput benefit or new hardware qualification is claimed. Target rules
remain those already qualified by the branch; this milestone adds composition
and independent path tests, not new visibility, ACC/proxy or cross-core rules.

## Executed validation and joint review

Validated on 2026-09-05 with LLVM/MLIR 19.1.7 and workspace Python 3.12.13,
over the named base plus the uncommitted continuation. Incremental compilation
used C++17 and the configured warning-as-error policy, with at most two build
workers. Only affected compiler/test targets were requested:

```bash
cmake --build build --parallel 2 --target \
  pto-protocol-sync-loop-memory-test PTOASCompiler pto-test-opt \
  pto-protocol-sync-local-memory-test pto-protocol-sync-scoreboard-test \
  pto-protocol-sync-one-shot-test pto-protocol-sync-ready-release-test \
  pto-protocol-sync-direct-repair-test pto-protocol-sync-mixed-test
PATH="$PWD/.venv/bin:$PATH" taskset -c 0,1 .venv/bin/python \
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit \
  -v -j 1 build/test/lit --filter protocol_sync \
  -o build/protocol-sync-structured-lit-results.json
```

The test invocation uses one lit worker and two allowed CPUs; this LLVM thread
pool implementation respects the affinity mask. PTOAS does not expose
`--mlir-disable-threading`, so that unsupported flag is not used for the final
validation. The C++ occurrence/scoreboard oracle drivers explicitly disable
MLIR multithreading.

- Focused native/occurrence tests: **2/2**, 4.31 seconds, recorded in
  `build/protocol-sync-structured-focused.json`.
- Full ProtocolSync selection: **44/45**, 72.96 seconds. The sole failure was
  an obsolete diagnostic count for `guarded_direct`: the new complete alternative
  raises attempted/feasible worlds from one to two. The same cheaper direct-only
  world and synchronization actions remain selected.
- After updating that exact assertion, the affected test passed **1/1**, 3.24
  seconds, using the same command with `--filter 'protocol_sync_mixed.pto$'` and
  `-o build/protocol-sync-structured-counter-recheck.json`. Thus all **45** current
  ProtocolSync tests have passing validation, without repeating the entire suite
  or rebuilding for an expectation-only change.
- The changed-code prefilter checks **24** C++/header/CMake files with **zero
  errors and zero warnings**; `git diff --check` passes. Applicable C++ bounds,
  nullability, explicit failure, scoped-build and compiler-test rules were also
  reviewed semantically. No external security analyzer was run.

The twelve structured worlds exercise 426 selected path/trip rows. Forty
instruction mutations require concrete unsafe-state witnesses, separately from
24 canonical-atom/certificate mutations. Four same-GM publication cases remain
rejected by both planning and fresh concrete verification. These counts are
fixture observations, not production kernel admission or hardware measurements.

Testing exposed and fixed the concrete eligibility treatment of fixed-sync
summary markers: actual emitted events advertise fixed supply, which must be
checked by instruction reconstruction rather than rejected as opaque macros.
The exception is concrete-only; planning still excludes pre-existing supply.

Both the algorithm reviewer and compiler-integration reviewer **accepted** the
joint steps 3/4 source/test delta with no blocking findings. They performed
read-only review, not independent test execution. Their nonblocking follow-ups
are deeper nested-choice fixtures and atom-mask-specific differential checks.
Both explicitly qualify this as a conservative serialized interface baseline,
not sparse structured placement or full GenerationSSA.

Result JSON files remain ignored local build artifacts, not a frozen evidence
archive. No new full-host, PyPTO/PyPTO-Lib corpus, device, CA-model, performance
or target-qualification campaign was run. No commit or push belongs to this
validation step.
