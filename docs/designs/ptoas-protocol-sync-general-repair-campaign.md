# Parallelism-first general frontier repair campaign

## Governing revision — 2026-09-06

The revised objective supersedes the count-first and serialized-coverage policy
recorded below. Preserve the pipeline overlap permitted by required dependencies
and qualified target mechanisms within the existing operation schedule. Do not
reorder computation or alter buffer allocation. Fewer synchronization actions
are not an improvement when they add blocking between independent operations.

Normal repair publishes each required completion immediately after its source
frontier and acquires it immediately before its target. Initially, only identical
endpoint obligations share a repair. Broader sharing needs a certificate that it
adds no blocking; intersecting legal placement intervals are insufficient.
For `L1, L2, C1, C2` with only `L1 -> C1` and `L2 -> C2`, retain two events:
`set A` after L1, `set B` after L2, `wait A` before C1, `wait B` before C2.
The independent execution oracle must permit L2 and C1 to remain outstanding
together. Delaying the first signal until after L2 is a race-free but unacceptable
normal placement, not a performance optimization.

Every admitted **pattern** identifies a pipeline structure and supplies its known
efficient synchronization implementation, including intended overlap, storage
reuse, participation and event lifecycle. Elementary OneShot handoffs belong to
general repair, not the pattern catalog. Capacity-one control handshakes are
general recurring machinery; proven multi-slot pipeline protocols are useful
specializations. Existing narrow ReadyRelease construction remains transitional
until its overlap contract and general/pattern classification are audited.

Serialization is permitted only as explicit resource recovery after certified
event pressure, not simply because selective synthesis is missing or an analysis
budget expires. First try a non-serializing plan and proven-safe event reuse.
Total-phase loop cycles and V-hub packages remain reference constructors; they
must not win normal selection by having fewer event pairs. Removing them from
normal selection can temporarily lower native admission. Report that regression
honestly rather than counting serialized coverage as campaign completion.

### Revised implementation order

1. **P0/P1 — readiness policy:** independent direct frontiers, matching policy
   verification and overlap regression; fold normal OneShot selection into direct
   repair; remove count-first and serialized normal-world selection.
2. **P2 — event lifetimes:** prove consuming waits happen before rearming; separate
   proven pressure, conservative unknown interference and exhausted analysis.
   Add explicitly attributed serialization recovery only after that distinction.
3. **P3 — bounded evidence and frontend facts:** finish frozen collection without
   losing failures; descriptor state, access precision, authoritative core/view
   provenance and both GM contracts. Preserve the historical 394-row population.
4. **P4 — canonical structured state:** shared atoms, generations, outstanding
   readers/writers, stable requirement IDs and compositional boundary summaries.
5. **P5 — selective structured synthesis:** ordinary loops, choices, their
   composition and nesting, entry/backedge/exit/bypass and slot distances, with
   semantic verification and arbitrary-trip event invariants in the same slice.
6. **P6 — efficient protocols and remaining effects:** qualify genuine pipeline
   specializations, fixed supply, visibility, queues, communication and ACC/L1
   domains individually. Do not weaken semantic gates for admission.
7. **P7 — closure:** frozen corpus in A2/A3, both GM contracts and patterns on/off;
   native compilation without fallback, C++ emission and fresh verification.
   Report normal, resource-serialized, unsupported, analysis and collection
   outcomes separately. Host overlap witnesses are not device throughput results.

Each accepted slice requires independent algorithm and compiler-integration
reviews and a local commit. No pushes. The historical N-stage ledger below
records prior work, not the current selection policy or acceptance result.

## Contract and frozen source baseline

The campaign starts at PTOAS `61606de36841b7c8d857ed2a5dfc5d8a923bcc1a`.
On 2026-09-06, read-only remote resolution confirmed current `main` heads:

- PyPTO: `9f657f37ed20ce148b46fb7229c267a152a0644e`.
- PyPTO-Lib: `57e9d6a9294d9c38edd042f1edb5bdc50b4622bb`.

These match the clean snapshots under `build/protocol-sync-native-corpus/sources/`.
The input population is not yet collected. The historical externally reported
152 kernels are not a current measurement. Keep the 394-row historical
differential corpus separate from the newly collected frontend population.

Completion means every frozen valid frontend row compiles natively with legacy
fallback disabled, emits C++, and passes fresh concrete verification, separately
for A2/A3 and both GM contracts. General-only measurements disable specialized
pipeline-pattern selection. Completion requires selective, obligation-driven
structured repair; serialized references do not satisfy normal admission.
Host acceptance is not a hardware-correctness measurement.

## Execution and commit policy

Implement, independently review, address findings, and commit each accepted
slice locally. **Do not push**, including backup branches. User authorization
covers these local commits without repeated permission prompts. Preserve
unrelated `third_party/` content and other user worktrees. Use at most two
resource-intensive workers across the entire machine; main owns targeted builds
and tests, review agents inspect read-only. Store durable evidence under `build/`,
not the RAM-backed `/tmp`. Do not rebuild unchanged targets or LLVM.

## Historical ordered stages (superseded by P0–P7 above)

1. **N0 — freeze and baseline:** immutable frontend revisions, unsynchronized
   generated inputs, parameters, collection failures, commands and hashes;
   general-only four-way acceptance and concrete/C++ follow-up; all exposed
   blockers per row. Keep collection failures in the denominator.
2. **N1 — semantic facts:** descriptor-state updates, authoritative core context,
   forwarded/view provenance and Exact/Conservative/Unknown regions; preserve
   both GM contracts and require independent byte-set/descriptor tests.
3. **N2 — canonical requirements:** typed obligations independent of recognition
   and selection, outstanding readers and writes, generations and stable
   provenance; replace old protection only after internal/boundary no-omission
   accounting and dual-run agreement on the existing subset.
4. **N3 — structured interfaces:** memory plus completion/token transfer through
   sequences, joins and ordinary loops including entry/backedge/exit/bypass;
   conditional/nested loops and SSA forwarding; independent path oracle and an
   arbitrary-trip invariant, not bounded unrolling as proof.
5. **N4 — selective backward repair:** query established supply, traverse memory
   links/lanes/region interfaces, share legal handoffs with balanced participation;
   semantic concrete verification and token lifetime/progress witnesses ship
   together. Demonstrate removal of unnecessary adjacent-phase ordering.
6. **N5 — fixed supply and allocation:** import certified existing mechanisms and
   reservations before repair; reuse event generations only after proven death;
   deterministic feasibility retries retain conservative interference otherwise.
7. **N6 — remaining domains:** GM visibility, communication, queue ownership,
   multistage cube/local storage and ACC/proxy effects, prioritized by actual
   corpus blockers. New target claims need documentation/model/qualification
   evidence; no operation-name whitelist or generic body ALL-barrier escape.
8. **N7 — closure:** rerun every frozen row and historical differential population;
   preserve per-row artifacts, metrics and hashes. Missing frontend facts or
   unqualified target effects stay explicit blockers, never denominator removal.

No new pattern optimizations precede the general baseline. Reuse proven
lifecycle facts and complete parameterized handshakes where useful, without
emitting partial cyclic protocols. Each stage requires algorithm/soundness and
compiler-integration review, focused regressions and the ProtocolSync checkpoint
suite before committing. Device validation and target claims remain separately
qualified. Exhausted search budgets are not semantic counterexamples.

## Progress ledger

### P0/P1 — independent readiness policy accepted

This slice over `58bad1ba37570711ccea8c0a3b0006d79e2cd4a3` replaces greedy
interval merging with exact-endpoint repair and matching placement verification.
Mixed normal synthesis no longer selects elementary OneShot candidates or ranks
complete plans by event count. Serialized loop/V-hub recipes remain explicitly
constructed reference tests, not normal selection alternatives.

The new `protocol_sync_independent_readiness.pto` checks separate publications
and late acquisitions in A2/A3 and both GM modes, plus fresh concrete verification
and C++ emission. The direct unit's independent asynchronous oracle establishes
an L2/C1 outstanding-overlap witness and exhaustive bounded safety. Delaying the
first signal or advancing the second wait remains safe but removes the witness.
Identical endpoint obligations still share a repair. Existing reference loop,
choice, boundary, zero-trip and event mutation coverage is retained.

Validation on 2026-09-06:

- Targeted `cmake --build build --parallel 2 --target PTOASCompiler
  pto-protocol-sync-direct-repair-test pto-protocol-sync-mixed-test
  pto-protocol-sync-loop-memory-test` and incremental test-target rebuilds pass.
- With the workspace venv on PATH, `taskset -c 0,1 .venv/bin/python
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit -v -j1
  build/test/lit --filter 'protocol_sync_' -o build/protocol-sync-p1-tests.json`:
  **47/47 pass**, 61.82 seconds. This is the focused host checkpoint, not the
  entire repository suite or a new frontend corpus/device campaign.
  Result JSON SHA-256:
  `613ee4e44ac75dc9a90d0030b6e52f0fde9ac5aa13a0ace8f73fae5e8e055468`.
- Changed-code compliance check against HEAD: 13 files, zero errors/warnings;
  `git diff --check` passes. New C++ remains C++17, target-scoped CMake reuses the
  existing oracle source, and no target-ordering semantics were strengthened.
  Proprietary static analyzers and device qualification were not run.
- Independent read-only algorithm and compiler-integration reviews by
  `campaign_n0_algorithm_review` and `campaign_n0_compiler_review`: accepted.

P2 follows with event-generation death/reuse and honest pressure
classification. No serialized resource-recovery path is enabled by this slice.
Selective structured admission and the frozen corpus closure remain unfinished.

### P2a — once-only acknowledged event reuse

`EventLifetime` proves event consumption using an action-order graph, separate
from physical-memory completion. For each qualified logical handoff, its signal
precedes its consuming wait. Strictly ordered synchronization actions on the
same physical core and pipe supply further edges. The graph crosses directed
event domains: a V-to-MTE2 acknowledgement can prove that an earlier MTE2-to-V
wait consumed its token before MTE2 signals its next generation.

Planned waits are before their target phases and signals after their source
phases; concrete reconstruction uses the actual synchronization operations.
Ambiguous same-point ties add no edges. Every edge advances lexical position or
the before/after offset, so the proof cannot depend on a cyclic assumption about
the assigned event ID. Safe sharing follows by induction over that order: every
previous consumption precedes the next rearm. No signal/wait is moved or added.

Scope is deliberately bounded: unguarded, non-recurring handoffs in a single-block
function, with actual ancestry checked independently of generation metadata.
Loops, choices, missing anchors, recurring protocol kinds, and more than 128
total logical generations yield no new reuse proof. Conservative interference
remains. An absent path is not proof of simultaneous liveness; the existing
`ResourceInfeasible` result describes conservative graph coloring, not certified
physical scarcity. P2b must establish that distinction before any serialization
recovery is enabled.

The regression has seven load/compute generations sharing exact local storage.
Required reverse reclamation already acknowledges each forward acquisition.
All seven forward generations and six reverse generations use one ID per
direction. The native fixture is tested in A2/A3 and both GM contracts, with
fresh concrete verification and C++ emission. The execution oracle checks safe
reused generations and requires an actual unsafe witness when an acknowledgement
is removed. Separate negatives cover absent acknowledgements, forged loop
metadata, guards, missing anchors, and valid-ID inputs exceeding the proof cap.
The P1 independent-readiness regression remains the check against newly added
serialization. This is host/model evidence, not a device performance result.

Validation over P0/P1 commit `ed7037859cc15faf717261fe714acd745afe2f15`
on 2026-09-06:

- Targeted two-worker build of `PTOASCompiler` and the direct, mixed,
  loop-memory, OneShot, ReadyRelease, local-memory and scoreboard unit targets:
  passed; no LLVM rebuild. The OneShot reference regression was updated to
  explicitly check its existing reverse acknowledgement before expecting reuse.
- With the workspace venv on PATH, `taskset -c 0,1 .venv/bin/python
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit -v -j1
  build/test/lit --filter 'protocol_sync_' -o build/protocol-sync-p2a-tests.json`:
  **48/48 passed**, 62.22 seconds. Result SHA-256:
  `84687677044b976222749989a8bd34ff5f43a41ae6da4ec16c9c87b435a8027f`.
  A subsequent named-boolean compliance cleanup changes no test semantics;
  its affected OneShot target is rebuilt and rerun separately.
- Changed-code compliance check: nine files, zero errors/warnings;
  `git diff --check` passed. Independent algorithm and compiler-integration
  reviews accepted this bounded slice. No full system/device or new corpus run.

P2a does not finish P2: unknown interference and proof-budget exhaustion still
need explicit propagation before any pressure-recovery policy is enabled.
Selective loop/choice synthesis and complete frontend admission remain open.

### P2b — allocation failure attribution

The allocation result now distinguishes:

- `event-interference-unresolved`: the conservative interference graph cannot
  fit the available colors. This is not a simultaneous-live hardware witness.
- `no-unreserved-event-ids`: declared reservations leave zero compiler IDs in
  the affected domain. This does not establish that serialization can help.
- `event-allocation-analysis-limit`: feasibility search, domain size, or the
  bounded consumption-order analysis prevents establishing an assignment.

Successful conservative assignments remain accepted when a consumption proof
or color minimization hits its limit. Failure attribution includes the affected
core/directed domain and available-ID count in the raw allocator result; plan
dumps and CLI records currently propagate the typed reason and aggregate counts.
Those domain details are not yet exposed in the CLI. The historical
`max_event_domain_pressure` field means colors used by the computed assignment,
not measured or proven maximum simultaneously live hardware generations.

Direct and mixed plans retain a separate handled allocation-analysis-limit
status. Explicit reference OneShot/ReadyRelease routes retain unsupported status
with allocation attribution. An optional protocol's allocation limitation no
longer aborts mixed selection as an internal error; the direct alternative can
still be tried. Failed allocations keep all event IDs unassigned, and direct/
mixed verification reproduces their status and cause. Budget limits must not
turn into internal-error records merely because verification repeats a bound.

No failure class authorizes serialization. A future pressure certificate needs
a complete supported event/control model and an independently replayable cut
with more live generations than available IDs. P2a's under-approximation of
guaranteed order cannot supply that certificate by an absent path or clique.

Validation correction: P2b inspection found that earlier P0/P1 and P2a checkpoint
runs used a stale statically linked `pto-test-opt` for several native lit tests.
The compiler library and affected unit binaries were current, but the reported
47/47 and 48/48 totals were not entirely current-head validation. P2b rebuilds
that driver explicitly with every affected unit target and reruns the checkpoint.
In particular, the old 16-world expectation and the reference OneShot distinct-ID
expectation must not be interpreted as current normal-planner behavior.

Validation over `95b083310f1caa88ad2c7c981f60ddefa784e7b8`, 2026-09-06:

- Targeted `taskset -c 0,1 cmake --build build --parallel 2 --target
  PTOASCompiler pto-test-opt pto-protocol-sync-direct-repair-test
  pto-protocol-sync-mixed-test pto-protocol-sync-loop-memory-test
  pto-protocol-sync-one-shot-test pto-protocol-sync-ready-release-test
  pto-protocol-sync-local-memory-test pto-protocol-sync-scoreboard-test` passed.
  Shared statistics/plan interfaces required dependent PTOAS recompilation;
  review corrections used incremental builds, with no LLVM rebuild.
- Fresh-linked `llvm-lit -v -j1 build/test/lit --filter 'protocol_sync_'
  -o build/protocol-sync-p2b-tests.json`, using the same venv/affinity command
  as P2a: **49/49 passed**, 66.71 seconds. SHA-256:
  `9447defe08b9b491348c1e33d5e551e53a855b146c318eeebede538e4cc6cce4`.
  This supersedes the stale-driver totals above. Corrected the old same-pipe
  fixture to retain its second late acquisition before the required WAW barrier.
- `.venv/bin/python -m unittest discover -s test/experiments/protocol_sync
  -p test_records.py`: **28 passed**. The corpus blocker matrix retains all
  three terminal allocation reasons, without treating failed optional
  alternatives inside admitted functions as terminal blockers.
- Changed-code prefilter: 24 code files in the worktree, zero errors/warnings;
  `git diff --check` passed. Independent algorithm and compiler-integration
  reviews accepted. No new device or frontend acceptance campaign in this slice.

The new native 129-generation fixture checks handled analysis limits in A2/A3
and both GM contracts. Unit regressions cover zero reserved IDs even above the
domain cap, successful bounded minimization, forged attribution and cleared
assignments. Certified physical-pressure recovery remains deferred; proceed
with P3 corpus collection and semantic facts without enabling serialization.

### P3a — preserve source-backed frontend imports

The static collector now places an example driver's directory on its import
path, matching its absolute sibling imports, while still checking that the
loaded entry is exactly the inventoried source file. Collector arguments remain
hidden from module-local parsers. This is collection infrastructure only: it
does not change compiler admission or invoke a kernel/device runtime. Trusted
module top-level Python still executes. Failed seeds and partial PTO outputs
retain their separate records.

Both independent algorithm and compiler reviews accepted the bounded change.
On 2026-09-06, `taskset -c 0 .venv/bin/python -m unittest discover
-s test/experiments/protocol_sync -p 'test_*.py'` passed all **38 tests**.
The changed-code prefilter and `git diff --check` passed. No compiler rebuild
was needed. The previous interrupted `static-n0` collection remains untouched;
the next collection uses a fresh disk-backed result directory and the same
frozen frontend source commits. Static seeds are not the complete parameterized
kernel acceptance population.

### Historical N0 progress

- N0 in progress: source heads resolved; added general-only mixed-mode selection
  control and evidence-policy propagation. No current-head corpus results yet.
- Collection environment: the PTOAS venv has no PyPTO, pytest or torch. Recover
  an isolated compatible frontend environment without altering user worktrees.
- Next: validate/review the selection-control slice, then collect the frozen
  frontend inputs and record all collection and compiler blockers.

### N0a — general-only selection control accepted

`--protocol-sync-mixed --protocol-sync-patterns=off` now retains general direct,
loop and structured alternatives while disabling OneShot/ReadyRelease synthesis.
This is selection-policy plumbing, not broader semantic admission. The
acceptance runner and C++ follow-up preserve the policy and require explicit
zero selected-protocol counters. Historical follow-ups omit the new switch
when their metadata predates it, preserving old compiler compatibility.

Independent read-only reviews `campaign_n0_algorithm_review` and
`campaign_n0_compiler_review` accepted this slice. The latter found the old
compiler compatibility issue; it was corrected and re-reviewed. Added tests
cover policy mismatches/missing counts and native plus concrete straight-line,
choice, loop-choice and prefix/loop/suffix paths with patterns disabled.

Validation on 2026-09-06:

- `taskset -c 0,1 cmake --build build --parallel 2 --target PTOASCompiler pto-test-opt`:
  passed. Adding the generated pass option required rebuilding its dependent
  translation units; no LLVM rebuild or parallel competing job was started.
- `.venv/bin/python -m unittest discover -s test/experiments/protocol_sync -p test_records.py`:
  27 passed.
- `.venv/bin/python .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py --repo . --base HEAD`:
  9 changed code files, zero errors/warnings. This prefilter does not substitute
  for unavailable external analyzers or hardware qualification.
- With `.venv/bin` prepended to `PATH`, run the configured LLVM `llvm-lit`
  through `.venv/bin/python` under `taskset -c 0,1`, `-v -j1 build/test/lit
  --filter protocol_sync -o build/protocol-sync-general-n0-suite.json`:
  46/46 passed in 58.61 seconds. The focused new test also passed separately.
- `git diff --check`: passed. No full system or device suite was run.

Next is N0 collection/environment recovery and the four-way native baseline;
N0 as a whole and the full campaign remain incomplete.

### N0b — static frontend collection adapter accepted

Built the frozen PyPTO snapshot's `pypto_core` incrementally in `build-corpus`
with an isolated Python 3.14.6 environment, system-installed CPU torch 2.12.0,
and nanobind. Initialized only the pinned libbacktrace, msgpack-c and Simpler
source submodules. The first build exposed a missing Simpler header; after
recovering that pinned source, the same targeted build resumed successfully.
No frontend tracked source, unrelated checkout, or global environment changed.
The build used two workers; nested libbacktrace make was explicitly limited to
one worker via `CMAKE_BUILD_PARALLEL_LEVEL=1` at configuration time.

The static adapter inventories 349 example/model entry seeds at these source
revisions. This is not a generated-kernel count or a complete acceptance
population: nested constructors, missing signature facts and parameterized tests
still require collection work. Raw PTO comes from public pre-pass specialization
and `ir.compile(skip_ptoas=True, memory_planner=PYPTO)`, not cached C++ or legacy
synchronization. Partial outputs retain their failed parent record. Module
imports execute trusted frontend top-level Python; the adapter never invokes
the compiled kernel or a device runner.

Both independent N0 reviewers accepted after fixing an implicit codegen thread
pool: `PYPTO_CODEGEN_MAX_WORKERS=1` now accompanies the BLAS/OpenMP limits in
both parent and standalone worker paths. Added source-file import verification,
runtime/package/module provenance, syntax-error accounting and explicit scope
documentation. The hello-world smoke emitted raw physically planned PTO.

Validation on 2026-09-06: 36 stdlib experiment tests passed; focused evidence lit
test passed; the 46-test ProtocolSync suite passed in 58.99 seconds without a
new native build. Changed-code prefilter: four files, zero errors/warnings.
Artifacts: `build/protocol-sync-collector-focused.json`,
`build/protocol-sync-collector-suite.json`, and
`build/protocol-sync-native-corpus/smoke-hello-world/`.
No full system/device campaign was run. Next: execute collection and extend
construction/parameterized adapters, preserving their separate denominators.
