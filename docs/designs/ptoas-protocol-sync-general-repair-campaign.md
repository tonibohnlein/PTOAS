# General frontier repair campaign

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
OneShot/ReadyRelease selection. Existing qualified serialized alternatives remain
feasibility fallbacks, but completion also requires selective, obligation-driven
structured repair. Host acceptance is not a hardware-correctness measurement.

## Execution and commit policy

Implement, independently review, address findings, and commit each accepted
slice locally. **Do not push**, including backup branches. User authorization
covers these local commits without repeated permission prompts. Preserve
unrelated `third_party/` content and other user worktrees. Use at most two
resource-intensive workers across the entire machine; main owns targeted builds
and tests, review agents inspect read-only. Store durable evidence under `build/`,
not the RAM-backed `/tmp`. Do not rebuild unchanged targets or LLVM.

## Ordered stages and acceptance gates

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
