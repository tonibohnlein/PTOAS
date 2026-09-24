# Integrated review of Phase A patches 2–6

Date: 2026-09-24. Worktree: `codex/handoff-foundation`, based on
`0a38c8e9f4bd4852177c1dd6f6e0d0ee26b6ca16`. These changes are uncommitted.
The pre-existing step-1 documentation changes are preserved.

## Integration and independent review

The five Downloads bundles each targeted the same base commit. Steps 4–6
therefore required three-way integration with earlier steps, followed by native
compilation and semantic review. They were not an ordered patch series.

| Step | Scope | Independent review |
| --- | --- | --- |
| 2 | Legal original cuts, owner/continuation and snapshot keys, qualified may-interval queries | Accepted at this scope |
| 3 | Shared full-write semantics plus exact-cell geometry; conservative unknown coverage | Accepted at this scope |
| 4 | Shared factored transfer, scoped queries, incoming writers/readers and guarded demands | Accepted at this scope |
| 5 | Original-value identity, integer arithmetic, four availability outcomes and typed prerequisites | Accepted at this scope |
| 6 | Stable guarded obligation families, lazy membership/enumeration and effect/typed witnesses | Accepted at this scope |

The independent reviewer accepted step 6 after the final integrated runs;
all five increments are accepted at their stated scopes.

Acceptance of these increments does not establish all checklist rows touched by
them. In particular, scoped incoming histories require later transport proofs;
D1–D4, exact boundary procedures, covering boundaries, descriptor formation and
subscription/consequence closure remain assigned to steps 7–13. Full Phase A
parity requires step 14. FrontierSynch still stops before construction.

## Corrections made during review

- Preserved the shared full-write interface through the importer and both sync
  paths. Fixed backward RMW handling: a definite RMW remains a future reader
  for an earlier writer while its forward write retires preceding readers.
- Connected factored, value and obligation caches to the original-program
  snapshot identity and revision. Stale queries fail conservatively.
- Fixed value availability immediately before its own definition and inside
  its defining region. Availability remains directional; legacy aggregate
  `DecodedRequirement::unresolved` may also mention the opposite endpoint.
- Integrated step 6 with the actual step-4 condition DAG, incoming reader roots
  and scoped projections. Conditional evaluation selects one arm; bypassed
  conditions need not be observable or evaluated.
- Made obligation universe tokens survive retained IDs, preventing reuse of an
  analysis address from accepting an old ID. Typed origins retain producing
  phase identity even when phases share an executable source cut.
- Distinguished unresolved marginal enumeration from a complete empty result;
  both marginal callbacks validate the same interval key. Missing native guard
  mappings use conservative marginal fallback.
- Replaced the obsolete standalone lifetime structure stub with a native MLIR
  fixture. Registered arithmetic, value, lifetime and obligation tests in lit.
  Test assertions remain active in release builds.

## Validation record

Native validation uses the installed LLVM/MLIR 19 toolchain and the existing
focused build under `.local/frontier-corpus`. Current Phase A, shared sync input,
PTO IR interfaces and test sources are compiled locally; unchanged dependencies
come from the existing donor build. This is not a clean top-level CMake build,
CANN compilation, device execution or performance measurement.

Executed during integration:

- Step 2: 27,941 interval checks, including 11,940 independent finite-execution
  comparisons; native structured-cut fixture.
- Step 3: 1,383,576 coverage checks; native full-write and RMW assertions.
- Step 4: 343 concrete executions against an independent scanner and a
  4,096-optional-reader formation check; native scoped/snapshot fixtures.
- Step 5: 1,981,828 arithmetic checks plus 64-bit boundaries; native value and
  snapshot tests; actual PTO scalar-producer/imported prerequisite checks.
- Step 6: portable obligation model with 256 reader valuations and a
  10,000-reader shared-DAG check; actual factored transfer against 12 concrete
  executions; stale ID, conditional bypass and unresolved-enumeration tests.
- Integrated native analysis: factored self-test (including the two-producing-
  phase typed prerequisite), coverage and TAXPY fixtures, and all five
  development kernels (two flash-attention components, two GEMMs, vector add).

Final integrated runs:

- Focused lit: **11/11 passed**, including interval, factored transfer, value,
  arithmetic, lifetime, obligation, full-write, TAXPY and development-corpus tests.
- Both actual pass modes on five development inputs plus the shared-coverage
  mode fixture: **12/12 passed**. Existing mode inserts synchronization and
  produces verified MLIR; FrontierSynch reaches its expected construction
  diagnostic. The coverage-mode FileCheck assertions also pass.
- Native lifetime fixture passes direct membership, incoming/replaced origins,
  lazy pair/subscription construction, support-before-pairs and loop fallback.
- AddressSanitizer and UndefinedBehaviorSanitizer pass both portable step-6
  executables. LeakSanitizer aborts under the sandbox's ptrace environment;
  the successful rerun used `ASAN_OPTIONS=detect_leaks=0`. Leak checking is
  therefore not claimed.
- `git diff --check` passes. No merge-conflict markers remain.

 Logs remain in the ignored
`.local/phase-a-patch-review/step2` through `step6` directories.

## Reproduction

With the native tools built through the repository CMake targets:

```sh
llvm-lit -j1 test/lit/pto/frontier_synch_intervals.pto \
  test/lit/pto/frontier_synch_factored_transfer_step4.pto \
  test/lit/pto/frontier_synch_obligations.pto \
  test/lit/pto/frontier_synch_write_coverage.pto \
  test/lit/pto/frontier_synch_write_coverage_modes.pto
bash test/FrontierSynch/run_interval_core.sh
bash test/frontier_synch/run_write_coverage_standalone.sh
bash test/standalone/run_frontier_obligations.sh
ASAN_OPTIONS=detect_leaks=0 CXX=clang++ SANITIZE=1 bash test/standalone/run_frontier_obligations.sh
```

The portable scripts use a disk-backed `.local/frontier-tests` scratch directory
by default and delete their own temporary run directory.

## Changed-code compliance scope

Applicable: C++17/ODS interface correctness, ownership and bounds, checked
arithmetic, CMake target wiring, shell quoting, license headers and test failure
handling. No Python production code, networking, release hardening or device
code generation is introduced by these patches.

The final changed-code prefilter inspected 45 files and reported 132 errors
(129 brace-pattern matches and three internal assertions) plus nine cast
warnings. These are contextual findings, not a zero-finding result.
The changed-code prefilter reports brace matches on nested/multiline expressions
and embedded TableGen C++, despite actual braced bodies after formatting.
All 129 flagged conditions were checked for their actual bodies. Its three
assertion matches are internal arena sort invariants; external qualification
failures use explicit conservative results. Pointer-to-integer casts encode
borrowed immutable IR identities only; they are not dereferenced or serialized.
These contextual checks follow the compliance skill's rule-resolution policy.
The prefilter is not claimed to pass without findings. External EChecker/SecK
and target-device validation were not run.
