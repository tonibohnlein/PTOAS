# M1 — Retained lifecycle and explicit capability upgrades

Baseline: `d6a920bf74785377998b35aa1c313198ce982e9b`. This milestone implements
the first batch of the [path-completion plan](frontier-sections56-path-completion-plan.md).
The [machine-readable receipt](frontier-sections56-path-m1-results.json) records
test names, binary and input hashes, construction counts and all timing pairs.

## Implemented contract

Completion attempts are scoped to retained evidence. A new child capability can
invalidate a refusal or refresh a parent snapshot. First successful completion
publication advances evidence once; subsequent refreshes do not cause alternating
parent/child invalidation. Generation rollover invalidates cached attempts.
Previously issued owners remain valid against the unchanged original source,
including after exact mathematics becomes available. New fallback selection
still cannot weaken retained exact mathematics.

Finite and eligible singleton-sequence producers export their qualified
completion-child capability. The parent no longer inspects their concrete
representation. The provider caches both successful and refused qualification;
this does not broaden the existing event-free child contract.

Finite, arithmetic, completion-child and session templates now instantiate from
one immutable `LogicalFamilyFragment`. Mutable inventories are fresh. Allocation,
generated SSA, private state and finalized protocols are never captured. Session
capture may retain immutable terminal choices without generated consumer guards;
mathematical ownership is restored on instantiation. Producer-specific resource
and invocation qualifications remain separate.

Mathematical-only sequence requests retain the compact repeated representation.
A synchronization request explicitly requests the existing endpoint-oriented
phase alternative, including flags in nested retained sequences. A refused
upgrade preserves the incumbent. Each successful alternative establishes its own
allocation evidence. This is one explicit capability upgrade, not a new search.

## Validation

Incremental native builds of `pto-test-opt` and `pto-sync-input-test` passed with
at most two workers. No LLVM rebuild. The focused lit gate passed **21/21**,
including finite pipeline/scaling, deferred contextual/cleanup output, arithmetic
exports, repeated region/storage oracles, completion storage/readiness and
serial/threaded multi-function insertion.

New regressions establish warm/cold phase upgrades with six-ID allocation,
weak-owner retention, stable repeated requests, generation rollover, cached child
qualification, new child evidence refreshing the parent, and old-owner preparation
and allocation after refresh. The mixed-parent fixture exercises alternating
proper/whole fallback scopes. The first test draft incorrectly expected the
`final_false` whole-function request to select fallback; it correctly selects the
exact native-scalar route. The test was moved, with no production-policy change.

Twelve geometric cases (four forms, sizes 4/8/16) passed logical matching,
physical reuse, flat/regional agreement where applicable, static guard and
construction bounds, evaluated-control accounting, and saved/combined lowering.
All measured constructed scalars and live added arithmetic equal baseline.
Each case checks four guard bindings. Three serial interleaved pairs at size 16:

| Form | Saved candidate/baseline | Combined candidate/baseline |
|---|---:|---:|
| Concrete | 1.019 | 1.010 |
| Guarded readers | 1.006 | 1.013 |
| Guarded writers | 1.010 | 1.000 |
| Guarded composition | 1.009 | 1.012 |

The committed scaling driver has a historical strict-improvement assertion for
any distinct binary hash. That assertion failed against this already-compacted
baseline. The local M1 harness leaves the committed tests unchanged, retains all
semantic/structural/evaluated-work checks and requires construction/live
non-regression in both routes. It additionally profiles baseline saved
materialization with the baseline trace tool: without that instrumentation,
its constructed count was unavailable (reported zero), not actual zero work.
These harness corrections are recorded rather than counted as compiler failures.

Three additional production cases used ordinary logical insertion plus physical
allocation, with one warm-up and three interleaved pairs each:

| Case | Candidate/baseline | Output comparison |
|---|---:|---|
| Regional completion | 0.999 | Byte-identical |
| Phase upgrade | 1.033 | Identical except recognition/selection diagnostics |
| Arithmetic regional | 0.996 | Byte-identical |

The phase cost includes retaining compact mathematics before the endpoint upgrade.
No speedup is claimed. The completion case uses the supported split route;
combined completion output remains unsupported in both binaries. No command or
physical-map differences were normalized away in the phase comparison.

Commands and full logs are under `.local/sections56-path-completion/`:
`m1-focused.log`, `m1-finite-regression/results.json`,
`m1-production/results.json`, and `m1-build-*.log`. The local comparison scripts
are hashed in the committed receipt. The complete synchronization gate and
49-module campaign remain the final-plan gate; no device tests were run here.

## Compliance and deletion

`git diff --check` passes. Changed-code compliance inspected C++/headers/CMake;
there are no new production scripts. The deterministic checker reports eight
`G.FMT.11-CPP` false positives: its regex terminates a condition at nested call
parentheses. All reported statements have braces. Locations are
`LogicalFamilyFragment.cpp:22`, `SequenceStorage.cpp:27,30`,
`CompletionRegionChecks.cpp:130,171`, and
`NormalizedControlChecks.cpp:346,352,601`. The actual long-line finding was fixed.
No suppression or checker change was introduced. No full static analyzer claim.

Removed duplicated finite/session template structures and arithmetic/child
inventory capture, and removed `finiteSequenceChild` plus parent-side inspection.
The specialized mathematical kernels and existing independent oracles remain.
Architecture reviewer `architecture_m1` and correctness reviewer `compiler_expert`
both granted final scoped acceptance after inspecting source and runtime receipts.
The correctness reviewer verified all 12 geometric inventory comparisons and the
local baseline-profile contract. Neither review claims a full-suite or device run.
