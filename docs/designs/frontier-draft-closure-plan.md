# Draft implementation closure

Baseline: compiler `ee181375`, paper `531eece` plus its October 16 working edits.
The existing exact demands, accepted order, matching and invocation contracts
remain the semantic constraints. Success, unsupported export, unresolved proof
and a checked fixed-protocol capacity bound remain distinct outcomes.

| Milestone | Deliverable | Acceptance | State |
|---|---|---|---|
| 1. Ownership | Module-owned allocation; common symbol-aware staging for saved recipes, contexts and executed counters | Threaded/serial multi-function agreement, globals/helpers, failure atomicity, existing traces | Implemented; focused validation passed |
| 2. Semantic gates | Independent wrapping and full arithmetic-membership oracles; resolve remaining broad failures | Complete fresh gate, with unfinished construction retained explicitly | Pending |
| 3. Guarantees | Explicit order precision, reduction and realization contracts; qualified completion/child/loss evidence | Assertions and regressions at producer, composition and emission boundaries | Pending |
| 4. Allocation scope | Separate fixed-protocol certification from alternate-protocol research | Per-protocol evidence and outcome classifications; no implicit ordering changes | Pending |
| 5. Composition cost | Reduce attributed Delta incoming and phase relation construction | Same semantics, structural scaling checks, paired total-construction measurements | Pending |

Use the configured incremental build, never a clean rebuild for a milestone.
Local intensive work is limited to two aggregate workers. Historical results
remain in the existing receipts; new runs are pinned under
`.local/draft-closure-20261016/`. Device execution is a separate qualification.

## Milestone 1 receipt (October 16)

The allocation pass now owns a module transaction. Saved-family, execution-context
and executed-counter staging use the same transitive symbol dependency closure;
function workers no longer read and clone sibling bodies. Explicit textual pass
pipelines must place `pto-frontier-allocate` at module scope. A failure in any
function leaves the original module unchanged.

Validation used the existing pinned incremental build (two workers), rebuilding
`pto-test-opt`, `pto-sync-input-test` and the compiler library. The new regression
passed directly with affinity restricted to two CPUs: four functions per staging
path, repeated threaded/serial comparisons, parallel independent modules, empty
budgets and a malformed certificate in the final function. The shared staging
unit check includes a transitive helper and global and excludes an unrelated body.

The focused lit gate passed all 16 affected tests. Its initial run had two Python
ABI import failures in the CLI, before compilation; both passed after placing the
configured Python 3.12 environment on PATH. No source expectation was weakened.
`git diff --check` passed. The code checker reported two braced multiline `if`
conditions as G.FMT.11-CPP; source review confirmed both have explicit braces.
There was no device execution or thread sanitizer run. Local raw logs are under
`.local/draft-closure-20261016/m1-*`.
