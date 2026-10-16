# Draft implementation closure

Baseline: compiler `ee181375`, paper `531eece` plus its October 16 working edits.
The existing exact demands, accepted order, matching and invocation contracts
remain the semantic constraints. Success, unsupported export, unresolved proof
and a checked fixed-protocol capacity bound remain distinct outcomes.

| Milestone | Deliverable | Acceptance | State |
|---|---|---|---|
| 1. Ownership | Module-owned allocation; common symbol-aware staging for saved recipes, contexts and executed counters | Threaded/serial multi-function agreement, globals/helpers, failure atomicity, existing traces | Implemented; focused validation passed |
| 2. Semantic gates | Independent wrapping and full arithmetic-membership oracles; resolve remaining broad failures | Complete fresh gate, with unfinished construction retained explicitly | Oracle repairs validated; four construction failures remain |
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

## Milestone 2 receipt (October 16)

The common independent membership oracle now checks arbitrary-width machine rows,
congruences and existential locals with certified finite bounds. Unsupported
local systems fail explicitly. A separate exhaustive oracle checks 7,000 bounded
systems; the existing quotient-local oracle also cross-checks the shared helper.
Captured loop-bound tests include signed 64-bit wrapping at both extremes. Tests
accept a newly supported arithmetic route only after occurrence/access/native
reachability or command-closure checks, rather than a producer-shape assertion.
The removed `nsw` fixture still rejects the nonwrapping repeated certificate.

Nine focused lit tests passed. Ten direct membership consumers also passed,
including 139,968 piecewise-bound, 280,500 guard and 458,392 program checks.
The fresh synchronization gate moved from **250 passed, 10 failed, one timed out**
to **258 passed, three failed, one timed out** (262 tests including the new oracle).
This is not an all-green gate: composed phases and repeated regions still time
out, rotating regions exhaust the bounded validation address space, and boundary
rotation still lacks an exact nonuniform deferred-crossing adapter. Those tests
remain unchanged and failing; they are construction gaps, not waived semantic
expectations. The committed JSON retains every outcome and the repaired source
hashes. Raw logs are `.local/draft-closure-20261016/m2-*`.

Only the existing native test targets were rebuilt incrementally. `git diff
--check` passed. The changed-code checker flags fully braced multiline conditions
in the two native test files; source review found no missing braces. No device
run, universal wrapping proof or complete draft-coverage claim is made.
