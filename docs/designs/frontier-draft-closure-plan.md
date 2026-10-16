# Draft implementation closure

Baseline: compiler `ee181375`, paper `531eece` plus its October 16 working edits.
The existing exact demands, accepted order, matching and invocation contracts
remain the semantic constraints. Success, unsupported export, unresolved proof
and a checked fixed-protocol capacity bound remain distinct outcomes.

| Milestone | Deliverable | Acceptance | State |
|---|---|---|---|
| 1. Ownership | Module-owned allocation; common symbol-aware staging for saved recipes, contexts and executed counters | Threaded/serial multi-function agreement, globals/helpers, failure atomicity, existing traces | Implemented; focused validation passed |
| 2. Semantic gates | Independent wrapping and full arithmetic-membership oracles; resolve remaining broad failures | Complete fresh gate, with unfinished construction retained explicitly | Oracle repairs validated; fresh 263-test gate has three construction failures |
| 3. Guarantees | Explicit order precision, reduction and realization contracts; qualified completion/child/loss evidence | Assertions and regressions at producer, composition and emission boundaries | Implemented for supported contracts; focused gate passed |
| 4. Allocation scope | Separate fixed-protocol certification from alternate-protocol research | Per-protocol evidence and outcome classifications; no implicit ordering changes | Implemented; three scoped witnesses rechecked |
| 5. Composition cost | Reduce attributed Delta incoming and phase relation construction | Same semantics, structural scaling checks, paired total-construction measurements | Qualified changes validated; remaining composition-size and watchdog failures stay open |

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

## Milestones 3–4 receipt (October 16)

A shared read-only guarantee view now serves retained mathematical outcomes and
prepared plans. The public pass serializes the same view before saved-recipe
sealing. It keeps modeled-order precision, cover quality and realization order
separate. Explicit adjacent local demands and the existing arithmetic adjacency
certificate can report direct order equality; nonadjacent local barriers can
strengthen order. Unsupported equality proofs, prefix matching and successful
allocation do not promote coverage to equality. Missing compact evidence stays
unavailable. The view retains its modeled-input owner, and failed exports do not
erase the mathematical guarantee or trigger new analysis.

Section 6 reports SCC-local placement, the qualified A3 completion protocol,
coarse storage classes, no additional parent event IDs, and distinct unique child
owners for event-free, completed-exit and closed-descendant contracts. Tests keep
pending primitive effects across calls and reject changed child seals. A general
realized-loss bound remains explicitly unavailable; this does not implement the
paper's unrestricted child import or numerical loss analysis.

The fixed-protocol capacity oracles now attach input/command-graph fingerprints,
admitted bindings and an outcome scoped to that protocol. Uniform assignment,
kernel compilation, alternative protocols and device qualification are separate
fields. Decode/prefill/Delta still require 9/20/46 directional IDs in the pinned
executions. The feedback control demonstrates that a changed protocol receives
a different graph identity. No new allocation or protocol replacement is claimed.

The existing native targets rebuilt incrementally; the 13-test focused gate
passed, covering guarantees, retained requests, completion children, closed child
seals, saved/combined output, threaded module allocation and capacity witnesses.
`git diff --check` passed. The checker flags the explicitly braced multiline
range check in `ResultGuarantees.cpp` as G.FMT.11-CPP; source review confirms its
body is braced. Local logs are `m3-*` and `check_*protocol*.out` under the receipt
directory. No device test was run.

## Milestone 5 receipt (October 16; qualified scope)

Reference-forward Boolean query projection now visits only the closed occurrence
interval from its source to its target. Earlier incoming edges cannot contribute
a path from that source and are not reconstructed. The original numeric answer,
endpoint presence, totality, dependency and availability checks remain intact.
A growing-prefix regression requires a four-occurrence query to visit exactly
four sites and three edges; the independent all-event-kind closure oracle
continues to check the resulting answers.

Arithmetic relation composition now lifts both inputs into the joined coordinate
frame and normalizes their conjunction once. It preserves aliases, fresh
coordinates, arbitrary-width rows and congruences. An independent membership
check covers 138,915 bounded assignments. This removes repeated normalization;
it does not replace the general relation algorithm or its potentially expensive
projection and difference operations.

Boundary-loop composition also retains the producer's ordered disjoint partition
through child export. Deferred GM effects can use that evidence only after
revalidating the original loop, arena and exact interval endpoints, and after the
existing distinct-visit storage-discharge proof succeeds. Overlapping or changed
endpoints retain the refusal. This repairs the missing symbolic-slice adapter
without introducing global expression assumptions or interpreting a necessary
domain as an exact activity guard.

The boundary regression now passes 56 ordinary and 64 multiple-cut unfolded
closure checks. Its affine/wrapping driver completes another 64 closure checks,
then fails the unchanged construction-size ceiling: 123,668 arena nodes versus
104,000. That final failure remains open. Passing these finite executions does
not establish general draft coverage or physical allocation feasibility.

The initial composition measurements predate the retained-partition repair and
are retained separately under `m5-measurements/`. They are not measurements of
the final source. The final-source measurements and complete gate are recorded below. No device
result or independent reviewer acceptance is claimed by this receipt.

### Attributed remaining composition work

A bounded debugger run of the large coprime rotating fixture catches its memory
allocation exception in `IntegerImplicationCache::subsumed` during general
arithmetic completion. The production stack reaches that construction through
`continueIdentityChild`; the independent sequence-analysis report already
succeeds. This is evidence to investigate dispatch into the compact sequence
provider before expanding another arithmetic closure, not evidence that the
physical allocator needs a larger event budget.

A separate 30-second sample of the first repeated-phase fixture is in
`RegionExpressions::RelationBuilder::conjunction`, reached through symbolic
regional sequence composition. The Boolean-to-integer conversion forms products
of alternatives before projection. This sample identifies a construction
boundary; it is not an exclusive timing breakdown or a completed analysis.
Replacing that conversion requires preserving its exact occurrence, machine
arithmetic and binding contracts. The rotating dispatch repair and this relation
representation problem are separate follow-ups.

### Final validation and measurement

The complete frozen-source gate passed **260 of 263 synchronization tests** in
1,462.75 seconds. It used one lit worker, a 4 GiB address-space ceiling and a
1,200-second per-test allowance. `sync_composed_phases.pto` passed in 607.90
seconds; its earlier 120-second aggregate timeout was therefore insufficient for
this multi-invocation test. This is not a planning-speedup result. Repeated phases
still fail their own 90-second subprocess watchdog, rotating insertion still
exhausts memory, and affine boundary construction still exceeds its unchanged
size ceiling. All expectations remain enabled.

The separate nonempty regular and sliced phase probes both pass their independent
payload-order oracle in baseline and candidate builds. Each has 97 events and
identical complete traces. Their concurrent diagnostic timings are not used as
performance evidence.

Final serial Delta timing pairs, in seconds, are **10.492/10.897**,
**12.696/11.599**, and **12.898/17.317** (baseline/candidate). The median paired
change is a 3.86% slowdown, with substantial variation; no total speedup is
established. Complete output IR is byte-identical. Projection occurrence visits
fall from 17,530 to 15,484 and edge visits from 301,332 to 245,957; constructed
prefix operations remain 311,732 and full-function arithmetic remains 313,274.
The final single phase pairs are 13.147/17.020 seconds for the regular case and
49.398/35.497 for the sliced case. These single pairs do not establish a general
performance improvement. The sliced timing input has zero trips; its separate
nonempty oracle above supplies the additional semantic check.

Nine initial focused tests passed, including independent query closure and
integer membership. The retained-partition checks pass, including refusal of
stale endpoints; the boundary lit file remains failed only at its size ceiling.
The configured native targets were built incrementally with at most two workers.
`git diff --check` passes. The changed-code checker reports nine G.FMT.11-CPP
matches on explicitly braced compound conditions; source review found no missing
braces and no new compiler warning. Raw receipts and tool/source fingerprints
are retained in `frontier-draft-closure-results.json` and the local receipt tree.
