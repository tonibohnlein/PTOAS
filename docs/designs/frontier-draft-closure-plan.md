# Draft implementation closure

Baseline: compiler `ee181375`, paper `531eece` plus its October 16 working edits.
The existing exact demands, accepted order, matching and invocation contracts
remain the semantic constraints. Success, unsupported export, unresolved proof
and a checked fixed-protocol capacity bound remain distinct outcomes.

| Milestone | Deliverable | Acceptance | State |
|---|---|---|---|
| 1. Ownership | Module-owned allocation; common symbol-aware staging for saved recipes, contexts and executed counters | Threaded/serial multi-function agreement, globals/helpers, failure atomicity, existing traces | Implemented; focused validation passed |
| 2. Semantic gates | Independent wrapping and full arithmetic-membership oracles; resolve remaining broad failures | Complete fresh gate, with unfinished construction retained explicitly | Oracle repairs validated; final continuation gate passes 262/263, with one existing repeated-phase watchdog |
| 3. Guarantees | Explicit order precision, reduction and realization contracts; qualified completion/child/loss evidence | Assertions and regressions at producer, composition and emission boundaries | Implemented for supported contracts; focused gate passed |
| 4. Allocation scope | Separate fixed-protocol certification from alternate-protocol research | Per-protocol evidence and outcome classifications; no implicit ordering changes | Implemented; three scoped witnesses rechecked |
| 5. Composition cost | Reduce attributed Delta incoming and phase relation construction | Same semantics, structural scaling checks, paired total-construction measurements | Qualified changes validated; rotating and boundary gates pass, repeated-phase relational expansion remains open |

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

## Milestone 5 continuation: request ownership and residual compaction

The whole-invocation identity adapter now reuses an adequate retained child but
never starts a nested same-region backend search. A cold miss resumes compact
Sequence construction, leaving further candidate selection to the outer request.
When that loop needs general arithmetic, the outer request also owns its export
requirements: a demand-only request no longer builds storage selectors merely to
create an identity wrapper. Genuine regional children retain their complete
composition interface. The existing ownership, query-upgrade and fresh-plan
checks remain enabled.

The compact route exposed two exclusive readiness alternatives that previously
reached a different producer. Shared endpoint normalization now proves exclusion
from a bounded slice of unsigned equality/strict-inequality guards, in addition
to producer-owned residue domains. It compares identical scalar identities,
keeps exact guards and matching tuples, and never merges simultaneous commands.
The new guard optimization requires a frozen family protocol; legacy concrete
exporters that append private lanes after preparation retain their existing
grouping across allocation requests. Review found that index literals retain
64-bit attributes on narrower targets; the new proof therefore also requires
a non-scalable index64 layout at the original cut. A real index32/index64
recipe-insertion regression checks that coincident truncated literals preserve
two active commands. The 64-node discovery bound limits optional compaction
work only; incomplete proofs retain the original commands. Finite coordinate-map
normalization runs first so a pairwise merge cannot prevent an already-supported whole-family map.
The asymmetric fixture retains one producer SET and two consumer WAIT sites.

The interval helper is checked against all 256 byte values for 576 comparison
pairs (147,456 evaluations), including zero and maximum bounds. Existing tests
also retain coactive commands, distinct matching tuples and intervening command
order. Sequence queries for the same selected occurrence now use exact presence
and the native start-to-completion edge, without requesting a child rank circuit;
all four event-kind pairs are checked with present and absent occurrences.
This shortcut alone does not reduce the boundary fixture's size.

The initial ten-test focused gate passed, including prepared protocols, family
emission, combined and deferred output, candidate selection, identity ownership,
sequence queries and rotating traces. A second complete run exposed legacy
allocation-lane grouping and outdated callback/static-site expectations. The
legacy path now retains its grouping; the native-event oracle still checks all
truth values, with no child callbacks needed. Profiling checks reported sites
against actual IR rather than requiring equal static SET/WAIT counts; the
separate repeated-crossing oracle still checks unfolded matching and order.
The follow-up eleven-test gate passes these repaired checks. The rotating driver
adds zero-trip and nonempty 13/11-bank closure cases. The initial dispatch-only complete run still had three failures:
it repaired rotating construction but exposed the family-site regression fixed
above. That initial result is retained separately from the final gate.

Before the final incoming-path change, Delta's executable function body is
byte-identical to the preceding commit;
313,274 arithmetic operations and all command-site counts remain unchanged.
Full-file equality does not hold: recognition-diagnostic ordering, location
aliases and request estimates differ. Request walks decrease from 17 to 15.
This was a concurrent output diagnostic, not a paired timing experiment, and
establishes no device or total compiler speedup.

### Shared incoming-path output

The boundary probe attributes most growth to alternative-path queries in crossing
reduction: 8,453 arena nodes at symbolic-reducer entry, 10,903 after endpoint
folding/consolidation, and 123,668 after closure and boundary export. The probe
counts 10,482 prefix requests and 10,189 suffix requests; all 20,671 incoming
requests use the residual query route. These are construction counters, not
exclusive timings.

Incoming path queries now retain private activity-scoped answers and apply one
activity mask to their combined output. The exact activity already includes the
candidate, entry and distinct-identity predicates; the caller no longer masks
those predicates again. Public single-query answers remain activity-masked and
cached by activity and original occurrence pair. No restricted answer is exposed
as an unrestricted query.

Prefix qualification still precedes suffix qualification. Review found that both
outputs must retain the existing per-output select simplifications: a prefix
that is false under activity must skip an unavailable suffix, and an opposite
branch that cannot execute must not remain as an availability obligation. The
final implementation shares the expression service's existing one-level select
arm fold, including negated conditions. It changes only local IDs; raw query
caches remain unchanged. The combined output retains its activity mask. This
creates no temporary expression or proof transaction.

Tests cover both activity polarities, unavailable queries, conditional-zero
prefixes and suffixes, removal of unavailable opposite-arm inputs, inactive/
unbound outputs and a false suffix with an unbound prefix. The first full-gate
attempt after the focused path tests was stopped for this review repair; it is
not counted as completed validation.

Removing caller masks alone reduced the boundary arena to 113,745 nodes. Shared
path output reduces it to **103,881**, below the unchanged **104,000** limit.
The complete boundary driver passes 56 ordinary, 64 multiple-cut and 64 affine/
wrapping closure comparisons. The final seven-test focused gate passes, including
the new path-query checks, public pipeline and family materialization tests.
Both independent source reviewers accepted the corrected implementation within
this scope. Full-gate and Delta replay receipts follow below.


### Final continuation measurements

Three serial interleaved Delta logical-compilation pairs are **11.749/12.396**,
**12.345/12.346**, and **12.698/12.146** seconds (baseline/candidate). The median
paired change is approximately **+0.005%**; total compilation time is effectively
unchanged in this measurement. The baseline is the preserved index-width-fixed
build immediately before the incoming-path changes, not the previous commit.
The final output has **313,258** arithmetic operations versus **313,274** and
93,166 selects versus 93,182. Static command counts remain 172 SETs, 47 WAITs
and 85 barriers. This is a small Delta output reduction, not a new major Delta
speedup or a device result.

The final fixed-execution replay preserves all **72 payloads and 358 commands**,
including matching and command causality. The same scoped 46-ID witness remains;
this change does not make that fixed protocol fit six IDs. Review acceptance is
for the source transformations and their stated contracts, not universal draft
coverage or device qualification.


The final frozen-source synchronization gate passed **262/263** tests in
**727.34 seconds**, with one worker and the unchanged resource limits. Its sole
failure is `sync_repeated_region.pto` at the existing 90-second phased-repeat
subprocess watchdog. The earlier rotating memory failure and boundary size
failure pass. No test expectation or timeout was relaxed for this continuation.
All 16 changed production/test files match the pinned build source fingerprints.

The incremental build and seven-test focused gate passed. `git diff --check`
passes. The changed-code prefilter reports 22 G.FMT.11-CPP matches; source
inspection confirms every cited control body is explicitly braced (zero reported
warnings). These are classified prefilter false positives, not a claim that an
unavailable external static analyzer passed. Both independent reviewers gave
scoped source acceptance after the index-width and inactive-arm repairs.
The machine-readable receipt retains intermediate results and the interrupted
pre-repair gate without counting that interruption as a completed run.
