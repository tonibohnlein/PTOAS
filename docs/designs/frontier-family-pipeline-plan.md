# Compact command-family pipeline

This revision supersedes the execution priorities of
[the consolidation plan](frontier-consolidation-plan.md), while carrying its
unfinished obligations forward. The historical implementation baseline is
`9e86b5232b92cf9b4efe27464a061d78a2ca7406`. The active revision below is pinned to
implementation `34b7ed8119f95e3a0b612a630ce7066c3c5296e2` and the October 14 draft
at `b5a5df8` in `synchronization_draft`. Older results below remain historical
receipts, not evidence that the new milestones are complete.

## Contract

Keep the existing mathematical providers and replace qualifying internal
algorithms with the revised draft's stronger constructions. Build executable command-family
recipes, normalize their logical protocol, request allocation for that protocol,
then materialize generated arithmetic and commands transactionally. These are
internal stages, not additional public passes or a new dialect.

Preserve the selected payload ordering, original hardware restrictions, and
retained exact mathematics on export failure. No ordering repair, new admission
cutoffs, general symbolic solver, runtime demand analyzer, or weaker fallback
after exact retention is introduced. Necessary activity domains remain one-way
exclusion evidence; they do not replace exact guards or matching identities.

## Active execution plan: revised Sections 5–8

This execution order supersedes the earlier post-rebase priorities. Retain the
completed family recipes, shared occurrence/availability service, demand-driven
shared bindings, mixed regular/residual batches, complete-map normalization,
owned allocation requests and parent retry. Deferred contextual singletons are
not a compactness guarantee. Do not add another dispatcher backend for an
internal algorithm specialization.

The scope of "implement Sections 6 and 7" here is the SCC completion policy,
closed-child integration, and applicable order-preserving allocation and span
algorithms. It does not include the section's ordering-changing repairs,
unrestricted protocol synthesis, or an assumption that every symbolic result
admits a cheap allocation. Theorem 7.23's cross-lane reversal issue remains a
draft obligation; its excess formula must not authorize a production decision.

| Step | Reviewable work | Completion gate | Existing obligations |
|---|---|---|---|
| R1 (implemented) | SCC-ordered completion transfer | Concrete pending-state oracle, fixed-child containment, bounded fact propagation pass | M6; Section 6 |
| R2 (native closure implemented; pair scores open) | Core-chain periodic frontier specialization | Exact demand/query agreement and linear scan/relaxation scaling; scalar scaling costs remain explicit | M1/M3; Section 5.4 |
| R3 | Writer-centered guarded storage and shared boundary scans | Exact conflict/boundary agreement, including macro envelopes and protected effects; read-heavy scaling | M1/M3; Section 5.2 |
| R4a | Ordered periodic/repeated first-reuse sweeps | Certified applicability, linear query counts, exact capacity and finite-restriction checks | M2/M4; Propositions 7.8/7.10 |
| R4b | Linear periodic allocation from suffix thresholds | Independent command-graph reuse/width oracle; no general assignment matrix on this route | M2/M4; Proposition 7.12 |
| R4c | Direct modulo-lane span extraction and qualified boundary export | Exact span/presence agreement without generation expansion; parent retry preserved | M2/M4; Proposition 7.13, Corollary 7.14 |
| R5 | Binding-aware multi-output control construction and producer query slicing | Less total constructed/live control, unchanged matching and payload order; Delta lifetime gate separate | M3/M4 |
| R6 | Normalized logical graph allocation, owned session and closed-child completion import | Shared unified/split stages, allocation before scalar materialization, deleted superseded paths | M3/M4/M5/M6; Sections 6/8 |
| R7 | Pinned pipeline, scaling and device closure | Distinct logical/physical/C++/device outcomes and reproducible receipts | M1/M6 |

R1 transfer is implemented; R2–R7 remain open. Existing primitives are starting
points, not completed migrations. Continue with R2, R3, R4a–c, R5, R6, then R7.
Run focused integration checks at each step; do not defer all production
validation to R7. A newly measured bottleneck may justify resequencing, recorded
with evidence rather than another untracked fallback.

### R1: SCC-local conservative transfer

Replace the global two-solve placement in `CompletionTransfer.cpp` with the
Theorem 6.2 policy. Preinstall mandatory prerequisite completions. Compute
reachable SCCs and process their condensation graph in topological order.
An acyclic singleton transfers finalized incoming facts once. A cyclic SCC
holds external inputs fixed, solves locally, marks conflicting entries, then
solves locally again. Publish only final outputs to successor components.
Reuse the mode-tagged delta worklist and existing summary contract; no change
to completion protocol legality or physical flag ownership is implied.

Keep the old global policy in test support for the stated containment check,
not as a second production policy. Compare selected sites and payload closure
only with identical child code, effects, mandatory entries and protocols.
Use a separate concrete pending-state oracle for summaries and coverage.
Include write/read/read, joins, self-loops, multi-node SCCs, unreachable nodes,
mandatory entries and final-false while visits. Count SCC visits, solves and
internal/inter-component fact propagation. No completion may be dropped merely
because another child decomposition selected fewer sites. This step does not
permit conservative replacement of retained exact mathematics.

### R2: core-chain periodic frontiers

Specialize `GuardedPeriodicQuotient` inside its current provider. Preserve
parallel-record normalization, winner/provenance rules and query interfaces.
For certified core chains, build ordered potential-site lists per pipe; compute
masked native closure by prefix/suffix scans including period wrap and payload
start-to-completion edges. Seed each required origin kind, then perform at most
`k` demand-edge rounds, each followed by complete native closure. Retain the
generic supplied-graph path when additional native semantics lack this contract.
Changing only the old relaxation round limit is invalid.

Use shared last-edge exclusion scans. Charge requested all-pair exports
separately; do not construct them merely to answer sparse queries. Evaluate
scaled `(period, rank)` frontier representation within the same interface to
avoid multiplication by pipe length, with explicit infinity, signed bounds and
overflow checks. A representation change is a separate reviewable part if
needed; the core native-closure migration must not depend on an unchecked
integer encoding.

Compare against independent shortest-path/unfolded graph oracles and the old
algorithm on small numerical and guarded inputs. Cover absent sites, empty
pipes, wrap, parallel ties, zero-distance edges, both query origins, partial
periods and zero trips. Scale sites/records at fixed pipes and count native
scans, demand relaxations, unique gates and exclusion work. The target is
Proposition 5.17's supplied-description bound, not a new general guarded
periodicity recognizer. Independently changing iteration guards do not qualify.

### R3: writer-centered guarded construction

Retain the guarded rank reducer. Replace qualifying finite storage input and
boundary construction with one shared per-cell ordered-access builder:
potential writers drive candidate generation, intervening-write predicates are
shared, equal endpoint witnesses are merged, and prefix/suffix scans build
writer/reader boundary selectors. Measure `c`, `W = sum(w_x * m_x)`, candidate
visits and gates. A read-only cell must construct no internal storage-conflict
candidates and must still export its reader boundaries in linear scan work.

Qualification itself must not retain a hidden all-access-pair loop. Separate
residual alias/conflict work and hardware-protected cases explicitly. Process
simultaneous macro envelopes against the same preceding state before updating
it. Preserve scalar-pipe restrictions and accumulator/tile-shape protections.
Use the shared builder for finite and qualifying bounded-lifetime consumers;
retain general paths where their stronger semantics are needed. Do not infer
an impossible write through an unrequested satisfiability proof.

Test read-heavy and all-writer cases, RMW, guarded absence, repeated witnesses,
macro envelopes, alias residuals and protected conflicts. Compare generators,
reduced demands and boundary queries with independent explicit executions.
Attribute Delta's large roots to producers before claiming this closes them.

### R4: specialized Section 7 allocation

Extend the existing allocation request with a certified first-reuse capability:
direction, ordered publication/consumption phases, exact suffix thresholds,
admitted context, finite-restriction evidence and owning query/index lifetime.
Do not introduce a second program representation. Canonical handoffs and
contracted child spans need different applicability evidence.

R4a precomputes endpoint-to-port projections and uses the ordered monotone
sweep/direct cyclic assignment where Propositions 7.8/7.10 apply. Both success
and insufficient-capacity results use the stronger algorithm. Zero-capacity,
empty and short invocations remain explicit cases.

R4b implements Proposition 7.12's residue/permutation construction from supplied
first-reuse positions `t_r = c*a_r + b_r`. Validate the suffix contract and
required reachability/finite thresholds before computing its minimum. Retain
the general assignment solver for arbitrary sufficient-link matrices; such a
matrix is not automatically an exact suffix-threshold description. Test
unordered matching, tied residues, large distances and malformed descriptions
against a separate command-graph oracle. Charge threshold construction
separately from linear allocation.

R4c derives first/last lane occurrences from certified modulo rules over
interval domains using gcd compatibility, modular inverses and interval
rounding. Compare original command positions, including within-cut order.
Split holes only through supported exact phase descriptions. Handle empty,
short and clipped intervals without enumerating their generations. Extend
qualified rotating-child boundary certificates with lane endpoints, charging
numerical residue expansion explicitly. Other children retain supplied span
selectors. Neither storage extrema nor lane endpoint extraction proves reuse.

All three parts preserve parent-aware candidate retry, reservations, direction
budgets and zero/partial-invocation obligations. Compare complete executable
ID maps among independently certified candidates and retain the incumbent on
unproved or unprofitable rewrites. No candidate may buy cheaper control by
adding payload prerequisites. Include a shared command-semantics regression:
an outgoing SET textually preceding an incoming WAIT does not itself establish
a causal SET-to-WAIT edge; test the local-barrier case separately.

### R5: shared control construction and Delta

Extend the common expression service with multi-output requests under one
arena, original cut, occurrence translation, history environment and private
state version. Slice to the union of live roots before scalar creation. Reuse
subexpressions only with equal bindings and legal placement. Original cut
identity alone is not a sufficient key; snapshots, resets, absent values,
rollback and temporary expression identities remain ownership boundaries.

Distinguish repeated sub-DAGs across roots from one already-large reachable
root. The pinned Delta profile has 294 distinct emissions and one regional
expression constructing 600,293 operations; whole-root caching is insufficient.
Attribute this root to query construction and coordinate/guard composition.
When endpoint maps are supplied, use the draft's restricted incoming queries
(`app:incoming-cut-queries`) inside existing producers where qualified: latest
incoming source per pipe plus preceding executed payload, with shared queries.
Their O(k^2) query count does not bound arbitrary query or selector complexity,
and does not prove availability at the source cut. Preserve early publication,
cleanup and the chosen protocol rather than introducing a runtime analyzer.

Acceptance measures unique reachable mathematical nodes, constructed scalar
operations, live lowered operations and dynamic evaluated control separately.
Test sharing and non-sharing across history versions, zero/empty/nested
invocations and guarded absence. Compare active communications, matching,
cuts and payload closure, not only command counts or output numerics.
Delta additionally requires an actual invocation-prefix lifetime export and
finite-ID reuse proof. If the fixed protocol cannot be certified, report that
separately; compact logical code is not physical success and no modulo-ID guess
or silent conservative substitution is allowed.

### R6: finish the retained-plan boundary

Normalize recipes and their coactive command order before allocation; consume
matching metadata without scalar materialization. Allocate the resulting
logical protocol, compose certified physical maps with occurrence selectors,
then lower its live control once. Preserve the current split-pass artifacts and
make unified and split entry points use the same stage functions. Saved
artifacts must be self-contained and versioned, with no borrowed callbacks.
Keep effect precision, cover reduction and realization-order guarantees
distinct in capability results. Delete superseded producer orchestration as
each migration completes; do not discard specialized mathematical kernels.

Complete Section 6's closed-child import with sound effects and executable,
closed matching across the imported whole invocation. An isolated iteration
with open inter-iteration handoffs is not a closed child. Preserve selective
child code without requiring arbitrary reachability/storage selectors or
draining at every boundary. Physical certification must include inter-visit
reuse and completion-protocol resources. Add the coarse shared-input path so
qualified completion requests do not first require fine physical partitioning.
Keep unsupported calls, scalar readiness, mixed-core and hardware protocol
restrictions explicit.

### R7 and remaining carryover

Refresh the eleven-kernel campaign with pinned planner, allocator and backend
revisions, flags and hashes. Report logical insertion, mathematical retention,
physical certification, C++ generation and device execution separately. Keep
GroupNorm/decode capacity certification, prefill allocation export and Delta
prefix lifetime failures distinct; refusal is not a lower bound above six IDs.
Close phased-repeat/corpus and semantic-oracle gaps as well as focused tests.
Keep source-fact, relational crossing, capture-closure and rank-transport work
from M3/M6 on the ledger until production callers and tests establish closure.

Run structural scaling on loop counts, banks, read/write ratios, SCC shapes,
lane/phase counts, bindings and record-label permutations. Retain the pipeline
inventory: records -> families -> query/proof work -> prepared/materialized
arithmetic -> live control -> static sites -> executed communications/control.
Use paired compiler timings before claiming speedup; device claims require
device runs. Hardware restrictions and exact-order acceptance remain fixed.

The fused explicit scan/reducer is a subsequent task inside the existing
explicit provider, primarily to reduce working storage. Keep materialized
generators for requested witnesses, supplied-graph inputs and test oracles.
It does not block R1–R7. The upstream dynamic-subview row proof integration
remains explicitly deferred, as previously agreed.

Use the existing configured build incrementally, targeting affected objects,
test drivers and `ptoas`. Do not reconfigure, reinstall or rebuild LLVM for
routine milestones. Run focused tests once after each meaningful change and
broaden only for integration risks or failures; at most two aggregate intensive
workers. Record each milestone's code, contracts, deleted paths, test results
and remaining limitations before a reviewable commit. This planning revision
does not restart builds or implement any R step.

## Existing milestone ledger (carried forward into R1–R7)

| Milestone | Work and acceptance | Current status |
|---|---|---|
| M0: preparation ownership | Reproduce prefill failure; remap every affected non-SSA preparation reference before deletion; validate complementary cleanup and failure atomicity; replay prefill separately from the failing baseline. | Complete: `32b38b426` |
| M1: measurement | Extend stage inventories with per-repeat lanes/extrema, candidate pairs/gaps, query construction, proof/cache work and actual traversal counts. Keep static records/sites distinct from dynamic commands and evaluated control. | Stage inventories, repeated-allocation work/traversal counters and concrete trace arithmetic/select/guard visits implemented; wrapper-discovery inventories implemented; complete ID-map cost attribution remains open |
| M2: repeated allocation | Normalize certified ordered extrema; share attempt-owned relative queries; try economical certified numerical/repeated assignments before general sharing, retaining alternatives needed by parent allocation. | Certified extrema scans, exact-activity deduplication and attempt-owned relative queries implemented; owned independent/shared candidate requests with parent retry implemented; explicit-capacity selection is available through the session and analysis-pass resource request; complete mapping-cost policy remains open |
| M3: family construction and binding | Plan availability and publication/cleanup before allocation; share occurrence/frame/capture/reset interpretation; migrate sequence and repeated crossings to shared recipe families before scalar lowering. | Numerical/rotating root recipes migrated in `a2ae91269`; shared occurrence-coordinate binding and availability recipes implemented; finite guarded publication/cleanup is now retained entirely in shared family recipes; ordinary numerical sequence and repeated children retain recipes through composition; exact fixed-coordinate sequence crossings now form recipes before scalar preparation; child activity masks retain those recipes through composition; repeated crossings now retain numerical or contextual recipes before scalar lowering; guarded sequence crossings now retain publication/cleanup and common-cut expression recipes; finite overlays and guarded/numerical periodic adapters retain recipes through allocation; relational crossings and capture closure remain open |
| M4: composed allocation and lowering | Allocate the normalized command graph; compose incoming selectors with independently certified ID maps; compare complete mapping costs; emit only live control at legal scopes. | Existing late normalization/candidate comparison retained; common nonconstant tuple terms compose before dispatch in `15f2007ea`; coordinate-selected phase shifts now compose under the same skeleton with a cost-checked incumbent; different skeletons and early graph allocation remain open |
| M5: session and capabilities | Retain facts, mathematics, recipes and allocation in one session; deterministic capability upgrades; common unified/split stage functions; self-contained versioned saved artifacts; delete redundant orchestration. | Explicit-capacity allocation requests now reuse the retained session and identity owner; capacity-specific answers are isolated from unrestricted exports. Unified physical materialization and remaining orchestration are open |
| M6: carryover and closure | Finish shared-fact and coarse-input integration, closed-child completion imports, guarded/relational rank transport and outstanding semantic oracles; run pinned eleven-kernel/corpus validation and report device results separately. | Eleven-kernel replay refreshed at `c088495d8`: 11 logical, 7 six-ID/C++ successes with the pinned planner/allocator and fixed retained C++ backend; corpus, device, and interface closure remain open |

Each milestone proceeds through focused, separately reviewable changes with
regressions and correctness/architecture review. Infrastructure does not complete a production
migration. A migrated producer must use the shared path and remove its redundant
preparation implementation. The late compactor remains only for explicitly
unmigrated producers.

## Refinement: fixed-demand control compilation

Minimum demands and maximum concurrency are hard requirements. The secondary
objective reduces construction and executable-control work while preserving
active communications, matching, cuts, coactive command order and invocation
behavior. Physical IDs may change only under an independently valid certificate.
This does not introduce command aggregation protocols or added-order repair.

Persistent's banking periods, demand translation structure and certified ID
periods are separate objects. Preserve the existing nested structure and boundary
cases; do not infer a universal period two. Regular extrema may use shared
first-present/last-present scans only under certified reference order. Mixed
completion directions retain every bound needed for causal reuse. Unresolved
order keeps the general guarded representation.

A cheap repeated assignment is usable only when the combined request fits,
including crossings and reservations. Preserve access to stronger sharing when
a locally feasible child assignment prevents its parent from fitting. Buffer
parity, storage extrema and emitted-WAIT counts are not reuse certificates.

Regional migration first separates availability planning from materialization:
publication, acquisition and cleanup must be settled before allocation. Existing
recipes retain child references and coordinate transformations. Exact activity
remains distinct from necessary exclusion evidence. Complete finite-domain
checking may compact supplied points; it does not justify extrapolation.

Extend complete-map normalization within the existing small candidate policy.
Missing optimization proofs retain the incumbent. Liveness includes cleanup,
state and immutable non-SSA evidence, not only SSA command operands. No new
solver, admission cutoff, work budget or public pass is introduced.

## Interfaces and ownership

The pre-materialization family retains original cuts and invocation coordinates,
exact activity, executable matching/selection recipes, necessary exclusion
domains, coexecution order, mathematical ownership and record provenance.
Original SSA inputs may be borrowed from the unchanged source; generated
detached SSA values must not be the recipe interface.

Direct, prefix and completion protocols remain distinct. Their matching,
publication, acquisition and cleanup behavior is fixed before allocation. One
coordinator owns matching for every selected prefix direction. A private-state
recipe is materialized once per invocation.

Allocation evidence belongs to the actual logical protocol. A change to that
protocol rebuilds affected evidence unless preservation is established. A
cheaper expression for one assignment is different from a new assignment: the
latter needs its own reuse proof. Logical-only requests build no allocation
evidence. Existing inexpensive certified candidates and cost weights remain the
starting policy; no unbounded candidate search is added.

The split public passes use the same stage implementations as the owned
production session. Serialized artifacts cannot retain borrowed pointers or
callbacks. Exact export failure retains mathematical ownership and reports its
specific unmet obligation.

## Verification

Test regular nested loops with varying trips and banks, record-label
permutations, exclusive and simultaneous commands, tuple maps, terminal cleanup,
zero-trip/skipped/nested invocations, and just-over-capacity allocation. Compare
payload order, matching and legal reuse separately from numeric ID labels.

Keep construction cost, executable-control cost and ordering quality separate.
Raw arithmetic includes dead code; static sites are not dynamic commands or
live notifications. Profile-on/off must preserve plans and certificates.
Claim compiler speedups only with at least three serial baseline/candidate pairs
without overlapping builds, reporting all samples and median/range. Device
latency comes from separate runs, not source counts or an interpreter.

Use at most two aggregate local intensive workers. Durable receipts belong in
the disk-backed workspace. Device-task documents remain outside the repository.

## Baseline evidence

The serial eleven-kernel MayNotAlias campaign at the baseline has ten logical
successes and seven six-ID allocations/C++ outputs. Prefill crashes during
terminal-cleanup validation. GroupNorm and decode lack a certified six-ID
assignment; Delta lacks an invocation-prefix lifetime export. Those diagnoses
are not interchangeable.

Persistent retains 1,544 record labels and emits 56 SET/60 WAIT sites. Delta emits
2,473,273 raw arithmetic operations. Neither count establishes dynamic cost or
physical capacity. Receipts and complete control inventories are retained at
`.local/generated-control-9e86b5232/`, including `REPORT.md`, `metadata.json` and
`results.json`. These single-run timings are not paired speedup evidence.

## M0 result

Preparation CSE replaced SSA uses but left an immutable terminal-cut choice
holding an erased predicate. The repair requalifies that evidence with its
replacement value before deleting redundant operations. The regression covers
both consumer-active and cleanup-active executions. Prepared insertion and
guarded publication lit tests pass; the assertions-enabled native build adds no
warnings. The compliance prefilter reports two brace false positives on braced
conditions in the new regression.

Prefill now succeeds logically in 85.909 seconds with 1,035,368 KiB peak RSS.
Six-ID allocation still refuses because the finite/uniform allocation export is
missing. This is neither a capacity lower bound nor a compiler-time improvement.
The original failing receipt is unchanged. Repair receipts are under
`.local/family-pipeline-m0/`.

## M1 stage inventories

`profile-stages=true` now emits aggregate `frontier-control-profile` records at
preparation, normalization and insertion. They count provenance families and
record members separately from endpoint sites, detached arithmetic and its
structural endpoint reachability. Function arithmetic explicitly includes
payload arithmetic. Private-state paths report their staged invocation rather
than pretending that an empty ordinary endpoint list means no generated work.
These are static inventories, not executed-instruction counts or proof costs.
The two focused profile tests establish unchanged public IR and diagnostics.

A Persistent GEMM probe at `f8917325d` reports:

| Stage | Record members | Endpoint sites | Detached arithmetic | Endpoint-reachable arithmetic |
|---|---:|---:|---:|---:|
| Prepared | 1,544 | 3,088 | 20,257 | 20,257 |
| Normalized | 1,544 | 116 | 18,283 | 11,468 |

Insertion emits 56 SET sites, 60 WAIT sites, one barrier, and 18,342 arithmetic
operations including payload arithmetic. Thus the late reduction in command
sites still follows substantial preparation; it is not demand elimination.
The 9.015-second diagnostic run is not a paired compiler-time result. Receipts
are `.local/family-pipeline-m0/persistent-control.{json,out,err}`. Traversal,
dynamic-control, and complete ID-map metrics remain outstanding.

## Earlier numerical/rotating migration (now M3)

`a2ae91269` splits the existing numerical family construction from scalar
materialization. The opaque immutable recipes retain validated coordinate maps,
ordered pieces, original cuts and record provenance. Affine/modular fitting and
irregular-domain construction run before emission; lowering consumes the chosen
map without reconstructing it. A cut directory replaces the old per-cut scan
through all pieces. Temporary point lists used to fit maps are not retained in
the expression recipe.

Numerical and rotating root requests now retain these recipes through allocation
support with zero detached scalar operations. Insertion consumes each recipe
once. Standalone numerical and rotating entry points use the same constructor;
the old immediate family-preparation wrapper is removed. Existing regional
consumers materialize explicitly before editing endpoint SSA. Physical-section
projection materializes before remapping its borrowed coordinates. Private-state
and completion plans reject mixed recipe ownership.

Eleven focused lit tests pass, covering production preparation with and without
allocation support, six-ID allocation, section projection, independent matching
and ordering oracles, irregular family maps, zero/short/nonunit loops, and fresh
plan ownership. The endpoint-family fixture needed a test correction: operation
names in diagnostic attributes are not payload statements. Its corrected check
still requires every original loop and payload operation to be present.

The saved rotating output is byte-identical. The numerical output differs only
in a location line moved by the new RUN directive; arithmetic, commands,
provenance and certificates agree. Persistent retains the M1 inventories and
passes six-ID allocation. Its output differs in ten `arith.andi` definitions;
flattening associative Boolean ANDs proves equality at all 5,443 uses outside
those AND chains, and all other lines agree. Receipts, source/binary hashes and
comparison results are under `.local/family-pipeline-m2/`.

This is the first production family migration, not completion of M3.
Sequence and repeated crossings still prepare individual alternatives. The late
compactor still runs, physical-ID expressions are still lowered in the separate
allocation pass, and producers do not yet export all known symbolic maps
directly. No planning-speedup or device-speedup claim follows from this change.
The next migration must preserve availability and direct/cleanup protocol
selection before allocation while removing the corresponding per-alternative
preparation; merely delaying the same expanded preparation is insufficient.


## Repeated-allocation refinement: measured result

The baseline is `49f7ca18053b3e3cdbe321bfcfdc92a8fccbae11`. Per-invocation
allocation profiling now records lanes, endpoint alternatives, extrema work,
matrix and partition work, relative-query construction, proof requests/cache
misses, traversal steps, and elapsed time. The temporary profile owner follows
recursive preparation without escaping the request. Logical-only requests do
not construct allocation evidence. Proof counters include revisits to shared
nodes; arena size is not used as a substitute for traversed work. Coalescing and
independent-lift fallback are included in elapsed time but not the lane-sharing
query/proof counters.

For certified reference-ordered endpoints, shared first-present and last-present
scans replace incremental pairwise guard remasking. When the supplied inventory
is unordered, constant coordinate tuples propose an order, and every adjacent
pair must pass the existing reference-order contract. Conditional orders keep
the general algorithm. Mixed completion directions retain all endpoints needed
for causal reuse. Original event coordinates and allocation contracts remain
unchanged. Relative-query results, including unavailable results, are cached
only for one append-only allocation attempt.

| Persistent diagnostic | Baseline | Candidate |
|---|---:|---:|
| Lanes | 21 | 21 |
| First / last endpoint alternatives | 1,544 / 1,544 | 1,544 / 1,544 |
| Matrix lane pairs | 105 | 105 |
| Boundary-pair visits | 894,568 | 894,583 |
| Relative queries constructed | 10,170 | 1,200 |
| Implication computations | 57,264 | 134,934 |
| Proof traversal steps | 828,111,780 | 2,047,458 |

The candidate certifies 42 ordered extrema lists using 4,534 reference-order
comparisons. More distinct implication requests do much less total traversal:
the resulting guards are simpler. Persistent has no relative-query cache hits
in this measurement; its improvement must not be attributed to that cache.
The large boundary-pair inventory remains an outstanding representation cost.

Three serial unprofiled pairs, with no overlapping builds or tests, measure the
logical pass **including allocation support**, not physical lowering, C++
emission, or device execution:

| Pair | Baseline seconds | Candidate seconds |
|---|---:|---:|
| 1 | 9.115 | 3.909 |
| 2 | 9.417 | 3.608 |
| 3 | 9.016 | 3.507 |
| Median | 9.115 | 3.608 |

The median improvement is **2.53x**. All non-AND IR, including allocation
certificates, agrees. The changed i1 AND chains have identical complete leaf
multisets, compared with SSA scope preserved. The plan retains 1,544 records,
3,088 prepared endpoints, 116 normalized command sites, 56 emitted SET sites,
60 WAIT sites, one barrier, and 18,342 raw arithmetic operations. Six-ID physical
allocation succeeds. This change reduces planning work; it does not reduce
executed handshakes or claim a device speedup.

Six focused lit tests pass. Independent checks cover all guard valuations for
eight extrema, zero and long lists, shuffled provenance, conditional order,
mixed directions, unknown/unavailable query caching, and rollback. Existing
oracles also validate 1,583 allocation envelopes/reuse chains and 125,856 event
queries. The broader phased-repeat test times out: its first case also times
out on the preserved baseline and is not counted as passing. Receipts and
source/binary hashes are in `.local/family-pipeline-refined/validation.json`.

The next allocation policy change must retain ancestor fallback: a locally
feasible independent assignment can prevent the enclosing plan from fitting.
The current general sharing matrix is therefore still attempted first. The
measured change does not claim that M2 is complete.


## Shared occurrence-coordinate binding

Direct sequence endpoints, invocation-prefix endpoints, singleton prefix
lifting, and repeated-region endpoints now use the same original-loop ordinal
and occurrence-constraint implementation. It returns arena expressions, never
generated SSA. It preserves event predicates, leaf ordinals, enclosing quotient
coordinates and fixed anchor coordinates. Callers retain their existing exact
activity, masking and protocol behavior; repeated preparation retains its index
IV requirement. Validation inspects the requested frame without scanning every
event type at each endpoint.

Six focused production tests pass across direct sequence preparation, endpoint
retry, prefix matching, singleton lifting, invocation ownership, and the repeated
allocation oracle. The new binding oracle checks quotient/fixed coordinates,
all relevant Boolean outcomes, malformed frames and absent loops. Site and
invocation predicates are deliberately excluded from its coordinate-only result.
Persistent's logical IR and allocation evidence agree with the measured
allocation refinement under the same scoped AND comparison.

This removes duplicated coordinate decoding, not all availability, capture and
history interpretation. Sequence/repeated crossings still prepare individual
scalar alternatives. Before deferring those operations, publication and cleanup
must be established as recipes; successful emission currently participates in
that decision. No whole-M3 completion claim follows from this refactor.


## Availability-recipe migration

Contextual expression preparation now separates original-value availability from
scalar materialization. Its immutable recipe records the selected expression,
original deterministic replay producers, and required existing binding keys.
No generated SSA is captured. The existing contextual emitter consumes this
same recipe, and sequence ancestor-sharing probes explicitly plan before
creating a detached arithmetic block. This removes the previous combined
preflight/clone implementation without adding an alternative replay policy.

The recipe borrows its arena and original invocation. It cannot survive arena
rollback or payload/control mutation. Lowering checks its arena, original cut,
required bindings and forbidden producers before emitting code. Failed planning
creates no scalar code and does not charge successful replay bindings.

Terminal cleanup still requires its existing singleton-cut certificate. Pure
availability planning alone does not establish that certificate or matching.
Early sequence/repeated family construction remains open until the publication
protocol and cleanup evidence can be retained before their scalar predicates
are materialized. This slice is a prerequisite within M3, not its completion.

Validation: ten focused lit tests pass, including malformed specialization,
missing binding, foreign arena/cut, replay prohibition, and production direct
and prefix paths. Persistent retains equivalent logical IR and allocation
evidence in all three serial pairs, and allocates with six IDs. Baseline and
candidate medians are 3.807 s and 3.608 s with overlapping ranges; these are an
overhead check, not evidence of a planning speedup. Prefill logical preparation
succeeds in 75.601 s; allocation still reports its missing supported export.
Receipts and manual review are retained in
`.local/family-pipeline-availability/validation.json` and `review.md`.

## Singleton cleanup qualification before materialization

Terminal cleanup selection now uses one narrow predicate interpreter for exact
arena expressions and emitted scalar SSA. It checks original structured cuts,
positive-step counted coordinates, invariant selected values, and original
then-arm facts. The same traversal serves serialized-command revalidation.
This is a multiplicity proof, not an availability proof, and does not interpret
necessary domains as exact predicates. No loop visits or branch valuations are
enumerated by production code.

Early-publication preparation requests this proof before constructing its
publication predicate. Unmigrated relational recipes may still need the existing
emitted-predicate check when their equality is exposed only during lowering.
They retain that acceptance path; a missed compact pattern never forces a new
refusal. Consumer guards and full publication recipes still need migration.

Four affected lit suites pass. The helper regression compares 216 exact
predicate cases before and after lowering across negative/zero/positive lower
bounds, unit/nonunit steps and zero/short loops, plus nested-coordinate,
original-branch and malformed-input cases. Persistent and prefill logical IR
and allocation evidence agree with `477b7ae5e` under the scoped associative-AND
comparison. Persistent still allocates with six IDs. Prefill logical compilation
takes 75.330 s in this single correctness probe; no speedup is claimed.
The manual review and receipts are in `.local/family-pipeline-cleanup/`.

## Finite guarded publication recipes

The direct/early publication decision is now a shared policy independent of
scalar construction. An immutable recipe retains original cuts and the exact
publication and terminal-cleanup expressions. The recipe planner requests
availability and singleton selection before creating code; the compatibility
adapter supplies the existing emitted-value checks to the same policy.

Finite guarded preparation now plans supported protocols and caches their
cut-specific availability recipes before endpoint arithmetic. Materialization
consumes those recipes, sharing original scalar replay values across roots.
Unsupported compact qualifications retain the incumbent materialized route.
Matching identities, cleanup cuts and allocation envelopes are unchanged.

This migrates publication planning, not the full regional family hierarchy.
Finite endpoints still materialize within their producer, and sequence/repeated
composition still requires concrete endpoint SSA. Family normalization before
allocation and pure child composition remain open M3/M4 work.

Six focused suites pass, including recipe-only planning, early cleanup, direct
publication, wrong-arena rejection and shared scalar replay. Persistent retains
equivalent logical commands and evidence and allocates with six IDs. Prefill
changes independent arithmetic definition order: a scoped expression-DAG check
preserves each arithmetic-run operation multiset and compares all other lines,
command order and attributes after SSA substitution. Its three negative controls
reject altered comparisons, selectors and commands. No device result or paired
compiler speedup is claimed. Receipts and the manual self-review are retained in
`.local/family-pipeline-publication/`.

## Cycle-local early family refinement

Numerical family provenance and independently partitioned endpoint sides now
share one ordering implementation. When grouping creates cyclic command-order
constraints, only groups in cyclic strongly connected components are refined.
Previously every vertex left after topological elimination was split, including
acyclic descendants. Refinement still reruns the complete coactive-order check.
Endpoint-piece candidate pairs are collected per original cut; unrelated cuts
no longer participate in the pair scan. Same-cut exclusion testing remains
pairwise, with no general linear-composition claim.

The regression exhausts all 65,536 four-node directed graphs against an
independent reachability-closure oracle and checks an 8,192-node descendant
chain. Early family fixtures vary downstream membership and record labels,
checking startup/tail activity and unfolded command order. A downstream family
with 16 alternatives remains one family and two endpoint pieces, while the
actual two-group cycle is split. Five focused suites pass. Persistent retains
equivalent IR/evidence under the scoped associative-AND comparison and allocates
with six IDs. This is a general early-construction fix, not evidence of another
Persistent speedup or completion of regional recipe composition.
Validation and manual review: `.local/family-pipeline-order/`.

## Shared nonconstant physical tuple maps

M4's bounded mapping normalization now covers members sharing a certified palette
and normalized dynamic tuple terms but differing in their constant base. Zero
terms are removed, strides/phases reduced, constant terms folded, and remaining
terms sorted. Original endpoint coordinates select the exact base through the
existing finite-family expression builder; the common dynamic terms and palette
lookup are lowered once. Different dynamic terms or palettes retain the original
dispatcher. This re-expresses the supplied assignment, not a new reuse proof or
allocation. Expensive complete selectors retain the existing conditional policy.

For a saved-certificate scaling family with 32/128/512 records, total emitted
arithmetic drops from 183/471/1,623 operations to 85/85/85. Sparse, nested-tuple
and modulo-three variants also have constant counts across these sizes. The
modulo-three regression checks that expensive arithmetic remains guarded. Tests
compare every executed physical ID with the supplied formula, including zero
trips, nonunit coordinates, noncontiguous IDs and unsigned-maximum ordinals;
a differing-term case exercises the incumbent dispatcher. Four affected suites
pass, followed by the extended targeted mapping oracle. Persistent's six-ID
allocated output is byte-identical. No planning or device speedup is claimed.
Receipts and manual self-review: `.local/family-pipeline-tuple/`.

This closes one useful nonconstant-map subclass. General differing tuple terms,
recipe-backed regional composition and allocation on its normalized graph remain
open; no unrestricted symbolic synthesis has been introduced.

## Pinned eleven-kernel replay at `15f2007ea`

All eleven inputs complete logical insertion; seven complete six-ID allocation and C++ emission. The planner/allocator revision is `15f2007ea`; C++ uses the same retained backend as the earlier campaign. Local commands, input and binary hashes, profiles, and failure receipts are in `.local/generated-control-15f2007ea/`. These are serial single-run observations, not paired timing or device results.

| Kernel | Logical SET / WAIT / barrier sites | Raw logical arithmetic | Logical seconds | Allocation / C++ |
|---|---:|---:|---:|---|
| tilelang_gemm | 14 / 14 / 1 | 1,401 | 0.203 | pass / pass |
| pypto_gemm | 12 / 12 / 1 | 1,196 | 0.202 | pass / pass |
| persistent_gemm | 56 / 60 / 1 | 18,342 | 3.910 | pass / pass |
| group_norm | 17 / 17 / 24 | 367 | 0.202 | six-ID assignment not certified |
| causal_conv1d_prefill | 59 / 84 / 62 | 92,786 | 74.524 | missing allocation export |
| causal_conv1d_decode | 22 / 22 / 22 | 2,761 | 0.202 | six-ID assignment not certified |
| gdn_chunk_cumsum | 3 / 3 / 1 | 41 | 0.303 | pass / pass |
| mhc_head_mix | 3 / 3 / 7 | 45 | 0.202 | pass / pass |
| elementwise_pipeline | 9 / 9 / 1 | 171 | 0.203 | pass / pass |
| gated_delta_rule | 172 / 47 / 194 | 2,473,273 | 81.013 | missing prefix lifetime export |
| lossless_block_cast | 5 / 5 / 4 | 83 | 3.423 | pass / pass |

Static sites do not count executed commands or simultaneously live notifications. Raw arithmetic includes payload scalar work and preparation that downstream lowering can remove. The ten previously successful logical outputs retain their earlier aggregate site/arithmetic counts; this is an inventory comparison, not a semantic equivalence proof. Prefill separately confirms the ownership repair. Delta still generates 2,473,273 arithmetic operations and lacks physical lifetime export. Fewer static sites cannot establish six-ID feasibility.

Persistent retains 1,544 record members. Its profile starts with 3,088 prepared endpoint sites and 20,257 arithmetic operations; late normalization leaves 116 sites, 18,283 prepared arithmetic operations, and 11,468 endpoint-reachable arithmetic operations. The final logical function contains 18,342 arithmetic operations including 59 original payload operations. These measurements confirm that early regional family construction remains unfinished; output compaction alone does not avoid preparation.

Its repeated allocation visits 894,583 boundary pairs across 105 matrix pairs and 162 candidate gaps. There are 895,783 implication requests but 134,934 computations. The measured proof and relative-query computations occupy about 21 ms and 36 ms inside a 3.53 s repeated-allocation stage. Those timers exclude cache lookup and activity construction, so the residual time needs direct measurement before attributing it to either.

## Allocation activity construction and coalescing attribution

The repeated allocator now retains each ordered pair's activity conjunction in
its existing attempt-owned proof context, before implication-cache lookup.
Candidate gaps reuse the exact expression ID. The cache belongs to one unchanged
command graph and append-only arena; it does not survive a rollback or invocation.
New counters distinguish activity requests, conjunction constructions, and their
construction time. The coalescing timer includes child validation/palette
coalescing and is contained in the repeated-region time, never added to it.

On Persistent, 894,583 requests construct 134,813 conjunctions. Proof answers,
matrix pairs, candidate gaps, relative queries, and proof-node counts remain
unchanged. Three alternating serial pairs measure baseline 3.609/3.908/3.809 s
and candidate 3.908/3.609/3.608 s. Their ranges overlap; no reliable planning
speedup is claimed. Each pair's full logical output agrees under the existing
scoped Boolean-AND operand normalization, including allocation evidence.

A separate profiled run locates 3.173 s of 3.382 s in child-palette coalescing,
before periodic allocation. Activity construction occupies 42 ms, proof
computation 21 ms, and relative-query computation 35 ms. This narrows the next
performance investigation to coalescing; the periodic boundary-pair count alone
does not explain the remaining elapsed time.

The focused profile, regional-session, and physical-allocation suites pass.
The direct repetition oracle passes 14 nested command closures, zero trips and
compact-size checks. The combined repetition suite fails after 91 s, consistent with its previously
recorded phased-loop timeout; it is not counted as passing. A separate nested
allocation run was interrupted while stalled and is also not counted as passing. Local receipts and manual
self-review are in `.local/family-pipeline-activity/`. Device latency is unmeasured.

## Certified envelopes before constant-palette pair queries

The remaining measured Persistent coalescing cost was query construction, not
implication proofs. A temporary local probe of the incumbent constant-palette
path counted 235,294 member pairs, 237,863 distinct reuse queries, and only two
distinct proof requests; the queries occupied about 3.17 s. The probe is retained
in `.local/family-pipeline-coalescing/` and is not production instrumentation.

Constant palettes now retain an optional unconditional envelope alongside their
member provenance. Every member must be active and present; original source and
consumer pipes must match the shared cross-pipe direction. Native reachability
certifies an earliest source start and latest consumer completion. The summary
is extended as palettes merge. If the two envelopes are provably separated,
their separation covers all member pairs by transitivity. Each accepted merge has an envelope or all-pair reuse proof. Numeric palette
assignments may change; record inventory and original command graph do not.

Envelope failure does not imply incompatibility. Conditional members, unavailable
bounds, and interleaved lifetimes retain the original all-pair proof. In particular,
individually disjoint lifetimes can share a palette even when their enclosing
intervals overlap. The optimization adds no handoff, drain, ordering edge, or
capacity assumption; buffer banking does not serve as its certificate.

An independent finite-lifetime oracle covers 32/128/512 disjoint groups in both
orders with permuted record labels and a linear bound on full child-reachability
queries. It also checks overlapping, coincident, interleaved, and complementary
conditional lifetimes; retained records and active interval reuse are checked
under both condition valuations. Four focused suites pass, and the final
cross-pipe qualification was rechecked with the complete region-expression oracle.
Seven compliance matches are braced-control regex false positives, manually reviewed.

Three alternating serial Persistent pairs measure baseline 4.020/3.707/3.606 s
and candidate 1.905/1.905/1.906 s: medians 3.707 versus 1.905 s, a 1.95x
improvement for logical compilation including allocation support. All three
complete outputs agree under scoped Boolean-AND operand normalization, including
allocation evidence; the candidate still allocates with six eligible IDs.
This is one-kernel compiler evidence, not a device or broad-corpus speed claim.
Receipts and manual self-review: `.local/family-pipeline-coalescing/`.

The optimization is a sufficient coalescing proof on qualified groups. It does
not replace general guarded allocation, establish minimum capacity, or complete
the planned candidate policy with parent retry. Early regional family planning
and owned capability upgrades remain open.

## Evidence-aware liveness before insertion

After family normalization and preflight, insertion now deletes detached generated
arithmetic that no required root reaches. The shared liveness helper also supplies
profiling's new `required_arithmetic` count. Roots include command guards/identities,
member coordinates, necessary-domain coordinates, terminal-cleanup predicates,
non-arithmetic/effectful preparation, and SSA users outside preparation. A pending
allocation callback conservatively retains all preparation because its captured
Values are opaque. Original payload IR is never part of the deletion set.

Tests cover dead generated chains, untouched original arithmetic, domain-only
values, pending allocation captures, external SSA users, and a cleanup predicate
used only by immutable evidence. The existing later-SSA-use test now makes that
use an actual matching operand. Four insertion/publication/family suites and the
stage-profile suite pass. The build initially exposed a missing CMake source-list
entry after extracting the shared helper; the entry is added and both tools link.

Persistent retains 56 SET / 60 WAIT sites, while raw logical arithmetic falls from
18,342 to 11,527 and selects from 5,858 to 2,596. Prefill succeeds in a separate
74.321 s diagnostic replay, retaining 59 SET / 84 WAIT / 62 barrier sites and
reducing arithmetic from 92,786 to 92,648. Its canonical IR is byte-identical after
normalizing the original input path spelling. Persistent's canonical IR agrees
under scalar-DAG substitution and complete Boolean-AND operand-multiset
normalization; every non-arithmetic line, ordering and attribute is retained.
Negative controls changing a comparison, selection or SET fail this comparison.

Persistent still allocates with six IDs and emits C++. Both old and new C++ have
1,360 lines, 119 `if` statements and 69 ternaries: the backend already removes the
dead arithmetic. This is an intermediate-IR reduction, not a device-speed result.
No paired planning-speedup claim is made for this slice. Four compliance matches
are manually reviewed braced-control false positives.

Receipts and manual self-review: `.local/family-pipeline-liveness/`. Preparation
still constructs 20,257 arithmetic operations and 3,088 endpoints for Persistent;
this deletion boundary does not complete early regional family construction.
Private-state prefix plans use their existing materializer and are unchanged.


## Numerical recipes through ordinary sequence composition

Sequence composition now retains immutable references to numerical child recipes,
with checked record translations and positions among concrete endpoints. Appending
a contribution does not copy its selector inventory or create generated SSA.
Family-metadata allocation views retain the same recipe, so requesting their reuse
summary does not force endpoint materialization. The final iterative traversal
lowers each contribution once and preserves the positions of concrete commands
between contributions. Original member records and both matching sides receive
the same translation; a shared namespace calculation preserves sequence-wide
naming by direction and displacement.

This migrates ordinary numerical/rotating sequence children. Guarded endpoint
masks, projection remapping, repeated-region tuple insertion and overlays retain
their explicit concrete compatibility boundaries. Crossings still prepare their
own arithmetic, and the late family compactor remains. This is not yet a compact
recipe for every regional producer or every repeated crossing.

Eight focused suites pass, covering prepared insertion, sequence/GM composition,
regional sessions, guarded publication, family emission, logical insertion and
stage profiling. Nested recipe tests cover placement, provenance translation,
common namespaces, immutable allocation views, overflow and transactional
rejection. The production sequence fixture moves from 10 concrete prepared sites
and 77 arithmetic operations to 5 concrete sites, 5 recipe sites and 37 operations;
both lower to 10 sites and 62 normalized preparation operations.

Persistent and prefill replay successfully (1.905 s and 75.816 s diagnostic runs).
Their complete canonical outputs agree with saved prior outputs under the same
scalar-DAG/Boolean-AND comparison, including allocation evidence and command
order; all negative controls are rejected. Persistent still allocates with six
IDs. Its prepared inventory remains 3,088 endpoints and 20,257 arithmetic
operations: the remaining repeated/finite compatibility path is not migrated by
this slice. No paired compiler-speedup or device claim is made.

Local receipts and manual self-review: `.local/family-pipeline-sequence/`.
Ten compliance matches are manually reviewed braced-control regex false positives.
Final formatting-only line wraps have the same token content as the tested
sources; the recorded comparison identifies those two files explicitly.


## Invocation coordinates retained in numerical family recipes

Repeated composition now keeps numerical child recipes when adding the enclosing
visit to matching tuples. The wrapper owns the original expression arena and
plans availability through the existing cut-expression service once per original
cut. It creates no generated SSA. Nested wrappers retain outer-to-inner tuple
order; local barriers receive no communication identity. The same shared
namespace calculation now serves concrete sequence and repeated endpoints.

Final lowering stages the complete recipe in a temporary prepared plan. Failure
of a retained binding preserves both the incumbent concrete inventory and the
unmaterialized recipe. Cut-local expression contexts share matching arithmetic
within the same arena and original cut. The lowered plan is adopted only after
all required coordinates have been emitted successfully.

This changes the internal repeated-child boundary, not the repeated crossing
algorithm. Crossings, guarded masks, projection and non-numerical singleton
producers still have concrete preparation paths. It adds no publication, cleanup,
ordering edge or private runtime state, and makes no new physical-capacity claim.

Six focused suites pass, as do 14 independent repeated-region command closures,
including zero trips and compact-size checks. The new profile fixture initially
requested allocation aggregates its root does not export; it now checks recipe
retention separately. Existing regional allocation-profile requirements remain
unchanged and pass. The repeated fixture changes from 17 concrete endpoints and
276 prepared arithmetic operations to 12 concrete endpoints, 5 recipe sites and
236 operations; final sites and normalized arithmetic remain 17 and 249.

Persistent and prefill diagnostic replays succeed (1.905 s and 74.413 s), agree
with saved complete canonical outputs under the existing checked comparison, and
reject all comparison negative controls. Persistent still allocates with six IDs;
its 3,088-endpoint prepared inventory remains unchanged. No compiler-speedup or
device claim is made. Six compliance matches are manually reviewed braced-control
false positives. Receipts and manual self-review are in
`.local/family-pipeline-repeat-recipe/`.


## Fixed-coordinate crossing families before preparation

Sequence preparation now qualifies finite crossing batches before generating their
endpoint arithmetic. Qualification uses the shared original occurrence constraints,
original-cut branch facts and typed DAG substitution. Both endpoints must have
source ordinal zero, no unresolved visit frame, and full activity equal to their
supplied fixed coordinate point at the original cut. A necessary coordinate domain
alone is insufficient. Unknown residual predicates keep the existing guarded path;
there is no sampling, new solver, payload replay or changed publication policy.

Qualified batches reuse the existing endpoint-piece ordering and family-expression
compiler. Finite leaves have source ordinal zero; enclosing invocation coordinates
still use the existing recipe wrapper. Original records and allocation lifetimes
remain unchanged. Mutually exclusive points can share a command site; simultaneous
commands retain multiplicity and order. Batch construction is transactional and
creates no scalar MLIR. Guarded child masks may still materialize a compact child
recipe before the root, so this is not yet one final lowering for every producer.

For Persistent GEMM, the four finite batches contain 379 crossing records each.
Compared with `d7f0ff0ae`, the production inventories are:

| Quantity | Before | After |
|---|---:|---:|
| Retained record members | 1,544 | 1,544 |
| Concrete prepared endpoints | 3,088 | 128 |
| Prepared arithmetic | 20,257 | 12,793 |
| Normalized detached arithmetic | 11,468 | 3,865 |
| Final SET / WAIT sites | 56 / 60 | 56 / 56 |

All ordered payload and command events, matching identities and numeric physical
IDs agree on cores 0, 15, 16 and 19. Core 0 still executes 5,011 matched pairs;
reducing static WAIT sites removes no dynamic communication. Six eligible IDs
suffice for the unchanged allocation evidence.

After canonicalization and CSE, the core-0 interpreter counts 359,137 -> 258,062
arithmetic evaluations and 105,209 -> 98,553 guard evaluations. Select evaluations
increase from 21,160 to 54,310. The fixed C++ backend produces 1,363 -> 1,326 lines,
119 -> 115 `if` statements and 69 -> 95 ternary selections. These are representation
and interpreter measurements, not device latency or instruction-cost estimates.

Focused validation includes sparse coordinate sets, simultaneous commands, arbitrary
record labels, invalid finite recipe contracts, guarded publication and production
finite-loop closures, including zero trips. The fixed 1-by-1 nested fixture is
rejected by both pinned baseline and candidate; its artifacts are retained separately
and it is not counted as passing coverage. The zero-trip fixtures use the public
compiler followed by trace-existing interpretation, since the test-only regional
trace entry rejects their empty mathematical region.

Three alternating serial pairs have medians 1.804 s before and 1.804 s after.
There is no demonstrated full-planning speedup. Allocation support remains the
major measured stage. The final-source output agrees with the traced candidate
under complete scalar-DAG/Boolean-AND comparison; changed-command, comparison and
Boolean-operator negative controls are rejected.

Nine focused suites and 14 independent repeated-region closures pass. Prefill
compiles in a 77.419 s diagnostic run and its canonical output agrees with the
pinned baseline. Nine changed-code checker matches are manually reviewed
braced-control false positives; no new compiler warning is introduced.

Local artifacts and manual self-review: `.local/family-pipeline-finite-crossing/`.
This is an M3 slice. Guarded/relational crossings, early normalized-plan allocation,
complete ID-map optimization, full pinned campaign and device validation remain open.


## Deferred activity masks on compact family recipes

Child site and invocation activity no longer force an otherwise deferred family
through scalar materialization. The existing invocation-context recipe now also
owns exact Boolean activity bindings. Both use the shared cut-expression planner
and lower through one cut-local cache. Activity applies to same-pipe barriers as
well as cross-pipe commands; invocation coordinates remain exclusive to matched
handoffs. Exact masks are not inferred from necessary coordinate domains.

Availability is checked at each original command cut before retaining the wrapper.
Final materialization revalidates the binding transactionally. A rejected binding
leaves the incumbent plan intact. Mask combinations occupy a stage after both the
shared cut value and the endpoint guard, including when the guard belongs to a
later leaf. This avoids a use-before-definition across detached preparation blocks.

Compared with `9d0e2ca18`, Persistent GEMM changes as follows:

| Quantity | Before | After |
|---|---:|---:|
| Concrete endpoints at root preparation | 128 | 56 |
| Deferred family sites at root preparation | 0 | 72 |
| Arithmetic at root preparation | 12,793 | 796 |
| Arithmetic after materialization, before cleanup | 12,793 | 11,808 |
| Normalized detached arithmetic | 3,865 | 3,865 |
| Final SET / WAIT sites | 56 / 56 | 56 / 56 |

The new `materialized` control-profile boundary is reported for direct plans
between `prepared` and `normalized`. It exposes deferred work separately from
eliminated work: the large preparation decrease is mainly postponement; total
arithmetic constructed before cleanup falls by 985 operations. Final complete
canonical output agrees with the pinned baseline and allocation still succeeds
with six eligible IDs. The 1.905-second planning probe is diagnostic only, not a
paired timing result or a demonstrated compiler/device speedup.

Focused checks cover nested masks and invocation coordinates, barriers, arbitrary
record labels, detached-stage dependencies, rejected bindings and transactional
failure. Eight independently checked finite-loop closures include zero trips and
outer-bank variants. The outer-bank variants can select another guarded producer;
recipe retention is asserted separately on the regular finite fixture rather than
assuming the same producer for every input.

Eight focused suites pass, as do the 14 independent repeated-region closures.
Prefill compiles in a 72.916-second diagnostic run and its complete canonical
output agrees with the pinned baseline. Canonical comparisons reject changed
comparison, command and scalar-operation negative controls. Six compliance
matches are manually reviewed braced-control false positives.

This extends M3 and its measurement boundary. Guarded/relational crossing
construction, projection ownership, early normalized-plan allocation and the
remaining full-pipeline milestones are unchanged. Local validation artifacts and
manual self-review are in `.local/family-pipeline-masks/`.


## Coordinate-selected phases in certified physical maps

Physical lowering now factors members whose tuple rules have the same palette,
arity, coordinate/stride/modulus/scale skeleton and different constant phases.
The existing exact finite-coordinate expression builder selects each phase and
constant base. One shared modular calculation then implements the complete map.
Every supplied member rule remains authoritative; no assignment, reuse proof,
logical command, cut or payload order changes.

Terms are normalized modulo their declared modulus and sorted by skeleton before
phase. Zero-stride terms fold into the base. Reordered duplicate terms retain their
full additive contribution. Unsupported skeletons use the incumbent dispatcher.
The candidate is compared with that dispatcher using the existing weighted live
ID-DAG cost; ties and higher costs retain the incumbent. This is one bounded
re-expression attempt, not a solver or additional allocation search. Original
record-label computation is not charged to the incumbent, making that comparison
conservative. Expensive calculations retain the existing conditional placement.

The independent formula fixtures compare every executed ID, including sparse
points, nonunit loops, noncontiguous palettes, nested source tuples, maximum
unsigned ordinals, zero trips and independently grouped waits. For sizes 8, 32
and 128 (four points per row before optional holes), raw arithmetic inventories
including original fixture arithmetic are:

| Phase-map fixture | `26ed9fb7f` | Candidate |
|---|---|---|
| One shifted parity term | 193 / 481 / 1,633 | 88 / 88 / 88 |
| Two nested shifted terms | 205 / 565 / 2,005 | 81 / 81 / 81 |
| Sparse points, maximum ordinal | 170 / 386 / 1,250 | 89 / 89 / 89 |
| Shifted modulus-three term | 241 / 721 / 2,641 | 95 / 95 / 95 |

Existing identical-term cases retain their previous inventories. Additional
exceptional-phase, duplicate-term and incompatible-stride cases check exact IDs
without claiming constant-size code. The initial duplicate-term growth assertion
was too strong: canonical pairing can require a non-affine phase selector. Its
semantic check remains; that residual language limitation remains explicit.

Four focused allocation/sequence suites pass. The baseline/candidate growth replay
retains all ID and zero-trip checks, disabling only the new growth assertion for
the old compiler. These fixtures consume saved producer-trusted certificates and
test map implementation, not independent discovery of reuse. The physical
allocation suite separately checks production reuse behavior.

Persistent still allocates with six IDs; its physical IR is byte-identical to
the baseline. All ordered physical events agree on cores 0, 15, 16 and 19.
Four compliance matches are manually reviewed braced-control false positives.
The initial build exceeded LLVM SmallVector's default inline-element size check;
packing optional composition state together fixed that build error.

No compiler-time or device-speed claim follows from these code counts. This is a
bounded M4 extension; differing term skeletons, family-level allocation before
materialization, and the M2 parent-aware cheap-candidate policy remain open.
Local artifacts and manual self-review: `.local/family-pipeline-phase-maps/`.


## Pinned eleven-kernel refresh at e0e17e5db

The serial campaign keeps all eleven inputs and the C++ backend pinned, verifies
tool/input hashes, uses MayNotAlias and IDs 0--5, and retains every diagnostic.
All eleven logical compilations succeed; seven allocate and emit C++. There are
no compiler timeouts. These are one-run stage diagnostics, not paired timings or
device results. The later requested-output change below is validated separately.

| Kernel | Logical SET / WAIT sites | Raw function arithmetic | Allocation / C++ |
|---|---:|---:|---|
| tilelang_gemm | 14 / 14 | 1,401 | pass / pass |
| pypto_gemm | 12 / 12 | 1,196 | pass / pass |
| persistent_gemm | 56 / 56 | 3,924 | pass / pass |
| group_norm | 17 / 17 | 365 | assignment uncertified |
| causal_conv1d_prefill | 59 / 84 | 92,648 | missing allocation export |
| causal_conv1d_decode | 22 / 22 | 2,761 | assignment uncertified |
| gdn_chunk_cumsum | 3 / 3 | 41 | pass / pass |
| mhc_head_mix | 3 / 3 | 45 | pass / pass |
| elementwise_pipeline | 9 / 9 | 171 | pass / pass |
| gated_delta_rule | 172 / 47 | 2,473,273 | missing prefix lifetime export |
| lossless_block_cast | 5 / 5 | 83 | pass / pass |

The direct-plan profiles now distinguish preparation, materialization,
normalization and insertion. Prefix/private-state output retains its own profile
boundaries; Delta's zero prepared direct endpoints do not mean zero construction
work. Its final 2,473,273 arithmetic operations remain an unresolved expansion.
Prefill retains 92,437 normalized preparation operations and requires a separate
allocation export. Neither refusal establishes minimum required hardware capacity.

Persistent's preparation/materialization/normalization counts are 796 / 11,808 /
3,865 arithmetic operations, with 56 / 56 final SET / WAIT sites. All static counts
remain distinct from active demand instances, executed commands and live flags.
The input/expert inventories, stage diagnostics and control/allocation counters are
retained in [the pinned results](frontier-family-control-validation.json).
Full local artifacts: `.local/generated-control-e0e17e5db/`.


## Requested family-expression outputs

Physical mapping requests only the selected member/base/phase value from the
existing family-expression planner. Its enclosing logical command already owns
the exact activity domain. The planner validates every coordinate and label as
before, but does not build the full presence-domain boxes or emit their guard.
Residual decision guards needed to select a value remain. Command-family clients
retain the default presence-and-member contract.

Coordinate normalization is lazy within one emitter. Unused axes and the divisor
of a constant modular result are never emitted. This replaces construction and
subsequent deletion of that unused arithmetic; it does not change domain fitting,
matching, certificates or synchronization protocols. Member-only values remain
meaningful only within the caller-established exact domain.

The construction regression checks sparse constant mappings before any cleanup:
ordinary and modular selected values each create exactly one constant, while the
command request retains its presence guard. Duplicate coordinates still reject
without emitting operations. Eight focused suites pass. Complete canonical
comparisons cover new logical compilation of all three GEMMs and physical
allocation of all seven successful saved campaign plans. Those outputs agree
with `e0e17e5db`; this is a construction improvement, with no paired compiler-time
or device-speed claim. Artifacts: `.local/family-pipeline-requested-output/`.

## M3 repeated crossing recipes

Repeated preparation now retains its crossing activity, source ordinal and
phase-residue evidence in the existing family recipe tree. The shared contextual
availability service certifies each expression at its original cut without
constructing MLIR arithmetic. The general repeated crossing emitter is removed;
final lowering uses the common family materializer, including namespace changes,
enclosing invocation coordinates, activity masks and transactional failure.

When the original loop domain and exact fixed-coordinate endpoint activity are
certified, a repeated crossing batch instead uses the existing numerical family
compiler. Sequence and repeated producers share the same attempt-owned
fixed-point activity proof. Quotient phases, clipped invocations and unresolved
activity use contextual recipes with their original exact boundary predicates;
necessary residue evidence never replaces those predicates.

This is a preparation migration, not a claim that every guarded family has been
compressed or that its arithmetic is cheap. Guarded sequence publication/cleanup,
relational crossings and projection-owned remapping remain separate work. The
allocation policy and selected demands are unchanged.

Validation: nine focused lit suites pass (eight from the first run, then the
corrected repeated-family regression), plus 14 existing repeated-region closure
cases. The new production regression checks 14 guarded/zero-trip executions;
six before/after traces preserve every ordered payload and logical command.
Contextual recipe tests cover residue validation, malformed activity/member
inputs and atomic failure after an availability binding is invalidated.

For the guarded nested fixture with surrounding work, preparation changes from
12 concrete endpoints plus 5 deferred pieces to 5 concrete endpoints plus 12
deferred pieces. Prepared arithmetic changes from 56 to 28; materialized and
normalized arithmetic remain 96 and 82. All three GEMM logical outputs match the
pinned canonical outputs; Persistent's six-ID physical output matches too. These
are representation/semantic checks, not paired compiler-speed measurements or
device results. Additional two/three-bank phase probes timed out (90 seconds in
the trace driver and a separate 20-second public probe); they remain unresolved
and are not counted as passing. The complete historical phase suite remains open.
Receipts are retained under `.local/family-pipeline-repeated-crossings/`.

## M3 guarded sequence crossing recipes

Sequence crossings now settle direct versus early publication and terminal
cleanup from exact expression recipes, then retain their guards, identities and
original coordinates in the common family tree. Cleanup alternatives own their
arena and singleton-selection evidence until final lowering binds the actual
consumer guard. That binding is revalidated transactionally before any generated
preparation is committed. Allocation sees the selected protocol and its terminal
lifetime before scalar materialization.

The sequence-specific scalar emitter and scratch-block sharing path are removed.
Requests to share an expression at a common dominating cut are retained as
recipes and lowered through the same contextual service. The final materializer
shares those accepted values only at dominated cuts. Exact activity remains
separate from necessary coordinate evidence. Physical-context projection still
materializes recipes before transferring original cuts; projection-owned recipe
remapping is not claimed by this migration.

Guarded singleton families are deferred, not automatically compressed into a
small modular map. This migration does not change the selected demand graph,
logical protocol, resource policy or direct/prefix ownership. Relational exports,
shared capture/reset closure and early allocation selection remain open.

The sequence preparation counter now attributes child namespace/identity
adapters to child import; crossing preparation counts only new crossing work.
The sum is unchanged. Historical counter splits need this attribution difference
when compared with the new diagnostic.

Protocol selection and family construction share successful availability recipes
within one append-only attempt. The cache retains no generated SSA and expires
before arena rollback; distinct roots still have their own traversal costs.

Validation: ten focused suites pass, with five relevant suites rerun successfully
after the shared planning cache. Tests cover retained cleanup selection, final
binding, failure atomicity and independent materialization of repeated requests.
The guarded sequence regression now requires zero new crossing preparation;
child import adapters remain measured separately.

The eleven-input candidate replay reports 11 logical, 7 six-ID allocation and
7 C++ successes, with the same four export/allocation refusals and no timeouts.
The final cache preserves prefill's candidate IR exactly. Canonical structural
comparison with the earlier pinned campaign preserves all three GEMMs and
GroupNorm. Prefill differs only at four guards where an enclosing branch already
establishes the removed conjunct; payloads, command order, identities and the
remaining canonical arithmetic match. Negative controls reject changed command
kinds, arithmetic operations and the relevant enclosing branch.

Prefill retains 131 record labels and 59/84 static SET/WAIT sites; raw arithmetic
falls from 92,648 to 90,050. Candidate prefill probes took 84.028 and 86.714 seconds;
these are not paired timing evidence and establish no planning speedup. No device
execution is claimed. Full receipts are retained under
`.local/generated-control-sequence-context/` and
`.local/family-pipeline-sequence-context/`.

## M3 finite guarded family recipes

Finite guarded insertion now retains exact activity, publication, acquisition
and complementary terminal cleanup as contextual family recipes. It builds no
per-endpoint arithmetic or generated identities before common materialization.
Allocation retains the same source/consumer envelopes and terminal-lifetime flag;
stream namespaces and record provenance preserve their previous meanings.

The producer-specific scalar emitter is removed. The former standalone
publication-recipe materializer, whose last production caller was this producer,
is removed too. Its tests now exercise the shared family materializer. The older
circuit endpoint adapter remains for the unmigrated bounded/arithmetic paths;
this slice does not claim those paths have migrated or that guarded records
have become numerical periodic maps.

Validation: seven focused suites pass; shared publication and prepared insertion
pass again after deleting the obsolete helper. The finite guarded analysis
regression checks zero preparation operations and nonempty materialized code for
live shared-storage cases. Thirty-two before/after traces preserve every ordered
payload and logical command across ordinary, branch-selected, early-publication
and local-barrier cases. The public prefill replay succeeds and matches the
preceding candidate after canonicalization; record namespace additions now fold
before lowering. Receipts are retained under
`.local/family-pipeline-finite-context/`. No compiler or device speedup is claimed.

## M3 finite overlay recipe composition

Finite overlays now retain filtered child recipes and append their exact added
handoffs through the common contextual family compiler. Allocation reads matching
tuple sizes directly from the same recipe composition walk used by lowering,
including namespace translation and invocation prefixes. Conflicting tuple
sizes are rejected. This metadata describes matching shape only; it does not
replace activity, matching or reuse proofs.

The overlay's duplicate loop/coordinate interpreter and new-handoff scalar
emitter are removed. Shared occurrence constraints retain the existing index-IV
contract and finite ordinal-zero check. Already concrete legacy children keep
their compatibility adapter; migrated children need no scalar materialization
merely to export allocation metadata. Existing augmented-order reuse checks and
physical allocation remain unchanged.

Eight focused suites pass, including malformed matching shapes and independent
lowering retries. Ten before/after cases preserve complete ordered logical and
physical traces, including physical IDs. The refreshed eleven-input campaign
reports 11 logical, 7 six-ID allocation and 7 C++ successes, with the same four
refusals and no timeouts. Persistent's arithmetic differences are associative
Boolean conjunctions; prefill matches the preceding finite-family candidate
after canonicalization. No paired planning-time or device-speed claim is made.
Receipts are under `.local/family-pipeline-overlay-context/` and
`.local/generated-control-overlay-context/`.

## M3 guarded periodic family recipes

The common guarded/numerical periodic adapter now retains exact source and
inverse ordinal maps, activity and entry-invariant recipes before scalar
materialization. It uses the existing contextual family compiler and matching
namespace, preserving original generator IDs as member provenance. Its private
SSA emitter and entry-to-endpoint Value caches are removed. This migrates both
guarded rotating exports and arithmetic-to-periodic conversion; it adds no
recognizer, protocol or allocation proof.

Shared entry expressions remain explicit planning bindings. The common service
preserves their masking and expression identity during cut specialization,
requires actual bound values before lowering, and scopes bindings to one recipe
build. Duplicate shared roots are retained once. Tests reject missing bindings
without generating arithmetic and exercise independent builds using one cache.

Nine focused suites pass. The guarded-rotating suite passes its regular, ACC,
branch-local replay and poison-mask checks, then times out on alternating-guard
rejection; the retained baseline also times out on that exact input at the same
90-second limit. This existing failure remains recorded, not counted as passing.
The nested-overlay fixture now checks recipe retention before materializing for
its unchanged coordinate/guard oracle. Eighty baseline/candidate ordered traces
match after explicit provenance-label decoding, and ten overlay logical and
physical traces match.

The serial eleven-input campaign retains 11 logical successes, seven six-ID
allocations and seven C++ outputs, without timeouts. The same four allocation
exports remain unavailable. All three GEMMs retain their command-site and raw
arithmetic counts. GroupNorm arithmetic falls from 361 to 335 and elementwise
from 171 to 164 through constant provenance-translation folding. Canonical
comparison establishes equivalent changed outputs; four Prefill guards differ
only by conjunction with an already-true enclosing condition. Delta's large
logical output is unchanged. No paired planning-time or device claim is made.
Receipts are `.local/family-pipeline-periodic-context/` and
`.local/generated-control-periodic-context/`.

## Shared materialization cache refinement

The staged family lowering now owns one `FamilyMaterialization` context. Its
cache key includes the expression arena, original command cut and shared-root
binding owner. Self-contained leaves and invocation/activity wrappers share the
empty binding environment. Distinct nonempty binding owners remain isolated;
no cache survives a failed materialization or an arena rollback.

Shared expressions and replay inputs are indexed once after their defining
recipes lower. A receiving cut imports required or reachable bindings only,
and checks the original defining cut's dominance. It no longer copies every
entry of every dominating shared memo. Wrapper preparation appends after earlier
leaf preparation, preserving detached SSA dependency order.

`frontier-control-profile` now includes `family_materialization` counters for
published bindings, lookup requests, inspected candidates, imported bindings,
local memo entries and cut contexts. These are construction-work diagnostics;
they do not influence admission or serialize into allocation evidence.

The regression with 512 shared predicates and 64 cuts requires exactly 63
imports and 63 candidate checks beyond the source cut. Retained entries equal
published entries plus those 63 imports. Another regression composes 32
self-contained leaves and activity wrappers using one cut context. Fresh retry,
missing bindings and a non-dominating shared definition are checked separately.

At baseline `078658748`, the focused rebased campaign passed 11 tests and the
historical phased-repeat trace timed out at 90 seconds. The candidate passes
all 11 selected checks, including the new regressions, plus 14 nested/zero-trip
command-closure cases. Ten overlay configurations preserve the complete ordered
logical and physical traces. TileLang/PyPTO canonical logical output is byte
identical; Persistent agrees after SSA naming and associative/commutative
Boolean-AND normalization, with a changed-command negative control rejected.

Persistent physical allocation succeeds with eligible IDs 0–5. Complete ordered
payload/synchronization traces, matching identities and numeric IDs agree on
cores 0, 15, 16 and 19: 5,011 pairs on cores 0/15 and 4,625 on cores 16/19.

For Persistent, arithmetic constructed at materialization falls from 11,811 to
11,550 operations. Final logical arithmetic remains 3,924 and static SET/WAIT
sites remain 56/56. The successful lowering reports 42 published bindings, 256
lookup requests, 89 candidate checks/imports, 1,207 local entries and 26 cut
contexts. These inventories establish scoped sharing; they do not establish a
compiler-time or device-speed improvement. The diagnostic timing pair was not
an isolated performance experiment.

The build passes with no new warnings. The changed-code prefilter reports 13
`G.FMT.11-CPP` matches: all are nested/multiline conditions followed by braced
bodies, verified against the complete conditions. No source suppressions were
added. This is scoped build/test/manual validation, not a full analyzer or
device campaign. Receipts are in `.local/family-pipeline-shared-context/`.

Remaining costs include shared-root availability planning per cut, wrapper
subtree discovery, the extra reachable-DAG lookup traversal, and conservatively
separate nonempty environments. This refinement does not claim compact guarded
families, allocation on a normalized recipe graph, or Delta lifetime support.

## Mixed numerical and guarded crossing batches

Sequence and repeated producers now retain the existing exact numerical
qualification for each crossing instead of losing all qualifications when one
crossing needs contextual activity. The repeated qualification still requires
an unquotiented, unclipped original counted loop with matching trip bounds.
Qualified points and residual publication/cleanup recipes enter one endpoint
inventory. Necessary coordinate evidence authorizes exclusion only; the
producer's exact activity proof authorizes numerical lowering.

The shared piece scheduler groups compatible numerical points while retaining
before/after constraints on coactive residual commands. Its complete order
check and SCC refinement remain authoritative. Residual singletons retain their
relative order through a chain, avoiding a dense residual-only order graph.
Numerical/residual comparisons are limited to common original cuts. Unsupported
mixed grouping keeps the exact contextual realization available. Matching
identities, allocation envelopes and publication policy are unchanged.

Mixed leaves reuse the numerical selector builder and contextual binding
service. No new emitter or expression language is introduced. The opt-in
`recipe_forms` inventory reports numerical sites, contextual sites and mixed
leaves by walking the retained recipe; it is not independently maintained
semantic state.

Validation against `6a19b11db`:

- Twelve focused lit tests pass, including production mixed crossings with both
  guard outcomes and zero-trip cases checked against an independent unfolded
  effect graph. All ten cases also allocate with IDs 0–5 and preserve the
  baseline ordered logical and physical traces, including numeric IDs.
- The grouping oracle checks 8/32/128 point families, residuals at the start,
  middle and end, reversed record numbering, excluded and coactive alternatives,
  simultaneous commands and malformed inventories. A 4,096-residual case checks
  preserved order without a dense residual-pair graph.
- TileLang and PyPTO canonical outputs are unchanged. Persistent allocates with
  eligible IDs 0–5 and preserves complete ordered payload/command traces,
  matching identities and numeric IDs on cores 0, 15, 16 and 19. Core 0 retains
  5,011 matched pairs.

This closes the all-or-nothing crossing-batch limitation for the existing
numerical qualification. It does not compress arbitrary residual guards, add
phase/partial-invocation qualifications, or complete allocation on the compact
protocol. No paired compiler-speed or device-performance claim is made.
Receipts are retained locally under `.local/family-pipeline-mixed-batches/`.
The historical phased-repeat timeouts and four physical/export refusals remain
open; this focused validation does not replace the full pinned campaign.

## Owned allocation requests and parent retry

The prepared plan's one-shot allocation callback is replaced by an owned
`AllocationPreparation` capability. Independent and shared constructions cache
separate answers, including failure, on the unchanged logical command graph.
Requesting a candidate does not select it. Sequence, repetition, conditional
masking and finite overlays propagate the parent's requested strength through
their retained child views. A stronger parent request can therefore revisit a
locally feasible independent assignment without recomputing demands or endpoint
arithmetic. Unmigrated producers retain one-shot compatibility, including
certificate-only results.

The independent repeated construction uses the existing certified lifting; it
does not infer reuse from buffer parity. Shared requests preserve the previous
periodic-lane-first policy. Once Shared has run, later requests reuse retained
answers instead of constructing a weaker result. Candidate state expires with
the plan; no callback enters a saved certificate.

An internal explicit-capacity selector checks the complete regional certificate,
including palette conflicts and reservations, against the supplied eligible ID
pool. It tries independent and shared candidates, requests optional lane
evidence only when the base certificate does not fit, and selects only a
successfully decoded certificate. Failed attempts preserve the incumbent plan.
This establishes the request lifecycle, not the full production cost policy:
the public split pipeline still requests Shared and retains its existing output
contract. It does not know a subsequent allocator's eligible pool.

The focused checks cover parent escalation from a seven-ID independent
composition to a six-ID shared composition, cached failure, repeated requests,
shared-first requests, moved ownership and legacy certificate-only output.
Production mixed-family fixtures exercise explicit-capacity selection before
scalar materialization, invalid/empty pools, failed-attempt preservation, retry,
physical allocation and independent unfolded command-closure checking.

Remaining work is to pass the real capacity through the owned production
session, compare complete executable mapping costs, and reduce the expanded
lifetime inventory. This milestone does not claim those are complete or that
a diagnostic count reduction proves a compiler-time improvement.

## Explicit capacity during session preparation

`FrontierAnalysis::attachAllocation` accepts an optional eligible-ID pool. The
existing public pass forwards `allocation-ids=0,1,2,3,4,5` to that request, before
materializing deferred families. IDs must be distinct and in 0–5. The option is
incompatible with analysis-only mode or `prepare-allocation=false`. An omitted
pool preserves the existing unrestricted shared export. This is a resource
request on the existing pipeline, not another demand-analysis route.

For an already available regional interface, the session tries the retained
independent/shared candidates using the whole-parent certificate decoder. It
does not run another recognizer or construct general queries to obtain that
interface. Capacity-specific certificates stay on their prepared plan and never
enter the unrestricted export cache used by fresh preparations. A failed
candidate attempt falls back to the existing certificate-export path; it does
not change the selected demands or substitute a conservative protocol.

The saved output remains a logical plan plus self-contained evidence. Run the
physical pass immediately afterward with its actual eligible pool:

```text
--pto-frontier-analysis="allocation-ids=0,1,2,3,4,5"
--pto-frontier-allocate="eligible-ids=0,1,2,3,4,5"
```

The physical pass remains authoritative for supported mappings, command
bindings, reservations and capacity. The analysis pass still permits logical
output when allocation support is unavailable. Its success is not a hardware
feasibility claim. Identity adapters forward to retained child exports through
the same ownership check as unrestricted allocation. Providers without a retained regional export
keep their existing allocation path.

This integration exposes parent-aware candidate selection in production but
does not yet score complete executable mappings before materialization or
replace record-level lifetime inventories by a normalized family graph. The
combined transactional physical session and broader interface closure remain
open.

### Share preparation across candidate requests

Candidate selection must not double unchanged child work. A producer-owned
input cache retains immutable summary identities within one fixed command graph,
arena, coordinate translation and binding context. Sequence namespace mapping,
conditional masks and finite overlays reuse their summary when their owned child
answers are unchanged. Repetition also reuses certified palette coalescing and
finite fallback preparation. A stronger child answer with a different identity
invalidates this reuse. No structural guard comparison or global cache is used.
The independent and shared allocation algorithms remain separate.

The profile now counts `coalescing_cache_hits`. Before this sharing fix, the
new explicit-pool retry performed two expensive coalescing passes on Persistent.
A final serial diagnostic pair measured allocation support at 1.483 seconds for
baseline `41eb8be66` and 1.484 seconds for the candidate, with the second
coalescing request served from the owned cache. This is evidence that the new
retry avoids that repeated work, not a compiler-speedup or device-latency claim.

### Deferred callees and validation

The module's event-use check now reads deferred recipe matching metadata. A
callee containing only recipe-backed SET/WAIT sites still needs the existing
interprocedural event-lifetime contract when a caller can retain occupied flags
across the call. The previous check saw only concrete endpoints/private state.
The regression's baseline accepted that unsupported call; the repaired library
and public paths reject it without mutating the original module.

Validation receipts are in `.local/family-allocation-session/`:

- Eleven distinct focused lit tests pass, covering candidate ownership and
  invalid/sparse pools, identity forwarding, overlays, guarded specialization,
  repeated recipes/profiles, closed callees, completion and stage accounting.
  The repeated profile test forces a parent retry and requires a coalescing
  cache hit. The broad repeated-region/storage checks were interrupted and are
  not counted as passing; the historical phase coverage remains open.
- Default TileLang, PyPTO and Persistent logical outputs retain canonical
  agreement with the baseline, allowing associative Boolean-AND ordering.
  A changed-command negative control is rejected.
- Explicit six-ID Persistent preparation retains all ordered logical events on
  cores 0, 15, 16 and 19. Physical traces independently pass matching and causal
  reuse checks. Cores 0/15 retain 5,011 pairs and cores 16/19 retain 4,625 pairs.
  Physical arithmetic remains 4,600 operations.
- No new device execution, full corpus closure or minimum-capacity claim is
  established by these checks.

## Compact ordered lifetime selectors

After certifying strict reference order, the repeated allocator now retains only
the first candidate for each identical exact activity in its forward scan and
the last in its reverse scan. If two candidates have the same activity, the
later one cannot be first: whenever it exists, the earlier one also exists.
The symmetric argument applies to last selection. This removes dead candidates
before constructing their selection guards or comparing boundary pairs. It
uses expression identity in the same arena, not guard-equivalence solving.

The existing order qualification remains mandatory. Conditionally ordered
endpoints keep the general algorithm; mixed-direction groups keep all required
completion bounds. Demand records, communication instances and physical reuse
obligations are unchanged. This compacts the selector interface used by reuse
queries; it does not erase record provenance or finish general family-level
allocation.

For Persistent's shared candidate, the retained first/last alternatives each
fall from 1,544 to 80, boundary-pair visits from 894,583 to 1,779, and proof-node
visits from 2,046,877 to 15,749. The 21 lanes and 1,544 input records are unchanged.
The `duplicate_extrema_guards` counter records the removed selection candidates
separately from the input inventory.

The independent occurrence oracle enumerates all 256 combinations of eight
predicates, including empty/all-absent cases, 128 repeated candidates, reversed
record order, conditional occurrence order, and mixed directions. It checks
actual selected occurrences and bounds both retained selector count and newly
constructed selection arithmetic by the distinct supplied predicates on the
certified path. This guarantee is restricted to identical activities; arbitrary
guarded selectors can still require their full supplied representation.

Validation against `4ef0797b6` passes the allocation/profile, repeated-family and
candidate-request tests. Persistent's full logical traces remain identical on
cores 0, 15, 16 and 19, and physical traces pass independent matching and causal
reuse checks with IDs 0–5. Its static command counts and 4,600 physical arithmetic
operations are unchanged. These are compiler-side lifetime-summary reductions,
not fewer executed synchronization commands. Receipts are retained under
`.local/family-allocation-extrema/`.

Three serial paired runs, alternating baseline/candidate order and using the
same explicit six-ID request, measured median logical-pass time including
allocation support at 1.734 seconds before and 1.499 seconds after (about 14%
lower). No concurrent builds or tests ran during the pairs. This is one host's
compiler measurement; it does not imply a device-latency improvement.

## Guarded lifetime envelopes and pinned campaign

The constant-palette coalescer now permits one exact common activity predicate
per group. It proves that the group's source/consumer bounds exist and enclose
all member lifetimes under that predicate. Separation is proved under the
conjunction of both groups' activities. When merging different activities,
the combined envelope becomes unavailable: selecting an absent group's bound
would be unsound. The existing all-member compatibility path remains available
for that case and for overlapping envelopes. Necessary activity domains never
replace the exact predicate.

This extends the existing lifetime-summary construction to a shared residual
predicate without adding an allocation algorithm or symbolic-equivalence
solver. The new concrete-interval oracle covers 1/8/128-member groups, shared,
complementary and independent predicates, interleaved lifetimes, and true
conflicts. All four predicate valuations are checked. Separated groups with a
shared predicate have bounded cross-pipe reuse-query calls independent of their
member count; native extremum certification still visits the supplied members.

Five focused tests pass. TileLang and PyPTO core-0 logical traces remain equal
to `c088495d8`; Persistent remains equal on cores 0/15/16/19. Physical matching
and causal reuse checks pass for all those executions. Physical arithmetic
counts remain 1,563 / 1,372 / 4,600 respectively. The single Persistent compiler
pair is effectively unchanged (1.519 / 1.521 seconds); no additional Persistent
speedup is claimed. Local receipts: `.local/family-guarded-envelopes/`.

The separately pinned eleven-kernel campaign is recorded in
[the validation inventory](frontier-family-pipeline-validation.json), including
input/tool hashes, stage outcomes, generated-control counts and repeated-work
counters. It uses `c088495d8`, before this guarded-envelope extension, and the
fixed retained C++ emitter identified by its hash. All eleven logical requests
succeeded; seven completed six-ID allocation and C++ emission. GroupNorm and
decode still lack a certified six-ID assignment, prefill lacks an allocation
export, and Delta lacks a prefix-lifetime export. Delta still contains 2,473,273
logical arithmetic operations. No device execution is included, and logical
success does not close those production blockers. Campaign timings are single
watchdog-run diagnostics, not paired speedup measurements.

## Sparse shared-binding planning

Shared lookup now applies to availability planning as well as scalar lowering.
The contextual family constructor indexes published roots by expression ID and
asks about a root only when cofactoring reaches it. A binding is admitted only
at the same original cut or a dominating original cut. Each receiving cut
retains the reached keys, rather than copying every dominating shared root into
its planning cache. The expression arena, original cut and immutable binding
environment still scope those caches; a different environment needs a fresh
owner. Final lowering must supply every retained binding or fail before
constructing scalar IR.

The diagnostic inventory separately counts published planning roots, lookup
requests, candidate-cut checks and retained planning bindings. Composition
counts construction once per immutable leaf, while counting command sites per
use. Mixed leaves retain the counters of their contextual contribution. The
restriction remains important: many publications of the same root can require
multiple candidate-cut checks, and the algorithm still traverses the requested
expression DAG. This removes eager root-by-cut copying; it is not a universal
linear bound for contextual preparation.

This change does not remove shared computations that the producer explicitly
requested, compress arbitrary guarded selectors, or change the logical
communications. It addresses compiler-side binding construction. Wrapper
subtree discovery and the remaining compact-allocation milestones stay open.

Eight focused tests pass, including shared-family scaling, sparse binding
failure, guarded publication, mixed batches, repeated allocation and stage
profiling. With 512 published roots and 64 receiving cuts that each use one
root, the constructor records 64 lookups, 64 candidate-cut checks and 64 retained
bindings. Reusing a recipe doubles its command sites without double-counting
construction. Later-cut bindings, independent retry environments and missing
binding failure atomicity remain covered. The separate 128-request fixture
retains one planning and one lowered binding per request. Local receipts are in
`.local/family-shared-planning/`.

Against `80c8d5779`, TileLang and PyPTO outputs are byte-identical. Persistent
and prefill retain canonical agreement after associative Boolean-AND and
equality normalization; no command, tuple or other output change is accepted
by that comparison. Persistent records 40 retained shared bindings and prefill
58. Prefill still visits 38,835 expression nodes through lookup, with 86
candidate-cut checks: sparse bindings do not eliminate its large requested
DAG. The single serial prefill pair is 88.58 / 88.48 seconds, and Persistent is
1.52 / 1.62 seconds. These diagnostics establish neither a planning speedup nor
a device result. The existing physical/export refusals remain open.


## Shared original-cut index for recipe wrappers

Activity and invocation wrappers now reuse an immutable index of the child's
original cuts. The index records whether a cut contains a nonlocal command;
activity still applies to every cut, while invocation coordinates apply only
to communication tuples. Namespace translations and wrappers preserve those
original cuts. Appending another child forms a new union lazily, importing
already indexed subtrees and visiting each remaining recipe node once. The
index retains neither expression bindings nor an availability certificate.
Every new wrapper still plans its own expression at each applicable cut.

The old discovery walk is removed. New counters distinguish discovered nodes,
examined endpoint pieces, index imports, imported entries and availability
requests. Imported entries are counted explicitly: union construction can still
copy a supplied cut inventory, and retaining an index does not imply constant
work for every composition. Counters describe construction retained by the
accepted recipe, counting shared context objects once; they are not a ledger
of discarded failed attempts. The cache is scoped to the existing owned
compilation and unchanged original IR, with no global lookup state.

The scaling fixture uses 128 alternating activity/invocation wrappers over 64
cuts, half carrying nonlocal commands. Discovery examines one leaf and its 64
pieces; subsequent wrappers reuse its index. All 6,144 applicable availability
checks remain, and nonlocal matching tuples have the expected 65 coordinates.
The old repeated walk would examine those 64 pieces 128 times. Appending a new
cut is checked separately: the new union includes it, importing 64 old entries
and inspecting the new leaf. An unresolved invocation predicate still refuses
even when the cut inventory is cached.

Four focused family/order/repeat/profile tests pass. TileLang and PyPTO output
is byte-identical to `55915f90b`; Persistent retains canonical agreement under
the existing associative-Boolean comparison. Its retained recipe reports 18
discovered nodes, 114 examined pieces, 52 imported entries and 67 availability
requests. No paired wall-time speedup, new physical capability or device result
is claimed. Receipts are in `.local/family-wrapper-index/`. This change stays in
the private family implementation; validation used the existing incremental
build, without another public-header rebuild.

## Normalize equivalent complete physical mappings

The existing complete-map key now folds zero-stride modular terms into the
base and sums the scales of identical dynamic terms. Zero-scale and modulus-one
terms remain irrelevant. Terms are sorted by coordinate, reduced stride,
reduced phase and modulus. The full palette remains part of the key. These
operations preserve the supplied function for every coordinate; they neither
fit a new assignment nor infer reuse. The existing palette-range check bounds
all nonnegative contributions, including the constant folds and scale sums.

This lets the existing identical-family path discard arbitrary record
selection before constructing its scalar selector, even when equal functions
have different additive spellings. The emitted map still comes from an original
certified member. The shared dispatcher cache uses the same normalized key.
The common-skeleton phase-fitting path is unchanged, so combining duplicate
terms cannot accidentally narrow its supported varying-phase representation.
Different complete functions still retain their existing selection and cost
checks. This does not complete allocation on the normalized recipe graph.

A pinned baseline at `fe5976a84` emits 184 / 472 / 1,624 arithmetic operations for
32 / 128 / 512 records representing one dynamic map. The candidate emits
62 / 62 / 62 for that family. Additional forms give constant counts of 63, 40
and 41, covering maximal unsigned ordinals and modulus three. Exact ID traces
are checked with noncontiguous palettes, independently grouped consumers,
constant terms, duplicate summands, reordered terms and zero trips. These are
saved-certificate lowering fixtures, not new allocator reuse certificates.

Four allocation tests pass, including the separate causal-reuse oracle and
existing genuinely different phase/stride mappings. Physical outputs for
TileLang, PyPTO and Persistent GEMM are byte-identical to the baseline, with
1,563 / 1,372 / 4,600 arithmetic operations respectively. No additional GEMM or
device speedup is claimed. Receipts: `.local/family-mapping-normalization/`.

## Share exact prefix source captures before materialization

One sequence contributor now owns a directory of primitive source-rank captures.
Its key contains the source phase, the exact occurrence-matching predicate in
that contributor's arena, and the qualified reset loop (or whole invocation).
Repeated requests for that key reuse the state slot, reset and source update.
The old per-crossing allocation of these identical captures is removed.
Consumer predicates and choices remain separate; no publication, acquisition,
barrier or retained demand is removed by this transformation.

The preservation argument is equality of state transitions: the replaced slots
start at zero, reset at the same original invocation boundary, and receive the
same executed-source ordinal under the same exact predicate. They therefore
hold the same value at every consumer. Different matching predicates and reset
scopes remain distinct. The directory is local to one collection call, owns no
generated SSA, and cannot leak slot numbers into a fresh retry or another arena.
This does not add cross-provider capture equivalence or general expression
sharing across changed private-state bindings.

The invocation-prefix oracle now includes 4-way and 128-way disjoint guarded
partitions that refer to two primitive source occurrences. Both retain two
capture slots, while preserving independent consumer decisions. A separate
case requires four slots for two different reset scopes: an inner consumer
needs loop-local captures, while an outer consumer on another pipe needs the
whole-invocation captures to survive a later empty invocation. The independent
payload-closure and exact publication-consumption checks cover 24 additional
executions. Existing guarded, singleton and bounded-prefix tests also pass.
The fixture's second direction is explicitly included in its allowed-direction
and publication-count oracle; no matching or closure check is relaxed.

The pinned Delta probe against `a00ba0ba9` is deliberately reported separately
from those semantic tests. Capture sharing reduces private-state slots from
294 to 284, selects from 853,003 to 852,993, and total arithmetic from 2,473,273
to 2,473,233. SET/WAIT/barrier sites remain 172 / 47 / 194, and consumer sites
remain 41. Logical output is still about 153 MB. Thus duplicate primitive
captures are not the main cause of Delta's control expansion. The single serial
compiler pair is 79.41 / 83.72 seconds; it establishes no speedup. Allocation
still refuses the missing invocation-prefix lifetime export. No full Delta
command-trace equivalence or device result is claimed from this structural
probe. Receipts and the explicit refusal are in `.local/prefix-source-captures/`.

## Prefix expression construction accounting

The invocation coordinator now owns an expression profile for its staged
materialization attempt. Regional prefix selectors and bounded-lifetime guards
register their original arena/root/cut identity; bounded entries also retain
the consumer translation distance. The profile counts requests, emissions,
direct/history attempts, and scalar operations constructed and retained at
those cuts. It reports the sixteen largest constructions through the existing
control-profile output. It changes neither expression planning nor emission.

A new attempt resets emission counters while retaining recipe-registration
counts. An accepted private-state plan copies the profile, so a later failed or
successful retry cannot change the accepted plan's diagnostic snapshot. The
profile retains no generated SSA values. Bounded roots are identified before
their temporary coordinate substitution and arena rollback.

These counts cover the detached scalar blocks constructed by the two guard
emitters. They exclude publication/acquisition bookkeeping, predicate-history
updates, mathematical proof traversals and work from abandoned attempts before
the latest materialization. They are not dynamic device operation counts.

The pinned Delta diagnostic against `e02aa4b6e` produces byte-identical logical
IR (SHA-256 `12d8b4ad7637ade129b6664fed14972da1fc9640296b15337a107dac28deaae1`).
It has 294 distinct registered root/cut/translation entries and 294 emissions,
not repeated whole-root requests. The 330 emission attempts include 36 failed
direct attempts, followed by 33 proven-inactive constants and three historical
bindings. Those failed attempts construct no retained or discarded scalar IR:
constructed and retained totals are both 2,471,573 operations. The full function
contains 2,473,233 arithmetic operations, including original payload arithmetic.

One regional expression at the `pto.tcvt` cut on input line 261 constructs
600,293 operations. The next largest entries are bounded guards at consumer
distance three, with 24,630 and 24,618 operations at line 257; several other
bounded guards construct about 22,000 operations each. Whole-root memoization
alone therefore does not address this output. The next investigation must
separate common subexpressions across bounded guards from expansion already
inside the retained regional expression, while preserving their original
availability and history bindings. These measurements do not establish that
arbitrary cross-cut or cross-history sharing is valid.

All six focused prefix/private-plan/stage-profile tests pass. Added checks cover
fresh retries, frozen accepted diagnostics, bounded zero-trip elimination and
nonempty bounded emission. The single Delta run is 96.31 seconds and is only a
diagnostic, not paired speedup evidence. Logical commands, private-state slots
and output bytes are unchanged; no new physical lifetime export or device
result is claimed. Receipts: `.local/prefix-expression-profile/`.

## R1 result: SCC-local completion transfer

The production transfer now finalizes reachable SCCs in topological order,
preinstalling mandatory completions. Acyclic singletons transfer once; cyclic
components solve twice against fixed external input. Internal delta queues and
external propagation are separate, so provisional cyclic facts never reach a
downstream component. Decomposition and propagation use iterative traversals.
The old global two-solve production implementation is removed.

The existing concrete-state oracle now checks SCC placement using an independent
transitive-closure component decomposition, final summaries and conflict safety.
It separately checks selected-site containment against the old global policy
with identical children. All 16,384 enumerated graph cases, 1,000 deterministic
random graphs and a 100,000-node chain with a two-node cycle pass. The
write/read/read regression protects only the first reader. Propagation counters
check internal and external fact bounds and one/two solves per component.

The three completion transfer/region/public lit tests pass with the existing
assertions-enabled incremental candidate build. No compiler warnings were
introduced. The compliance prefilter's five brace findings in the test driver
are false positives: each reported conditional has a braced body, including
the multiline conditions. This establishes the transfer-policy migration, not
closed-child import, broader hardware protocols or device execution. Receipts:
`.local/r1-build.log`, `.local/r1-tests.log`, `.local/r1-compliance.log`.

## R2 result: native closure between bounded demand rounds

Core-chain guarded periodic analysis now closes all native paths with masked
prefix/suffix scans, then alternates k demand rounds with complete native
closure. Both query-origin kinds use this construction. Native final-edge
alternatives use strict scans including wrapped self/later starts; demand
exclusions retain winner identities and ties. Additional native prerequisites
retain the generic relaxation path. The core route stores only demand edges.

The existing scalar-score representation remains for this algorithm change. Its
checked bound accounts for transient native-wrap candidates, and multiplication
by source-pipe length still uses addition chains. The planned period/rank pair
representation is not implemented yet; do not claim the full gate bound without
charging these scaling additions.

The independent guarded periodic and frontier scaling suites pass, as do the
arithmetic-periodic conversion and order-covered periodic suites. At three
pipes and 6/12/24 sites, demand relaxations are 108/216/432 and native scan visits
are 612/1224/2448. Constructed expression counts are 1,946/4,966/10,970; emitted
operations in the construction fixture are 559/1,767/4,231. No paired compiler
speedup or device result is claimed.

Guarded rotating integration passes regular, ACC-protected, branch-local and
poison-masking cases before timing out at its historical alternating-guard
rejection case (90 seconds). A retained baseline also times out on the isolated
input (20-second diagnostic); the earlier report already records the same
90-second baseline failure. The final nonadjacent case passes separately. The
whole guarded rotating suite remains unresolved, not counted as passing.

Incremental compilation passes. Two covered-switch warnings arise in unchanged
SessionProducers and SequenceBoundary code, already recorded by the configured
baseline builds; the changed kernel introduces no warning. The compliance
checker flags the braced arithmetic-bound conditional as a parser false
positive. Receipts: `.local/r2-build.log`, `.local/r2-build-final.log`,
`.local/r2-tests.log`, `.local/r2-adapter-tests.log`, `.local/r2-validation/`.
