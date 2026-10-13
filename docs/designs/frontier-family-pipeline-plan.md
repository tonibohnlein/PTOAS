# Compact command-family pipeline

This revision supersedes the execution priorities of
[the consolidation plan](frontier-consolidation-plan.md), while carrying its
unfinished obligations forward. The implementation baseline is
`9e86b5232b92cf9b4efe27464a061d78a2ca7406`.

## Contract

Keep the existing mathematical algorithms. Build executable command-family
recipes, normalize their logical protocol, request allocation for that protocol,
then materialize generated arithmetic and commands transactionally. These are
internal stages, not additional public passes or a new dialect.

Preserve the selected payload ordering, original hardware restrictions, and
retained exact mathematics on export failure. No ordering repair, new admission
cutoffs, general symbolic solver, runtime demand analyzer, or weaker fallback
after exact retention is introduced. Necessary activity domains remain one-way
exclusion evidence; they do not replace exact guards or matching identities.

## Milestones

| Milestone | Work and acceptance | Current status |
|---|---|---|
| M0: preparation ownership | Reproduce prefill failure; remap every affected non-SSA preparation reference before deletion; validate complementary cleanup and failure atomicity; replay prefill separately from the failing baseline. | Complete: `32b38b426` |
| M1: measurement | Extend stage inventories with per-repeat lanes/extrema, candidate pairs/gaps, query construction, proof/cache work and actual traversal counts. Keep static records/sites distinct from dynamic commands and evaluated control. | Stage inventories and repeated-allocation work/traversal counters implemented; dynamic-control and complete ID-map inventories remain open |
| M2: repeated allocation | Normalize certified ordered extrema; share attempt-owned relative queries; try economical certified numerical/repeated assignments before general sharing, retaining alternatives needed by parent allocation. | Certified extrema scans and attempt-owned relative queries implemented; economical candidate policy with parent retry remains open |
| M3: family construction and binding | Plan availability and publication/cleanup before allocation; share occurrence/frame/capture/reset interpretation; migrate sequence and repeated crossings to shared recipe families before scalar lowering. | Numerical/rotating root recipes migrated in `a2ae91269`; shared occurrence-coordinate binding implemented; early regional/repeated family construction and availability planning remain open |
| M4: composed allocation and lowering | Allocate the normalized command graph; compose incoming selectors with independently certified ID maps; compare complete mapping costs; emit only live control at legal scopes. | Existing late normalization/candidate comparison retained; early composition and nonconstant tuple extension pending |
| M5: session and capabilities | Retain facts, mathematics, recipes and allocation in one session; deterministic capability upgrades; common unified/split stage functions; self-contained versioned saved artifacts; delete redundant orchestration. | Pending |
| M6: carryover and closure | Finish shared-fact and coarse-input integration, closed-child completion imports, guarded/relational rank transport and outstanding semantic oracles; run pinned eleven-kernel/corpus validation and report device results separately. | Pending |

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
