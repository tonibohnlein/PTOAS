# Frontier synchronization compilation routes

The source and public-header trees use the same directories. A route owns its
mathematical analysis **and** its existing synchronization preparation and
allocation adapters. These are components of one compiler pass, rather than
three independently scheduled MLIR passes.

| Directory | Paper correspondence | Responsibilities |
| --- | --- | --- |
| `Finite/` | Section 5 finite occurrence lists, including runtime guards | Explicit lifetime generators and reduction, guarded analysis and queries, original-cut endpoint preparation, finite event-ID reuse evidence and allocation |
| `Structured/` | Section 5.4 bounded lifetime and periodic storage use | Window circuits and refresh certificates; rotating and numerical periodic producers; periodic reduction; route-specific selectors, endpoint recipes and allocation certificates |
| `Arithmetic/` | Section 5.4 restricted arithmetic | Exact primitive extraction and class checks, integer relation algorithms and reduction, selectors, regional interfaces, endpoint preparation and arithmetic allocation certificates |
| `Shared/` | Common machinery and Section 5 regional composition / Section 8 orchestration support | Original occurrence and access contexts, normalization, expression circuits, sequence/repetition composition, fallback constructions, endpoint families, transactional insertion and physical event-ID allocation |
| Directory root | Section 8 request-driven dispatch | `FrontierAnalysis`, request/result ownership, structural candidate inventory, cached producers, cost selection and detached preparation |

Section 5.4 actually names three compact classes. `Structured/` groups the two
that use storage structure and numerical or circuit representations; it does
not merge their premises or algorithms. `Arithmetic/` remains a separate route.
Rotating extraction and numerical templates are producers for periodic analysis,
not additional tractable classes. Small-count expansion is a shared normalization
alternative and retains original coordinates; it is not a fourth route.

## Compilation flow and ownership

The stages below describe ownership and dependency relationships. Physical ID
lowering may run after the logical fragments have been attached, in the separate
allocation pass; detached preparation and certificate construction precede it.

1. The session constructs the shared `SyncInput` and structural index once for
   an unchanged function and policy. The supplied shared access model remains the
   authority for pipes, reads, writes and storage identities.
2. The dispatcher requests a route's form checks, exact generators, reduction
   and required mathematical interfaces. Each route retains its native result
   representation; composition requests child queries and storage selectors.
3. Route-specific preparation uses the retained demands and the structure that
   established them. Numerical and rotating root plans retain immutable family
   recipes, validated coordinate maps and command order without generating
   scalar MLIR. Other producers still construct detached fragments. Shared
   endpoint and insertion helpers preserve matching identities.
4. Route-specific allocation adapters certify reuse for the selected completed
   graph. `Shared/LogicalInsertion.cpp` invokes deferred allocation preparation;
   `Shared/PhysicalAllocation.cpp` lowers certified logical families to physical
   event IDs. `Finite/FiniteAllocation.cpp` and shared compact/handoff engines
   perform their supported allocation strategies. Capacity remains six eligible
   IDs under the default policy, with explicit failure and no scarcity repair.
5. Numerical family recipes are materialized once at insertion, after requested
   allocation support. Regional composition and physical-section projection
   explicitly materialize before rewriting concrete coordinates and endpoints;
   those adapters have not yet migrated to recipe composition. Accepted
   fragments are attached transactionally. Failed preparation never changes
   the original IR. A missing emission or allocation adapter does not discard
   exact mathematical results.

Mathematics, code preparation and allocation remain separate requests inside
each end-to-end route. IDs must be assigned against the selected composed graph;
allocating children independently does not establish parent allocation.

A sequence containing the sole whole-invocation loop can forward the loop's
retained mathematical result without constructing storage selectors. The scope
certificate checks all modeled payloads and outside operations independently of
child success. The regional identity consumes its Sequence attempt once and
continues the shared family order; it never recursively analyzes the same node.
Stronger exports stay on the same retained child graph. Preparation revalidates
the original scope, and allocation uses that graph's record namespace and existing
certificate or query interface. This identity adapter introduces no crossing
requirements or new storage representation.

## Shared descriptor and scalar facts

`SyncInput` owns the reaching valid-shape index for its unchanged function. Access
contracts, physical geometry and matrix protection consume that same descriptor
state. Identity follows the descriptor SSA value, independently of physical
allocation reuse. Equal branch states survive joins; unknown loop counts include
zero visits, and loop-local scalar values cannot survive a backedge unchanged.
Opaque calls and unknown control invalidate dynamic metadata. Queries do not
rescan descriptor users or block prefixes.

Scalar normalization is cached by block and effective index width, with injective
canonical symbols rebound to each requested region. Fixed-coordinate analyses
retain separate contexts. Rebuilding the input replaces both owners; retained
storage results keep the original descriptor owner. Source IR must outlive the
borrowed anchors.

Access-audit records distinguish declared regions from retained allocation
enclosures and report failed qualification provenance. A materialized range is a
representation of the supplied model, not a claim about exact native footprints.
These facts refine shared inputs without changing Section 5.4 algorithms or
introducing an address-precision admission gate.

## Entry points and current limits

`FrontierAnalysis` is the common session entry point. `SessionProducers.cpp`
connects requests to route implementations; `SessionPreparation.cpp` connects
retained results to route-specific detached preparation. Shared logical insertion
and physical allocation consume those prepared plans. Route adapters may reuse
another route's supported representation through an explicit conversion.

The public `pto-frontier-analysis` pass uses the session dispatcher and the
transactional module coordinator to insert logical synchronization. Follow it
with `pto-frontier-allocate="eligible-ids=0,1,2,3,4,5"` to lower supported
allocation certificates to physical commands. Missing route-specific exports
still produce explicit failures; mathematical class membership alone does not
establish executable synchronization or allocation support.

The arithmetic producer checks each completed primitive with the same
normalization and profile checks used by batch recognition. A failed primitive
stops construction of that representation. Occurrences and accesses are checked
before pairwise order and native relations; access extraction runs once per
attempt, and recipe indices follow the actual retained relation order. Later
profiles and routes remain available. Failed optional residue reconstruction restores the earlier accepted
result. Machine scalar recipes use sparse, simultaneous substitution: profile
dimension ceilings never determine allocation sizes or placeholder identities.

The counted-domain certificate also recognizes an original signed 64-bit bound
`upper = lower + c` for a nonnegative signed constant `c`. A nonempty signed
loop has exact unsigned distance `c`; wrapping across the signed maximum makes
that invocation empty. Its maximum trip count is `ceil(c / step)`, which fits signed 64 bits
because `c <= INT64_MAX` and `step >= 1`. Runtime trip expressions still use
the original bounds and preserve wrapping behavior. Certified selected-view
bound overrides use the existing range proof instead of this correlation.

Integer relations share immutable normalized constraints across endpoint-tagged
values. Derived intersections, remappings and projections own fresh results;
each value keeps its own emptiness memo. This preserves the numerical relation
API without copying all arbitrary-precision coefficients for every endpoint.

Bounded regional selectors retain the physical-refresh producer's collective
writer-coverage certificate, including its original guards, phases and interior
tail bound. A full window of at least `period + tail` visits can reuse that
certificate for writable atoms. Read-only per-pipe coverage and effect-specific
confinement still require their separate proofs; inconclusive reuse leaves the
existing exact proof available.

`pto-frontier-analysis=analysis-only` retains the Step 2 certification checkpoint:
it reports original-region mathematical outcomes and stops before emission with
the original IR unchanged. The standalone diagnostic test tools also expose
demands-only and detached-preparation requests.

Mixed recognition helpers remain in `Shared/Recognition.cpp` and
`Shared/GuardedRecognition.cpp`: they serve both finite and compact candidates.
The shared folder also retains conservative and deferred algorithm libraries;
their location does not claim that they are exact tractable-class routes.

## End-to-end regression coverage

`test/lit/pto/sync_frontier_public_pipeline.pto` exercises the public logical
and physical passes with explicit, finite guarded, numerical periodic, bounded
lifetime and arithmetic inputs. Finite guarded and nested arithmetic inputs
may select sequence composition; the arithmetic fixture also checks that
regional arithmetic construction ran. Existing command-graph oracles check
matching and dependency preservation across guards, nested loops and zero trips.

The public regression compares the complete original module after failed
preparation, intentional analysis-only stopping and allocation failure. It also
checks that guarded rotating logical commands remain available when their
allocation export is unsupported. Closed-callee and section regressions cover
nested modules, alias policies and physical execution contexts. These are
compiler and command-graph checks, not device runtime validation.

## Retained-order coverage of residual memory effects

`Structured/OrderCoveredPeriodic` can certify a periodic required order even
when the original GM footprints are not periodic. It first builds an exact
unguarded graph from invariant local cells. A separate owned certificate then
checks every original potential hazard involving an omitted GM effect: the
retained completion-to-start threshold must cover all forward occurrence pairs,
including a site's later visits. Any possible hazard outside the original loop
prevents this proof. The hypothesis cannot use an omitted effect's generator
or a uniform conflict involving that effect to prove its own redundancy.

This adapter preserves the original shared input and ordinary form diagnostics.
Its reported class is `periodic-required-order`, with representation
`order-covered-periodic`. The omitted effects are never marked disjoint or put
in `dischargedEffects`. The initial implementation supports an independent
counted loop directly in a single-block function with a return terminator,
unconditional payloads, invariant exact local cells,
and no uniform modeled relationships. Materialized local effects use the shared
physical-cell partition, so overlapping allocation SSA roots and partial byte
reuse retain their original common identity. Exactness is relative to those
supplied modeled ranges, including preexisting allocation-bound models; the
adapter does not claim a more precise hardware footprint. Missing materialized
cells retain the prior invariant-family attempt and its explicit obligations.
Counted-domain representability,
carried-state, scalar-control prerequisites and macro restrictions remain in
force. These are the scope of this implemented proof adapter, not a claim that
other inputs are outside the mathematical class.

The retained graph supports existing periodic queries, detached logical plans
and their allocation certificate. Original-coordinate GM selectors and regional
re-entry/composition adapters remain unavailable. A stronger export request
retains these exact demands and can select another supported route. Root
promotion still requires the independent whole-function scope check.

The common endpoint adapter also accepts a certified `CountedLoop` domain,
including negative or runtime lower bounds. It keeps signed span selection
and unsigned distance division, so a distance above INT64_MAX need not
overflow when the certified visit count fits. The original normalized-domain
admission, 64-bit layout and endpoint-cut checks remain in force.

The shared overlap query can additionally prove disjoint modular byte bands
for the same canonical GM base under a known 64-bit index layout. It derives
independent origin residues and validates each original linear access map;
dimension strides divisible by the common power-of-two period contribute no
residue span. Disjoint enclosing bands prove disjoint bytes across independently
varying visits, without cancelling equal SSA symbols. Unknown bases, layouts,
extents or unsupported maps retain the existing conservative overlap answer.
This optional proof refines the shared conflict query; it introduces no analysis
admission gate and does not change the retained access descriptions.

Scalar normalization refines a certified fixed-coordinate constant to a singleton
range in that query context, after the existing machine-width and operation
checks. Dependent min/max and checked arithmetic can then use the specialized
occurrence without borrowing the original loop-wide range. Unbound contexts,
wrapping operations and incompatible range facts retain existing behavior.

## Maintaining the layout

Keep new route-specific recognition, analysis, emission and allocation adapters
in their route directory, with matching public headers. Put helpers consumed by
multiple routes and regional composition in `Shared/`. Keep selection and cache
policy in the session, rather than duplicating dispatch inside route folders.
`lib/PTO/Transforms/CMakeLists.txt` lists translation units explicitly in these
same groups; all continue to link into `PTOTransforms`.

An original complete root sequence can privately consume an order-covered
periodic child through its materialized local-cell boundaries and retained
periodic queries. Omitted effects remain owned deferred obligations; every
actual sibling is checked for conflicts. This does not provide complete GM
selectors: the root retains partial storage capabilities and refuses generic
repetition, slicing, or relational conversion that requires complete storage.
Endpoint preparation uses the shared deterministic scalar replay checks at
original cuts, including late pure loop-bound definitions. Stronger unavailable
selector requests retain the same exact mathematics.

### Whole numerical words with recurring global storage

The numerical template adapter can retain a writing GM base whose original
shared byte maps have one invariant signed machine origin and fixed aligned
inner offsets. It re-extracts every original effect on that base at its original
fixed coordinates. A common power-of-two segment size must divide the machine
circle and the signed-representative cut; exact residual intervals must fit
within one segment. Unknown origin translation permutes segment identities, so
relative atom overlap is exact without treating mathematical byte addresses as
wrapped machine values. Different bases retain the original alias-policy
conflict generators, and local target protection and prerequisites are unchanged.

This certificate belongs to one complete whole invocation. Its sealed snapshot
checks the full payload/effect word, including modes, source IDs and geometry.
Relative atoms are not absolute shared storage cells and are never declared
GM-disjoint. Regional body/selectors and storage-pattern composition explicitly
refuse this representation; retained whole numerical mathematics and its direct
endpoint/allocation adapters remain available. A missing map, changing outer
origin, incomplete scope or fragment limit leaves an unmet adapter obligation.

The recurring adapter preflights a conservative materialization slice bound
before allocation. `estimated_attempted_fragments` separately records saturated
construction charges, including failed attempts and conservative slice bounds;
it is not an exact allocation or wall-time measurement. `fragments` retains the
existing published/budgeted representation count. The generic regression uses
disjoint local load/store buffers so recurring GM alone creates cross-pipe
hazards, and compares emitted logical/physical command closure against original
signed-64-bit address evaluation at wrap boundaries. Full corpus and device
validation remain separate gates.

### Whole-invocation aligned private translations

The same owned relative-storage adapter can certify an original machine row
`T + c + q * ordinal` when the loop has constant signed bounds and a positive
constant step. Decomposition preserves the original machine width and accepts
only exact same-width additive arithmetic and constant multiplication of the
origin-free ordinal term. An absent origin means literal zero; it does not mean
an unknown value. Every map of the writing base must agree on the original
invariant origin, width, byte stride and ordinal stride.

Full aligned segments conservatively enclose each nonempty access. Different
payloads with a writer must occupy different segments within a visit, and the
existing translation-disjointness proof checks all visits. An arbitrary-precision
bound prevents relative segment wrap over the complete invocation; the unknown
original origin may still cross the signed cut. Empty footprints remain in the
sealed membership snapshot but contribute no hazards. This is a sufficient
certificate; unsupported casts, changing origins, relative wrap or overlapping
segments preserve the later producer attempts.

Private effects retain an explicit certificate kind and no relative atoms. They
are not published as absolute GM discharges. The whole-invocation owner retains
all original effect maps and loop bounds, while regional selectors and generic
storage composition remain unavailable. The regression evaluates the original
signed machine addresses for zero, one and nonunit two-visit invocations under
both alias policies, and checks logical closure and physical event-ID pairing.
These are compiler-level checks; device execution and the complete corpus remain
separate validation gates.

### Shared local atom references

A numerical template's finalized local partition now owns each distinct atom
reference vector once. A cache scoped to that partition canonicalizes complete
storage ranges, including address space and base identity, before expanding
range-to-atom references. Equal keys reuse immutable backing; copied effect
records retain ownership. Source identity, access modes, invocation protection
and the number of logical occurrences remain per effect and are unchanged.

Each canonical range union must be exactly covered by complete disjoint partition
atoms. Partial
atoms, missing coverage and malformed dual legacy/shared representations are
refused. Sealed relative-storage owners compare reference contents rather than
pointer identity. No cache survives into a different partition or invocation.

The existing fragment limit charges each constructed backing ID once and one
retained handle per effect. Other maps, ranges and partition atoms retain their
existing charges. Estimated attempted work includes canonical range processing
and expansion misses, including failed construction; sharing is not reported as
fewer physical occurrences. Focused checks cover repeated keys, independent
partition index orders and bases, failed retries, lifetime after cache/form
destruction, and protected read/write parity with the legacy representation.
The numeric diagnostic also retains actual cache requests, hits, expansions and
expanded IDs on success and failure, separately from estimated attempted work.

## Opt-in stage diagnostics

Use `pto-frontier-analysis="profile-stages=true"` to write one
`frontier-profile` JSON record per analyzed function to stderr, including failed
preparations. `analysis-only=true` measures exhaustive certification separately
from the production root request. The test driver accepts `--profile-stages`
before `--certify-regions`, `--insert-logical`, or `--recognition-boundary`.
Profiling adds no IR attributes and does not change family selection.

Stage IDs and request IDs are local to one unchanged-input session. Parent stage
and request IDs identify nested work. A top-level preparation/allocation stage also
records its originating analysis request in `parent_request`. `elapsed_ns` includes descendants;
`exclusive_ns` subtracts immediate child durations, so summing exclusive times
does not count nested work twice. `first_dispatches` counts newly opened backend
attempts, including immediate refusals and internal cache reuse; it is not a
measure of mathematical work. `constructions` and `cache_hits` describe the
instrumented normalization, export, preparation, and allocation-support stages.
A null construction count means that canonical construction work is not
instrumented by that scope; it never means zero mathematical work. Session-wide
canonical builder totals are reported separately. Timings cover the named
instrumented stages, rather than every internal relation operation. These
timings do not include subsequent IR attachment, physical-ID lowering,
C++ emission, or device execution. External campaign receipts measure complete
process elapsed time and peak RSS separately.

Repeated allocation additionally publishes one `repeated-allocation-work`
inventory per invocation of its regional exporter, with the original loop
location and owning request. Counts distinguish lane/endpoint alternatives,
extrema comparisons, lane-pair matrix cells, tested repeat gaps, relative-query
requests/constructions, proof requests/computations, proof traversal steps, and
partition submatrix construction. Traversals count repeated visits, not only
distinct DAG nodes. Relative-query and proof nanoseconds describe cache-miss
work; the inventory's `work.elapsed_ns` includes the whole repeated export.
These diagnostic durations overlap the enclosing allocation-summary timing and
are not added to stage totals. Logical-only requests construct no such support.
The query/proof counters cover the lane-sharing attempt, including failures;
coalescing and the independent-lift fallback remain in the enclosing elapsed
time but do not yet contribute query/proof counters.

Ordered extrema retain original coordinates and share first-present/last-present
prefix/suffix guards after certifying strict reference order. Mixed-direction
bounds and unresolved order keep the general representation. Relative-query
reuse is local to one repeated allocation on one unchanged graph and append-only
arena, including false and unavailable answers; it never survives rollback or
an enclosing invocation change.

Outcomes retain the session diagnostic ledger and request identity instead of
copying cumulative estimate history. `costRecords()` explicitly materializes a
report snapshot. Contract-audit refreshes in a session mark the report dirty in
constant work; `result()` publishes it once at a reporting boundary. Reacquire
`result()` after a mutating analysis request before reading its audit. Standalone
program recognition keeps its eager audit behavior.


Original occurrence coordinates are bound in `Shared/OccurrenceBinding.h`,
implemented by `RegionalEvents.cpp`. Direct sequence, prefix and repeated
preparation share the same exact coordinate tests and original loop ordinal
normalization. This interface returns arena expressions without generated SSA;
activity, availability and publication protocols remain caller-owned. Necessary
endpoint scheduling domains are a separate, one-way exclusion abstraction.


Contextual scalar availability is planned by `prepareContextual` before
`emitPreparedContextual` creates code. The immutable cut recipe retains original
producers, expression IDs and required binding keys, with no generated SSA.
Recipes and their cut-local contexts are scoped to an append-only arena and an
unchanged original invocation; neither may survive rollback or source mutation.
Materialization rejects the wrong arena/cut or missing bindings before creating
operations. Payload producers and nondeterministic operations remain forbidden.
Sequence ancestor sharing uses this split directly; the compatibility emitter
calls the same planner and materializer. This does not yet replace terminal-cut
qualification or all per-alternative regional preparation.

Cleanup multiplicity can also be qualified directly on a retained exact arena
predicate, before scalar emission. `EndpointCutChoices` uses the same original
loop/branch and invariant-coordinate interpretation for that query and for
emitted command validation. Availability is a separate obligation. The legacy
emitted-predicate route remains for recipes whose singleton equality is only
exposed by relational lowering.

`GuardedPublication` separates the shared direct/early protocol policy from its
availability adapter. Finite guarded preparation retains immutable publication
and cut-expression recipes before generating scalar operations. Existing
relational and historical adapters keep their materialized acceptance route
through the same policy. Recipe lowering reuses cut-local scalar captures made
by other prepared roots; it does not replay a shared original producer twice.

The physical allocation pass runs at module scope, including in textual pass
pipelines: `builtin.module(pto-frontier-allocate{eligible-ids=0,1,2,3,4,5})`.
It owns one module transaction while staging saved families, execution contexts
and executed counters. A failure preserves the complete original module.
Function transactions copy the target and its transitive symbol
dependencies; unrelated siblings are not cloned. Function-level helper APIs
require exclusive ownership of the enclosing module.
