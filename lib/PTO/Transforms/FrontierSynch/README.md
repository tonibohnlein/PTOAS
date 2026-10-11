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
   established them to construct detached synchronization fragments at original
   cuts. Shared endpoint and insertion helpers preserve matching identities.
4. Route-specific allocation adapters certify reuse for the selected completed
   graph. `Shared/LogicalInsertion.cpp` invokes deferred allocation preparation;
   `Shared/PhysicalAllocation.cpp` lowers certified logical families to physical
   event IDs. `Finite/FiniteAllocation.cpp` and shared compact/handoff engines
   perform their supported allocation strategies. Capacity remains six eligible
   IDs under the default policy, with explicit failure and no scarcity repair.
5. Accepted fragments are attached transactionally. Failed preparation never
   changes the original IR. A missing emission or allocation adapter does not
   discard exact mathematical results.

Mathematics, code preparation and allocation remain separate requests inside
each end-to-end route. IDs must be assigned against the selected composed graph;
allocating children independently does not establish parent allocation.

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

## Maintaining the layout

Keep new route-specific recognition, analysis, emission and allocation adapters
in their route directory, with matching public headers. Put helpers consumed by
multiple routes and regional composition in `Shared/`. Keep selection and cache
policy in the session, rather than duplicating dispatch inside route folders.
`lib/PTO/Transforms/CMakeLists.txt` lists translation units explicitly in these
same groups; all continue to link into `PTOTransforms`.
