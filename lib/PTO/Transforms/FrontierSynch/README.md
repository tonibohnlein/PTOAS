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

## Maintaining the layout

Keep new route-specific recognition, analysis, emission and allocation adapters
in their route directory, with matching public headers. Put helpers consumed by
multiple routes and regional composition in `Shared/`. Keep selection and cache
policy in the session, rather than duplicating dispatch inside route folders.
`lib/PTO/Transforms/CMakeLists.txt` lists translation units explicitly in these
same groups; all continue to link into `PTOTransforms`.
