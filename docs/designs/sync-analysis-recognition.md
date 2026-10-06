> Access contract update: all routes reduce the supplied shared storage model.
> Original-effect precision is not an admission gate. See
> [Shared modeled storage accesses](sync-effect-precision.md) for representation,
> residual alias obligations, composition and cost. Historical milestone notes
> below describe the implementation at their respective commits.

# Analysis-route recognition

## Active implementation

The executable routes are explicit analysis, numerical-template periodic
analysis, direct and immutable guarded rotating-footprint analysis, exact
sequence composition, finite guarded analysis, and restricted arithmetic analysis.
They use common logical insertion. Physical allocation
supports numerical templates, direct rotating results, finite explicit/guarded/sequence
plans, and the sufficient compact strategies described in M13. Other compact
results still require a uniform allocation export before physical compilation.
Same-pipe demands always emit a pipe barrier immediately before the consumer,
including when the source and consumer are not consecutive on that pipe. The
barrier also completes intervening same-pipe payloads. Demand reduction remains
exact for the supplied model; order equality and leastness apply only when this
placement adds no ordering. Required-order queries remain valid sufficient
witnesses for event-ID reuse, even if insertion strengthens the order.

A failed route is not a missing shared memory-effect input by default. Distinguish
an input outside that route's mathematical contract from an unimplemented
normalization, regional interface, selector, or command emitter. Current aggregate
failure messages do not establish which of these occurred; inspect the anchored
recognition diagnostics and the rejecting adapter. In particular, blanket checks
on carried values or runtime lower bounds are implementation restrictions, not a
claim that the draft excludes all such loops.

The implementation is in `lib/PTO/Transforms/FrontierSynch/`:

| Stage | Entry point and implementation |
|---|---|
| Shared physical access extraction | `SyncInput::build`, in `InsertSync/`; consumed by both synchronization passes |
| Structure and route checks | `FrontierAnalysis::initialize`, `ProgramRecognition.cpp`, `PhaseIndex.cpp` |
| Numerical effect template | `NumericTemplate.cpp`, `NumericTemplateControl.cpp`, `NumericTemplateEffects.cpp`, `NumericTemplateStorage.cpp` |
| Direct rotating generators and quotient | `RotatingExtraction.cpp`, `RotatingAnalysis.cpp` |
| Guarded rotating generators and parameterized quotient | `GuardedRotatingAnalysis.cpp`, `GuardedPeriodicQuotient.cpp`, `GuardedRotatingInsertion.cpp` |
| Explicit scan, ranks and boundaries | `ExplicitAnalysis.cpp`, `ExplicitReduction.cpp`, `LifetimeScan.cpp` |
| Sequence composition and boundary reduction | `SequenceAnalysis.cpp`, `SequenceStorage.cpp`, `SequenceQueries.cpp`, `SequenceAdapters.cpp` |
| Finite guarded closure and insertion | `FiniteGuardedAnalysis.cpp`, `FiniteGuardedQueries.cpp`, `FiniteGuardedInsertion.cpp` |
| Restricted arithmetic reduction and selectors | `ArithmeticDemandAnalysis.cpp`, `DifferenceBoundRelations.cpp`, `IntegerRelations.cpp`, `ArithmeticSelectors.cpp`, `GeneralArithmeticSelectors.cpp` |
| Exact generators and reduction | `NumericTemplateAnalysis.cpp`, `LifetimeScan.cpp`, `PeriodicDemandGraph.cpp`, `PeriodicFrontier.cpp` |
| Endpoint preparation and insertion | `NumericTemplateEndpoints.cpp`, `NumericTemplateInsertion.cpp`, `LogicalInsertion.cpp` |
| Reuse certificate and physical IDs | `PeriodicAllocation.cpp`, `FiniteAllocation.cpp`, `CompactAllocation.cpp`, `AllocationCertificate.cpp`, `PhysicalAllocation.cpp` |
| Compact endpoint emission | `FamilyExpressions.cpp`, `FamilyInsertion.cpp` and `PhysicalAllocation.cpp` |

Arithmetic recognition builds its relation bundle only when a caller requests
`FrontierAnalysis::recognizeArithmetic()`. Same-policy requests reuse the result;
changing the alias policy resets it. MLIR invalidates the analysis after an
IR-changing pass. Direct callers must keep the borrowed IR unchanged for the
analysis lifetime. The full `--recognize` diagnostic requests it; the
numerical insertion path does not. This avoids constructing an unused
arithmetic candidate without changing the periodic result or adding a compiler
mode. Recognized differences, integer octagons and bounded coefficients have
production reduction and logical-insertion backends. A successful recognizer
does not discharge endpoint availability, local adjacency or physical allocation.
The arithmetic dispatcher tries the fixed period-one class first, then the fixed
period-two class if the full first contract fails; it never infers class limits
from observed maxima.

## Next route-completion milestones

These milestones extend the existing pass. Each is complete only after its
analysis, endpoint preparation, integration tests, and architecture, correctness,
and performance reviews are accepted. Recognition alone is not completion.
Physical allocation and scarcity repair are separate.

| Milestone | Deliverable | Status |
|---|---|---|
| M5 | Shared ordinary-access contracts, based GM cells, and common explicit/guarded effect acceptance | Implemented; three reviews accepted |
| M6 | Immutable guarded rotating extraction, parameterized quotient circuits, and executable logical endpoints | Implemented; three reviews accepted |
| M7 | Difference-bound reduction, endpoint selectors, and route integration | Implemented; three reviews accepted |
| M8 | Integer octagon and bounded-coefficient reduction and selectors | Implemented; three reviews accepted |
| M9 | Mixed regional composition and finite overlays using the implemented exports | Implemented; three reviews accepted |

M5 retains unknown geometry and alias relationships as explicit obligations.
Operation-specific access semantics belong to the shared IR interface consumed
by both passes; demand analysis contains no instruction footprint table.
M6 must preserve shared circuits rather than enumerate guard valuations or
expand numerical dependence distances. M7 and M8 must satisfy the draft's exact
integer projection and fixed-class restrictions. M9 accepts only exports that
its crossing analysis can consume; it does not assume a general compact theorem
for repeatedly composing arbitrary nested regions.

Each milestone is committed separately after review. The corpus report under
`.local/composition-corpus-20261005/` is the baseline for additional accepted
regions and whole functions. Partial recognition is reported separately from
successful logical insertion and from physical compilation.

## Next end-to-end milestones (M10–M13)

Baseline: `.local/route-coverage-m9-20261006/REPORT.md`, 20 nonempty original
modules with logical insertion, no original-module allocation, and two separately
pinned GEMMs with C++ output. Each milestone requires architecture, correctness
and performance acceptance before its own commit. All use the existing pass and
shared access interface; no kernel-specific recognizers or new emission modes.

| Milestone | Deliverable | Acceptance | Status |
|---|---|---|---|
| M10 | Finite allocation exports, explicit interval allocation and guarded compatibility | Existing straight-line logical successes compile; independent reuse/closure checks | Implemented; review gates accepted |
| M11 | Shared matrix valid extents, views and justified element/layout contracts | Independently checked exact ranges; affected corpus and existing mode checked | Implemented; review gates accepted |
| M12 | Shared, domain-simplified arithmetic endpoint formulas | Nested corpus example has at most 207 arithmetic operations, unchanged demands | Implemented; review gates accepted |
| M13 | Uniform allocation for supported compact results using existing queries | Nested arithmetic and representative rotating/guarded cases compile in IDs 0–5 | Implemented; review gates accepted |

Allocation preserves endpoints and order, reports unsupported proof separately
from capacity, and leaves scarcity repair and shared-pool optimization separate.
Finite guarded assignments use only proved compatibility; failure is not a proof
of minimum capacity. Existing numerical certificates remain supported. M13 does
not assume a general theorem for repeatedly composing nested summaries.
Matrix precision is strengthened only from instruction semantics and shared
geometry, retaining explicit conservative results otherwise. Arithmetic emission
retains exact integer and divisibility semantics and safe cut availability.
Each gate records recognition, logical insertion, allocation and C++ separately;
new device validation is a subsequent task. The final gate repeats the pinned
corpus and compares compile times and generated sizes.

### M10 finite allocation

Explicit producers export ordered release indices from their completion-rank
rows. The allocator uses release buckets and a free-ID list, optimal within each
direction for these ordered handoffs. Grouping costs O(h log k); the scan and
allocation cost O(h) for h handoffs and fixed hardware pipe count k.
Finite guarded and loop-free sequence producers export sufficient pairwise
compatibility using endpoint presence and their existing exact all-event queries.
Presence overapproximates retention: uncertain implications prevent reuse, never
justify it. Every previous incompatible user of an ID is checked, including
users separated by a skipped branch. Failure of this assignment does not prove
capacity infeasibility. Export costs include O(sum h_pq² * (query + implication))
work and quadratic evidence, with cut lookup and expression construction charged
separately. Sequence queries reuse the parent port graph, including internal
endpoints which are not boundary ports; child allocations are not concatenated.

The internal `pto.finite_allocation` certificate is an alternative to the existing
cyclic certificate. It records original IDs, domains and release/compatibility
evidence. Allocation checks its structure and endpoint coverage, trusts producer
semantic evidence, and emits stride-zero phases using the common lowering.
Empty/barrier-only analyzed plans need no IDs. Unanalyzed functions and stray
notifications are not accepted as empty plans. Failure changes no IR.

Validation: 20 finite logical/physical traces plus overlapping lifetime exhaustion
and malformed evidence checks. All 21 finite original-corpus logical successes
(including two empty modules) now generate C++; the nested arithmetic case awaits
M13. Both supplemental GEMMs still generate C++. Local results are in
`.local/route-completion-m10/`.

## Implementation milestones

These milestones concern exact demand analysis in current Section 5; Section 4
supplies direct insertion. Both extend the existing production pass. Recognition
alone is not completion: an accepted route must compute demands and prepare the
endpoint code needed by its caller. Neither milestone changes physical allocation
or scarcity repair.

### M5 implemented access contract

Ordinary aligned ND load/store and aligned TADD/TADDS publish selections through
`MemoryEffectOpInterface`. All operand valid-shape preconditions are checked at
the accessing instruction, including metadata updates. Native repeat-width
limits, alignment, layouts, padding and special transfer modes constrain these
contracts. Other operations retain their existing shared declarations; the
frontier analysis has no instruction-specific footprint recovery.

Finite GM cells carry their canonical function-entry pointer base. Equal byte
offsets on different bases do not merge; different bases require the configured
may-not-alias assumption before an exact cell partition is accepted. Local
storage continues to use absolute addresses across SSA roots. Explicit,
guarded, arithmetic and sequence adapters preserve this identity. Independent
GM effects can be discharged where single-occurrence scope is established;
the shared input caches those proofs.

The 2026-10-06 validation passed 102 regression checks and all three reviews.
On the original 784 prepared modules, logical insertion increased from two
empty functions to 21 whole-module successes, adding 19 nonempty modules.
Exact shared access records increased from 5,754 to 11,347 of 37,524.
Existing insertion still succeeds on all 784; its 782 C++ successes are
unchanged. The two supplemental device GEMMs preserve their dynamic commands,
physical IDs and compact C++ output. Physical allocation for the new explicit
and guarded results remains separate. Detailed results are recorded locally in
`.local/route-completion-m5/REPORT.md`.

Range partitioning costs `O(R log R + I)` plus shared geometry recovery; checking
an exact common partition is linear in the selected ranges. Current valid-shape
resolution can still require quadratic metadata work in long blocks. These
costs precede the explicit scan/reduction bound, rather than being hidden in it.

### M6 implemented guarded rotating route

A fixed potential loop body can have immutable conditional participation and
parameter-dependent slot offsets. The extractor normalizes conditional aliases
and read-modify-write accesses, selects strict previous/next writers with modular
arithmetic, and constructs guarded generator records. Slot counts and strides
remain compile-time constants; neither slots nor trip counts are enumerated.
Signed offset arithmetic uses Euclidean residues with checked intermediate bounds.

The weighted quotient preserves native start/completion order between present
payloads, including self-wraps when a pipe has only one active site. It deduplicates
conditional endpoint records and constructs shared shortest-distance and retention
circuits. All-event threshold queries distinguish unreachable from distance zero.
The pure interface requires immutable, defined guards and certified forward
integer distances; the rotating extractor establishes those conditions.

One lifetime optimization omits a self-refresh WAW when a distinct read-only
payload accesses the same rotation orbit. Under the self-predecessor guard there
is no intervening writer, and the reader's RAW/WAR edges replace that WAW.
Only self records are omitted, so their distinct-site replacement edges remain.
The witness is conditional and does not enumerate branch valuations.

Logical insertion evaluates invariant circuits once before the loop. Endpoints
retain original cuts and guards and match by record and source iteration ordinal.
The source checks `distance < trips - ordinal`; the target checks
`distance <= ordinal`. Zero trips and partial final lifetimes therefore leave no
unmatched notifications. Inactive branch predicates/offsets are masked with
`select`, preserving safety when invariant scalar code is replayed at loop entry.
Local barriers require a proof of executed-payload adjacency; a failed proof
leaves the original program unchanged.

This production adapter currently accepts one whole-function loop. It exports
exact generators, retention circuits and all-event query circuits, but no regional
storage selectors or physical allocation certificate. Conditional accumulator
protection not represented by this backend is an explicit unsupported case;
the shared hardware rule detects potential protected writer pairs. Ordinary
unprotected ACC accesses are supported. The numerical protected route remains
first, and both frozen device GEMMs retain identical plans and C++ output.

With `A` fragments and `m` potential payloads, extraction and quotient construction
use expected `O(A² + m³)` circuit operations under fixed 64-bit arithmetic.
Modular multiplication costs `O(log B)` gates when modulus bit lengths are counted.
There is no expansion in the numerical dependence distance. The insertion cost
also includes contextual scalar replay and sound Boolean implication checks:
for `g_local` local candidates and `G` DAG nodes the sufficient adjacency checks
cost at most `O(g_local * m * G)`. These checks are not included in the core
quotient bound. Only demanded circuit roots are emitted, sharing invariant work
at entry and arithmetic at each cut.

Validation passed 104 regression commands, including 166 concrete guarded-loop
traces compared with independent physical-conflict graphs, 104 quotient
valuations with all-event reachability checks, and transactional rejection cases.
The quotient tests cover singleton wraps, absent intermediate sites, equal
records, billion-sized distance bounds and arithmetic overflow rejection.
Architecture, correctness and performance reviews accepted the milestone.
Local build and validation artifacts are under `.local/route-completion-m6/`.

### M7 difference-bound analysis and insertion

The exact arithmetic backend now consumes the shared normalized primitive
relations for difference bounds. It joins physical reads/writes by memory space,
canonical base and byte, restricts conflicts to reference order, then applies the
bounded-pipe-witness construction. Every composition keeps one shared parameter
tuple and requires matching occurrence residues. Reflexive native order is kept
separate from strict required reachability; subtracting native order and two-step
strict reachability yields the minimum demands.

`DifferenceBoundRelations` uses arbitrary-precision closed integer DBMs.
Projection is the exact closed-submatrix operation. Union subtraction partitions
fixed difference directions by their thresholds, with exact empty/subset pruning;
it never replaces a union by an abstract join. Integer constants are not truncated
by intermediate closure or strict-complement operations.

Forward and inverse selectors retain original site tags and coordinate tuples.
Their guards are exact projected domains; each output coordinate is a maximum of
closed lower bounds. The minimum-demand functionality theorem makes these bounds
select a unique endpoint. First-match selection removes duplicate piece matches.
The insertion adapter prepares guards and tuple expressions at the original cuts
in detached blocks, then uses the common logical insertion mechanism. Boolean
parameters are zero-extended; index coordinates use signed widening. Checked i128
arithmetic supports the producer's period one/two and 64-bit index contract.
Unrepresentable selector constants cause explicit failure before IR mutation.

The route accepts whole arithmetic regions, including supported nested counted
loops and index-dependent branches. It does not unroll them. Local demands must
pass an exact executed-adjacency test. Programs with potentially hardware-protected
accumulator interactions remain with a route that models that protection; the
arithmetic backend does not silently reinstate those software demands. Regional
storage selectors and physical-allocation exports are not provided by this route
yet. Integer octagons and bounded coefficients are covered by M8 below.

Validation passed 107 RUN checks, including exact DBM/selector tests, 48 finite
physical-conflict oracle valuations and 12 nested conditional insertion traces.
Five full-CLI commands were skipped by the overlay runner. Both frozen GEMMs
retain their dynamic plans and C++ sizes, with analysis/allocation around 0.097 s.
Architecture, correctness and performance reviews accepted. Local evidence is
in `.local/route-completion-m7/REPORT.md`.

For fixed pipe count and arithmetic dimension, the construction is polynomial in
the primitive representation and encoded constants’ bit lengths. Its degree can be large.
A join/project on d coordinates uses O(d³) arbitrary-precision operations; union
insertion also performs subset comparisons against existing pieces. Subtraction
has a fixed-dimension threshold-arrangement cost. The cost counters record
primitive pieces, joins, projections, compositions, differences and output pieces;
they do not claim to count every inclusion test or arrangement cell. Runtime trip
counts and numerical address distances are never enumerated. Logical emission is
charged separately to the resulting selector representation.

### M8 integer octagons and bounded coefficients

The bounded-witness analysis is one policy-based implementation. Differences
retain closed DBMs; integer octagons and bounded coefficients use exact affine
inequalities and congruences. Physical identity, parameter/residue matching,
native identities, cover subtraction and adjacency checks are shared.

Octagonal projection pairs integral lower and upper bounds, with exact rounding
of doubled unary coefficients. General projection scales the eliminated
coordinate to unit inequality coefficients and keeps its divisibility condition.
It enumerates congruence-period offsets from candidate lower bounds (or upper
bounds when no lower exists), substitutes each candidate into all constraints,
and retains the resulting exact union. This is integer bound-candidate
elimination; it does not use a rational projection or a convex join. Coefficients,
constants and modular periods use arbitrary-precision integers.

For the configured fixed pipe/coordinate/coefficient/initial-period class, the
elimination periods and derived directions are class-dependent constants.
They are not dependence distances or trip counts. The fixed-depth witness
construction therefore keeps polynomial representation and bit costs, although
its degree and class constants can be large. The standalone integer-operation
API makes no polynomial claim for an arbitrary binary-encoded modulus. Boolean
subtraction uses an affine-threshold/modular arrangement rather than repeated
unbounded piece splitting. Feasibility and exact inclusion prune that arrangement.

Selectors eliminate output coordinates jointly and back-substitute rational
witnesses. Their domains retain the divisibility conditions that make each
rational output integral. Tagged first-match selection returns a consistent
complete endpoint tuple; inverse selectors recover the same original source
coordinates used by SET. The common arithmetic insertion adapter emits these
formulas using checked i128 intermediates. It rejects formulas beyond its checked
machine representation before changing IR.

Pure arithmetic is folded and interned separately at each insertion cut before
any Value escapes into an endpoint recipe. This shares repeated conditions and
coordinate formulas without moving synchronization commands or invalidating
stored endpoint identities. The fixed period-one recognizer avoids unnecessary
parity pieces when all primitives already fit that class; period-two indexing
and steps retain their existing route.

Validation and review evidence are recorded in `.local/route-completion-m8/`.
This milestone adds logical analysis and insertion. It does not supply physical
allocation certificates or solve general repeated nested-region composition.

### M9 mixed regions and finite overlays

The sequence composer now accepts immutable guarded rotating children alongside
explicit, numerical periodic and finite guarded children. The guarded producer
exports its existing shared reachability circuits, exact physical-cell boundary
selectors and detached internal endpoint recipes into the common regional
interface. Parameter offsets remain expressions. Zero trips disable selectors;
restarting a later sibling loop preserves storage identity across the boundary.
The composer partitions overlapping ranges together and reduces all crossings.

The export enumerates the finite physical slot table, not dynamic iterations.
Identical raw offsets share a poison-safe active-user mask; constructing these
masks takes O(F log(F+1)) map work and O(F) Boolean nodes for F fragments.
For V materialized fragment-slot visits and C cells, partitioning costs
O(V log V), cell incidence scanning O(CV), and selector comparisons
sum_c O(U_c^2), where U_c counts visits covering cell c. Modular multiplication
adds O(log B) circuit operations per visit for slot count B. Native boundary
selection costs O(m^2). The existing shared port-graph composition costs remain
charged separately. Arbitrary nested arithmetic tuples still require regional
selector/query adapters; a recognized whole-region arithmetic result alone
is not accepted as a child export.

`analyzeFiniteOverlay` implements the supplied-base-plus-finite-exception
contract. Every exception identifies at most one actual edge per context in the
unchanged base occurrence frame. It checks endpoint presence and forward order,
constructs one shared port closure, deduplicates conditionally identical endpoint
pairs and refines both base and added retention guards. It preserves storage
selectors and wraps required-order queries. It does not infer that a proposed
decomposition covers the physical effects, discover arbitrary exception sets,
or treat an unbounded exception family as a finite list.

Filtered endpoint preparation is an optional regional export, implemented by
finite guarded and guarded rotating producers. Both SET and inverse WAIT apply
the filter to the same actual source/target pair before emission. The overlay
adapter retains endpoint maps and matching identities, checks availability at
original cuts, and drops inherited allocation certificates. Its result can be
supplied to sequence composition. Producers without filtered recipes cannot
claim executable overlay insertion. Further nested overlays need a new filtered
export from their producer; the query-only wrapper does not invent it.

For P overlay ports, preprocessing takes O(P^2 q + P^3) query/circuit work.
Each query or base retention evaluation adds O((P+1)q + P^2) work for base-query
cost q, with logarithmic cache lookup factors. The r forwardness checks cost
O(rG) sufficient-implication work over G shared DAG nodes. Local-adjacency checks
also charge their complete DAG scans. Per-cut emission/replay and generated
output are charged separately; the guarded adapter shares its memo in place.
No SAT enumeration or trip-count expansion is used. A missing adjacency or
endpoint-availability proof leaves the input unchanged.

Validation: 115 regression RUN checks passed, with five full-CLI checks skipped.
Independent checks cover 100 mixed-region physical closures, 1,152 overlay event
pairs, and ten finite/compact overlay insertion traces. The two frozen GEMM
plans, four-ID allocation and C++ sizes remain unchanged. Architecture,
correctness and performance reviews accepted; evidence is under
`.local/route-completion-m9/`.

### Milestone 1: complete the explicit straight-line route

**Scope.** A supplied sequence of payload occurrences with resolved control and
exact effects. Straight-line functions qualify; a known loop bound does not by
itself authorize unrolling. Unresolved `if/else` is a separate guarded case.
Fuse adjacent explicit children before analysis, but initially insert only when
the accepted result covers the whole function. A child result cannot justify
insertion without accounting for crossings with its surrounding regions.

**Reuse.** `SyncInput`, `PhaseIndex`, the explicit recognizer,
`scanStorageLifetimes`, shared hardware protection, and `PreparedLogicalPlan` /
`insertLogicalSynchronization` already exist. The dedicated explicit rank reducer and route adapter now connect
these components to production logical insertion.

**Deliver.**

1. Adapt shared exact ranges to one physical-cell partition, preserving overlaps
   across distinct SSA roots and the configured GM alias policy. Retain payload
   occurrence identities, pipes and legal cuts. Reuse shared geometry code.
2. Run the existing lifetime scan, including supplied prerequisites, then add
   the Section 5.2 completion-rank reducer. Return the retained demands and
   required completion frontiers; keep storage boundary state for later reuse.
3. Prepare logical SET/WAIT endpoints and coalesced local barriers through the
   existing insertion interface. Recognize the paper's local-adjacency scope;
   do not silently replace an out-of-scope local requirement with stronger order.
4. Select this route in the existing pass for eligible straight-line functions.
   Missing premises leave the original IR intact and report the unmet contract.

**Acceptance.** Compare generator closure, minimum demands and frontier rows
against an independent all-conflict graph on small examples and generated
executions. Include RAW/WAR/WAW, RMW, cross-buffer transitive paths, physical
reuse across roots, exact subregions, shared hardware protection, empty input,
and rejected unknown effects. Check emitted logical edges and original payload
preservation. Once cell incidences are supplied, scan, reduction and insertion
have expected cost `O(n+c+e+nk+k|F*|)`; report range partitioning separately.
No quadratic conflict graph is built by the production route.

**Implemented contract.** Whole-function explicit input uses the exact enumerated
physical cells already provided by `SyncStorageEffects`. Symbolic geometry,
including unresolved GM base relationships, is rejected; the adapter does not
infer disjointness from distinct SSA roots or bypass the configured alias policy.
The pure scan/reducer supports supplied forward prerequisites; the current MLIR
adapter rejects additional value prerequisites that it does not model and supplies
only storage generators. It preserves starts and completions as distinct events.
Boundary identities and rank rows are retained for reuse, but regional composition
and an all-event query adapter were not implemented by this milestone.
Milestone 3 below adds both for supported sequences.

**Validation.** The independent reducer oracle covers 402 valid and five invalid
inputs. The MLIR integration checks physical reuse across SSA roots, subviews,
disjoint bytes, actual endpoint cuts, hardware-protected accumulation, empty
functions and rejection of symbolic effects/nonadjacent local demands. Shared
range partitioning and original-IR traversal costs are additional to the bound
above. All three milestone reviews accepted; the local suite passed 90 checks with five
full-CLI checks skipped. Committed as `75d01a49b`.

### Milestone 2: make rotating-footprint recognition executable directly

**Scope.** The paper's fixed-body, constant rotating-footprint language, with
exact disjoint slot families, a common stride per family and fixed within-slot
atoms. Trip count may remain symbolic. This is a generic access contract, not a
GEMM recognizer or a theorem for arbitrary nested-loop composition. Immutable
guarded offsets remain a subsequent circuit-backend extension.

**Reuse.** Rotating recognition, scalar selector normalization, numerical
weighted-quotient reduction and common logical insertion already exist. The
numerical-template route scans a concrete effect word. The new direct route
uses the paper's circular-phase generator extractor and shares quotient reduction
and endpoint insertion with that route.

**Deliver.**

1. Export a normalized compact family/fragment contract. Certify regular slot
   geometry from its supplied layout where possible, without enumerating all
   slots; retain an explicit unmet obligation for unsupported geometry.
2. Implement GCD/residue grouping, modular phases, sorting and circular writer
   scans from the rotating-footprint theorem. Generate and deduplicate the
   sparse RAW/WAR/WAW records, then reduce their union across all families.
3. Feed these records to the existing quotient reducer, preserving the derived
   period-one and refresh-distance certificate. Do not expand the trip count,
   the numerical dependence distance or the joint address period.
4. Adapt the retained records to the common endpoint interface, including zero
   trips, first/last iterations and nonzero lower bounds/positive steps. Publish
   each available query/selector interface explicitly; a missing regional export
   must prevent composition from treating this as a complete child result.
5. Enable logical insertion for a qualifying whole-loop function, with the
   existing invocation-completion policy. Reuse the explicit route as the
   independent finite-execution comparison, not as the production implementation.

**Acceptance.** Compare finite unfoldings with the explicit route and independent
conflict oracle. Include mixed 2/3/5-slot families, noncoprime strides, stationary
storage, offset selectors, RMW, partial exact atoms, and trips shorter than a
refresh period. A large encoded slot count/distance must leave the extractor's
record count and quotient size independent of its numerical value. Extraction
has `O(A log(A+1))` arithmetic/comparison work after per-family preprocessing;
state geometry-recovery and integer bit costs separately. Endpoint tests verify
matching and absence of terminal unmatched notifications.

After these milestones, explicit and constant rotating inputs have real
analysis-and-insertion backends. Finite guarded reduction, immutable guarded
rotating circuits, arithmetic reduction/selector synthesis, and regional
composition remain separate milestones; an applicable recognizer alone does
not establish their implementation.

**Implemented direct route.** `extractRotatingGenerators` groups fixed fragments
by family, atom and GCD residue, sorts modular phases, and emits the strict
previous-writer and next-overwrite records. `analyzeRotating` reduces their union
with the existing per-pipe quotient and binds the retained endpoints to original
cuts. Both endpoint preparers share the counted-loop arithmetic. Hardware
protection is restricted to certified stationary accumulator groups within one
body visit; it never silently suppresses an inter-iteration record.

The production pass uses this route for an eligible whole-function loop when no
accepted numerical template is available. Existing numerical templates retain
their allocation export. The direct route exports logical insertion and
completion-origin queries; it supplies neither general regional selectors nor a
physical-allocation certificate. Extra payloads outside the loop prevent this
whole-function acceptance. No compact nesting theorem is assumed.

After exact atom normalization, extraction takes expected `O(A log(A+1))`
arithmetic/comparison work, plus GCD/inverse work per family, and returns `O(A)`
records. The quotient size depends on the fixed body and records, not the
numerical slot/refresh distance or trip count. Integer costs depend on encoded
bit widths; arithmetic overflow is an explicit failure. This is a backend bound:
Step 0 still materializes some address vectors/footprints, atom partitioning has
its own cost, and existing numerical-template recognition may run first.
Contiguous constant-base slot families are certified by one interval per family;
explicit planner address lists still require inspection of their supplied entries.

**Validation and status.** All three milestone reviews accepted. The local suite
passed 92 checks (including one targeted rerun after correcting a stale diagnostic
expectation), with five full-CLI checks skipped. The extractor oracle checked 392
finite executions and eight invalid inputs. Actual-IR tests checked seven loops
and two rejection cases. Milestone 2 is complete for the scope above; physical
allocation and the remaining regional/guarded/arithmetic backends are separate.

The `--explicit-analysis` and `--rotating-analysis` test diagnostics expose actual
backend results. A recognizer's `backend=available` indicates an implementation
exists, not that the region satisfies all insertion/interface obligations.

## Recognizer interfaces

The first recognizers inspect original MLIR using `SyncInput`, `PhaseIndex`,
and `SyncStorageEffects`. They neither unfold loops nor construct demands.
The library API is `FrontierSynch/Recognition.h`; its diagnostic client is:

```sh
pto-sync-input-test --recognize input.pto
```

The client reports the function's explicit and finite-guarded candidacy and
inspects both rotating variants of every `scf.for` separately, including loops
inside rejected outer loops. Arithmetic recognition checks the whole function.
Diagnostics identify missing premises. It verifies that the original IR remains
unchanged. This is a developer test tool, not a production synchronization mode.

## Separate recognition from a ready analysis

| Recognition result | Meaning |
|---|---|
| `applicable` | The implemented input checks establish this recognizer's contract, relative to complete supplied effects and valid input execution. |
| `missing-premise` | The shape may qualify, but an effect, geometry, or arithmetic obligation remains unresolved. |
| `not-applicable` | The supplied syntax is outside the implemented class, or contradicts one of its structural premises. |

An applicable input is not Section 8's `Ready` result. The base recognizers
report `backend=unavailable` when no analysis backend is attached. The late
numeric-template path additionally exports minimum demand records,
completion-origin queries and logical endpoint recipes, described below; it
supports the allocation-only physical path for complete whole-function plans,
but does not provide general regional composition or scarcity repair. Failed
recognition neither proves physical infeasibility nor changes the selected
ordering. No compiler fallback is invoked by the diagnostic.

As with the precision layer, completeness of the shared effect producer is an
input contract. Unrepresented side-effecting operations are reported, but this
does not audit every registered operation for missing implicit effects.

## Milestones 3 and 4: regional composition and guarded analysis

**Status: Milestones 3 and 4 implemented and accepted by architecture,
correctness and performance reviewers.** Each milestone requires all three
reviews before its commit. The target is exact demands and logical
insertion; newly combined plans must not concatenate child allocation
certificates. The existing complete GEMM route remains available.

### Milestone 3: exact sequence composition

Accept sequences of fused explicit runs and supported numerical-template or
direct-rotating loops, including empty children. Analyze each child independently
of its predecessor and preserve original occurrence identities. Do not repeatedly
instantiate a summarized body inside an enclosing loop.

A regional result must export internal minimum demands, structured endpoint
recipes, exact reflexive all-event queries, per-cell first/last writers and
prefix/suffix readers per pipe, native first/last payloads, presence predicates,
constructed-interface flags and a cost ledger. Start-origin queries are required;
completion-origin ranks alone are not the interface. Periodic selectors use trip
counts and modular access structure without unfolding runtime iterations.

Use one physical-cell partition across children, preserving aliases across SSA
roots. Any slot/cell enumeration is charged. A regional numerical template must
prove its discharged GM accesses have no conflicts with outside accesses; a
whole-function discharge cannot silently be reused for a child.

Fold storage and native summaries to produce all crossing records, including
links spanning empty children and supported additional prerequisites. Reduce
crossings across every cell together through one shared boundary-event graph,
using exact child queries. Coalesce equal actual identities before the cover
test. Internal child minimum demands remain unchanged. The parent query circuit
shares the boundary closure rather than recursively duplicating child queries.

Prepare child and crossing endpoints together, remap logical identities, and
insert only after complete preflight. Keep independent SET/WAIT grouping and
root-only invocation completion. Unsupported placement, nonadjacent local
covers or incomplete external effects return an unmet obligation without IR
mutation. Hardware protection must have the same certified scope as in the
original analysis; do not infer new protection at boundaries.

Acceptance compares actual emitted order and matching against independent finite
unfoldings: explicit/loop/explicit, two loops, aliasing roots, partial ranges,
cross-buffer redundancy, RMW, read-only children, zero/one/many trips, coincident
ports, absent native endpoints and transactional rejection. Large trip counts
must not enlarge the representation. For finite port lists report
`O(L + P²(Q+d+1) + P³ + rP)` construction, with selector, effect normalization and
crossing-generation costs separately. Here P is the number of endpoint slots,
r the crossing records, Q the child query circuit size, d identity size and L
total supplied descriptions.

#### Sequence implementation and costs

`RegionalAnalysis` is the common export: an owned shared expression arena,
original occurrence anchors, guarded storage/native selectors, exact all-event
query callbacks, a detached internal-plan factory, capability flags and a cost
ledger. `composeRegionalSequence` accepts these exports; `sequenceRegionalResult`
exports its result through the same interface. Query results remain usable when
endpoint preparation fails. The normal pass reaches this route after the
existing whole-function numerical and rotating routes.

The concrete adapters cover root-level explicit runs, direct rotating loops and
regional numerical templates. They preserve the existing recognizers' effect
and control restrictions. Unknown external GM conflicts are rejected. This is
sequence composition, not repetition of a summarized inner region.

The shared graph uses already-closed child query blocks. Crossings advance the
child index, so two matrix-vector stages propagate reachability through each
populated block; empty children do not add a factor to this graph computation.
Equal actual endpoint pairs are coalesced by Boolean matrix products before
cover reduction. Graph construction/reduction costs `O(P³+rP)` after selectors
and child queries are supplied, with `O(P²)` graph matrices. Arithmetic and
interning counts are separate from integer bit costs.

Additional work is charged explicitly:

- Rotating adapters enumerate each access's `slots/gcd(stride, slots)` residues;
  numerical templates retain their existing charged inner expansion.
- For `A` physical fragments, `C` partition cells, `H` children and `I_hc`
  incidences in one child/cell, range endpoints need `O(A log A)` sorting.
  Selector construction includes `O(CA+HC+sum I_hc²)` comparisons. Importing
  supplied child partitions adds their range/cell incidence work.
- The ledger counts physical fragments, rotating residues, numerical visits,
  selector comparisons, candidate crossings, implication checks and final DAG
  nodes. Crossing generation includes its candidate loops and
  `sum_j O(A_j G_j)` Boolean implication work, where `A_j` is the number of
  conjunct obligations and `G_j` the current DAG size. This is additional to the
  graph bound above.
- A query between exported port events is `O(1)`. A query between arbitrary
  supplied occurrences makes linear-many child queries and at most `O(P²)`
  combinations against the shared closure.
- Preparation shares expressions at each cut. It emits `O(sum_cut G_cut)`
  operations, sorts newly visited DAG nodes, and charges same-pipe adjacency
  implication checks separately. Guard code inside a loop executes at that cut;
  compact static size alone is not a device-time guarantee.

The independent checker executes inserted IR and compares its entire payload
order with physical-conflict unfoldings. It includes zero trips, two loops,
RMW, reader-only regions, partial physical ranges, overlapping allocation roots,
and export/recomposition after destroying the original result wrapper. A late
unavailable endpoint predicate must reject insertion without destroying exact
queries. Actual constant bounds of 101 and 1,000,000,101 must produce equal
port/cell counts and bounded expression/emission sizes. Existing GEMM gates
remain unchanged: identical physical traces, 24/28 SET/WAIT sites and
13,700/17,182 generated C++ bytes.

### Milestone 4: finite guarded analysis

Implement the existing finite-guarded recognizer's backend for loop-free nested
if/else with exact effects and forward requirements. Reuse its predicate DAG;
keep parent-path conditions and branch exclusivity. Derive exact storage
conflicts from shared cell incidences. Include every ordered same-pipe start pair
and completion pair under endpoint presence, so skipped payloads retain native
order. Unmodeled prerequisites remain an explicit rejection.

Build shared Boolean Warshall closure and cover-selection circuits. No branch
valuation enumeration, formula expansion or satisfiability solver is required.
Export guarded demands, all-event queries, and conditional storage/native
selectors through the regional interface from Milestone 3.

For insertion, use dominating values or safely recomputable expressions at the
actual cuts, preserving parent-path evaluation. SET and WAIT must execute under
matching conditions with the same identity. An unavailable future condition may
leave exact analysis successful but insertion unsupported; do not publish
unconditionally or move a command to a cut that strengthens order.

Use these results in sequence composition. Guard-select complete arm results
when the common guard is available, never connecting mutually exclusive arms.
Otherwise run finite guarded analysis on the enclosing loop-free region.
Guarded rotating circuits and arithmetic reduction remain later milestones.

Acceptance includes conditional producers/consumers, empty arms, nested and
arm-local guards, post-conditional reuse, selected same-pipe adjacency,
recomputable predicates, unavailable future guards, and guarded children between
explicit/periodic siblings. Enumerate feasible valuations only in the independent
checker and compare required closure, covers and actual command matching. The
analysis circuit bound is O(N³) in potential events; charge effect recovery and
emitted guard duplication separately.

#### Finite guarded implementation and costs

`analyzeFiniteGuarded` accepts adjacent loop-free roots containing nested
`scf.if` operations and exact shared physical ranges. It constructs one event
pair per potential payload. Every ordered same-pipe pair contributes native
start/start and completion/completion edges under endpoint presence. Shared
Boolean closure and cover predicates give the exact selected demands without
enumerating branch outcomes. Storage selectors distinguish first/last writers
and the per-pipe readers before/after those writers, preserving physical reuse
across SSA roots. The backend exports the same `RegionalAnalysis` interface as
explicit and periodic children. Sequence traversal fuses each maximal loop-free
span containing conditionals and composes it with its compact loop siblings.

Endpoint preparation simplifies predicates using the branches enclosing the
original cut. It reuses dominating values and can replay unavailable scalar
expressions from the deterministic `arith` and `index` languages when their
operations are region-free, memory-effect-free and speculatable. This contract
excludes nondeterministic replay such as `llvm.freeze`. Nested presence uses
`select(parent, condition, false)` so a poison value from an inactive arm cannot
reach a synchronization guard. Unknown replay semantics and unavailable
structured results reject placement while preserving exact query results.
Same-pipe insertion additionally establishes executed-payload adjacency.
Preparation is detached; failure leaves the original IR unchanged. Child plans
contain no completion drain and do not carry concatenated allocation proofs.

For `N` potential payloads and `U_c` accesses to cell `c`, conflict construction
and storage selectors cost `O(sum_c U_c^2)` incidence work. Native selectors cost
`O(N^2)`; closure and cover selection use `O(N^3)` shared-circuit operations and
space. Physical partition construction is charged to the shared input layer.
Omitting unresolved GM effects additionally compares each candidate effect with
all effects in the shared input; charge those overlap queries separately (at
most the regional effect count times the program effect count).
The cubic bound describes analysis, not all emitted code. Each cut separately
charges reachable DAG cofactoring (including sorting), emitted gates and scalar
replay; cached values are shared only at that cut. Adjacency certification also
charges its sufficient Boolean implication checks. Diagnostics record analysis
nodes before preparation and total nodes/operations after cut specialization.

Validation compares all-event queries and actual inserted command closure with
independent finite graphs across nested/empty arms, conditional producers and
readers, RMW, aliasing, late scalar guards and compact loop siblings. Additional
cases check inactive-arm poison masking, rejection of nondeterministic replay,
nonadjacent local covers and unavailable future structured guards. Guard
valuations are enumerated only by tests. Independent-guard size checks exercise
8 and 24 potential branches without expanding their executions. The final
gate passed 98 targeted checks (five full-CLI checks were skipped by the local
overlay harness), 64 guarded executions, and 32 sequence executions. The
64/32 execution checks also passed with AddressSanitizer on the changed sources;
leak checking was disabled because of the local tracing restriction. Existing
frozen GEMM dynamic traces, IDs and generated-code sizes remain unchanged.

Both milestones must preserve the current GEMM dynamic traces and IDs, at most
24/28 static SET/WAIT sites and 13,700/17,182 C++ bytes for the frozen PyPTO/TileLang
inputs. Record compilation-time and representation-size changes. Physical
allocation for the new result classes is a separate interface obligation.

## Explicit regions

The explicit recognizer accepts a single-block sequence without nested regions
or ambiguous multi-phase anchors, with assigned pipes and exact supplied cell
effects. It uses the original `PhaseIndex` sequence and never makes a compact
loop explicit. Shared buffer bounds and unresolved alias predicates remain modeled inputs; their precision does not reject a region.
The result describes one invocation of the block, not repetitions of an
enclosing loop.

## Rotating storage

The rotating recognizer accepts `scf.for` loops with a constant nonnegative
lower bound, a constant positive step, no loop-carried SSA arguments, and a
fixed body without nested regions or ambiguous phase order. It normalizes the
induction variable to an iteration ordinal. The upper bound may be a parameter;
no enumeration of its values is needed. Within-loop conditionals use the guarded
route. An enclosing loop containing an inner loop is rejected by this recognizer
even if the inner loop qualifies independently.

An access identifies a fixed physical allocation or a `multi_tile_get` family.
The shared scalar analysis recovers constant-stride/offset selectors modulo a
fixed slot count when integer semantics justify the normalization. Low-bit
masks such as `i & 1` also express modulo a power of two, including for negative
operands and with the constant on either side. The rotating recognizer additionally
handles wrapping additions, subtractions and constant multiplications under an
unsigned power-of-two remainder or its equivalent mask. This preserves the low
bits without assuming that the intermediate arithmetic does not overflow.
Unproved signed remainders, arbitrary masks, runtime moduli and loop-carried
selectors remain unsupported. Invariant parameter offsets use the guarded route.

`sync_recognition_rotation_extended.pto` checks masked parameter offsets and
rejects noncontiguous masks. `sync_recognition_arithmetic_extended.pto` checks
the resulting byte relations against concrete accesses, including negative
masked operands. This extends recognition; it does not add arithmetic reduction
or guarded endpoint synthesis.

The recognizer records each fragment's family, normalized stride and offset,
slot count, and candidate refresh distance `slots / gcd(stride, slots)`.
All accesses to one family must have a common stride. Known physical slot
extents must be disjoint, both within and across families. Distinct SSA roots
with overlapping physical addresses are rejected by this initial family
recognizer, not treated as separate storage. Refining such roots into a common
family remains a possible extension.

Within-slot fragments must come from an exact shared access region. The earlier
recognizer-local `tgetval`/`tsetval` recovery shortcut has been removed. Access
regions and alias predicates come from the shared access producer; the
recognizer does not maintain a second instruction-effect table. An unresolved
footprint reports `inexact-footprint`, while structural checks still report
slot expressions, families and refresh distances.

## Guarded regions

The finite-guarded recognizer traverses loop-free, single-block regions with
nested `scf.if` arms. It retains a shared predicate DAG: each node records its
parent conjunction, condition value and selected arm. Payload phases and
rotating accesses refer to those nodes. Missing arms are empty; arbitrary loops
and other region control are rejected. Supplied footprints must be exact.

Conditions may be computed within a finite region. Recognition records whether
all conditions are available at entry; a late condition does not invalidate the
finite guarded analysis class, but recognition does not establish that paired
synchronization endpoints can evaluate their guards. Predicate evaluation must
respect the parent path, particularly for values defined inside an arm.

The guarded rotating variant permits nested conditionals when their conditions
are available before the loop or can be recomputed there using a retained recipe
of invariant, safely speculatable expressions. This check establishes immutable
participation across iterations; predicates that depend on the induction
variable do not satisfy it. The same physical
slot and footprint checks as the unguarded route apply. No retention circuits
or guarded endpoint code are generated yet.

## Arithmetic relations

`ArithmeticRecognition.h` checks a supplied bundle of exact primitive relations.
Every role must be present: context, occurrences, order, native order, reads,
writes and extra prerequisites. A present empty union denotes false, whereas a
missing role is an unmet input obligation. Completeness and equivalence to the
program, finite executions and native-order semantics remain producer contracts.
The checker does not infer these facts from arithmetic syntax.

Each primitive declares named coordinates and their roles, including parameters,
storage coordinates and auxiliaries. All coordinates count toward the configured
dimension limit. Every piece is an MLIR `IntegerSet` over quotient coordinates
with explicit residues under one common fixed period. The producer must already
have split modular constraints into these pieces. Raw division, remainder and
nonlinear products are rejected.

The caller supplies fixed limits on pipe count, dimension, period and coefficient
magnitude. These limits must not be inferred from the current kernel. All rows
are collected with checked integer arithmetic and normalized by their coefficient
GCD. Inequalities use floor division, including negative constants; an equality
with a nondivisible constant makes its piece empty. The normalized rows identify
difference constraints, integer octagons, or the broader bounded-coefficient
class. For example, `x+y<=N` has three coordinates and belongs to the broader
class when `N` is a parameter; `u=7v+3w+c` needs coefficient bound at least seven.

An accepted result retains normalized rows and indices into the supplied schema
and residue pieces. Failed recognition publishes no partial normalized bundle.
This is a representation check, not feasibility, projection, reduction or selector
synthesis. Integer coefficients are currently limited to signed 64-bit values;
overflow is reported rather than wrapped.

### Primitive extraction from kernel IR

`ArithmeticProgram.h` now connects the shared input to the arithmetic checker.
It accepts sequences and nested canonical `scf.for` loops (lower bound zero,
step one, no loop-carried SSA arguments). Upper bounds may be constants, function
entry index arguments, or enclosing induction variables. This includes rectangular
and triangular nests and restarted inner-loop buffer rotation. It analyzes the
whole nest without unfolding any trip count or repeatedly composing an inner
summary.

The producer exports a site table and relations for occurrence domains, strict
reference order, strict native reachability, physical byte reads/writes, parameter
context and an empty extra-prerequisite relation. Relation metadata identifies
source/target sites, their coordinate spans, start/completion event kinds and
address space. Order compares common enclosing iterations first and static
program positions afterward. Native relations include same-pipe start/start,
completion/completion and start/completion paths, including each payload's own
start-to-completion edge. They represent native closure, not just adjacent edges.

Fixed exact effects use the shared physical ranges, so overlapping SSA roots
refer to the same bytes. Dynamic slots require `iv rem slot_count`, actual
physical slot intervals, and exact supplied within-slot footprints. Slots need
not be evenly spaced, and distinct allocations need not be disjoint for this
arithmetic route. The slot count must divide the configured residue period.
All quotient substitutions use checked integer arithmetic.

This first producer supports **P in {1,2} and D<=8**, with at most 256 residue
tuples per conjunction. Larger producer configurations are rejected before
splitting. Pipe/dimension limits are checked before constructing site pairs.
The test client fixes `(k,D,P,C)=(8,8,2,8)`; the library takes an explicit
configuration. The broader supplied-bundle checker remains available separately.

Conditionals, noncanonical loops, computed bounds, unresolved footprints,
existing synchronization and additional SSA prerequisites are currently rejected.
In particular, a payload result consumed by another payload, loop control or a
return is not silently treated as free metadata. Rejected inputs export no
partial primitive/site/parameter bundle. Completeness of the shared operation
effect registry remains an input premise.

`pto-sync-input-test --arithmetic input.pto` emits structured test data for these
relations. `pto-arithmetic-recognition-test` exercises the generic checker and
its integer normalization. Neither tool generates demands or synchronization.

## Cost and scope

For a candidate with `N` immediate operations, `A` supplied access records, and
`S` physical slot intervals, recognition takes expected
`O(N + A + S log(S+1))` time and `O(A + S)` additional space. Family lookups
use hash maps; a sorted interval sweep checks disjointness. Expression matching
checks only the documented bounded-depth syntax. These costs exclude building
the shared input and modeled storage partition, and charge the slot intervals
explicitly: this first implementation enumerates known physical slots. It does
not claim the paper's slot-count-independent extraction bound. The output has
at most one normalized fragment per supplied record; deduplication and
generator extraction are subsequent work.

Guarded traversal adds expected linear work in visited operations, phases and
access records, plus the rotating geometry checks when requested. For arithmetic,
let `L` count expression DAG nodes across rows, `D` the configured coordinate
limit, and `M` the input schema/residue metadata. Checking takes expected
`O(M + LD)` fixed-width arithmetic operations, plus coefficient GCD operations.
Expression collection shares nodes within each row. These bounds exclude
constructing the primitive relations and residue splitting.

For the IR producer, let `n` be the number of operations, `h` the nesting depth,
`s` the number of payload sites, and `a` the supplied access/range count. At fixed
supported `D,P`, collection and construction take expected `O(nh+s^2+a)` work,
with `O(s^2+a)` relation output; residue splitting has the explicit factor
`P^D` and polynomial factors in `D`. The quadratic term constructs reference and
native order between static sites, not conflicting dynamic access pairs. Costs
are independent of trip counts. These operation counts exclude shared input
recovery and the later symbolic reduction backend.

Counted-pattern and compositional recognizers remain unimplemented. Practical
coverage next needs producer-supplied coverage or exact subregions through the
shared interface, broader bounds/views and arithmetic conditions. The arithmetic reduction/selector backend and rotating
generator/quotient backend are also still separate work.

## Validation

The lit tests cover missing access-region premises, parameterized
two- and three-slot loops, zero and large trip counts, independent assessment
of nested loops, unknown geometry, shifted-index obligations, mismatched
strides, physical aliasing across SSA roots, runtime moduli, conditional bodies
and unsupported loop steps. Existing shared-input and precision tests remain
applicable.

New tests also cover nested then/else paths, absent arms, immutable versus
iteration-varying loop guards, late finite guards, all three arithmetic classes,
negative-bound normalization, impossible equalities, residue/schema mismatches,
configured bounds and checked coefficient overflow.

The arithmetic program fixture now verifies that missing access contracts and
rejected control/prerequisite cases export no partial bundle. Its former
occurrence-level comparison relied on the removed scalar footprint shortcut;
it is no longer evidence for exact IR extraction. The supplied-relation checker
continues to test the arithmetic classes and immediate rejection of oversized
producer configurations.

## Arithmetic physical identity

Arithmetic access relations use `(memory space, canonical base, byte)` as the
physical identity. For planned local storage the base tag is absent and the
byte is an absolute address. Distinct allocation SSA values therefore still
conflict when their assigned addresses overlap.

For GM accesses the shared view/pointer mapper retains a canonical function-entry
pointer and an exact relative byte map. The arithmetic producer accepts this
base only when it agrees with the shared memory record's root. Views, pointer
casts and `addptr` retain that identity and contribute their byte offsets through
the existing shared maps. A base tag is not an extra integer variable or a
numerical pointer address. Consumers must compare both identity tags before
comparing byte coordinates. The IR owns the borrowed base values, as it owns the
payload anchors.

One canonical GM base is supported under either existing alias policy. Distinct
GM bases are disjoint only under the supplied `MayNotAlias` policy; under
`MayAlias` the producer rejects multiple bases because their relative displacement
is unknown. Opaque selected/carried pointers and mixtures of based and absolute
GM addresses also reject without exporting partial primitives. This changes no
instruction footprint or shared alias policy and introduces no new local
allocation identity. The `--arithmetic` test dump exposes the base's argument
number (`-1` denotes an absolute address); it never prints an address value.

## Fixed local byte-map materialization

The shared range materializer recognizes separable sums of constant multiples
of a coordinate, its quotient and its remainder by one positive constant divisor
per coordinate. Constant extents split exactly into full digit rectangles and a
partial final rectangle. Within each rectangle, sorted strides merge byte
intervals only when they touch or overlap; negative strides are reflected, zero
strides add no bytes, and gaps remain explicit. The result is an exact union of
physical ranges, independent of instruction names and layout names.

Cost is charged to the parsed expression, generated rectangles, dimensions per
rectangle and emitted sparse slices, rather than to every byte or element of a
contiguous tile. Rectangles and aggregate sparse slices are each capped at
4,096; parsing also has explicit depth and work limits. Arithmetic is checked,
and failure preserves the caller's output. Unsupported forms use the previous
exact materializer when possible, otherwise retain their symbolic access map.
This does not guarantee concrete ranges for every layout or resolve symbolic
origins whose simplification requires loop-domain facts.

## Late numeric inner templates

The `numeric-template` candidate keeps one outer counted loop and explicitly
expands certified numeric inner visits into a fixed body. This is the draft's
charged late expansion, not a nested-region composition theorem. It is reported
after the existing candidates and does not select a backend or set
`analysis_ready`. Each expanded payload retains its original phase, before/after
cuts and full tuple of original inner induction values. The outer induction is
`lower + step * ordinal`; bounds remain in the original IR. A statically proven
zero outer trip count produces an empty-invocation certificate without expanding the body or
normalizing unreachable accesses. The certificate is tied to those constant
bounds. Runtime-zero loops retain their body schema and use endpoint guards.

Bounds, carried scalar recurrences and branch decisions must specialize exactly
through the shared scalar analysis. Unknown state, remaining outer-dependent
branches and additional payload prerequisites reject. Shared effects are
substituted for each template occurrence. Local footprints must become exact
constant byte unions; their common atoms use memory space and physical address,
including reuse across distinct allocation SSA values. The resulting local
storage certificate has period one and writer refresh at most one outer visit.

GM effects remain in the records. A complete GM base is discharged only if it
is read-only, or if exactly one **template payload** accesses it and all of that
payload's read/write fragments share a constant translation per outer ordinal.
For the latter case, the translation magnitude must cover the entire union's
span, proving different visits disjoint. A separate reader or another expanded
visit of the same original instruction invalidates this discharge. Distinct GM
bases require the existing `MayNotAlias` policy. Because this first candidate
exports no external GM storage selectors, it is accepted only when the outer
loop is directly in the function and contains all its payloads; no surrounding
payload interactions are discarded.

Preflight counts expanded operation visits and potential payloads before
allocating template occurrences. Fixed representation limits are 65,536 visits,
4,096 payloads, 65,536 footprint/range/atom/reference fragments, and nesting depth
16; callers may only narrow them. Both branch arms contribute to the preflight
bound. Actual work is charged to expanded visits, shared scalar/map processing,
materialized ranges and atom references. Atom endpoints are sorted once, and
range-to-atom lookup uses binary search plus the emitted references. Exceeding a
limit rejects with no partial template export. Recognition supplies the effect
word to the exact numerical analysis below. It does not supply external storage
selectors, emitted endpoint code, event-ID allocation or synchronization commands.


## Exact numerical periodic demand analysis

An accepted numeric template now feeds a generic storage scan and numerical
periodic reducer. The scan accepts explicit payload identities, pipes and
physical atom read/write modes. It consolidates modes before changing state,
tracks the previous writer and the latest read-only reader on each pipe, and
emits RAW, WAR and consecutive-writer WAW generators. Native completion order
protects the earlier readers on that pipe. Deduplicated endpoint pairs retain
their storage witnesses. Supplied forward prerequisites are also supported by
the scan. Its expected cost is `O(n + c + e)` for payloads, atom incidences and
supplied prerequisites; resetting a lifetime discards its reader-map capacity.

For the constant local effect word, every written atom is refreshed in the next
visit. Scanning two visits therefore finds every generator sourced in visit
zero. Keeping those records derives the period-one generator certificate.
The template's existing whole-base GM discharge excludes internal GM conflicts;
all original payloads, including those with only discharged GM effects, remain
in native pipe order. Type numbers continue to identify the original template
payload, its phase, fixed inner coordinates and insertion cuts.

The reducer itself takes arbitrary forward numeric displacement records, not
only distance-zero/one records from this adapter. It deduplicates them, builds
start/completion vertices with native chains and wrap edges (including one-site
pipes), and computes one completion-seeded shortest-path index per pipe. A
best/second incoming-edge scan decides each record's redundancy against the
original graph, preserving ties between distinct edge identities. This avoids
rerunning shortest paths after deleting individual records and never expands
the dependence distance into an iteration window.

Canonical record sorting costs `O(g log g)`. After deduplication, graph and
arithmetic work is `O(k(V + E) log(V + 1) + kE + g)`, with
`O(V + E + km)` space for `m` payload types, `V = 2m` event types and
`E = O(m + g)` edges. The ordered priority set holds at most one entry per
vertex. Integer arithmetic is checked; representational overflow rejects the
complete result without exporting a partial proof. These are arithmetic-operation
bounds, not unit-cost claims for arbitrarily wide integers.

The result contains the exact minimum-demand records and completion-origin
reachability thresholds, including completion-to-completion queries. Unreachable
thresholds are distinct from invalid queries. Finite-prefix helpers check endpoint
presence, support partial final periods, and distinguish reflexive from strict
completion queries. Threshold lookup uses indexed arrays; a profile query adds
an expected constant-time pipe lookup. These results do not provide start-origin
queries, external storage selectors, emitted endpoint code, physical IDs or
repairs; `analysis_ready` remains false until the required interfaces exist.

The numerical test helper accepts synthetic effect words and arbitrary periodic
records. Its independent Python oracle enumerates all conflicting occurrence
pairs, adds native order, and computes finite DAG reachability and covers without
using the lifetime scan. It checks exact demands, per-pipe ranks, finite-prefix
queries and thresholds, alongside very large distances, overflow rejection,
partial periods, initial readers, read-modify-write and tied alternatives.


## Direct logical endpoint recipes

Each retained periodic record now produces a logical direct-insertion recipe.
For a cross-pipe record `a -> b` at displacement `d`, SET follows the original
source payload and executes at source ordinal `t` exactly when
`d < T && t < T - d`. WAIT precedes the original target at ordinal `j` exactly
when `j < T && j >= d`. Both identify the same logical notification by
`(record, source ordinal)`, with source ordinal `j - d` at WAIT. This namespace
belongs to one plan and must be preserved if plans are later composed. These
guards publish no notifications for absent targets, so this route needs no
cleanup for surplus publications. Same-pipe records request a barrier before
the consumer under the same-pipe adjacency premise of the model.

The numeric adapter retains each endpoint's own original phase and complete
inner-coordinate tuple. It also records the actual IR insertion points:
`Before(op) = (block, op)` and `After(op) = (block, op.next)`. Recipes sharing
that static key form a cut group. At each dynamic visit, coordinate predicates
and ordinal guards select the active instances before grouping commands;
different visits of one static cut are never coalesced. Active commands at one
cut are ordered SET, barrier, WAIT, with one barrier per pipe. Metadata or region
boundaries do not make distinct cuts equivalent automatically.

Counted-loop helpers compute trip counts and original-IV ordinals using checked
integer relations, including nonunit steps, zero trips and the full signed
bound span. Invalid steps or off-grid induction values are errors; an endpoint
whose occurrence or partner is absent is inactive. Recipe construction takes
linear work in retained records and anchor-coordinate output, with expected
linear static-cut grouping. Per-recipe ordinal selection is constant arithmetic
work; matching an anchor also reads its coordinate tuple. Ordering `q` active
commands at one cut costs `O(q log q)`. No production helper enumerates the
outer trip count.

The read-only recognition API exports selection recipes and placement
descriptions. The integrated pass below now emits their logical PTO operations;
physical event IDs are assigned by the separate pass described below. No
start-origin query interface, external storage selectors or scarcity repair is added, and the
whole Section 8 result is still not ready. The periodic index remains unchanged.
The command oracle checks fresh logical matching and models the mechanisms
separately: SET observes prior completions without gating later payload starts;
WAIT gates later starts without waiting for prior payload completions. Under
same-pipe adjacency, the resulting payload-event closure equals the selected
required order. For a nonadjacent local demand, the test checks coverage only.


## Periodic directed budgets and cyclic offsets

The numerical periodic result also supplies exact resource budgets for its
canonical direct handoffs, under the draft's command and local-adjacency
premises. The allocation summary groups retained cross-pipe records by direction
and orders each group by source type. It verifies unique source and target
phases and strict target ordering, including the wrap to the next period.
These checks establish that truncating the payload execution retains a prefix
of that direction's communication sequence. Same-pipe records consume no event
IDs and are not included in these groups.

For each phase, completion-to-start thresholds give the first later handoff
whose publication is ordered after its consumption. The maximum phase distance
is the exact uniform budget; absence of every such return path is represented
as infinity. An exact finite-prefix query counts present handoffs and clips
the phase lifetimes, handling zero trips and partial final periods without
unfolding. Infinity is separate from an arithmetic failure. Entire candidate
calculations use widened integers before minimization and subtraction; only an
unrepresentable final finite result fails. No partial budget summary is
published on failure.

The builder makes `O(sum c_direction^2) <= O(m^2)` threshold queries.
Hash grouping is expected linear; per-direction sorting is bounded by the
same quadratic sum. Stored summaries are linear in the retained cross-pipe
records. A finite-prefix budget takes linear work in that direction's phases;
a uniform capacity comparison is constant work. Integer bit costs remain
separate from these arithmetic-operation counts.

For capacity `E > 0`, a handoff of phase `r` in source period `n` has cyclic
local offset `(c*n+r) mod E`, evaluated with widened arithmetic. Capacity
sufficiency is a separate query: evaluating an offset does not establish that
reuse is safe. A zero-capacity direction is sufficient only for an empty finite
handoff sequence; no modulo operation accepts zero capacity. The logical
consumer reconstructs the source period using the existing endpoint identity.

These are per-direction budget and offset recipes, not hardware event IDs.
They assume no device capacity, perform no new hardware qualification, and do
not prove that two directions mapped to overlapping physical resources can
reuse those resources concurrently. A physical resource assignment still needs
its own compatibility and capacity checks. No PTO commands, physical mapping,
scarcity repair or new ordering is emitted here. The summary is relative to the
canonical direct plan on the supplied covers, with no exceptional entry or
interface paths that supply additional reuse.

The independent finite oracle enumerates handoffs from all-conflict DAG covers,
finds their first causally reusable successors, measures interval overlap, and
checks every same-offset pair. Random quotient tests certify these abstract
reuse-order formulas; physical canonical-plan claims additionally retain the
model's local-adjacency premise. Tests also cover infinite uniform budgets with
finite invocation budgets, large prefixes, widened offsets, malformed endpoint
ordering, and representable minima with larger intermediate arithmetic.


## Integrated logical insertion

`pto-frontier-analysis` now computes the numerical periodic demands and inserts
their synchronization in one pass. Its supported input is one accepted
whole-function numerical template, including the tested nested PyPTO and
TileLang GEMMs. The diagnostic `--recognize` continues to inspect inputs without
mutation; `pto-sync-input-test --insert-logical input.pto` runs the production
pass and prints its transformed IR. Other recognized classes still need their
analysis and endpoint backends before they can use this pass.

`prepareNumericTemplateInsertion` is the route adapter: it validates the
numerical result and constructs a `PreparedLogicalPlan`. This common plan
contains legal cuts, source/destination pipes, presence guards, logical record
and source-occurrence identities, and an explicit plan namespace. Guard and
identity arithmetic is built in owned, detached MLIR blocks. Other demand
backends can supply this same interface without numerical-template recipes.

`insertLogicalSynchronization` validates SSA availability, types, cuts and
namespace freshness before changing the function. It then installs the prepared
arithmetic and inserts commands, grouping endpoints by their actual cuts. The
producer must still establish demand correctness, dynamic endpoint matching,
safe evaluation of its arithmetic and legal placement; structural validation
cannot prove those analysis obligations. All endpoints sharing a cut must be
submitted together, and different plans must use distinct namespaces.

The numerical adapter preserves
original payloads, loops, branch decisions and allocation geometry. A command's
guard selects its own inner-coordinate tuple and checks that its partner's
outer iteration exists. Commands at a common original cut are emitted in
SET, barrier, WAIT order. Local barriers on the same pipe at that cut share
the disjunction of their guards. Zero trips execute no handoffs;
a last iteration never publishes a handoff whose consumer is absent. A whole-function
plan adds one terminal `PIPE_ALL` before return to complete outstanding payloads,
including the final store. It is outside all computation loops. A statically
empty invocation needs no drain. This is invocation completion, separate from
the internal minimum-demand relation. The
[Ascend static-Tensor contract](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html)
requires a terminal `PipeBarrier<PIPE_ALL>()`; existing InsertSync emits the same
completion mechanism.

Cross-pipe commands are `pto.logical_set` and `pto.logical_wait`. Both carry
source and destination pipes, a plan/record identity and the source outer
ordinal. The identity is logical and scoped to the function invocation; it
is not a physical hardware event ID. The IR operations survive round trips
and have side effects. Physical lowering must replace them after allocation;
EmitC and VPTO conversion reject unresolved logical commands. Same-pipe
requirements use `pto.barrier` directly. No internal all-pipe barrier is introduced.

Insertion does not assign physical IDs. When the producer has a valid periodic
capacity summary, it preserves the budgets and phase numbering in an owned
`pto.cyclic_allocation` function attribute for the second pass. Insertion
invalidates the borrowed analysis result after mutation. A rejected
input receives a diagnostic instead of a partial synchronization plan.
The independent insertion checker executes the generated integer predicates
and original structured control, observes actual command occurrences, and
compares their payload order with an all-conflict graph.

## Allocation-only physical lowering

Run the second pass immediately after logical insertion:

```text
pto-test-opt input.pto --pto-frontier-analysis \
  --pto-frontier-allocate='eligible-ids=0,1,2,3,4,5'
```

The caller supplies an eligible subset of 0–5. The pass rejects reserved IDs
6–7 even when explicitly supplied. There is no
default pool. `pto-frontier-allocate` consumes the numerical producer's uniform
cyclic certificate and allocates independently within each directed event domain.
For A2/A3, the event identity is `(source pipe, destination pipe, numeric ID)`:
equal numeric IDs in different directions identify different events. The
[2201 synchronization documentation](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)
states this explicitly. The
[TQueSync guidance](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/TQueSync.html)
recommends 0–5 for static-Tensor programming, reserving 6–7 for system/framework
uses. No numeric ID is permanently tied to a pipe pair: every supported pair
can use any eligible number. SET/WAIT matching uses the pair and number together.
The native enum's 0–7 range alone does not establish eligibility.
It lowers the logical endpoints at their existing cuts, preserving their
executed command sequence, payloads and control flow. A one-ID subset emits static flags; larger subsets
emit dynamic flags with offset `(c*n+r) mod E`. Reducing the factors modulo `E`
before multiplication prevents overflow for large source ordinals. Both ends
use the producer's source ordinal and therefore select the same ID.

Success removes every logical endpoint and the certificate. Failure emits a
diagnostic and leaves the function unchanged. Preflight validates the eligible
IDs, certificate shape, endpoint identities, complete static endpoint pairs,
index width and all directed capacities before any rewrite. Existing physical
event commands are rejected because their ID occupancy is not certified here.
The certificate is an internal producer contract, not verification of arbitrary
annotations: payload order, matching, guards and occurrence ordinals must not
change between the two passes. Other demand producers can use the same lowering
when they establish and encode this uniform cyclic contract.

The two current GEMMs have directed budgets 4, 4, 2, 2, 1, 1. Each fits the
six eligible numeric IDs; allocation uses only 0–3 across the six domains.
Summing these budgets to 14 and comparing against six numeric IDs was an
incorrect global-pool restriction. Repeating a number in different directions
does not require a cross-direction reuse witness: those are different events.
Same-direction reuse still requires the certified WAIT-before-SET order.
Finite-invocation budget minimization, shared-resource allocation for a different
target model, and scarcity repair remain separate work. Allocation failure does
not establish infeasibility under those other strategies. Statically empty
examples require no IDs.

Allocation consumes membership and coordinates prepared by analysis. For N IR
operations, R represented records and F provenance families, membership preflight
uses O(N log(F+1) + R) work apart from certificate decoding and hash-table costs.
Coordinate-to-phase preparation additionally costs
O(sum_p [r_p D_p² log(r_p+1) + E r_p D_p]) for endpoint pieces with r_p members
and D_p coordinates, where E is at most six. It emits at most
O(sum_p r_p D_p) arithmetic operations. Whole-function dominance analysis and CSE have separate
additional costs; CSE runs after lowering. Allocation does not
unroll runtime loops or reconstruct coordinates from guards. Exact irregular
maps remain decision expressions; regular maps use verified modular arithmetic.
The endpoint interface and its preparation bounds are described below.

Tests execute the emitted commands, check their closure against an independent
all-conflict graph, and check physical phases against the original analysis
record order. They match notifications by complete logical identity and require
every reuse to have a causal WAIT-before-SET path. Saved version-one IR has
separate compatibility tests; malformed certificates and insufficient pools must
fail without changing IR. Device speed is assessed separately: preserved
ordering and command traces alone do not establish unchanged latency.

## Hardware-protected storage conflicts

`HardwareProtectionBuilder` records hardware protection on physical-cell
accesses before the lifetime scan generates demands. The numerical-template
adapter supplies the already resolved accumulator cells and original operations;
the same builder can serve other occurrence producers. Recognition, periodic
reduction and logical insertion contain no matrix-specific exemption.

The first rule implements the A2/A3 [Mmad accumulation contract](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html):
consecutive accumulation into the same accumulator does not require a local
barrier when `(m / 16) * (n / 16) >= 10`. Dimensions are the valid dimensions
used by the native lowering. The implementation requires known compatible
dimensions, an initializing matrix operation, subsequent in-place accumulations,
the same accumulator type and physical cells, and `PIPE_M`. Another initializer,
an incompatible matrix operation, or an intervening access to those accumulator
cells ends the group. Separate template visits get separate group identities.
Unknown dimensions and other targets receive no exemption. This source was
checked against the A2/A3-compatible documentation preview built from
`5c07153f668d` on 2026-09-30; it does not grant all M-pipe conflicts protection.

A group certifies access protection for every ordered pair of its writers on
each grouped cell. It does not assert completion-before-start. The scan retains
every physical read and write and updates the lifetime state normally, but skips
the protected RAW/WAW witnesses. Other cells' witnesses and supplied prerequisites
remain. Filtering after reduction would be wrong: a protected accumulator edge
could already have hidden an indispensable conflict on another cell. Group IDs
are local to a builder; independently constructed inputs must remap IDs before
combination. Repeating a scope must create fresh groups.

The sparse scan remains sufficient for the remaining software requirements.
Along the lifetime-scan witness path for a conflict on one cell, protected links
before a retained software edge can be replaced by native completion order;
protected links after it by native start order. If all links are protected, all
endpoints belong to the same protected
writer group. Protection annotation adds expected linear work in the access
records and does not change the scan or quotient asymptotic bounds.

Tests separately check target and shape eligibility, accumulator resets and
intervening readers. An independent all-pairs conflict oracle removes only
certified protected witnesses, then checks the sparse generators and actual
inserted commands against the resulting required order. Cross-pipe event pairs
are still selected by reduction. UnitFlag, physical ID allocation and command
grouping are separate concerns and are not enabled by this rule.


## Endpoint-family migration

The endpoint interface preserves exact source/target coordinate maps and original
record identities before logical commands are materialized. `EndpointFamilies`
retains paired coordinate provenance. `EndpointPieces` groups SET and WAIT sides
independently by their own cut and coordinate domain. Matching namespaces group
records with the same pipes and displacement; a member label identifies the
original record within that namespace. Grouping one side does not require the
other side to share a cut. Overlapping coordinate tuples retain their
multiplicity, and cyclic proposed command orders are split before emission.
The finite planners cost O(R²(1 + D² + log(R+1))) time and O(R² + RD) space for
R retained records and coordinate rank D; they never expand the runtime trip
count.

Logical SET/WAIT accept optional index member-coordinate operands. Their identity
is (plan, static namespace/record ID, source ordinal, member tuple); old operations
have an empty member tuple and remain valid. Coordinate provenance is serialized
as versioned `pto.endpoint_families` metadata with plan-local loop and cut IDs;
borrowed compiler pointers do not cross the IR boundary. This is producer-owned
provenance, not a verification proof for arbitrary hand-edited annotations.
Version 3 adds executable pieces, each naming its endpoint side, cut and original
records. The logical command identifies its piece and evaluates the original
record label, so independently grouped SET and WAIT sides still match exactly.

Explicit producers emit singleton families. Rotating and numerical-template
producers emit one logical command per endpoint piece directly from retained
coordinate maps. The presence guard is an exact union of coordinate boxes;
the member selector is a verified affine expression or an exact decision
expression. Holes, nonunit steps and different source/target coordinates remain
represented. For N members and D coordinates, expression preparation uses
O(N D² log(N+1)) work and O(ND) emitted arithmetic in the worst case. Piece
partitioning and preparation are additional costs, separate from demand reduction.

Allocation reads the retained original-record map and the existing cyclic
certificate. It checks the partition, pipe assignments, cuts and endpoint
presence before mutation. For each endpoint piece, it composes the retained
coordinates directly with the certified physical phases. Modular affine fits
are checked at every represented tuple; an unsuccessful fit uses exact selection.
This avoids retaining a large logical label expression merely to decode it back
into a physical phase. It never reconstructs coordinates from emitted guards.
Expressions are prepared in detached blocks, with induction-value availability
checked before insertion. Dominance permits hoisting total phase arithmetic;
SET/WAIT commands keep their original guards. CSE removes dead logical selectors
and shares arithmetic after lowering. The costs are stated in the allocation
section above. Arbitrary irregular pieces may still require expressions
proportional to their member count.

Version 3 accompanies independently grouped commands. Saved version 2 paired
families and version 1 singleton provenance remain supported; old IR without
provenance is accepted as singleton families under its original allocation
certificate. There is one production emission path. Compatibility adapters
preserve existing saved commands; they do not introduce alternate modes for
newly analyzed programs. Allocation removes consumed provenance and certificates.
A failed preflight leaves the input unchanged. This changes neither the certified
resource policy nor the demand relation.

### M11 shared matrix access precision

Full, 16-aligned F16 and F32 matrix operands use shared checked access regions.
The accessing operation resolves current valid extents, including constant
allocation operands behind dynamic types and intervening `set_validshape`.
All matmul effects require compatible full extents; unresolved/partial extents,
padding and unsupported layouts retain conservative ranges. ACC accumulation
requires the existing same-destination read-modify-write contract. TEXTRACT
continues to retain its smaller selected physical range.

Both InsertSync and frontier consume these declarations. Native justification:
PTO-ISA `tload_common.hpp::TLoadGm2L1Nd2nz`, `TMatmul.hpp::TMATMUL_IMPL`
and `tstore_common.hpp::TStoreAccNz2nd`. F32 changes element width, not the
full-rectangle contract; 16 alignment avoids native K padding differences.
There is no iteration expansion. Metadata lookup retains the shared resolver's
block-prefix scan, so repeated dynamic descriptors can cost quadratic time in
block length; this change makes no linear recovery claim.

Validation: full F16/F32 byte spans, dynamic full extents, metadata shortening
and restoration, unknown extents and existing-mode smoke checks passed.
The 256 affected or previously successful corpus cases retain all prior C++
successes and add three: pypto-lib matmul, PyPTO DDR full-K, and staged matmul.
Detailed results: `.local/route-completion-m11/corpus/results.json`.

### M12 arithmetic endpoint simplification

DBM selectors coalesce two pieces only when their hull equals their exact union
and one tagged affine endpoint map agrees on both domains. The integer proof
checks each pair of complemented inequalities; a gap in the union prevents
merging. Max terms dominated throughout a piece's domain are then removed.
Boolean constants and repeated operands are folded before endpoint recipes
retain their SSA handles. General integer selectors keep their existing exact
emission path; this optimization specializes DBM selectors.

One greedy sweep performs at most quadratic piece-pair trials. A conservative
bound is O(K^2 D^7) arbitrary-precision arithmetic operations, with separate bit
costs and O(K D^2) output. This need not find the smallest selector representation.
There is no trip-count, distance or parameter-value expansion.

The nested corpus example emits 133 arithmetic operations instead of 414
(target: at most 207). Five serial samples per version measured median analysis
at 0.116 s before and 0.115 s after, while the regression job used the second
worker; these are local smoke timings, not an isolated benchmark. Independent
expanded conflict graphs cover zero, singleton and repeated visits; selector
unit checks cover integer-adjacent unions, holes, parameters, signed residues
and inverse endpoint maps. Artifacts: `.local/route-completion-m12/`.

### M13 uniform allocation for compact results

Direct rotating results export their existing numerical periodic reuse budgets.
DBM arithmetic results can export a sufficient dedicated-family assignment:
each static endpoint pair gets one ID, after proving that every earlier
retained handoff's target completion precedes every later handoff's source
start in that family. Checks cover all relation pieces and residue combinations,
share invocation parameters, and compare lexicographically ordered source
coordinates. Exact projection and union subtraction discharge coverage queries.
The source coordinate tuple remains the logical matching identity; the certified
constant assignment does not need to evaluate it to choose an ID.

Immutable guarded rotating results use a sufficient cyclic assignment. For
each possibly retained record, Boolean specialization builds a numerical witness
graph containing only payloads and generators guaranteed active under that
record's guard. A completion-to-start threshold certifies a reuse gap. Distinct
record phases occupy disjoint cyclic subsequences, including when other records
are skipped. Unknown displacement/activation proofs remain unsupported. The
assignment is uniform in loop trip count; it does not enumerate valuations.

These strategies carry explicit names in the existing cyclic export. Their
capacity failure states that the chosen sufficient assignment does not fit,
not that the logical plan requires more IDs. Failed allocation leaves the
logical IR unchanged. Scarcity repair, general-integer allocation and mixed
compact-region allocation without a whole-plan reuse proof remain separate.

Arithmetic proof work is quadratic in a family's piece count times its source
coordinate dimension and exact coverage-query cost. Union subtraction can
construct a large threshold arrangement; the polynomial guarantee requires
fixed dimension. Guarded proof work includes one numerical quotient per record
and O(R(R+M)) circuit queries. Each constant specialization costs at most O(G^2)
forced-fact work in the expression DAG; the implementation performs no search
over Boolean valuations. These are compile-time proofs, not runtime solvers.

Validation includes 35 independently checked logical/physical matching traces,
zero/single/repeated trips, guard alternatives, impossible guards, two
transactional rejection cases, and constant-size physical output for 10 versus
one billion trips. The nested arithmetic corpus case, a numerical rotating loop
and an immutable guarded loop each generate C++ with eligible IDs 0–5. No new
device execution was performed. Local artifacts: `.local/route-completion-m13/`.


Final M10–M13 validation used 786 pinned inputs (784 original corpus modules
and two supplemental GEMMs), with all input hashes verified. Physical allocation
and C++ generation succeeded for 25 original modules, including two empty
modules, and both supplemental GEMMs. All prior C++ successes were retained;
there were no timeouts. The other 759 inputs still require unsupported logical
or allocation capabilities. This is coverage progress, not complete corpus
support. The campaign used the explicit GM may-not-alias mode and eligible
IDs 0–5.

The development-overlay regression run passed 119 RUN checks. Five full-CLI
checks were skipped because the unified release CLI was unavailable. Three
serial timing samples per version on 24 common logical successes measured
median per-case analysis times of 0.0956 s before and 0.0959 s after. The nested
arithmetic case decreased from 414 to 133 arithmetic operations while retaining
four logical commands. These timings measure logical analysis preparation,
not full compilation or device execution. Architecture, correctness and
performance reviewers accepted all four milestones. Final artifacts and tool
hashes: `.local/route-completion-m13/REPORT.md` and `metadata.json`.


### Periodic GM accesses with invariant origins

The numerical template producer now retains loop-invariant integer SSA values
in GM access maps, including pure arithmetic computed inside the retained loop.
The map's first symbol is the iteration ordinal; subsequent symbols name these
invariant values. Local storage still requires concrete geometry. Shared scalar
normalization removes signed min/max clamps only when proven ranges select the
same operand throughout the loop; it preserves unknown or overflowing cases.

For a GM base written by one payload type, the existing translation test may
cancel a common invariant origin. All maps must agree on that origin and its SSA
symbols, their relative offsets/extents must be concrete, and translated spans
must be disjoint. Read-only bases retain their existing discharge. Distinct GM
bases still require the supplied non-alias policy. This yields the same exact
periodic generators, quotient reduction, endpoint recipes and allocation path;
no demand approximation or kernel-specific pattern is introduced.

The pypto-lib prefill kernels `save_prefill_last_token_group` and
`save_prefill_last_token_single` now generate C++ through this path. Each retains
one loop with two payload types and period one. Their demands are load-to-store
readiness and store-to-next-load UB release. The emitted code has two SET and
two WAIT sites and uses EVENT_ID0 for both directions; tests check causal
consumption across directions as well. The existing invocation-completion
barrier remains. No device validation is claimed.

Validation covers ten independently checked traces, overlap/clamp/alias
rejections, six scalar range checks and 120 regression RUN checks. Five full-CLI
checks were skipped. All 49 pypto-lib modules and all prior C++ successes were
rerun: four pypto-lib modules now generate C++ (two new), and all prior successes
remain. Artifacts and remaining class-specific blockers are recorded in
`.local/pypto-lib-tractable/REPORT.md`.


### Scalar prerequisites and contiguous periodic intervals

Scalar SSA dependencies are classified using the shared phase pipe and result
kind. A synchronous scalar phase supplies native completion-to-start edges;
other scalar-producing phases require explicit completion demands. Both kinds
follow pure value expressions and conditional results to their payload users.
Loop-carried values still need an occurrence map. Storage conflicts remain
independent: a vector writer must finish before a scalar read of that storage.
The explicit, periodic, finite-guarded and arithmetic reducers retain these
native edges when testing demand redundancy.

Rotating recognition also accepts physical address maps of the form
`base + bank_bytes * ((stride * i + offset) mod banks)` from the shared access
representation. Equal physical families are unified across allocation SSA
roots, and views keep their exact within-slot byte ranges. This uses no
instruction-specific or kernel-specific recognition. MLIR integer-range
interfaces and the index data layout establish nonwrapping arithmetic; values
without a proof remain symbolic.

The sequence route can partition a zero-based, unit-step loop at one invariant
comparison threshold. Supported predicates compare a proved `i + c` (`c >= 0`)
with an entry-available bound, using signed or unsigned `<`, `<=`, `>=` or `>`.
All varying branch predicates must describe the same split. Invariant branches
can remain guarded inside either interval. More general changing predicates
and nested loop bodies keep their existing route obligations.

Each half-open interval is analyzed with its fixed branch choices and exported
through the existing periodic storage selectors and all-event queries. Endpoint
recipes filter **both** endpoints to the interval. Original occurrence ordinals,
branch cuts and loops are retained. The sequence composer generates and reduces
all crossing edges, including those to explicit prologue/epilogue payloads.
Clamping the split handles zero trips, empty intervals and partial final bank
cycles without expanding the runtime trip count. Numerical interval summaries
also preserve cyclic allocation witnesses and their lifetime envelopes;
unsupported allocation proofs or insufficient capacity remain explicit failures.

This is an application of contiguous-region composition and periodic interval
locality, not a rule for repeating arbitrary compact nested regions. The cost is
that of at most two periodic analyses, finite slot-boundary export and the
existing crossing reducer. Runtime trip count does not control analysis size.
The materializing regional adapters preflight a cumulative limit of 256 slot
visits per child before constructing selectors. Larger compact bank families
need a compact selector representation; they remain eligible for standalone
quotient analysis. Exported intervals carry their first original ordinal so
public composition preserves incoming scalar prerequisites as well.

Validation of this increment (2026-10-06):

- The unchanged audited `elementwise_pipeline` input completes logical analysis,
  allocation with eligible IDs `0..5`, and PTO C++ emission. Its original loop is
  retained; the interval split does not expand its payload body.
- `causal_conv1d_decode` completes logical analysis. Its current allocation
  certificate does not fit the supplied six-ID capacity. This is an allocation
  limitation, not a proof that every valid plan needs more than six IDs; scarcity
  repair is still unimplemented.
- Existing InsertSync also generates C++ for both unchanged inputs.
- Regression checks cover 56 boundary-guard executions and their physical-ID
  reuse, six physical-bank alias executions, scalar prerequisites, and the
  existing explicit/periodic/guarded/arithmetic/sequence oracle suites. The
  shared-interface checks exercise both GM alias policies with loop-dependent
  offsets: canonical pointer provenance survives those offsets, while unknown
  pointer provenance remains conservative.

These are host analysis, emitted-order and C++ generation checks. They do not
constitute device execution or performance measurements.
