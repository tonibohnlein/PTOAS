# Section 5 construction contracts

## Integration follow-up (2026-10-08)

The construction implementations below do not imply complete original-IR
recognition, regional exports, or executable plans. The follow-up work proceeds
in the following order. Each milestone requires focused semantic validation,
two independent accepting reviews, and its own commit before proceeding.

| Milestone | Integration work | Status |
|---|---|---|
| 1 | Lazy arithmetic dispatch, preserve difference-bound routing, and retain bounded demand results after endpoint failure | Accepted |
| 2 | Constraint import for piecewise bounds and shared positive-step ordinal normalization | Accepted |
| 3 | Shared symbolic storage certificates and crossings between symbolic siblings | Accepted |
| 4 | Reuse ownership/phase/boundary certificates in varying repetition and whole-visit type extraction | Accepted |
| 5 | Collective guarded refresh and reusable bounded-lifetime regional results | Pending |
| 6 | Numerical composition for general sequences without unnecessary dense port closure | Pending |
| 7 | Original-IR contract audit and pinned corpus validation through available emission/allocation | Pending |

All milestones analyze the existing shared modeled accesses. They add no
exact-address admission gate or kernel-specific recognition. Successful demands
must survive unavailable queries, selectors, endpoint recipes, or allocation.
Finite visit-type tables remain demand-only when arbitrary-word queries or
executable choices are unavailable. Streaming retention at a consumer does not
make a future predicate available to an earlier SET.

The baseline full campaign at `ee26eec8a` covered 786 prepared inputs: 594
logical outputs, 560 C++ outputs, and 17 logical-analysis watchdog cancellations.
Commit `2075425c4` then restored nine C++ outputs and ten allocations by fixing
unrelated metadata rejection. Its focused rerun did not repeat the full corpus.
The benchmark set has eleven prepared inputs, including the two GEMMs: nine
logical outputs and seven C++ outputs. One main-corpus input already contains
manual synchronization and passes through unchanged. These are compiler
outcomes, not device validation or proof of class membership.

This records the five completion milestones after the original recognizer
implementation. Analysis is relative to the shared modeled accesses, including
conservative ranges. An unresolved real address is not a separate admission
condition. Recognizing a mathematical input class, constructing its demands,
exporting executable endpoints and assigning physical event IDs are separate
results.

## Implemented construction stages

| Construction | Implementation | Required input / result |
|---|---|---|
| Phased and varying regions | `RepeatedPhases`, rotating boundary and varying-region adapters | Preserve the analyzed child graph, native prerequisites and storage selectors. An enclosing ownership or phase proof is still required for outer-dependent maps. |
| Weighted repetition | `BoundingRepetition`, `NumericalWeightedRepetition` | Shared last-crossing reduction, with semantic record deduplication and native priority. Numerical chain frontiers answer displacement queries without expanding trips. |
| Numerical repeated squaring | `NumericalRepeatedSquaring`, `NumericalRepeatedExports` | A shared logarithmic-depth composition DAG. Its immutable numerical snapshot describes the selected graph; graph overlays must invalidate it. |
| Bounded-lifetime streaming | `LifetimeStream`, `ResolvedLifetimeStream`, `GuardedRanks` | Certified bounded-span generators and ordered pipe fronts. The symbolic executor uses current register inputs; the resolved executor uses stamped ring slots. Both decide retention at the consumer. |
| Finite visit types | `FiniteVisitDemands`, `FiniteVisitRecognition` | Exact child and ordered-pair demand libraries, with cached original-IR alternatives. No arbitrary-word global query or allocation is asserted. |
| Arithmetic distance intervals | `ArithmeticPeriodicConversion`, `ArithmeticPeriodicImport` | Exact distance-interval pieces become periodic records before general closure. A paused arithmetic generator stage can resume if the adapter cannot supply the requested exports. |

The streaming interface is an analysis transition. It does not make future
branch outcomes available at an earlier SET. Existing stateless window recipes
or the draft's executable-guard conditions remain necessary for endpoint code.

Arithmetic conversion separates mathematical integers from executable integer
expressions. A correct interval result survives an unavailable bounded-integer
export. Signed floor/ceiling and fixed residues preserve integer semantics.
Native completion-to-start edges remain native during reduction.

## Finite visit types

The draft's finite-visit proposition assumes a fixed list of exact child types.
For each immutable parameter valuation, every type refreshes the same
persistent written cells and contains every common present pipe. A pipe or
written cell may be absent from every type under the same parameter condition;
its guarded records then disappear without enumerating parameter cases. Other effects must be shared read-only and disjoint from every type's writers,
or certified visit-owned. Supplied inter-visit prerequisites must have exact adjacent-visit
templates.

Analyze each type once and each ordered pair of types once. Retain each child's
internal demands and the pair's reduced crossings. A word of visit types selects
these records; neither the table nor its construction expands the word length.
A refresh or pipe present in some types but absent in another under the same
parameter valuation invalidates this adjacent-visit contract: the next relevant
access may lie beyond the next visit.

This is a demand result. An arbitrary type word does not imply a compact
whole-loop query, endpoint implementation or allocation certificate. The
implementation must retain its child owners and pair records without pretending
that they satisfy the richer `RegionalAnalysis` interface. In particular, a
parent needing global reachability cannot consume the table as a complete
regional summary.

Branch-arm analysis must be independent of the outer selection predicate.
Adjacent visits may choose different arms. Reusing one symbolic Boolean for both
visits would incorrectly make opposite-arm crossings impossible. Invariant
parameters may be shared; visit-dependent values require separate bindings or
an explicitly unmet adapter obligation.

## Limits and diagnostics

| Situation | Meaning |
|---|---|
| One type omits a refresh or pipe used by another under the same parameters | Violated finite-visit criterion. |
| A boundary predicate cannot be proved under the supplied parameter context | Unproved contract obligation. |
| Symbolic effects lack a common-cell or ownership adapter | Missing regional export; not proof that no tractable class applies. |
| Slot/phase enumeration exceeds `maxRegionalSlotVisits` | Limit of the enumerating producer, not the quotient theorem. |
| A configured arithmetic profile admits only selected fixed periods | Profile/producer restriction, not exclusion from all fixed-period arithmetic. |
| A numerical ordinal, count or displacement cannot fit its representation | Representation limit; retain any independently established mathematical result. |
| An endpoint guard is unavailable at its command cut | Emission obligation; retain recognized class and computed demands. |
| No supported reuse certificate assigns the physical IDs | Allocation failure; not proof of minimum ID scarcity. |

The original-IR arithmetic-to-periodic adapter handles one common mandatory
counted loop with a positive constant step and an affine origin in its existing
parameter bindings. Its ordinal must fit the shared signed-64 representation.
The generic interval constructor
has a broader supplied-input contract. Other original-loop forms can retain the
ordinary arithmetic route; their adapter limitation must not become an interval
class rejection.

## Cost accounting

Count normalized effects, generated records, ports, explicit phases, expression
nodes and queried interfaces. Charge every deliberately materialized phase or
slot. Arithmetic-operation bounds do not include arbitrary-precision integer bit
cost for free. Shared DAG nodes and cached child results are counted once; the
cost of constructing a child or answering its query remains in the ledger.

All these routes use the existing synchronization and allocation interfaces.
Consumer-adjacent same-pipe barriers follow the implementation policy. For a
nonadjacent demand this placement can strengthen payload order, so accepting the
plan does not assert the draft's adjacent-demand order-equality premise.

## Verification of the finite-visit implementation

The independent finite-graph oracle checks 121 type words and 56,808 event pairs
against all raw ordered storage conflicts, including exact cover membership.
Additional cases cover native and nonnative prerequisites, different physical
partitions, absent refresh/pipes, shared invariant absence and failed partial
constructions. Original-IR cases check successful and unproved contracts and
cached demand-only stage reporting. Existing phased and conditional composition
regressions remain passing after propagating analysis-only requests to children.

Serial incremental compilation and both tool links passed. Independent reviewers
accepted the construction and integration. The changed-code style prefilter's
39 brace findings were inspected: their control bodies are braced; the parser
misreads nested parentheses. No line-width or whitespace findings remain.
These checks do not claim a new corpus campaign, device run or scarce-ID repair.

## Integration milestone 1 validation

Production dispatch no longer constructs arithmetic profiles before successful
native-scalar, explicit, or structural routes. Explicit diagnostic recognition
still requests all configured profiles. Periodic source eligibility is checked
before generator construction; both early source rejection and rejection after
generator construction preserve the dedicated difference-bound reducer.

Bounded demand results own their window, DAG, predicate placeholders and source
records. Whole-function coverage is checked before publishing them, and endpoint
failure cannot discard that result or silently select a weaker fallback.

Focused dispatch, arithmetic conversion, bounded endpoint/retained-predicate,
and section allocation regressions passed. Two independent reviewers accepted
the final code. The changed-code checker reports four false brace findings in
compound test conditions; all four bodies have explicit braces. Line-width and
whitespace checks are clean. Validation logs are in the local milestone ledger;
this does not constitute a full corpus or device run.

## Integration milestone 2 validation

Arithmetic loop domains now import signed-index `min`, `max` and supported
`select` bounds as exact constraint alternatives. A selected lower bound also
selects the origin of the step congruence. Existing machine-safe scalar
normalization remains responsible for the leaves; unsigned comparisons are not
reinterpreted as signed. Expansion limits report producer limitations.

Arithmetic-to-periodic import substitutes original-IV quotient coordinates into
counted-loop ordinal coordinates, retaining inequalities and congruences.
Negative and parameter-dependent origins and noncoprime stride/period pairs
retain their distinct ordinal phases. Endpoint emission uses the shared
`CountedLoop` implementation.

Serial incremental compilation and both tool links passed. Independent
occurrence/order membership checks passed 139,968 comparisons; 42 original-IR
origin/step variants passed exact adjacent-demand and paired-endpoint checks.
Existing arithmetic dispatch/conversion and guarded periodic/rotating checks
also passed. Two independent code reviewers accepted the changes.
The style prefilter's compound-condition brace findings were inspected; their
bodies are braced. No line-width or whitespace findings remain.

## Integration milestone 3 contracts

Symbolic storage exports retain physical family membership, reservation owners,
and per-cell selectors separately. Finite uniform-selector atoms require a
certificate covering every byte; a sampled byte is insufficient. Disjoint and
shared-read-only symbolic siblings preserve their selectors through composition.

Compatible arithmetic siblings export their already computed occurrence, order
and boundary-selector relations. Binary composition joins selectors on physical
cell identity, projects the cell, and reduces crossings through the children's
strict/reflexive order relations. Native pipe order across child boundaries is included.
The parent retains the resulting symbolic storage selectors for another merge.
No child instruction analysis or trip/byte enumeration is repeated.

The initial algebraic adapter requires identical parameter bindings, periods and
enclosing invocation contexts. Deferred conflicts, incoming scalar prerequisites
and uniform relationships without crossing relations remain explicit adapter
obligations; children are retained. Opaque per-cell callbacks alone do not
provide a representation for a parameterized family of crossing demands.

Visit-owned recovery now supports a bounded affine inner run with symbolic
length, preserving holes and selecting its inner ordinal algebraically. Other
coordinate shapes still need an ownership producer. Exact membership and owner
queries remain distinct when a reservation contains untouched bytes.


Milestone 3 validation: serial incremental compilation and both tool links
passed. Symbolic arithmetic sibling composition matched independently unfolded
all-event reachability and covers, including zero trips and overlapping and
disjoint windows. Ownership, read-only symbolic siblings, ordinary sequence
insertion/allocation, and phased three-level composition regressions passed.
The phased oracle was corrected to honor existing same-scalar storage protection
and to count loop operations without counting diagnostic strings. Two independent
reviewers accepted the source and these test corrections. The style prefilter's
56 compound-condition brace findings were inspected; bodies are braced. No
line-width or whitespace findings remain. This is focused validation, not a new
corpus or device campaign.


## Integration milestone 4 contracts

Periodic phase recognition discovers periods in loop bounds as well as storage
selectors. It proves whole-value periodicity before specializing counted domains;
a residue identity alone cannot justify a trip-count replacement. Queries,
selectors and endpoint filters use the same specialized domain, including empty
phases and partial outer periods.

Phase-owned storage uses a joint certificate over all phase bodies. Original
physical maps determine actual owners; only logical context is canonicalized
under the proved storage renaming. Crossing views omit certified owned effects,
while original internal demands and endpoint recipes remain intact. Exported
selectors convert actual visit coordinates to period/phase coordinates after
checking physical ownership and the active visit interval. Combined read-only
selectors retain the first/last access per pipe.

Arithmetic regional exports retain complete finite physical families alongside
symbolic families. A single unbounded piece invalidates finite coverage for its
whole physical family, not for unrelated families. Finite atom expansion remains
a charged optional adapter.

Finite visit extraction now constructs whole visits with shared prefixes,
suffixes and sequential choices. Original child analyses are cached; each type
is composed without its outer selection predicate. Repeated tests of the same
SSA predicate reuse one decision. Distinct-choice products are explicitly
materialized and their producer limit is reported separately from class failure.
Automatic storage projection checks ownership across every type, including all
persistent and residual accesses; a per-type proof cannot discharge cross-type
aliases. The resulting finite-type library remains demand-only where executable
selection and arbitrary-word global queries are unavailable.

The rotating-child boundary producer is independent of the outer length
schedule. It retains the child's original quotient and uses separately scoped
protection for cross-visit storage. The affine-length route consumes this
certificate; other schedules must supply their own boundary-type sequence proof.

Finite-neighbor composition queries the symbolic child's complete byte selectors
on the finite counterpart's support. Pair-local coverage suppresses only those
residual geometric tests; full symbolic effects and callbacks remain exported.
Uniform modeled relationships are retained. Byte expansion is charged and uses
the existing finite-interface limit, with unsupported projections left explicit.

Milestone 4 validation: serial incremental compilation and both tool links passed.
The phase oracle checked 27,776 event queries plus storage selectors. Whole-visit
types, independent/repeated decisions and explicit type limits passed. Periodic
length tests covered empty, unequal and sliced phases and compact emission.
Owned phase tests checked 20 single/two-consumer byte closures, holes, zero trips,
partial periods and rejected overlapping ownership. Existing arithmetic sibling,
varying-length, repeated-storage and numerical/regional oracle checks passed.
The rotating boundary oracle ran 1,440 comparisons; its stale expected count was
corrected. Read-only effects discharged by the shared analysis are not required
to reappear as finite arithmetic boundaries. Two independent reviewers accepted
the final source including the finite-neighbor adapter. The style prefilter's
compound-condition brace findings have braced bodies; no line-width or whitespace
findings remain. This is focused validation, not a new full corpus/device run.
