# Canonical synchronization covering

## Objective

CanonicalSync treats synchronization selection as completion covering over a
fixed kernel execution order. Covering is the only selection path: the
`pto-canonical-sync` pass builds the graph, generates mechanisms, solves a
grounded constrained set-cover instance, allocates event IDs, and verifies
the selected plan before emission. There is no alternate selector and no
solver option; the pass options are `event-id-num-max`,
`assume-distinct-gm-args-noalias`, and `assume-all-gm-accesses-noalias`, and
`pto-print-canonical-sync-plan` exposes the covering diagnostics through its
`view` option (`view=covering`, text format only). Version one does not
change physical addresses, pipe assignments, or operation order. A later
memory planner may rebuild the demand set for another placement, and a later
scheduler may replace issue-order edges without changing coverage or
mechanism semantics.

## Graph

Each schedulable operation is one weighted node carrying its issue resource
(pipe), scope, timeline order, control guard, and the set of event
destinations that can observe its completion. Zero-distance structural edges
form a DAG. A positive-distance edge belongs to one structured loop and
connects different dynamic occurrences; a positive-distance self-recurrence
is therefore valid even though a zero-distance self-edge is not.

Structured control is represented by conjunctions of branch alternatives. A
node or edge can contribute to a demand proof only when the joint execution
condition of the demand endpoints implies its guard. Unsupported unstructured
control flow and unknown asynchronous effects remain fail-closed.
Control IDs are static IR identities; virtual loop unrolling contextualizes a
literal by its dynamic copy, so alternatives from different iterations are not
treated as mutually exclusive merely because their static control ID matches.

Scopes form a graph-owned tree. Each scope records its parent, a
must-execute-within-parent fact, a loop flag, and an optional timeline
interval covering the orders of the operations it contains; resource-use
lifetimes and component decomposition are expressed over these timelines.
Each control records the scope in which it is evaluated. A recurrence names a
non-root loop scope that encloses both endpoints; sibling regions cannot
manufacture a recurrence. Edge and demand guards preserve source- and
target-occurrence conditions separately. Controls inside the recurrence scope
are contextualized per virtual copy, while outer controls remain shared.
Coverage paths may traverse nested operations only when the
must-execute-within-parent fact is proven by the IR provider; unknown or
potentially zero-trip nested loops remain fail-closed.

Structural edges have explicit temporal semantics in three kinds. A
completion-supply edge establishes completion. Completion-preserving issue
order can carry an already established completion fact but is not traversable
before completion has been established; it supports merged waits but cannot
aggregate earlier operations into a later event set. Non-completion-preserving
issue order cannot participate in a completion proof. This distinction avoids
treating command issue order as hardware completion order.

The graph stores immutable completion demands separately from structural
edges. Demands are SSA, RAW, WAR, or WAW, each with a recurrence scope and
iteration distance. Memory demands additionally carry storage provenance:
storage domains model address spaces, storage accesses record one exact
physical extent of one normalized logical access (with a family identity that
distinguishes logical accesses reusing the same interval), and storage
witnesses record the overlapping byte interval between one source and one
target access. A demand states whether its witness list is complete;
incomplete provenance disqualifies it from slot-protocol coverage. Graph
structure is frozen before mechanism construction, and generation counters
plus an RAII edge transaction keep candidate insertion append-only and
rollback-safe.

## Mechanisms and coverage

An event contributes only its explicitly verified source-to-target completion
edges. A barrier contributes a completion cut on one pipe. Verified protocols
(ownership cycles and slot protocols) remain atomic candidates whose actions
are certified as a whole. The mechanism universe assigns one dense identity to
each mechanism; adding a candidate atomically attaches all of its
completion-supply edges to the graph, so partial protocol selection is not
representable. Directed event domains and their lifetimes are stored
independently of supply, while conflicts encode alternative protocol
implementations. The side-effect-free protocol verifier runs before commit,
the universe validates construction epochs and revalidates after external
graph mutation, and frozen selection evaluators record both graph and universe
generations and fail closed after either changes.

Every non-barrier supply edge is bound to a resource-use lifetime with
matching endpoints, scope, and recurrence distance. Buffer-token pool
identities must be globally unique; a shared pool spanning multiple resource
domains is rejected until feasibility and component decomposition can model
it jointly (no production mechanism uses buffer tokens today). A barrier
names a concrete anchor on the pipe that it drains. The covered target may be
on another pipe, but its timeline position, scope, and control guard must
guarantee that the barrier executes before it; conditional and potentially
zero-trip nested anchors fail closed.

Resource feasibility is exact for the modeled inclusive interval lifetimes:
the maximum weighted overlap equals the required number of interchangeable
IDs. Unique reservations inside the hardware budget reduce availability;
out-of-range reservations remain diagnostic. A feasible result also carries
the deterministic nonreserved physical-ID assignment (expired IDs are reused
in order); emission consumes that assignment rather than running a different
allocator.

For a demand at loop distance `d`, the coverage oracle evaluates a virtual
`d+1`-copy DAG. Reachability state records whether the path has crossed a
completion edge; a demand is covered only when its target is reachable with
that state set. The oracle returns witnesses for covered demands and
reachable-cut certificates for uncovered demands.

## Mechanism universe generation

The universe is generated from the conservative demand set before no-alias
filtering. Alias information can deactivate demands but cannot remove a
candidate that could still cover another active demand transitively.

Candidate sources, in construction order:

- **Per-requirement canonical mechanisms.** For each conservative
  requirement, same-pipe requirements without hardware completion get a
  canonical same-pipe barrier (deduplicated by anchor), and cross-pipe
  requirements get canonical events grouped into bundles. Ownership cycles
  produce verified ownership bundles plus one composite bundle; bundles whose
  physical slots overlap are recorded as conflicts.
- **Verified slot protocols.** Slot-lifecycle discovery finds physical slots
  with a producer/consumer resource pair, a recurrence scope and distance,
  and ready/release opportunity sets, using only demands with complete
  storage provenance. Each materializable candidate is admitted through
  `addVerifiedProtocol` and must additionally build a distance-one release
  event recipe that passes correspondence and protocol verification; a
  candidate without a recipe is excluded and demotes optimality.
- **Standalone events.** A distance-zero cross-pipe active demand whose
  source can signal the target's pipe synthesizes a canonical set/wait
  event bundle when none exists.
- **PIPE_ALL fallback barriers.** One fallback candidate per active-demand
  group `(target, recurrence scope, distance)`, anchored before the target
  and marked as draining all resources. Candidates at the same loop depth
  whose active-demand coverage is a subset of another's are dominance-pruned.
- **Generated mechanisms.** `SyncCoverColumnGeneration` factories run over
  the universe, gated by `SyncCoverTargetCapabilities`. The pass wires
  exactly two generators into production: `canonical` (per-demand cross-pipe
  events) and `pierce-barrier` (same-pipe barriers on pipes without hardware
  completion, each covering every demand whose interval crosses the anchor).
  A merged-prefix generator existed but was removed: prefix-set completion
  semantics carry no evidence on any supported target, so it could never
  activate; it returns only with a hardware microbenchmark.
  Hardware-completion resources come from the pass's per-arch predicate, and
  the event-ID budget is `event-id-num-max`.

Every admitted mechanism must have a verified emission recipe expressed in
the CanonicalBarrier/CanonicalEvent vocabulary before selection: generated
barriers and events are translated back into recipes and re-verified through
the event-protocol verifier, and any unmaterializable generated mechanism is
a hard internal error.

## Grounding

`groundSyncCoverInstance` grounds structural coverage once; the search
consumes only bitset columns and immutable mechanism metadata and never
invokes the graph oracle. Coverage is a pure function of a demand's
endpoints, scope, distance, and guards over the frozen graph, so the
instance is skylined: one row per distinct coverage key, represented by its
first active demand, with oracle queries issued once per key. Covering a
row covers every duplicate demand behind it; the final verification still
spans the deduplicated original demand set. Grounding builds singleton
columns from one context-batched propagation that evaluates every mechanism
simultaneously per key, then adds independently verified factory columns
whose declared extra coverage is re-proved by the oracle. The adapter contributes those factory
columns: resource-feasible pairs of ownership/slot protocols, one column per
slot protocol bundling its release with greedily chosen event bundles that
cover the lifecycle's ready demands (extended to demands whose witnesses are
exactly the managed slot extent), and per-recurrence-scope pipeline merges of
those columns.

Composite coverage is proposed structurally, never searched for: the adapter
enumerates round-trip columns — a carried supply mechanism paired with the
event bundle that exactly reverses its endpoints, joined to the demand by
fixed issue-order paths on either leg — for every recurrence demand, ranked
tight-anchored and specific-mechanism first and capped at 512 columns with at
most four distinct pairs per demand. The claims are untrusted: grounding
proves every claimed demand with an oracle witness and silently keeps only
what it proves. The only grounding cap is `maximumColumns` (65536); hitting
it sets `columnsTruncated` and is never interpreted as infeasibility. A
demand with no column at all fails selection as incomplete search, distinct
from proven-uncoverable demands, which fail as proven infeasibility.

## Selection and cost

Selection is one deterministic global greedy pass over the grounded bitsets:
anchor each round on the first uncovered demand in instance order and adopt
the cheapest feasible column covering it, ranking that demand's columns by
per-depth action-per-new-coverage ratios. A column that uses a barrier only
counts coverage of demands no barrier-free column can take, so breadth that
events could supply never justifies a broad drain. Fallback barrier columns
hold no event resources, so the pass completes whenever every demand has a
column; there is no exact search, no component decomposition, no evaluation
budget, and no rescue path — earlier revisions carried all four, and every
production kernel was solved by this greedy anyway. Every evaluated state
runs the frozen evaluator: exact per-domain weighted-interval pressure
feasibility plus structural cost; infeasible states are pruned, never
costed. A single-deletion sweep (`removeRedundant`), a grounded-column
exchange (`improveWithGroundedColumns`), and a bounded oracle-checked
deletion pass (`oracleRedundancyLimit`, default 32) then improve the plan;
each re-evaluates full feasibility before accepting.

Every returned plan passes one final verification with a fresh evaluator and
a fresh coverage-oracle traversal over the deduplicated active demands;
failure is reported as `FinalVerificationFailed`, never silently emitted.

Structural cost is symbolic and lexicographic: per loop depth from deepest
nesting outward, compare barrier actions then event actions, then mechanism
count, then the stable mechanism signature. A barrier that drains all
resources is weighted by the number of distinct issue resources in the graph
(at least two), so the solver never prefers it over a same-depth targeted
barrier. Peak pressure, total domain pressure, event-domain count, and
minimum headroom remain reported diagnostics; exact coloring is a hard
constraint, but these nonseparable aggregates do not order plans. Invalid or
incomplete results carry an explicitly unevaluated cost, never a
valid-looking zero cost.

## Allocation and emission

The selection snapshot authenticates every selected resource use against the
live mechanism universe (domain, scope, distance, width, and lifetime must
match exactly) and validates the solver's per-domain interval allocation,
including reserved-ID exclusion and reuse only across disjoint inclusive
lifetimes. Materialization then maps each selected mechanism to its stored
recipe: barriers to `CanonicalBarrier`, event bundles to
`CanonicalEventBundleCandidate` with physical event IDs assigned through an
explicit per-use materialization event index (never inferred from vector
position or event direction), and slot protocols to synthetic bundles whose
identities are allocated above all existing bundle IDs. Barriers, events, and
event domains are replaced together only after bundle projection and
allocated protocol verification succeed. The legacy whole-plan verifier is
not re-run: the solver's independent oracle has already checked every active
demand, and the legacy check is both quadratic and incomplete for
verifier-proved hierarchical ownership consequences.

## Diagnostics and honesty

The covering diagnostics never overstate the result. The greedy selector
never claims optimality, so no `optimal=` field is reported; plan quality is
pinned by the regression corpus instead. The topology
line proves the grounded-search contract: `grounding-queries` is positive,
`coverage-queries=0` during search (structural coverage is grounded once and
searched over bitsets), and `final-validations=1` with the final traversal's
coverage queries reported separately.
