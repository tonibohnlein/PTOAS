# Canonical synchronization covering

## Objective

CanonicalSync treats synchronization selection as completion covering over a
fixed kernel execution order. Version one does not change physical addresses,
pipe assignments, or operation order. A later memory planner may rebuild the
demand set for another placement, and a later scheduler may replace issue-order
edges without changing coverage or mechanism semantics.

## Graph

Each schedulable operation is one weighted node. Zero-distance structural
edges form a DAG. A positive-distance edge belongs to one structured loop and
connects different dynamic occurrences; a positive-distance self-recurrence is
therefore valid even though a zero-distance self-edge is not.

Structured control is represented by conjunctions of branch alternatives. A
node or edge can contribute to a demand proof only when the joint execution
condition of the demand endpoints implies its guard. Unsupported unstructured
control flow and unknown asynchronous effects remain fail-closed.
Control IDs are static IR identities; virtual loop unrolling contextualizes a
literal by its dynamic copy, so alternatives from different iterations are not
treated as mutually exclusive merely because their static control ID matches.

Scopes and controls are graph-owned, and each control records the scope in
which it is evaluated. A recurrence names a non-root loop scope that encloses
both endpoints; sibling regions cannot manufacture a recurrence. Edge and
demand guards preserve source- and target-occurrence conditions separately.
Controls inside the recurrence scope are contextualized per virtual copy,
while outer controls remain shared. A path therefore cannot claim execution
outside either endpoint or reject a valid `then(i) -> else(i+1)` recurrence.
Nested scopes additionally carry a must-execute-within-parent fact. Coverage
paths may traverse nested operations only when that fact is proven by the IR
provider; unknown or potentially zero-trip nested loops remain fail-closed.

Structural edges have explicit temporal semantics. A completion-supply edge
establishes completion. Completion-preserving issue order can carry an already
established completion fact but cannot create one. Non-completion-preserving
issue order cannot participate in a completion proof. This distinction avoids
treating command issue order as hardware completion order.

The graph stores immutable completion demands separately from structural
edges. Physical overlap creates RAW, WAR, and WAW demands. SSA and architecture
ordering may contribute structural completion requirements or fixed completion
edges, but do not imply a synchronization mechanism.

## Mechanisms and coverage

An event contributes only its explicitly verified source-to-target completion
edges. A barrier contributes a completion cut on one pipe. Cyclic ownership
protocols remain atomic candidates with independently verified token state,
prime, steady-state, and drain actions.

The mechanism universe assigns one dense identity to each event bundle,
barrier, or ownership protocol. Adding a candidate atomically attaches all of
its completion-supply edges to the graph; partial protocol selection is not
representable. Directed event/token domains and their periodic lifetimes are
stored independently of supply, while conflicts encode alternative protocol
implementations. Ownership candidates are admitted only after their protocol
verifier has accepted the complete prime/steady-state/drain lifecycle.
Every non-barrier supply edge is bound to an event-ID or ownership-token
lifetime with matching endpoints, scope, and recurrence distance. Event
bundles use event-ID domains; verified ownership protocols may additionally
use independent physical token pools. A barrier names a concrete same-pipe
anchor and can cover an edge only when the covered target's scope and control
guard guarantee that the anchor executes. Conditional and potentially
zero-trip nested anchors therefore fail closed.

For a demand at loop distance `d`, the coverage oracle evaluates a virtual
`d+1`-copy DAG. Reachability state records whether the path has crossed a
completion edge. A demand is covered only when its target is reachable with
that completion state set. The oracle returns witnesses for covered demands
and reachable-cut certificates for uncovered demands.

Coverage, token validity, and exact event-ID coloring are hard constraints.
The final emitted plan is checked by a non-incremental verifier independently
of the optimizing oracle.

## Selection and cost

The mechanism universe is generated from the conservative demand set before
no-alias filtering. Alias information can deactivate demands but cannot remove
a candidate required to repair another plan.

Small affected mechanism components use deterministic exact search. Larger
components use a bounded deterministic beam guided by uncovered cuts. Every
complete candidate receives exact coverage, protocol, and coloring checks.

The initial cost model uses one scalar unit weight per operation. It compares
straight-line critical paths, periodic loop serialization, loop-aware dynamic
set/wait actions, barrier drains, and event pressure. Calibrated target weights
can replace unit weights without changing the graph or solver.
