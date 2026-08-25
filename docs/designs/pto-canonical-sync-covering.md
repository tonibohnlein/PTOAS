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
established completion fact but is not traversable before completion has been
established. It therefore supports merged waits but cannot aggregate earlier
operations into a later event set. Non-completion-preserving issue order cannot
participate in a completion proof. This distinction avoids treating command
issue order as hardware completion order.

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
Adapter code constructs these candidates through the descriptor builder:
canonical events are one operation, while typed action references let protocol
lanes share one physical action across event-ID and buffer-token uses. Adapter
call sites never manipulate descriptor-local edge, action, use, or binding
indices directly.

Graph and universe construction are append-only. Candidate edges are normalized
without mutation, the side-effect-free protocol verifier runs before commit,
and an RAII graph transaction restores both edges and generation if the final
non-allocating commit does not complete. The universe validates a construction
epoch once, revalidates after external graph mutation, and performs full
graph/universe validation at phase boundaries. Frozen selection evaluators
record both graph and universe generations and fail closed after either changes.
`SyncCoverMechanism.h` intentionally remains the universe API: mechanism
descriptors, resource-feasibility results, and structural-cost results stay
together because one selection evaluator produces the latter two from one
authoritative coloring.
Every non-barrier supply edge is bound to an event-ID or ownership-token
lifetime with matching endpoints, scope, and recurrence distance. Event
bundles use event-ID domains; verified ownership protocols may additionally
use independent physical token pools. Buffer-token pool identities are unique
in version one; a shared pool spanning multiple resource domains is rejected
until coloring and component decomposition can model it jointly. A barrier
names a concrete anchor on the source pipe that it drains. The covered target
may be on another pipe, but its timeline position, scope, and control guard
must guarantee that the barrier executes before it.
Conditional and potentially zero-trip nested anchors therefore fail closed.

Ordinary event bundles are deliberately limited to distance-zero canonical
actions: produce immediately after the supplied edge's source and consume
immediately before its target. Positive-distance events require prime,
steady-state, and drain behavior that these two anchors cannot describe. They
therefore enter the universe as verified protocols, including the stock bare
recurrence-event protocol that the CanonicalSync adapter will use. The stock
protocol is deliberately limited to one event and distance one, with an exact
prime, body wait/set, and drain lifecycle. Its endpoints may be in mandatory
straight child scopes but not nested loops, whose execution multiplicity needs
an explicit proof. Wider, longer-distance, conditional, nested-loop, or
ownership recurrences require an adapter verifier for the exact submitted
descriptor; there is no permissive generic fallback.

A verified cyclic lifetime may bind distance-zero startup or drain supplies in
a nested ancestor or descendant scope when all physical actions remain within
the lifecycle timeline. This represents one event ID across preheader,
steady-state, and epilogue phases. Unverified and ordinary event bundles still
require exact scope and distance matching.

The protocol verifier is a soundness boundary. For a verified ownership or
recurrence protocol, it certifies that the submitted actions implement every
declared supply edge, including token state across prime, steady-state, branch,
and drain paths. The generic oracle trusts those certified edges; it does not
reconstruct their endpoint semantics from action anchors.

Resource feasibility is exact for the modeled inclusive interval lifetimes:
the maximum weighted overlap equals the required number of interchangeable
IDs. Straight lifetimes span their bound physical actions. Positive-distance
lifetimes conservatively span the complete explicit loop timeline in version
one, so feasibility may reject a runtime-valid circular allocation but never
accept one through interval splitting. Unique reservations inside the hardware
budget reduce availability; out-of-range reservations remain diagnostic. The
result includes the stable maximum-pressure point and
`(mechanism, resource-use, width)` owners. Mutually exclusive guards are not
used to share IDs in version one. A feasible result also carries the
deterministic nonreserved physical-ID assignment; emission consumes that
assignment rather than running a different allocator.

For a demand at loop distance `d`, the coverage oracle evaluates a virtual
`d+1`-copy DAG. Reachability state records whether the path has crossed a
completion edge. A demand is covered only when its target is reachable with
that completion state set. The oracle returns witnesses for covered demands
and reachable-cut certificates for uncovered demands. One immutable oracle
epoch validates the graph once and lazily caches each demand's virtual DAG.
Its topology query conservatively ignores completion state while retaining
only mechanisms on a structural source-to-target path; disconnected kernel
regions therefore do not collapse into one search component.

Coverage, token validity, and exact event-ID coloring are hard constraints.
Structural cost is defined only for resource-feasible selections; infeasible
partial search states require a separate overflow-first search heuristic.
The final selected plan is checked by fresh graph/resource validation and a
second exact coverage traversal that bypasses the optimizer's witness cache.

## CanonicalSync adapter and emission

The CanonicalSync adapter translates the conservative completion requirements,
fixed issue-order graph, barriers, ordinary event bundles, bare recurrence
events, synthetic bundles, and verified ownership protocols into one mechanism
universe. Active requirements retain their conservative identities, so
deactivating a no-alias requirement cannot remove candidates needed by another
plan. Shadow mode builds and solves this universe but leaves legacy emission
unchanged.

The graph and mechanism adapters own every callback they retain. Non-owning
`llvm::function_ref` views are limited to helper calls that complete before the
view's caller returns; they are never stored in an adapter member.

Direct emission is explicitly selected with `solver=covering` or the driver
option `--canonical-sync-solver=covering`. It authenticates every selected
resource use against the immutable mechanism universe, requires exact equality
with the solver's final allocation, and validates physical ID reuse over the
inclusive lifetimes. Event-ID uses carry an explicit provider-local event
index; materialization never infers the mapping from vector position or event
direction. Buffer-token allocation is rejected in version one.

Selected providers are materialized atomically. Barriers, event bundles,
flattened events, event domains, and exact IDs are replaced together only
after bundle projection, allocated protocol verification, direct final
coverage, and the CanonicalSync whole-plan coverage check succeed. A sound
truncated solver result may be emitted, but its diagnostics retain
`truncated=yes` and do not claim optimality. The direct path is opt-in; the
legacy selector and emitter remain the default.

Version one still constructs the legacy seed before direct selection. It
therefore cannot rescue a kernel that fails during an earlier legacy scarcity
repair. Removing that bootstrap dependency requires moving conservative
candidate generation ahead of all legacy feasibility decisions.

## Selection and cost

The mechanism universe is generated from the conservative demand set before
no-alias filtering. Alias information can deactivate demands but cannot remove
a candidate required to repair another plan.

Small affected mechanism components use deterministic exact search. Larger
components use a bounded deterministic beam guided by uncovered cuts. Every
complete candidate receives exact coverage, protocol, and coloring checks.
Components conservatively join demands and mechanisms whose lifetimes overlap
inside one event/token domain or that carry an explicit conflict. Disjoint
lifetimes in the same domain remain independent because their union cannot
increase instantaneous pressure. Covered witnesses are reusable while all
mechanisms in the witness remain selected; removing any witness mechanism
causes an oracle query, while additions preserve the proof. The cache retains
only a bounded antichain of small witnesses per demand.

The direct-cover beam follows the legacy affected-slice search's bounded,
stable-search discipline but does not call that helper directly. The legacy
helper is coupled to MLIR-side mechanism references, static requirement
support lists, and the legacy score. Direct covering instead branches on the
current oracle cut, prunes hard resource failures, supports exact components,
and keeps this layer free of MLIR data structures. Resource-feasible seeds are
warm starts and incumbents, not recognizer authority: they are evaluated
outside the exploration budget and remain available for the final score
comparison. Over-budget conservative seeds are accepted as input but are not
search states; the empty-state cut search repairs them from the candidate
universe.
After selection, redundancy removal considers complete atomic mechanisms and
chooses the best valid single-mechanism deletion at each step.

Exact components compose globally because the version-one ordering is limited
to separable metrics: loop-aware event-action profile, barrier profile,
mechanism count, and stable identity. Peak pressure, total domain pressure,
event-domain count, and minimum headroom remain reported diagnostics; exact
coloring is a hard constraint, but these nonseparable aggregates do not order
plans. A future global Pareto combiner may promote them into the objective.
Beam width, depth, and evaluation truncation are reported separately. Failure
without truncation proves infeasibility; bounded failure reports incomplete
search, and a bounded success does not claim optimality. Exact and beam search
both obey the per-component evaluation budget; seed evaluation remains outside
that budget so a valid incumbent is not discarded by truncation. Every
returned plan is finally rechecked with fresh graph/resource validation and a
second exact coverage traversal over the immutable prepared demand topology.
The traversal bypasses search witness caches; only deterministic virtual-graph
expansion is shared to avoid retaining a second full topology cache. Invalid or
incomplete results carry an explicitly unevaluated cost, never a valid-looking
zero cost.

The initial mechanism cost is symbolic rather than a runtime estimate. It
counts each physical synchronization action once, even when one action carries
both an event-ID and a buffer-token use. Profiles are compared from deepest
loop nesting outward; loop-entry/exit prime and drain actions exclude the
loop's own iteration depth. Barriers use a separate profile. Mechanism count
and stable identities provide deterministic later tie breakers. Event pressure
and headroom remain feasibility diagnostics because they do not compose across
independent mechanism components. Mutually exclusive guarded actions are
counted separately.
Calibrated trip counts and target weights can replace this structural profile
later without changing graph, mechanism, or feasibility contracts.
