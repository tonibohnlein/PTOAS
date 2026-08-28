# PTOAS CanonicalSync Bounded Pattern Cover

CanonicalSync inserts synchronization by solving a bounded pattern-cover problem
over a completion-aware dependence graph. The pass is designed around three
separate concerns:

1. analysis constructs sound completion requirements;
2. selection chooses synchronization mechanisms with exact event-ID
   feasibility;
3. finalization validates the selected plan before materialization.

Pattern coverage is certified while the immutable problem is built. Selection
uses only those certified bitsets, and finalization checks the selected IDs
against the frozen pattern table before emission.

## 1. Completion Requirements

The graph contains asynchronous PTO operations, structured scopes, guards, and
completion edges. A demand states that a source operation must complete before a
target operation may issue. Demands include SSA dependencies and physical-memory
RAW, WAR, and WAW hazards.

The graph owns an explicit scope tree. A zero-distance demand belongs to the
lowest scope containing both endpoints, while a recurrence belongs to its loop.
Loops remain structured: a distance-`d` recurrence is checked in that loop's
`d + 1`-copy arena, which is sufficient to represent the finite periodic
dependence without unrolling a runtime trip count. Descendant operations remain
visible in an enclosing arena, while nested recurrences retain their own arena;
the implementation does not form a Cartesian product of nested trip counts.
Mutually exclusive control paths and loop-local guards remain explicit.

Issue order is not automatically completion order. The graph distinguishes:

- completion-preserving issue-order edges;
- issue-order edges that do not preserve completion;
- completion supplied by a synchronization mechanism.

Target capabilities license completion-preserving behavior. In particular, the
initial implementation does not assume that an arbitrary MTE1 `set_flag` drains
all earlier MTE1 work. A protocol that requires such prefix semantics is admitted
only when the target capability records suitable evidence.

## 2. Mechanisms And Patterns

A mechanism is an independently selectable and materializable unit:

- one complete event handshake;
- one targeted same-pipe barrier;
- one generic loop-carried event channel with its lane-indexed prime, body, and
  drain;
- one verified ownership or slot-lifecycle protocol.

Each mechanism owns its completion edges, physical actions, resource lifetimes,
cost, and materialization recipe. A set and its matching wait are never selected
independently.

A pattern is a named combination of mechanisms with precomputed demand coverage.
For selected mechanism set `X`, coverage is:

```text
Covered(X) = union coverage(p) for every pattern p
             whose members are a subset of X.
```

Mechanism cost is paid once even when several patterns share that mechanism.
Selecting members independently also activates every pattern whose complete
member set is then present.

The production catalog is deliberately finite:

- singleton events and targeted barriers;
- direct pairs whose exact joint coverage is larger than the union of their
  singleton coverage;
- generic recurrence events for every distance that fits the event-ID budget;
- recurrence round trips;
- verified ownership protocols;
- verified slot-lifecycle protocols;
- bounded pipeline-scope aggregates;
- targeted-barrier event frontiers used only under event-ID scarcity;
- localized `PIPE_ALL` rescue mechanisms.

Candidate construction is bottom-up over the scope tree, but selection is not.
An inner mechanism remains selectable globally and its coverage is evaluated
against every active demand, including demands owned by an ancestor scope. All
singleton mechanisms are propagated together for each demand. Likewise, every
bounded two-mechanism proposal is propagated together in the demand's owning
base or recurrence arena. This extends inner coverage outward without one graph
walk per mechanism or pair.

CanonicalSync does not enumerate arbitrary mechanism subsets, run pricing, or
maintain a second exact solver. Direct-pair proposals are restricted to related
scope ancestries and bounded before exact joint coverage is computed. If that
bound is exceeded, the whole optional pair family is skipped rather than taking
an order-dependent prefix. Every supported demand retains a safe precise event,
targeted barrier, or final rescue, so a missing optimization pattern affects
plan quality rather than correctness.

For a release-style distance-`d` recurrence, one verified mechanism owns `d`
physical IDs, one scope-entry prime and scope-exit drain per lane, and body
set/wait actions selecting `iterationOrdinal % d`. A forward-ordered recurrence
uses the stronger legal same-iteration event when available. Guarded recurrence
channels remain fail-closed until a complete path-balanced recipe is available.

## 3. Selection And Event IDs

One deterministic greedy solver repeatedly selects the feasible pattern with the
best exact, integer-computed marginal cost density. The marginal cost contains
only newly introduced mechanisms, while the gain includes every pattern
activated by the resulting selected set. A reverse-deletion pass then removes
mechanisms whose removal preserves claimed coverage and resource feasibility.

Selection runs in three ordered tiers. Precise events, recurrence channels,
verified protocols, and targeted same-pipe barriers are considered first. A
targeted barrier is a precise local completion mechanism, not a broad rescue.
If ordinary greedy selection reaches an event-ID dead end, the same precise
catalog is retried with exact interval pressure and event-lifetime span as
deterministic tie breakers. Only then are targeted-barrier event frontiers
enabled. `PIPE_ALL` is enabled only in the final retry and is
reported with a diagnostic. Once a retry enables a fallback family, every
enabled candidate other than `PIPE_ALL` competes by the same marginal-density
objective. This lets a frontier avoid a greedy event-ID dead end rather than
arriving only after the scarce ID was already consumed. In the final retry,
`PIPE_ALL` ranks after every precise or scarcity candidate and is considered
only when those candidates cannot advance the anchored demand under the current
resource assignment.

No semantic graph traversal occurs during greedy selection. Selection reads the
precomputed pattern coverage and resource intervals.

Event IDs are a hard feasibility constraint. Each directed event domain is
colored independently with inclusive interval lifetimes, reserved IDs, recurrence
width, and the hardware budget. A candidate that cannot be colored is not a
feasible selection. The final assignment is recomputed authoritatively after
selection.

The structural objective is loop-aware and calibration-free. A targeted
same-pipe barrier is counted as one precise synchronization action, while an
event handshake contributes its set and wait actions. Actions in deeper loops
are compared first using exact integer ratios. Broad `PIPE_ALL` barriers are
both isolated in the rescue tier and weighted by the resources they drain.

## 4. Finalization And Materialization

After selection, CanonicalSync recomputes active coverage from the selected
mechanism IDs and the frozen, certified pattern bitsets. It also rechecks
resource feasibility, the event-ID assignment, mechanism conflicts, and the
availability of every materialization recipe. No semantic graph traversal is
repeated after selection. Failure aborts the pass without partially modifying
the IR.

Materialization then emits each selected mechanism exactly once using its stored
recipe and assigned IDs. Protocol actions remain atomic: failure cannot leave a
standalone set, wait, prime, or drain action in the program.

Every non-event completion claim in an ownership mechanism carries a typed proof
kind. Its verifier reconstructs the relevant ready/release token path from the
cycle, checks exact managed-storage witnesses and endpoint roles, and validates
the claimed recurrence scope and distance. Nested-loop summaries additionally
require a matching verified inner recurrence edge. These claims are not accepted
because the generator produced the same descriptor; the verifier derives their
premises independently.

Unknown asynchronous effects, unsupported control flow, ambiguous ownership, and
unmaterializable protocols fail closed with a diagnostic. The pass does not
silently fall back to an unverified synchronization plan.

## 5. GM Alias Contracts

CanonicalSync is conservative for GM unless the caller supplies an explicit
contract:

- `--canonical-sync-assume-distinct-gm-args-noalias` treats distinct GM pointer
  arguments as non-aliasing;
- `--canonical-sync-assume-all-gm-accesses-noalias` treats all distinct GM
  accesses as non-aliasing.

The second mode is stronger and unsafe unless the complete caller and allocation
contract guarantees disjoint storage. Neither option changes local-memory
ownership analysis.

## 6. Historical GEMM Gate

With `assume-all-gm-accesses-noalias`, the historical GEMM selects four verified
protocols: accumulator ownership, atomic L0 ownership, stable hierarchical L1
ownership, and alternating hierarchical L1 ownership. The current regression
requires:

- complete demand coverage;
- exact event IDs below 8;
- zero body barriers;
- one required tail `PIPE_ALL` barrier;
- 72 set actions and 74 wait actions.

The lower event count of an earlier donor implementation used one event after the
later of two MTE1 producers to cover both. That requires ordinary-node MTE1 prefix
completion, which the current target capability does not license. The bounded
pattern implementation therefore keeps one completion event per producer. This
is a target-semantics distinction, not a solver or pattern-generation failure.

## 7. Future Extensions

The graph and mechanism model intentionally leave room for evidence-gated merged
events, more selective pair discovery, guard-complete recurrence channels, and
richer target capabilities.
Longer term, the same completion requirements and structural cost model can
participate in memory-placement and out-of-order scheduling optimization. Those
extensions may change addresses or issue order and then rebuild the bounded
pattern problem; they do not require weakening construction-time coverage
certification.
