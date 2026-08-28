# PTOAS CanonicalSync

CanonicalSync is an automatic synchronization pass for PTO programs whose
operation order, pipeline assignment, and physical buffer placement are already
fixed. It is an alternative to InsertSync and GraphSync.

The pass follows this correctness boundary:

```text
structured completion graph and hazards
  -> direct synchronization mechanisms
  -> exact singleton and pair coverage
  -> global deterministic cover
  -> reverse deletion
  -> event-ID allocation and bounded repair
  -> fresh semantic verification
  -> atomic materialization
```

CanonicalSync does not reuse GraphSync's graph-solver IR, coloring search, or
code generation. It does not run an exact solver, synthesize ownership
protocols, or use a broad barrier as an ordinary cover candidate.

## 1. Input contract

Every ordinary operation result completes at its single scheduled graph node.
A multi-phase synchronization macro must instead name exactly one authoritative
completion phase for every SSA result. Missing, duplicate, or invalid mappings
fail before graph construction, and SSA provenance never crosses an unscheduled
effectful or region operation without an explicit rule. Tile allocation and
declaration results are explicit provenance roots because they name storage
rather than completed asynchronous data.

CanonicalSync runs after scheduling and memory planning. Local allocations must
therefore have concrete physical addresses. The pass analyzes:

- asynchronous PTO operations and their assigned pipelines;
- structured `scf.for` and result-free `scf.if` regions;
- SSA dependencies;
- physical-storage RAW, WAR, and WAW hazards;
- loop-carried hazards at their inferred recurrence distance;
- target-supported completion behavior;
- reserved event IDs already owned by hidden target protocols.

Unsupported region control flow, incomplete effects, ambiguous physical storage,
and unbalanced recurrence recipes fail closed before the IR is changed.

Operations that consume the intra-core event resources owned by CanonicalSync
are rejected as input. Whole-core synchronization and memory-fence operations
are fixed input constraints and are not treated as CanonicalSync output.

## 2. Completion graph

The graph contains one node per asynchronous operation and an explicit tree of
function, branch, and loop scopes. Each demand records source, target, owning
scope, recurrence distance, guards, hazard provenance, and physical-storage
witnesses.

Equivalent demands are interned by this coverage key:

```text
(source, target, owner scope, distance, source guard, target guard)
```

All SSA and storage causes remain attached to the one row for diagnostics.

Issue order alone does not prove asynchronous completion. Coverage propagation
tracks whether a path has acquired a completion fact. A synchronization supply
can establish completion; subsequent issue-order edges can preserve that fact
only where target semantics permit it.

Distance-zero demands use one immutable base arena. A loop with maximum active
distance `d` uses `d + 1` virtual iterations. Nested loops retain separate
recurrence arenas rather than forming a Cartesian product.

Loop-local DAGs are summarized bottom-up with resource-specific entry and exit
nodes, an explicit zero-trip transfer, recurrence-carry resources, and copied
periodic-control phase relations. Each arena contains its locally owned
operations and only the transfer interfaces of its immediate children; child
bodies are not copied into parent arenas. The interface retains a distinct port
for each externally relevant child operation, so early and late operations on
one resource cannot alias. Port discovery closes over internal operations on
fixed paths between exposed endpoints, preserving each path's node availability
and guards. Each summary owns only its local ports and resource sets; parent
arenas resolve descendant metadata through hierarchical child references and
charge ports and resource boundaries against the arena node budget before
materialization. Fixed and selected completion supplies use those
identity-preserving ports, while only certified issue-order edges connect ports
to resource entry/exit boundaries. A recurrence protocol exports completion
only when common validation certifies balanced priming, body lanes, and one
scope-exit drain per lane. Guarded protocols remain valid locally but do not
export until phase-qualified export semantics are supported.

## 3. Restricted mechanism catalog

A mechanism is the smallest selectable and materializable unit. Version one
contains only:

- a direct cross-pipeline event handshake;
- a targeted same-pipeline barrier;
- a lifecycle-complete generic recurrence event channel;
- a targeted-barrier plus event frontier reserved for event-pressure repair;
- a localized target `PIPE_ALL` barrier reserved for the last backstop.

A set and its matching wait are one mechanism and can never be selected
separately. A recurrence channel owns its entry priming, loop body actions,
modulo lane selection, and exit draining.

Ownership/slot-lifecycle protocols, pipeline aggregates, named round trips,
merged-prefix events, and arbitrary protocol paths are not part of the catalog.

The ordinary cover sees only precise direct mechanisms. Repair frontiers and
localized `PIPE_ALL` mechanisms are present in separately ranked fallback
tiers and cannot win normal cover selection.

A fixed architecture-required return drain is outside the covering problem. It
has no demand coverage and cannot make the optimization instance trivial.

## 4. Singleton and pair coverage

For every precise mechanism `m`, the graph oracle computes exact singleton
coverage `C(m)` over all active demand rows.

For two plausible mechanisms `m1` and `m2`, the oracle computes exact joint
coverage. The pair is retained only if it adds coverage that neither singleton
has:

```text
extra(m1, m2) = C(m1, m2) - (C(m1) union C(m2))
```

The pair records only this extra bitset. Mechanisms are selected and paid for
once even when they activate several retained pairs.

Pair proposals are generated bottom-up. A pair is owned by the lowest common
ancestor of the mechanisms' supply scopes, so mechanisms in sibling regions can
compose at their parent. Precise targeted barriers may participate in pairs.
A pipeline/guard prefilter removes pairs that cannot plausibly form a completion
chain before exact propagation.

The exact-evaluation bound is applied per owner scope. If one scope exceeds the
bound, optional pairs for that scope are skipped as a group; singleton direct
mechanisms remain available.

## 5. Selection strategies

All selection strategies operate on the same frozen bitsets and scan every
candidate with positive marginal gain. None anchors its decision on the first
uncovered demand.

`fixed-cover`
: Treats each retained pattern as a fixed column and charges its complete member
  cost whenever that column is considered. This is the simplest comparison
  baseline.

`action-aware-singleton`
: Considers one missing mechanism at a time and measures all coverage activated
  by adding that mechanism, including pairs whose other member is already
  selected.

`pair-lookahead`
: Considers the action-aware singleton moves plus a two-mechanism move for each
  retained pair when both members are still missing. This is the production
  default.

Each round chooses the best exact marginal density globally. Ratios use
128-bit cross multiplication and stable ID tie-breaks.

The calibration-free cost is lexicographic:

1. action counts by natural loop depth, with deeper loops compared first;
2. serialization breadth induced by the supplied completion edges;
3. inclusive event lifetime area;
4. number and stable IDs of newly added mechanisms.

After greedy cover, mechanisms are examined in reverse selection order and
removed whenever exact activated-pattern coverage remains complete.

## 6. Event allocation and repair

Normal greedy selection does not color events and does not reject a logical
cover because its partial event allocation is infeasible.

After reverse deletion, each directed pipeline-pair event domain is allocated
independently. Lifetimes are inclusive, so IDs can be reused only when one
lifetime ends strictly before the next begins. Allocation respects widths,
reserved IDs, and the configured hardware budget.

On failure, the allocator reports:

- the domain;
- required and available IDs;
- the maximum-pressure timeline point;
- the selected mechanisms live at that point.

Bounded repair first forbids one conflict-core mechanism at a time and reruns
the same global selector. If those alternatives remain infeasible, the
targeted-barrier event-frontier tier becomes eligible. The default repair bound
is eight rounds.

If repair is exhausted, CanonicalSync makes one last attempt using localized
target barriers, including `PIPE_ALL` only at affected targets. This backstop is
reported explicitly and is never an ordinary weighted-cover candidate. If even
that plan cannot be freshly verified, the pass fails and leaves the IR
unchanged.

## 7. Independent final verification

The final verifier does not trust the mutable greedy coverage state. It rebuilds
the completion-supply list from only the selected mechanisms and reruns the
semantic coverage oracle over every active demand.

It also recomputes event allocation and validates mechanism conflicts, guarded
and recurrence actions, physical anchors, lanes, and reserved IDs. Only then are
all actions staged and emitted. No failure can leave a partial set/wait
protocol in the function.

## 8. CLI

Enable the pass with:

```text
--enable-canonical-sync
```

Relevant driver options are:

```text
--canonical-sync-event-id-max=8
--canonical-sync-pattern-mode=direct|direct-pair
--canonical-sync-selection-strategy=fixed-cover|action-aware-singleton|pair-lookahead
--canonical-sync-maximum-repair-rounds=8
--canonical-sync-assume-distinct-gm-args-noalias
--canonical-sync-assume-all-gm-accesses-noalias
```

`direct` disables optional pair construction. `direct-pair` is the default.
The two GM alias contracts are mutually exclusive; the all-accesses contract is
unsafe unless the caller guarantees that even accesses through the same
argument are disjoint.

To compare all three selectors on the same frozen problem without inserting
synchronization:

```text
--enable-canonical-sync
--canonical-sync-analysis-only
--canonical-sync-comparison-report=report.json
--emit-pto-ir
```

Analysis-only mode requires textual IR emission so it cannot accidentally
produce executable output without synchronization.

The human summary and JSON report contain demand/key counts, direct mechanisms,
pair proposals and evaluations, retained synergistic pairs, selected event and
barrier counts, all structural cost components, maximum domain pressure, repair
rounds, assigned physical IDs, work limits, and localized `PIPE_ALL` use.
