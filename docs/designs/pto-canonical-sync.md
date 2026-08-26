# PTO Canonical Synchronization

`pto-canonical-sync` inserts synchronization for a PTO function whose operation
order, pipe assignment, and local-memory addresses are already fixed. It has one
selection path: construct a sound dependence graph, generate synchronization
opportunities, solve a grounded constrained set-cover instance, and verify the
selected plan before changing the IR.

## Scope

The pass does not reorder operations, choose pipes, or plan local-memory
addresses. Its inputs are:

- structured PTO/SCF IR;
- the fixed issue order of operations on each pipe;
- operation memory effects and exact local-memory ranges;
- SSA and structured-control relations;
- the target's completion and event-ID capabilities.

The pass may add barriers and set/wait event protocols. A barrier is a fallback
serialization mechanism. Events and verified ownership protocols preserve more
cross-pipe overlap when they cover the same completion requirements.

## Pipeline

### 1. Structural dependence graph

Each schedulable PTO operation becomes one `SyncCoverNode`. A node records its
pipe, structured scope, issue order, guards, storage accesses, and completion
targets. The graph contains:

- fixed issue-order edges;
- SSA and explicit operation dependences;
- memory RAW, WAR, and WAW dependences;
- positive-distance loop recurrence edges;
- structured scope and alternative-control metadata.

Zero-distance edges must form a DAG. Loops remain structured and finite in the
IR. Positive-distance edges represent the periodic dependence between virtual
loop copies, so the pass does not unroll an unbounded loop.

The graph distinguishes issue ordering from completion. A same-pipe edge is
completion preserving only when the target capability says that completion of
the later operation also proves completion of the earlier one. No undocumented
prefix-set behavior is assumed.

### 2. Synchronization opportunities

The pass generates mechanisms without mutating a selected plan:

- barrier opportunities for same-pipe completion hazards;
- canonical event bundles for cross-pipe requirements;
- verified unit and hierarchical slot-ownership protocols;
- interval-piercing barriers for unconditional straight-line same-pipe cuts;
- merged-prefix events only when target capabilities explicitly permit prefix
  set semantics;
- unit recurrence, ring-release, and capability-gated buffer-token protocols.

Ownership generation uses exact storage provenance. Slot/front structure may
choose anchors, but the protocol verifier must independently establish the
managed storage lifecycle, token balance, guards, recurrence distance, and
resource width. A recognizer is not its own verifier.

Unknown control, address, effect, or slot relations fail closed. The pass keeps
a barrier or rejects the function instead of assuming a protocol is valid.

### 3. Grounded columns

`groundSyncCoverInstance` freezes the opportunity universe into a reusable
incidence instance:

- each active completion requirement is a row;
- each mechanism or verified composite is a column;
- each column has a precomputed covered-demand bit set;
- resource lifetimes, conflicts, and loop-aware structural cost are attached
  once;
- composite columns share the cost of their member mechanisms.

Singleton mechanism incidence is computed in one context-batched traversal.
Verified factory columns are inserted directly. Bounded depth-two pricing is
used only for demands that still lack a column. Selection never invokes the
coverage oracle.

Grounding happens before decomposition. Consequently, a composite column that
connects two incidence regions also connects their components; it is never
silently dropped by an earlier decomposition.

### 4. Incidence decomposition

The grounded bipartite demand/column graph is decomposed into connected
components. Resource interactions are included in component formation, so
solutions of distinct components can be composed safely. Shared resource pools
use the same identity for decomposition and feasibility.

### 5. Constrained weighted set cover

Each component is solved deterministically:

- greedy construction for a feasible incumbent;
- exact search for small components under an explicit evaluation budget;
- bounded local deletion and exchange improvement for larger components.

Every candidate selection is checked against cached column coverage, mechanism
conflicts, event lifetimes, and the exact interval colorer. The objective compares
loop-aware barrier actions before event actions at each depth, then applies stable
structural tie-breakers. This prevents the solver from buying a frequently
executed pipe barrier merely to remove a cheaper event pair.

Budget exhaustion reports a valid truncated result, never a proof of
infeasibility. A selected plan is emitted only if it is feasible and strictly
validated.

### 6. Independent final verification

After selection and exact event-ID allocation, the coverage oracle verifies all
active requirements against the selected completion edges. Demands sharing the
same structured expansion are checked in a batch. This is the only selected-plan
oracle invocation; it is a soundness boundary, not the search procedure.

The pass then materializes complete mechanisms atomically. It never emits an
isolated half of an event or ownership protocol. A final `PIPE_ALL` tail barrier
is retained to establish function completion at the external boundary.

## Event IDs

Current hardware exposes at most eight event IDs per directed pipe domain.
Event mechanisms carry explicit lifetimes. The exact colorer allocates IDs while
respecting reserved IDs and overlapping lifetimes. If events alone do not fit,
the grounded solver may select a barrier column that serializes the affected
requirements. It must still pass final whole-plan verification.

## GM Aliasing

Unannotated GM accesses are `MayAlias`. Same-root constant subranges and
supported view geometry are analyzed precisely. The function attribute
`pto.noalias_pairs` may prove selected GM arguments disjoint.

Two explicit command-line assumptions exist for experiments and controlled
frontends:

- `--canonical-sync-assume-distinct-gm-args-noalias` assumes different GM
  arguments are disjoint while preserving same-root hazards;
- `--canonical-sync-assume-all-gm-accesses-noalias` ignores all GM memory
  dependences, including same-argument and cross-iteration relations.

Both options are unsafe contracts supplied by the caller, and they are mutually
exclusive. The all-access mode is not a lifetime-analysis inference. Overlapping
logical lifetimes do not imply different pointers are disjoint.

## Diagnostics

`pto-print-canonical-sync-plan` supports `all`, `dependencies`, `plan`,
`events`, `ownership`, and `covering` views. The covering view reports:

- graph and structured-control sizes;
- slot-lifecycle and verified protocol factories;
- generated opportunity counts and truncation;
- grounded components and solver evaluations;
- demands without an event column;
- exact resource allocations;
- final verification statistics.

These counters distinguish candidate-generation gaps, resource infeasibility,
bounded-search truncation, and final-verification defects.

## Current Limits

- Merged-event staircase generation remains disabled unless prefix-set semantics
  are explicitly supported for the target pipe.
- Buffer-token rings are generated only for targets that expose such a resource;
  current A2/A3 event emission uses verified event protocols.
- Depth-two pricing is a bounded missing-column repair, not an unbounded search.
- Operation scheduling, pipe reassignment, and DSA co-optimization are future
  extensions. The graph and cost layers preserve the required boundaries: those
  extensions may alter node order, storage witnesses, or cost, while the grounded
  covering and final verifier remain reusable.

## Performance Contract

Search cost must scale with the grounded incidence instance rather than repeated
global graph reconstruction. In particular:

- no candidate-deletion loop may rebuild the virtual graph;
- selection performs zero coverage-oracle queries;
- exact search has an explicit evaluation budget;
- final verification is batched and runs once;
- release builds should keep the historical GEMM within a small constant factor
  of `pto-insert-sync`, not minutes.

The historical no-alias GEMM regression requires a complete eight-ID plan with
53 set/wait pairs, zero planned body barriers, deterministic output, and one
successful final verification.
