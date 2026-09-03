# CanonicalSync graph-guided structural cover plan

## Purpose

CanonicalSync begins with a hardware-grounded demand graph and one conservative
direct synchronization mechanism for every demand. A direct mechanism may
establish more completion relations than the demand that caused it: an event is
a source-pipeline-prefix to target-pipeline-suffix cut, and a pipe barrier is a
same-pipeline cut. Several direct mechanisms can also establish additional
relations through transitivity.

The optimization problem is therefore not limited to choosing isolated direct
mechanisms. It must also discover a small number of promising **sets** of direct
mechanisms whose joint effect may cover substantially more demands. Exhaustively
grounding all pairs or larger subsets is not viable.

This plan uses graph structure together with normalized IR semantics to nominate
bounded mechanism groups. Nomination is only a search heuristic. The ordinary
hardware-grounded coverage oracle remains the sole authority for admitting a
group to the set-cover instance.

## Central invariant

Let:

- \(D\) be the complete set of required synchronization demands;
- \(M\) be the catalog of target-legal direct mechanisms;
- \(F(S)\) be the demands proved covered when a set \(S\subseteq M\) is
  installed in a fresh achieved-happens-before world.

For a proposed mechanism group \(S\), define its genuinely joint benefit as:

\[
\operatorname{Extra}(S)=
F(S)\setminus\bigcup_{m\in S}F(\{m\}).
\]

A structural proposal may enter the set-cover instance only if:

1. every member is an already legal, independently verified direct mechanism;
2. the complete group is grounded with the same summary, flat, and bounded
   unrolled coverage oracles used by final verification;
3. those oracles agree; and
4. \(\operatorname{Extra}(S)\ne\varnothing\).

Graph shape, operation names, storage roles, or similarity never constitute a
proof of coverage.

## What a group column means

Initially a group column is a **composition**, not a new synchronization
recipe. Selecting it selects all direct member mechanisms. Its joint coverage
is active only while all of those members remain selected. Cost is the number
of member mechanisms not already selected.

This distinction avoids a common modeling error:

- a composition group emits the union of its direct member actions and may gain
  transitive coverage;
- a synthesized mechanism emits a different, cheaper physical protocol and
  requires a separate hardware certificate and materialization recipe.

The first can be discovered and verified entirely by structural grouping. The
second needs lifecycle synthesis described later in this plan.

## Information used to nominate groups

### Graph structure

The structural analysis operates on the typed demand graph and its regional or
bounded iteration expansion. Useful structural signals include:

1. **Topological level boundaries.** For every level \(k\), collect demands
   crossing from nodes at levels \(\le k\) to nodes at levels \(>k\). Test the
   direct mechanisms attached to that boundary as one group.
2. **Region-local transitive bases.** Compute the transitive reduction or an
   equivalent reachability basis of the unique phase-edge DAG. The mechanisms
   on the basis edges are a natural small set to test because their composition
   may cover long edges.
3. **Dominance and separator structure.** Dominator, post-dominator, articulation,
   bridge, and small separator analyses identify connectors through which many
   required paths pass. Groups are built from the direct mechanisms incident to
   those connectors, not from every pair in the graph.
4. **Loop-expanded SCCs.** Positive-distance storage edges turn an otherwise
   acyclic body into ownership cycles. SCCs in a bounded lifecycle graph identify
   ready/release groups that may need one balanced recurrence protocol.
5. **Repeated structural roles.** Bottom-up and top-down labelled hashes can
   identify repeated or symmetric subgraphs. Hash equality is only a proposal
   key; candidate groups still undergo exact grounding.

Here, a graph cut is an analysis construct used to choose mechanisms worth
testing. It does not imply inserting a broader barrier or moving an event to a
wider physical frontier.

### Normalized IR semantics and annotations

Graph-only grouping loses the distinctions that make a candidate physically
meaningful. Every structural key may therefore be refined by authoritative IR
facts:

- AIC versus AIV execution domain;
- source and target physical pipelines;
- macro phase and authoritative result-completion phase;
- operation class and dataflow role;
- structured region, guard, and coexecution class;
- enclosing loop and iteration-distance vector;
- physical storage identity, alias root, address space, and access mode;
- logical slot or modulo-lane identity when proven;
- tensor role such as L1 input, L0A/L0B operand, ACC result, UB staging, or GM
  output;
- target capability and legal directed hard-event domain.

These facts should come from normalized interfaces or explicitly versioned
target providers. Operation-name matching is a last-resort diagnostic aid, not
the semantic contract.

For Cube programs, semantic role labels are particularly valuable. A pipeline
can be classified from dataflow as A-stationary, B-stationary,
output/ACC-stationary, or streamed without encoding one rigid instruction
sequence. Structural symmetry can then group repeated load/compute/release
stages within that classification.

## Bounded proposal families

The implementation should build candidates by intersecting graph structure and
semantic keys, rather than enumerating arbitrary subsets.

### 1. Level-boundary groups

For a topological boundary \(L_k\), start with all direct mechanisms associated
with crossing demands. Also consider bounded semantic slices such as:

- one hardware unit;
- one directed pipeline pair;
- one operation-stage pair;
- one storage space or alias root;
- one guard/coexecution class.

Retain the full group and a small number of deterministic leave-one-out or
Pareto variants. Do not enumerate the power set.

### 2. Transitive-basis and connector groups

Compute a scalable reachability basis on unique phase edges. Propose the direct
mechanisms corresponding to non-redundant basis edges. Use a typed connector
index to form only pairs or small sets that share a connector, dominance
frontier, or compatible completion transfer.

This replaces \(O(|M|^2)\) blind pair enumeration with work proportional to the
small connector buckets plus exact grounding of retained proposals.

### 3. Storage-lifecycle groups

Group demands by physical storage identity and region. Within each family,
construct the bounded ownership graph:

```text
writer(i) -> reader(i) -> writer(i + distance)
```

SCCs and cycle bases nominate the ready/release mechanisms that should be tested
together. Exact address/slot identity, guards, loop distance, and access roles
must remain part of the key.

### 4. Symmetry groups

Label nodes with normalized semantic roles and compute deterministic structural
signatures from predecessor and successor level sets. Repeated signatures can
nominate corresponding mechanism sets across unrolled stages or stationary
Cube pipelines.

Merkle-style or Weisfeiler-Lehman-style hashes are useful for indexing only.
Hash collisions or matching shapes never bypass exact identity and coverage
checks.

### 5. Scarcity-local groups

Event scarcity is a separate phase. If an otherwise verified direct plan cannot
be colored, use the allocator's current conflict core to nominate local cut or
lifecycle alternatives. These candidates must be grounded and compared only in
that core. Broad source/target-frontier factoring and `PIPE_ALL` are not part of
normal optimization.

## Proposal-generation bounds

Every family needs explicit deterministic limits:

- maximum proposals per function and per family;
- maximum member mechanisms per proposal;
- maximum leave-one-out variants;
- maximum connector bucket size;
- maximum closure/SCC workspace;
- maximum grounded coverage work;
- maximum seeded selection trials.

When an optional structural budget is exhausted, truncate that family and keep
the direct singleton problem valid. Budget exhaustion must not turn a
singleton-correct program into a compiler rejection.

## Exact grounding and selection

For each retained proposal:

1. install the fixed synchronization supply plus all group members in a fresh
   coverage world;
2. evaluate hierarchical summaries;
3. compare with the flat scoreboard;
4. compare with the bounded structured interpreter when exhaustive;
5. compute singleton-union coverage and additional joint coverage;
6. admit only a strictly useful, fully agreed group.

Selection then operates only on immutable columns. It should compare the normal
singleton greedy result with a bounded set of group-seeded trials, perform
reverse deletion over physical mechanisms, and freshly verify the final
materialized plan. Selected group provenance must survive into reports.

## Lifecycle protocol synthesis

Structural composition alone cannot reduce the physical action count when a
useful ownership plan requires a different protocol. A lifecycle SCC may
therefore nominate a synthesis attempt, but a synthesized column is admitted
only after constructing and certifying a complete recipe:

- priming for every recurrence lane;
- wait-before-reuse and set-after-release;
- legal directed event pairs and usable event IDs;
- periodic lane circulation;
- balanced guards and coexecution;
- zero-trip behavior;
- exit draining;
- event-state safety across iterations;
- fresh demand coverage and post-materialization verification.

The synthesized mechanism records its parent direct mechanisms and witness SCC.
It must provide a non-additive benefit in action count, event pressure, lifetime,
or verified coverage. Merely concatenating parent recipes remains a composition
group.

## Historical GEMM experiment

The historical GEMM ownership kernel is the primary structure-discovery test.
The experiment should report, for every proposed and selected group:

- structural family and semantic key;
- member direct mechanisms and origin demands;
- singleton-union, grounded, and additional coverage;
- selected-before-cleanup and survived-reverse-delete status;
- emitted actions, event domains, lifetimes, and pressure;
- direct-parent fallback cost.

The current level/transitive prototype finds two genuine region-transitive
groups with 48 additional covered rows. Group-seeded selection still retains 80
non-baseline direct mechanisms and remains infeasible in the
`AIC:PIPE_MTE1 -> PIPE_M` event domain. This validates the bounded group-search
idea but also demonstrates its boundary: retaining the original direct recipes
cannot rediscover the fused ready/release ownership plan.

An initial connector-neighborhood ablation admitted 201 leave-one-out-expanded
groups with 2,025 additional covered rows, but selected exactly the same plan.
This showed that connector structure is semantically productive while the
subset expansion is not cost-effective. The implementation therefore retains
one full group per connector and applies independent fixed proposal budgets to
level, transitive, connector, semantic, and storage families.

With those bounds applied, the same fixture produces 42 connector proposals,
admits 36, and grounds 420 additional rows. Enabling every family produces 204
proposals, admits 38, and grounds 468 additional rows. Both configurations still
select 81 mechanisms including fixed baseline supply and fail the same event
domain. The reduced proposal volume therefore preserves the conclusion while
removing most of the unproductive subset search.

The storage family now constructs bounded SCCs over phase nodes for each
physical alias root and requires at least one positive-distance demand. On the
existing repeated ready/reuse regression it rediscovers the two-phase ownership
cycle and its ready/release direct mechanisms. Its joint coverage equals the
singleton union, so it is correctly retained as synthesis provenance but not
admitted as a composition column.

The next GEMM milestone is therefore:

1. recover the ownership SCCs from storage identity, recurrence, Cube role, and
   symmetry information;
2. synthesize balanced lifecycle recipes from those SCCs;
3. reproduce the compact historical ownership structure without importing a
   rigid instruction-sequence recognizer;
4. retain zero body `PIPE_ALL`;
5. pass fresh graph, lifecycle, allocation, and device correctness checks.

### Ownership-synthesis checkpoint

The opt-in storage experiment now has two compact, unsynchronized PTO inputs
under `test/samples/CanonicalSyncReference/`.  They preserve the ownership
topology of the hand-tuned A2/A3 GEMM and TopK kernels without copying their
manual synchronization.  The corresponding lit test checks both analysis and
full materialization.

The storage analysis records an explicit generation for every exact physical
slot lifetime.  A generation contains its logical family and stable family
slot, producer completion cut, first consumer cut, last consumer cut, next
overwrite, control residues, period, and reuse distance.  This identity is
independent of whether a grouped or singleton protocol alternative is being
tested.  Physical storage slots and event-token lanes are deliberately
distinct: two slots may share a serialized token lane, but a witness for one
exact slot can never prove coverage for the other.

The compact GEMM input currently yields 21 storage generations and five
selected protocols: two L1 panel channels, two L0 operand channels, and one
stationary accumulator handoff.  The TopK input yields four generations and
two selected depth-two channels while retaining its same-Vector barrier
obligations.  Independent protocols with periods one through four may coexist
in one loop; materialization and the bounded verifier use their checked
least-common schedule period.

This is a discoverability and correctness checkpoint only.  Coverage is still
grounded from physical cuts and simulated token transitions, materialized
events still pass the independent event verifier, and no recognition result is
accepted as a proof by itself.  However, synthesized protocols currently have
unit mechanism weight.  Device repeatability and a dynamic action/stall cost
model are required before comparing their performance with direct plans or
enabling the family by default.

## Staged implementation

### Stage A: composition prototype

- level-boundary proposals;
- region-local transitive bases;
- AIC/AIV and operation-stage semantic slices;
- physical-storage lifecycle families;
- bounded storage-lifecycle SCC proposals requiring a positive-distance edge;
- exact grounding and additional-coverage admission;
- group-aware greedy selection and reverse deletion;
- provenance, statistics, and dumps.

This stage is implemented behind
`--canonical-sync-structural-cover=all` and defaults to off. Individual
families can be ablated with a comma-separated list containing `level`,
`transitive`, `connector`, `semantic`, and `storage`.

### Stage B: richer bounded graph structure

- directed dominator/post-dominator and separator indexes;
- connector-indexed transitive pairs;
- recurrence-expanded storage SCCs and cycle bases;
- labelled structural signatures for repeated Cube/Vector stages;
- independent family masks for same-branch ablation.

### Stage C: certified synthesis

- translate selected lifecycle SCCs into candidate ready/release recipes;
- prove priming, circulation, guard balance, zero-trip, and drain contracts;
- ground synthesized recipes as independent set-cover columns;
- retain parent-mechanism and graph-witness provenance.

### Stage D: evaluation and retirement

- run direct-only, composition-family, and synthesis-family ablations on the
  corpus and historical GEMM;
- calibrate action, serialization, and event-pressure objectives on device;
- use barrier-heavy regressions as negative controls;
- retire rigid ownership recognizers only after generic synthesis matches their
  correctness and performance envelope.

## Acceptance criteria

The work is accepted only when:

1. structural mode cannot weaken the complete direct-demand basis;
2. default `none` mode remains byte-identical to the hardware-graph baseline;
3. every admitted group has positive exactly grounded joint coverage;
4. optional family-budget exhaustion truncates instead of rejecting;
5. final materialization passes independent demand and event-lifecycle checks;
6. all accepted device kernels are repeatable and agree with the correctness
   baseline;
7. no ordinary plan uses body `PIPE_ALL`;
8. GEMM gains come from certified ownership synthesis rather than broader
   serialization cuts.
