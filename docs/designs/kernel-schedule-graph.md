# Kernel scheduling graph

`pto-print-kernel-schedule-graph` is the analysis entry point for whole-kernel
compute/transfer scheduling work. It does not change PTO IR.

## Graph contract

Each leaf operation that implements `OpPipeInterface`, reports a concrete pipe,
and is not itself a synchronization operation becomes one node. The node's
OneStopParallel vertex type is the numeric `PIPE` value returned by
`OpPipeInterface::getPipe()`. The legacy structural mode uses unit work
weights; communication and memory weights are zero. For latency scoring,
`pto-print-kernel-schedule-graph` accepts a pinned PTO-ISA
`formula_params.csv` through `--duration-table`. It derives an exact
`(opcode, dtype, rows, cols)` signature for every priced operation and, with
`--require-exact-durations`, refuses the graph if any scheduled node lacks a
matching formula. It never substitutes a family average. Transfers,
conversions, and signatures requiring additional PTO-ISA parameters remain
unavailable until their complete signatures are represented.

The graph separately records a latency on each dependency. Ordinary
SSA/control/memory edges have zero added delay. A placement analysis adds
explicit `placement-reuse-{raw,war,waw}` hazards for selected overlapping DSA
allocations, with one globally chosen synchronization delay. The base graph
therefore remains non-reusing and reusable across placements.
`getLongestPathCycles()` scores the whole selected placement—not a sum of
independent pair penalties—as the longest path through operation work plus
dependency delay. Positive-distance recurrences remain outside this
per-iteration path and must be scored through an initiation interval model.

The underlying `ScheduleGraph` is PTOAS-owned and has no MLIR dependency. Its
query API satisfies OneStopParallel's typed computational DAG contract:

- `Vertices`, `NumVertices`, and `NumEdges`
- `Parents`, `Children`, `InDegree`, and `OutDegree`
- vertex work, communication, and memory weights
- `VertexType` and `NumVertexTypes`

This keeps PTOAS independent of a particular OneStopParallel checkout while
allowing the graph to be passed directly to its templated algorithms.

## Dependencies

The first implementation records:

- SSA producer-to-consumer dependencies, tracing through pure scalar/view ops.
- Block-local memory RAW, WAR, and WAW dependencies from
  `MemoryEffectOpInterface`; structured boundaries conservatively order memory
  effects that cross blocks.
- Conservative structured-control dependencies around regions, calls, and
  unmodelled side-effecting operations.
- Memory recurrences for all `LoopLikeOpInterface` loops and SSA recurrences
  for `scf.for` iteration arguments.

Positive-distance dependencies retain the owning loop operation in their
metadata. This distinction is required for nested loops, where a distance of
one has a different meaning at each loop level.

Alias-producing views are currently collapsed to a common root. This is safe
but may introduce false dependencies between statically disjoint partitions;
range-sensitive alias analysis is a later refinement.

The initial builder supports structured control flow. It rejects operations
with explicit CFG successors instead of returning a graph with missing
cross-block dependencies. CFG scheduling can be added later with explicit
backedge and region metadata.

Loop-carried dependencies have a positive iteration distance. They are stored
as dependency metadata and shown as dashed DOT edges, but are not inserted into
the zero-distance adjacency DAG. This preserves the DAG contract required by
OneStopParallel schedulers.

DOT uses blue nodes for transfer operations and amber nodes for compute
operations. Data-dependency kinds have distinct edge colors, control edges are
purple, and positive-distance recurrences are dashed.

The current structured-control model materializes conservative boundary edges.
This can be quadratic for large regions; a compact boundary representation or
structural transitive reduction is required before using the graph as a
large-kernel scheduling input.

## Inspection

Stable text output:

```bash
pto-test-opt input.pto '-pto-print-kernel-schedule-graph=format=text'
```

Graphviz output:

```bash
pto-test-opt input.pto '-pto-print-kernel-schedule-graph=format=dot' \
  | sed -n '/^digraph /,/^}$/p' > schedule.dot
dot -Tsvg schedule.dot -o schedule.svg
```
