# Kernel scheduling graph

`pto-print-kernel-schedule-graph` is the analysis entry point for whole-kernel
compute/transfer scheduling work. It does not change PTO IR.

## Graph contract

Each leaf operation that implements `OpPipeInterface`, reports a concrete pipe,
and is not itself a synchronization operation becomes one node. The node's
OneStopParallel vertex type is the numeric `PIPE` value returned by
`OpPipeInterface::getPipe()`. The legacy structural mode uses unit work
weights; communication and memory weights are zero. For latency scoring,
`pto-print-kernel-schedule-graph` accepts either a pinned PTO-ISA
`formula_params.csv` through `--duration-table`, or PyPTO's resolved composite
provider output through `--node-durations`. The latter combines calibrated
formula/signature evidence with pinned analytical and generic Perf-Sim
approximations without duplicating that logic in PTOAS. Its JSON identifies the
evidence class and binds every nonnegative cycle value and provenance string to
the function, exact unweighted graph SHA-256, node ID, operation, and
`pypto_access_order`. `--require-exact-durations` is the historical name of the
completeness gate: it refuses unresolved/fallback nodes, while the evidence
class distinguishes calibrated signatures from resolved approximations. The
two duration inputs are mutually exclusive; neither may substitute an
unpinned family average or unsupported fallback.

The graph separately records a latency on each dependency. Ordinary
SSA/control/memory edges have zero added delay. A placement analysis adds
explicit `placement-reuse-{raw,war,waw}` hazards for selected overlapping DSA
allocations, with one globally chosen synchronization delay. The base graph
therefore remains non-reusing and reusable across placements.
`getLongestPathCycles()` scores the whole selected placement—not a sum of
independent pair penalties—as the longest path through operation work plus
dependency delay. Positive-distance recurrences remain outside this
per-iteration path. `getRecurrenceInitiationIntervalCycles()` scores their
return paths as a loop initiation-interval lower bound, and
`getLatencyLowerBoundCycles()` reports the maximum of that bound and the
acyclic longest path.

The analysis indexes zero-distance edge latencies once. Longest-path scoring
is `O(V + E)`. Recurrence return paths are computed once per distinct
recurrence target, for `O(E + T(V + E) + R)` total work, where `T` is the
number of distinct recurrence targets and `R` the number of recurrence edges.
This deliberately avoids rescanning all dependencies for every traversed edge.

These are bounds for the supplied graph, not a complete invocation-latency
claim. The recurrence bound includes cycles with one positive-distance edge,
not cycles composed of multiple recurrence edges. Control-path feasibility,
fixed per-pipe issue order, and finite loop execution must be validated before
interpreting it as a kernel score. Do not count a numeric bound as a complete
latency prediction.

For low-memory source validation, the `pto-reuse-model-opt` target registers
only this graph pass and links its implementation via `PTOKernelScheduleModel`.
It uses the same sources as the full tool, but avoids unrelated transform
objects. It is not an assembler or a replacement for product code generation.

PTOAS deliberately does not reconstruct DSA buffer identity from lowered PTO:
that information belongs to the producer that exported the DSA problem and
solution. `--placement-reuse-edges` accepts its verified join as a small JSON
document:

```json
{
  "schema_version": 1,
  "function": "kernel",
  "edges": [{
    "source_node": 4,
    "target_node": 9,
    "kind": "war",
    "iteration_distance": 1,
    "recurrence_loop_depth": 1,
    "provenance": "buffers=12,37;accesses=18,24"
  }]
}
```

Together with `--reuse-sync-latency-cycles=W`, this adds every selected reuse
edge to one complete graph before scoring. The topology-only PyPTO contract
also binds the edge set to the exact unweighted graph SHA-256. The importer
validates function identity, node IDs, direction/kind, recurrence distance and
common-loop depth, and non-empty provenance. A file naming another function is
an error rather than an empty edge set. Positive-distance placement edges stay
out of the acyclic adjacency graph but contribute to the reported
recurrence-II and combined latency lower bounds. It does not infer missing
relations.

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

Complete placement latency using graph-bound PyPTO inputs:

```bash
pto-test-opt input.pto '-pto-print-kernel-schedule-graph=format=text' \
  -o /dev/null > ptoas-schedule-graph.txt
# Produce durations.json and edges.json against ptoas-schedule-graph.txt.
pto-test-opt input.pto \
  '-pto-print-kernel-schedule-graph=node-durations=durations.json placement-reuse-edges=edges.json reuse-sync-latency-cycles=64 require-exact-durations=true'
```

Graphviz output:

```bash
pto-test-opt input.pto '-pto-print-kernel-schedule-graph=format=dot' \
  | sed -n '/^digraph /,/^}$/p' > schedule.dot
dot -Tsvg schedule.dot -o schedule.svg
```
