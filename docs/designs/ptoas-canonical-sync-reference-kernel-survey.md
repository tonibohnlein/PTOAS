# CanonicalSync reference-kernel synchronization survey

## Purpose

This document inventories public PTO-ISA and Ascend C kernels that are useful
for studying high-performance synchronization.  The first pass is descriptive:
it records pipeline structure, buffer ownership, and synchronization topology.
It does not yet prescribe CanonicalSync recognizers or claim that a pattern is
universally optimal.

The survey distinguishes three evidence classes:

- **performance reference**: a kernel explicitly presented and measured as a
  performance implementation;
- **production operator source**: an implementation from a maintained CANN
  operator repository, whether or not its synchronization has already been
  audited here;
- **programming example**: authoritative documentation or sample code that
  explains a lifecycle clearly but is not itself evidence of peak performance.

The distinction matters.  A simple double-buffered vector-add sample is a good
semantic oracle for ready/release lifecycles, but it is not a suitable source
for calibrating the cost of barriers or event placement.

## Reference repositories

### PTO-ISA

Repository: <https://github.com/hw-native-sys/pto-isa>

This is the closest source-level match for PTOAS.  The repository contains
manual PTO kernels, an explicit event interface, device tests, and performance
notes.  Its optimization guide describes optimized kernels as overlapped
TLOAD, transform, compute, and TSTORE stages, with explicit warm-up,
steady-state, and drain phases.

Initial high-value directories are:

| Kernel family | Path | Evidence | Why it is useful |
| --- | --- | --- | --- |
| A2/A3 GEMM | `kernels/manual/a2a3/gemm_performance/` | performance reference | Nested L1 and L0 ping/pong lifecycles around Cube compute |
| A2/A3 FlashAttention | `kernels/manual/common/flash_atten/` | performance reference | Cross-Cube/Vector stage graph with several independently sized FIFOs |
| A5 FlashAttention | `kernels/manual/a5/flash_atten/` | performance reference | Same algorithmic graph mapped to a different target and communication mechanism |
| GEMM plus AllReduce | `kernels/manual/a2a3/gemm_ar/` | candidate for detailed audit | Compute/communication overlap and completion at a non-local boundary |
| TGET bandwidth | `kernels/manual/a2a3/tget_bandwidth/` | candidate for detailed audit | Asynchronous transfer ownership and throughput-oriented buffering |

Primary references:

- optimization guide:
  <https://github.com/hw-native-sys/pto-isa/blob/main/docs/coding/opt.md>
- PTO event model:
  <https://github.com/hw-native-sys/pto-isa/blob/main/docs/coding/Event.md>
- GEMM implementation:
  <https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp>
- GEMM tuning notes:
  <https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/a2a3/gemm_performance/README.md>
- common FlashAttention implementation:
  <https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/common/flash_atten/fa_performance_kernel.cpp>
- common FlashAttention notes:
  <https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/common/flash_atten/README.md>
- A5 FlashAttention implementation:
  <https://github.com/hw-native-sys/pto-isa/tree/main/kernels/manual/a5/flash_atten>

### Current CANN operator repositories

Organization: <https://gitcode.com/org/cann/repos>

The current public CANN organization splits operator sources by domain.  These
repositories are important because they contain Ascend C implementations and
tiling variants beyond the small PTO corpus.

| Repository | Contents relevant to this study | Initial audit targets |
| --- | --- | --- |
| <https://gitcode.com/cann/ops-transformer> | Attention, MoE, MC2, and fused transformer operators | FlashAttention, grouped matmul, matmul/collective fusion, paged or sparse attention |
| <https://gitcode.com/cann/ops-nn> | Matmul, convolution, normalization, pooling, quantization, and other neural-network operators | Matmul variants, RMSNorm/LayerNorm, convolution pipelines |
| <https://gitcode.com/cann/ops-math> | Conversion, mathematical, reduction, sorting, and random operators | Reductions, scans, sort/merge networks, conversion pipelines |
| <https://gitcode.com/cann/ops-tensor> | Tensor-oriented operators and reusable Tensor APIs | Queue and storage-lifecycle abstractions shared by kernels |
| <https://gitcode.com/cann/cann-samples> | High-performance practice examples and tuning material | Small examples with an explicit performance narrative and runnable validation |

The older <https://gitee.com/ascend/cann-ops> repository is still valuable as a
historical source because it exposes public Ascend C operators organized under
`src/matmul`, `src/norm`, `src/conv`, `src/quant`, and related categories.  Its
Gitee page now says that the repository is no longer maintained, so new surveys
should prefer the domain-specific GitCode repositories above.

### Authoritative Ascend C programming examples

The static-Tensor programming guide is the clearest reference for the intended
manual protocol:

<https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html>

It explicitly presents:

- forward MTE2-to-Vector and Vector-to-MTE3 dependencies;
- reverse Vector-to-MTE2 and MTE3-to-Vector dependencies protecting reused
  buffers;
- a ping/pong event lane chosen by `loopIdx & 1`;
- prologue priming of both release directions;
- epilogue draining of both lanes;
- the restriction that manually managed event IDs 6 and 7 are unavailable.

The queue-based Ascend C add sample is a useful higher-level version of the
same three-stage pipeline:

<https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/3_add_kernellaunch/AddKernelInvocationNeo/add_custom.cpp>

Its `TQue` objects have depth two and express ownership through
`AllocTensor`, `EnQue`, `DeQue`, and `FreeTensor`.  The underlying synchronization
is hidden, but the storage lifecycle is explicit in the IR-level operations.

## Inspected synchronization structures

### A2/A3 PTO GEMM: nested ping/pong ownership

The tuned PTO GEMM is organized as:

```text
GM --TLOAD/MTE2--> L1 panels --TEXTRACT/MTE1--> L0A/L0B
                                             --TMATMUL/M--> L0C
                                             --TSTORE--> GM
```

The kernel partitions `M` and `N` across Cube cores.  Within a core it uses
three nested loops over output tiles and K slices.  Its synchronization is not
one flat chain.  It contains several coupled lifecycles:

1. **L1 panel lifecycle.**  Two L1 slots hold A and B panels.  A load waits for
   the previous extract to release the selected slot.  After TLOAD, separate A
   and B ready signals allow TEXTRACT to consume them.  Release is delayed until
   all `stepK` slices from that panel have been extracted.
2. **L0 operand lifecycle.**  Two L0A/L0B slots alternate.  TEXTRACT waits until
   Cube has released a slot, fills it, and signals that the operands are ready.
   TMATMUL waits for readiness, consumes the operands, and releases the slot for
   a later extract.
3. **Accumulator lifecycle.**  L0C is output-stationary across the K loop:
   `TMATMUL` initializes it and `TMATMUL_ACC` updates the same accumulator.  It
   is handed to the store path only after the final K slice.
4. **Protocol boundary.**  Reverse directions are primed before the loop and
   drained after all nested loops.  Ping/pong indices remain live across output
   tile boundaries rather than being reinitialized for each inner loop.

This is a strong lifecycle-synthesis example.  It is a difficult first test for
composition-only covering because the efficient recipe is not just a subset of
tight demand-owned event pairs: it shares periodic ready/release channels over
many dynamic demand instances.

### PTO FlashAttention: a cyclic stage network with several FIFO depths

The inspected common PTO FlashAttention kernel has four major stages:

```text
QK Cube matmul -> QK FIFO -> Vector softmax
Vector softmax -> P FIFO -> PV Cube matmul
PV Cube matmul -> PV FIFO -> Vector running update/output
```

Its structure is more informative than a simple linear pipeline:

- QK, P, and PV transfers use distinct `TPipe` ring FIFOs;
- FIFO depth and warm-up/preload depth are tuning parameters, not universally
  fixed at two;
- Cube and Vector sides use named ready/consumed flags, `TALLOC`, `TPUSH`,
  `TPOP`, and `TFREE` to represent ownership transfer;
- local MTE2/MTE1/M/Vector/MTE3 events synchronize on-core movement and
  computation inside each stage;
- several local ping/pong indices advance independently;
- the prologue supplies initially free tokens and the epilogue waits for all
  remaining event lanes before the mandatory final drain;
- the A2/A3 and A5 variants keep the high-level stage graph but use different
  Cube/Vector communication directions and buffer mappings.

This kernel is a good source for graph-structural experiments because it has
multiple repeated motifs, meaningful separators between stages, and IR-visible
queue roles.  It also demonstrates why graph shape must be combined with
storage and FIFO annotations: two isomorphic-looking stage edges may use
different buffer depths and therefore require different token protocols.

### Ascend C static-Tensor vector pipeline: the minimal complete lifecycle

The official manual double-buffer example has the stage graph:

```text
MTE2 load -> Vector compute -> MTE3 store
```

For each ping/pong slot it contains both ready and release relations:

```text
MTE2(i) -> V(i)       input is ready
V(i)    -> MTE3(i)    output is ready
V(i)    -> MTE2(i+2)  input slot is free
MTE3(i) -> V(i+2)     output slot is free
```

The resulting graph becomes cyclic only after positive-distance edges are
included.  This is the best initial test for recurrence-expanded SCC discovery:
it is small, its intended protocol is specified directly by the vendor, and
its lifecycle has an unambiguous two-lane implementation.

### Ascend C `TQue`: ownership expressed above raw flags

The queue-based add example has the same load/compute/store stages, but
`AllocTensor`/`FreeTensor` describe slot ownership and `EnQue`/`DeQue` describe
producer-to-consumer transfer.  This gives a useful semantic mapping:

```text
AllocTensor  -> acquire a free storage slot
EnQue        -> publish a ready slot
DeQue        -> acquire the ready slot
FreeTensor   -> publish the slot as reusable
```

An autosynchronizer should exploit these authoritative roles when they survive
in IR.  The dependence graph remains the correctness authority, while queue
roles provide a much better proposal key than operation names or graph shape
alone.

## Pattern inventory for the next source-audit pass

The following structures should be recorded for every reference kernel.  They
are observations, not proposed built-in recognizers.

| Structure | What to record | Representative sources |
| --- | --- | --- |
| Linear staged pipeline | ordered stages, true dependency boundaries, same-pipe barriers | Ascend C static Tensor, queue-based add |
| Ping/pong ownership cycle | storage slot, ready edge, release edge, lane count, prime/drain | static Tensor, PTO GEMM |
| Multi-depth ring FIFO | producer/consumer stages, depth, occupancy/backpressure contract | PTO FlashAttention |
| Stationary Cube operand/result | A/B/ACC stationary role, streamed dimension, reuse distance | PTO GEMM and CANN matmul variants |
| Decimated producer | one load supplies several extracts or computes; release after the final use | PTO GEMM `stepK` panels |
| Cross-Cube/Vector handoff | intermediate storage, direction, FIFO ownership, target-specific transport | PTO FlashAttention, CANN attention |
| Reduction/update state | persistent running maximum/sum/output and scratch reuse | FlashAttention, RMSNorm/LayerNorm, reductions |
| Same-pipeline sequencing | barriers needed inside a multi-op Vector or MTE phase | CANN norm/math operators |
| Compute/communication overlap | local result lifecycle plus remote collective completion | GEMM-AllReduce and MC2 kernels |
| Warm-up and drain | initial free tokens, partial first iteration, exit consumption | all manually pipelined references |

For each concrete implementation, the survey should eventually store:

```text
target and core domain
loop nest and dynamic stage order
physical storage slots and alias roots
producer/consumer operation classes and hardware pipelines
forward ready demands
positive-distance release demands
event/FIFO lane count
prologue, steady-state, and epilogue actions
guards, tail iterations, and zero-trip behavior
same-pipe barriers and whole-pipeline drains
measured performance context, when published
```

## Status of the current CanonicalSync structural experiment

The historical GEMM experiment did **not** explicitly unroll the complete
nested loop nest and then run level-set or symmetry discovery on that large
graph.

What the current prototype does is:

1. build a region-level graph of phase nodes and zero-distance demand edges;
2. retain typed positive-distance recurrence edges;
3. compute region-local topological levels, transitive bases, connector
   neighborhoods, semantic slices, and storage-lifecycle SCC proposals;
4. nominate bounded sets of existing direct mechanisms; and
5. ground every proposed set through the normal completion analysis and a
   bounded structured interpreter.

Thus, bounded unrolling is used as a **coverage verifier**, but an explicitly
unrolled multi-iteration graph is not yet used as the primary **proposal
generator**.

On the historical GEMM fixture, the existing analysis found many genuine joint
coverage effects:

- the transitive family found two admitted groups and 48 additional covered
  rows;
- the bounded connector family admitted 36 groups and 420 additional rows;
- all enabled families admitted 38 groups and 468 additional rows.

None reduced the selected physical plan: it still selected 81 mechanisms
including fixed baseline supply and remained infeasible in the
`AIC:PIPE_MTE1 -> PIPE_M` event domain.  This is evidence that graph-guided
proposal selection works, but also that composition of the original direct
recipes cannot by itself reproduce GEMM's cheaper periodic ownership recipe.
The corresponding detailed plan is in
`docs/designs/ptoas-canonical-sync-structural-cover-plan.md`.

## Better graph-structure study set

The historical GEMM should remain an end goal, but it should not be the only or
first structural benchmark.  A staged sequence is more diagnostic:

1. official one-buffer and two-buffer Vector load/compute/store lifecycles;
2. a three-buffer variant to test whether lane depth is discovered rather than
   hard-coded;
3. a decimated producer in which one load feeds several consumers;
4. an output-stationary Cube microkernel with one L1 and one L0 lifecycle;
5. the full nested PTO GEMM;
6. PTO FlashAttention with several independently sized inter-stage FIFOs;
7. CANN reductions/norms and compute/communication fusion as negative and
   generalization tests.

For proposal discovery, a useful next experiment is a small explicit expansion
of two or three loop iterations, with every node labelled by:

```text
operation role, hardware pipe, physical storage slot,
loop/iteration coordinate, guard, and recurrence distance
```

Level sets, separators, cycle bases, and structural hashes can then nominate
sets of direct mechanisms in the expanded graph.  Repeated nodes are quotiented
back to region roles before exact grounding.  The graph analysis still does not
choose a synchronization recipe; it only identifies a small number of direct
mechanism groups worth testing for unexpectedly large joint coverage.

## Immediate follow-up

The next survey pass should inspect concrete kernels, not merely repository
names, in this order:

1. PTO A2/A3 GEMM and common/A5 FlashAttention, already partially catalogued;
2. official static-Tensor single- and double-buffer examples;
3. CANN `ops-nn` matmul and normalization implementations;
4. CANN `ops-transformer` attention and MC2 implementations;
5. PTO GEMM-AllReduce and TGET bandwidth examples;
6. CANN math reductions, scans, and sorting networks.

Each audit should add a compact lifecycle diagram and a table of concrete
synchronization actions.  Only after that inventory exists should recurring
structures be evaluated as graph/IR proposal families.
