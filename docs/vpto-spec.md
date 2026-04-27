# PTO micro Instruction Spec — Merged Draft (A5)

> **Status:** DRAFT for review
> **Base:** [vpto-spec.md](https://github.com/mouliangyu/PTOAS/blob/feature-vpto-backend/docs/vpto-spec.md) (2026-03-20)
> **Updated:** 2026-03-27

---

## Part I: Architecture Overview

### Overview

This document defines the PTO micro Instruction, a compiler-internal and externally facing specification designed to represent vector compute kernels within the PTO architecture. Much like NVVM provides a robust IR for GPU architectures, the PTO micro Instruction serves as the direct bridge between high-level programming models and the underlying hardware ISA, providing a precise, low-level representation of vector workloads explicitly designed for the Ascend 950 architecture.

#### Position in the Stack and Layer Modeled

The PTO micro Instruction operates as a very low-level intermediate representation within the PTO compiler stack. It is uniquely designed to accurately and comprehensively express all architectural information of the Ascend 950 hardware. It specifically models the bare-metal vector execution layer, making hardware-specific capabilities and constraints, such as exact vector lane configurations, memory space hierarchies, and hardware-specific fusion semantics, fully transparent and controllable.

#### PTO Instruction Modes and Compilation Flows

Within the end-to-end PTO software stack, PTO instructions may appear in three closely related authoring or lowering modes:

- **PTO Tile Instruction**: tile-oriented PTO code that serves as a nano-kernel encapsulation of Tile operations, primarily expressing computation and data movement in terms of tile buffers, tile shapes, and tile-local layout.
- **PTO micro Instruction**: vector-execution-oriented PTO code that makes DMA setup, vector registers, masks, synchronization, and `__VEC_SCOPE__` boundaries explicit. This document is centered on this mode.
- **PTO Tile+micro Instruction**: a hybrid PTO form that keeps tile-level orchestration while embedding explicit micro-instruction regions where direct vector-pipeline control is required.

From these PTO instruction forms, the stack can proceed along two main compilation flows:

- **CCE generation flow**: PTO ISA is lowered into a CCE-oriented representation, which is then compiled by the BiSheng toolchain into Ascend device binaries.
- **Bytecode generation flow**: PTO ISA is emitted as bytecode, which is then compiled by the BiSheng toolchain into Ascend device binaries.

```text
High-level frameworks / DSLs / library kernels
                    |
                    v
         +----------------------------------+
         |          PTO ISA layer           |
         |                                  |
         |  (1) PTO Tile Instruction        |
         |  (2) PTO micro Instruction       |
         |  (3) PTO Tile+micro Instruction  |
         +----------------+-----------------+
                          |
             +------------+------------+
             |                         |
             v                         v
 +-------------------------+   +-------------------------+
 | Path A: generate CCE    |   | Path B: generate        |
 | (CCE-oriented form)     |   | bytecode                |
 +------------+------------+   +------------+------------+
              |                             |
              v                             v
   +-----------------------------------------------+
   |               BiSheng compiler                |
   +---------------------------+-------------------+
                               |
                               v
                 +-----------------------------+
                 |   Ascend device binaries    |
                 +-----------------------------+
```

#### Why External Developers Read or Author PTO micro Instruction

While the majority of users will interact with the PTO architecture via higher-level frameworks, external developers may need to read or author PTO micro Instruction directly for several key reasons:

- Custom Toolchain Development: build custom compiler frontends or domain-specific languages (DSLs) that target the Ascend 950 architecture with maximum hardware utilization.
- Performance Engineering: inspect the output of high-level compiler passes, verify fine-grained optimization behaviors, and pinpoint performance bottlenecks at the architectural level.
- Micro-Optimization: hand-author highly optimized, critical mathematical kernels using a stable, precise IR when higher-level abstractions cannot achieve the theoretical peak performance of the hardware.

#### Relationship to CCE

The PTO micro Instruction is designed to express the full semantic capabilities of the Compute Cube Engine (CCE), but with significant structural and pipeline advantages for compiler development.

- Bypassing the C/Clang Pipeline: while CCE heavily relies on C/C++ extensions parsed by Clang, the PTO micro Instruction operates entirely independently of the C language frontend. By bypassing Clang AST generation and frontend processing, utilizing the PTO micro Instruction significantly reduces overall compilation time and memory overhead.
- Enhanced IR Verification: because the PTO micro Instruction is a strongly typed, SSA-based (Static Single Assignment) compiler IR rather than a C-wrapper API, it provides a much more rigorous and detailed IR verification process. Structural inconsistencies, invalid memory access patterns, and operand type mismatches are caught immediately with precise, explicit diagnostic feedback, providing developers with much higher visibility into kernel correctness than traditional CCE error reporting.

#### Intended Audience

This document is written for compiler engineers, library writers, and advanced performance architects. We expect the reader to have a working understanding of modern compiler infrastructure, specifically MLIR, the principles of Static Single Assignment (SSA) form, and a deep understanding of the vector-processing capabilities of the Ascend 950 architecture.

### Getting Started

The PTO micro Instruction is architected as a performance-critical layer within the compiler stack, specifically designed to exploit the **Decoupled Access-Execute** (DAE) nature of the Ascend 950 hardware.

#### Hardware Pipeline Modeling

The IR is structured to mirror the three primary hardware pipelines of the Ascend 950 architecture. Correct PTO micro Instruction authoring requires managing the interaction between these asynchronous units:

**MTE2** (Memory Transfer Engine - Inbound): Responsible for moving data from Global Memory (GM) to the Unified Buffer (UB).

**Vector Core** (Computation): The primary engine for executing SIMD operations on data stored in UB.

**MTE3** (Memory Transfer Engine - Outbound): Responsible for moving processed data from UB back to GM.

#### Architecture Detail: Vector Lane (VLane)

The vector register is organized as **8 VLanes** of 32 bytes each. A VLane is the atomic unit for group reduction operations.

```
vreg (256 bytes total):
┌─────────┬─────────┬─────────┬─────┬─────────┬─────────┐
│ VLane 0 │ VLane 1 │ VLane 2 │ ... │ VLane 6 │ VLane 7 │
│   32B   │   32B   │   32B   │     │   32B   │   32B   │
└─────────┴─────────┴─────────┴─────┴─────────┴─────────┘
```

Elements per VLane by data type:

| Data Type | Elements/VLane | Total Elements/vreg |
|-----------|---------------|-------------------|
| i8/si8/ui8 | 32 | 256 |
| i16/si16/ui16/f16/bf16 | 16 | 128 |
| i32/si32/ui32/f32 | 8 | 64 |
| i64/si64/ui64 | 4 | 32 |

#### Memory and Synchronization Model

The PTO micro Instruction enforces a strict memory hierarchy. The Unified Buffer (UB) is the only valid operand source for vector compute instructions. Consequently, the architecture of a PTO micro Instruction program is defined by the explicit management of data movement:

**Address Space Isolation**: The IR uses `!pto.ptr<element-type, space>` to distinguish between GM (`!pto.ptr<T, gm>`) and UB (`!pto.ptr<T, ub>`). The verifier ensures that vector compute operations do not access GM directly; data must first be moved into UB.

**UB Capacity**: The Unified Buffer provides 256KB of on-chip SRAM (also referred to as "vecTile").

**Data Flow**:

```
┌─────────────────────────────────────────────┐
│                 Global Memory (GM)           │
│              (Off-chip HBM/DDR)              │
└─────────────────────┬───────────────────────┘
                      │ DMA (MTE2 inbound / MTE3 outbound)
┌─────────────────────▼───────────────────────┐
│              Unified Buffer (UB)             │
│            (On-chip SRAM, 256KB)             │
└─────────────────────┬───────────────────────┘
                      │ Vector Load/Store (PIPE_V)
┌─────────────────────▼───────────────────────┐
│           Vector Register File (VRF)         │
│     vreg (256B each) + mask (256-bit each)   │
└─────────────────────────────────────────────┘
```

1. **GM → UB**: DMA transfer via MTE2 (`pto.dma_load`)
2. **UB → vreg**: Vector Load instructions (`pto.vlds`, `pto.vldsx2`, etc.)
3. **vreg → vreg**: Compute instructions (`pto.vadd`, `pto.vmul`, etc.)
4. **vreg → UB**: Vector Store instructions (`pto.vsts`, `pto.vstsx2`, etc.)
5. **UB → GM**: DMA transfer via MTE3 (`pto.dma_store`)

The grouped DMA surface in this specification covers GM↔UB transfer only.
Low-level raw copy families such as UB→UB copy use separate operand contracts
and are outside this grouped DMA interface.

**Load/Store Access Patterns**:

For UB↔vreg data movement, besides contiguous load/store, the architecture provides rich access pattern support including strided access, pack/unpack, interleave/deinterleave, broadcast, upsample/downsample, channel split/merge, gather/scatter, and squeeze/expand operations. For detailed instruction syntax and distribution modes, refer to the [Vector Load/Store](isa/03-vector-load-store.md) group in the ISA specification.

#### Synchronization Model

The Ascend 950 architecture employs a cluster-based design with a 1:2 ratio of Cube cores to Vector cores. The PTO micro Instruction provides multiple levels of synchronization to manage concurrent execution across pipelines and cores:

**Inter-Core Synchronization (within a cluster):**

Synchronization between cores within the same cluster is achieved via the core sync mechanism using `pto.set_intra_core` and `pto.wait_intra_core` operations. This enables coordination between Cube and Vector cores sharing the same cluster resources.

**Vector Core Pipeline Synchronization:**

Within a single core, multiple pipelines operate asynchronously:

- **MTE2 (PIPE_MTE2)**: DMA copy-in from GM to UB
- **MTE3 (PIPE_MTE3)**: DMA copy-out from UB to GM
- **Vector Compute (PIPE_V)**: Vector ALU operations
- **Scalar (PIPE_S)**: Scalar unit running the kernel program

Pipeline synchronization can be achieved through two mechanisms:

1. **Flag/Event mechanism**: `pto.set_flag` and `pto.wait_flag` operations resolve Read-After-Write (RAW) and Write-After-Read (WAR) hazards between pipelines.

2. **Buffer-ID mechanism**: `pto.get_buf` and `pto.rls_buf` provide finer-grained synchronization through buffer acquisition and release semantics for producer-consumer coordination.

**Intra-Pipeline Memory Barriers (within `__VEC_SCOPE__`):**

Within the vector execution scope, the hardware does not track UB address aliasing between reg↔UB accesses. When UB addresses overlap or alias between vector load/store operations, explicit memory barriers are required:

```c
pto.mem_bar "VV_ALL"      // All prior vector ops complete before subsequent
pto.mem_bar "VST_VLD"     // All prior vector stores visible before subsequent loads
pto.mem_bar "VLD_VST"     // All prior vector loads complete before subsequent stores
```

Without proper barriers, loads may see stale data or stores may be reordered incorrectly.

#### Execution Scopes (__VEC_SCOPE__)

`__VEC_SCOPE__` is the IR-level representation of a Vector Function (VF) launch. In the PTO architecture, it defines the hardware interface between the Scalar Unit and the Vector Thread.

In PTO micro Instruction source IR, vector execution scopes are modeled as dedicated region ops. The default form is `pto.vecscope`; when the scope body must reject implicit capture and require explicit region arguments, use `pto.strict_vecscope`.

**Scalar-Vector Interface:**

The execution model follows non-blocking fork semantics:

- Scalar invocation: the scalar processor invokes a vector thread by calling a VF. Once the launch command is issued, the scalar unit does not stall and continues executing subsequent instructions in the pipeline.
- Vector execution: after invocation, the vector thread independently fetches and executes the instructions defined within the VF scope.
- Parallelism: this decoupled execution allows the scalar and vector units to run in parallel, so the scalar unit can prepare addresses or manage control flow while the vector unit performs heavy SIMD computation.

**Launch Mechanism And Constraints:**

- Parameter buffering: all arguments required by the VF must be staged in hardware-specific buffers.
- Launch overhead: launching a VF incurs a latency of a few cycles. Very small VFs should account for this overhead because launch cost can rival useful computation time.

**MLIR Representation:**

```mlir
pto.vecscope {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}
```

**Strict MLIR Representation:**

```mlir
pto.strict_vecscope(%ub, %ub_out, %lane) {
^bb0(%in: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, ub>, %iv: index):
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %in[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %out[%iv], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
} : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index) -> ()
```

`pto.strict_vecscope` is the strict form of `pto.vecscope`.

- `pto.vecscope` allows the body to use surrounding SSA values directly.
- `pto.strict_vecscope` requires every external value used by the body to be passed through the op operand list and received as a body block argument.
- `pto.strict_vecscope` rejects implicit capture from the surrounding scope.
- both ops still represent one explicit VPTO vector interval.
- regardless of whether the source form uses `pto.vecscope`,
  `pto.strict_vecscope`, or a lowered carrier loop with
  `llvm.loop.aivector_scope`, every op that produces or consumes `!pto.vreg`,
  `!pto.mask<...>`, or `!pto.align` must be enclosed by exactly one vector
  interval
- nested vector intervals are not part of the legal VPTO surface; ordinary
  nested `scf.for` structure is fine, but one vector interval may not contain
  another vector interval

### Example: VecScope

```mlir
pto.dma_load %7, %2, %c0_i64, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64)
  : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64

pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

pto.vecscope {
  scf.for %lane = %c0 to %9 step %c64 {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %v = pto.vlds %2[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  }
}

pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.dma_store %8, %14, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64)
  : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64
```

### Example: Strict VecScope

```mlir
pto.strict_vecscope(%ub_in, %ub_out, %lane, %remaining) {
^bb0(%in: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, ub>, %iv: index, %rem: i32):
  %mask, %next_remaining = pto.plt_b32 %rem : i32 -> !pto.mask<b32>, i32
  %v = pto.vlds %in[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %out[%iv], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
} : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index, i32) -> ()
```

Use `pto.strict_vecscope` when the source form should make all vector-scope inputs explicit in the region signature instead of relying on surrounding SSA visibility. The scope op itself only defines the vector-interval boundary and region argument contract.

### Cluster Programming Model

#### Overview

An A5 cluster contains one **Cube block** (AIC) and two **Vector blocks** (AIV0, AIV1). Each
block runs an **independent program** under its own Scalar Unit (SU), with its own issue queues:

| Block | Issue Queues |
|---|---|
| Cube (AIC) | MTE2, MTE1, CUBE, FIXP |
| Vector (AIV) | MTE2, VEC, MTE3 |

There is no implicit synchronization between blocks. All coordination between the Cube and Vector
programs is **explicit**, via the primitives described below.


```
┌─────────────────────────────────────── A5 CLUSTER ───────────────────────────────────────┐
│                                                                                           │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐           │
│  │   CUBE CORE (AIC)   │    │  VECTOR 0 (AIV0)    │    │  VECTOR 1 (AIV1)    │           │
│  │                     │    │   subblock_id = 0   │    │   subblock_id = 1   │           │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │           │
│  │  │  Scalar Unit  │  │    │  │  Scalar Unit  │  │    │  │  Scalar Unit  │  │           │
│  │  │  (SU)         │  │    │  │  (SU)         │  │    │  │  (SU)         │  │           │
│  │  │  runs cube    │  │    │  │  runs vec     │  │    │  │  runs vec     │  │           │
│  │  │  program      │  │    │  │  program      │  │    │  │  program      │  │           │
│  │  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │           │
│  │   ── Issue Queues ─ │    │   ── Issue Queues ─ │    │   ── Issue Queues ─ │           │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │           │
│  │  │     MTE2      │  │    │  │     MTE2      │  │    │  │     MTE2      │  │           │
│  │  │    GM → L1    │  │    │  │    GM → UB    │  │    │  │    GM → UB    │  │           │
│  │  ├───────────────┤  │    │  ├───────────────┤  │    │  ├───────────────┤  │           │
│  │  │     MTE1      │  │    │  │      VEC      │  │    │  │      VEC      │  │           │
│  │  │   L1 → L0A/B  │  │    │  │  SIMD compute │  │    │  │  SIMD compute │  │           │
│  │  ├───────────────┤  │    │  ├───────────────┤  │    │  ├───────────────┤  │           │
│  │  │     CUBE      │  │    │  │     MTE3      │  │    │  │     MTE3      │  │           │
│  │  │  MMAD (L0C)   │  │    │  │    UB → GM    │  │    │  │    UB → GM    │  │           │
│  │  ├───────────────┤  │    │  └───────────────┘  │    │  └───────────────┘  │           │
│  │  │     FIXP      │  │    │                     │    │                     │           │
│  │  │  L0C → UB     │  │    │                     │    │                     │           │
│  │  │  (fixpipe)    │  │    │                     │    │                     │           │
│  │  └───────────────┘  │    │                     │    │                     │           │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘           │
│                                                                                           │
│  ┌────────────────────── SC (System Controller) ──────────────────────────────────────┐  │
│  │                                                                                     │  │
│  │   32 semaphores · 4-bit counter each · shared for C→V and V→C directions           │  │
│  │                                                                                     │  │
│  │   ┌──────────────────────────────────────────────────────────────────────────────┐ │  │
│  │   │  sema_id 0 –15  │ [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7][ 8][ 9][10][11][12][13][14][15] │ │  │
│  │   │                 │                    ↕  C→V / V→C  ↕                         │ │  │
│  │   │                 │              communicate with AIV0 (subblock_id=0)          │ │  │
│  │   ├──────────────────────────────────────────────────────────────────────────────┤ │  │
│  │   │  sema_id 16–31  │ [16][17][18][19][20][21][22][23][24][25][26][27][28][29][30][31] │ │  │
│  │   │                 │                    ↕  C→V / V→C  ↕                         │ │  │
│  │   │                 │              communicate with AIV1 (subblock_id=1)          │ │  │
│  │   └──────────────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                                     │  │
│  │   → 16 sema_id pairs (0–15) available for 1:2 C:V sync per slot                   │  │
│  │                                                                                     │  │
│  │   set_intra_block(trigger_pipe, sema_id)  ──►  increments semaphore                │  │
│  │   wait_intra_core(wait_pipe,    sema_id)  ──►  stalls pipe until semaphore > 0     │  │
│  │                                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Intra-Cluster Synchronization

Within a cluster, the PTO micro ISA provides two levels of synchronization:

**Intra-core pipeline sync** (`pto.set_flag` / `pto.wait_flag`): coordinates the asynchronous
pipelines *within a single block* — for example, ensuring MTE2 completes a GM→UB load before
the VEC pipeline begins computation. This does not cross block boundaries.

**Inter-block sync** (`pto.set_intra_block` / `pto.wait_intra_core`): coordinates between the
Cube block and a Vector block within the same cluster. The sender specifies which **local
pipeline** commits the signal, ensuring the preceding operation on that pipeline has completed
before the signal is issued. The receiver specifies which **local pipeline** should stall until
the signal arrives. This is the fundamental IPC primitive for Cube–Vector cooperation on A5.

> **Note:** `pto.set_cross_core` / `pto.wait_cross_core` operate at **multi-cluster** scope and
> are not used for intra-cluster communication.

#### Intra-Cluster Data Paths

A5 provides dedicated on-chip data paths between the Cube and Vector blocks, bypassing Global
Memory entirely. These are the **recommended high-performance paths** for intra-cluster tile
exchange.

##### C→V: Cube L0C → Vector UB (fixpipe)

The **fixpipe** instruction transfers data directly from Cube's L0C buffer to a Vector block's UB.
Because Cube natively produces results in **NZ fractal layout** and Vector operates on **ND
(row-major) layout**, fixpipe performs the layout conversion in hardware:

```
Cube L0C  (NZ layout)  ──[fixpipe, NZ2ND]──▶  Vector UB  (ND layout)
```

Fixpipe supports a **dual-destination mode**: a single transfer can write to *both* AIV0's UB and
AIV1's UB simultaneously, with the tile split in hardware along either the row axis
(`DualModeSplitM`) or the column axis (`DualModeSplitN`):

| Split | AIV0 receives | AIV1 receives |
|---|---|---|
| Split-M (rows) | Upper `[M/2, N]` in ND | Lower `[M/2, N]` in ND |
| Split-N (cols) | Left `[M, N/2]` in ND | Right `[M, N/2]` in ND |

This 1→2 broadcast with in-hardware tile split is the architectural basis for 1:2
Cube-to-Vector tile distribution.

##### V→C: Vector UB → Cube L1 (TMOV ub2l1)

The reverse path uses `TMOV ub2l1` to transfer data from a Vector block's UB into Cube's L1
buffer. A key architectural constraint: Cube's L1 stores tiles in **NZ fractal layout** (e.g.
`K1M1M0K0` — for fp16: `K0=16`, `M0=16`) so they can be loaded into L0A/L0B for MMAD
computation. Since Vector produces tiles in **ND layout**, the layout conversion from ND to NZ
must be applied as part of the V→C transfer:

```
Vector UB  (ND layout)  ──[TMOV ub2l1, ND→NZ]──▶  Cube L1  (NZ K1M1M0K0)
```

For 1:2 mode, both AIV0 and AIV1 each transfer a sub-tile into Cube's L1. The two sub-tiles are
assembled into a single contiguous NZ Mat tile in L1, ready for use as a LeftTile or RightTile
input to MMAD:

| Split | AIV0 writes to L1 | AIV1 writes to L1 | Assembled in L1 |
|---|---|---|---|
| Split-M (rows) | `[K/2, N]` NZ at base | `[K/2, N]` NZ at offset | Full `[K, N]` NZ Mat tile |
| Split-N (cols) | `[K, N/2]` NZ at base | `[K, N/2]` NZ at offset | Full `[K, N]` NZ Mat tile |

##### Fallback: GM-Staged Transfer

When the local data path is not applicable, data can be exchanged via a **Global Memory staging
buffer**: the producer DMAs data to GM, and the consumer DMAs from GM. This path incurs off-chip
bandwidth cost and higher latency, but serves as a general fallback.

#### Programming Model

The common pattern for Cube–Vector co-programming is a **software pipeline**: the Cube and Vector
programs run a coordinated loop where each iteration the Cube produces a tile and the Vector
consumes it (or vice versa), with explicit `pto.set_intra_block` / `pto.wait_intra_core`
handshakes at each step to maintain correct data ordering.

The PTO micro ISA exposes all the hardware primitives above directly. Higher-level constructs
that simplify this pattern (such as in-order FIFO abstractions) can be implemented as software
libraries on top of these primitives; they are not part of the ISA itself.


### Scope

This document is the interface specification centered on the `mlir::pto` dialect and the shared MLIR surface used alongside it in PTO micro Instruction programs.

It only describes:

- operation names
- operand and result lists
- operand and result types
- important attributes
- C-style semantics for each operation

It does not describe lowering strategy.

PTO micro Instruction source programs are not restricted to `pto` operations alone. In practice they also use shared MLIR dialect ops, most notably the full scalar operation surface of `arith` together with structured control-flow ops from `scf`, to express scalar constants, scalar arithmetic, type conversion, comparisons, and structured control flow around PTO vector or tile regions. These shared-dialect ops are part of the supported PTO micro Instruction source surface and should be regarded as part of PTO-ISA alongside `pto` dialect operations.

### Shared MLIR Dialects

- `arith`: the full scalar `arith` surface is supported in PTO micro Instruction programs, covering scalar integer, floating-point, boolean, and `index` operations. In current samples the most common uses are still constants, offset/bounds arithmetic, casts, compares, and selects.
- `scf`: structured control flow used to model counted loops, conditional regions, loop-carried state, and break-like control around PTO compute and data-movement ops.
- Shared dialect ops remain in standard MLIR form so that PTO analyses and backend passes can reason about control flow and scalar state without re-encoding them as PTO-specific instructions.

### BlockDim Query Operations

These ops expose the current kernel instance's execution coordinates to scalar code. They are the PTO-level equivalent of runtime queries such as `GetBlockIdx()` and `GetBlockNum()` in kernel programming models.

Use them when the same kernel body is launched across multiple blocks or subblocks and each execution instance must figure out which slice of the global workload it owns.

A common pattern is:

- split the full input/output tensor into `block_num` disjoint block-sized regions
- let each block compute its own starting offset from `block_idx`
- within one block, further tile the local region and drive the tile loop with ordinary scalar `arith` / `scf` ops

For example, if a tensor is split evenly across 8 blocks and each block handles `block_length = 2048` elements, then block `b` owns the global range `[b * block_length, (b + 1) * block_length)`. The per-block GM base pointer can be formed by adding `block_idx * block_length` elements to the original base pointer.

At the PTO micro Instruction level, these runtime-query ops are pure scalar producers. They do not perform data movement, do not allocate memory, and do not by themselves create tiling or double buffering. Instead, they provide the scalar values used by surrounding address computation and structured control flow.

#### Example: block-level data partitioning

```mlir
%block = pto.get_block_idx
%block_num = pto.get_block_num
%block_len = arith.constant 2048 : index
%base = arith.index_cast %block : i64 to index
%offset = arith.muli %base, %block_len : index
%block_in = pto.addptr %gm_in, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
%block_out = pto.addptr %gm_out, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

In this pattern, all blocks execute the same kernel body, but each block sees a different `%block` value and therefore computes a different GM window.

#### `pto.get_block_idx`

- **syntax:** `%block = pto.get_block_idx`
- **result:** `i64`
- **semantics:** Return the current block ID in the range `[0, pto.get_block_num())`.

```c
block = block_idx();
```

#### `pto.get_subblock_idx`

- **syntax:** `%subblock = pto.get_subblock_idx`
- **result:** `i64`
- **semantics:** Return the current subblock ID in the range `[0, pto.get_subblock_num())`.

```c
subblock = subblock_idx();
```

#### `pto.get_block_num`

- **syntax:** `%block_num = pto.get_block_num`
- **result:** `i64`
- **semantics:** Return the total number of launched blocks visible to the current kernel instance.

```c
block_num = block_num();
```

#### `pto.get_subblock_num`

- **syntax:** `%subblock_num = pto.get_subblock_num`
- **result:** `i64`
- **semantics:** Return the total number of visible subblocks for the current execution instance.

```c
subblock_num = subblock_num();
```

Typical usage:

```mlir
%block = pto.get_block_idx
%subblock = pto.get_subblock_idx
%block_num = pto.get_block_num
%subblock_num = pto.get_subblock_num
```

### Core Types

### Element Types
`vreg<T>`: `!pto.vreg<NxT>` Fixed-width PTO micro Instruction vector type with total width exactly 256 bytes (2048 bits). `N` is the lane count, `T` is the element type, and `N * bitwidth(T) = 2048`.

| Type | Bits | Description |
|------|------|-------------|
| `i8` / `si8` / `ui8` | 8 | Signless/signed/unsigned 8-bit integer |
| `i16` / `si16` / `ui16` | 16 | Signless/signed/unsigned 16-bit integer |
| `i32` / `si32` / `ui32` | 32 | Signless/signed/unsigned 32-bit integer |
| `i64` / `si64` / `ui64` | 64 | Signless/signed/unsigned 64-bit integer |
| `f16` | 16 | IEEE 754 half precision |
| `bf16` | 16 | Brain floating point |
| `f32` | 32 | IEEE 754 single precision |

### Mask Types

`mask<G>`: `!pto.mask<G>` Typed predicate-register view. `G` is one of `b8`, `b16`, `b32` and records the byte-granularity interpretation used by VPTO ops and verifiers.

Typed masks are also the primary legality contract for predicated VPTO code:

- vector ops over `f32`, `i32`, `si32`, and `ui32` consume `!pto.mask<b32>`
- vector ops over `f16`, `bf16`, `i16`, `si16`, and `ui16` consume
  `!pto.mask<b16>`
- vector ops over 8-bit element families consume `!pto.mask<b8>`
- compare families keep seed-mask and result-mask granularity aligned with the
  compared vector family
- carry families keep carry-in, carry-out, and execution-mask granularity
  aligned with the data-vector family
- mask-only ops that do not explicitly change granularity preserve the same `G`

### Address Space Conventions

PTO micro Instruction memory operands use `!pto.ptr<element-type, space>`. This specification models the following memory-space attributes:

| Space | Interpretation |
|-------|----------------|
| `gm` | Global Memory (GM), off-chip HBM/DDR storage |
| `ub` | Unified Buffer (UB), on-chip vector buffer |

Typical pointer construction and pointer arithmetic follow the same `!pto.ptr<..., space>` form:

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
```

### `!pto.ptr<T, space>`

`!pto.ptr<T, space>` is the typed pointer form used for explicit memory operands in PTO micro Instruction.

- `T` is the element type associated with the pointed-to storage.
- `space` is the memory domain, typically `gm` or `ub` in this specification.
- A `pto.ptr` value carries an address plus its element-type / memory-space interpretation, but it does not carry tensor shape or stride metadata by itself.
- Tensor semantics are introduced separately through view-building operations such as `pto.make_tensor_view`.
- Pointer arithmetic is element-based rather than byte-based.

Typical examples:

- `!pto.ptr<f32, gm>`
- `!pto.ptr<f32, ub>`
- `!pto.ptr<bf16, gm>`

### Tensor View Metadata Query Ops

VPTO source programs may keep GM tensor operands in logical `!pto.tensor_view`
form instead of exposing them as raw memrefs. Two metadata-query ops are used to
read shape and stride information from that logical view:

#### `pto.get_tensor_view_dim`

- **syntax:** `%dim = pto.get_tensor_view_dim %tv, %idx : !pto.tensor_view<...> -> index`
- **semantics:** Returns the runtime extent of dimension `%idx` from the logical tensor view.

```c
dim = tv.shape[idx];
```

Example:

```mlir
%d2 = pto.get_tensor_view_dim %src, %c2 : !pto.tensor_view<?x?x?x?x?xf32> -> index
```

#### `pto.get_tensor_view_stride`

- **syntax:** `%stride = pto.get_tensor_view_stride %tv, %idx : !pto.tensor_view<...> -> index`
- **semantics:** Returns the logical stride of dimension `%idx`, measured in elements rather than bytes.

```c
stride = tv.strides[idx];
```

Example:

```mlir
%s2 = pto.get_tensor_view_stride %src, %c2 : !pto.tensor_view<?x?x?x?x?xf32> -> index
```

Notes:

- These ops are metadata queries only and do not trigger any hardware pipeline activity.
- In authoring-form IR, they operate on `!pto.tensor_view`.
- During compiler-internal lowering, they may be rewritten to equivalent memref metadata queries such as `memref.dim` and extracted strided metadata.

### Pointer Operations

#### `pto.tensor_view_addr`

- **syntax:** `%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> memref<...>`
- **syntax:** `%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> !pto.ptr<T, gm>`
- **semantics:** Extract the underlying address view from a `tensor_view` or `partition_tensor_view`.

```c
result = addr_of(src);
```

`pto.tensor_view_addr` is an address-extraction operation. It does not move data and does not by itself imply any hardware side effect. When the result type is a memref, it exposes the lowered view directly. When the result type is `!pto.ptr<..., gm>`, it exposes the same address in pointer form. After compiler-internal view lowering, the operand may already be a memref; in that case the op is folded away or rewritten to an equivalent memref-to-ptr cast.

#### `pto.castptr`

- **syntax:** `%result = pto.castptr %addr : i64 -> !pto.ptr<T, space>`
- **semantics:** Reinterpret a scalar address value as a typed PTO pointer in the target memory space.

```c
result = (ptr<T, space>)addr;
```

`pto.castptr` is a pointer-construction operation. It does not perform data movement and does not by itself imply any load/store side effect.

#### `pto.addptr`

- **syntax:** `%result = pto.addptr %ptr, %offset : !pto.ptr<T, space> -> !pto.ptr<T, space>`
- **semantics:** Compute a new pointer by advancing the base pointer by an element offset.

```c
result = ptr + offset;  // offset counted in elements, not bytes
```

`pto.addptr` preserves both the element type `T` and the memory-space tag `space`.

#### `pto.load_scalar`

- **syntax:** `%value = pto.load_scalar %ptr[%offset] : !pto.ptr<T, space> -> T`
- **semantics:** Load one scalar element from a pointer-like operand.

```c
value = ptr[offset];
```

- **inputs:**
  `%ptr` is a typed PTO pointer `!pto.ptr<T, space>`, and `%offset` is an
  `index` displacement counted in elements.
- **outputs:**
  `%value` is the loaded scalar element.
- **constraints and limitations:**
  The result type MUST match the element type of `%ptr`. This op is a scalar
  memory helper; unlike `pto.vlds`, it does not produce a `vreg` result and
  does not participate in vector load `dist` families.

#### `pto.store_scalar`

- **syntax:** `pto.store_scalar %value, %ptr[%offset] : !pto.ptr<T, space>, T`
- **semantics:** Store one scalar element to a pointer-like operand.

```c
ptr[offset] = value;
```

- **inputs:**
  `%value` is the scalar value to store. `%ptr` is a typed PTO pointer
  `!pto.ptr<T, space>`, and `%offset` is an `index` displacement counted in
  elements.
- **constraints and limitations:**
  The stored value type MUST match the element type of `%ptr`. This op is a
  scalar memory helper; unlike `pto.vsts`, it does not consume a mask and does
  not target vector-store `dist` families.

#### Pointer-Based Vector Access Example

The following lowered-style fragment shows how typed PTO pointers flow through
pointer construction, pointer arithmetic, structured control flow, and PTO
memory ops. Scalar memory access is expressed on `!pto.ptr<T, space>` in
general, but the common VPTO pattern here is UB-local scalar access alongside
UB vector loads/stores:

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
pto.vecscope {
  %16 = scf.for %arg3 = %c0 to %11 step %c64 iter_args(%arg4 = %12) -> (i32) {
    %mask, %scalar_out = pto.plt_b32 %arg4 : i32 -> !pto.mask<b32>, i32
    %s = pto.load_scalar %1[%c4] : !pto.ptr<f32, ub> -> f32
    pto.store_scalar %s, %1[%c8] : !pto.ptr<f32, ub>, f32
    %17 = pto.vlds %1[%arg3] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %18 = pto.vabs %17, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %18, %10[%arg3], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    scf.yield %scalar_out : i32
  }
}
```

In this pattern, `pto.castptr` materializes a typed UB pointer, `pto.addptr` shifts the base by 1024 `f32` elements, and the subsequent `[%arg3]` indexing on `pto.vlds` / `pto.vsts` applies an additional element offset relative to that base.

### Special Types

#### `!pto.mask<G>`

`!pto.mask<G>` models an A5 predicate register (256-bit) under a typed granularity view, not an integer vector.

`G` is part of the type and MUST be one of:

- `b32`
- `b16`
- `b8`

All three forms describe the same physical 256-bit predicate-register class. The type parameter does not encode how many lanes are currently active. Instead, it records how VPTO interprets the register when matching mask-producing ops, mask-consuming ops, and verifier legality rules.

In the ISA chapters below, this document uses `!pto.mask<G>` as shorthand when a
family is generic over granularity. For op families whose names already encode
the granularity, such as `pset_b32`, `pge_b16`, `plt_b8`,
`pdintlv_b8`, and `pintlv_b16`, examples use the corresponding concrete typed
mask.

**Mask Granularity:**

The predicate register is 256 bits in length, where each bit controls 1 byte of data. `G` therefore describes how many bytes form one logical element slot:

| Mask Type | Bytes / Element Slot | Typical Element Family | Derived Logical Lanes |
|-----------|----------------------|------------------------|-----------------------|
| `!pto.mask<b32>` | 4 | `f32` / `i32` | 64 |
| `!pto.mask<b16>` | 2 | `f16` / `bf16` / `i16` | 128 |
| `!pto.mask<b8>` | 1 | 8-bit element family | 256 |

This is intentionally different from a lane-vector model such as `mask<64xi1>`:

- `!pto.mask<b32>` still denotes a 256-bit predicate register;
- `64` is only the derived logical lane count for the `b32` view;
- value-level patterns such as `PAT_VL32` describe which lanes are active, not a different type.
- `pto.vaddc`, `pto.vsubc`, `pto.vaddcs`, and `pto.vsubcs` use `!pto.mask<G>`
  to carry their per-lane carry results, interpreted with this same
  granularity.

**Predication Behavior (Zero-Merge):**

The native hardware predication mode is **ZEROING** — inactive lanes produce zero:

```c
dst[i] = mask[i] ? op(src0[i], src1[i]) : 0    // ZEROING mode
```

```mlir
// Predicated add: inactive lanes produce zero
%mask = pto.pset_b32 "PAT_VL32" : !pto.mask<b32>   // first 32 logical b32 lanes active
%result = pto.vcmp %a, %b, %mask, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
```

```mlir
// Compare and select: generate mask from comparison, use for conditional select
%mask = pto.vcmp %lhs, %rhs, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
%out = pto.vsel %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

#### `!pto.align`

`!pto.align` models the A5 vector-align carrier state. It is not payload data.

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align

%store_align = pto.init_align : !pto.align
%next_align = pto.vstus %store_align, %offset, %vec, %ub
    : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
```

---

## Part II: Notation Convention

This section defines the MLIR syntax patterns and C-style semantic notation used throughout the ISA reference (Part III).

### MLIR Op Syntax Patterns

All PTO micro Instruction operations follow standard MLIR syntax. The common patterns are:

**Unary (one vector in, one vector out):**

```mlir
%result = pto.<op> %input : !pto.vreg<NxT> -> !pto.vreg<NxT>
```

**Binary (two vectors in, one vector out):**

```mlir
%result = pto.<op> %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

**Vec-Scalar (one vector + one scalar in, one vector out):**

```mlir
%result = pto.<op> %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>
```

**Load (memory to register):**

```mlir
%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

**Store (register to memory):**

```mlir
pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>
```

**Dual Load (one load, two results — deinterleave):**

```mlir
%low, %high = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

**Dual Store (two inputs, one interleaved store):**

```mlir
pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>
```

**Compare (two vectors + seed mask in, mask out):**

```mlir
%mask = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.mask<G>
```

**Conversion (one vector in, different-typed vector out):**

```mlir
%result = pto.vcvt %input, %mask {rnd = "R", sat = "SAT", part = "EVEN"} : !pto.vreg<NxT0>, !pto.mask<G> -> !pto.vreg<MxT1>
```

**Predicate construction:**

```mlir
%mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%tail = pto.pge_b32 "PAT_VL16" : !pto.mask<b32>
```

**Sync operations:**

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

**Pointer construction and arithmetic:**

```mlir
%ptr = pto.castptr %addr : i64 -> !pto.ptr<T, SPACE>
%ptr2 = pto.addptr %ptr, %offset : !pto.ptr<T, SPACE> -> !pto.ptr<T, SPACE>
```

### Shared Dialect Syntax Patterns

PTO micro Instruction programs may interleave PTO ops with standard MLIR `arith` and `scf` ops.
The examples below emphasize common index-heavy patterns, but `arith` support is not limited to index arithmetic.

**Scalar / index constant:**

```mlir
%c0 = arith.constant 0 : index
%zero = arith.constant 0.0 : f32
```

**Scalar arithmetic (integer / float / boolean-style bitwise):**

```mlir
%sum_i = arith.addi %lhs_i, %rhs_i : i32
%sum_f = arith.addf %lhs_f, %rhs_f : f32
%bits = arith.andi %flags0, %flags1 : i32
```

**Scalar compare and select:**

```mlir
%cond = arith.cmpi eq, %lhs, %rhs : index
%bound = arith.select %cond, %a, %b : index
```

**Counted loop with loop-carried values:**

```mlir
%result = scf.for %iv = %lb to %ub step %step
    iter_args(%acc = %init) -> (index) {
  %next = arith.addi %acc, %iv : index
  scf.yield %next : index
}
```

**Structured conditional region:**

```mlir
%selected = scf.if %cond -> (index) {
  scf.yield %then_value : index
} else {
  scf.yield %else_value : index
}
```

**Structured while loop:**

```mlir
%state:2 = scf.while (%iv = %c0, %alive = %true) : (index, i1) -> (index, i1) {
  %keep_going = arith.cmpi slt, %iv, %limit : index
  scf.condition(%keep_going) %iv, %alive : index, i1
} do {
^bb0(%iv_in: index, %alive_in: i1):
  %iv_next = arith.addi %iv_in, %c1 : index
  scf.yield %iv_next, %alive_in : index, i1
}
```

### C-Style Semantics Convention

For each ISA operation in Part III, semantics are expressed as C code. The convention:

```c
// Vector register contents as arrays:
T dst[N];       // destination
T src0[N];      // first source
T src1[N];      // second source (binary ops)
T scalar;       // scalar operand (vec-scalar ops)
int mask[N];    // per-lane predicate (0 or 1)

// N = lane count determined by type:
//   N = 256 for i8/si8/ui8
//   N = 128 for i16/si16/ui16/f16/bf16
//   N = 64  for i32/si32/ui32/f32
//   N = 32  for i64/si64/ui64
```

**Example — pto.vadd semantics:**

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

**Example — pto.vcgadd (group reduction per VLane) semantics:**

```c
int K = N / 8;  // elements per VLane
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        sum += src[g*K + i];
    dst[g*K] = sum;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

For A5 reduction result types:

- `pto.vcadd` widens `i8 -> i16`, `u8 -> u16`, `i16 -> i32`, and `u16 -> u32`,
  with the lane count halved in each widening case.
- `pto.vcadd` keeps the same result type for `f16`, `f32`, `i32`, and `u32`.

### Template Placeholder Conventions

| Placeholder | Meaning |
|-------------|---------|
| `"SRC_PIPE"`, `"DST_PIPE"` | Pipeline identifiers: `"PIPE_MTE2"`, `"PIPE_V"`, `"PIPE_MTE3"` |
| `"EVENT_ID"` | Event identifier: `"EVENT_ID0"` etc. |
| `"DIST"` | Distribution mode string (see the relevant load/store ISA group in Part III) |
| `"CMP_MODE"` | Compare predicate: `eq \| ne \| lt \| le \| gt \| ge` |
| `"RND"` | Rounding mode: `R \| A \| F \| C \| Z \| O` |
| `"SAT"` | Saturation: `SAT \| NOSAT` |
| `"PART"` | Half selector: `EVEN \| ODD` |
| `"PAT_*"` | Predicate pattern literal |
| `T` | Element type (f32, f16, bf16, i32, i16, i8, etc.) |
| `N` | Lane count (`N * bitwidth(T) = 2048`) |

---

## Part III: ISA Instruction Reference
# Part III: ISA Instruction Reference — Summary

This section provides a categorized overview of all PTO micro Instruction operations plus the shared MLIR `arith` and `scf` ops that may appear in PTO micro Instruction programs. Detailed documentation for each group is available in the linked files.

---

## Instruction Groups

| # | Group | Description | Count | Details |
|---|-------|-------------|-------|---------|
| 1 | [Pipeline Sync](isa/01-pipeline-sync.md) | Intra-core pipeline synchronization | 5 | `pto.set_flag`, `pto.wait_flag`, `pto.pipe_barrier`, `pto.get_buf`, `pto.rls_buf` |
| 2 | [DMA Copy Programming](isa/02-dma-copy.md) | Public DMA transfer interface between GM↔UB and UB↔UB | 3 | `pto.dma_load`, `pto.dma_store`, `pto.dma_copy` |
| 3 | [Vector Load/Store](isa/03-vector-load-store.md) | UB↔vreg data movement with various access patterns | ~20 | `pto.vlds`, `pto.vldsx2`, `pto.vgather2`, `pto.vsts`, `pto.vstsx2`, `pto.vscatter`, etc. |
| 4 | [Predicate Load/Store](isa/04-predicate-load-store.md) | UB↔mask register movement | 5 | `pto.plds`, `pto.pldi`, `pto.psts`, `pto.psti`, `pto.pstu` |
| 5 | [Materialization & Predicate Ops](isa/05-materialization-predicate.md) | Scalar broadcast, predicate generation and manipulation | ~17 | `pto.vbr`, `pto.vdup`, `pto.pset_b*`, `pto.pge_b*`, `pto.plt_b*`, `pto.ppack`, `pto.punpack`, `pto.pnot`, `pto.psel`, etc. |
| 6 | [Unary Vector Ops](isa/06-unary-vector-ops.md) | Single-input element-wise operations | 6 | `pto.vabs`, `pto.vexp`, `pto.vln`, `pto.vsqrt`, `pto.vrelu`, `pto.vnot` |
| 7 | [Binary Vector Ops](isa/07-binary-vector-ops.md) | Two-input element-wise operations | 13 | `pto.vadd`, `pto.vsub`, `pto.vmul`, `pto.vdiv`, `pto.vmax`, `pto.vmin`, `pto.vand`, `pto.vor`, `pto.vxor`, `pto.vshl`, `pto.vshr`, `pto.vaddc`, `pto.vsubc` |
| 8 | [Vec-Scalar Ops](isa/08-vec-scalar-ops.md) | Vector-scalar operations | 9 | `pto.vadds`, `pto.vmuls`, `pto.vmaxs`, `pto.vmins`, `pto.vlrelu`, `pto.vshls`, `pto.vshrs`, `pto.vaddcs`, `pto.vsubcs` |
| 9 | [Conversion Ops](isa/09-conversion-ops.md) | Type conversion with rounding/saturation control | 4 | `pto.vcvt`, `pto.vtrc`, `pto.vbitcast`, `pto.pbitcast` |
| 10 | [Reduction Ops](isa/10-reduction-ops.md) | Vector reductions | 7 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin`, `pto.vcgadd`, `pto.vcgmax`, `pto.vcgmin`, `pto.vcpadd` |
| 11 | [Compare & Select](isa/11-compare-select.md) | Comparison and conditional selection | 4 (+1 not A5) | `pto.vcmp`, `pto.vcmps`, `pto.vsel`, `pto.vselr` (`pto.vselrv2` removed: not A5) |
| 12 | [Data Rearrangement](isa/12-data-rearrangement.md) | In-register data movement and permutation | 2 (+2 not A5) | `pto.vintlv`, `pto.vdintlv` (`pto.vintlvv2`, `pto.vdintlvv2` removed: not A5) |
| 13 | [DSA/SFU Ops](isa/13-dsa-sfu-ops.md) | Specialized ops, index generation, and sorting helpers | 9 | `pto.vlrelu`, `pto.vprelu`, `pto.vexpdif`, `pto.vaxpy`, `pto.vmull`, `pto.vmula`, `pto.vci`, `pto.vbitsort`, `pto.vmrgsort4` |
| 14 | [Arith (Shared MLIR Dialect)](isa/14-shared-arith.md) | Full scalar `arith` surface used around PTO ops; the companion page lists categories and representative examples | all scalar ops | `arith.constant`, `arith.addi`, `arith.addf`, `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.index_cast`, `arith.extsi`, `arith.trunci`, `arith.andi`, `arith.shli`, etc. |
| 15 | [SCF (Shared MLIR Dialect)](isa/15-shared-scf.md) | Structured loops, branches, and loop-carried state around PTO regions | 5 | `scf.for`, `scf.if`, `scf.while`, `scf.condition`, `scf.yield` |
| 16 | [Cube Matrix Multiply (MAT)](isa/16-cube-matmul.md) | GM↔L1 cube staging, L0A/L0B loads, L0C matmul, and L0C/L1 side-buffer moves | 10+ | `pto.copy_gm_to_cbuf`, `pto.copy_gm_to_cbuf_multi_nd2nz`, `pto.copy_gm_to_cbuf_multi_dn2nz`, `pto.load_cbuf_to_ca`, `pto.load_cbuf_to_cb`, `pto.mad`, `pto.copy_matrix_cc_to_gm`, `pto.copy_matrix_cc_to_cbuf`, `pto.copy_matrix_cc_to_ub`, `pto.copy_cbuf_to_bt`, `pto.copy_cbuf_to_fbuf` |

---

## Quick Reference by Category

### Memory Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| GM→UB DMA | 2 | `pto.dma_load` |
| UB→GM DMA | 2 | `pto.dma_store` |
| GM→L1 (cube staging) | 16 | `pto.copy_gm_to_cbuf` |
| GM→L1 (multi layout staging) | 16 | `pto.copy_gm_to_cbuf_multi_nd2nz`, `pto.copy_gm_to_cbuf_multi_dn2nz` |
| L1→L0A / L1→L0B | 16 | `pto.load_cbuf_to_ca`, `pto.load_cbuf_to_cb` |
| L0C→GM (cube writeback) | 16 | `pto.copy_matrix_cc_to_gm` |
| L0C→L1 / L0C→UB | 16 | `pto.copy_matrix_cc_to_cbuf`, `pto.copy_matrix_cc_to_ub` |
| L1→BT / L1→FB | 16 | `pto.copy_cbuf_to_bt`, `pto.copy_cbuf_to_fbuf` |
| Contiguous Load | 3 | `pto.vlds` with `NORM` dist |
| Broadcast Load | 3 | `pto.vlds` with `BRC` family dist |
| Gather | 3 | `pto.vgather2`, `pto.vgatherb` |
| Contiguous Store | 3 | `pto.vsts` with `NORM_B8` / `NORM_B16` / `NORM_B32` dist |
| Scatter | 3 | `pto.vscatter` |

### Compute Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| Element-wise Arithmetic | 6, 7 | `pto.vadd`, `pto.vmul`, `pto.vabs`, etc. |
| Scalar Operations | 8 | `pto.vadds`, `pto.vmuls`, etc. |
| Transcendental | 6 | `pto.vexp`, `pto.vln`, `pto.vsqrt`, etc. |
| Reduction | 10 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin` |
| Cube matmul (L0A×L0B→L0C) | 16 | `pto.mad` |
| Comparison | 11 | `pto.vcmp`, `pto.vcmps` |
| Selection | 11 | `pto.vsel`, `pto.vselr` |

### Type & Data Manipulation

| Operation | Group | Description |
|-----------|-------|-------------|
| Type Conversion | 9 | `pto.vcvt`, `pto.vbitcast`, `pto.pbitcast` |
| Interleave/Deinterleave | 12 | `pto.vintlv`, `pto.vdintlv` |
| Interleave/Deinterleave (not A5) | 12 | `pto.vintlvv2`, `pto.vdintlvv2` |

### Synchronization

| Operation | Group | Description |
|-----------|-------|-------------|
| Intra-core Sync | 1 | `pto.set_flag`, `pto.wait_flag` |
| Pipeline Buffer Sync | 1 | `pto.get_buf`, `pto.rls_buf` |

### Scalar & Control Operations

Group 14 covers the full scalar `arith` surface. The rows below list common PTO micro Instruction patterns rather than an exhaustive partition of `arith` ops.

| Operation | Group | Description |
|-----------|-------|-------------|
| Scalar Constants | 14 | `arith.constant` |
| Scalar Integer / Index Arithmetic | 14 | `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.remui`, `arith.ceildivsi`, etc. |
| Scalar Floating-Point Arithmetic | 14 | `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.maximumf`, etc. |
| Scalar Compare & Select | 14 | `arith.cmpi`, `arith.cmpf`, `arith.select` |
| Scalar Casts / Width Changes | 14 | `arith.index_cast`, `arith.index_castui`, `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.sitofp`, etc. |
| Scalar Bitwise / Shift Ops | 14 | `arith.andi`, `arith.ori`, `arith.xori`, `arith.shli`, `arith.shrsi`, `arith.shrui`, etc. |
| Counted Loops | 15 | `scf.for` |
| Conditional Regions | 15 | `scf.if`, `scf.yield` |
| Break-like Structured Loops | 15 | `scf.while`, `scf.condition`, `scf.yield` |

### Recent A5 Additions (Implemented)

- `pto.set_quant_pre` (lowered to `llvm.hivm.SET.QUANT.PRE.v300`)
- `pto.set_atomic_s32`, `pto.set_atomic_s8` (A5-selectable atomic mode controls)
- Cube-side movement additions:
  - `pto.copy_gm_to_cbuf_multi_nd2nz`
  - `pto.copy_gm_to_cbuf_multi_dn2nz`
  - `pto.copy_matrix_cc_to_cbuf`
  - `pto.copy_matrix_cc_to_ub`
  - `pto.copy_cbuf_to_bt`
  - `pto.copy_cbuf_to_fbuf`

### Verified Op List (Current Batch)

- `pto.copy_cbuf_to_bt`
- `pto.copy_cbuf_to_fbuf`
- `pto.copy_gm_to_cbuf_multi_dn2nz`
- `pto.copy_gm_to_cbuf_multi_nd2nz`
- `pto.copy_matrix_cc_to_cbuf`
- `pto.copy_matrix_cc_to_ub`
- `pto.load_cbuf_to_ca_mx`
- `pto.load_cbuf_to_ca_s4`
- `pto.load_cbuf_to_cb_mx`
- `pto.load_cbuf_to_cb_s4`
- `pto.set_atomic_s32`
- `pto.set_atomic_s8`
- `pto.set_channel_para`
- `pto.set_fpc`
- `pto.set_loop1_stride_outtol1`
- `pto.set_loop2_stride_outtol1`
- `pto.set_loop3_para`
- `pto.set_loop_size_outtol1`
- `pto.set_mte2_nz_para`
- `pto.set_pad_val_outtol1`
- `pto.set_quant_pre`

---

## Supported Data Types

| Type | Bits | vreg Lanes | Description |
|------|------|-----------|-------------|
| `i8` / `si8` / `ui8` | 8 | 256 | Signless/signed/unsigned 8-bit integer |
| `i16` / `si16` / `ui16` | 16 | 128 | Signless/signed/unsigned 16-bit integer |
| `f16` | 16 | 128 | IEEE 754 half precision |
| `bf16` | 16 | 128 | Brain floating point |
| `i32` / `si32` / `ui32` | 32 | 64 | Signless/signed/unsigned 32-bit integer |
| `f32` | 32 | 64 | IEEE 754 single precision |
| `i64` / `si64` / `ui64` | 64 | 32 | Signless/signed/unsigned 64-bit integer |

---

## Common Patterns

### Softmax (Numerically Stable)

```mlir
// 1. Find max
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %max_vec, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
%max_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 2. exp(x - max) using fused op
%exp = pto.vexpdif %logits, %max_bc, %mask, "ODD" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// 3. Sum
%sum = pto.vcadd %exp, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %sum, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
%sum_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 4. Divide
%softmax = pto.vdiv %exp, %sum_bc, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

### ReLU Variants

```mlir
// Standard ReLU
%relu = pto.vrelu %input, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Leaky ReLU (scalar alpha)
%lrelu = pto.vlrelu %input, %alpha, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>

// Parametric ReLU (per-element alpha)
%prelu = pto.vprelu %input, %alpha_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

```

### Data Layout Conversion

```mlir
// AoS → SoA (deinterleave)
%x, %y = pto.vldsx2 %ub_xy[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// SoA → AoS (interleave)
pto.vstsx2 %x, %y, %ub_xy[%offset], "INTLV_B32", %all_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>
```

---

*For detailed semantics, C-style pseudocode, and CCE mappings, see the individual group documentation files.*

---

## Appendix: Discussion Points

### Part I

1. **mem_bar as pto op:** Should `pto.mem_bar` be a formal pto dialect op, or is there an existing mechanism?
2. **UB size parameterization:** Is 256KB always fixed, or should spec allow for architecture variants?
3. **MERGING predication:** Intentionally omitted (SW-emulated, perf overhead). Revisit if needed later.

### Part II

1. **Predication in C semantics:** Should every op's C code explicitly show the `if (mask[i])` guard, or assume all-active and note predication separately?
2. **VLane terminology:** Using "VLane" instead of "DataBlock" — confirm this naming is preferred.

### Part 3A

1. **pto.vdupi:** Is this distinct from `pto.vdup` with an immediate operand, or can `pto.vdup` handle both?
2. **Predicate ops (pand/por/pxor and predicate movement forms):** These need MLIR op definitions and verifier rules. Confirm priority.

### Part 3B

1. **Section 10 removals:** 4 interleave ops removed (not on A5). If multi-arch support is needed later, these would need conditional inclusion.

### Part 3C

2. **Store dist family completeness:** `vsts` currently covers `NORM_B8`, `NORM_B16`, `NORM_B32`, `1PT_B8`, `1PT_B16`, `1PT_B32`, `PK_B16`, `PK_B32`, `PK_B64`, `PK4_B32`, `MRG4CHN_B8`, `MRG2CHN_B8`, and `MRG2CHN_B16`, while `vstsx2` covers `INTLV_B8` / `INTLV_B16` / `INTLV_B32`. `MRG4CHN_B8` / `MRG2CHN_B8` / `MRG2CHN_B16` are preserved in the VPTO surface, but the current hardware still reports them as unsupported via verifier warning and they are not expected to validate at runtime on A5 today.
3. **vcvt width-changing pattern:** The even/odd + `vor` pattern for forms such as `f32 -> f16` is the standard compiler lowering. Confirm this is the intended representation in the spec.
4. **Stateful store ops (Section 14):** These are complex with SSA state threading. Are they all needed for A5, or can some be simplified?
