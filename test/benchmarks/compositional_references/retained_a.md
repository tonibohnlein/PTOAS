# Task 1: retained A across output tiles

Read `COMMON.md`; its matching, correctness, profiling and packaging requirements
apply. The goal is to discover whether automatic synchronization preserves one
physical input generation across several consumers while another input turns
over. **Do not substitute ordinary double-buffered GEMM or Shenggan.**

## Exact reference

In the pinned CATLASS archive:

- `examples/25_matmul_full_loadA/matmul_full_loadA.cpp` and its README/CMake.
- `include/catlass/gemm/kernel/matmul_full_loadA.hpp`.
- `include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp`.
- The referenced full-load-A block scheduler, tile-copy and tile-MMA helpers.

The example uses L1 shape 128×256×256 and L0 shape 128×256×64, half A/B/C.
The caller compares the next physical GM A offset with `gmOffsetAPreload` and
passes `needLoadL1=false` when the same A block remains in L1. Preserve that
caller, its scheduling and the full lifetime across blockMmad calls. Merely
calling the block helper once does not exercise the mechanism.

The source admission check is:

`128*K*sizeof(half) + 256*256*2*sizeof(half) <= target L1_SIZE`.

Use admitted K=512 and 1024 after verifying the target limit. Do not use the
ordinary 6144-K GEMM configuration; it does not fit this full-load-A strategy.
Both M<=128 and M>128 scheduler branches matter. Unit flags are enabled by the
default `MmadAtlasA2FullLoadA<true>`; follow the common contract rule.

## Compact case matrix

Let C be the actual Cube core count. Use the original scheduler/core launch.

1. Upstream smoke shape M=256, N=512, K=1024.
2. Retention: M=128, N=256*C*4, K=512 and 1024. Confirm from the actual per-core
   trace that each active core uses the same loaded A through multiple N tiles.
3. Multi-row retention: M=256, N=256*C*4, K=1024. Confirm both retained and
   reloaded generations in the original schedule; adjust N within legal limits
   if necessary to expose the intended transitions and report that choice.
4. Reload control: M=128*C*2, N=256, K=512. Verify actual A addresses change on
   successive owned tiles and that the adapter does not retain stale readiness.
5. One source-supported N tail, for example N=256*C*4-64, with unchanged copy
   geometry/masks. If the original cannot admit it, report that restriction.

Choose the two retention cases and reload control for the main timing table;
the rest are correctness/structure cases unless they reveal a distinct result.
Record A loads per physical generation, B refills, and consumer counts. A case
without measured source-level A reuse is not evidence for retained-A handling.

## Construction and quality questions

Derive matched manual/unsynchronized PTO from the full caller and block body.
Preserve A once loaded, B ping-pong, L0 staging, ACC mode, original output type,
and all copy/layout details. No extra A load may be introduced in any matched
arm. Keep the first consumer's A readiness and B's independent deadlines.

At each remaining blocking acquisition identify:

- Physical A generation and the reader that first needs it.
- Whether A completion persists through later B loads and output-tile entries.
- B's previous same-bank readers and actual refill deadline.
- L1 last-copy versus L0 matrix-reader release; an M wait is not automatically
  the right L1 release boundary.
- The real final reader before an A reload or owning exit.

Add a reload negative: replacing A's contents must restore the new-generation
readiness requirement. Add a multi-reader negative: release after only the first
child must fail when a later child still reads that A generation. Inspect whether
any later B load gates an earlier extraction unnecessarily; assert absence only
where the original payload/manual protocol establishes that independence.

## Success signal

Report whether OAHS reproduces the manual's retained-input overlap, how existing
differs, and the exact source/deadline of any extra wait. A lower pair count is
secondary. Deliver matched timing/profile evidence or a concrete importer
limitation—not another ordinary GEMM timing table.
