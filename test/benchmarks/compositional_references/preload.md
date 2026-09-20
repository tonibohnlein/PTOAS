# Task 2: prefetch across output-tile boundaries

Read `COMMON.md`. The goal is to determine whether synchronization preserves the
reference's **next-output-block prefetch while current-block work is still in
flight**. This is distinct from buffering K panels within one output tile.

## Exact reference

In the pinned CATLASS archive:

- `examples/06_optimized_matmul/optimized_matmul.cpp`, README and CMake.
- `include/catlass/gemm/kernel/optimized_matmul.hpp` and its actual selected
  helpers; inspect the dispatch rather than assume a block helper is the kernel.
- `include/catlass/gemm/block/block_mmad_preload.hpp`.
- `include/catlass/gemm/kernel/basic_matmul_preload.hpp` as a smaller caller
  reference if useful; label a reduced caller separately from example 06.

The inspected default is half A/B/C, row-major A, column-major B, with
`ENABLE_UNIT_FLAG=true`, `ENABLE_SHUFFLE_K=true`, L1 128×256×256 and L0 128×256×64.
The optimized wrapper can include packing/padding kernels. Preserve them or
measure a separately identified already-packed compute scope identically for
all arms. Preserve shuffled K's initialization/accumulation order too.

In the block helper, `shuffleKIdx == lastTileIdx && hasNextBlock` loads the next
output block's initial panels. `isFirstBlock` controls startup. Carry their real
caller state, buffer indices and priming/draining across block invocations.
Do not reset the protocol at every lexical output-tile boundary.

## Compact case matrix

Let C be the actual Cube core count. Start from a source-supported layout and
the original tiling/scheduler. Suggested logical shapes, subject to its checks:

1. A one-output-block-per-active-core control, with no cross-block preload.
2. M=128*C, N=1024, K=512: several output blocks per core, short K.
3. M=128*C, N=2048, K=4096: several output blocks per core, long K.
4. One uneven per-core block-count shape, e.g. M=128*(C+1), N=768, K=512,
   to exercise the final `hasNextBlock=false` transition.
5. One source-supported partial output tile to test final copy masks/padding.

Verify actual per-core ownership rather than assuming M/N imply it. If a
suggested shape does not exercise the required branch with the source scheduler,
choose the nearest legal shape that does and record the reason. Keep the shape
matrix small; prioritize two multi-block cases plus the no-preload control.

## Matched-payload requirements

All three matched arms must contain the same explicit next-block loads in the
same original control positions. An autosync pass cannot create payload prefetch
that the input lacks. Moving loads, changing K order or switching to a different
tile scheduler is a separate optimization, not an autosync result.

Distinguish these intervals in the source trace and timeline:

- Last reads of a current L1 bank and its next refill.
- Next output block's initial A/B loads.
- Current block's final matrix operations and FIX store.
- Next block's first L1→L0 consumers and ACC initialization.

Attribute any wait/drain inserted before a next-block load to the exact storage
reuse or event-republication requirement. Test whether required consumption
knowledge can survive the block boundary without a whole-pipeline drain.
Retain separate A/B readiness; moving a publication later to reduce keys is not
automatically equivalent to the source protocol.

## Discriminating checks and deliverable

On a legal multi-block path, inserting a drain before prefetch must measurably
broaden the command graph or block the intended timeline overlap. Removing a
true L1 same-bank return must fail the relevant memory or rearming obligation.
Include first block, interior block and last block paths, plus uneven per-core
tails. Do not impose a forbidden-order assertion where the source's own shared
storage makes that order necessary.

Measure original versus matched manual first, then OAHS versus matched manual
and existing. Report compute-kernel and wrapper/padding/prefetch costs separately.
Show the actual next-block DMA/current-block compute or store overlap and identify
the additional blocking endpoint if an automatic plan loses it.
