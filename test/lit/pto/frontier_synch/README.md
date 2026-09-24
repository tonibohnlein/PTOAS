# FrontierSynch kernel corpus

Four kernel families, expressed as five standalone PTO inputs for the existing
Phase A query boundary. These are original payload programs: allocations,
engine assignments, scalar control, reads and writes. They contain no active
synchronization. Do not execute the payload files directly on a device.

The manual plans are comparison references. The analysis must derive storage
relationships from the program, without recognizing a kernel name, a particular
loop spelling, or a reference flag sequence. No algorithm implementation changes
are part of this corpus.

## Inputs and provenance

| Input | Specialization | Properties retained |
| --- | --- | --- |
| [tilelang_gemm.pto](tilelang_gemm.pto) | FP16 A[256,4096], B[4096,256], C[256,256]; FP32 accumulation | Two output tasks; L1 prefetch prologue; guarded lookahead; two L1 and two L0 banks; four L0 extracts per L1 panel; last-read L1 release before the final MMA |
| [pypto_gemm.pto](pypto_gemm.pto) | Same A/B geometry, FP32 C | Carried Mat/L0 slot counters reset at their respective loop entries; physically persistent banks; release after the inner loop; accumulator reuse across output tasks |
| [expert_fa_cube.pto](expert_fa_cube.pto) | FP16 Q[256,128], K/V[4096,128]; two query tasks; FP32 MMA | Resident Q; single K/P/V L1 buffers; shared double-buffered L0A/B/C across two GEMM phases; Q read only on the first two visits of each QK batch |
| [expert_fa_vector.pto](expert_fa_vector.pto) | One of two vector lanes, 64 rows per lane; FP16 output[256,128] | Full online softmax and output accumulation; early release of `io`; separate store-buffer reuse; alternating maxima; per-stage factors and sums |
| [vector_add.pto](vector_add.pto) | FP32 A/B/C[128,128], one of two vector lanes; four 16x128 chunks | Prefetch prologue, guarded next load, three separate two-slot UB pools, MTE2/V/MTE3 handoffs and drain |

`lane` must be 0 or 1. All tensors are row-major and disjoint unless a shared
workspace is explicitly described below. The fixed extents are part of the
input contract; these files do not promise arbitrary shapes or empty reductions.

Sources are pinned to avoid silently changing the comparison:

- TileLang Ascend commit `1986dd76569f321c821bb8ce85d5b5b397010d71`:
  [intrinsic GEMM](https://github.com/tile-ai/tilelang-ascend/blob/1986dd76569f321c821bb8ce85d5b5b397010d71/examples/gemm/example_gemm_intrinsic.py),
  [expert Flash Attention](https://github.com/tile-ai/tilelang-ascend/blob/1986dd76569f321c821bb8ce85d5b5b397010d71/examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py),
  [pipelined vector add](https://github.com/tile-ai/tilelang-ascend/blob/1986dd76569f321c821bb8ce85d5b5b397010d71/examples/elementwise/elementwise_add_pipeline.py).
- Shenggan/pypto-gemm commit `4b7fe5558996f8468f0c55189fc44092222c5085`:
  [manual pipeline](https://github.com/Shenggan/pypto-gemm/blob/4b7fe5558996f8468f0c55189fc44092222c5085/hpgemm_step4_manual_pipeline.py).

The [Flash Attention benchmark](https://github.com/tile-ai/tilelang-ascend/blob/1986dd76569f321c821bb8ce85d5b5b397010d71/examples/flash_attention/fa_opt/bench_mark.md)
reports roughly 80% of native AscendC throughput for the expert implementation.
The [PyPTO benchmark](https://github.com/Shenggan/pypto-gemm/blob/4b7fe5558996f8468f0c55189fc44092222c5085/README.md)
reports 267.94 TFLOPS for the manual pipeline and 226.06 for the automatic
swizzled version at M=K=N=4096. Those measurements motivate the source selection;
they are not measurements of these PTO ports or an isolated InsertSync comparison.

## Deliberate porting differences

- Core assignment, swizzle, head/batch indexing and the Python launchers are
  specialized away. Two output/query tasks retain local-storage reuse across
  task boundaries. Tile sizes and reduction geometry remain substantive.
- Local byte addresses are explicit. Repeated `alloc_tile` declarations at the
  same address are descriptors of the same physical storage, not new runtime
  allocations. Scalar calculations and manual event operands do not determine
  aliasing by themselves.
- Both GEMMs use a conditional `tmatmul`/`tmatmul.acc` to express the source MMA
  initialization predicate. TileLang GEMM's reference event numbering is
  normalized so its FIX-to-M return event uses ID 1.
- Flash Attention keeps 14 stages, synchronization every two stages, and 32 KV
  blocks: the batch sizes are 14, 14 and 4. Workspace arrays are flattened to
  [1792,128]. QK and PV accumulation are converted to FP16 by `tstore`, as in the
  source workspace interface.
- The QK transpose is expressed through a strided GM view of K. The source puts
  this transpose on the L1-to-L0 copy; the PTO port loads the transposed logical
  tile into L1. This preserves the participating K elements and engine chain,
  but is a lowering difference to account for before comparing performance.
- Flash Attention row statistics use 64x8 FP32 storage with one valid column,
  instead of compact 64x1 storage, to satisfy the A3 vector layout contract.
  Reductions have an explicit scratch buffer. `LOWERING` annotations add
  conservative vector barriers between PTO vector instructions; the TileLang
  source relies on its backend for those dependencies. These barriers are not
  claimed to reproduce TileLang's optimized instruction schedule.

## Flash Attention execution boundary

The cube and vector functions describe the cooperating bodies for one cube core
and its two vector lanes. They share three separate FP16 workspaces:

| Workspace | Producer | Consumer | Forward / return cross-event IDs |
| --- | --- | --- | --- |
| ws1: QK scores | cube FIX | vector MTE2 | 0 / 1 |
| ws2: softmax weights | vector MTE3 | cube MTE2 | 2 / 3 |
| ws3: partial output | cube FIX | vector MTE2 | 4 / 5 |

Each workspace has 14 slots of 128x128 elements. Each vector lane accesses its
own 64-row half. Forward notifications occur every two slots (or at batch end);
return notifications release a complete batch. Both task bodies use the same
task and batch order. The manual annotations express the A3 FFTS block-mode
protocol and require an appropriate `ffts` base address and paired launch.

Phase A runs independently on each function. It does **not** prove this
cross-core protocol, associate an incoming workspace load with its remote
producer, or validate the launch configuration. There is no host launcher or
device-validation claim in this initial corpus.

A future automatic local-sync device comparison must preserve the external
cross-core protocol and confirm that its presence does not bypass local
insertion. Running either synchronization-free body alone is not a Flash
Attention execution test.

## Physical memory maps

Offsets and strides below are bytes within the named memory space. Distinct
memory spaces may use the same numeric offset.

| Family | Space | Allocation: base, slot stride, slots |
| --- | --- | --- |
| Both GEMMs | Mat | A: 0, 65536, 2; B: 131072, 131072, 2 |
| Both GEMMs | Left / Right / Acc | A: 0, 16384, 2 / B: 0, 32768, 2 / C: 0, 131072, 1 |
| FA cube | Mat | Q/K/P/V: 0/32768/65536/98304, each 32768 bytes |
| FA cube | Left / Right / Acc | A/B: 0, 32768, 2; C: 0, 65536, 2 |
| Vector add | Vec | A/B/C: 0/16384/32768, stride 8192, 2 slots each |
| FA vector | Vec | acc_o: 0..32768; io: 32768..49152; s_half: 49152..65536 |
| FA vector | Vec | work: 65536..98304; broadcast: 98304..131072; scratch: 131072..163840 |
| FA vector | Vec | sumexp: 163840..165888; maxima: 165888, stride 2048, 2 slots |
| FA vector | Vec | factors: 169984, stride 2048, 14 slots; sums: 198656, stride 2048, 14 slots |

Ranges are half-open. The vector-add reference waits for MTE3 before reloading
the corresponding input slot even though input and output pools are disjoint.
It is a useful conservative reference, not an assertion of optimal placement.

## Run Phase A now

In a configured build of this checkout:

```bash
cmake --build build --target pto-frontier-analysis-test --parallel 2
build/tools/pto-test-opt/pto-frontier-analysis-test \
  test/lit/pto/frontier_synch/pypto_gemm.pto
lit -j 1 -v build/test/lit --filter 'frontier_synch/'
```

The test executable uses `SyncInput`, `importOriginalStructure` and
`ProgramAnalysis` from the current sources. It interprets **every** indexed
storage requirement, checks its source subscription, and verifies that the
original IR remains unchanged. It reports RAW/WAR/WAW counts, physical bank
relations and unknown occurrence results. It neither
allocates events nor invokes the unfinished constructor.

`complete()` here is preparation completeness. Unknown correspondence, missing
full-write proofs remain unresolved; a successful process
exit does not certify a complete implementation of draft 0.42.

## Materialize the manual reference

```bash
python3 test/lit/pto/frontier_synch/materialize.py \
  test/lit/pto/frontier_synch/pypto_gemm.pto > manual.pto
build/tools/pto-test-opt/pto-frontier-analysis-test --verify-only manual.pto
```

The materializer only activates `HAND`, `CROSS` and `LOWERING` annotations. All
payload operations, allocations and control flow are shared with the input.
The annotations are executable reference data, not disabled implementation.
Each lit test verifies both the payload and the materialized reference, including
the presence of actual local/cross synchronization operations in the latter.
Parsing and verification do not prove event balance, race freedom or numerics.

With a working full compiler, the intended manual compile entry is:

```bash
ptoas --pto-arch=a3 --pto-level=level3 manual.pto -o manual.cpp
```

For a future same-payload comparison, run `pto-insert-sync{algorithm=existing}`
or `pto-insert-sync{algorithm=frontier-synch}` on the **payload** file using the
normal pass pipeline. The latter still intentionally fails at construction.
Do not feed the materialized manual file to automatic insertion: manual flags
can cause the existing pass to skip the function.

## Current validation checkpoint

On 2026-09-24, all five corpus lit tests and the TAXPY regression passed using a
freshly compiled focused runner and the existing LLVM 19 toolchain. The runner
compiled the current Phase A, shared translator and PTO dialect implementation,
with operation/interface headers regenerated from this checkout and the remaining
available dialect dependencies. This follows the handoff's existing probe approach. The checkout's
normal full compiler build remains unavailable; the CMake target is registered
for a normally configured checkout. Local artifacts are under
`.local/frontier-corpus/`, including the runner, build script/logs and lit output.
The already-built local runner can be used immediately from this checkout:

```bash
.local/frontier-corpus/pto-frontier-analysis-test \
  test/lit/pto/frontier_synch/expert_fa_vector.pto
```

| Input | Translated phases | Physical bank relations | Interpreted requirements | Unknown occurrence |
| --- | ---: | ---: | ---: | ---: |
| TileLang GEMM | 9 | 4 | 77 | 77 |
| PyPTO GEMM | 7 | 4 | 35 | 35 |
| FA cube | 12 | 6 | 96 | 96 |
| FA vector | 28 | 2 | 1976 | 1976 |
| Vector add | 6 | 3 | 58 | 58 |

Counts are this checkpoint's observations, not desired synchronization counts.
Both algorithms consume the same instruction effects through `SyncInput`;
FrontierSynch performs no additional effect audit or instruction whitelist check.
TAXPY's shared effect definition includes reading and writing its destination,
matching `dst += src * scalar`. The focused
[`frontier_synch_taxpy_effects.pto`](../frontier_synch_taxpy_effects.pto) regression
checks dependencies on both loaded inputs and the destination write.

No full EmitC/VPTO compilation, simulator/NPU execution, numerical comparison,
event-protocol validation or performance measurement has been performed for
these ports. Those are the next gates before using them as performance baselines.
