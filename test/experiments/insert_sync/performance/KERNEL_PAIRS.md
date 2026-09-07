# Source-derived kernel pairs

These twelve PTO files extend the GEMM campaign with six hand-tuned source
extractions. Both members of each pair have identical scalar operations,
control flow, allocations, views, payload operations and queue operations.
The `.auto.pto` member removes only local `set_flag`, `wait_flag` and `barrier`
operations. It is an input to InsertSync, not a synchronized executable.

All six pairs now compile with follow-up v2 in default report mode. The table
below preserves the frozen `60db1026a` checkpoint used by the checked-in
baselines. See [current counts](MANUAL_VS_FOLLOWUP_V2.md) for the new results.

| Pair | Scope | Frozen 60db InsertSync result |
| --- | --- | --- |
| [topk_128.manual.pto](inputs/topk_128.manual.pto) / [automatic](inputs/topk_128.auto.pto) | Source algorithm specialized to float32, 128 columns, K=128, two slots, two rows per slot | Combined and staged compile; bounded payload traces match manual |
| [conv2d_interior.manual.pto](inputs/conv2d_interior.manual.pto) / [automatic](inputs/conv2d_interior.auto.pto) | One interior 3×3 convolution output tile; real image-to-column operation, L1 release every three L0 uses | Manual/plain C++ emission succeeds; automatic synchronization rejects missing helper contracts |
| [flash_attention_cube.manual.pto](inputs/flash_attention_cube.manual.pto) / [automatic](inputs/flash_attention_cube.auto.pto) | Reduced cube-side QK/P/PV FIFO extraction, with persistent local credits | Manual/plain C++ emission succeeds; automatic synchronization rejects queue effects |
| [triangular_inverse_16.manual.pto](inputs/triangular_inverse_16.manual.pto) / [automatic](inputs/triangular_inverse_16.auto.pto) | Complete float32 16×16 column sweep, recurring load/compute/store credits | Manual/plain emission and payload replay; automatic rejects scalar-dependent TAXPY effects |
| [gdn_wy.manual.pto](inputs/gdn_wy.manual.pto) / [automatic](inputs/gdn_wy.auto.pto) | GDN WY vector and cube peers, 128×128 chunks, two workspaces, half-chunk tail | Manual/plain emission and per-peer replay; automatic rejects FFTS setup effects |
| [kda_wy.manual.pto](inputs/kda_wy.manual.pto) / [automatic](inputs/kda_wy.auto.pto) | KDA WY vector and cube peers, overlapping UB phases, persistent A2, workspace drain | Manual/plain emission and per-peer replay; automatic rejects FFTS setup effects |

At that checkpoint, all except TopK were auxiliary **contract-admission regressions**.
Those rejections do not show that production InsertSync misses synchronization.
FlashAttention is a cube-side extraction, not a full
attention implementation with its vector peer. Conv2D is an interior-tile
extraction, not the complete multicore convolution launcher. No generic GEMM or
elementwise surrogate replaces their missing operations.
The triangular inverse and GDN/KDA source mapping, argument domains, tail
behavior and peer requirements are detailed in [RECURRENCE_PAIRS.md](RECURRENCE_PAIRS.md).

## Source pins and reproducibility

The original four complete relevant upstream files are frozen under `references/`, from
PTO-ISA commit `1216c55831fcc4ed2f096e4ca582ff75a633fbb6` in the local checkout
`/home/toni/work/pypto3/pto-isa-main`. `kernel-pairs-manifest.json` records their
original paths and SHA-256 hashes, both PTO inputs per case, and the Conv2D
support header. The runner validates these identities before compiling and
copies the references/header into each campaign's `support/` directory.
The three added routines and their utility header/test/license are pinned to
`huawei-csl/pto-kernels` commit `e118ec71ed0f170111d4f3380a7a93550c8c22ed`.
Each added reference has its own repository, commit and hash in the manifest;
the original top-level source commit applies to the PTO-ISA references only.

`generate_kernel_pairs.py` is a reviewed transcription, not a general C++→PTO
translator. Its output is deterministic. It preserves the upstream sequence
within the selected extraction, with the explicit adaptations below.

```bash
python3 test/experiments/insert_sync/performance/generate_kernel_pairs.py
python3 test/experiments/insert_sync/performance/generate_recurrence_pairs.py
```

Regeneration does not refresh hashes or baselines automatically. This makes
source edits visible rather than silently accepting a changed reference.

## TopK mapping

Reference: [topk_kernel.cpp](references/topk_kernel.cpp), especially
`InitBuffers`, `SortEachGroup`, `MrgsortSingleRow`, `ExtractDataOrIndex`,
`ProcessSingleRow`, `ProcessIteration` and `runTOPK`.

The specialization uses `validCol=128`, `gWholeShape4=128`, `topk=128`, float32
and `SINGLE_LOOP_ROW=2`. This exercises a complete four-way merge without the
format-two tail merge or executed-count vector. It is not the source launcher's
1024-column/top-1000 configuration. There are two slots; each outer iteration
processes four rows in the original order. `%groups` may be 0 through 16 and
all three data arrays must have backing for 64×128 elements. `%inidx` contains
the 128 column indices. The GM arrays are disjoint.

The frozen manual sequence retains:

1. One index preload and two primed V→MTE2 release credits.
2. One two-row load per slot, followed by an MTE2→V handoff.
3. Two TSORT32 operations and their vector barriers.
4. V→MTE2 publication after the final source read.
5. Per-row TMRGSORT and TMOV stages with the original phase barriers.
6. Separate score/index gathers, separate V→MTE3 publications and two stores.
7. The two final V→MTE2 waits.

Physical backing offsets are obtained from `InitBuffers` with those constants:
sort buffers 0/2048, merge buffers 4096/6144, index 8192, scratch 8704/9728,
scores 10752/11776, indices 12800/13824 and input buffers 14848/15872 bytes.
Row aliases and the source/scratch views' physical versus valid widths are
explicit. The score gather's destination is materialized as its active
128-column view at the same address: PTO's TGATHER verifier requires its
destination valid width to equal its physical width. The original C++ uses a
256-column row descriptor with 128 valid columns at that address. No score or
index address or executed element count changes.

The source's fixed two-row loops and one merge iteration are unrolled; the
outer reuse loop remains. Manual/automatic comparisons retain the same
transcription, including aliases. The source has no explicit output-release
credit or terminal ALL barrier; the manual reference is preserved as written.
Its synchronization is not asserted race-free merely because it is hand-tuned.

This local source uses MTE2/V/MTE3. It does not provide the MTE1 phase mentioned
in an earlier general description of TopK.

## Conv2D mapping

Reference: [conv2d_forward_kernel.cpp](references/conv2d_forward_kernel.cpp),
`Compute`, `ProcessKIteration`, `MacroMatmul`, `StoreResult` and the flag
initialization/drain helpers.

The extraction selects `mIter=3`, `nIter=0`, retaining the source's
`baseM=128`, `baseK=48`, `baseN=256`, `stepKa=stepKb=3`, 3×3 filter,
`hin=16`, `win=96`, C0=16 and N=6144. This is the interior row interval
384–511: `hinStart=3`, `hinCount=4`, `woutStart=0`, pads [1,1,0,0].
Arguments are the original core-relative fmap/weight/output base pointers.
`%panels` is in [1,32]; 32 reproduces all 4608 K elements. Zero panels would
store an uninitialized accumulator and is outside this fixture's domain.

The 3-use L0 body is explicitly expanded inside the retained panel loop. Both
L1 parities are represented, and the odd number of uses reverses the L0 parity
between panels. A shared MTE1→MTE2 release occurs after the third TIMG2COL and
weight extraction, before the last matmul. The accumulator is initialized only
on the first K slice. L1 fmap bases 0/18432, weight bases 131072/204800, L0
bases 0/32768 and accumulator base 0 follow the source's interior calculation.
The enclosing full-kernel M/N loops and launch wrapper are outside this slice.

PTO currently has neither ConvTile descriptors nor TIMG2COL/SETFMATRIX ops.
Those stages are explicit `func.call @benchmark_conv_*` operations, with
actual implementations in [conv2d_helpers.h](conv2d_helpers.h). The header
reconstructs the original ConvTile and NC1HWC0/FRACTAL_Z GlobalTensor types;
the PTO Mat tiles passed to it represent backing addresses, not replacement
image-to-column computations. SETFMATRIX, fmap load, weight load, TIMG2COL,
weight TEXTRACT and the NC1HWC0 store remain distinct calls at their source
positions. Include the header before the emitted Conv2D C++.

No `pto.tileop.effects` or fictitious pipeline summary is attached. The auxiliary
strict coverage check requires payload and descriptor/configuration contracts.
The frozen checkpoint rejects this input; follow-up v2 reports the same gap as
a remark in default report mode and retains the established translation:

```
InsertSync unsupported helper: complete pto.tileop.effects contract required
```

The support header has not been compiled with a device toolchain. An attempted
CPU-header syntax check exposed upstream `_Float16`/`std::exp` ambiguity and a
CPU-only TIMG2COL requirement for column-major L0A, while the A2/A3 source uses
row-major L0A. This does not validate or invalidate the A2/A3 device header;
the retained evidence is PTO parsing and C++ emission only.

## FlashAttention mapping

References: [fa_performance_kernel.cpp](references/fa_performance_kernel.cpp),
`compute_qk`, `compute_pv`, local credit setup/drain, and
[pto_macro_matmul.hpp](references/pto_macro_matmul.hpp).

This is a noncausal, explicit-flag cube-side extraction with dimensions reduced
to 16×16, one subtile, one L0 K segment, no QK preload and one L1 slot for each
of Q/K/P/V. The two shared L0 ping-pong slots remain at 0/32768 and alternate
between QK and PV; the two accumulators have distinct addresses. These are
documented reductions of the vendor kernel, not its full performance settings.

Three distinct GM-only FIFOs remain: QK C2V, P V2C and PV C2V, each depth eight,
with flag bases 0/2/4 and split=1. `%tiles` is in [0,16]; Q is 16×16, K is
viewed transposed with strides [1,16], and V has 256×16 backing. FIFO backing
sizes are respectively 8192, 4096 and 8192 bytes. A vector peer must supply
P and consume QK/PV with the same FIFO protocol; none is synthesized here.
The caller must configure the cross-core synchronization environment.

The sequence preserves QK allocation, Q's first-use-only load, K load, local
operand readiness/release, QK store/push, P pop, V/P loads, P free after its
last load, PV compute, PV allocation/store/push, and local credit drain.
The automatic member **retains all FIFO declarations and
TALLOC/TPUSH/TPOP/TFREE operations**. It removes only the local flag/barrier
plan. Replacing FIFO operations with ordinary local events would change the
program's ownership and progress contracts.

The frozen checkpoint's effect gate rejects these queue effects. Its baseline
retains that result and separately verifies unsynchronized PTO/C++ lowering; it never
counts the rejection as accepted automatic insertion. Dynamic queue execution,
softmax/vector-peer computation and a full attention numerical result are not
measured by this extraction.

The compiler's custom printer omits `local_slot_num` for a GM-only pipe, while
its custom parser demands it. Adding the attribute is invalid without a local
address. For measurement only, `normalize_gm_pipe_assembly` rewrites that exact
printed form into equivalent generic MLIR, preserving every printed attribute
and the one-GM-operand segmentation. It retains both original output and
`*.metrics.pto`, with hashes and the conversion count in `results.json`.
Unknown forms/attributes are not rewritten. A round-trip test checks that the
converted operations print back identically. The source/compiler is unchanged.

## Run the new pairs

Use the same matching Python/native runtime as the GEMM benchmark:

```bash
python3 test/experiments/insert_sync/performance/run.py \
  --python-root /path/to/build/python \
  --manifest test/experiments/insert_sync/performance/kernel-pairs-manifest.json \
  --arch a3 \
  --baseline test/experiments/insert_sync/performance/kernel-pairs-baseline-a3.json \
  --output /path/to/disk-backed/results/kernel-pairs-a3
```

Use the A2 architecture/baseline for A2. The existing five-case manifest and
its baselines remain separate and unchanged. A compiler timeout, parse failure,
unexpected rejection or failed plain lowering is a failure. Only the declared
contract diagnostics qualify as `unsupported-contract`. Baseline comparison
also reports admission changes for review, including newly accepted kernels.

The frozen 60db baseline for each architecture has eighteen case/arm rows:
six manual emissions, two accepted
TopK automatic arms and ten unsupported automatic arms.
The latter retain separate plain-lowering outputs. They are not eighteen accepted
autosynchronized kernels. Fourteen accounting/source tests cover reproducible
generation, frozen source hashes, sync-only pair differences, unchanged FIFO
operations, Conv2D last-use release, assembly round-tripping and the earlier
action/placement/allocation mutation controls. Device correctness, event-lifetime
proofs and timing remain separate work.
The new tests also cover scalar-dependent triangular payloads, unchanged
cross-core publications, empty-stripe participation and the different GDN/KDA
drain counts. For these three cases, rejected automatic arms retain bounded
payload replay of their separately compiled **unsynchronized** input. This
is payload parity evidence, not successful automatic synchronization.

## Frozen host checkpoint

The A2/A3 baselines use compiler source `60db1026a`, native SHA-256
`879c0705280ced29a0dd43617d3fab6e7b5b0d0419051921ef7cabc8bf2c3ab3`.
Both architectures have the same TopK counts at this checkpoint:

| TopK measurement | Manual | Combined / staged |
| --- | ---: | ---: |
| Static set / wait | 10 / 10 | 15 / 15 |
| Static barriers | 22 V | 14 V + 2 MTE3 + 1 ALL |
| Set / wait, 16 groups | 130 / 130 | 180 / 180 |
| Barriers, 16 groups | 352 V | 224 V + 32 MTE3 + 1 ALL |

The ALL barrier is outside the loop. Payload traces match for 0, 1, 2, 3 and
16 groups. These counts expose changes worth investigating; they do not rank
device performance or prove the manual synchronization sufficient.

Full local evidence for all six cases is under the workspace's
`insertsync-builds/campaign/kernel-pairs-six-{a2,a3}-verified/` and fresh
baseline reruns under `kernel-pairs-six-{a2,a3}-final-v2-baseline-check/`. Each directory
contains `results.json`, commands, diagnostics, emitted PTO/C++, input copies
and frozen supporting sources. Earlier probe/round directories are development
artifacts, not this checkpoint.
