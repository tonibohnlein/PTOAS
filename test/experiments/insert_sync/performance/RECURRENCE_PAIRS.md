# Triangular inverse, GDN and KDA PTO pairs

These are the three previously missing hand-tuned kernel families. The
triangular fixture implements the selected column-sweep routine. The GDN and
KDA fixtures each implement the selected WY stage, including its vector
producers and cube consumer. They are not full multistage GDN/KDA networks or
megakernels. The automatic member removes only local flags/barriers; scalar
accesses, branches, workspace addresses, FFTS setup and cross-core signals
remain identical.

Follow-up v2 compiles all three automatic pairs in default report mode.
Admission diagnostics below describe the frozen `60db1026a` baselines and
remain available through strict coverage mode. They do not establish a defect
in production InsertSync. See [current counts](MANUAL_VS_FOLLOWUP_V2.md).

| Family | Manual | Automatic |
| --- | --- | --- |
| Triangular inverse | [triangular_inverse_16.manual.pto](inputs/triangular_inverse_16.manual.pto) | [triangular_inverse_16.auto.pto](inputs/triangular_inverse_16.auto.pto) |
| GDN WY | [gdn_wy.manual.pto](inputs/gdn_wy.manual.pto) | [gdn_wy.auto.pto](inputs/gdn_wy.auto.pto) |
| KDA WY | [kda_wy.manual.pto](inputs/kda_wy.manual.pto) | [kda_wy.auto.pto](inputs/kda_wy.auto.pto) |

## Sources and regeneration

All three are transcribed from `huawei-csl/pto-kernels` commit
`e118ec71ed0f170111d4f3380a7a93550c8c22ed`. The WY implementations in that
repository are ports of the MegaGDN routines identified in the earlier survey.
Frozen complete files, their utility header and the Clear BSD license are in
`references/`. Each is independently hashed in `kernel-pairs-manifest.json`.

- [Triangular source](https://github.com/huawei-csl/pto-kernels/blob/e118ec71ed0f170111d4f3380a7a93550c8c22ed/csrc/kernel/kernel_tri_inv_col_sweep.cpp): `runTTriInv<float,16>`.
- [GDN source](https://github.com/huawei-csl/pto-kernels/blob/e118ec71ed0f170111d4f3380a7a93550c8c22ed/csrc/kernel/kernel_gdn_wy_fast.cpp): fixed-length `wy_fast_kernel` body and `gemm_v0`.
- [KDA source](https://github.com/huawei-csl/pto-kernels/blob/e118ec71ed0f170111d4f3380a7a93550c8c22ed/csrc/kernel/kernel_kda_wy.cpp): fixed-length `kda_wy_kernel` body and `gemm_v0`.

```bash
python3 test/experiments/insert_sync/performance/generate_recurrence_pairs.py
```

The generator does not refresh manifest hashes or baselines. Input hashes,
reproducible generation and exact sync-only differences are checked by the
accounting tests. Generation needs no upstream checkout or network access.

## Triangular column sweep

`triangular_inverse_16` receives disjoint `src`/`dst` pointers and `%matrices`
in [0,16]. Each is backed by sixteen 16×16 float32 matrices. The core's
contiguous work range is supplied directly; multicore work partitioning and
the host launch wrapper are outside the fixture. The payload works on the
source routine's raw matrix layout (unit lower-triangular matrices in raw
row-major storage). The frozen upstream Python test documents its Torch
transpose/view convention. The fixture adds no transpose or diagonal division.

The complete 16-column sweep remains: clear output, load the matrix, create
each basis vector, use TAXPY with tile-derived scalar coefficients, write the
inverse through scalar tile accesses, and store the result. The two descending
loops are unrolled at the fixed size 16. The outer matrix reuse loop remains.
PTO `tgetval`/`tsetval` represent the actual `GetValue`/`SetValue` calls; none is
replaced by a dummy helper. The input, basis and output UB bases are 0, 1024
and 1088 bytes. Row aliases preserve the 64-byte pitch.

The manual source primes MTE3→MTE2 and MTE3→V once, returns both credits after
each store, and drains both at exit, including zero work. Its V↔S handoffs and
105 vector barriers per matrix are retained. At size 16 each matrix executes
120 TAXPYs and 120 scalar tile reads. Set and wait counts are each
`2 + 276 * matrices`. This records the upstream synchronization; it does not
certify its sufficiency.

The frozen 60db coverage gate rejects the first TAXPY whose scalar coefficient depends
on a preceding tile read through `arith.negf`, reporting:

```
InsertSync unsupported effect: resource or memory operand is not modeled
```

Manual and plain unsynchronized PTO/C++ emission succeed. Symbolic payload
replay matches for 0, 1, 2, 3 and 16 matrices. Scalar tile values are hashed
symbolically, not numerically evaluated; no numerical or asynchronous
correctness claim follows from the trace match.

## GDN and KDA WY stages

Both specialize the source to C=D=128 and one value/key head (H=Hg=1), with
one already assigned per-core contiguous sequence. These are source-default
tile dimensions; the multicore/head enumeration is replaced by `%chunks` in
[0,16]. Each input/output matrix array has 2048×128 backing. Beta is stored as
a contiguous 1×2048 vector. GDN's gate is float32 1×2048; KDA's gate is float32
2048×128. A, K, V, beta, U, W and both workspaces use float16. Every pointer
argument denotes disjoint GM storage; the repeated reuse of each workspace
through that same pointer is intentional.

`%tail_half=false` means all chunks have 128 valid rows. With `%tail_half=true`,
the last nonempty chunk has 64 valid rows; earlier chunks remain full. Input
loads, zero padding and output stores follow that domain exactly. The vector
function's `%stripe` is 0 or 1, corresponding to the two 64-row subblocks.
The lower stripe of a half tail performs no A load; KDA also skips its K/G
loads. Both stripes still produce and signal both full workspace stripes.
Other partial-tail sizes, multiple sequences and grouped heads are outside
this specialization.

Each file contains two separate functions, suffixed `_vector` and `_cube`.
A device wrapper must run the cube function and both vector stripes as
cooperating peers with the same `%chunks`, `%tail_half`, FFTS configuration and
workspace pointers. Each workspace has 128×128 float16 backing (32768 bytes)
per cube core. The wrapper and a device launch are not supplied by this host
regression. The benchmark replays each peer independently; it does not
serialize them into a supposed device execution.

### GDN mapping

The vector stage computes `A2 = A * beta` and `A1 = A * (exp(g) * beta)`, with
column broadcasts. The cube computes `U = A2 @ V` and `W = A1 @ K`.

All live UB offsets match the source: beta half 0, A1 half 256, beta float
16640, beta row 17152, beta broadcast 17664, A1 float 75008, A2 float 107776,
A2 half 140544, gate float 156928, gate row 157440 and gate broadcast 157952.
The unused scratch declaration is omitted. L1 K/V/A2/A1 bases are
0/32768/65536/98304; L0 A/B each use base 0; U/W accumulators use 0/65536.

The single MTE2→V publication after the beta/A loads is preserved, along with
the original vector phase barriers and two PIPE_ALL barriers before workspace
publication. Ready flags are A2=2 and A1=1; free flags are A2=3 and A1=4.
Each vector peer skips the free waits on its first work item. The source has
**no final free-credit drain**: for N>0 each vector waits N−1 times per free
flag while the cube publishes N times. This observed trailing credit is
preserved and tested; the fixture does not silently repair the manual plan.

### KDA mapping

The vector stage computes `A2 = INV * beta` and `Keff = K * exp(g_cs)`. The
cube computes `U = A2 @ V`, then `W = A2 @ Keff` while keeping A2 resident in
L1 across both GEMMs.

The two vector phases share backing bytes exactly as in the source. Phase 1
uses beta half 0, beta float 512, beta broadcast 1024, inverse float 33792,
A2 float 66560 and A2 half 99328. Phase 2 uses K float 0, gate float 32768,
Keff float 65536 and Keff half 98304. The half A2 and Keff destinations also
serve as their phase's input staging buffers. L1 V/A2/Keff bases are
0/32768/65536; L0 A/B each use base 0; U/W accumulators use 0/65536.

Ready flags are A2=10 and Keff=11; free flags are A2=12 and Keff=13. In-loop
free waits skip the first work item, and the guarded trailing drain consumes
both final free credits. Zero-work peers perform neither publications nor
drains. An empty lower tail stripe still stores zeros and publishes both
ready flags. Tests check these counts independently of local synchronization.

### Cross-core preservation and frozen admission

`SetCrossFlag` uses `1 | (2 << 4) | (flag << 8)` in the pinned utility header.
The PTO representation retains `sync.set`/`sync.wait` with explicit FFTS mode
2. The local flag/barrier stripper leaves both intact. The pinned
`pto-kernels` routines assume FFTS base setup by the caller. PTOAS requires
`pto.set_ffts` in the same function before it will emit these cross-core ops,
so each generated peer takes `%ffts` and performs that setup explicitly. This
is a documented adaptation in both pair members; the original MegaGDN
variants also perform setup inside the routine.
A2/A3 are the target architectures; the source's A5 intra-block variants are
not transcribed here.

The frozen checkpoint's first rejection is at the required `pto.set_ffts`, with
`InsertSync unsupported effect: resource or memory operand is not modeled`.
A diagnostic probe omitting setup reaches the recurring `pto.sync.wait` and
reports `InsertSync unsupported effect: missing semantic summary`, but that
probe cannot emit C++ because same-function setup is required. It is not an
alternative runnable fixture. Neither rejection proves that the rest of the
cross-core protocol is modeled. Both manual and plain unsynchronized lowering emit PTO/C++.

The manifest contains 30 scenarios per WY pair: cube and both vector stripes,
0/1/2/3/16 chunks, with full and half-tail forms. Manual/plain payload hashes
match for each. Fixed protocol counts and placement hashes are retained
separately from local set/wait/barrier metrics. No cross-core completion,
quorum, progress proof, device numerical result or runtime is measured.

## Regression command

Use the six-case manifest and the matching A2/A3 baseline:

```bash
python3 test/experiments/insert_sync/performance/run.py \
  --python-root /path/to/build/python \
  --manifest test/experiments/insert_sync/performance/kernel-pairs-manifest.json \
  --arch a3 \
  --baseline test/experiments/insert_sync/performance/kernel-pairs-baseline-a3.json \
  --output /path/to/disk-backed/results/six-kernel-pairs-a3
```

The earlier five-case pipeline/GEMM manifest and its baselines remain separate.
An unexpected compiler error, failed plain lowering, payload mismatch or
changed baseline fails the campaign. Declared contract rejections remain
visible as `unsupported-contract`, never as accepted automatic kernels.
