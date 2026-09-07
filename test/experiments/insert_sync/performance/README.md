# InsertSync performance regression fixtures

Source-derived **TopK, Conv2D, FlashAttention, triangular inverse, GDN and KDA PTO pairs** are available
in [KERNEL_PAIRS.md](KERNEL_PAIRS.md), using `kernel-pairs-manifest.json`.
All six compile automatically with follow-up v2's default report mode. The
frozen `60db1026a` baselines record five auxiliary coverage-gate rejections.
GDN/KDA include both WY vector and cube peers;
see [RECURRENCE_PAIRS.md](RECURRENCE_PAIRS.md). The original
five-case manifest and measurements below remain frozen separately.

The historical [original-versus-60db InsertSync counts](REVISION_COUNTS.md)
cover all seven kernel families and four controls on A2/A3, with static and
executed action counts and explicit rejection results.

The follow-up v2 compiler now admits every pair in report mode. Its fresh
[hand-tuned/original/follow-up comparison](MANUAL_VS_FOLLOWUP_V2.md) records
static counts and 102 replay scenarios per arm on A2/A3. It restores
compatibility; GEMM and TopK counts do not improve over the original pass.

This is the first runnable slice of the earlier benchmark ladder: the exact
historical full GEMM manual/automatic pair, plus four small synthetic pipeline
controls. It measures compiler admission, synchronization placement, directed
event-ID footprint, static actions and concrete dynamic action counts. It does
not measure device time or prove asynchronous memory/event correctness.

## Recovered instance

The authoritative pair was found in:

```
/home/toni/work/pypto3_sync/canonical-sync-step4-experiment/historical-artifacts/
  step3_swizzle.pto
  step4_manual_pipelining.pto
```

Both files are copied byte-for-byte into `inputs/`. Their SHA-256 values match
the earlier `DEVICE_TASK_CANONICAL_SYNC_GEMM_OWNERSHIP_PERFORMANCE.md` pins:

| Input | SHA-256 |
| --- | --- |
| step 3, automatic | `3db5d353475ee920857d16e5c3533ab23e0c6d2c38e22b57235c0e85a9a6a380` |
| step 4, manual | `ccd3abfb22eb6f81d87f6bdb72dfba655a456b54512f2b6def9557766922dae1` |

Provenance: `huawei-csl/pto-dsl`, commit
`36cd417e230a84a80870f564de27539ccc7c4a75`, directory
`examples/aot/matmul_optimization_guide`. The nearby `pto-dsl-historical`
checkout has moved since the original campaign; its present HEAD is not the
source pin. The pinned commit remains available in its Git object database.

This GEMM exercises nested L1 panel reuse, L0A/L0B ping-pong and a stationary
L0C accumulator together. The original records identify nine allocations and
82 non-sync PTO operations, including seven loads, 32 extracts, two initial
matmuls, 16 accumulating matmuls and one store. Counts are static across both
branches, not executed counts. Step 4 also adds scalar predicates and sync-only
branches, so text equality after deleting sync lines is an invalid parity test.

## Frozen population

`manifest.json` fixes input hashes, compilation levels, GM contract and replay
scenarios. Automatic arms receive the same bytes. Manual compilation omits
`--enable-insert-sync`; the runner checks actual analysis/allocation markers
and explicit-sync bypasses in the automatic arms.

| Case | Role | Concrete replay |
| --- | --- | --- |
| `one_buffer` | Basic MTE2 → V → MTE3 ready/free cycle | Zero, one and repeated trips |
| `two_buffer` | Two independent physical slots | Both slots and distance-two reuse |
| `three_buffer` | Synthetic depth-three diagnostic | Partial fill, wrap and repeated reuse |
| `four_use` | Two slots, four consumers per load, release after last consumer | Zero, one and repeated groups |
| `historical_gemm` | Full original L1/L0/ACC schedule | Idle core; 1/2/3 K panels; outer reuse; 4096³ and 2048×4096×4096 |

`generate_controls.py` deterministically creates the four small pairs. These
are synthetic counterparts to the early ladder, **not verbatim official
one/two-buffer examples**. GM input and output arguments must refer to disjoint
allocations; each output tile has a distinct address. Their manual protocols
prime and drain every slot, including slots unused by short or zero-trip runs.
The four-use case releases the input after its fourth vector read. It remains
a diagnostic, not a device-qualified optimal manual implementation.

GEMM scenarios use A[M,K], B[N,K], C[M,N] and compute A × Bᵀ. K is a positive
multiple of 512; M and N are multiples of 128 and 256. The empty case is an
idle core with no assigned output tile, not an invalid K=0 GEMM. Large-shape
replay samples named cores 0 and 23; its counts are per core, not whole-device
totals. `measure.py` interprets scalar arithmetic and structured control in the
actual emitted MLIR. Unknown operations/control and budget exhaustion fail the
row. It never substitutes a static count for an unavailable dynamic count.

Payload trace hashes include executed PTO operations, operands, physical
allocation addresses, types and concrete view descriptors, independent of SSA
names and extra sync-only branches. Equality is checked against the manual arm
for every listed scenario. This is bounded schedule/storage evidence, not a
proof for all symbolic arguments or numerical equality.

## Run and compare

Use a Python interpreter matching a **complete** PTOAS Python/native runtime:

```bash
python3 test/experiments/insert_sync/performance/run.py \
  --python-root /path/to/build/python \
  --arch a3 \
  --baseline test/experiments/insert_sync/performance/baseline-a3.json \
  --output /path/to/disk-backed/results/new-a3
```

Use `--arch a2 --baseline .../baseline-a2.json` for A2. Omit `--baseline` to
collect a new candidate measurement. Existing output directories are rejected.
Every compiler invocation runs serially with a timeout. No build is started.

The runtime used for the initial checkpoint is currently at
`/home/toni/work/pypto3_sync_more/insertsync-builds/PTOAS-insertsync-baseline/python`,
with Python
`/home/toni/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`.
Despite the directory name, its native library is the R2–R3 InsertSync checkpoint
at source commit `60db1026a`, SHA-256
`879c0705280ced29a0dd43617d3fab6e7b5b0d0419051921ef7cabc8bf2c3ab3`.
The runner fingerprints the loaded runtime, not a presumed checkout name.

Each case/arm retains its input, emitted PTO and C++, complete commands,
diagnostics, pass-boundary dumps, placement records and manual-to-auto IR diff.
`results.json` contains all rows, including failures; `summary.md` is the short
report. C++ success means emission, not device C++ compilation. Compile times
are diagnostic and excluded from the regression gate.

The checked-in baselines freeze status, audit verdicts, event footprint, action
counts, placement digests and dynamic traces. Any change returns nonzero and
requires review, including fewer actions: a count reduction can introduce
blocking. Read the retained placement/IR diffs before deliberately refreshing
a baseline. Native hashes may differ between revisions; input/scenario hashes
and architecture must match. This runner compares manual, InsertSync combined
and InsertSync staged; old direct/shared-frontier/lifecycle planners are not
implemented as extra modes of this branch.

Accounting mutation tests run independently of compiler output:

```bash
PYTHONPATH=/path/to/build/python python3 \
  test/experiments/insert_sync/performance/test_measure.py
```

They cover prime/steady/drain formulas, slot wrap, four-use counts, deleted and
moved waits, changed allocations, unsupported scalar operations, input identity
and the baseline comparison gate. Moving a wait changes the action/placement
trace even when counts and payload remain equal.

## Initial checkpoint

A2 and A3 each pass **15/15** case/arm host gates (30 PTO/C++ emissions per
architecture). All listed manual/automatic payload traces match. The current
local auditor reports `unsupported` for all ten automatic rows per architecture
because these are symbolic-loop/cube fixtures. They are not verified-local
passes. Manual rows have no automatic audit verdict.

Local artifacts are retained under
`/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/` in
`performance-r1-a2` and `performance-r1-a3-verified`. The latter is a fresh
rerun against `baseline-a3.json`, with zero baseline changes. All seven
accounting tests and the scoped Ruff check pass.

| GEMM measurement, A3 | Manual | Combined | Staged |
| --- | ---: | ---: | ---: |
| Static sets / waits | 53 / 53 | 44 / 44 | 44 / 44 |
| Static barriers | 0 | 25 | 25 |
| Dynamic sets / waits, 4096³ core 0 | 3921 / 3921 | 3437 / 3437 | 3437 / 3437 |
| Dynamic barriers, same core | 0 | 1783 | 1783 |
| Dynamic body PIPE_ALL, same core | 0 | 0 | 0 |
| Dynamic outside-loop PIPE_ALL, same core | 0 | 1 | 1 |

Automatic barriers in that scenario comprise 1,408 M, 176 MTE1, 176 MTE2,
22 FIX and one terminal ALL. These are candidates for dependency/placement
investigation, not evidence that every barrier can legally be deleted. The
two/three-buffer and four-use controls also expose recurring MTE3 barriers
absent from their manual counterparts. Event IDs are reported per directed
pipe pair; distinct IDs are a footprint, not proven peak simultaneous liveness.

## Remaining ladder and device work

`reference_sources.json` records exact local source files and hashes from
`/home/toni/work/pypto3/pto-isa-main` at
`1216c55831fcc4ed2f096e4ca582ff75a633fbb6`:

| Ladder entry | Located instance / remaining preparation |
| --- | --- |
| Official one/two-buffer examples | Earlier `prompt6.md` cites the Ascend C static-Tensor programming guide; original vendor snippets are not vendored here. Synthetic controls above are explicitly separate. |
| Stationary accumulator, L1 panel, L0 ping-pong microkernels | Present together in the exact full GEMM; independent reduced PTO pairs remain to be extracted. |
| Conv2D | Interior-tile pair and explicit convolution helper header now in [KERNEL_PAIRS.md](KERNEL_PAIRS.md); automatic helper contracts remain unsupported. |
| Additional full GEMM | `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`; historical pair is the executable reference here. |
| FlashAttention | Cube-side QK/P/PV FIFO pair now in [KERNEL_PAIRS.md](KERNEL_PAIRS.md); queue contracts and the vector peer remain outside the runnable local domain. |
| TopK | Source-derived 128-column pair now in [KERNEL_PAIRS.md](KERNEL_PAIRS.md), retaining same-pipe phase barriers. |
| Triangular inverse | Pinned `pto-kernels` column-sweep pair; scalar-dependent TAXPY admission regression. |
| GDN / KDA | Pinned `pto-kernels` WY-stage pairs include both vector/cube peers, workspace signals and half-tail handling; cross-core admission regressions. See [RECURRENCE_PAIRS.md](RECURRENCE_PAIRS.md). |
| AllGather+GEMM / GEMM+AllReduce / MoE | Later cross-core/progress/queue track; excluded from this local flag benchmark. |

For eventual GEMM device timing, recover `caller.cpp`, `run_matmul.py`,
`bench_matmul.py` and `compile.sh` from the pinned PTO-DSL commit with `git show`.
The caller already supports `KERNEL_CPP` and `KERNEL_FN` overrides, so each
retained generated C++ arm can use the same wrapper and launch geometry.
Use separate live device allocations for A/B/C, identical input seeds, a
numerical golden A × Bᵀ, repeated launches and finite timeouts. Preserve the
existing shape constraints, stress both parities and outer reuse, and compare
interleaved warmed timing samples with the same toolchain/device. Record
compiler/ISA/runtime/device pins and numerical results separately from timing.
No device run, hardware latency model or peak event-liveness proof is included
in the host checkpoint.

## MMAD r3 and device follow-up

[MMAD_R3_COUNTS.md](MMAD_R3_COUNTS.md) records the complete A2/A3 rerun with
MMAD-chain analysis off/on, keeping pairs, each named pipe and PIPE_ALL
separate. Use `run.py --mmad-chains` to opt in; it still runs manual,
combined and staged arms for the selected manifest.

[FRONTIER_R4_COUNTS.md](FRONTIER_R4_COUNTS.md) records the matched r4
frontier-refinement off/on campaign over all eleven fixtures, on A2 and A3 and
with MMAD analysis both off and on. Use `run.py --frontier-refinement` to opt
in. The current supported proof domain changes no static synchronization
inventory in these fixtures; all replay and cross-configuration checks pass.

[device/README.md](device/README.md) supplies launch adapters and numerical
goldens for Conv2D, FlashAttention, GDN and KDA. Those adapters require their
first device compilation/validation. The FlashAttention pair's entry-aware
FIFO release is an intentional source-population repair, applied to both
members and reflected in the manifest hashes; historical baseline JSON files
remain historical, not updated expected outputs for this repaired population.

The follow-up instructions are in
[DEVICE_TASK_INSERTSYNC_MMAD_R3_WALLTIME.md](../../../../DEVICE_TASK_INSERTSYNC_MMAD_R3_WALLTIME.md).
