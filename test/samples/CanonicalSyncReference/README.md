# CanonicalSync protocol reference kernels

This directory contains compact, unsynchronized PTO programs whose storage
lifetimes mirror synchronization protocols found in hand-tuned A2/A3 kernels.
They are analysis inputs for CanonicalSync, not claims of peak kernel
performance.  Synchronization is intentionally absent so each planner can be
compared from the same semantic program.

## Provenance

The protocol shapes were distilled from the following PTO-ISA reference
kernels at commit `0c112d61f41342bd0867ce1080c29f1590d72484`:

- `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`
- `kernels/manual/a2a3/topk/topk_kernel.cpp`

The reference checkout contained unrelated generated cost-model files and
results when this corpus was created.  Only the two tracked kernel sources
above were read; no files were copied from that checkout.

## Cases

### `gemm_ownership_pipeline.pto`

This kernel retains the synchronization-relevant shape of the performance
GEMM while reducing its dimensions and loop nest:

- two physical L1-A slots and two physical L1-B slots;
- four L1 consumers before a panel may be released;
- two physical L0A and two physical L0B slots reused within each panel;
- one L0C accumulator owned by Cube across the four-slice reduction;
- one final Cube-to-FIX handoff per output generation.

The expected storage-generation analysis is:

| Family | Producer | Consumer | Depth | Consumer span |
| --- | --- | --- | ---: | ---: |
| L1-A | MTE2 | MTE1 | 2 | 4 extracts |
| L1-B | MTE2 | MTE1 | 2 | 4 extracts |
| L0A | MTE1 | Cube | 2 | 1 matmul |
| L0B | MTE1 | Cube | 2 | 1 matmul |
| L0C | Cube | FIX | 1 | 4 Cube updates before store |

The eventual optimized plan should synthesize independent ownership channels
for L1-A, L1-B, L0A, and L0B, keep the accumulator stationary, and avoid a
body `PIPE_ALL`.

### `topk_ownership_pipeline.pto`

This kernel retains the two-slot storage flow of the reference TopK:

```text
MTE2 load -> Vector sort -> merge stage -> Vector copy -> MTE3 store
```

It intentionally contains same-Vector read/write reuse around the sort and
merge results.  Those dependencies remain real `PIPE_V` barrier obligations;
an ownership protocol must not erase them.  The input slot lifetime ends after
`tsort32`; the merge/output slot lifetime ends only after the MTE3 store.  Both
logical families have depth two.

## Local use

Parse or lower a sample with InsertSync:

```sh
ptoas --pto-arch=a3 --pto-level=level3 --enable-insert-sync \
  test/samples/CanonicalSyncReference/gemm_ownership_pipeline.pto \
  -o /dev/null
```

Inspect CanonicalSync without committing to materialization:

```sh
ptoas --pto-arch=a3 --pto-level=level3 --enable-canonical-sync \
  --canonical-sync-analysis-only --canonical-sync-dump \
  --canonical-sync-structural-cover=storage \
  test/samples/CanonicalSyncReference/gemm_ownership_pipeline.pto \
  -o /dev/null
```

The TopK sample's ABI treats `src` and `out` as disjoint buffers.  The current
PTO form cannot encode that argument contract, so its local CanonicalSync
analysis additionally needs the explicitly unsafe research assumption:

```sh
ptoas --pto-arch=a3 --pto-level=level3 --enable-canonical-sync \
  --canonical-sync-analysis-only --canonical-sync-dump \
  --canonical-sync-structural-cover=storage \
  --canonical-sync-gm-alias-policy=distinct-roots-unsafe \
  test/samples/CanonicalSyncReference/topk_ownership_pipeline.pto \
  -o /dev/null
```

That option is justified here only by the sample ABI.  It must not be used as
a general correctness shortcut.

The existing lit cases `canonical_sync_ownership_depths.pto` and
`canonical_sync_stationary_accumulator.pto` remain the small depth-one,
depth-two, depth-three, last-consumer, and stationary-accumulator regression
oracles.  Stable behavior discovered with these larger samples should be
reduced into lit tests rather than checking large plan dumps here.
