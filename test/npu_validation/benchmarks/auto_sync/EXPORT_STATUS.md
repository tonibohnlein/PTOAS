# PTO IR preparation status

Updated 2026-10-06. New source revision:
`tile-ai/tilelang-ascend@cbc67f2697e8ee18357adc447272691432b3948d`.

## Available artifacts

All **nine additional kernel families** have one explicitly specialized PTO IR
port, with reproducible generators, an authored-protocol `expert.pto`, a
synchronization-free `input.pto`, and `port.json`. The files are under
`prepared/<benchmark>/`. Both synchronization passes receive the same input.
The new ports have passed IR verification and a textual check that removing
synchronization from the expert leaves exactly the input's payload, storage,
and control code. This checks the two port variants; it does **not** prove that
the translation from upstream is correct or that the manual protocol is sound.

They are benchmark ports, not automatic TileLang exports or production
recognizers. Source inspection and the translation decisions are recorded in
[AUDIT.md](AUDIT.md). Native-device validation remains outstanding for all nine.

## Compilation results

These are baseline results captured before the subsequent analysis changes.
C++ generation used the recorded local compiler binaries, all inputs planned,
extra synchronization and memory planning disabled in the final backend:

| Port | Specialization | Expert C++ | Existing C++ | Frontier C++ |
|---|---|---|---|---|
| Persistent GEMM | Full default 8192×1024×8192, 20 workers | Yes | Yes | Analysis rejected |
| GroupNorm | Serial FP32, N=4,C=64,H=W=16,G=4, four spatial steps | Yes | Yes | Analysis rejected |
| Causal conv1d prefill | FP16, two sequences, dim128, width4, sequence lengths divisible by4 | Yes | Yes | Analysis rejected |
| Causal conv1d decode | BF16, batch8, dim2048, width4, SiLU | Yes | Yes | Analysis rejected |
| GDN chunk cumsum | B2,H16,L16384,C128, eight chunks per task | Yes | Yes | Analysis rejected |
| mHC head mix | Forward, 8192×4 reshaped to2048×16 | Yes | Yes | Analysis rejected |
| Pipelined vector add | Full1024×1024, two stages, two vector lanes | Yes | Yes | Analysis rejected |
| Fused gating/delta rule | 24 sequences, runtime lengths1..8, dk=dv32, normalization enabled | Yes | Yes | Analysis rejected |
| Lossless block cast | Generic64×64 output-scale blocks; BF16→FP32 | Yes | Yes | Analysis rejected |

Including the two earlier GEMMs: **24/33 C++ generations succeed**. Those two
still compile with all three plans. New-port frontier failures happen in
structured analysis, before allocation; these results do not establish any
ID scarcity. No production compiler code was changed for this preparation.

Diagnostics group as follows:

- Persistent GEMM, cumsum, mHC, vector add and gating: `sequence loop has no
  exact regional rotating or numerical template`.
- GroupNorm, both convolutions and lossless cast: `finite guarded analysis has
  unsupported control or additional prerequisites`.
- Both groups also report that the arithmetic route's required primitives are
  unsupported. These diagnostics identify the rejecting routes, not a proof
  that the kernels are intrinsically outside the paper's tractable classes.

The durable local compilation record is
`.local/expert-nine-ports-20261006-final/generation.json` at the worktree root.
It contains tool hashes, manifest hash, commands, per-step wall times, exit
codes and artifact hashes. Each kernel directory includes the exact diagnostics.
These are single compilation observations, not a compilation-time benchmark.

## Reproduce

```sh
# Verify source/input pins, regenerate in memory, check the pairs and verify IR.
python3 test/npu_validation/benchmarks/auto_sync/export_ports.py \
  --pto-test-opt .local/modeled-access/pto-test-opt

# Regenerate C++ into a new directory. A nonzero exit reports rejected variants.
python3 test/npu_validation/benchmarks/auto_sync/prepare.py \
  --pto-test-opt .local/modeled-access/pto-test-opt \
  --ptoas .local/shared-alias/ptoas \
  --timeout 60 --output .local/expert-sync-comparison
```

`export_ports.py --install` updates the registered artifacts and their hashes
only after every newly generated pair passes verification. The default command
checks reproducibility without rewriting files.

## What can run

We have **PTOAS-generated C++**, not nine device-qualified executables. This
workspace has no CANN compiler, TileLang/torch-npu installation or NPU runtime.
The new ports also need launch wrappers, input generation and reference checks;
`port.json` records the selected factory configuration and restrictions.

Before timing, the device work must:

1. Build and verify each port's authored variant against the independent source
   reference, including cache/final-state and scale-factor outputs.
2. Qualify launch indexing (logical core and vector-lane indices), partial
   loads, helper expansion and the source protocols that use ID6. Keep the
   tested frontier capacity at IDs0..5; do not silently remap source flags.
3. Validate existing/frontier against the same reference, resetting all mutated
   storage outside timed regions. Frontier needs the analysis gaps fixed first.
4. Report native upstream timing separately from the ported expert. Passing the
   same numerical test does not establish equal storage schedules or speed.

Scope limitations: GroupNorm's channel-pipeline alternative, mHC backward and
lossless cast's optimized default max4 path are **not** ported. The lossless port
is a valid generic source configuration, not a replacement claim for max4 HPC
performance. Prefill excludes zero trips and partial four-token groups because
its source prefetches all four rows. Gating excludes empty sequences because
its source prefetches the first token unconditionally.

## Why an explicit port was needed

At the pinned revision, TileLang's `target="pto"` produces PTO-ISA **C++**.
`tilelang/engine/lower.py` calls `target.build.tilelang_ascend_pto` and retrieves
`get_source()`; the target generator emits C++ including `pto/pto-inst.hpp`.
It does not emit PTOAS `.pto` MLIR. Relevant compiler/helper files are pinned
under `sources/` and listed in `benchmarks.json`. A reusable TIR-to-PTOAS importer
would be a separate frontend project; it is not part of these benchmark ports.
