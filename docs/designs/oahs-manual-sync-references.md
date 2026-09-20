# Hand-written Ascend C / PTO-ISA synchronization references

Inspected 2026-09-19. Source/build audit only; no external kernel was built,
numerically validated, or timed. The OAHS device task stays at `495fb9cbd`.

## Provenance: manual and automatic paths are different

Pinned public sources, with per-file SHA-256 manifests in this workspace:

- CATLASS GitHub mirror `ascend-catlass/catlass`:
  `2b85ed307b281baa76d663f11a9c9aa228d56652` (16 selected files).
- PTO-ISA `hw-native-sys/pto-isa`:
  `c0d7148e95ef73bd12a73165fdce4b723a3b7e72` (10 selected files).

PTO examples must not be assumed manual. The relevant distinction is visible
in both source and build:

| Reference | Authored synchronization | Inspected compilation path | Classification |
| --- | --- | --- | --- |
| CATLASS C++ BlockMmad examples | Explicit AscendC SetFlag/WaitFlag, buffer event lists, initialization/drain | CMake ASC language via CANN; no PTOAS invocation in inspected build files | Hand-written template pipeline |
| PTO-ISA manual A2/A3 GEMM | Explicit set_flag/wait_flag and physical TASSIGN | Direct bisheng C++; `--cce-pto-enable`, without the automode flag | Manual PTO-ISA reference |
| PTO-ISA manual common flash attention | Explicit intra-core flags, queue calls and macro bodies | Direct bisheng mixed-core C++; no automode flag | Manual reference with separate queue contracts |
| PTO-ISA automode A2/A3 GEMM | Compiler-managed alternative | Build adds `--cce-pto-auto-enable` | Automatic comparison arm, not hand-tuned evidence |
| PyPTO / PTOAS emitted examples | Depends on the invoked lowering/pass | Must inspect exact invocation and emitted output | Do not use as independent manual oracle |

Build evidence:
[manual GEMM](https://github.com/hw-native-sys/pto-isa/blob/c0d7148e95ef73bd12a73165fdce4b723a3b7e72/kernels/manual/a2a3/gemm_performance/CMakeLists.txt),
[automode GEMM](https://github.com/hw-native-sys/pto-isa/blob/c0d7148e95ef73bd12a73165fdce4b723a3b7e72/kernels/automode/a2a3/gemm/CMakeLists.txt),
[manual attention](https://github.com/hw-native-sys/pto-isa/blob/c0d7148e95ef73bd12a73165fdce4b723a3b7e72/kernels/manual/common/flash_atten/CMakeLists.txt),
[CATLASS example](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/examples/00_basic_matmul/CMakeLists.txt),
[CATLASS build](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/examples/CMakeLists.txt).

This establishes where source synchronization comes from. It does not assert
that lower-level instruction implementations or the device compiler add no
synchronization. A benchmark adoption must archive emitted instructions and
actual compiler flags too. CATLASS also has newer DSL work; the references here
are its explicit C++ templates, not a blanket claim about every CATLASS path.

## Ranked references and actual lessons

### 1. CATLASS BlockMmadPingpong — first projection target

[Source](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/include/catlass/gemm/block/block_mmad_pingpong.hpp)

Relevant source locations:

- 164–183: separate L1 A/B and L0 A/B event primes and final drains.
- 203–212, 242–251: independent A/B refills and readiness publications.
- 276–285: acquire A readiness only on its first copy; publish its release
  after the last L1-to-L0 A copy.
- 300–311: B has its own first-copy acquisition and last-copy release.
- 337–344: matrix execution and subsequent L0 operand releases.

The important distinction is three lifetimes: L1 source, L0 operand, and ACC
result. Releasing L1 waits for the copy engine, not the compute using its L0
copy. B readiness need not gate A's first extraction. The template constrains
L1 and L0 M/N dimensions to match (lines 132–135); do not generalize its
first/last guards to arbitrary M/N subdivision.

**OAHS target:** reproduce these boundaries from physical effects and original
loop occurrences on the unchanged projection payloads. This independently
supports the first-consumer defect found in the preceding local study. Avoid
making one broad loop-entry receipt stand in for both operands.

### 2. CATLASS preload and retained-input variants — lifetime composition

[Preload block](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/include/catlass/gemm/block/block_mmad_preload.hpp)
prepares the next output block's first L1 panels during the final current-block
K tile (the `hasNextBlock` branch at line 219). This is a concrete reason not
to drain every buffer at a lexical child exit. Some of this performance comes
from explicit payload prefetch/reordering; autosync cannot promise that change
when the original program does not expose it.

The [async callback variant](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)
keeps pending work and L1/L0 ownership in an object, with explicit final
synchronization. Its L0A/L0B/L0C stage state is separate. The implementation
should preserve ownership across a callable region only when original control
and the selected endpoints establish that continuation.

The [full-load-A variant](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp)
loads A under `needLoadL1` and retains it while B panels turn over. That is a
useful source for the shared-input projection case. It must not be applied to
Qwen gate/up's actual activation reload. The caller's retention promise needs
proof from native physical effects; the Boolean alone grants no completion.

[Optimized example](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/examples/06_optimized_matmul/optimized_matmul.cpp)
selects the preload dispatch; its defaults use half operands, padding and their
own tile sizes. This is an implementation-pattern reference, not a measured
BF16 Qwen performance baseline.

### 3. PTO-ISA manual GEMM — small independent benchmark candidate

[Source](https://github.com/hw-native-sys/pto-isa/blob/c0d7148e95ef73bd12a73165fdce4b723a3b7e72/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp)

`ProcessKIteration` explicitly publishes A and B separately after their loads,
acquires them at first extraction, and releases MAT after the last extraction.
L0 reuse waits for the previous matrix user of that bank. Invocation primes and
drains are explicit. This is a compact source for a manual-versus-stripped-sync
pair, and the [README](https://github.com/hw-native-sys/pto-isa/blob/c0d7148e95ef73bd12a73165fdce4b723a3b7e72/kernels/manual/a2a3/gemm_performance/README.md)
reports author measurements; none were reproduced here.

Its shared MAT release must not become a general merging rule. CATLASS's
separate A/B releases are the stronger reference where the earlier A refill
must remain independent of the later B reader. Preserve each input program's
actual useful boundaries, rather than copying whichever event count is smaller.

### 4. PTO-ISA manual attention — separate ingress and egress lifetimes

[Source](https://github.com/hw-native-sys/pto-isa/blob/c0d7148e95ef73bd12a73165fdce4b723a3b7e72/kernels/manual/common/flash_atten/fa_performance_kernel.cpp)

The vector path has two different ownership returns:

- Line 451: V→MTE2 protects reuse of the incoming score storage before loading.
- Lines 454–478: queue receive and score loads, then MTE2→V readiness.
- Line 499: MTE3→V protects outgoing probability storage before vector work.
- Lines 510 and 533: input and output ownership are returned separately.

The output-reader receipt therefore gates V, while the score ingress has its
own reuse requirement. This is a much sharper reference than moving every
MTE3 return after every receive. Source queue/free operations, physical aliases
and macro scratch effects still matter. The Qwen tail overlap remains a real
negative test, not something this different buffer arrangement disproves.

[Matmul macro](https://github.com/hw-native-sys/pto-isa/blob/c0d7148e95ef73bd12a73165fdce4b723a3b7e72/kernels/manual/common/flash_atten/pto_macro_matmul.hpp)
and [softmax macro](https://github.com/hw-native-sys/pto-isa/blob/c0d7148e95ef73bd12a73165fdce4b723a3b7e72/kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp)
are also archived. Top-level calls must be expanded when reconstructing a
protocol; treating a macro as one opaque payload can hide its actual deadlines.

### 5. CATLASS attention — prologue / delayed PV / epilogue

[FAI kernel](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/examples/23_flash_attention_infer/fai_kernel.cpp)

The steady schedule explicitly offsets PV from QK with `preLaunch` and rotates
workspace slots over `preLaunch+1` positions. QK, softmax and PV communicate
through distinct cross-core flags; masked/tail loops carry the continuation.
A PV semantic row, its workspace message and the current physical operand-bank
user are consequently different identities. This is a useful comparison for
single-block attention's lagged QK/PV schedule.

**OAHS target:** preserve actual previous participating bank use through
prologue/body/epilogue while keeping queue generations separate. A full-contract
benchmark is larger work; do not import a local protocol by ignoring cross-core
flags or paged-cache address qualification.

## Hardware-contract boundary

[CATLASS TileMmad](https://github.com/ascend-catlass/catlass/blob/2b85ed307b281baa76d663f11a9c9aa228d56652/include/catlass/gemm/tile/tile_mmad.hpp)
contains a shape-dependent PIPE_M barrier at lines 69–74. Other selected paths
use `unitFlag`/accumulation-phase mechanisms. A source file with few visible M
barriers is therefore not proof that issue order alone orders completion.

Keep the current BF16 contract until the exact dtype, shape, target, instruction
parameters and lowering are independently qualified. Access-local ordering and
whole-operation completion remain different claims. Do not add hardware-specific
exceptions to the general theoretical paper merely to copy a reference count.

## Concrete work order

1. **Projection first-consumer placement.** Use CATLASS's separate A/B pattern
   and the existing same-payload reference PTO tests. Qualify invariant physical
   generations through non-unit-step children; acquire once at the actual first
   consumer. Keep real rearming and separate early sources.
2. **Separate L1 and L0 release deadlines.** Audit whether any MAT release is
   delayed by a later M wait or child-exit drain. The last L1 copy is the relevant
   reader, not the following compute. Use the completed-plan fence diagnostics
   as evidence of missing construction-time support, not a deletion algorithm.
3. **Retained-input/sibling continuation.** Start with a source that actually
   retains the generation, such as the full-load-A contract. Keep reload and
   retained-generation negatives paired.
4. **Attention input/output interfaces.** Follow the explicit split in manual
   PTO attention, with the existing queue and tail-storage contracts. Preserve
   the delayed QK/PV schedule separately from physical-bank succession.
5. **Benchmark adoption.** Pin one manual payload, preserve its tiling and data
   movement, strip only synchronization that OAHS is meant to reconstruct, and
   compare the resulting selected words and device timeline to the manual arm.
   Keep required queue, collective and target-specific primitives under their
   contracts. Do not erase all synchronization indiscriminately.

For each imported reference record the eligible key pool, precise payload and
footprints, publication prefixes, first/last uses, primes/drains, source compiler
flags and actual binaries. Compare same-payload arms before changing layouts,
fusing operations or adding prefetch. Those can be later experiments, but are
not effects of autosync alone.

## Evidence limits and retained files

This audit inspected 26 pinned files, about 318 kB, plus repository indexes.
`catlass-files.json` and `pto-isa-files.json` contain source URLs and checksums;
`fetch_sources.py` reproduces the selection. Files are stored under
`/home/toni/work/pypto3_sync_more/manual-sync-reference-work/` on disk.
No downloaded build script was executed. No new compiler or numerical test
result is claimed. The earlier projection reference experiment remains the
local tested evidence, and the first-consumer production mechanism is still open.

## External manual attention update, 2026-09-20

The user reports both original manual cases pass on device: smoke
(128×1024, TILE_S1=128) max diff 3.85e-5 and pipeline
(2048×2048, TILE_S1=512) 1.25e-4, against the upstream 1e-3 threshold.
Three pipeline repetitions give the same 1.80e-4. These are remote reports,
not locally reproduced or a derived numerical bound. The earlier failure came
from mixing one case's binary with another case's input/golden files.

The remote agent records two adaptations: joining the spaced kernel-launch
closing tokens, and omitting unavailable --cce-pto-enable on CANN 9.0.0. The
alternative --cce-enable-pto-passes selects AUTO mode and fails this manual
kernel, so it is not a replacement. Matched-manual PTO transcription remains
pending; preserve QK_PRELOAD=4, physical buffers, queues and the full pipeline
schedule before comparing autosync arms.
