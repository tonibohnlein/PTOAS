# Manual synchronization benchmark

## First adopted reference

PTO-ISA `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`
at `c0d7148e95ef73bd12a73165fdce4b723a3b7e72`, compared with PTOAS
`495fb9cbda7649f15a7fc09d6b1d91ffb7737d54`.

This is a benchmark harness, not a production pass change. No device is
available on the preparation machine. Device correctness and performance are
pending. CATLASS and attention remain separate benchmark adoptions; they are
not covered by this GEMM result.

| Arm | Purpose |
| --- | --- |
| `original_cpp` | Upstream manual device body, unchanged; common host launcher |
| `manual` | PTO transcription of exactly the same payload and event sequence |
| `manual_banked_keys` | Explicit amendment: separate readiness keys for each MAT bank |
| `handoff` | OAHS applied once to the same unsynchronized PTO |
| `existing` | Existing InsertSync applied once to that PTO |

The original C++ control guards a four-chunk panel load inside a flat K loop.
The PTO transcription factors this into panel/substep loops. It also expresses
the flags as equivalent two-state carried values. Both automatic arms receive
that same normalized control, tiling, addresses, layouts and operation order.
The admitted K values are multiples of 512: both original bank flags return to
zero at each output tile. This is not a general normalization for arbitrary K.
The original C++ arm measures any device cost of this transcription separately.
It is not valid to attribute an original-C++/OAHS difference entirely to sync
without first examining original-C++/manual-PTO.

The main configuration is the original 6144×6144×6144 FP16→FP32 GEMM, 24 cores,
1536×1024 output per core, 128×256 output tiles, K=64 compute chunks and
K=256 MAT panels. A is row-major; **B is column-major**, stored as N contiguous
rows of K elements by the harness. LEFT bank spacing is 32768 bytes.
The 256×512×512 one-core configuration is a smoke test, not a performance proxy.

## Source and protocol checks

`probe/` executes the unchanged upstream template body with trace intrinsics.
It checks addresses, shapes, layouts, GM strides, extraction offsets, operation
order and initialization/accumulation roles. Its aliases were checked against
the pinned A2/A3 `pto/common/pto_tile.hpp`. It is not the native PTO backend or
a numerical emulator.

Full reference traces match on **all 24 cores / 388,224 payload operations**;
the manual transcription also matches the original event sequence exactly.
All generated arms preserve those payload records. Local causal tests cover
one and four output tiles with the full K length, plus the smaller smoke case.
The independent oracle checks conservative whole-tile local effects; GM
addressing enters identity, while numerical GM coverage is a device check.

The upstream manual uses readiness keys 0 and 1 for A/B, irrespective of the
MAT bank. On the second panel, the initial free token for the second MAT bank
does not prove that the first panel's readiness keys were consumed. The local
ordinary-event causal-rearming check therefore rejects that republication.
This is a **contract-level finding**, not an observed device wrong result.

`manual_banked_keys` changes only these key identities: A/B use 0/1 for MAT
bank 0 and 2/3 for MAT bank 1. It passes the bounded rearming check, with the
same number and placement of events and the same concrete payload ordering.
It is labelled as an amendment, never silently substituted for upstream.

`check.py` explicitly records the original's missing rearming edges. It can
continue that concrete matching graph for diagnostics without inventing edges;
the resulting original graph is **not** marked certified. All other arms must
pass the full local check. Removing readiness fails; inserting a whole-pipeline
drain adds order. `test_check.py` exercises those distinctions.

### Local results, full K=6144

Counts include invocation priming/draining. ALL is separate from named barriers.

| Arm | One-tile pairs | Named barriers | Four-tile pairs | Named barriers | Local rearming |
| --- | ---: | ---: | ---: | ---: | --- |
| manual | 270 | 0 | 1,068 | 0 | Rejected |
| manual, banked readiness | 270 | 0 | 1,068 | 0 | Pass |
| OAHS | 296 | 0 | 1,166 | 0 | Pass |
| existing InsertSync | 245 | 97 | 971 | 388 | Pass |

OAHS and existing retain one terminal ALL; the manual source uses its explicit
drains. Against the banked manual's finish-to-issue graph, OAHS adds zero and
removes 22/94 relations on one/four tiles; existing adds 1,375/5,632. These
relations do not measure latency or all effects of pipe drains. Native OAHS
construction/reconstruction succeeds: 671 sites, 3 selected updates, 18,782
replay evaluations, 12 recurring channels, zero recurring omission trials
(one local run approximately 0.44 s). All eight PTO arms lower to C++.

## Reproduction

Run `generate.py OUTPUT` and use `prepare.py` with explicit paths to one
compiler build's `pto-test-opt`, `pto-oahs-selected-test`, and Python/ptoas
command. `--source` names the pinned upstream C++ and `--oracle` names
`test/oahs/check_carried_slot_trace.py`. Both are included in the device bundle.
The independent full-source trace campaign is test work; its cost does not
belong in the compiler timing.

On device, `build_device.py` uses one explicit JSON flag list, identical PTO
headers and `-O2` for every arm. It rejects downstream PTO automode. For the
original body it excludes only the upstream host launcher; **it does not define
`__COSTMODEL`**, which would change the device header path. Every command,
cc1 driver trace and loaded library hash must be archived.

`run_device.py` uses ACL events and a vectorized blocked FP64 reference, computed
once per seed and shared across all arms and four queued output slots. It gates
timing on three seeds, all outputs, output sentinels/guards, finite values and
unchanged inputs. The predeclared numerical bound is
`gamma_K * sum(abs(a*b))`, with `u=2^-24`, `gamma_K=K*u/(1-K*u)`, on exact FP16
dyadic inputs and FP32 accumulation without overflow/underflow. Report the
actual error/bound and bitwise comparisons too; do not loosen a failed bound.
The bound is a numerical model to check against the target implementation,
not a theorem about undocumented accumulator modes.

Six rotated rounds ×30 samples give 180 timings per arm after every arm's
correctness gate completes. Use matched timeline profiles to attribute MTE2,
MTE1, M and FIX stalls and overlap. Event counts and relation subsets alone are
not performance predictions. Device scripts are syntax-checked locally; their
ACL/CANN execution remains to be validated by the remote task.
