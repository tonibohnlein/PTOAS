# Clean upstream extraction: M1 and M2

Native reproduction passes on pinned upstream `7e2ec3e29`, using the constructor from accepted research checkpoint `5d6766d69`.
The software architect, compiler expert and algorithms/performance reviewer accepted the extraction with no blocking findings. `codex/oahs-upstream` is now the authoritative development branch; the research branch is a frozen reference.

The initial compiler build was clean. A mechanical license-header update was followed by an incremental rebuild and complete constructor acceptance rerun. The compiler library and native driver fingerprints are recorded in [the evidence](UPSTREAM_EXTRACTION_RESULTS.json).

## Static mechanisms

Each logical row exactly matches the accepted M2 checkpoint. Frozen refiner outputs are comparison artifacts; its implementation is absent.

| Fixture | Upstream default sets/waits | Frozen refiner sets/waits | Clean logical sets/waits | Logical named barriers | Logical PIPE_ALL |
| --- | ---: | ---: | ---: | --- | ---: |
| one_buffer | 6/6 | 6/6 | 4/5 | none | 0 |
| online_softmax | 12/12 | 12/12 | 13/13 | V: 20 | 0 |
| q_proj | 19/19 | 29/29 | 24/27 | M: 5 | 0 |
| qk_matmul | 21/21 | 22/21 | 19/20 | M: 2 | 0 |

Unequal static sets/waits represent alternate guarded sites; all checked executions balance and consume the matching tokens. Upstream default and the frozen revised/refiner configurations are different baselines and remain labeled separately.

## Executed mechanisms and scalar cost

| Fixture / scenario | Refiner pairs | Clean logical pairs | Logical named barriers | Logical PIPE_ALL | Logical scalar/control operations |
| --- | ---: | ---: | --- | ---: | ---: |
| one_buffer / trips_16 | 66 | 63 | none | 0 | 344 |
| online_softmax / 16 | 96 | 94 | V: 257 | 0 | 680 |
| q_proj / core0 | 519 | 509 | M: 127 | 0 | 3327 |
| qk_matmul / 16 | 261 | 255 | M: 32 | 0 | 1049 |

Q projection uses 32 loop iterations on core 0. Counts and scalar traces are unchanged from the checkpoint, including its substantial guard overhead. This extraction adds no new synchronization or device-performance improvement.

## Validation

- Four unchanged strict-constructor inputs, PTO and C++ emission; exact checkpoint static/executed metrics, action traces, payload, allocations, views and ABI.
- Fourteen supported boundary observations against the frozen refiner, with no later acquired cross-lane prefix. QK explicitly excludes the independent Q1 preload at its first Q0 consumer for bounds 1, 2 and 16.
- 63 native MLIR/libisl relation checks; 15 occurrence-import cases including 1,936 softmax phase pairs.
- Five independent known-local requirement comparisons, including overlap, added readers and equivalent predicates. Softmax checks 439 local obligations and 2,025 occurrence relations. This oracle excludes GM.
- Nine physical-admission cases and three focused GM-query cases (twelve query answers), including unknown bounds and actual output overlap.
- 24 structured empty/skipped-reader executions; six deliberate emitted-IR corruptions rejected after positive native controls.
- Two- and three-buffer strict construction reaches the recorded publication-domain lowering limitation. Hybrid output exactly matches upstream fallback. Zero-budget strict failure and fallback equivalence also pass.
- Seven observer/Boolean-replay tests and the native pass selector/strict/fallback lit test pass.

## Dependency and development boundary

The relation and occurrence engines retain the accepted code. Physical import no longer depends on lifecycle/frontier graph construction; GM occurrence proofs are separate utilities. Upstream translator, alias analyzer and slot analysis remain byte-identical. See [the dependency map](../../../../docs/designs/oahs-upstream-extraction.md).

Memory-carrying `scf.for` arguments and `scf.while` are explicitly unsupported because research-only forwarding changes were not extracted. General guard lowering, physical-section exits, MMAD discharge, wider allocation and GEMM remain later work.

The research worktree and its three unfinished guard files are preserved byte-for-byte. No device correctness or wall-time qualification was performed.

## Review acceptance

All three reviewers inspected the final sources and recorded native evidence; none independently reran device tests. Their acceptance covers this M1/M2 extraction. Guard generalization and the remaining milestones continue on the clean branch, using the preserved research changes as unfinished work rather than accepted code.
