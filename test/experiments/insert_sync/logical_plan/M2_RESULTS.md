# Milestone two: Q projection and QK

Native acceptance passes on the M2 implementation committed as `68a31b6c8`.
The software architect, compiler and algorithms/performance reviewers accepted
this native milestone. Device qualification remains outstanding.
The constructor starts from each unchanged unsynchronized input; neither legacy
insertion nor the handoff refiner supplies its seed.

| Fixture | Arm | Sets | Waits | Named barriers | PIPE_ALL |
| --- | --- | ---: | ---: | --- | ---: |
| Q projection | Revised | 39 | 39 | M: 6 | 1 |
| Q projection | Refiner | 29 | 29 | M: 6 | 1 |
| Q projection | Logical | 24 | 27 | M: 5 | 0 |
| QK | Revised/refiner | 22 | 21 | M: 2 | 0 |
| QK | Logical | 19 | 20 | M: 2 | 0 |

Alternative guarded sites explain unequal static sets/waits. All observed
executions balance and drain their actual tokens; reconstruction separately
proves occurrence matching and reuse. At the recorded larger scenarios:

| Fixture / scenario | Arm | Executed pairs | Named barriers | PIPE_ALL |
| --- | --- | ---: | --- | ---: |
| Q projection / core 0, 32 iterations | Refiner | 519 | M: 128 | 1 |
| Q projection / core 0, 32 iterations | Logical | 509 | M: 127 | 0 |
| QK / 16 iterations | Refiner | 261 | M: 32 | 0 |
| QK / 16 iterations | Logical | 255 | M: 32 | 0 |

Scalar/control operations increase from 435 to 3,327 for Q projection and from
354 to 1,049 for QK at those bounds. These are replayed scalar/control operations,
not target instruction counts. The increase is a material unresolved cost; the
pair reductions alone do not establish an overall runtime benefit.

Payload, allocations, views and ABI match. The five Q-projection/QK boundary
observations introduce no later acquired cross-lane completion prefix. QK's
first MTE1 reader explicitly acquires Q0 without acquiring independent Q1.
QK at zero trips retires both preloads using one pair instead of five.
Finite observations supplement symbolic reconstruction; they are not a device
performance result or a universal schedule-equivalence proof.

## Implementation and checks

- Interior first-use domains can propose representable constant-IV comparisons
  from rational extrema. Existing guards are tried first. Exact integer-domain
  equality, operand availability and fresh reconstruction decide acceptance;
  rational relaxation never supplies a synchronization proof.
- The pinned MLIR subtraction path asserted in `Simplex::detectRedundant` on
  the actual Q-projection query. Exact subtraction now separates total floor
  definitions from membership constraints, merges floor witnesses, preserves
  equalities, and partitions by the first violated integer constraint. Fresh
  implication checks avoid the failing mutable redundancy routine. Repeated
  bounded qualification handles newly exposed integer witnesses. The captured
  native query and residue/equality/nested-floor cases remain differential tests.
- Assignment queries share completion progress for one unchanged logical plan.
  Adding barriers reopens all affected source searches. Deletion/replacement
  still uses fresh completion; emitted reconstruction has independent state.
- 63 native MLIR/libisl relation checks pass. M1 online-softmax and one-buffer
  acceptance passes again on the changed engine. Online-softmax still executes
  94 pairs, 257 V barriers and no PIPE_ALL at bound 16.
- 24 structured executions include empty/skipped readers and a first/else
  reader split. Six deliberate emitted corruptions are rejected; zero-budget
  strict failure and unchanged hybrid fallback pass.

The saved Q-projection/QK run is `m2-native/acceptance43`; its interrupted
remainder was completed in `acceptance44`. The retained rows, input hashes,
scalar counts, invocation times and compiler/driver fingerprints are in
[M2_RESULTS.json](M2_RESULTS.json). The binary was built incrementally before
the user committed the unchanged worktree. This is not an independent clean
build. No device compilation or runtime was performed.

## Costs and remaining scope

Recorded PTO-emission invocation times are 21.93 s for Q projection and 4.21 s
for QK, versus 0.52 s and 0.62 s for the refiner. The new M1 rerun takes 9.33 s
for softmax. These include invocation overhead, not isolated planner timing.
Q projection consumes 272.7 million work units; the experimental allowance is
384 million, increased from M1's 128 million. The allowance is neither a
wall-time nor a memory bound. Further admission should first address repeated
relation work and scalar overhead rather than routinely increasing it.

Native requirements remain conservative may-access conflicts. General affine
guard materialization, slot-relative final-use guards, target/reservation-aware
pools, dynamic storage-derived keys, physical-section exits, qualified MMAD,
historical GEMM construction and full corpus acceptance remain later work.
The OAHS document sharpens those contracts; it does not restart this constructor.
