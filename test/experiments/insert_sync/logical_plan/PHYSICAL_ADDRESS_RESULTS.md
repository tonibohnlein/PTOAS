# Physical-address and fallback checkpoint

This checkpoint repairs shared physical semantics before automatic planner
selection. It does not accept the unfinished slot campaign, the compilation-cost
milestone, or device qualification. The OAHS work allowance remains unchanged.

Integer constants and retained casts are evaluated with APInt width and
signedness semantics. Unresolved explicit local addresses remain potentially
aliasing with other roots in the same space. Invalid known intervals are rejected
before either constructor runs. Translation and actual multi-tile lowering now
use the same checked aligned slot addresses. Identity-copy removal requires a
precise singleton range, so equal unknown placeholders or may-address unions
cannot erase a real copy. See [the contract](../../../../docs/designs/oahs-physical-addresses.md).

## Exact-source validation

Built the changed compiler library and native drivers with the established
LLVM 19.1.7 Release `-O1`/assertions configuration. Local build/test concurrency
never exceeded two workers; native test contexts disable threading. Results are
from the pre-commit worktree, with source status and binary hashes retained by
the runners.

| Check | Result |
| --- | --- |
| Physical addresses | 16 direct-pass cases, all three modes; two evaluator cases; 66 native invocations passed |
| Physical slot mapping | 17 cases, 23,018 assertions passed |
| Actual buffer-select lowering | 12 cases passed |
| Default-pipeline lit subset | All 34 relevant tests passed; new RAW test rerun after fixing its FileCheck syntax |
| Mandatory `oahs_focused` CTest | Passed, 51.79 seconds |
| Accepted four kernels | Strict construction, payload/allocation/ABI preservation, replay, boundary and C++ checks passed |

The direct cases include truncation, signed/unsigned extension, narrowing index
casts, unknown runtime and arithmetic addresses, subview offsets, overflow, and
the 32-byte LEFT footprint with 512-byte stride. The evaluator cases distinguish
i1 sign/zero extension and refuse index interpretation with a conflicting
32-bit data layout. Shared known and unknown cast graphs test memoization.

The retained-cast cross-lane RAW reproducer fails on the saved old binary:
the vector reader lacks completion of its aliasing MTE2 producer. All three new
planner modes establish that completion. An earlier two-TLOAD WAW fixture was
discarded as evidence because upstream deliberately exempts that relationship;
this patch does not change that target rule.

| Kernel | Sets / waits | Named barriers | Terminal ALL |
| --- | ---: | --- | ---: |
| One buffer | 4 / 4 | None | 1 |
| Online softmax | 12 / 12 | V: 20 | 1 |
| Q projection | 23 / 21 | M: 5 | 1 |
| QK | 17 / 17 | M: 2 | 1 |

These preserve the previous accepted inventories and useful boundaries. Static
set/wait counts may differ across alternative guarded sites; executed matching
is checked separately. This campaign is not a controlled timing comparison.

Artifacts are under
`insertsync-builds/campaign/logical-plan/physical-addresses-m1/` in the parent
workspace: `final3`, `baseline-raw`, `mapping-lowering1`, `four-kernels1`,
`lit-final`, and the focused/default-lit logs. The focused runner's full output
is in build `test-results/oahs/focused-f02jv4q6`.

## Reproduction

Use the README's established build and Python environment, with mandatory
`PTOAS_EVENT_MODEL_ISL_LIBRARY` where needed:

```sh
python test/experiments/insert_sync/logical_plan/check_physical_addresses.py \
  --opt "$BUILD/tools/pto-test-opt/pto-test-opt" \
  --driver "$BUILD/tools/pto-test-opt/pto-logical-sync-test" \
  --python-root "$BUILD/python" --output "$NEW_RESULTS/addresses"
python test/experiments/insert_sync/logical_plan/check_slot_mapping.py \
  --driver "$BUILD/tools/pto-test-opt/pto-sync-slots-test" \
  --output "$NEW_RESULTS/mapping"
ctest --test-dir "$BUILD" --output-on-failure -R '^oahs_focused$' --parallel 1
python test/experiments/insert_sync/logical_plan/check_constructor.py \
  --focused --case one_buffer --case online_softmax --case q_proj --case qk_matmul \
  --python-root "$BUILD/python" \
  --native-driver "$BUILD/tools/pto-test-opt/pto-logical-sync-test" \
  --output "$NEW_RESULTS/four-kernels"
```

The lit subset selects `insert_sync|buffer_select|multi_tile|multitile|identity_tmov|bufid`.
The local run used one lit worker and serial compiler wrappers. Larger default
regression and hardware tests remain separate gates.
