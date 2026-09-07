# InsertSync follow-up v2: native validation

Validated on 2026-09-07 in `PTOAS-insertsync-r1`, on top of `60db1026a1de8f1bd0154f9bdcb21e0f6d944a7b`. The package at `../insertsync_followup_v2` was applied after checking all 28 package hashes, the exact contents of 20 existing files, and the absence of the 10 addition paths. Changes are uncommitted.

## Native integration fixes

- Copy the lightweight MLIR `Value` handle before calling the non-const MLIR 19 `getParentRegion()` accessor.
- Restrict the report-mode diagnostic handler to the thread performing the coverage query. The supplied handler could intercept a parallel function’s diagnostic, causing a spurious “coverage query failed without a diagnostic” failure; it could also absorb unrelated hard errors.
- Match the native printer’s `call` spelling and extend the coverage fixture with complete and incomplete functions sharing one context.
- Add the required license headers to five restored regression fixtures.

The package itself is retained unchanged. The complete installed 30-file patch, including these fixes, is in `insertsync-builds/campaign/followup-v2-validation/applied-with-native-fixes.patch` relative to the workspace root. Its reverse application was checked against the installed files without modifying them. Originals are backed up in `insertsync-builds/campaign/followup-v2-apply-backup`.

## Validation

- Built `PTOASCompiler`, `pto-test-opt`, and `PTOASPythonPackage` in the checkout’s dedicated configured build using `--parallel 2`.
- Package tests: **25/25 passed**, including 6,000 synthetic traces under two signal-order models.
- Supplied native fixture commands: **42/42 passed** after the integration fixes. The first run’s two failures and diagnostics are preserved separately.
- Related native lit fixtures: **147/147 passed in combined mode and 147/147 in staged mode**.
- Frozen benchmark inputs: **11/11 emit both PTO IR and C++** in report mode, on A2 and A3, in combined mode, staged mode, and staged mode with pruning. Strict staged mode admits **6/11** on both architectures and retains the five expected coverage rejections.
- Original frozen corpus: **208/213 emit both PTO IR and C++** in A3 staged report mode with the established disjoint-argument contract. This restores **43** rows from the pre-patch revision’s **165/213**. The five remaining failures are exactly the original compiler’s five failures. All 208 emitted rows executed analysis and event allocation.
- `git diff --check` passed.

The corpus retains all 213 rows / 186 distinct input hashes; its manifest SHA256 is `a08791b3255bf7944ac8cf6b0bf2f0168a4718d35666711b930792309613adf2`.

## Benchmark synchronization counts

Counts are static local `set_flag + wait_flag + barrier` sites, including `PIPE_ALL` barriers and excluding fixed cross-core/FIFO operations. The three report configurations have identical counts on both architectures.

### Totals

| Fixture | Hand | Original | v2 | v2 − hand |
| --- | ---: | ---: | ---: | ---: |
| One buffer | 13 | 13 | 13 | 0 |
| Two buffers | 25 | 27 | 27 | +2 |
| Three buffers | 37 | 40 | 40 | +3 |
| Four-use producer | 49 | 51 | 51 | +2 |
| GEMM | 106 | 110 | 113 | +7 |
| TopK | 42 | 47 | 47 | +5 |
| Conv2D | 48 | 9 | 9 | −39 |
| FlashAttention | 37 | 55 | 27 | −10 |
| Triangular inverse | 661 | 555 | 555 | −106 |
| GDN | 40 | 80 | 80 | +40 |
| KDA | 43 | 84 | 86 | +43 |

### Set / wait / barrier breakdown

| Fixture | Hand | Follow-up v2 |
| --- | ---: | ---: |
| One buffer | 6 / 6 / 1 | 6 / 6 / 1 |
| Two buffers | 12 / 12 / 1 | 12 / 12 / 3 |
| Three buffers | 18 / 18 / 1 | 18 / 18 / 4 |
| Four-use producer | 24 / 24 / 1 | 24 / 24 / 3 |
| GEMM | 53 / 53 / 0 | 44 / 44 / 25 |
| TopK | 10 / 10 / 22 | 15 / 15 / 17 |
| Conv2D | 24 / 24 / 0 | 0 / 0 / 9 |
| FlashAttention | 18 / 18 / 1 | 13 / 13 / 1 |
| Triangular inverse | 278 / 278 / 105 | 277 / 277 / 1 |
| GDN | 16 / 16 / 8 | 32 / 32 / 16 |
| KDA | 17 / 17 / 9 | 35 / 35 / 16 |

The count differences from the original compiler include earlier revision changes. **The new pruning option makes no additional reduction in these 11 inputs:** every pruning count is zero. Conv2D, FlashAttention, triangular inverse, GDN, and KDA carry `gap-legacy-retained`, which is the outcome of the translator-coverage check added inside the revised InsertSync pass by commit `60db1026a`. It says that this additional check did not certify the production translator's existing summaries. It does not establish that production InsertSync omitted required synchronization. The table therefore reports their measured counts without marking them as failed results.

The supplied standalone pruning regression removes one already-completed named barrier and retains the barrier needed after new source work. An additional automatic-insertion probe derived from that fixture passes the local audit on A2/A3, but its normal InsertSync output offers no further barrier removal.

## Artifacts and limits

Workspace artifact root: `insertsync-builds/campaign/followup-v2-validation`. It contains source/native fingerprints, the complete applied patch, native command results, both lit reports and the 147-fixture selection, all eight benchmark configurations, the 213-row corpus result, and exact campaign commands.

Native library SHA256: `b60849b397f67efe50a26ae64702f4dcbbc299406c81a23f84fc768e285f73b2`.

The broad corpus was checked in A3 staged report mode; the full corpus was not repeated in strict mode or with pruning enabled. Native compilation, static synchronization counts, and the bounded local audit are not device correctness or timing measurements. No device execution or speedup is claimed. Existing benchmark snapshots and original comparison results were retained.
