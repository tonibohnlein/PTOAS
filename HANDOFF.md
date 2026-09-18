# OAHS current handoff

Updated: 2026-09-18

## Checkout

- Repository: `/home/toni/work/pypto3_sync_more/PTOAS-oahs-clean-m1`
- Branch: `codex/oahs-clean-m1`
- Current milestone base: `948dafb40`
- Expected working tree at this handoff: clean

Verify these facts at the start of a new session. A newer user commit supersedes
the recorded base.

## Active milestone

Improve the selected synchronization plan before doing more construction-time
optimization.

The immediate target is the remaining dynamically executed `PIPE_M` barrier in
the exact Shenggan GEMM. The implementation must preserve the compact MAT
readiness/release protocol and its measured overlap improvement.

### Concrete next task

Represent the original first-initialization fact

```text
outer_k == 0 && inner_k == 0
```

without expanding the complete nested control graph. The fresh `tmatmul`
initializer executes only on the first inner visit of the first K group. Its
cross-tile predecessor is already intended to be ordered through

```text
previous M work -> M-to-FIX readiness -> FIX store -> FIX-to-M entry acquisition
```

The analysis currently admits fresh initialization after arbitrary interior M
operations. That produces guarded local fences even though those paths do not
exist in the original program.

Use a general qualified first-use/region-boundary mechanism. Do not mark fresh
initialization as an accumulating MMAD, assume same-pipe completion, or add a
GEMM recognizer. Do not restore the discarded full nested-mode expansion; it
increased the selected graph and construction work substantially.

Success means:

- zero dynamically executed named barriers for the exact GEMM, while retaining
  the terminal `PIPE_ALL`;
- no new finish-to-launch ordering between current-bank compute and next-bank
  preparation, or between child compute and independent parent DMA;
- unchanged payload, memory coverage, event balance, and rearming checks;
- a negative test where initialization can genuinely repeat still retains its
  completion requirement;
- focused native/core tests and the exact emitted-order checker pass before a
  new device task is issued.

## Current verified GEMM state

The compact MAT mechanism carries a complete MAT readiness/release cycle before
same-pipe fence decisions. It removes all 16 dynamically executed `PIPE_MTE2`
barriers and adds 18 event pairs per tile.

Correctly optimized device build results:

| Arm | Median | TFLOPS | MAC ratio |
| --- | ---: | ---: | ---: |
| Manual | 229.632 us | 299.3 | 0.917 |
| Compact MAT | 357.290 us | 192.3 | 0.605 |
| Previous handoff | 391.147 us | 175.7 | 0.550 |
| Existing | 533.640 us | 128.8 | 0.434 |

Compact MAT is 8.66% faster than the previous handoff and 33.05% faster than
existing. Its matrix work time equals the previous handoff; the shorter kernel
and higher MAC occupancy show that removing the MTE2 drains restored overlap.
The 18 added event pairs had no measurable cost in this experiment.

The manual plan remains 1.56 times faster than compact MAT, so substantial plan
quality headroom remains.

## Latest frontier diagnosis

The immutable requirement-frontier index reports 1,683 original storage
relationships for the exact GEMM:

- 320 same-visit;
- 160 previous-use;
- 24 region-entry;
- 32 region-continuation;
- 1,147 currently unclassified pairwise relationships.

Most unclassified records are the cross-product of first/middle/final
observations of already qualified inner operand/ACC cycles. Raw relationship
count is therefore not a synchronization target.

Every remaining fence record is on `PIPE_M`, for ACC cell 9, immediately before
a represented fresh initializer. Each has the same residuals:

```text
previous M read  -> new ACC write
previous M write -> new ACC write
```

The emitted static observations simplify to two guarded barrier words, with one
barrier executed per output tile. This is the evidence for the active first-use
task.

The frontier-indexed constructor still emits a byte-identical compact MAT plan:

```text
sha256 6e089c9c2f0d693a1ce0716699e7d3b2253172e1568b546f9842793db4c9eea5
```

One- and two-entry trace checks pass, including all forbidden-overlap checks.

## Validation and artifacts

Useful local artifacts:

- `/home/toni/work/pypto3_sync_more/gemm-cycle-work/frontier-analysis.log`
- `/home/toni/work/pypto3_sync_more/gemm-cycle-work/frontier-details.tsv`
- `/home/toni/work/pypto3_sync_more/gemm-cycle-work/frontier-unknown-modes.tsv`
- `/home/toni/work/pypto3_sync_more/gemm-cycle-work/frontier-analysis.pto`
- `/home/toni/work/pypto3_sync_more/gemm-cycle-work/compact-final.pto`
- Device archive: `/opt/pypto/oahs-compact-mat-device.tar.gz`
- Device archive SHA-256:
  `e115f57710febac032ccc50f97a732e8f9f3cedb6bcf569682616a37b99f94cd`

Last completed local validation at the milestone base:

- standalone OAHS CTest: 18/18;
- native `pto-oahs-selected-test`;
- exact Shenggan one- and two-entry trace checker;
- `git diff --check`.

Common commands:

```bash
cmake --build /home/toni/work/pypto3_sync_more/oahs-m1-core-build --parallel 2
ctest --test-dir /home/toni/work/pypto3_sync_more/oahs-m1-core-build --output-on-failure -j2
cmake --build /home/toni/work/pypto3_sync_more/oahs-m1-native-build \
  --target pto-oahs-selected-test --parallel 2
```

Run the exact GEMM through construction and its checker:

```bash
/home/toni/work/pypto3_sync_more/oahs-m1-native-build/tools/pto-test-opt/pto-oahs-selected-test \
  --construct test/lit/pto/oahs_carried_slot_shenggan.pto \
  > /home/toni/work/pypto3_sync_more/gemm-cycle-work/current.pto
python3 test/oahs/check_carried_slot_trace.py \
  /home/toni/work/pypto3_sync_more/gemm-cycle-work/current.pto
```

## Parked work

The tracked TODO contains the full acceptance criteria. In priority terms:

1. Continue synchronization-plan improvements, starting with first-use ACC
   initialization and then re-measuring the manual-plan gap.
2. Rework guarded attention bank reuse from remaining obligations, without the
   event-count and replay-cost increase of the preserved prototype.
3. Restrict recurring coalescing and make optional specialization decline
   cleanly under key pressure.
4. Optimize construction time by sharing immutable structure and replacing
   recurring changed-plan trials with direct certificates.

Do not start item 4 merely because the current GEMM constructor performs 24,752
recurring-analysis site evaluations. The current project decision is to improve
the selected plan first, then optimize its construction.

## Fresh-session prompt

```text
Read AGENTS.md, HANDOFF.md, and docs/designs/oahs-todo.md. Verify the branch,
HEAD, and working tree. Continue the active first-use ACC milestone. Preserve
the compact MAT protocol and do not use full nested graph expansion. Implement,
run focused tests, regenerate the exact GEMM, and report ordering and mechanism
counts before proposing a device task.
```
