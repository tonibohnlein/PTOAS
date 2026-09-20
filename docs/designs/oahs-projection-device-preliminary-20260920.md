# Preliminary projection device feedback — 2026-09-20

Source: device-agent report pasted by the user. Raw samples, profiles and final
archive have not been inspected locally. This is feedback for the dispatched
`495fb9cbd` campaign; verify exact arm revisions and loaded hashes in the final
manifest before treating that association as independently confirmed.

All times below are reported medians in microseconds. The agent reports disjoint
IQRs against both comparators for every projection row.

| Kernel | Existing | Candidate | Before | Latency reduction vs before | Excess latency removed | Remaining slowdown vs existing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| down_proj | 30.262 | 38.024 | 49.165 | 22.66% | 58.94% | 25.65% |
| q_proj / out_AIC | 33.580 | 44.041 | 57.462 | 23.36% | 56.20% | 31.15% |
| kv_proj | 33.725 | 44.272 | 57.588 | 23.12% | 55.80% | 31.27% |
| gate_up_proj | 65.512 | 88.235 | 114.839 | 23.17% | 53.93% | 34.69% |
| lm_head | 434.843 | 663.915 | 845.928 | 21.52% | 44.28% | 52.68% |

Here `excess latency removed = (before-candidate)/(before-existing)`. The
reported "41–47%, consistently about 45% closed" column does not follow from
the supplied medians. The recalculated values are above; they must replace that
column in the final report. Similar relative gains support a shared mechanism,
but do not identify every remaining stall or its cause.

## GEMM

Reported medians: candidate 224.507, before 225.961, reference 225.355, manual
227.881, existing 533.572. The before/reference binaries are reported identical.
Candidate versus before is -0.643%, versus reference -0.376%; the identical-
binary median spread is 0.269%. Therefore the statement that both improvements
fall *inside that 0.269% spread* is arithmetically incorrect. A single spread
between identical binaries is not a calibrated significance threshold anyway.
Retain the conservative parity/no-regression interpretation pending raw rounds;
do not claim a small speedup solely from these summary medians.

## Reported validation and pending work

- 153 device correctness runs, zero failures: down 27, projections 90, LM 21,
  GEMM 15; three seeds, identical input hashes across arms.
- Nine static SET/WAIT pins match; GEMM 200/394/782 pairs pass the committed
  checker. Host matrix: 267 successful rows. OAHS: 21/21.
- Initial aggregate profiles have now been reported (below). Remaining campaign:
  resource-conflict/MemoryL0 attribution, vector controls, post-RMSNorm, coupled
  attention, final archive. Per-instruction timelines are unavailable on the
  task's CANN/msprof build; do not claim they were collected.
- No timing here validates the separately prepared first-consumer reference
  plans or the three new external-reference benchmark tasks.

## Next interpretation/implementation step

The independent-bank placement fix has a substantial reported device benefit;
the projection regression remains unresolved. First-consumer MAT readiness is
still the next concrete implementation target from the local source study:
publish after the actual B load, acquire once at its first participating
extraction, preserve separate A readiness and real consumption/rearming credit.
That reference change has local ordering evidence but no device result.

Map each remaining blocking acquisition in the compiler/emitted plan to its
protected load, physical bank/generation and first consumer. Use available
aggregate profiles and controlled device arms to test that attribution. Distinguish
late/broad readiness, genuine BF16 M completion, missing construction-time MAT
cycle support, and event-reuse constraints. Compare candidate with both before
and existing. Do not replace remaining fences or waits based on counts alone.
The larger LM slowdown is a useful scaling case after the mechanism is isolated.

## Follow-up aggregate profiles

User-reported sum of per-pipe active times divided by core time:

| Kernel | Existing | Before | Candidate | Summed active time, existing / before / candidate (µs) |
| --- | ---: | ---: | ---: | --- |
| down_proj | 1.679 | 1.030 | 1.216 | 77.5 / 78.3 / 76.4 |
| q_proj | 1.808 | 1.036 | 1.225 | 89.5 / 89.6 / 91.0 |
| kv_proj | 1.686 | 1.031 | 1.211 | 98.5 / 96.2 / 95.6 |
| gate_up | 2.075 | 1.081 | 1.388 | 139.9 / 131.3 / 133.0 |
| lm_head | 1.577 | 1.013 | 1.189 | 1324 / 1288 / 1297 |
| Shenggan | 1.103 | 2.628 | 2.628 | 559 / 568 / 571 |

This supports reduced aggregate pipeline concurrency as the projection issue,
with partial recovery by the candidate and the opposite advantage on Shenggan.
It does not identify the responsible command or exclude idle periods/resource
stalls. ResourceConflictRatio and MemoryL0 campaigns are queued for down, q,
gate/up and Shenggan; vector controls/x_gamma are running.

Interpretation requirements for the final report:

1. Verify pipe counters use the same core population, weighting and observation
   interval as the denominator and do not double-count overlapping categories.
   Under that assumption the factor is average active-pipe multiplicity. Above
   one proves some concurrency; near one does not prove no overlap. Two pipes
   active for half the interval followed by half idle also give factor one.
2. Call the numerator summed active time, not invariant payload work. Gate/up
   changes -6.15% before/existing and -4.93% candidate/existing, contradicting
   the supplied blanket ±3% claim. Payload identity needs its own trace/hash.
3. With a nearly constant numerator, factor and duration are algebraically
   related. Their matching rank is not independent evidence identifying a
   particular wait as the cause. A controlled placement change plus the native
   requirement witness is the useful next causal test.
4. A missing instruction timeline is an instrumentation limit, not a reason to
   invent one or postpone the already-supported first-consumer experiment.
   Preserve genuine memory, native-M and event-reuse obligations in that change.
