# Review of the 2026-09-21 full sweep

Reviewed locally on 2026-09-21. This is an audit of saved device records, not
new device execution or instruction-level attribution.

## Provenance and scope

Downloaded `oahs-full-20260921-report.tar.gz`, SHA-256
`73ca52af6b539946fdc2754a5edb70fd0c40f8a7324b207dc8b71da67ad4738a`.
All 164 members listed in its report manifest match their hashes. The 165-file
archive is **report-only**: source manifests, result records and raw measurements
are present; actual prepared kernels, harnesses, inputs and binaries are in the
separate full `oahs-full-20260921-results.tar.zst` archive. The bundled reproduction
instructions describe that larger archive, not this downloaded report bundle.
Local extracted evidence: `../full-sweep-review-20260921/`.

Compiler: `2cc458cbe1e5aff6f77ffc6fea62b13686f38a94`, clean tracked source.
PyPTO: `166bf7ac40924bacf6035b9fea6a0b9bbc08f9db`.
pypto-lib: `205255b4770ee84dfa176bcbc7bbef651953c7e1`.
This predates last-reader and static-FIFO relay changes. Current-source recapture,
framework/toolchain and timing scope differ from earlier MAT campaigns; do not
combine their absolute times or infer a historical regression from them.

- 48 matched original entries: 18 PyPTO examples, 30 pypto-lib entries.
- 77 isolated configurations, collapsing to 54 normalized payload/ABI/shape groups.
- Coverage inventory: 266 Python files, with exclusions recorded individually.
- Queue: 125 finished, 91 host-blocked, 19 failed. All eight cards participated.
- 111 blocker records: 91 host, 18 original-entry device, one isolated reference
  failure, one additional distributed-compile blocker. These are not 111 OAHS
  wrong-code cases, nor a denominator for the 125 successful comparisons.

## Measurement audit

Each arm has 100 consecutive invocations; discard the first 50, report the last
50. I recomputed medians and inclusive quartiles for all 250 arm summaries from
25,000 raw samples; every summary matches and every sequence contains 1..100.
Original entries record three correctness invocations per arm; isolated kernels
record eight. Qwen's explicit golden checks cover seeds 7 and 23 on both arms.

Original-entry duration is runtime device-orchestration wall time. Isolated
measurements are single-launch ACL-event stream intervals, which can include
host submission gaps. They are not interchangeable or profiler-exclusive times.
The inspected KDA/Qwen records contain one existing block followed by one handoff
block, not rotated matched rounds. Disjoint quartiles and small split-half drift
identify follow-up candidates, not causal attribution or independent replication.

Thirteen isolated pairs record identical device binaries. They nevertheless show
apparent latency reductions from -37.89% (`ffn_swiglu_matmul`) to +65.55%
(`split_k_atomic`). The latter even has disjoint IQRs; its existing half-sequence
drift is +165.3%. Thus disjoint IQRs alone cannot certify isolated improvements.
Original-entry identical-binary controls show much smaller descriptive differences
(0.07%, 0.31%, 0.42%), but are not a universal noise threshold for other workloads.

## Original-entry findings

These are the strongest new workload leads. First seven listed regressions have
separated quartiles and less than about 1.1% absolute split-half drift per arm.

| Case | Existing us | Handoff us | Handoff / existing | IQR overlap |
| --- | ---: | ---: | ---: | --- |
| models__glm5_3_flash__kda_projection | 237.389 | 316.950 | 1.3351 | no |
| models__deepseek_v4_flash_dspark__decode_compressor_ratio4 | 99.620 | 111.330 | 1.1176 | no |
| models__deepseek_v4_flash_mtp__rmsnorm | 48.210 | 52.130 | 1.0813 | no |
| models__deepseek_v4_flash_dspark__rmsnorm | 51.610 | 55.190 | 1.0694 | no |
| models__deepseek_v4_flash_dspark__hc_pre | 79.450 | 83.850 | 1.0554 | no |
| models__deepseek_v4_flash_dspark__lookup_embedding | 100.600 | 104.190 | 1.0357 | no |
| models__deepseek_v4_flash_mtp__hc_pre | 60.890 | 63.020 | 1.0350 | no |
| models__qwen3_14b__prefill_fwd | 25198.820 | 24736.260 | 0.9816 | yes |

KDA projection adds about 79.56 us and is the first investigation target. Check
which constituent kernels changed before attributing it to lost pipeline overlap.
Qwen prefill is the authentic **two-layer, batch-16, capacity-128 fixture**, not
the complete 40-layer model. Both arms pass; its descriptive 1.84% reduction has
overlapping IQRs. Crucially, a coupled runtime now exists on the device machine.
Reuse it for the pending FIFO campaign only after checking that its prepared
inputs and queue/ABI contract match the pinned row48/49 fixtures.

## Isolated-kernel findings

| Case | Existing us | Handoff us | Handoff / existing | IQR overlap |
| --- | ---: | ---: | ---: | --- |
| down_proj | 34.770 | 31.380 | 0.9025 | no |
| gate_up_proj | 69.600 | 62.580 | 0.8991 | no |
| kv_proj_variant | 37.220 | 34.620 | 0.9301 | no |
| q_proj | 36.580 | 33.350 | 0.9117 | no |
| out_proj_cube | 37.320 | 33.700 | 0.9030 | no |
| lm_head | 475.510 | 481.690 | 1.0130 | yes |
| post_rmsnorm | 31.810 | 32.940 | 1.0355 | no |
| attention_finalize_0 | 12.860 | 19.250 | 1.4969 | no |
| matmul_acc | 14.380 | 18.020 | 1.2531 | no |

Projection results generally support the earlier direction, but do not replace
the prior matched MAT campaign. `kv_proj`'s larger apparent 26.6% reduction is
confounded by -22.3% baseline split-half drift; its variant is a cleaner lead.
LM head has no resolved benefit here. Post-RMSNorm's small regression should be
checked alongside the original DeepSeek RMS entries, not equated with them.
`attention_finalize_0` and `matmul_acc` are useful small-kernel follow-ups, but
require a controlled timing check before ranking them as confirmed regressions.
Finalization variants share a normalized group yet produce differing outcomes.
The `qkpv_plan*` names map to prefill rows28–31 (small plan-building inputs), not
the coupled attention kernels48/49. Do not credit them as attention timings.

## Failures and missing evidence

Seven original-entry device records fail numerical checks after an existing-arm
launch, before any handoff launch. Nine report runtime lane/finalization errors,
one is a DFX timing-contract refusal, and prefill_indexer_compressor fails a later
check after both arms have passed seed checks. Their root causes are unassigned.
The isolated `mlp_cast` failure is a rejected rounding oracle; the corrected,
separately named `mlp_cast_round` passes. Do not count it as compiler wrong code.

Of the 91 host blockers, 23 point to handoff `codegen_errors.txt`, two to existing
codegen diagnostics; others include framework legalization/allocation, target and
distributed-fixture requirements. The report bundle does not contain those full
compiler stderr files. Preserve them and classify refused contracts versus
construction failures before broadening native admission. The distributed LM-head
record is separate and did not execute with substitute collectives.

## Changed local priorities

1. Obtain the small KDA projection reproducer from the full archive: prepared PTO,
   both selected plans, emitted C++, build records and per-kernel runtime mapping.
   Reconstruct at the frozen and current revisions, identify a concrete additional
   dependency and its physical requirement. Request only a short rotated device
   confirmation if needed; do not repeat the entire 100-invocation sweep.
2. Repeat the same diagnosis on decode compressor ratio4 and DeepSeek RMSNorm as
   transfer candidates; inspect hc_pre afterward.
3. Obtain full stderr plus prepared input for the 23 handoff-codegen blockers;
   deduplicate by actual diagnostic. Keep admission coverage separate from timing.
4. Keep the coupled FIFO device candidate fixed. Reuse the newly qualified Qwen
   runtime after input/contract matching rather than claiming the harness is absent.
5. Let these real misses guide first/last composition, required-return sharing,
   descriptor qualification and grouping certificates. Model improvements remain
   useful, but no specific new mechanism is established by these timings alone.
