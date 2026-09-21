# GLM KDA projection: initial attribution

2026-09-21; current local compiler `21f95f9b7`. Analysis of the frozen
`2cc458cbe` campaign only; no compiler changes or new device runs.

## What the saved records establish

Fixture: `models/glm5_3_flash/kda_projection.py`, input X `[67,4096]` bf16,
Q/K/V weights each `[512,4096]`; additional low-rank and gate projections.
The paired entry ran on card0, job `task_20260921_104235_18236833798`.
Both arms pass seeds7/23 with identical input and golden hashes. The full
100-invocation sequences use existing then handoff, with the first50 discarded.

All three corresponding prepared modules have identical recorded hashes:

| Prepared module | SHA-256 |
| --- | --- |
| kda_projection | 832a68408907b9d100d7af9bc56b3b8b363907c76ee1b65d4bbcd87826bd7e04 |
| kda_projection_b_zero | 9de9b5406dc2b5505d4767e2d053e9b255807f8f81e71f835c9936cca591cd1a |
| kda_projection_b_pad | 2c9fb546e6ef2030115092abd6f81a0d2d0a98da5b4797ae1b44418c665b1754 |

The four device binary records narrow the changed code:

| Binary | Existing vs handoff |
| --- | --- |
| kda_projection_b_pad (AIV) | identical |
| kda_projection_b_zero (AIV) | identical |
| kda_projection_aic | different |
| kda_projection_aiv | different |

Kernel configuration hashes agree. Generated orchestration source hashes differ;
without their text, this cannot be dismissed as a path/comment difference or
certified as the same runtime task graph. Upstream pass dumps also differ in
hash, although the final prepared modules agree. Compare actual content first.
These are archived hash records, not locally rehashed binaries.

## Runtime-window attribution

Recomputed from the final50 invocations of each arm, converting recorded
nanosecond span durations to microseconds. All200 recorded full device-wall
spans equal their corresponding reported timing samples exactly.

| Span / per-run derived quantity | Existing median us | Handoff median us |
| --- | ---: | ---: |
| Whole device run | 237.3895 | 316.950 |
| Scheduler dispatch/execution window | 202.880 | 281.390 |
| Orchestrator window | 3.910 | 3.870 |
| Preamble | 6.920 | 6.960 |
| Post-orchestration | 11.209 | 11.219 |
| Whole device run minus scheduler window | 35.430 | 35.839 |

The increased scheduler window is consistent with almost all of the overall
~79.56us increase. It does not identify a particular task, pipeline stall or
synchronization endpoint. Spans overlap; do not add these medians or interpret
`graph_build` as compiler time. The per-run complement is calculated before
its median, not by subtracting median durations.

The local copies of the runtime timing guide, device-phase guide, chip-swimlane
guide and onboard span-export source match the campaign source-manifest hashes.
They define `sched` as the dispatch/execution window, not pure scheduler CPU
work. No per-task swimlane, dependency graph or name-map capture is in this
report bundle; a critical-path breakdown would be unsupported.

This is stronger localization than whole-entry latency but still permits main
AIC cost, main AIV cost, coupled backpressure or changed dispatch behavior.
The unchanged padding/zero binaries rule out changes to their instructions;
they do not prove unchanged scheduling of those tasks.

## Required next evidence

The report-only archive contains no prepared PTO or generated source. A local
filename search found no alternate KDA source/plan bundle; the pinned pypto-lib
object is also unavailable in the older local checkout. Do not regenerate a
different source and present it as the measured kernel.

[Small device artifact request](../../test/benchmarks/kda_projection/DEVICE_TASK.md)
collects the three prepared modules, both plans/generated sources and runtime
orchestration/harness. No new device work is needed to export them. Compare
orchestration semantics first, reconstruct the same prepared input at current
head, then attribute blocking acquisitions/fences to physical lifetimes on both
AIC and AIV. Use matched per-task capture only if code inspection cannot select
the responsible component. Keep the authentic coupled path throughout.

Reproducible local analysis: `../kda-projection-work/analyze_report.py`;
structured results: `../kda-projection-work/analysis.json`. These require the
extracted report bundle under `../full-sweep-review-20260921/`.
