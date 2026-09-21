# GLM KDA projection: export the measured case, then confirm narrowly

## First deliverable: small artifact packet, no device required

We are investigating the full-sweep result at compiler
`2cc458cbe1e5aff6f77ffc6fea62b13686f38a94`:
`models/glm5_3_flash/kda_projection.py`, 237.3895us existing vs316.950us handoff.
Use the preserved campaign artifacts, not a new source or different shape.

The local report proves all three prepared modules are paired identically;
padding/zeroing binaries match, while the main AIC and AIV binaries differ.
Almost all additional duration appears in the dispatch/execution window.
Generated orchestration hashes differ and need a semantic comparison.

Export the existing text artifacts without rebuilding or modifying them:

```bash
tar -C /opt/pypto/oahs-full-20260921/expanded/lib-pairs/models__glm5_3_flash__kda_projection \
  -czf /opt/pypto/oahs-kda-projection-inputs.tar.gz \
  existing/ptoas handoff/ptoas \
  existing/kernels handoff/kernels \
  existing/orchestration handoff/orchestration \
  existing/kernel_config.py handoff/kernel_config.py \
  library_pair.py prepare_model.py resident_timing.py device-result.json
sha256sum /opt/pypto/oahs-kda-projection-inputs.tar.gz \
  > /opt/pypto/oahs-kda-projection-inputs.tar.gz.sha256
```

Verify contents against the full sweep's `artifact-identities.json` entry for
`models__glm5_3_flash__kda_projection`. Main prepared SHA-256:
`832a68408907b9d100d7af9bc56b3b8b363907c76ee1b65d4bbcd87826bd7e04`.
Include effective compile commands/flags if retained elsewhere. Missing artifacts
must be listed explicitly. No multi-GB inputs, framework source archive or full
campaign transfer is needed for initial local compiler diagnosis.

Compare the two orchestration sources: identify whether differences are only
paths/names or change task submission, dependencies, core placement or counts.
Preserve an exact diff and any normalized comparison rules. Do not assume their
hash difference is harmless.

## Small confirmation if device capacity is available

Keep the separate FIFO attention campaign's source and task unchanged. Use a
free device for this independent matched block; both KDA arms must share it.
Reuse frozen correctness-qualified binaries and verified inputs. Retain authentic
coupled AIC/AIV execution; do not time one half alone.

- Start with10 warmups/arm, then three alternating rounds, four samples/arm/round.
- Report same `device_wall` and `sched` windows, medians, quartiles and arm order.
- Check outputs and restore mutable state outside timing as in the original task.
- Record hashes and confirm the same shape, launch configuration and runtime.
- Do not repeat the full100-invocation sweep or repair the compiler in this task.

If task-level localization is needed, make a separate small paired capture using
the pinned runtime's documented chip-swimlane/dependency mechanism. Return
`chip_swimlane_records.json`, `deps.json`, `name_map*.json` together and logs with
zero dropped/overflow reconciliation. Use the runtime version's supported
configuration and output prefix; no invented runner flags. Compare capture
windows with that run's wall time. Capture repeats for both arms are needed
before interpreting critical-path shares. Keep profiled and unprofiled results
separate; a first-round capture is not steady-state timing.

Acceptance: report whether the slowdown persists and, if captured, which task
family lies on the longer path. Do not attribute it to a particular SET/WAIT or
fence until the exact physical requirement and a discriminating change are shown.
