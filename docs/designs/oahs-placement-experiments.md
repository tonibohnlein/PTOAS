# Placement-oriented views and isolated experiments

2026-09-20. Placement/admission milestone based on `8afb90f17`; the device
bundle captures its precommit source snapshot. Keep the ongoing
MAT device campaign pinned to its supplied source and plans.

## How the first pass guides placement

We extend the interface between existing analyses and construction. We do not
add an authoritative completion analysis or turn every physical cell into an
event channel.

| Existing view | Placement fact | Construction must establish |
| --- | --- | --- |
| `RequirementFrontiers.at(deadline)` over storage relationships | Physical cell, source access, useful publication boundary, overwrite/first-consumer deadline | Whether that boundary's actual selected prefix covers the remaining requirement |
| Original control, canonical words and occurrence indexes | Straight corridor, shared-word executions, exactly-once participation | Every emitted endpoint executes with a matching partner |
| Native cell effects and loop access sets | Input class is not regenerated, even if its producer pipe performs disjoint work | Actual readiness acquired once at the qualified first consumer |
| Selected replay cache and versioned source handles | Post-origin word-start state and post-word state, separately | Coverage and key binding use the SAME gap and ledger version |
| Existing key frontier and ledger endpoints | Occupancy, known consumption and neighboring selected uses | A legal physical binding; structural storage facts grant no event credit |

The first linked placement improvement follows this chain:

```
indexed physical requirement and publication cut
  -> current coverage at the post-origin gap
  -> virgin-key / occurrence certificate at that gap
  -> SET before unrelated incoming WAIT
  -> replay + cold check + independent forbidden-relation test
```

`SelectedSource.postOrigin` exposes the stable word-start boundary. Ordinary
post-word snapshots remain available. This first experiment supports only
acyclic, unique-word, straight-corridor sources with a virgin eligible key.
It is not general arbitrary-gap or recurring-key placement, and never sorts
all SETs ahead of WAITs. A wait supplying necessary credit must remain before
the publication that carries that credit.

## Implemented status

### 2. Admission and trial accounting — default hardening

The exact-fit witness has two independent bank episodes consuming the two
eligible keys per direction, plus unrelated input readiness before the loop.
At exact base `8afb90f17`, the linked constructor fails with four reserved
channels: `no reusable key or nonrecursive consumption acknowledgment`.
The new constructor declines that cohort and ordinary construction succeeds.

An optional candidate is checked privately before reservations or endpoints
are committed. Protocol, phase-resource, analysis-completion or diagnostic
failure rejects it. Uncovered payload requirements may remain pending. A
publish-only invalid proposal test verifies unchanged ledger version,
endpoints, reservations and contextual-replay state, then constructs normally.

Resource admission examines uncovered cross-engine demands in the staged
result. If the proposed cohort exhausts their direct eligible direction, it
declines. This is conservative direct-vocabulary admission, not a complete
proof of future allocation and not a fixed spare-key rule. Fully supported
protocols may still use the whole pool. Certified general recurring-key scope
reuse remains open.

Provider selection now computes only the next maximal provider using the old
tie rule; the next actual receipt invalidates the discarded ranking.

Mandatory proposal work has separate `proposal_sites` / `proposal_microseconds`
counters and protocol/resource rejection counts. Existing counters separately
report qualification time, replay evaluations/contextual updates, recurring
omission trials, structured entry solves, final helper trials and final cold
certificate work. Native preparation time remains separate; these counters do
not claim an exhaustive disjoint decomposition of all importer work.

### 3. Frontier motion and source gaps — explicit experiments

`--no-frontier-motion` groups only identical publication AND acquisition
frontiers in the generic coalescer. Legacy motion remains the default pending
this controlled comparison; its general ordering certificate is still open.

On Shenggan, no-motion changes 200/394/782 pairs to 330/652/1296 for 1/2/4 tiles.
Checked payload ordering is identical, named barriers stay zero and terminal
ALL stays one. Memory, rearming, ACC and forbidden-overlap checks pass. This
is a cost-of-conservative-grouping experiment, not a demonstrated overlap win.

`--source-gaps` reproduces and repairs the five-operation linked witness. Its
outward-publication variant also passes, and removing real readiness is still
rejected. The numerical native witness adds initialization/output stores around
that structure: 7 pairs stay 7; two finish-to-issue dependencies disappear with
none added. The old unrelated reader no longer gates the overwrite.

### 4. Deferred acknowledgment — restricted acyclic experiment

`--defer-acyclic-acks` requires acyclic control, a unique command word and an
exactly-once acquisition with a straight continuation to exit. It preserves empty-but-not-known-consumed key state
after the forward handoff. Later actual key reuse still invokes ordinary F7.
The one-key reuse regression checks the necessary consumption path. Optional
acquisitions and recurring cases retain the closed fallback.

The native conditional witness executes 5 versus 4 pairs; it removes three
payload dependencies on the false arm and two on the true arm, adding none.

The continuation restriction was added after a linked regression on
`2cc458cbe1e5aff6f77ffc6fea62b13686f38a94`. With one forward key, an optional
producer followed by an unconditional consumption and a later conditional
producer/consumer succeeds under the closed policy, but deferral fails with
`no reusable key or nonrecursive consumption acknowledgment`. The old
consumption is exactly once; the later branch is outside F7's straight-corridor
repair vocabulary.

Admission now uses the existing `Control::straight(current, exit)` index and
retains the immediate helper before a future branch. This adds a constant-time
control query, no new analysis state or scan. It deliberately also retains the
helper when a future branch would not reuse the key. Conditional repair and
more precise future-use admission remain open; this is not a general resource
feasibility certificate. The option remains disabled by default.

The linked test checks all four combinations of the two choices, with final
helper trials both disabled and enabled. Removing the actual reverse transfers
is rejected by the causal checker and independent graph oracle. The existing
straight-line deferral and real-reuse tests remain active. See `HANDOFF.md` for
the current validation record; no device speedup is claimed for this fix.

### 5. Access-class invariance — native admission experiment

`--class-invariant-inputs` admits disjoint producer work while retaining exact
input-cell invariance, original positive-loop bounds, unconditional consumer,
source-occurrence and participation checks. It shares the existing native
effect/loop sets and the existing core invariant-class test. This does not
relax MAT-cycle admission or support arbitrary guarded first participants.

Twelve native admission variants cover both policies and regenerated input,
guarded consumer, empty/unknown bounds and disjoint producer work. The new
numerical fixture constructs/reconstructs, passes the local graph oracle and
lowers. Default OAHS refuses it with a rearming diagnostic. Existing InsertSync
lowers, but the ordinary-core local oracle reports a missing same-pipe WAW
edge. This incomplete contract comparison is not a native wrong-code finding. Device numerical gates are required for both runnable arms;
there is no matched working-default latency comparison for this fixture.

### 6. Exact-equal-coverage binding — negative native experiment so far

The attention AIV function supplies two real equal-coverage candidate pairs.
`--equal-coverage-binding` probes exact coverage SETS in the same priority
population, retains an already helper-free baseline winner, and otherwise
uses the old deterministic tie rule among positively certified alternatives.
Unsupported reuse queries return Unknown. Ordinary positive probes require a
virgin key and exact source-time state; structured probes reuse an existing
current certificate without another solve. No ledger mutation occurs in the
probe. Binding revalidates the certificate.

Four probes on the attention AIV example produced **zero selection changes**.
All eight representative outputs remain identical. Keep the option disabled;
find a positive discriminating binding witness before expanding admission or
spending device time. No plan-quality benefit is claimed.

## Reproducible variants and evidence

The diagnostic native driver accepts flags after `--construct INPUT`:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--no-recurring-trials` | trials on | Isolate changed-plan recurring omissions |
| `--no-helper-trials` | trials on | Isolate final engine-pair helper pruning |
| both flags | both on | Restricted constructor without deletion trials; mandatory checks remain |
| `--no-frontier-motion` | legacy motion | No-motion grouping control |
| `--source-gaps` | off | Exact word-start publication experiment |
| `--defer-acyclic-acks` | off | Real key-deadline acknowledgment experiment |
| `--class-invariant-inputs` | off | Disjoint-producer first-consumer qualification |
| `--equal-coverage-binding` | off | Conservative read-only resource tie probe |

No flag disables final certification or emitted-plan reconstruction.

Local results:

- 23/23 portable suites, including 200 deterministic small programs with the
  gap/deferred/probe options and independent memory/event checks.
- Native selected tests pass, including the expanded first-consumer negatives.
- Default hardening: **88/88** corpus inputs construct/reconstruct with
  **88 byte-identical outputs** relative to 8afb.
- Initial matrix: 144 successful host cases across default and isolated
  options. The equal-coverage option was added afterward; a final 16-case
  representative default/probe matrix passes with unchanged outputs.
- Eight representative kernels show no changes from gap/deferred/invariance
  options. Use the native microkernels to measure those mechanisms.
- No-recurring-trials changes none of the eight samples. No-helper-trials
  changes post-RMSNorm and partial attention; the both-disabled outputs match
  no-helper-trials there. This is sample evidence, not a global equivalence.
- Eight runnable microkernel arms pass construction and C++ lowering. The
  ninth, default class-invariant fixture, is the explicit refusal above.
- Default GEMM retains the committed 200/394/782 oracle and zero named barriers.

The host logs include elapsed times but are not controlled compiler speed
benchmarks. FIFO AIV remains expensive: the sampled run reports 813,458 replay
evaluations and 125 contextual updates. New proposal validation accounts for
8,969 evaluations separately. **FIFO replay has not been optimized here.**

See [targeted device tasks](../../test/benchmarks/placement/DEVICE_TASKS.md).
No device measurement of these experiments or sanitizer result is claimed.
The separate earlier MAT down_proj feedback is recorded in the MAT-cycle note.
