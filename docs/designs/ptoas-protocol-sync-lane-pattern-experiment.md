# ProtocolSync lane-pattern experiment

## Status and scope

This document records the read-only experiment following Checkpoint C.5. It
does not change the ProtocolSync planner, candidate selection, materialization,
verification, or fallback behavior. Every pattern record is emitted with
`selectable=no`.

The experiment answers three narrower questions:

1. Can execution-lane structure recover useful synchronization shapes after a
   strict storage or channel recognizer rejects a function?
2. Do the recovered shapes reproduce placements selected by the earlier
   CanonicalSync implementation?
3. Which facts must remain target-specific rather than being inferred from
   lane order?

Stable/alternating L1 and accumulator protocols are deliberately excluded.
Those require repaired storage identity, core assignment, and iteration facts
before their recognition rules can be stated soundly.

The [experiment and related-work evidence
ledger](ptoas-protocol-sync-evidence-ledger.md) places these results beside the
C.7 storage-track measurements, records the complete claim/limitation matrix,
and compares the design with primary compiler and research sources. This
document remains the detailed C.6 construction and measurement record.

## Diagnostic model

Storage analysis now retains every exact physical-access endpoint even when it
rejects the generation timeline. The lane-pattern analysis forms RAW, WAR, and
WAW pairs only when both endpoints have known, overlapping physical intervals
and a linear order in the same block. Cross-block and loop-carried ordering is
not invented by this experiment.

The following read-only recognizers consume either the Checkpoint C.5
frontiers or the raw access pairs:

| Diagnostic kind | Required structure | Earlier CanonicalSync analogue | Cost recorded |
|---|---|---|---|
| `shared-one-shot-frontier` | At least two compatible, cross-lane one-shot ready windows with a common cut | shared event-frontier synthesis, or a direct event already interned at the same cut | one logical candidate, two steady-state actions |
| `same-lane-completion-cut` | Linear same-lane hazard intervals for which the target requires the same pipe barrier; greedy earliest-end interval stabbing finds each common cut | barrier coverage/coalescing (reported as `multi-demand-pipe-barrier`) | one logical candidate and one action per cut |
| `choice-balanced-round-trip` | A loop lifecycle whose ready demands cover every arm at choice entry and whose release demands balance at the same choice exit | lifted choice ready/release recurring event | one logical candidate, four steady-state actions |

The earlier implementation did not name multi-demand barrier coverage as an
independent pattern. That label is an experiment-side description of its
barrier coverage/coalescing behavior, not a claim about an old public enum.

Each candidate records its target-query result, old pattern and mechanism kind,
Checkpoint E admission status and rejection, logical cost, steady-state action
cost, and an operation-name placement used for the old-branch comparison. Raw
pairs separately record the same-lane completion query.

## Target boundary

Lane membership proves neither pipeline completion nor synchronization
legality. The A3 target model classifies each same-lane raw hazard as:

- `intrinsic` for the currently modeled MTE2 `tload`-to-`tload` WAW case;
- `pipe-barrier` when that lane has a supported same-pipe barrier;
- `unsupported-target` or `unsupported-mechanism`; or
- `not-applicable` for a cross-lane pair.

Only `pipe-barrier` pairs may enter `same-lane-completion-cut`. This is why the
old shared-MTE2-barrier fixture now produces two intrinsic raw pairs and no
barrier candidate. The rule agrees with the current `ProtocolSyncTarget` and
InsertSync exemption; it is not, by itself, an ISA or device qualification.

The event and ready/release target queries similarly report feasibility only.
No candidate becomes selectable in this experiment.

## Frozen-corpus method

The corpus run used the 394 A3 rows from
`canonical-frontier-394-a3-19674e06-final.tar.gz`. The archive contains the old
branch's selected mechanism dumps. The current compiler was run in analysis
mode with lane-frontier diagnostics and JSON statistics. The disk-backed local
runner used at most two workers.

The run was repeated after rebasing onto the E/F base `51fa6e338` on
2026-09-05. All semantic counts below were identical to the earlier
`4975d7523` run; aggregate lane-pattern time changed from 205,892 to 187,178
microseconds.

The worktree was later rebased through `7a09ed458`, including `faeb8e71d`, and
host-validated, but this 394-row lane-pattern census has not yet been rerun. The
counts below remain versioned `51fa6e338` evidence.

```text
ptoas --pto-arch=a3 --pto-level=level3 --emit-pto-ir \
  --enable-insert-sync --protocol-sync-analysis-only \
  --protocol-sync-dump=lane-frontiers --protocol-sync-statistics \
  INPUT.pto -o /dev/null
```

The comparison keys are resource, mechanism kind, and operation-name
placement. The old dump did not preserve occurrence-stable program-point IDs,
so a placement-name match is supporting differential evidence, not an identity
proof when operation names repeat.

## Corpus results

All 394 rows completed without a compiler failure or timeout. They contained
405 functions.

| Measure | Result |
|---|---:|
| Raw endpoints retained | 18,151 |
| Raw hazardous pairs | 66,839 |
| Raw pairs: intrinsic completion | 470 |
| Raw pairs: pipe barrier completion | 51,261 |
| Raw pairs: cross-lane / not applicable | 15,108 |
| Raw pairs: unsupported target or mechanism | 0 |
| Pattern candidates | 1,050 |
| Total logical cost | 1,050 |
| Total steady-state action cost | 1,096 |
| Aggregate lane-pattern analysis time | 187,178 microseconds |

| Candidate kind | Count | Target query | Checkpoint E | Old selected placement comparison |
|---|---:|---|---|---|
| Shared one-shot frontier | 22 | 22 supported | 22 not applicable | all 22 matched: 10 old shared-frontier syntheses and 12 direct events |
| Same-lane completion cut | 1,020 | 1,020 supported | 1,020 not applicable | 507 matched selected barriers; 513 had no selected name-level match |
| Choice-balanced round trip | 8 | 8 supported | 8 rejected: multiple consumers | no selected old placement in the corresponding fail-closed rows |

Candidate target queries are all supported because the same-lane recognizer
admits only raw pairs already classified as barrier-requiring. The zero
unsupported candidate count therefore does not imply universal A3 support; the
raw-pair classification is the relevant denominator.

All 481 same-lane candidates belonging to rows that the old branch admitted
matched an old selected barrier placement. Another 26 matches occurred in rows
where the old all-or-nothing plan later failed for another function or demand.
Most of the 513 unmatched candidates are consequently observations recovered
inside old fail-closed rows, not direct disagreements with a successful old
plan.

The eight choice-balanced candidates occur in four corpus rows that the old
branch ultimately rejected for a later single-lane recurring demand. They show
that the local choice shape survives that row-level failure. They do not prove
that the entire function has a complete recurring protocol.

## Earlier-fixture results

The old shared-event fixture reproduced the selected
`after(tload)->before(tmuls)` event placement. Its disjoint-window negative did
not form a shared candidate. The old shared-MTE2-barrier fixture instead
reported intrinsic completion for both WAW pairs, correcting the earlier
prototype's unnecessary barrier under the current target model.

The lifted-choice fixture reproduced the ready placement at the choice entry
and the balanced release at the choice exit, while reporting Checkpoint E's
`multiple-consumers` rejection. The earlier L0 ownership fixture exposed four
of the same choice-balanced shapes. Guarded/boundary recurrence, L1 parity and
path coverage, and accumulator RAR formed no candidate, as intended. The old
MMAD accumulator fixture is no longer accepted by the current verifier because
its A5 lhs layout is not `col_major`; it remains a deferred compatibility case.

## Conclusions and next steps

The experiment supports a hybrid architecture:

- storage timelines and Checkpoint E remain the high-confidence recognizers for
  complete recurring ownership protocols;
- execution lanes and frontiers provide a useful structural discovery layer,
  especially across branches and after strict channel rejection;
- raw access pairs are required for residual same-lane hazards that disappear
  before timeline admission; and
- target queries, not graph or lane order, decide intrinsic completion and
  available synchronization mechanisms.

The evidence is strong enough to keep the three recognizers as diagnostics and
to use them as differential checks while ProtocolSync grows. It is not strong
enough to turn the lane layer into a complete synchronization pass. Before any
recognizer becomes selectable it still needs a complete obligation proof,
target legality, atomic selection, independent verification, and device
qualification.

Stable/alternating L1 and accumulator work should resume only after physical
storage identity, core assignment, loop-carried ordering, and iteration
distance are authoritative. Visibility/GM fences, cross-core events, fixed
fences, and exit/tail synchronization are also outside this experiment and
need separate demand types and target contracts.
