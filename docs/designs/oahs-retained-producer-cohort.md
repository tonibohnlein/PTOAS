# Retained inputs with complete producer support

Local follow-up to the uncommitted retained-reader snapshot on `d6e9b5365`,
2026-09-21. No device measurement is claimed.

## What changed

The reader-region qualifier previously admitted a generation spanning children
only when its producer wrote one physical cell. This extension admits several
written cells when their individually qualified cycles form a closed producer
cohort. It does not allocate a generic private channel for every discovered cell.

The first-pass view reuses original cell incidences, reader-region boundaries,
nearest-role propagation, and straight-corridor facts. Candidate records retain
separate readiness publications, first consumers, last readers, and overwrite
acquisitions. Each producer cohort is checked once through a cell index.

The initial multi-input scope requires one writer occurrence per cell, all those
writers in one original straight corridor, and the same reader engine. Every
producer-written cell must have a qualified complete candidate. Unknown cells,
exclusive effects, independent reader engines, unqualified participation, and
reloads do not gain broader admission. Existing single-generation protocols
retain their previous rules.

An internal `repairFreeProducers` premise accompanies the extended proposal.
The existing staged mandatory solve must show **no remaining payload requirement
on those producer engines** before the proposal commits. The solve observes only
actual selected commands. A safe event protocol with an uncovered producer
requirement declines transactionally, with `rejected_support` reported separately
from protocol and capacity rejection. Missing reader-side requirements remain
ordinary obligations. No extra full solve, construction mode, key reset, or
runtime guard is introduced.

## Why this addresses the counterexample

In `P: write X; P: write Y`, selecting X's complete cycle alone can remove the
fence before X and leave a Y fence after the new X write. That would add
`finish(new X) -> issue(new Y)`. The old counterexample still declines: an unread
Y has no qualified cycle. Even after structural qualification, the precommit
check refuses a cohort whose incoming or recurring producer requirements remain
uncovered. It never repairs this situation by moving a fence speculatively.

For an admitted cohort, the actual returns discharge producer requirements at
their own original deadlines. There is no remaining producer fence to relocate.
This certificate addresses that repair-motion mechanism; it is not a theorem
that the whole constructor produces a globally least-order plan. Source-word
motion and unrelated coalescing remain separate work.

## Evidence

- All 23 portable suites and the native diagnostic suite pass.
- Sixteen multi-input traces, including empty/short children and different
  lengths on successive parent entries, remove 164 full payload launch/finish
  relations against ordinary construction, with none added. The original eight
  single-input traces still remove 62 relations.
- Separate publications, first consumers, and last-reader frontiers remain.
  Missing readiness/necessary return, premature release, actual reload, and
  independent-reader negatives pass. A missing earlier Y return can be safe
  when X's actual later return covers it; both independent checkers agree. This
  extension does not implement that optional channel omission.
- Capacity rejection leaves ordinary construction's outcome unchanged. For the
  deliberately tight multi-input fixture, ordinary construction itself refuses;
  this is not presented as a successful small-pool allocator. A separate direct
  support-rejection test proves unchanged ledger version, endpoints, reservations,
  and replay state, followed by successful ordinary construction.
- Native A3 `oahs_retained_multi_children.pto` constructs and reconstructs. Its
  two-parent execution has 22 payloads and 251 local conflict checks. Full order
  shrinks 926 -> 910 (16 removed, zero added); executed pairs grow 6 -> 10.
  All five V barriers remain, and the producer MTE2 repair is discharged.
- All 88 corpus modules construct/reconstruct; every plan is byte-identical to
  the frozen retained-reader baseline. This includes GEMM, attention, projections
  and post-RMSNorm. There is no new corpus device-speedup claim.

The native relation comparison uses the independent scalar/command interpreter
and declared dense local effects, including both inputs of each vector add.
It does not certify GM visibility, numerics, or device execution.

## Work accounting

The nearest-role work is unchanged: two constant-height graph traversals per
eligible cell, plus existing qualification. The added writer/candidate indexes
and one scan per producer cohort do not form a bank-period or branch product.
Ordered-map operations cost logarithmic lookup; straight-corridor tests use the
existing constant-time frame/position index. There is one additional scan of
staged residuals, inside the already required proposal check, not another solve.

On the native witness, proposal checking visits 88 sites. Selected replay falls
614 -> 480; final checking remains 80 sites. Recurring omission and final helper
trials are both zero for this witness. Qualification, proposal checking, selected
replay, structured entry solves, and final checks remain separately reported;
these counts are not an end-to-end runtime claim.

## Device follow-up and next boundary

Two observable A3 microkernels use distinct GM data for every parent tile and
retain both local inputs across reader children. Their exact integer-valued FP32
oracles avoid expensive reference computation. Baseline and candidate both lower.
The smaller case removes 12 full payload relations (10 -> 14 executed pairs);
the larger removes 48 (20 -> 26 pairs). Neither adds a relation. A dedicated task
and source-complete bundle are prepared; neither has been dispatched here.

Artifacts: `../retained-multi-work/` relative to the workspace repository,
including baseline source hashes, paired native plans, portable/native logs,
ordering reports, and the serial corpus reproducer. The dispatched overnight
corpus bundle is unchanged.

Next: find an actual corpus input with this retained multi-child lifetime. Broader
admission should start from a concrete refused requirement, especially an incoming
producer dependency or non-straight writer correspondence. Keep release sharing,
within-word placement, and scoped key reuse separate until each has its own
ordering witness.
