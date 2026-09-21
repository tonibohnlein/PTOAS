# KDA pipeline investigation, 2026-09-21

**Latest status supersedes the historical next steps below:** native first-write
qualification is now a WIP implementation. KDA constructs/reconstructs, but a
native outward-publication regression fails and the local order diagnostic
adds serialization. Read [the complete first-write handoff](oahs-kda-first-write-handoff.md)
for exact code, failures, scope, hashes and continuation steps. No device
speedup is claimed. This branch is being pushed as a requested research checkpoint.

This is an attribution milestone, not an implemented compiler fix or a new
device speedup claim. Host attribution below was performed at
`a0c1d761e7624b42a0f494853cc70eecaabe9c44`. Development has since rebased onto
`bec7dfe37e7c6a6bcde4860ba878f65ab1b1abae`, without rebuilding or retiming. The joint-reader agent's
baseline/candidate device libraries were frozen before updating this worktree.

## Established measurement and source boundary

The controlled original-runtime comparison measures compiler `2cc458cbe`, not
the latest compiler. Existing median is 241.810 us and handoff is 318.160 us
(ratio 1.31574). The identical-existing control is 238.9995 us. All three
alternating rounds reproduce the large regression; correctness gates pass.
Device scheduler-window medians are 205.460 versus 283.260 us. This localizes
the difference to task dispatch/execution, not individual instructions/pipes.

The exact prepared main input SHA-256 is
`832a68408907b9d100d7af9bc56b3b8b363907c76ee1b65d4bbcd87826bd7e04`.
Both arms use the same input. Pad/zero binaries match. The main AIC/AIV pair
changes. Orchestration becomes byte-identical after normalizing generated
`inline` numeric suffixes, and launch configuration matches. The full coupled
entry has five AIC and ten AIV main tasks plus pad/zero tasks; no independently
timed attention or projection half substitutes for that entry.

At a0c1d761e, default construction and reconstruction pass for the exact input.
Selected PTO is `c73b9089f4a3895f38efbc6bdc137037d28eeb51b348c2badd5039bcf3de5308`.
Its placement differs from frozen handoff, so the old device ratio must not be
presented as a measurement of latest HEAD.

Two host-only ablations on a0c1d761e also construct/reconstruct successfully:
`--class-invariant-inputs` and `--no-frontier-motion`. Both emit byte-identical
PTO to that revision's default (SHA-256 above). Neither existing switch addresses this
case. No experimental option has been promoted or device-timed.

## Executed instruction population

An uncommitted local `count_trace.py` diagnostic interprets the original scalar
loops/conditions, rejecting unsupported scalar control. This is a count-only
diagnostic, not a memory/queue oracle or latency model. Queue operations are
opaque; instructions internal to their lowering are excluded. Scalar bindings
come from the original wrapper: arguments 16/17/18 are 67, block index 0,
block count 5. Counts below are per AIC block invocation.

| Explicit operation | Existing | Frozen handoff | a0 default |
| --- | ---: | ---: | ---: |
| MTE2 barrier | 0 | 67 | 67 |
| MTE1 barrier | 100 | 196 | 196 |
| M barrier | 392 | 196 | 196 |
| FIX barrier | 4 | 4 | 4 |
| FIX → M SET/WAIT pairs | 4 | 76 | 76 |
| M → FIX SET/WAIT pairs | 7 | 77 | 77 |
| All SET/WAIT pairs | 1,724 | 1,453 | 1,453 |
| All barriers, including terminal ALL | 497 | 464 | 464 |

Payload populations match: 319 loads, 784 extractions, 488 matrix operations,
128 moves, two inserts and six queue pushes. Fewer synchronization instructions
do not establish better overlap. The eleven static MTE2 barriers expand to 67
executions, including four sites in 15-iteration hot loops.

## Concrete compiler leads

The uncommitted read-only `--explain` diagnostic uses existing imported effects, Control,
selected decisions and fence residuals; it introduces no completion state.

1. **Mixed prologue/loop/epilogue physical generations.** The MTE2 fences report
   outstanding same-engine write/write obligations, mostly for MAT cells
   47–49. `qualifyReaderRegionCycles` rejects a cell if *any* of its writers is
   outside a cyclic component. Those cells have actual acyclic prologue/tail
   writers; the diagnostic records their operation/site IDs. Thus this global
   condition excludes them before the complete reader-cycle placement logic.
   This proves a qualification boundary, not that relaxing it alone is safe or
   sufficient. Other boundaries include uncovered reads outside leaf reader
   children and retained-cohort producer support.
2. **Repeated external-reader completion repair.** FIX → M decisions explicitly
   cite old FIX reads of ACC cells 66/67 before subsequent matrix writes. Many
   are common-cut exchanges in later matrix loops, with reverse handshakes for
   key consumption. The source phase includes the preceding queue push.
   Determine why a narrow first-conflicting-consumer receipt is unavailable,
   and whether earlier construction decisions become redundant later. This is
   separate from the MAT reuse problem.
3. **Keep outward ordering.** Existing loop-entry placement rejects moving an
   acquisition across observer publications. Native first-input observations
   qualify invariant reads, not arbitrary writes conflicting with an invariant
   external reader. A broad entry wait could delay operand-release publications
   and would not meet the maximum-parallelism objective merely by passing safety.

The next regression should express physical generation episodes crossing a
one-shot producer, repeated reader children, hot repeated producers, and a
one-shot tail. A second linked witness should expose first conflicting writes
after an external read, with an outward publication before the true deadline.
Use full payload-order comparisons and missing-support negatives. Do not remove
fences by name, waive producer support, assume key rearming, or recognize KDA.

## Profiling limit

The warmed combined swimlane/PMU attempt failed on its sixth invocation during
warmup: runtime finalization error 507018 and unflushed collector buffers.
Five completed warmup launches passed numerical checks. No measured samples
were reached. That generic AICPU error does not by itself prove kernel deadlock
or explain the regression. Earlier four cold captures vary substantially and
also cannot support latency attribution. Pipe-level attribution remains open;
do not turn these captures into an instruction-stall claim.

## Reproducibility and validation

External artifact directories: `oahs-kda-fix-20260921/`; controlled timing:
`oahs-methodcheck-20260921/glm/`. The former contains exact a0
plans/logs, original and latest diagnostic output, count JSONs and preserved
profiling failure. Scalar count sanity checks cover a two-visit conditional
loop and unsupported-operation rejection; lint passes. The unchanged native
suite on the pre-update baseline passed. No new full portable/corpus run or
production fix is claimed at this milestone.

Compact [raw launch records](device-feedback-20260921/kda-records.json) and
[measurement summaries](device-feedback-20260921/method-summary.json) are committed.
The diagnostic code and full profiling artifacts are deliberately not part of
this feedback-only commit. See [device feedback](oahs-device-feedback-20260921.md)
for launch counts, measurement limits and the other device tasks.

## Follow-up: linked mechanism tests (local, not promoted)

Worktree base is `a513ffe30` (feedback on `bec7dfe37`). Rebuilding the current
native compiler reproduces the a0 KDA plan byte-for-byte, including both AIC/AIV
functions. Construction and reconstruction pass. This closes the earlier
unrebuilt-checkout boundary; it is not a new device measurement.

### Mixed producer qualification is real but insufficient for KDA

`mixedProducerReaderRegions` starts with an acyclic producer/reader child,
then repeated producer/reader-child episodes, then an acyclic tail episode.
The original constructor misses the complete two-direction reader-region
cycle. A local prototype changes the all-writers-cyclic requirement to require
at least one cyclic writer, retaining complete reader coverage, physical role
checks, exact participation, staged support and final cold validation.
Purely acyclic inputs retain their ordinary entry placement. Removing the
cyclic restriction entirely disturbed that existing placement, so that broader
experiment was not retained.

The narrowed prototype selects two channels without a named pipe fence.
Nine finite paths vary outer iterations and child reads over 0/1/3. Independent
graph checking validates memory, event balance and rearming; the complete
payload-order sets are equal to ordinary construction (zero removed or added).
Deleting either channel fails cold checking. An independent tail reader or a
read outside the qualified children declines specialization. Prefix-only replay
and normal sibling replay emit identical words.

Crucially, the exact KDA input still emits the same SHA-256
`c73b9089f4a3895f38efbc6bdc137037d28eeb51b348c2badd5039bcf3de5308`.
The prototype therefore is **not a KDA fix** and has no device benefit claim.
KDA's one-shot reads and unqualified first-reader boundaries remain additional
limits. Do not remove those safeguards just to make the case qualify.

### First conflicting write, with outward publication

`firstConflictingWriterAfterPublication` is a linked constructor witness, not
a supplied synchronization plan or kernel-name recognizer. An external reader
is followed by an observer loop that exports completion to a third engine
before writing the conflicting cell. With ordinary observations the constructor
correctly declines a broad loop-entry receipt. A test-supplied original first-
visit prefix permits the existing constructor to acquire once at the first
conflicting write, with no repeated receipt at subsequent writes.

Independent graph checks for 1/2/5 visits compare complete payload-order sets
and forbid the external reader from gating the earlier outward consumer.
Missing acquisition fails cold checking. Moving the same acquisition to entry
remains memory/protocol-safe but introduces the forbidden outward order.
Thus safety alone is demonstrably insufficient, while the existing causal state
can represent the desired protocol once suitable occurrence boundaries exist.
This is a portable qualification witness, not proof of native KDA improvement.

Native observation inspection explains the next implementation boundary:
`importFirstConsumers` currently selects invariant **reads** of cells not
written inside the loop. That does not expose first writes conflicting with an
external invariant reader. The KDA report has six qualified first-input sites,
all MAT/MTE1 reads; it has no first-write counterpart and reports zero loop-entry
transfers. Some first matrix writes also sit behind original choices. A blanket
entry hoist would cross useful publications and is not an acceptable fallback.

Next concrete action: extend the shared original-control/physical-conflict view
to expose exactly-once first-conflicting-write frontiers, including alternative
first writes, without moving earlier publications. Retain source-class
invariance, source-time receipts, participation and key rearming. Native tests
must reject missing/ambiguous observations and compare full emitted order,
then the unchanged KDA input must actually produce a discriminating plan before
another device experiment. No new authoritative completion state is needed.

Validation of the local prototype and both regressions: 24/24 portable suites,
the native selected-test executable, and exact KDA construction/reconstruction
pass. The 88-input corpus has not been rerun; no production promotion, commit,
push or device run was performed for this follow-up. Logs and plans are in the
development worktree's `build-kda/` (`portable/final-ctest.log`,
`native-tests.log`, `kda-observations.txt`, `kda-bec7.plan.pto`,
`kda-mixed-final.plan.pto` and corresponding construction logs).
