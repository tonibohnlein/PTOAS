# Placement and optional-protocol review: revised plan

Assessment: 2026-09-20, base implementation `8afb90f17`.

**Follow-up implemented:** see [placement experiments](oahs-placement-experiments.md)
for current status. The assessment below records the pre-implementation review.
The supplied reviews inspect `fe1fc454b` against draft v0.33. The local paper
now identifies itself as v0.34; its F7/F8 and resource-selection sections retain
the relevant distinctions. This assessment re-reads current source and those
draft sections. It does not execute the reviewers' linked archives, reproduce
their native witnesses, run the compiler, or claim new device results.

## Disposition

Keep the shared causal frontier and two-phase storage-analysis/construction
architecture. The reviews refine placement, admission and evaluation; they do
not establish that the frontier is fundamentally wrong or that a globally
least-order plan is attainable. First-pass facts guide selection, while actual
selected commands establish completion and consumption knowledge.

| Finding | Current-source assessment | Action |
| --- | --- | --- |
| A: cut-only source handles | Confirmed representation limitation. `SelectedSource` has a cut/version but no stable word gap; `refreshSources()` uses the post-word `before` state and ordinary `edge()` appends its SET. | Reproduce the five-payload example with the linked constructor, then add exact-gap coverage and binding. Current projection cost is not attributed to it yet. |
| B: immediate common acknowledgment | `needsCommonAcknowledgment()` still returns true on `mayIssueAfter(current)`. This is a conservative fallback allowed by the draft, not unsafe output. | Separate acyclic experiment using the next selected key republication deadline. |
| C: optional rejection | Initial resource failure declines safely. An invalid initial omission-analysis candidate breaks the trial loop but can still proceed to reservation/append. | Stage mandatory protocol checks transactionally; reject the complete optional state before commitment. |
| D: trial-enhanced constructor | Non-qualified recurring omission analyses and final engine-pair helper deletion checks remain. Qualified MAT cycles skip the former, not necessarily the latter. | Expose independent diagnostic switches and charge all changed-plan checks. Direct-without-trials is still a restricted implementation of Algorithm 1. |
| E: common-frontier motion | `joinedCycle` is removed, but generic `combine()` still moves one frontier when the other matches, using occurrence/straightness tests. No general contextual order certificate appears there. | Establish a no-motion grouping baseline; retain moving cases only with a specific certificate. |
| F: global recurring reservation | `recurringKeys` remain excluded by ordinary allocation; entry-protocol reuse is narrower. | First reproduce exact-fit starvation, separately from failed-proposal rollback. Later add certified scope reuse only if needed. |

The latest MAT change already addresses part of the cross-child lifecycle
recommendation: selected readiness and return cycles preserve physical reader
boundaries before repair. It does not fix ordinary within-word placement,
optional rollback, the global reservation policy, or the remaining generic
frontier merge. Device latency of this MAT candidate remains pending.

## Ordered work

### 0. Keep the current device experiment pinned

Keep the four-arm MAT task: existing, fe1 baseline, placement control retaining
MTE2 fences, and production snapshot corresponding to the MAT implementation.
Do not silently replace that snapshot with any following hardening. Reconcile
profiles using the same binary, launch arguments and warm-up. Down is the first
measurement, LM the transfer check. Current host counts do not predict latency.

### 1. Admission hardening and explicit evaluation variants

- Reproduce two distinct linked cases: an initially invalid optional protocol;
  and a valid, exactly-fitting bank cohort that starves an unrelated ordinary
  requirement. A protocol-valid proposal need not be resource-admissible for
  the rest of construction.
- Stage endpoints, reservations, support records and mandatory protocol checks
  privately. Remaining payload requirements may remain explicitly pending.
  Rejection must resume ordinary construction with unchanged semantic state;
  only work/failure diagnostics may change. Do not add subset search.
- Do not reserve an arbitrary spare key per direction. A fully supported
  protocol may legitimately occupy the pool. Admission must account for the
  uncovered requirements, or qualify actual inactive-key reuse.
- Add independent diagnostic controls for recurring omission and final helper
  pruning; keep the current default and device arm fixed during comparison.
  Disabling trials must retain qualification, final checks and reconstruction.
  Label any failure honestly; the direct variant has no new completeness claim.
- Retain the small next-provider correction as a separate, behavior-preserving
  task: select only the next maximal provider with existing ties instead of
  sorting the whole population discarded after the next receipt. This is NOT
  the later equal-coverage resource-aware policy experiment.

### 2. Protect exact endpoint positions

First narrow the remaining uncertified common-frontier motion. Identical
publication AND acquisition frontiers can group without motion; sharing just
one frontier requires an additional contextual argument. Test crossed outward
publications and physical-key effects, not only the immediate consumer.
Measure extra key pressure/fallbacks after narrowing; fewer commands are not
an acceptance criterion.

The next parallelism mechanism is exact within-word source placement:

- Add the linked five-payload R-read-y / P-write-x / Q-copy-x / Q-write-y /
  P-overwrite-x fixture. Assert that R's read does not gate P's overwrite.
- Preserve a source as cut plus stable endpoint-relative gap, engine,
  occurrence/guard interpretation and premise version. A numeric offset alone
  is unstable under insertions. Use existing endpoint identities and replay
  snapshots where possible; do not create a competing causal state.
- Query coverage and bind the key at that SAME gap, with matching/rearming and
  neighboring-use checks. Also identify the target position. Never reuse the
  broader end-of-word snapshot to justify the earlier publication.
- Start with release before an unrelated incoming WAIT. Add the negative in
  which the WAIT supplies necessary completion or key-rearming evidence, so
  the SET must stay after it. No global SET-before-WAIT sorting.
- Include an outward publication between endpoints, shared observations and
  source-word edits. Compare incremental replay with a cold evaluation of the
  identical ledger and with the independent finite graph oracle.

### 3. Rearm at actual physical-key deadlines

Start with acyclic, one-shot common-cut handoffs. Later payloads alone must not
force an acknowledgment. An empty key with consumption unknown to its publisher
may remain pending until its next selected publication. At that deadline use
actual return credit, an eligible reusable/virgin key, or a qualified helper.
Keep the closed recurring fallback until its boundary interface is supported.
Test no-republication, real reuse with no return, natural-return reuse, and
outward-publication/key-neighbor mutations. This is prospective construction,
not blanket helper deletion.

### 4. Generalize existing lifecycle facts without expanding history

Expose a shared, inspectable VIEW over existing access/occurrence, boundary and
requirement indexes: physical generation, separate last readers per engine,
first consumer/next overwrite, useful source boundaries, participation and
empty cases, and conditional return support. Do not eagerly materialize a
cross-product record or allocate one channel for every discovered relation.

Admit class-invariant first-consumer corridors with disjoint producer work in
one small case; retain source-occurrence and exactly-once participation proofs.
A guard with no expressible first participating use must remain conservative.
A child exit is only a useful release boundary when its actual last-use and
participation facts justify it; it is not the definition of storage lifetime.
Use current projection, paired-child and attention examples as transfer tests.

### 5. Exact-equal-coverage resource probe (separate policy experiment)

First exhibit two providers generated by the real constructor with the same
nonempty certified residual SET. Equal sizes are insufficient. A read-only
probe checks the exact source/target gaps, current eligibility, both neighboring
uses, matching, rearming and premise versions. Failure is Unknown, not proof
of impossibility. Prefer a certified helper-free provider within that same
coverage/priority class; otherwise retain existing ties. No changed-ledger
solve in a supposedly read-only probe. Equal coverage does not ensure equal
incidental ordering, so measure ordering effects separately.

## Evidence and work accounting across these tasks

For every plan-quality claim retain: first-pass fact -> constructor decision ->
actual endpoints -> checked relation-set difference. Record the first selected
path introducing each targeted unnecessary relation, with physical source/use,
key generation, cut AND word gap. Keep this diagnostic lazy and targeted; do
not enumerate all payload reachability during production construction.

Require set inclusion, not merely fewer relations, and keep native fence/pipe
checks: the historical compact/666 GEMM plans had equal recorded relations yet
different device latency. Distinguish unavoidable source-prefix ordering under
the admitted primitive contract from order introduced by placement or fallback.
No metric here substitutes for device measurements.

Report occupied versus reserved keys; completion transfers versus helpers;
qualification/staged solves, replay, recurring omissions, final helper trials
and final certification separately. Several counters already exist: extend
missing distinctions rather than describe all accounting as absent. The FIFO
replay regression (26,206 to 813,458 evaluations in the recorded experiment)
remains open and is not solved by this plan. Address explicit accounting and
repeated propagation when that work blocks experimentation; do not hide it in
cheap slot-qualification cost or discard a useful mechanism without measurement.

All-path qualification/checking and emitted reconstruction remain required.
Finite unrollings and the independent oracle provide complementary evidence;
neither substitutes for qualified loop contracts. Native A3/ACC results remain
separate from ordinary-core and unqualified target/queue contracts.
