# CanonicalSync planning-efficiency milestone

This document records the completed planning-efficiency milestone on
`codex/canonical-sync-hardware-graph` and the work deliberately deferred until
after its device gate.

## Starting point

The checked-out branch is based on:

```text
2de9ccfdfc9576ade88fc3717c01b99c0fc7a00a
feat(pto): add explicit unsafe GM alias policy
```

Two prerequisite changes are already committed:

1. `8259cdff6642ec87eddabfab2e7d2b8f8f9136a2` makes generated cross-core
   synchronization fail closed until collective AIC/AIV participation can be
   represented and proven.
2. `2de9ccfdfc9576ade88fc3717c01b99c0fc7a00a` adds the explicit
   `conservative` and `distinct-roots-unsafe` GM alias policies. Conservative
   aliasing remains the default.

## Intended milestone

The milestone implements the following plan:

1. instrument CanonicalSync stages and state growth;
2. replace known-local all-pairs alias discovery with interval indexing;
3. precompute immutable region children, event execution loops, and source-cut
   prefixes used by singleton coverage evaluation;
4. integrate immutable fixed supply before freezing the residual demand graph;
5. replace dense candidate incidence with sparse, bidirectional incidence;
6. replace repeated greedy scans with a deterministic lazy max-heap;
7. measure verifier loop-state growth, without changing its semantics yet.

Fixed supply means intrinsic completion, qualifying existing fences, and the
mandatory exit drain. Obligations already satisfied by that immutable supply
are removed before the residual demand graph is frozen. Existing fences and
tail drains remain explicit physical mechanisms for coverage and
materialization; they are not selectable set-cover columns.

The singleton policy remains unchanged: every selectable direct mechanism is
evaluated once with the fixed graph supply through the regional summary, flat
scoreboard, and bounded oracle. The solver consumes only the authenticated,
cached incidence produced by those evaluations. It does not rerun coverage in
the greedy loop.

## Implemented milestone

### Statistics

`CanonicalSyncStatistics` and `--canonical-sync-stats` currently record JSON
statistics for:

- graph sizes and residual/fixed-covered demands;
- alias pair tests, candidate pairs, and indexed intervals;
- mechanism and coverage-world counts;
- sparse incidence size and greedy heap work;
- precomputed prefix entries;
- per-stage wall time;
- verifier loop transfers and maximum retained loop states.

The new regression is
`test/lit/pto/canonical_sync_statistics.pto`.

### Local interval index

Known, physical, non-GM accesses with finite byte intervals are sorted by
address space and interval start. A sweep emits only overlapping local access
pairs. GM, unknown-space, dynamic-range, and otherwise unresolved accesses use
the existing conservative pair test. Thus the optimization removes the common
local quadratic path without weakening conservative GM behavior.

### Fixed supply integration

`integrateCanonicalFixedBaseline` compacts the demand list before
`freezeGraph()`. It currently removes:

- exit-completion obligations supplied by mandatory tail drains;
- completion supplied by synchronous/intrinsic execution;
- completion supplied by an existing qualifying fixed fence;
- same-pipeline recurrence completion supplied by an existing fixed fence;
- visibility supplied by an existing fence plus required cache maintenance.

Unresolved MTE3-to-MTE2 GM visibility remains excluded from this fixed-supply
rule.

### Precomputed hierarchy and cut reachability

The model stores explicit immediate children for each region. For each
mechanism it also caches enclosing execution loops and the source-prefix phases
that may precede its physical source cut. The regional and flat coverage paths
consume these caches; the bounded structured oracle remains independent.

Singleton evaluation also computes the immutable baseline region summaries and
flat fixed point once. For one direct mechanism, only its action region and the
ancestor path to the function root are recomputed; unaffected child summaries
are reused. The flat differential oracle starts from its own baseline fixed
point and closes the singleton delta. The bounded structured oracle still
executes every singleton world independently from scratch, so it continues to
detect an unsafe incremental overclaim.

Detailed region summaries remain live only while a world is checked. Once the
regional result agrees with the flat oracle and every exhaustive bounded
oracle, the persistent catalog keeps the authenticated mechanism and demand
incidence plus oracle verdicts, but releases the concrete summary payload. The
baseline summaries remain available locally until all singleton deltas have
been evaluated.

### Sparse cover and heap solver

Each singleton candidate stores sorted covered demand IDs. The instance also
stores the reverse mapping from each demand to provider candidates. The greedy
solver uses a deterministic lazy max-heap keyed by current uncovered gain, then
performs the existing count-based reverse deletion. No coverage oracle is run
during selection.

## Validation

The following local gates completed successfully with at most two aggregate
workers:

- all 11 directly modified C++ objects compiled;
- a fully linked `PTOASCompiler` build completed, including every generated
  pass-option dependent;
- 52/52 CanonicalSync lit tests passed;
- the `comm_p2p_emitc.pto` unannotated-fence regression also passed;
- `git diff --check` passed;
- the changed-code compliance check reported:

```text
checked_files=17 errors=0 warnings=0
```

The six inputs that crashed in the wrong-revision device campaign no longer
crash. With the current safety policy they fail cleanly and atomically on the
device-unproven MTE3-to-MTE2 GM publication hazard. Stage statistics identify
mechanism construction as the intentional failure point.

The implementation audits established that:

- interval indexing is used only for finite physical local ranges; GM,
  unknown, dynamic, and nonphysical cases retain the previous alias predicate;
- fixed-fence loop requirements follow the existing coverage rule that every
  required loop contains at least one protected endpoint;
- complementary immutable fence guards are checked as an exact binary-choice
  cover before a demand is removed;
- source-prefix caches reproduce the old physical program-point query and are
  not consumed by the independent bounded oracle;
- sparse candidate coverage and reverse-provider lists are sorted, unique, and
  validated bidirectionally;
- exit obligations are deliberately removed from the residual universe while
  mandatory tail mechanisms remain explicit and materialized;
- statistics are emitted for successful functions and partial failed stages;
- verifier loop state is measured but its representation is unchanged.

## Deferred work

This milestone does not add general mechanism groups, indirect-only providers,
or a new cost model. Those remain later work after singleton planning and the
current device gate are sound.

The coverage-efficiency follow-up based on the device measurements below is
deliberately narrower than a semantic redesign. It:

- hashes mechanism interning and indexes recurring mechanisms by carrying loop;
- replaces linear SSA trace discovery with keyed forward/backedge sets;
- indexes completion facts and boundary transfers, including resource-keyed
  propagation and worklist transfer closure;
- retains baseline child summaries by reference and avoids copying clean
  subtrees or the complete baseline summary catalog for every singleton;
- updates greedy gains from demand-to-provider incidence instead of rescanning
  complete candidate columns; and
- records fact, transfer, boundary, and bounded-oracle work counters so the
  remaining coverage cost can be attributed on the next corpus run.

### Event-ID scarcity follow-up

The branch now implements four fail-closed scarcity tools: certified ready-lane
reuse, compatible physical-cut coalescing, a fully verified serialized
ready/release protocol, and the recurring release pool described below. The
earlier comparison of the other CanonicalSync branches identified an additional
conflict-core repair layer for later work:

1. prove allocation infeasibility before changing physical cuts;
2. report the directed event domain and a minimal pressure/conflict core;
3. propose only bounded, conflict-core-local alternatives, such as a targeted
   source-pipeline barrier/frontier followed by one legal event, or another
   already supported direct/recurring mechanism;
4. rerun selection, allocation, and independent verification for each repair;
5. retain strict failure as the default if no verified repair is feasible.

The release-pool repair does not change the singleton cover model. Non-nested
non-boundary recurring protocols in one reverse event domain may use one pooled
ownership token:

```text
Set<release> before the complete structured frontier
  -> each executed recurring body Wait<release> ... Set<release>
  -> Wait<release> after the complete structured frontier
```

The pool is also mandatory for an inner non-boundary recurring loop whose
ordinary reverse prime/drain would otherwise repeat inside an outer loop. This
closes a latent reset-before-drain lifecycle hole. Zero-trip inner or outer
executions preserve the primed token, while every executed body consumes and
returns exactly one token. Recurrence loops in one pool must be non-nested; an
ancestor/descendant pair could attempt a nested wait before the outer body
returns ownership. A boundary recurrence has a second, forward prime/drain
lifecycle; nested boundary recurrences remain fail-closed until both directions
can be lifted over the complete outer lifecycle. Boundary handshakes are also
excluded from release-pool scarcity repair because their two independently
primed directions do not use the pool's forward-handoff chain.

For a non-boundary recurring protocol, the four steady-state actions either
execute at the unconditional recurrence-loop body level or remain together in
one nested concrete action block. The guarded form consumes the release token,
performs the ready handoff, and restores the release token only when its arm
executes. An iteration that skips the arm leaves the token live and unchanged.
The guarded form remains outside a repeating outer loop and is excluded from
release pooling because neither case has a proven complete outer lifecycle.
A loop wholly nested in a choice remains supported because each dynamic
execution of that loop executes its body handoff.

The allocator treats the complete pool as one globally live reverse generation
and may reuse a forward ready key only across different pool member loops. The
materializer emits explicit pool tags, and the independent event verifier
reconstructs the boundary, every member loop, the exact directed key, and the
non-nesting proof before accepting either release-key or ready-key reuse. It
does not accept lexical drain-before-prime ordering as a proof.

The focused seven-protocol regression is
`canonical_sync_recurring_release_pool_scarcity.pto`. The qproj corpus family
that motivated this work contains forward handoffs confined to conditional
loop arms. The guarded single-lane protocol recovers the subset whose source
and target share one concrete arm; a host admission ablation recovered 19 of
the 60 admissions lost by the unconditional-body hardening. The remaining 41
still fail closed because they need a different recurrence shape, exceed the
event pool after admission, or fail independent memory verification. This is
host evidence only; the guarded protocol still requires focused device stress.

No internal `PIPE_ALL` fallback is enabled or proposed as an ordinary cover
column. A compile-total broad fallback would require a separate explicit policy
and device evidence; it is not part of the current roadmap.

Historical corpus results from the pre-publication-gate revision provide a
useful pressure corpus even though the current pass now rejects those kernels
earlier when they contain an unproven MTE3-to-MTE2 GM round trip. The 48
historical allocation failures were concentrated in five directed domains:

```text
21  AIC M    -> MTE1
12  AIV MTE2 -> V
11  AIV V    -> MTE2
 2  AIV MTE2 -> MTE3
 2  AIC M    -> FIX
```

The first inspected `qproj_matmul` failure was dominated by reverse release
lanes for recurring MTE1-to-M protocols, not by ordinary forward M-to-MTE1
events. A historical release-pool experiment made two variants compile by
sharing releases in opposite arms, but those choices were re-evaluated inside
outer loops. The arms can therefore execute on different dynamic iterations;
the experiment did not prove consume-before-reset and must not be reused.
Release-key sharing is valid only across opposite arms of a choice with no
repeating ancestor, because only one complete prime/body/drain lifecycle can
then execute per function invocation. The allocator and independent verifier
both enforce this restriction. Sequential release lifecycles also remain
interfering: lexical drain-before-prime order crosses physical pipelines and is
not a hardware consumption proof.

The historical `scatter_softmax_pool` and `score_reduce` plans contain
plausible connector chains such as MTE2-to-MTE3 plus MTE3-to-V, or V-to-MTE3
plus MTE3-to-MTE2. Singleton worlds cannot credit a demand that only the two
mechanisms cover jointly. This is evidence for a later bounded affected-slice
group check, not for enumerating all mechanism pairs. Candidate groups should
be proposed only from the failing domain and resource-compatible neighboring
cuts, then certified by the concrete hierarchical coverage evaluator.

The other local CanonicalSync implementations are not suitable for wholesale
reuse. The older scarcity search colors lexical set/wait intervals and relies
on a wait-before-later-set lifetime assumption that this branch deliberately
rejects. The pattern-cover experiment improved synchronization counts on some
accepted kernels but, with ordinary internal `PIPE_ALL` candidates removed,
accepted only 7 of 28 comparison cases and did not expand admission. Its useful
idea is limited to conflict-local barrier-plus-event frontiers. The retained
policy is therefore:

1. exploit exact mutual exclusion and already-certified lifecycle ordering;
2. coalesce compatible physical cuts;
3. use the verified serialized ready/release repair;
4. examine bounded connector groups or alternative singleton covers only in
   the allocation pressure slice; and
5. fail closed if no independently verified plan fits the hardware IDs.

## Device-gate interruption on 2026-09-02

The delivered archive
`canonical-394-vs-insertsync-d99520397-final.tar.gz` must not be treated as the
requested current-candidate gate. Its internal provenance records:

```text
HEAD   d99520397eabcd708f81febee5a2a8836c061ae1
parent 5798ee20ce80d2ebf050ecc669769b8e2f6835bc
```

The dispatched task instead required:

```text
HEAD   257fcad4c3cb89959554067333502a7442b5a478
parent 842e4bd1bcff8e68c7e6b47cc409bb0c6d91fc55
```

The campaign therefore measured the earlier corpus-collection commit. Its six
deterministic CanonicalSync crashes are useful historical evidence, but they do
not establish a regression at `257fcad4c` or the current branch. Before the
current relink, all six frozen crash inputs successfully analyzed with an older
local linked binary. That binary did not embed an independently queryable
source revision. The completed current build instead rejects all six inputs
cleanly on the restored MTE3-to-MTE2 publication rule. Together these results
show that the old crash is gone without treating an unproven protocol as safe.

Useful historical measurements from the wrong-revision run are:

- 388/394 CanonicalSync admissions and 394/394 InsertSync admissions;
- 388/388 CanonicalSync `VERIFY ok` and non-synchronization equivalence;
- CanonicalSync compile-time geometric mean 1.259 times InsertSync;
- CanonicalSync emitted 6,404 set/wait pairs and 6,112 barriers, versus 5,477
  pairs and 3,882 barriers for InsertSync;
- no standalone frozen kernel had an independent per-kernel golden, so no
  device correctness or timing was performed.

The current-candidate device gate must use strict provenance failure when the
checked-out SHA differs from the task SHA.

## Device gate for `0ec958f1e`

The corrected 394-kernel campaign completed with matching source and corpus
provenance. It reported:

- 394/394 crash-free planning runs;
- 251 admitted kernels and 143 atomic, policy-driven rejections for the
  device-unproven MTE3-to-MTE2 GM publication case;
- 53/53 focused CanonicalSync tests and 1898/1899 full `check-pto` tests;
- sparse cover incidence at 6.7% of the equivalent dense matrix;
- at most three verifier loop states, so verifier-state rework is not
  currently justified;
- no event-scarcity rejection, with three rows using serialized repair;
- exact non-synchronization IR equivalence on all admitted rows; and
- seven repository-golden A3 cases passing both CanonicalSync and InsertSync
  on cards 6 and 7. No distinct A2 target or silicon was available.

The principal performance result is that demand discovery fell to 2.2% of
planning time while coverage grew to 92.4%. The existing counters did not
explain a 162-times coverage spread, which is why the follow-up patch adds
inner-loop coverage and bounded-oracle counters before changing more
algorithms. The current work after `0ec958f1e` is not covered by that device
verdict.

## Host frontier ablation after the guarded-lifecycle repair

A matched ON/OFF run used one compiler binary and the frozen 394-row corpus.
The OFF arm disabled only synthesized non-recurring shared event frontiers;
direct mechanisms, demand semantics, selection, allocation, materialization,
and both independent verifiers were unchanged.

Both arms admitted 199 rows and failed closed on 195. With frontier synthesis
enabled, 577 candidates were proposed and 84 were selected across 50
successfully verified functions. Selected logical mechanisms fell from 3,462
to 3,345, a reduction of 117 (3.38%) without changing admission.

The extra candidates increased coverage work from 10,383 to 10,960 singleton
worlds (5.56%), sparse incidence entries from 1,463,135 to 1,583,100 (8.20%),
and indexed boundary-phase tests from 1,937,469 to 2,138,996 (10.40%). These
deterministic work counters are the appropriate compile-cost evidence; the
single sequential timing sweep was not randomized and is not a stable timing
claim.

The same run measured the existing exact same-pipe interval behavior. Of
2,732 unique direct barrier candidates, selection retained 1,763 and avoided
969 (35.47%). Of the selected barriers, 1,685 covered multiple demands across
141 verified functions. This confirms that target-adjacent direct barriers
already provide useful interval-frontier sharing; adding a duplicate barrier
candidate family is not justified.

These are host planning results, not device-performance evidence. The next
device task should compare the default and frontier-disabled arms on the same
launchable kernels and report dynamic execution time separately from static
mechanism counts.

## Diagnostic storage-ownership projection

The next bounded milestone adds no synchronization candidate. It indexes
residual local demands into ownership-channel signatures only when one storage
root has a same-iteration RAW publication and a reverse positive-distance WAR
reuse edge under the same guard, carrying loop, and same-core resource pair.
The diagnostic exposes channel count, ready and release edges, static
multi-buffer depth, and whether every edge retains a slot expression.

This deliberately does not infer a lifecycle from an arbitrary SCC or from a
WAW recurrence. A depth-two allocation remains insufficient to synthesize two
event lanes until distance-aware slot analysis proves that adjacent
generations use different slots and the reuse generation returns to the same
slot. The frozen corpus run for this patch must therefore preserve admission,
selected-mechanism count, and emitted IR while measuring how often complete
channels, depth-two channels, and fully slot-tracked channels occur. Only the
proven depth-two subset is eligible for the following atomic
`ReadyRelease<2>` experiment.

The matched 394-row host run preserved every function's status, failure stage,
demand count, mechanism count, and selected-mechanism count relative to the
frontier-enabled baseline. It again admitted 199 rows, and the 212 successful
function records selected 3,345 mechanisms. Among successful functions, the
projection found 734 complete channels in 106 functions; failed functions
contained another 1,945 channels. None of the frozen inputs retained a
multi-buffer root or per-access slot expression at this compilation stage, so
the run found zero depth-two and zero fully slot-tracked channels. The signal
is common, but this corpus cannot evaluate a multi-lane mechanism. The first
`ReadyRelease<2>` ablation therefore needs a separate slot-preserving corpus
drawn from pre-selection PTO IR or focused manual kernels; the frozen corpus
remains useful only for behavior and compile-cost non-regression.
