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
recurring protocol variants, new event-scarcity fallbacks, or a new cost model.
Those remain later work after singleton planning and the current device gate are
sound.

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

The branch already implements three fail-closed scarcity tools: certified
ready-lane reuse, compatible physical-cut coalescing, and a fully verified
serialized ready/release protocol. The earlier comparison of the other
CanonicalSync branches identified one useful missing layer, to be considered
only after the coverage work is measured:

1. prove allocation infeasibility before changing physical cuts;
2. report the directed event domain and a minimal pressure/conflict core;
3. propose only bounded, conflict-core-local alternatives, such as a targeted
   source-pipeline barrier/frontier followed by one legal event, or another
   already supported direct/recurring mechanism;
4. rerun selection, allocation, and independent verification for each repair;
5. retain strict failure as the default if no verified repair is feasible.

No internal `PIPE_ALL` fallback is enabled or proposed as an ordinary cover
column. A compile-total broad fallback would require a separate explicit policy
and device evidence; it is not part of the current roadmap.

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
