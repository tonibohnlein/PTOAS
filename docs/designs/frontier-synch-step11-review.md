# Phase A step 11: independent directional covering boundaries

**Integration update (2026-09-25): accepted at this component's scoped v0.44 gate.**
See [current integration evidence and corrections](frontier-synch-steps7-13-review.md).
Patch-delivery validation/status statements below describe the original submission.


## Baseline and status

Target: draft **v0.44**, sections 3.3 and I.3, with the original-program and
interval contracts of sections 3.2/3.4. The relevant rules are unchanged from
v0.43. Base: `codex/handoff-foundation` at
`4122dd1531fbdb2859bd7b27772dba56930be393` (integrated steps 2–6), not the older
`0a38c8e9f` baseline. The remote head was rechecked before packaging.

**Status: implementation prepared and portable production-core tests passed;
native compilation, native test execution and independent reviewer acceptance
are still pending.** This is not an accepted increment or a claim of complete
Phase A parity. No remote branch was modified.

The existing `OriginalBoundaryResult` and exact/may-frontier interfaces remain
unchanged. The new service does not materialize legacy requirement pairs,
change obligation identities, allocate events, insert synchronization or grant
completion. It does not implement descriptor/subscription preparation early.

## Implementation and public interface

`ProgramAnalysis::sourceCovering(sourceInterval)` and
`ProgramAnalysis::targetCovering(targetInterval)` return independent
`OriginalCoveringBoundary` records. The intervals are prepared using the existing
`ProgramAnalysis::prepareInterval` interface. Each side needs its own appropriate
selector, engine, owner/occurrence interpretation and horizon; the interval of a
triggering edge must not be substituted for the family of all source uses.

The lazy `OriginalCoveringQueries` adapter consumes the unchanged shared imported
effects and the shared `OriginalValueQueries` service. Its implementation lives in
`lib/PTO/Transforms/FrontierSynch/OriginalCoveringQueries.cpp`. The portable
production algorithm is `DirectionalCovering.h`, not a separate test model.

The result retains the complete original interval, a legal cut only on a
`Covering` answer, static may-access witnesses, reference-coverage meaning,
explicit per-interval multiplicity, possible no-access executions, repetition
scope, and references to enclosed/trimmed original work. Cut-delimited work
includes original scalar/control operations even when they have no translated
payload leaf. The whole actual source prefix still needs Phase B's checks;
work preceding the queried interval is not asserted absent.

The covering endpoint's new guard is **true within its existing original
control**, not a reconstructed last-read condition. Finite-loop qualifications
retain their separate typed prerequisites, including `NeedsCompletion`, without
crediting those prerequisites. Unknown physical incidence or opaque selector
qualification conservatively retains possible empty-family executions. A
may-footprint is not used as a proof of full read coverage or participation.

## Step-11 checklist

| Required behavior | Implementation / evidence |
| --- | --- |
| Original cut-delimited intervals and complete cache identity | Existing preparation validates version, owner, cuts and selector; answers use `(OriginalInterval, direction)` keys. `cacheAndFormation()` checks all selector/occurrence fields, incoming identity and stale snapshots. |
| Maximal source suffix / target prefix trimming | `DirectionalCovering::derive`; `examples()` checks both directions and partial sequence intervals. |
| Shared sequence recursion and transparent wrappers | One immutable sequence index flattens only sequence wrappers. Choices/repeats remain atomic. Wrapped and unwrapped examples return the same original cut. |
| Conditional boundary, not a guessed last branch access | Source stops at the join; target at choice entry. Optional empty paths retain extra endpoint executions. |
| Qualified finite-loop boundary | Native adapter calls `OriginalValueQueries::counted`. The covering rule does not impose D3's invariant-participation premise. Unqualified whole loops retain may facts and a finite-exit obstruction. |
| NoHit and uncertain effects | NoHit emits no endpoint. May/unknown matching blocks trimming. No-hit subregions do not grant completion or prove an otherwise positive interval finite. |
| Legal cuts and correct occurrence multiplicity | Non-executable internal phase cut is rejected without widening; the other direction remains usable. Qualified body intervals report one visit per body interval, not one per invocation. |
| Extra work / executions remain visible | Entire retained choices and loops remain in work references. `mayExecuteWithoutAccess` is conservative, not an executable no-hit guard or an exactly-participating certificate. |
| Preserve independent answers and original requirements | Separate covering APIs do not replace exact results. Native fixture checks existing may frontiers, unchanged pair counts, unavailable earlier guard, legal IR positions and unchanged IR. Native execution is pending. |

## Qualification boundary

The implemented I.3 SESE procedure accepts forward cut intervals in one original
sequence visit, including contiguous subsequences, structured entry/exit cuts,
and transparent sequence wrappers. An interval inside a repeating child needs
its explicit `FirstReach` interpretation and is counted per such child visit.
An external finite-loop boundary executes once per interval, including zero
trips, without finding the last participating iteration.

A cross-arm, backedge-crossing or otherwise non-SESE interval does not obtain a
positive boundary from a may-control walk. Those results keep conservative
witnesses and identify the missing visit/SESE premise. They are not declared
physically impossible. This patch supplies no new D1/D2/D4 occurrence transport
certificate. A claimed supported cross-boundary case must first provide the
qualified interval/multiplicity premise, rather than bypass this check.

An unavailable prescribed cut also returns an obstruction, not an ancestor
search or a broader replacement. This applies even when a containing original
instruction has an executable outer cut: the internal phase cannot silently be
relabelled as that later boundary.

## Correctness argument and self-review

Complete may-effects justify exclusions. Sequence summaries use disjunction for
possible hits; a guaranteed hit survives a choice only when both children have
one. Repetition requires a separate finite-exit qualification for a positive
boundary. A may-hit does not establish guaranteed participation.

After removing a source's no-hit suffix, every matching issue is in the retained
prefix. Its final atomic child is either a payload, a conditional, or a qualified
finite repetition. The prescribed after-cut follows every issue of that child;
it also follows earlier sequence children. The target proof is the dual. Neither
proof adds any completion edge. Keeping a choice/repetition whole establishes
one endpoint visit per interval, not one per matching access.

Self-review caught and fixed a cut-ordering bug: two different cuts at the same
sequence gap were initially treated as interchangeable. In particular, including
the stopping access must not turn `after(a)..before(a)` into an empty same-visit
interval. The index now preserves within-gap cut order; both cases have explicit
regressions. Selector keys are also shared once per summary table, rather than
copying long predicate-dependency lists into each node's memo key.

These checks and the separate concrete scanner are **not an independent
reviewer's acceptance**. That acceptance must be recorded after the native gate.

## Executed validation

The exact production core, using the existing original-point/control/interval
headers, passed GCC and Clang C++17 builds with `-Wall -Wextra -Werror -pedantic`,
and Clang AddressSanitizer + UndefinedBehaviorSanitizer. Leak checking was
disabled (`ASAN_OPTIONS=detect_leaks=0`); no LeakSanitizer result is claimed.

Each run reports **439,633 checks**, including **110,718 independent
finite-execution comparisons**. The reference scanner executes original syntax
and cut visits; it does not use the production control graph, sequence index,
summary recurrence or trimming procedure. It covers zero/one/multiple loop trips,
per-visit changing reader guards, modes, engines, optional paths, uncertain
incidence and stop inclusion. Enumeration occurs only in these small tests.

The independent-reader formation case at 4,096 optional readers produces 8,192
summary evaluations, 12,288 effect queries and 16,384 returned array references
across the two directional queries. These are measured work/output counts, not
wall-time measurements or a claim that ordered-map/key-comparison costs vanish.
No production guard-valuation or reader-set enumeration is performed.

Six deliberate mutations were all rejected: skipping source trimming, treating
unknown effects as empty, ignoring finite-repeat qualification, erasing extra
empty-path executions, accepting an unavailable cut, and forgetting within-gap
cut order. These mutation checks ran on temporary copies, not the delivered files.

`git diff --check` passed. Patch application was checked against a clean local
source subset whose existing files were verified against the remote Git blob
hashes. Direct network cloning and the LLVM/MLIR/native toolchain were unavailable;
this was **not a full repository build or a native regression-suite run**. The
native adapter, CMake target and PTO fixture are supplied but uncompiled here.

## Reproduce and finish the acceptance gate

From the patched repository:

```sh
bash test/FrontierSynch/run_covering_core.sh
CXX=clang++ SANITIZE=1 ASAN_OPTIONS=detect_leaks=0 \
  bash test/FrontierSynch/run_covering_core.sh

# Use the repository's configured LLVM/MLIR build directory.
cmake --build build --target \
  pto-frontier-covering-core-test pto-frontier-covering-test
llvm-lit -j1 test/lit/pto/frontier_synch_covering_step11.pto
```

The new native fixture uses real shared PTO import. It checks
`A; compute(g); if(g) B; U`, with the source before U despite g being unavailable
after A; the target and optional-path dual; a qualified counted loop with varying
read participation; an unqualified while; cache/version behavior; and unchanged IR.
Run it and the existing focused Phase A/pass-mode tests, then obtain independent
review against the checklist above before marking step 11 accepted. The existing
shared instruction-effect interface and existing sync path are not modified.

Steps 12 and 13 must attach these directional records to immutable obligation
IDs, generate the declared descriptor slots, and subscribe their structured cuts
before construction. Step 14 must repeat the integrated parity review. None of
those unimplemented procedures is being relabelled as an unsupported case or
claimed completed by this increment.
