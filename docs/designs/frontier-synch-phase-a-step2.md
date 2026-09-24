# Phase A step 2: occurrence, cut and continuation records

> **Integration update, 2026-09-24:** this increment has passed local native
> validation and independent review at its stated scope. See the
> [integrated review](frontier-synch-steps2-6-review.md) for fixes and evidence.
> The candidate/pending statements below record the original patch submission,
> not the current integration status. Later Phase A gates remain open.

## Scope and status

Implementation increment against upstream
`0a38c8e9f4bd4852177c1dd6f6e0d0ee26b6ca16`, branch
`codex/handoff-foundation`. The accepted factored transfer core from
`9bba6552055d5386ff7242c58031b0412885a0d6` is not rewritten.

Target: revision 0.44, sections 3.2, 3.4, 3.6, the original-position interface
of section 4.1, and Appendix I.1/I.3. This increment implements the shared
records and their original-control interpretation, not the later D1-D4,
qualified-write, factored-provenance or descriptor-completion work.

**Status: submitted for native validation and independent review, not accepted.**
The standard-library-only production core was compiled and exercised. Native
PTO/MLIR integration and the full existing suite were not compiled or run in
the patch-preparation environment. No independent human/agent reviewer has
accepted this increment. The independently coded trace interpreter is a test
oracle, not an independent review of the implementation.

## Common records and their consumers

`OriginalProgramPoints.h` contains the original `Region` record, the unchanged
`NoControlId` convention, and the common records:

| Record | Meaning and consumer |
| --- | --- |
| `OriginalCut` (`SourceMilestone` alias) | Payload before/after, external structured scope entry/exit, and numbered child-region entry/exit. Used by the control projection, interval queries, native cut resolver, and endpoint candidates. |
| `OriginalProgramVersion` | Opaque shared snapshot identity plus revision. A prepared query cannot be rebound to another original import. |
| `OriginalAccessSelector` | Cell, read/write modes, optional engine and physical-relation witness filter, qualification identity and predicate dependencies. A relation ID filters translated incidences; it does not prove a bank permutation. |
| `OriginalOccurrenceContext` | Original source/target roles, stop-visit interpretation, named original backedge, explicit incoming interface and qualification identity. These records do not create an incoming writer or establish completion. |
| `OriginalIntervalRequest` / `OriginalInterval` | Version, selector, occurrence, actual starting/stopping cuts, stopping-access inclusion, declared continuation, and derived physical owner. |
| `OriginalContinuationCases` | Represented incoming, child-entry, bypass, backedge, stop and owner-exit cases. The original graph retains the actual transitions; these flags are may observations, not guarded exact participation. |

`OriginalLifetimes::prepareInterval` is the public record-preparation boundary.
`ProgramAnalysis` forwards it. The canonical overloads of `mayAfter`, first/last
may-use and `supportBetween` use these records. Construction-facing decoding and
interpretation retain the same identity and expose explicit-continuation
overloads. Original source/deadline requirements and translated effect witnesses
remain unchanged.

The old raw-site frontier API and the old payload-continuation API remain for
source compatibility. Raw-site queries are uncached and version-check any supplied
stamp. Their positive access frontiers now also require executable original cuts.
The payload-continuation wrapper prepares a canonical interval and retains its
old hard owner-containment premise; it does not silently widen that premise.
New clients should use cut-delimited intervals, not persist private graph node IDs.

## Cuts and ownership

`resolveOriginalCut` returns an actual block and insertion-before operation in the
unchanged original IR. A child exit is before its existing terminator; an if join
or loop exit is after the complete structured operation. The function's outer
cuts denote body entry and the position before its terminator. An absent else
region has no child-local insertion point: no block or cut is fabricated.

The control graph adds explicit epsilon ports for these boundaries while keeping
payload IDs `0..n-1` and the function-exit ID `n`. In particular, external loop
entry is not the repeated header/body entry. While exits still execute the final
before region. Anonymous sequence wrappers add no public cut or owner identity.
All original alternatives and conservative loop bypasses remain represented.
The graph remains a may control language; it does not decide predicate feasibility.

The owner is the least original scope containing the source role, target role,
both cut interpretations and declared continuation. A named backedge also
participates when requested. An omitted continuation asks for the least owner;
an explicitly supplied `NoControlId` means the function horizon. A local index
reset cannot reset this identity. A child exit and an enclosing overwrite are
different intervals even when their triggering static source and target agree.

Internal translated phase boundaries can remain analytical access milestones,
but they are not executable cuts unless the existing lowering contract says so.
Interval preparation and positive frontier/endpoint results reject unavailable
cuts. They do not substitute `enclosingAfter` silently. Existing subscriptions
continue to keep their distinct sufficient and enclosing executable positions.

## Interval interpretation

The starting gap is literal: starting after a payload excludes that visit's
access. Stopping-access inclusion is independent of the stop's spelling:

| Stop | Inclusion | Behavior |
| --- | --- | --- |
| Before payload | false | Stop without inspecting its access. |
| Before payload | true | Inspect that stopping access, without following its outgoing continuation. |
| After payload | false | Exclude that final stopping occurrence's access. Earlier occurrences remain. |
| After payload | true | Include the stopping occurrence's access. |

A structured stopping boundary has no access, so requesting inclusion there is
invalid. Starting and stopping at the same after-gap does not import the access
preceding the start, even with inclusion enabled.

`FirstReach` means the first represented stop visit. `AfterBackedge` means the
first stop visit after at least one backedge of the named original loop. A
single finite state bit implements that language; it does **not** certify a
constant distance, an exactly matched bank use, or D2/D4 transport. An
`Unqualified` repeated stop retains the legacy Unknown exclusion outcome.
Qualification identities are retained in keys, never accepted as Boolean proof
flags. Future qualified descriptors must have distinct identities within their
original snapshot.

First/last **may** frontiers operate on this same interval state graph. A reverse
last-use walk must keep predecessors of a revisited starting state: encountering
its static ID is not proof that the dynamic interval began there. `Present` is
not an exact guarded frontier or an exactly-once matching certificate.

The legacy D3 reader summary has its own syntax slice. A record adapter converts
that slice to actual original cuts; it is not labelled with the triggering
source-to-deadline interval, which may end before the first reader. Support
qualification compares the full original support identity and the reader's
actual span. Unsupported slice/occurrence transport stays unresolved rather than
reusing a lexical-owner answer for a different interval. General exact
cut-delimited D3 queries remain a later step.

## Cache contract

Continuation and may-frontier caches include the complete `OriginalInterval`;
first and last have distinct cache entries. Decoder/interpreter keys additionally
contain the original requirement ID. Fixed-visit caching also retains the full
interval. Changing a cell, engine, role, physical selector, qualification,
predicate dependency, either cut, inclusion, continuation, owner or original
version cannot hit an old answer accidentally. Unknown, NoHit and positive
results retain their separate outcomes.

Whole-original marginal indexes (such as all-access sets and lifecycle tables)
remain snapshot-owned: they do not depend on a caller's interval. Original edits
require a new version and rebuilding Phase A; the version is not a hash that
automatically detects mutations of borrowed IR/effects. `OriginalLifetimes`
rejects an externally changed snapshot before reading its cached query results.
The selected synchronization ledger has a separate version and does not grant
credit to these original-only records.

## Focused evidence

`pto-frontier-interval-core-test` compiles the **production** records, control
graph and interval traversal directly. It uses no MLIR/PTO stubs. Its independent
Region interpreter enumerates small finite executions with dynamic cut and
backedge tokens; it does not traverse the implementation's control graph.

The current test reports **27,941 checks**, including **11,940 all-access,
first-may and last-may interval queries** across choices, counted loops, while
before/after control, conditional readers, and nested loops. Loop fixtures
include zero through three iterations. These are finite regression checks, not
a general recurrence proof. Separately, 512 independent optional accesses check
linear graph population and bounded traversal without enumerating valuations.

Additional checks cover complete key-field separation, declared owner horizons,
child exit versus enclosing overwrite, before/after inclusion, transparent
sequences, unavailable internal phase cuts, source-after exclusion, explicit
incoming identity, and wrong original versions. Five deliberately broken variants
were compiled and rejected by semantic tests: omitted inclusion/qualification
key fields, ignored continuation owner, accepted illegal internal cuts, and a
last-use walk truncated at a revisited start.

The added `pto-frontier-interval-test` and `frontier_synch_intervals.pto` exercise
real import, actual MLIR insertion points, the public lifetime/ProgramAnalysis
APIs, actual cache results and IR preservation. The two-phase adapter subtest is
explicitly a simulated translation of one real operation, not a claim that a
native multi-phase opcode was imported. **These native tests are registered but
were not run in the patch-preparation environment.**

Run the standalone checks from a checkout:

```sh
CXX=g++ test/FrontierSynch/run_interval_core.sh
CXX=clang++ test/FrontierSynch/run_interval_core.sh
```

In a configured PTOAS build, run the native and regression gates:

```sh
cmake --build build --target pto-frontier-interval-core-test \
  pto-frontier-interval-test pto-frontier-analysis-test --parallel 2
lit -j 1 -v build/test/lit --filter 'frontier_synch_intervals'
lit -j 1 -v build/test/lit --filter 'frontier_synch'
cmake --build build --target check-pto --parallel 2
```

## Remaining acceptance gates

The focused core passed GCC and Clang release builds, an assertion-enabled GCC
build, and Clang AddressSanitizer/UndefinedBehaviorSanitizer execution. Patch
application and whitespace checks are separate packaging checks. Native
compilation, the registered public-interface test, all five development kernels,
existing autosync regressions and independent step-2 review remain required.

Later plan steps still own qualified full-cell writes (3), conditional provenance
and incoming histories (4), original-value availability (5), guarded obligations
(6), D1/D2/D4 qualification (7-9), exact frontier/interval participation (10),
directional cover and descriptor generation (11-12), and complete structured-cut
subscriptions/support closure (13). This patch changes neither shared instruction
effects nor event construction. It does not implement a complete Phase A pass or
claim native performance improvement.
