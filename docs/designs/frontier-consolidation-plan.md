# Frontier consolidation implementation plan

This plan supersedes the orchestration and runtime priorities in the earlier
review plans. It retains their exact repeated-guard emission goal. The reference
compiler is `f7bf2483e15ce1223cc02d474f163f6d463d69aa`. The revised draft is
`25bae62ceee5d80351d1c39c2dab7b5d7c7ac2c5` in the synchronization paper repository.
The reviews of compiler `93f3431e9` are evidence to recheck, not a description of
all code at the reference compiler.

## Constraints

- Preserve Section 5.4 mathematical algorithms and physical-access semantics.
- Runtime work comes first; architecture, exact logical emission and semantic
  validation are required deliverables too.
- Instrument construction work; introduce no new internal work-budget or
  admission cutoffs. External experiment limits are not class criteria.
- Follow Section 8: retain exact whole-region mathematics when an export fails.
  Section 6 fallback does not replace an exact result with weaker ordering.
- Keep mathematical readiness, executable logical plans, physical allocation,
  source emission and device execution as separately reported outcomes.
- Exclude scarcity repair, a streaming demand evaluator, kernel-specific
  recognition and unrestricted symbolic allocation from this implementation.

## Milestones and acceptance

Each milestone has focused tests, architectural and correctness review, and a
separate commit. Infrastructure alone does not complete a production milestone.

| Milestone | Implementation | Acceptance | Status |
|---|---|---|---|
| 1. Measured baseline | Aggregate allocation summary/query/proof/lane timing and inner-loop counts; precise retained-prefix refusal; pinned baseline replay. | Profiling preserves commands and certificates; bounded diagnostic storage; classify the Persistent GEMM hot stage and Delta blocker. | Complete; evidence below |
| 2. Directional finite allocation | Implement Section 7's ordered SET/WAIT threshold sweep and interval coloring under explicit premises, including insufficient-capacity cases. Make logical-only allocation evidence optional; request stronger lane evidence only when useful. | Independent exact allocation oracle, linear reuse-query count, just-over-capacity regression; full compilation preserves allocation successes. | Finite sweep complete; logical-only allocation request complete; lane scheduling pending |
| 3. Capability ownership | Separate mathematical provider, recognized form, conversion, composition and requested export. Upgrade endpoint capabilities on retained evidence. Add distinct logical-plan, closed-matching and completion-summary capabilities. | Owner/context/occurrence identities survive upgrades; failed upgrades retain mathematics; unchanged work is shared. | Sequence preparation failure isolation complete; broader capability consolidation pending |
| 4. Shared compact input | Delay fine cell memberships; preserve compact selections and borrowed memberships through regional adapters. Index/shared-own aliases, pointer/view origins, scalar/descriptor/control facts; compactly represent universal descriptor invalidation. | Select-chain, wide-access, view-depth and descriptor/call/depth scaling counters; same physical overlap and alias semantics. Coarse Section 6 input avoids fine atomization. | Alias duplicate-search indexing complete; persistent memberships and shared fact views pending |
| 5. Numerical composition | Project to actual parent ports with hierarchical interior queries; reduce crossings once; construct guarded boundaries with prefix/suffix passes; share immutable PhaseIndex facts. | Independent reachability/boundary oracle including macro envelopes and empty/guarded regions; bounded-port and sibling-region scaling families. | Ordinary boundary prefix/suffix construction complete; numerical port projection and shared source facts pending |
| 6. General Section 6 fallback | Closed executable child interface and mode-tagged completion transfer on immediate-child CFG; two delta fixed points; qualified completion protocols, explicit loss and independent physical reuse. | Joins, backedges, while final-false visit, opaque/scalar prerequisites, child cleanup and repeated invocation tests; strict Section 8 dispatch policy. | Pending |
| 7. Exact repeated-guard integration | Derive certified incoming executed-source ordinals from retained demands, including relational sequence crossings. Use shared predicate histories, one invocation-owned recipe and one staged clone. | Public Gated Delta MayNotAlias logical emission succeeds with 29/29 exact regions; independent matching/closure and payload preservation; correct skipped-source, nested-reset and cleanup behavior. MayAlias gaps remain separately reported. | Regional prefix ownership, singleton forwarding and crossing presence complete; executable selectors and public Delta emission pending |
| 8. Proof reuse and validation | Share lazy inactive-proof and premise contexts, replay mappings and safe Boolean conversion work. Close known guard-equivalence/oracle gaps and run pinned paired campaigns. | No unexplained acceptance loss or changed output; report eleven kernels and 783-input corpus by stage, wall time, memory and inner work. | Lazy inactive-proof setup shared; broader proof reuse and campaign pending |

## Algorithm boundaries

The explicit scan remains proportional to actual access incidences, even when
membership storage is shared. Removing copies does not establish a compressed
scan bound. Guarded windows retain their stated quadratic terms. Numerical
periodic reductions require sparse supplied generators; counters distinguish
alias relations, expanded pairs and retained records.

The Section 7 directional sweep requires ordered SET and WAIT command chains
and exact consumption-before-republication queries. It must not replace a
more general allocator without proving these premises. Parent ports are
projected under Section 5.5 while interior queries remain available through
the composition hierarchy.

Section 6 requires closed child matching and executable completion protocols;
it does not require selective child queries. Closed matching alone does not
certify physical reuse across visits. Qualified barriers do not consume child
flags. Cleanup precedes the enclosing completion boundary.

The review's Theorem 7.23 concern remains a separate paper obligation: the exact
excess formula must account for all cut reversals, including cross-lane ones.
Do not use that formula to certify arbitrary cut orders. General repair is
outside this implementation plan.

## Measurement contract

Record the source and binary hashes, build settings, input hashes, alias policy,
requested capabilities and external limits. Resume interrupted campaigns
without treating interruptions as compiler refusals. Use at most two local
resource-intensive workers in aggregate. Baseline and candidate corpus timings
must run serially without an overlapping build before making a performance
comparison; overlapping exploratory runs are coverage evidence only.

Aggregate allocation buckets report exclusive accumulated durations, not
contiguous spans. Summary callbacks include uninstrumented descendant work.
Group timing includes setup and some tuple encoding; serialization covers the
explicitly instrumented attribute construction only. Lane query-key creation
is outside query timing. Pair-count records have zero duration.
Query/proof calls, cache hits/misses and examined group/lane/boundary pairs are
separate from mathematical backend dispatches. Reports must distinguish a
measured hot stage from source-derived scaling risks.

## Milestone 1 results

The pinned baseline replay completes all eleven inputs: ten logical successes
and seven allocation successes. Gated Delta is the logical refusal; GroupNorm,
causal-convolution prefill and decode are the allocation refusals. Certification
retains Delta's 29/29 MayNotAlias and 23/29 MayAlias exact regional results.
This checkpoint does not claim C++ emission or device execution.

Eight focused lit tests pass. They include independent profile-on/off public
output equivalence, bounded aggregate storage and timing conservation, retained
mathematical results, and the existing history/prefix regressions. Architectural
and correctness source reviews accepted this change. The changed-code prefilter
reports brace-rule false positives on conditions containing nested calls; all
reported selection bodies have braces. The build emitted only pre-existing
covered-switch-default warnings in unchanged files.

Persistent GEMM takes 29.840 seconds without profiling and 32.544 seconds with
profiling in these isolated single runs. Its profiled allocation-support scope
is 32.174 seconds, of which recursive summary preparation is 32.170 seconds.
The subsequent query and proof aggregates are each below 0.2 milliseconds, and
lane work below 0.3 milliseconds. There are 864,272 examined group-member pairs,
196 lane pairs and 28 boundary pairs in the instrumented certificate stage.
This narrows the next profiling target to recursive summary construction; it
would be incorrect to attribute the 32 seconds to the final lane proof.
Profiling has observable overhead here; these are diagnostic measurements,
not evidence of a speedup. Persistent GEMM's emitted IR is byte-identical with
profiling on and off.

Delta still refuses public logical insertion (0.904 seconds in this probe).
The newly distinguished prefix diagnostic is `sequence prefix collection
retains an analysis failure: regional child endpoint preparation failed`.
Ordinary endpoint preparation writes this error into shared sequence state,
and prefix collection refuses it before reaching its relational-crossing check.
Milestones 3 and 7 must isolate preparation failure from retained mathematical
state before claiming that a particular later adapter is the active blocker.
No Delta recovery is claimed by this instrumentation checkpoint.

Pinned input hashes, binary hashes, source hashes, counters and receipts are in
[frontier-consolidation-validation.json](frontier-consolidation-validation.json).

## Milestone 2: finite directional sweep

The ordered finite allocator now implements Proposition `sparse-shared-allocation`:
visit WAITs in command order, advance one pointer through the SET chain, and
bucket-color the resulting integer lifetime intervals. It returns the exact
per-direction requirement even when more than six IDs are needed. Insufficient
capacity on this path never constructs the dense reuse relation or invokes
matching. It does not add ordering or change the logical plan.

Canonical explicit certificates carry a target-bucketed WAIT permutation.
The decoder validates its type, completeness and uniqueness, nondecreasing
SET-prefix and WAIT ranks within each direction, equal-cut rank consistency,
and exclusion of reflexive reuse. Certificates without this extension retain
the generic allocator. Rank evidence still derives its semantic authority from
the fixed logical plan; structural checks do not establish arbitrary supplied
facts about a different command graph.

The compiled independent oracle constructs SET and WAIT chains with all
three-handoff matching permutations and arbitrary return edges, computes their
closure, rejects cycles and compares allocation with brute chain partitions.
A 4,096-handoff seven-ID case checks at most 8,192 reuse queries and valid lane
reuse. Existing generic allocation checks cover 2,048 causal orders. Five
focused lit tests pass, including public pipeline, allocation-session, finite
allocation, physical allocation and sequence composition checks. Malformed WAIT
permutations/ranks and legacy certificates are covered. Both source reviews
accepted the sweep. This is the completed finite-allocation component of
milestone 2; allocation-capability scheduling remains open.

## Milestone 3: sequence preparation failure isolation

Sequence endpoint preparation now owns its attempt diagnostic separately from
mathematical construction errors. Direct preparation, regional callbacks and
numerical/arithmetic adapters use this boundary. Failed emission does not erase
exact class membership, invalidate queries or block a later recipe on the same
retained owner. Genuine expression-construction errors remain failures; shared
expression IDs are not rolled back beneath child caches.

Seven focused lit tests pass. The new regression injects child preparation
failure, retries through direct and exported callbacks, verifies retained
mathematical membership and queries, and checks that an unavailable endpoint
predicate cannot be hidden by a successfully emitted false placeholder. Existing
guarded-publication checks validate the selective-SET failure followed by early
publication and terminal cleanup. Both independent source reviews accepted this
component. The compliance prefilter reports three nested-condition brace-rule
false positives; each reported body has braces.

The isolated Delta MayNotAlias replay still refuses logical insertion in 0.904
seconds, now with `sequence child lacks its exact retained prefix owner`.
The previous `sequence prefix collection retains an analysis failure` blocker
is gone. This is progress in ownership and retry behavior, not completion of
Delta emission or the broader capability milestone. The next prefix task is to
preserve the exact child owner through the relevant regional adapter.

## Milestone 8: lazy inactive-proof setup

Bounded endpoint preparation now validates its original window and constructs
monotone-presence patterns once per preparation attempt. Reached records share
that proof context and its cache; an early endpoint refusal still avoids proving
later records. Per-record proof transactions roll back only temporary proof
expressions. Pattern IDs and endpoint expressions appended between calls remain
valid. Original record IDs are checked against the arena size before context
construction, so newly created patterns cannot legitimize an invalid input ID.

Four focused lit tests pass: inactive emission, bounded prefix realization,
bounded allocation sessions and sequence analysis. The inactive-proof oracle
checks repeated lazy calls with interleaved expression creation, stable results,
malformed-record refusal and recovery. The prefix oracle still checks 4,745
executions. Both architectural and correctness reviews accepted the change after
the original-ID validation regression was found and corrected. Receipts are under
`.local/three-paths/consolidation/m8-inactive/`.

Delta's isolated MayNotAlias replay remains an explicit logical refusal with the
same missing retained prefix-owner obligation (0.804 seconds, 259,608 KiB RSS).
This single probe is not evidence of a general speedup. No demand predicate,
Section 5.4 algorithm or allocation policy changed.

## Ordinary sequence boundary construction

The ordinary composer now constructs first accesses in a forward pass and last
accesses in a backward pass. Each child contributes all simultaneous envelope
alternatives before its presence updates the accumulated mask. Constant-false
alternatives are discarded; surviving alternatives keep their original order.
This removes repeated remasking and rescanning of accumulated boundary vectors.
It does not change crossing reduction, requested exports, or event allocation.

`boundary_summary_visits` counts visits to supplied boundary alternatives only.
The composer still traverses child-by-cell combinations; expression construction,
port retention, prerequisite recovery and crossings are separate costs. This
change therefore does not establish a linear bound for complete composition.

A concrete occurrence-group oracle compares exported first/last writers,
prefix/suffix readers and native boundaries over all 32 valuations of five
independent guards. The fixture includes simultaneous writers on two pipes,
reader-only groups, an optional overwrite and entirely empty executions.
The scaling fixture uses 8, 16 and 32 separate runtime-loop children, which
cannot be fused into one adjacent explicit run.

Validation: eight focused sequence/public/macro/singleton lit tests pass. The
separate repeated-region script passes 14 independent nested command closures;
its loop-count assertion now counts operations instead of diagnostic substrings.
The fixture's subsequent phased-repeat script still exceeds its 90-second
watchdog, reproduced with the pinned baseline binary. This remains unresolved.
The 8/16/32-loop summary-only probes visit 280/536/1048 alternatives. Their
initial full-preparation version exceeded 90 seconds in downstream preparation
work; this is not a claim of improved complete-request latency. Architecture
and correctness reviews accepted the component, including the boundary oracle.
Receipts are under `.local/three-paths/consolidation/m5-summary/`.

## Alias-union duplicate search

Each translator alias union now builds one call-local hash membership index,
using every field of the existing exact record equality. Clones substitute the
destination SSA identity before lookup. Existing destination records seed the
index, accepted records retain first-seen order, and borrowed keys expire before
other translator operations can mutate geometry. Self-source unions are no-ops.

This removes the expected cubic duplicate-comparison path on select chains.
Explicit prefix vectors still retain quadratic storage; no compressed alias-set
or linear source-size bound is claimed. Translator counters separately record
candidates, seeded records, hashes, hashed address entries, equality checks and
retained clones, and reset at each Build invocation.

Four focused lit tests pass, including both alias policies, selected views and
A3/A5 selections. The independent chain fixture checks every ordered root set
and duplicate-arm union at 32/64/128/256 roots. At 256 roots it records 65,790
retained records and 32,895 equality checks; all four sizes satisfy the quadratic
work checks. Both source reviews accepted the change. Eight previously passing
sequence/public tests also pass with this implementation. Two additional CLI
fixtures cannot finish because this build lacks `ptoas._core`; their native
storage/alias portions passed. Receipts are under
`.local/three-paths/consolidation/m4-alias/`.

## Retained prefix capabilities through singleton composition

Regional prefix contributions now retain the source input, expression arena and
producer state independently of ordinary endpoint availability. Composition
retains proper child mathematics. The factory rejects mixed input owners,
nonempty children without provenance, and mismatched arenas. Generic event or
domain transformations clear the contribution. Only the proven whole-original
singleton normalization restores it: original branch cuts and nested invocation
resets remain in force, with no intervisit demands to reconstruct. Constant
slice endpoints now also supply their exact distance as an upper bound, allowing
interior singleton intervals without weakening the existing proof.

Eight focused lit tests pass. The direct singleton test invokes the actual
normalizer and collector after destroying intermediate wrappers, materializes
one clone, and checks eight executions against independent payload-coordinate
and command-closure oracles. Cases include first/interior/last visits, skipped
opaque branches, and lower bound seven with step two. The bounded prefix oracle
still passes 4,745 executions; ownership, foreign input, stale arena and generic
transformation rejection are checked separately. Architectural and correctness
source reviews accepted the changes. The compliance prefilter reported six
brace false positives (all bodies are braced) and mistook weak_ptr::lock for a
mutex operation; no new warning suppression was added.

The final isolated Delta MayNotAlias replay retains exact whole-function demands
but still refuses logical emission: `prefix crossing lacks current-context
source presence`. It has passed the former missing-prefix-owner barrier. The
probe took 17.623 seconds and 283,220 KiB RSS; reaching more preparation work
increased refusal latency relative to the earlier 0.804-second probe. This is
functional adapter progress, not a complete Delta recovery or a runtime win.
Physical allocation and source/device execution were not attempted for it.
Receipts are under `.local/three-paths/consolidation/m7-prefix-capability/`.

## Reusing certified crossing presence

Successful sequence closure now seals the construction invariant that each
retained crossing guard implies existence of its exact selected source and
target. Storage/value bridges establish this invariant, selected-coordinate
folding preserves active alternatives, and numerical/symbolic reduction only
retains or narrows those guarded records. Prefix collection checks this evidence
once instead of asking a child to reprove presence at a newly folded coordinate.
The certificate binds the owning state, arena, input/function/enclosing context,
edge tuples, port coordinates, choice identities, and original endpoint metadata.
A closure retry clears it; changed snapshots cannot use it. Regional semantic
callbacks remain immutable after closure. Original-cut, executable-guard,
coordinate and executed-source-rank checks remain separate and unchanged.

Snapshot construction and validation cost is proportional to retained edges,
choices, and port coordinate metadata. This introduces no new mathematical
route, admission limit, conservative fallback, or event-allocation repair.

Ten focused lit tests pass. The crossing oracle independently enumerates source
presence/latest selection under four valuations, exercises both reducers, and
rejects changed graph/context snapshots and failed reconstruction. Eight actual
sequence-prefix executions check independently guarded writers and a guarded
consumer, payload preservation, token matching, exact command closure and
surplus epilogue cleanup. This test explicitly composes supplied child interfaces;
it does not claim that the dispatcher selects that representation. Existing
singleton and bounded-prefix execution oracles also pass. Architecture and
correctness reviews accepted the component. The compliance prefilter reported
12 brace-parser false positives; all reported control bodies are braced.

The isolated Delta MayNotAlias replay still refuses logical emission, now with
`guarded expression input belongs to another invocation`, after passing the
crossing-presence barrier. It took 17.628 seconds and 283,624 KiB RSS. This is a
later endpoint ownership/availability obligation, not a successful logical plan
or a performance improvement. Exact-retention policy is unchanged. Physical
allocation and C++/device execution were not attempted. Receipts are under
`.local/three-paths/consolidation/m7-crossing-presence/`.


## Optional allocation evidence for logical-only emission

The public pass accepts `prepare-allocation=false` to omit the allocation-support
and certificate request after logical preparation. The default remains `true`,
including the library preparation entry point, so existing full compilation
continues to obtain evidence for `pto-frontier-allocate`. The report distinguishes
`not-requested` from `unavailable`. The logical plan may still carry inexpensive
allocation metadata assembled with its endpoints; the option avoids the deferred
proof construction, rather than promising zero allocation-related bookkeeping.

A retained library session can request allocation evidence before original IR
mutation. A public logical-only emission does not serialize that retained session:
its handoffs cannot later be allocated by the physical pass without the missing
evidence. Empty or barrier-only plans may not need such evidence. This option
changes the requested capability; it does not optimize the allocation proofs
required by a full compilation, add event IDs, or weaken synchronization order.

Five focused lit tests pass, including the five public backend fixtures, absent
allocation stages/certificates for logical-only requests, retained-session tests
and transactional refusal when evidence is missing. Comparing the bounded path
removes only its appended allocation lane coordinate and dead helper arithmetic;
other fixtures require unchanged member tuple arity. Original logical coordinates,
guards, payloads and commands agree after symmetric canonicalization. Source
architecture and correctness reviews accepted this change.

An isolated pair on Persistent GEMM with profiling enabled takes 29.748 seconds
with allocation evidence and 0.404 seconds without it (185,524 and 144,584 KiB RSS).
The latter records no allocation construction stages. This kernel's commands and
payloads match after removing only reports and certificates; no lane-coordinate
normalization is needed. The default output still allocates six IDs successfully
in 0.304 seconds. These are serial single runs of one binary under MayNotAlias,
not a full-compilation speedup or broad corpus result. Deferred allocation-proof
optimization and demand-driven lane strengthening remain open. Receipts are in
`.local/three-paths/consolidation/m2-optional-allocation/`.


## Reuse the numerical merge's crossing reduction

The sequence wrapper now receives the retained-crossing mask from the numerical
merge it already constructs, instead of reducing the same crossings twice.
The optional mask follows the supplied edge order, is cleared before construction,
and is published only after a successful merge. The mathematical reducer,
selection policy and retained graph are unchanged. Operation accounting charges
one reduction. Parent-port projection and shared phase facts remain open.

Six focused sequence/session/public-pipeline lit tests pass. The numerical oracle
checks 109,584 independent event pairs, including 1,280 graph/projection cases,
retained crossings, failed-construction output clearing, nested child-index reuse,
interior queries, overlays and a symbolic case. Its old hierarchy fixture also
failed on the pinned baseline because it assumed cost selection always chose the
numerical adapter. The revised fixture explicitly requests that existing adapter
without changing eligibility checks or production dispatch. Architecture and
correctness reviews accepted the change and fixture repair. Receipts are under
`.local/three-paths/consolidation/m5-single-reduction/`.
