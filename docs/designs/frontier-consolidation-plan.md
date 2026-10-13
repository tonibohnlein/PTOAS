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

### Priority revision after the review of `a166645de`

Keep the eight milestones and their acceptance criteria. Execute the remaining
work in this order:

1. **Milestone 5 bridge construction:** remove constant-false writer/reader
   frontier entries after processing complete occurrence groups. Count bridge
   candidates, remasked entries and retained bridges independently of boundary
   summary visits. Test separate compact children, simultaneous macro envelopes,
   conditional replacement and untouched intermediate children. Then address
   sparse child/cell participation; pruning dead history alone does not remove
   the child-by-cell traversal or residual alias pair products.
2. **Milestone 7 executable selectors:** retain producer-owned occurrence-choice
   recipes across exports and coordinate adapters. Capture primitive executed
   source ranks at their actual cuts, and evaluate choices at consumers. Preserve
   target occurrence selection as well, including same-pipe barriers. Predicate
   snapshots are supporting machinery, not a substitute for lost construction
   evidence. Public Delta logical success remains the completion criterion.
3. **Milestones 2, 3, 4 and 6 consolidation:** finish demand-driven allocation
   evidence, session-owned facts and capability upgrades, coarse input admission,
   and import of closed executable children into Section 6. Preserve the exact
   retention policy; the conservative fallback does not rescue a failed export
   of retained exact whole-region mathematics.
4. **Milestone 8 measurement and closure:** measure threshold substitutions and
   implication traversals as well as expression counts; share safe attempt-owned
   proof preparation. Close outstanding semantic oracles and run the pinned
   full pipeline with the new counters. Report successful compilation, retained
   mathematics, refusal, allocation and device execution separately.

The seven-kernel device comparison is a separate measurement task in
`test/npu_validation/benchmarks/auto_sync/DEVICE_TASK.md`; it does not expand the
compiler milestone scope to event-scarcity repair. The current eleven-kernel
replay gives ten logical plans and seven allocated C++ outputs, with 240/240
MayNotAlias exact regions. This is current-head coverage evidence, not the final
paired corpus campaign or device validation.

| Milestone | Implementation | Acceptance | Status |
|---|---|---|---|
| 1. Measured baseline | Aggregate allocation summary/query/proof/lane timing and inner-loop counts; precise retained-prefix refusal; pinned baseline replay. | Profiling preserves commands and certificates; bounded diagnostic storage; classify the Persistent GEMM hot stage and Delta blocker. | Complete; evidence below |
| 2. Directional finite allocation | Implement Section 7's ordered SET/WAIT threshold sweep and interval coloring under explicit premises, including insufficient-capacity cases. Make logical-only allocation evidence optional; request stronger lane evidence only when useful. | Independent exact allocation oracle, linear reuse-query count, just-over-capacity regression; full compilation preserves allocation successes. | Finite sweep complete; logical-only allocation request complete; repeated-lane proof reuse complete; lane scheduling pending |
| 3. Capability ownership | Separate mathematical provider, recognized form, conversion, composition and requested export. Upgrade endpoint capabilities on retained evidence. Add distinct logical-plan, closed-matching and completion-summary capabilities. | Owner/context/occurrence identities survive upgrades; failed upgrades retain mathematics; unchanged work is shared. | Sequence preparation failure isolation complete; broader capability consolidation pending |
| 4. Shared compact input | Delay fine cell memberships; preserve compact selections and borrowed memberships through regional adapters. Index/shared-own aliases, pointer/view origins, scalar/descriptor/control facts; compactly represent universal descriptor invalidation. | Select-chain, wide-access, view-depth and descriptor/call/depth scaling counters; same physical overlap and alias semantics. Coarse Section 6 input avoids fine atomization. | Alias duplicate-search indexing complete; borrowed regional adapter complete; global compact memberships and universal descriptor dependency indexing complete; shared fact views and delayed atomization pending |
| 5. Numerical composition | Project to actual parent ports with hierarchical interior queries; reduce crossings once; construct guarded boundaries with prefix/suffix passes; share immutable PhaseIndex facts. | Independent reachability/boundary oracle including macro envelopes and empty/guarded regions; bounded-port and sibling-region scaling families. | Dead bridge-frontier pruning and counters complete; ordinary boundary prefix/suffix construction and single numerical crossing reduction complete; parent-port projection and scoped source facts implemented through sequence and compact routes; session-wide consolidation remains |
| 6. General Section 6 fallback | Closed executable child interface and mode-tagged completion transfer on immediate-child CFG; two delta fixed points; qualified completion protocols, explicit loss and independent physical reuse. | Joins, backedges, while final-false visit, opaque/scalar prerequisites, child cleanup and repeated invocation tests; strict Section 8 dispatch policy. | Qualified A3 structured completion complete; larger closed-child imports and coarse input preparation pending |
| 7. Exact repeated-guard integration | Derive certified incoming executed-source ordinals from retained demands, including relational sequence crossings. Use shared predicate histories, one invocation-owned recipe and one staged clone. | Public Gated Delta MayNotAlias logical emission succeeds with 29/29 exact regions; independent matching/closure and payload preservation; correct skipped-source, nested-reset and cleanup behavior. MayAlias gaps remain separately reported. | Regional prefix ownership, singleton forwarding, crossing presence and qualified predicate snapshots complete; general executable selectors and public Delta emission pending |
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


## Borrow memberships through the regional numerical-body adapter

The finite numerical-body adapter now scans borrowed atom memberships and retains
rank rows and compact per-cell writer/reader extrema. It no longer retains an
expanded occurrence/access word or unrequested per-atom scan witnesses. The
metadata-only reducer input has one entry per payload. Query exports use the
retained reduction directly. Original occurrence coordinates, input/frame checks,
physical protection and recurring-GM/macro refusals are preserved.

This saves incidence-sized copies and retained intermediates; it does not reduce
the mathematical incidence term. The scan visits the incidences once and a second
borrowed pass constructs extrema with per-occurrence mode consolidation. Ordered
containers add logarithmic work. `scanWork` counts the scanner, not that second
boundary pass. Global fine partition expansion remains a separate open task.

Seven focused lit tests pass. The atom/effect scaling oracle compares borrowed
and materialized generators/rank rows, verifies shared memberships remain shared,
and checks absence of retained witnesses. Independent boundary expectations cover
read-only cells, multiple pipes, prefix/suffix readers, duplicate accesses and
read-modify-write occurrences. Invalid atoms and macro envelopes are rejected.
Session tests cover contexts, ownership and coordinates. Both source reviews
accepted the component.

An initial validation build preserved mirror timestamps and left a stale test
object across the result-type change. Rebuilding all affected header consumers
resolved that infrastructure failure; the final seven-test receipt is the
acceptance evidence. Future mirror updates must refresh timestamps. Receipts are
under `.local/three-paths/consolidation/m4-borrowed-regional/`.


## Preserve opaque evaluation identity across regional exports

The shared expression arena now records an opaque Boolean input's original SSA
value, tracked loop, and exact visit ordinal. The arena retains the placeholder
owner after its producer is destroyed. This records evaluation identity only;
it does not certify availability at an endpoint, definition presence, or unique
execution through an intervening loop. Consumers must establish those conditions.

Imports translate ordinal dependencies and preserve the original source identity.
They reject conflicting annotations and cycles created by merging two arenas.
Failed imports and nested speculative construction roll back both DAG nodes and
new provenance records, including records attached to preexisting inputs.
Ordinary substitution leaves opaque inputs atomic; coordinate rebinding requires
a distinct evaluated input. This prevents an unrelated arithmetic substitution
from silently changing which original predicate evaluation an input denotes.

Eight focused lit tests pass, including producer lifetime, ordinal remapping,
conflicting metadata, nested rollback, and cross-arena cycle rejection. Source
architecture and correctness reviews accepted the component. The initial broad
header rebuild reached its external watchdog; the resumed build found a test-only
const MLIR accessor error, which was corrected before the final passing build and
tests. The compliance prefilter's brace findings were inspected as parser false
positives; every reported body is braced. This component supplies provenance for
the consumer adapter; it does not itself make Gated Delta logically executable.
Receipts are under `.local/three-paths/consolidation/m7-evaluation-provenance/`.


## Capture primitive source ranks before selecting a boundary

Sequence prefix realization now decomposes a folded source boundary into its
original primitive ports. Each primitive saves its executed-source ordinal at
its source cut. The consumer evaluates the retained choice DAG and selects the
saved rank. Port discovery is iterative, validates topological child identities,
and visits shared nodes once. It preserves occurrence coordinates and all phase
site/invocation/event guards; executing an original operation alone does not
identify which phase view it represents. One invocation coordinator still owns
publication, watermark advancement and terminal cleanup.

Eight focused lit tests pass. The new independent graph fixture checks 24
executions with a source selected by a later loop result, zero/short loops,
repeated enclosing invocations and two phase views sharing one original anchor.
It checks exact payload closure, matching, cleanup and unchanged source IR using
supplied regional interfaces and the ordinary presence check. The phase case
would detect dropping phase identity guards. The nested case covers repeated
execution, but is not independently a proof that removing rank resets is unsafe:
its absent selected source also makes the consumer guard false. Architecture and
correctness reviews accepted the component.

The public Gated Delta MayNotAlias logical-only replay still refuses with
`guarded expression input belongs to another invocation`. Provenance and delayed
choice do not yet materialize historical predicate evaluations at the consumer.
That replay overlapped one serial focused-test process, so its timing is coverage
evidence only. Physical allocation, C++ emission and device execution were not
attempted. Receipts share the `m7-evaluation-provenance` directory above.


## Partial Boolean constants and expression-oracle discovery

Shared constant specialization now propagates false through AND and true through
OR even when the other operand remains unknown. Numeric operations retain their
existing requirements. Literal roots return immediately. A proposed self-premise
shortcut was rejected during review because allocation uses that query to find
contradictory active guards; the final implementation preserves those proofs.

Six focused lit tests pass, including a new discoverable wrapper for the broad
region-expression oracle. The existing `.mlir` fixture is not discovered by the
suite's `.pto` suffix rule. The wrapper runs its region-expression check, not all
other RUN lines in that input. The oracle's repeated-region numerical fixture now
explicitly requests the existing numerical adapter and retains its eligibility
checks and independent unfolded-graph comparison. The new checks cover absorbing
values, nearby live expressions, distinct SSA operands and contradictory
self-premises. Architecture and correctness reviews accepted the component.

Diagnostic-only Delta probes identify the first unavailable prefix expression as
a same-pipe barrier guard before the load at input line 148. Its 8,717-node DAG
references the original comparison at line 156 at visits zero through three.
Uniform implication and constant-specialization probes did not prove inactivity.
Finite valuations suggesting inactivity are not a certificate. The probes and
their emission fallback were removed from the build mirror; no production guard
is replaced by a sampled or assumed value. Public Delta logical recovery remains
open. Receipts are under `.local/three-paths/consolidation/m8-partial-constants/`;
earlier diagnostic probes are under `m7-evaluation-provenance/`.


## Exhaustive threshold proofs for inactive endpoint recipes

The shared expression owner has a sufficient uniform inactivity proof for roots
containing direct constant comparisons of an integer Input. For each candidate
Input independently, it partitions all uint64 bit patterns at zero, comparison
literals, safe successor literals and the sign boundary for signed comparisons.
Each comparison has a constant truth value throughout each resulting cell.
Signed comparisons use the existing exact APInt64 folding semantics.

Each cell fixes only those comparison atoms. The integer Input is replaced by a
constant only in singleton cells; nonsingleton cells retain its symbolic identity
and exact unsigned bounds. Existing typed substitution rebuilds selected ranks,
then the constant proof must establish false under those bounds. Every cell must
succeed. Other Inputs, including opaque predicate evaluations, remain independent
and unconstrained. Trying several Inputs does not take their Cartesian product.
Transactions discard each cell's temporary nodes and substitution memo.

This establishes uniform falsity rather than a fact valid only at one source cut.
Sequence prefix emission requests it lazily after ordinary expression emission
fails. A proved-inactive Boolean recipe discards failed scratch emission and
emits false; same-pipe barriers and unused incoming-rank choices are omitted.
Retained demands, occurrence identities, source publications and terminal cleanup
remain owned by the existing invocation coordinator. Inconclusive proofs retain
the ordinary endpoint failure. No sampled guard value, new backend, input
assumption or internal admission budget is introduced.

The optional counters record root-DAG discovery visits, input families and cells.
They do not count every substitution or implication traversal. Construction still
pays for rebuilding and proving each attempted cell; this is not a new fast-path
complexity guarantee and does not complete general historical-predicate emission.

Eight focused lit tests pass, including 32 new independent inactive/live prefix
executions and the expression, bounded-prefix, allocation and public-pipeline
checks. Both architecture and correctness reviews accepted this component. The
initial build reached its external 900-second watchdog; the resumed build
completed in 79.39 seconds. A serial public Delta MayNotAlias logical-only replay
still refuses with `guarded expression input belongs to another invocation`
(11.62 seconds, 284,984 KiB peak RSS). This component does not establish public
Delta recovery. Receipts are under
`.local/three-paths/consolidation/m7-threshold-inactivity/`.


## Allocation-summary proof reuse

A user-space CPU sample of Persistent GEMM identifies repeated implication proofs
inside `periodicLaneAllocation`, called during repeated-region summary creation,
as the dominant work. `refutesNegation` accounts for 44.83% of samples in its own
body; additional samples are in its graph traversal and map operations. This is
inside the previously measured summary stage, not the later top-level lane
certificate stage. The 49 Hz sample collected 1,287 samples with no reported loss.

One periodic allocation call now owns separate caches for implication and
constant-specialization requests in its expression arena. Both positive and
negative/unknown results are retained. Existing IDs keep their meaning as the
arena grows; the caches die before any enclosing attempt rollback. Proof order,
reuse matrices, allocation policy and selected synchronization remain unchanged.
Optional counters measure requests and actual computations; they impose no
admission limit and do not claim that distinct proof requests are inexpensive.

The independent repeated-allocation oracle checks 1,583 member envelopes and
reuse chains. Its instrumented cases issue 1,904 implication requests and compute
196 distinct answers. A new invariant, unknown return-edge fixture refuses reuse
across every candidate gap, caches its unknown specialization, and repeats in the
same arena after rollback with fresh computations. Four focused lit tests pass,
including the expression, allocation-session, physical-allocation and public
pipeline checks. Architecture and correctness source reviews accepted the change.

The comparison pins baseline `5a3f01401` and changes only this component. Two
serial unprofiled Persistent GEMM MayNotAlias runs take 30.139 and 30.042 seconds
before the change, and 9.018 and 9.116 seconds afterward: means 30.090 and 9.067
seconds, a 3.32x improvement for logical preparation including allocation support.
Full emitted IR, including allocation evidence, is byte-identical in all runs.
Peak RSS ranges from 185,636 to 185,812 KiB in the baseline and 187,816 to
188,396 KiB in the candidate.
The candidate also passes physical allocation with six eligible IDs (0.202 s).
This is a kernel-specific compiler measurement, not a corpus or device-speed claim.
Receipts and sampling data are under
`.local/three-paths/consolidation/m2-summary-profile/`. The original source mirror
has been restored to the candidate and rebuilt; no profiling-only source remains.


## Project numerical composition onto parent boundary ports

Each numerical merge now retains only its requested parent ports. Children also
retain the endpoints needed by crossings at that merge. Interior queries descend
through the immutable hierarchy instead of flattening every child event into the
root index. Original sequence event IDs are mapped explicitly to projected dense
IDs. The one boundary-summary snapshot is reused when publishing the result.

A parent can request a new prerequisite to an interior event absent from a
child's projected index. The adapter builds an index over just those requested
events using the child's exact queries and reference order. Unknown presence or
order refuses the adapter. A fallible stable merge sort uses O(P log P) reference
comparisons; this does not bound the cost of each child query or claim compact
interfaces for arbitrary symbolic nesting.

Six focused lit tests pass. The independent hierarchy oracle checks all event
pairs after introducing a crossing between previously omitted interior events.
The 8/16/32/64-child family checks bounded root ports and linear total retained
parent ports. Existing tests cover empty children, overlays, guarded fallback,
retained mathematics, prefix emission and the public pipeline. The numerical
hierarchy check is now discoverable through the .pto expression fixture.
Architecture and correctness reviews accepted the component. The compliance
prefilter's eleven brace findings are false positives on already braced bodies.
Receipts are under `.local/three-paths/consolidation/m5-projection/`.


## Compact global physical-cell memberships

The physical partition retains its existing byte geometry and cell identities.
Each effect now stores sorted half-open runs of cell IDs. The endpoint sweep
opens and closes these runs instead of appending each new cell to every active
effect. Consumers iterate the same memberships lazily; overlap compares runs.
This removes incidence-sized backing storage and sweep appends. It does not
remove the explicit scan's incidence term or defer fine partition construction.

Eleven focused lit tests pass, including the independent byte-intersection and
range-union oracle, carried/reused storage, blocked layouts, sequence composition,
regional exports, allocation and the public pipeline. The 512-wide-access fixture
represents 262,656 incidences with 1,024 stored runs. The oracle also covers
overlapping/duplicate ranges, adjacency, gaps, separate memory spaces, empty and
reversed ranges, maximum addresses, overlap and rebuilding the partition.

Three additional storage fixtures have stale descriptor/recognition expectations.
A control build using the original per-cell sweep produces byte-identical output
for all three, establishing that this change does not cause those failures.
The GM-alias fixture's native checks pass; its final CLI check cannot load the
existing Python extension. These failures remain visible and are not counted as
passing tests. Both source reviews accepted the representation. The compliance
prefilter's brace/assertion matches were inspected as braced bodies and internal
partition invariants. Receipts are under
`.local/three-paths/consolidation/m4-compact-memberships/`.


## Bind only reached replay inputs

Endpoint expression emission creates its temporary IR mapping from reached
inputs and replay dependencies. It no longer copies every input accumulated
at the cut on each emission. Existing result caching, dominance checks and
deterministic scalar replay remain unchanged. A saturating diagnostic counter
records bindings actually copied; it imposes no admission limit.

The independent 128-expression fixture adds unrelated cached inputs between
emissions and checks exactly one new binding per expression, the correct
original operand, and no new bindings for a repeated cached result. Its initial
assertion incorrectly required a fixed operand order for commutative addition;
the corrected check permits canonical operand ordering. Eleven focused tests
pass with this change, including expression, sequence-prefix, allocation and
public-pipeline checks. Both source reviews accepted the component. This closes
the growing replay-mapping copy issue, not general historical-value availability.
Receipts share `.local/three-paths/consolidation/m4-compact-memberships/`.


## Section 6 completion transfer and qualified structured fallback

The new completion-transfer provider owns mode-tagged A/E/S summaries and the
transparent-visit bit independently of exact mathematical results. Its compact
CFG retains branch joins and loop backedges. Two delta worklists propagate each
storage-class mode over each reachable edge at most once per solve. Unreachable
children do not contribute. A forward prerequisite is an explicit caller
obligation; no generic callback silently declares it completed.

The initial production adapter treats original payload operations as the closed
children and uses one conservative all-memory class. It supports structured
if/for/while control, including accesses on a while loop's final false visit.
It inserts common completion at the selected original cuts and at function exit.
Qualification requires the A3 single-core PIPE_ALL contract, modeled phases and
effects, and synchronous scalar prerequisites. Unknown calls, mixed cores, macro
protocols, manual flags and asynchronous scalar-readiness obligations refuse.
A barrier completes prior work; it is never used to consume a child's flags.

The completion owner binds the original function, source-input phase identities,
index, operation identities and unchanged source text. Insertion stages one
module clone, verifies it and then transfers the selected function body. Its
logical plan cannot be mixed with exact endpoints or private-state plans. It
needs no physical event IDs and exports no exact event queries or storage
selectors. Reports explicitly identify conservative completion. Dispatch tries
it only at the root in fallback mode after failing to retain exact mathematics;
an exact result with an unavailable export remains an explicit refusal.

This is the first production Section 6 subclass. Importing larger closed exact
children and bypassing initial fine atomization remain open. Consequently the
complete general fallback milestone is not yet finished.

Validation: ten focused lit tests pass, including the 16,384-graph independent
execution oracle, original-cut checks, stale-owner and transactional refusal,
public logical and physical passes, and strict retained-exact fallback refusal.
The initial failures exposed two test-driver/printer issues and missing empty
logical-plan metadata; all are fixed. The standard empty endpoint-family
contract now lets the physical pass accept the barrier-only plan without IDs.
The C++17 build passes. The compliance prefilter's brace findings were inspected
as braced control bodies. No device execution was performed. Receipts:
`.local/three-paths/consolidation/m6-completion/`.


## Qualified predicate snapshots for prefix expressions

Sequence crossing recipes can retain actual Boolean evaluations at selected
constant visits of an original unit-step loop. One invocation-owned collector
shares each value/validity slot pair and resets it before every loop invocation,
including an empty invocation after a nonempty one. The recipe snapshots the
original mapped SSA result after its definition; it never replays an opaque
producer. Layout is collected before the existing single staged clone.

A read requires either original-cut availability or an independent cofactor
proof that changing the missing evaluation cannot change the expression. These
proofs leave other historical inputs unconstrained and use numeric control
premises, so independently proved substitutions compose. Unsupported frames,
nonunit strides, non-64-bit index layouts and overflowing selected coordinates
remain explicit refusals. Emission binds only reached mapped inputs and replay
dependencies. This extends the existing finite-fringe representation; it does
not complete general executable selector construction.

Public Gated Delta remains unresolved. A parent port can carry an ordinal whose
selection was folded inside its child. Capturing historical predicate values
does not recover the construction evidence needed to match that selected
occurrence at both source and target cuts. The next integration step must
preserve that producer-owned selector evidence across exports. No exact result
is replaced and no physical allocation or device recovery is claimed.

Validation: eight focused snapshot/expression/sequence/public-path tests pass,
plus the two descriptor tests in the same isolated build (ten total). Snapshot
checks cover nine acceptance/refusal modes, visits zero through four, two
outer invocations, simultaneous unavailable values, signed lower bounds,
nonunit-stride refusal and coordinate overflow. Both source reviews accepted
the snapshot design; final sparse mapping review also accepted. The C++17 build
passes with existing warnings in unchanged files. The compliance prefilter's
brace matches were inspected as braced bodies; a test line was wrapped.

The final public Delta logical-only probe still refuses another-invocation
expression input: 15.727 seconds, 285,448 KiB peak RSS. This is a single diagnostic
replay, not a speed comparison. No Cartesian threshold-product proof is included.
Receipts: `.local/three-paths/consolidation/m7-snapshots/`.


## Shared universal descriptor dependencies

Opaque calls and unsupported region operations now mark enclosing loop subtrees
once. Dependency indexing enumerates the dynamic descriptors once per marked
loop, excluding descriptors defined inside that loop. Repeated calls reuse the
mark, preserving the same loop inputs and conservative call invalidation. This
removes the descriptor-count times call-count times nesting-depth dependency
probe term. It does not eliminate per-call invalidation or all ancestry walks.

The new counter measures actual universal descriptor probes. Eighteen generated
families vary descriptor count (4/8), call count (1/16/64), and depth (1/3/6),
requiring unknown shapes after opaque calls and at exit while bounding universal
probes by descriptors times depth. Both descriptor tests pass in the final
ten-test snapshot/descriptor run, with a successful C++17 build. Source review
accepted the component. The shared-input milestone remains partial. Receipts
are in `.local/three-paths/consolidation/m7-snapshots/`; the separate commit
manifest is in `.local/three-paths/consolidation/m4-descriptor-dependencies/`.


## Scoped shared phase source facts

PhaseIndex now retains an immutable owner for original source discovery,
phase-independent relevance and proved carried ordinals. A regional view
constructs its own selected-phase relevance delta and prerequisite edges. It
does not import another view's prerequisite relation. Original function and
phase-pointer membership are checked before publishing a view; an uncovered
larger symbolic request explicitly rebuilds facts for its full input. Owners
survive destruction of the initial index and borrow unchanged IR as before.

Singleton, finite-visit, phased and rotating composition pass the source owner
through the sequence interface. The symbolic fallback still requests its full
original phase set, reusing discovery only when the owner covers that set.
Compact-route forwarding is a separate follow-up; this is not a claim that
every session request now shares all preparation.

Validation: all 32 subsets of a five-phase fixture compare shared views against
fresh indexes for relevance, carried proofs and native/demand prerequisites.
The fixture includes duplicate envelopes, structured yields and dead carried
state. Foreign-function, foreign-phase, expanded-universe and owner-lifetime
checks pass. Twelve focused lit tests and two independent composed/carried-phase
closure scripts pass. Source review accepted the changes. The C++17 build passed
after adding a missing SCF include to the new test header. Compliance brace
matches were inspected as braced bodies.

The initial broader run hit an external 300-second timeout in composed-phase
validation. Its periodic-length and owned-storage subtests also exceed separate
60-second watchdogs on the saved M1 candidate. The bounding native-reclosure
test fails on that saved binary too: it requires a particular merge count. That
route-count assertion is being corrected separately, retaining the native-edge
oracle. These observations do not establish full current-head corpus validation
or a whole-pass speedup. Receipts: `.local/three-paths/consolidation/m5-source-facts/`.


## Compact composition forwards source facts

Compact body construction now forwards its immutable source facts to nested
sequence and repetition adapters. Both signs of bounding composition retain the
same owner, while rebuilding their own selected prerequisite relations. Repeated
compact bodies retain the full original input view they previously used. A caller
that supplies value links can retain an empty phase view for later symbolic
conversion: it adds neither phase entries nor value edges.

The bounding oracle checks owner identity in both modes and verifies all 69,984
independent event pairs. Its native-reclosure check still requires the original
completion-to-start link and no software crossing. An obsolete assertion of two
numerical merges was removed after reproducing its failure on the saved M1
candidate; this test no longer claims to force a particular numerical route.
The dedicated numerical-session test remains part of validation.

All ten focused tests pass, including nested compact repetition, bounding,
numerical, sequence and public-pipeline checks. Source review accepted the
forwarding, empty-view semantics and oracle correction. The C++17 incremental
build passes. The compliance prefilter's one brace match is a braced body.
Top-level compact requests still build their initial full-input index; local
prerequisite tracing remains local work. No whole-pass timing improvement or
complete session-wide sharing is claimed. Receipts:
`.local/three-paths/consolidation/m5-compact-facts/`.

## Milestone 5 follow-up: live bridge frontiers

Bridge construction now removes constant-false alternatives after masking a
complete writer/reader group. An unconditional replacement clears the old
frontier before appending the new simultaneous envelopes. Incoming alternatives
already known false are not appended. Conditional alternatives remain until the
expression arena proves them false; no guard-equivalence solver or admission
limit was added.

The diagnostic interface separately counts examined storage bridge candidates,
individual frontier entries remasked, and newly retained live endpoint pairs.
These counts exclude the residual-alias pair loop and boundary-summary work.
At 128, 256 and 1,024 unconditional writer children, candidate visits are 127,
255 and 1,023, with no individual remasking visits. Repeated identical symbolic
guards have the same linear candidate counts and one remasking per replacement.
All-false children produce no frontier candidates. An independent dense conflict
closure agrees across 32 guarded executions, including simultaneous writer
envelopes and skipped intermediate regions. The existing sequence regression
also checks candidate/remasking growth for 8/16/32 separate compact loop children.

The focused bridge and sequence-analysis lit tests pass in the optimized
assertions-enabled build. A separate source review accepted the group handling,
pruning and counter definitions. This does not close the sparse child/cell
participation work, residual alias products, or the full composition milestone.
Local build and test receipts are under
`.local/three-paths/consolidation/m7-selector-recipes/` (the shared validation run
also exercises the in-progress selector work).
