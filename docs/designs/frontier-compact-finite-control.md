# Compact finite synchronization control

Status: implemented and accepted by architecture and compiler reviewers for the
qualified finite-control scope. Validation: 266/267 Frontier tests and 302/304
broad tests; both failures and the separate allocation-cost limitation remain
explicit below.
Baseline: `e4cd258888182087fbb9cae0de0fa25c9df1badd`.
This follows the qualified [finite consolidation](frontier-finite-consolidation.md)
and its [measurement receipt](frontier-finite-qualification-results.json).

The objective is to reduce construction work, generated control size and evaluated
control work for the existing finite synchronization protocol. Minimum demands,
matching and accepted payload order are constraints, not cost terms. This is one
reviewable implementation milestone with four work packages below; instrumentation
alone does not complete it.

## Baseline and scope

At size 64, the production combined route retains 509 added arithmetic operations
for the concrete fixture, 575 for the read-heavy fixture, 12,830 for guarded writers
and 157,272 for guarded composition. Explicit regional composition retains 8,896
for the last fixture with the same checked logical behavior. These are total
added arithmetic counts, not guard sizes. They establish a representation gap;
they do not establish which transformation is responsible or that choosing the
regional route unconditionally is sound or economical.

Use the existing finite owner, regional exports, sequence composition, publication
policy, expression service and family allocation/materialization. Do not add a
parallel finite backend, kernel recognizer or scalar emitter. Keep concrete
analysis numerical. Do not expand the scope to unrelated periodic/prefix kernels,
alternative protocols, runtime analysis or a general Boolean solver.

## 1. Attribute the growth

Extend the existing diagnostic and finite-test infrastructure. Measure after
mathematical query construction, finalized protocol recipes, selector/physical-map
composition, scalar materialization and generated-code cleanup. Retain provenance
from each large root to its producer, original cut and requested output.

| Quantity | Required interpretation |
| --- | --- |
| Guard roots | Actual finalized SET, consumer-WAIT, cleanup-WAIT and barrier predicates, including their controlling branch conditions; report demand predicates separately. |
| Distinct guard nodes | Union of reachable expression nodes for guard roots, including arithmetic dependencies of comparisons. Count once within an arena and compatible binding environment. |
| Largest guard | Maximum reachable-node count for one root, with producer/cut and occurrence binding. |
| Shared nodes | Nodes reachable from at least two distinct roots; also report sum of per-root sizes versus union size. Structural sharing and legal emitted sharing are separate measurements. |
| Selector/ID work | Reachable nodes for occurrence selection and physical IDs, separately from guards. Report exclusive nodes and overlaps; do not sum overlapping sets as total control size. |
| Construction work | Nodes created, traversal visits, query records inspected/contributed, substitutions, copied entries and fresh versus reused bindings. Report work later discarded as well as live output. |
| Emitted control | Unique generated operations by the same root roles, total union, command sites and branch-local placement. Exclude original payload control from added-work totals. |
| Evaluated inactive tests | Guard evaluations yielding false and actual executed guard-dependency operations, including tests shared by several commands. Report skipped evaluations separately; never charge the whole DAG when only one branch executes. |
| Total evaluated control | Actual unique executed helper operations, selector/ID work and role overlaps for each checked execution. Shared work is counted once in the total, even when attributed to multiple outputs. |

An expression ID alone is not a binding identity. Scope accounting and reuse by
arena, original cut, occurrence translation, historical/private-state version and
availability environment as applicable. Do not imply that identical expressions
at different cuts can automatically share emitted values. Preserve rollback and
owner lifetimes. Diagnostic traversal must itself share topology discovery rather
than introducing a root-by-whole-DAG rescan; report diagnostic overhead separately
from uninstrumented compiler timings.

Pin the source and tool hashes, flags and selected producer/representation for
each route. For the composed fixture, compare public dispatch with explicitly
composed exports at the first stage where their work diverges. Attribute costs to
retained mathematics, crossing/query construction, publication recipes and lowering
before choosing the repair.

## 2. Fix construction in the existing production path

Use the attribution to preserve the smaller exact representation where its
contracts apply. Investigate repeated crossing queries, repeated substitution,
reconstruction of selected endpoints and requested outputs that build unused rank
components. Share or restrict construction under the existing binding contracts;
keep known selectors and numerical answers through regional exports.

If selection contributes to the gap, compare marginal requested work using the
existing request machinery. A different decomposition or producer must establish
the same accepted demands and selected communication protocol before its smaller
control is admissible. Do not hard-code the fixture or unconditionally prefer a
route based on this one comparison. Retain the valid incumbent when qualification
or equivalence is unavailable.

The production route must exercise the repair. A smaller test-only composition,
deferred work reconstructed during lowering, or a smaller final graph after equally
large construction is insufficient. Remove any replaced production adapter or
duplicate construction; record its callers and deletion gate. Generic kernels and
independent semantic oracles remain available.

## 3. Add enforceable scaling bounds

Extend `sync_finite_qualification.pto` and `sync_finite_emission_scaling.pto` rather
than starting another campaign framework. Keep independent small semantic oracles
and geometric public-pipeline checks through saved/reparsed and combined output.

For each supported fixture class, specify the counted input size, applicable
premises and formulas for constructed nodes, traversal work, live control and
evaluated work. Freeze concrete constants in regression assertions before accepting
the optimized implementation; a table of measurements alone is not a bound. Do not
raise limits automatically to accommodate a regression.

* Concrete and read-heavy subclasses: require linear construction and live control
  in supplied occurrence/effect incidences plus supplied expression size, with fixed
  pipe count and the tested bounded-write/input-binding premises. Require linear
  evaluated control for the fixture's execution length, counting inactive tests.
* Write-heavy finite inputs: charge `W = sum_x(w_x * m_x)` explicitly. Do not call
  a pair-record representation linear in source sites when its output is quadratic.
* Guarded composition: start with the existing alternating fixture at sizes
  8, 16, 32 and 64, extending safely after attribution. Require an explicit
  quadratic-or-better envelope in fixture size for construction, traversal and
  emitted control, plus a strict reduction of both constructed and live control
  against the pinned production baseline. Freeze the envelope coefficients from
  the justified construction and reviewed measurements, not from a permissive
  post-hoc fit. Claim linearity only for a separately established subclass.

Test balanced and skewed valid parenthesizations, empty/untouched children, shared
and independent guards, arbitrary record renaming, nested skipped conditions and
repeated finite invocations. General bounds must name supplied crossing/interface
size and binding multiplicity; do not hide those dimensions in a source-size
claim. Different partitions require semantic equality, not identical graph sizes.

Keep guard/selector/ID categories visible in assertions so shifting work between
them cannot masquerade as an improvement. Compare evaluated total control on every
checked valuation; investigate regressions rather than choosing by static command
count alone. Record three serial interleaved timing pairs per representative
fixture/output route with the same build configuration and no overlapping heavy
jobs. Report full process time and available exclusive stages separately.

## 4. Preserve semantics and qualify the result

Require unchanged active minimum demands and accepted realization-order guarantee.
Compare original payload occurrences, all four event-reachability answers, boundary
occurrences, actual publications/acquisitions, matching, cuts, coactive command
order, invocation history and cleanup. Harmless provenance renaming may be
normalized only across plans; matching within each plan uses its full identity.

Exclusion uses finalized command activity: exclusive demand predicates do not
justify merging simultaneous early publications. Each publication has exactly one
consumer or cleanup acquisition on its admitted execution. Preserve unavailable
and skipped-value semantics, machine wrapping and state versions. No extra barrier,
boundary drain, implicit resource reset, weakened predicate or lost communication
is permitted to meet a size target. Keep the same certified allocation for
representation-only changes; a changed ID mapping needs independent reuse evidence
for the same protocol. Retain precise allocation refusals.

Run focused component/semantic and scaling checks as construction changes land.
At acceptance, rerun the finite/publication/deferred/module-ownership gate, the
49-module primary corpus against the pinned baseline and the full synchronization
gate. Record inherited failures separately; do not relax expectations or timeouts.
Device compilation and latency remain separate outcomes, not implied by IR or C++
success.

Completion requires the production repair, enforced structural bounds, unchanged
semantic checks, paired measurements, a deletion ledger and machine-readable
receipts. Correctness and architecture reviewers must accept the complete batch
before its commit. Use incremental builds and at most two aggregate heavy workers;
do not push automatically. Report the optimized subclasses and remaining limits,
not a universal small-guard guarantee.

## Implemented construction and accounting

The production finite reducer now projects a *fixed reference-rank threshold*
through the numeric choices and masked maxima that it already constructs. Under
an incoming candidate's existing latest-source predicate, that candidate's rank
is a constant; there is no need to emit a comparison against a selected numeric
rank containing the full earlier history. The numeric frontier remains the
query/analysis representation. The projected predicate denotes the same retained
edge, and the existing family path owns its publication, cleanup and lowering.

The first supported subclass requires one identical atomic presence predicate per
pipe (constants are allowed), no additional atomic inputs in generator/native
edge guards, total expressions on defined inputs, and inputs available at the
original function entry. This last check prevents guard simplification from
silently changing an early-publication protocol. Independent per-occurrence
presence, unavailable inputs and unsupported expressions keep the original path.
There is no enumeration of Boolean valuations in production and no new recognizer,
backend, allocation policy or emitter.

For a recorded masked maximum, `max(old, active ? incoming : old) < K`
is equivalent on defined inputs to `old < K` and
`select(active, incoming < K, true)`. A recorded choice distributes the fixed
comparison into its arms while keeping the choice lazy. Grouped incoming tests
substitute a candidate's fixed label only under its exact latest-source predicate.
These transformations do not change the numeric join decisions or introduce
the tested incoming edge into its own exclusion proof.

Projection uses a reducer-local memo of `(numeric node, fixed threshold)`, retaining
lazy selection at masked arms. A range can discharge a comparison; otherwise the
projector follows only recorded numeric choices/maxima. Its cumulative discovery
budget is `8 * (recorded decisions + requested comparisons + 1)`. Exhaustion
retains the exact original numeric comparison, including failed-attempt work in
the counter. Memo entries are bounded by visits plus requests. Qualification
DAG visits are counted separately in the existing guard-qualification ledger.
That ledger counts distinct visited nodes per scan plus the explicit payload
qualification positions; it is not a count of every repeated root-list entry.
Supplied candidates, writer-pair visits, recipe root uses and cut counts are
reported alongside it so repeated input inventories remain visible.
The bound concerns inspected work and entries; ordered maps still charge their
ordinary logarithmic lookup cost.

The prior large control was not caused by missing demands or extra handshakes:
numeric rank/selection expressions were carried into retention guards and then
lowered at many original cuts. A union counted once in the source arena is not
a placement cost. This implementation therefore uses the stated structural
qualification, not a global guard-union comparison to choose emitted code.

The root profiler discovers dependency topology once and propagates root-membership
bits in batches of 64. Scratch storage is linear in discovered nodes; membership
work is reported separately and can grow with the number of roots. Profiles
separate guard, occurrence-selector and physical-ID dependencies and their overlap,
including the original cut and representative largest-root provenance. Arena
sharing is explicitly distinguished from emitted sharing. Original arithmetic
can be a control dependency, so root unions are not labelled wholly generated.

The trace profiler counts executed arithmetic instances, selected branches and
loop-carried bindings. Static possible roles also account for eagerly evaluated
ID/selector arithmetic even when its command is inactive. Unattributed executed
arithmetic is reported separately; it includes payload work and dead helper
constants with no command use. The role union plus unattributed work equals total
arithmetic executions. Added-work comparisons subtract the original input trace.
Skipped branches do not acquire fictitious arithmetic evaluations.

### Regression envelopes and their scope

The existing public-pipeline driver enforces the following envelopes for its
fixed two-pipe templates; `n` is its supplied template size, not an arbitrary
program's source size. Concrete and read-heavy templates have linear supplied
incidences. Writer-heavy and alternating composed templates have quadratic
writer-related candidate/record inventories.

| Measured quantity | Concrete/read-heavy | Writers/composed |
| --- | --- | --- |
| Constructed and live added scalar operations | `10n + 8` | `4n² + 16n + 16` |
| Added evaluated arithmetic per checked execution | same scalar bound, and no baseline regression | same scalar bound, and no baseline regression |
| Finite mathematical arena nodes | `16n + 16` (guarded only) | `16n² + 64n + 32` |
| Largest emitted guard dependency slice | 1 concrete; `6n + 4` guarded | `6n + 4` |
| SET/WAIT sites | `8n + 8` | `8n + 8` |

These are conservative regression envelopes for the supplied constructions, not
universal theorems about arbitrary guarded inputs. They account for both original
cuts and pair inventory: the quadratic cases allow four scalar operations per
potential pair plus linear endpoint work. The mathematical envelope separately
allows the retained numeric frontier and bounded threshold work. Counters also
check writer-driven pair visits against `2W`, projection visits/entries against
the exact local budget, and static/dynamic role partitions. Repeated cumulative
profile snapshots are never added together. The physical scalar construction count
is the maximum of existing preparation and cumulative family-emission counts,
avoiding double-counting the materialized preparation.

**Allocation is a separate remaining scaling cost.** The `prepared` profile is
printed after allocation support, and its whole shared arena includes allocation
proof expressions. It must not be confused with the requested guard slice or the
finite mathematical result. The unchanged fixed-record allocator can inspect
quadratically many record pairs; the composed fixture itself has quadratically
many records in `n`. Its recorded pair work is checked against `2R²`, where `R`
is retained record membership, and its arena occupancy is reported explicitly.
This milestone does not establish quadratic full-compiler work in `n` or repair
that allocator. It establishes bounded finite control construction and emission
for the stated subclasses while exposing the separate allocation cost.

### Replacement ledger

| Existing component | Change | Remaining use |
| --- | --- | --- |
| Finite fixed-reference retention comparison | Project constant thresholds through known rank decisions under qualified shared presence | Numeric fallback retained for unsupported predicates or exhausted local work budget |
| Numeric frontier and regional queries | Retained unchanged as the authoritative mathematical/query representation | All existing query and allocation consumers |
| Family publication, allocation and emission | No replacement or additional implementation | Existing selected protocol and certified allocation |
| Existing scaling and trace drivers | Extended in place with role accounting and enforced construction/evaluation bounds | Semantic, physical-reuse and public/saved pipeline oracles |

No complete production adapter became unused in this change. The optional local
projection replaces the qualified retention-comparison construction inside the
existing reducer; generic numeric construction is still required by other inputs.

## Qualification results

The [machine-readable receipt](frontier-compact-finite-control-results.json) pins
inputs, tool hashes, changed source hashes, flags, root profiles, construction
snapshots, per-valuation control work and all timing pairs. The comparison uses
`e4cd2588`, the immediate baseline, with the same configured native build and
shared LLVM. Only native incremental targets were built; LLVM was not rebuilt.

All 16 geometric cases (`n=8,16,32,64` for concrete, read-heavy, writers and
composed) pass the enforced bounds and the independent payload-order/command
matching/physical-reuse checks. All four guard bindings are checked. The composed
cases also compare explicit regional logical, saved and combined output against
the production path by executed communication identity, not static record names.
The numeric/guarded event oracle passes 6,144 valuations and 960 anchored windows;
a separate 66-valuation test forces threshold-budget exhaustion and checks its
fallback. Unit accounting checks include shared roots across 64-root batches,
masked unbound operands, conditional results, loop-carried values, eager inactive
ID arithmetic, and skipped branch arithmetic.

| Composed size | Baseline constructed scalars | Candidate constructed scalars | Baseline live arithmetic | Candidate live arithmetic | Candidate largest emitted guard |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 1,817 | 238 | 1,816 | 225 | 8 |
| 16 | 8,665 | 750 | 8,664 | 721 | 16 |
| 32 | 37,721 | 2,542 | 37,720 | 2,481 | 32 |
| 64 | 157,273 | 9,198 | 157,272 | 9,073 | 64 |

Both saved/reparsed and combined routes retain the same arithmetic counts in this
table. At size 64 the SET/WAIT site counts remain 127/127. Across guard bindings
`00,01,10,11`, added evaluated arithmetic falls from
`[0, 78194, 78570, 157272]` to `[0, 4283, 4282, 9073]`. The ordered logical communications, matching and payload order remain unchanged;
each physical assignment independently passes the reuse oracle. This comparison
does not require identical numeric IDs across assignments. The concrete,
read-heavy and independently guarded writer outputs retain their baseline
construction and execution counts. The explicitly regional size-64 result remains
8,896 arithmetic operations; this change improves the normal production path
without forcing its decomposition to match that test route.

Three serial interleaved uninstrumented timing pairs at size 32 give median
candidate/baseline ratios of 0.759 for saved/reparsed output and 0.706 for combined
output on the composed fixture. Other fixture ratios range from 0.991 to 1.019;
no speedup is claimed for them. These are full-process compiler measurements on
the fixtures, not device latency or another kernel's planning speedup. Diagnostic
profiles are collected separately from timed runs.

At composed size 64, the finite mathematical arena has 25,290 nodes and the
prepared command-root union has nine nodes. The whole arena **after allocation**
has 2,175,206 nodes. Allocation inspects 8,386,560 member pairs and issues
4,193,280 proof requests. Those numbers are retained in the receipt; the smaller
control measurements do not erase that remaining construction cost. The existing
fixed-record allocator is the next separately attributable scaling limitation.

The primary PyPTO-lib comparison passes 49/49 logical compilations and 49/49
six-ID allocations. Every logical and physical output is byte-identical to the
pinned baseline. This campaign requests neither corpus-wide C++ output nor device
compilation/execution.

The broad synchronization filter selects 304 tests: 302 pass and two fail.
Its frontier-named `pto/sync_` subset is 266/267, with the inherited
`sync_repeated_region.pto` 90-second phased-repeat subprocess timeout. All 20
finite/publication/deferred/module-ownership focused tests pass within that gate.
The additional broad failure is `insert_sync_slot_transitive.pto`, a legacy
InsertSync `SYNC-NEXT` expectation using the separately built CLI/Python runtime.
That target was not rebuilt by this milestone, and its legacy sources and fixture
are unchanged. Its tool hashes and scope are retained in the receipt. It is not
counted as passing or repaired. No expectation or timeout was relaxed.

The changed-code checker reports 20 `G.FMT.11-CPP` findings. Each inspected body
has explicit braces; these are the checker's nested-parenthesis regex false
positives, retained with source excerpts in the receipt. No suppression was added.
`git diff --check` passes. Both architecture and compiler reviewers gave final scoped acceptance after
reviewing the source and completed receipts. Their decisions are recorded in the
receipt; neither acceptance promotes the two failures to passes or establishes a
whole-compiler quadratic bound.

Reproduce geometric qualification with the configured test binaries:

```sh
python test/lit/pto/Inputs/check_finite_emission_scaling.py \
  --optimizer <candidate-pto-test-opt> --trace-tool <candidate-pto-sync-input-test> \
  --baseline <pinned-baseline-pto-test-opt> \
  --fixture test/lit/pto/sync_guarded_publication.pto \
  --output <disk-artifact-directory> --sizes 8,16,32,64 --pairs 3 --timing-size 32
```

The regular `sync_finite_emission_scaling.pto` test runs the same assertions at
sizes 2 and 4 without timing. The larger pinned campaign establishes the measured
production improvement; the test oracle does not require an older compiler at
ordinary test time.
