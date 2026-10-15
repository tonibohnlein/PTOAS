# Construction-first checkpoint evidence

This checkpoint is based on M1 `629371af160a5a17fe8f247eb4c53c372fff2cb2` in
`frontier-shared-input`. It is a reviewed implementation checkpoint, not
completion of the six milestones. The [current plan](frontier-family-pipeline-plan.md)
records the remaining work; the [historical log](frontier-family-pipeline-history.md)
retains earlier measurements.

Implementation acceptance, kernel compilation and fixed-protocol capacity are
separate outcomes. The C++ output stage below generates source; it does not
compile a device binary or execute it. No device results were obtained.

## Source and measurement boundaries

`S1` is the refreshed construction checkpoint before lifted arithmetic containment
and cross-call threshold preparation caching. `S2` adds lifted containment. `S3`
also retains immutable threshold preparation across calls for an unchanged root.
`S4` packs operand storage in that immutable preparation. `S5` prunes impossible
same-occurrence crossing tails. `S6` reuses structural identities for immutable
projection-cache outputs. These labels identify local source snapshots, not the
six milestone numbers.
The source and tool manifests in `.local/construction-first/` identify these
boundaries. Retained binaries are under the corresponding `s1-retained/` through
`s5-retained/` directories.

An early snapshot copy preserved timestamps and left compilation objects stale.
Results preceding the forced compilation-input refresh describe those binaries
only. They are not evidence that the final source passed or became faster. The
refreshed source was copied with fresh timestamps, rebuilt, and checked against
the active compilation inputs before the S1 campaigns. Subsequent checkpoints
use the same byte-comparison and fresh-timestamp procedure.

All intensive work is bounded to two aggregate workers. Measurements run serially
with one CPU affinity, a 4 GiB address-space limit, and no overlapping build.
Production stage watchdogs are 120 seconds. A watchdog is unfinished work, not an
allocation impossibility result. Profiling does not affect admission.

## Reviewed implementation

| Component | Change | Independent evidence and limit |
|---|---|---|
| Threshold specialization | Reuse immutable topology and reached input-family dependencies; preserve unaffected IDs; specialize selected branches lazily | Signed/wrap/opaque/integer cases, exact refinement, repeated roots, nested rollback and reused IDs. Specialized nodes and results remain transaction-local. |
| Threshold ownership | Retain source preparation by root, evict removed roots on rollback with ordered range erasure | Existing roots first prepared inside a rolled-back transaction survive; committed inner roots removed by outer rollback cannot lend preparation to reused IDs. |
| Varying boundaries | Ordered prefix/suffix cycle products and exact fixed-point power truncation | Independent full 64-level, noncommuting matrix oracle; no fixed point retains full construction. |
| Prefill relational construction | Exclusive buckets and exact lifted composition containment | Materialized composition/subtraction and finite enumeration agree, including complete required unions, coupled coordinates, a two-coordinate witness, parameter residues and an uncovered second disjunct. Zero explicit composition projections does not mean zero internal emptiness elimination. |
| Sequence crossings | Use certified child order for singleton dynamic crossing endpoints | All 16 fixture guard valuations and nonunit coordinates; incomparable or same-child dynamic groups retain the exact residual path. This does not complete restricted incoming queries through sequences. |
| Selection | Cache successful exact answers before candidate traversal, keyed by full request and audit context | Repeated requests, invalidation, audit/production separation, capability retries and fresh mutable preparations. General ranking by requested unbuilt work remains open. |
| Periodic estimates | Shared core-chain scan and generic graph-relaxation counts | Producer-counter comparisons and overflow-to-unknown checks. Unknown cost does not remove eligibility. |
| Child composition | Summary-only SCC transfer for qualified event-free children with pending payload effects | Independent pending-state and command-graph checks, conditional/loop cases, precise refusals, unchanged child IR. Pending eventful children remain unsupported by this extension. |

Both correctness and architecture reviewers accepted these bounded source
changes. Their source reviews do not establish a full milestone, a successful
kernel compilation, or an allocation result. Tests and measurements are recorded
separately below. The checkpoint commit does not mark a milestone complete;
publication remains a separate action.

## Refreshed S1 and S2 results

S1 focused checks pass **18/18**. Ordered production campaigns pass **49/49**
PyPTO-lib modules and then **22/22** PyPTO captures at both logical and six-ID
allocation stages. Their generated logical/allocation metrics match the earlier
71-case observations. The prepared campaign has **10/11** logical successes,
**8/11** six-ID allocations and **8/11** generated C++ outputs.

| Prepared protocol | Logical | Six-ID allocation | C++ source | Capacity interpretation |
|---|---|---|---|---|
| GroupNorm | Success | Success, split route | Generated | Partial allocation checkpoint; combined output remains open |
| Prefill | 120-second watchdog | Not reached | Not reached | Unresolved proof/construction; no impossibility claim |
| Decode | Success | Refusal | Not reached | Checked nine-ID bound for the retained fixed execution/protocol only |
| Delta | Success | Refusal | Not reached | Actual prefix-protocol reuse remains unresolved |
| Other seven prepared cases | Success | Success | Generated | Accepted assignments for those outputs |

GroupNorm replays pass all eight checked core/subblock combinations, preserve the
baseline logical traces and independently validate physical reuse. The lane-span
regressions also pass. The decode fixture checks command causality and a matching
lower bound against an independent antichain; textual SET-before-WAIT does not
create a causal edge. Its bound is not a universal hardware lower bound.

Three serial paired baseline → S1 runs give Delta logical-analysis medians **19.717 s →
18.517 s**. The short varying case gives **0.218 s → 0.265 s**, so no timing
improvement is claimed there. The baseline optimizer SHA-256 is
`4044d6115dc2d43ac3e7bcabf71df49c062ae846cba6944533286a285421bff5`;
S1 is pinned in `s1-retained/final-tool-manifest.json`. Delta still emits 454,698 control operations;
one sequence-incoming root at `pto.tcvt` after the token loop (input line 261)
contributes 208,060. Preparation
savings therefore do not establish smaller materialized control.

S2 passes the same **18/18** focused checks, including the asymmetric coupled
containment oracle. One serial S1/S2 prefill pair times out at **120.026 s /
120.028 s**. This establishes neither a speedup nor capacity closure. The
containment equivalence and structural projection reduction are separate evidence.

Receipts: `focused-refreshed.*`, `focused-s2.*`, `primary49-refreshed/`,
`secondary22-refreshed/`, `prepared11-refreshed/`, `groupnorm-replay-refreshed/`,
`paired-refreshed/`, `paired-prefill-s2/`, and `s2-source-manifest.json` under
`.local/construction-first/`.

## S3 validation and S4 memory follow-up

S3 passes **18/18** focused checks and repeats the ordered 49/22/11 campaigns.
All 82 case outcomes agree with S1. Logical and allocation metrics agree for
all cases. Persistent GEMM generates ten fewer C++ lines; the logical and
allocated IR differ only in Boolean conjunction association and operand order.
Flattening and sorting those conjunctions makes the complete IR texts identical.
This is a separate normalization check, not evidence of improved device code.
The eight GroupNorm replays and the current-output decode witness replay pass.

Three serial S1 → S3 Delta pairs have median logical-analysis times **18.865 s →
18.165 s** and median peak RSS **378,420 KiB → 423,012 KiB** (about **370 MiB →
413 MiB**). S3 therefore shows a modest measured time reduction with about 11.8% more
peak memory in this workload; the comparison includes S2 as well as the cache. Materialized control
remains unchanged. S4 replaces per-node inline operand vectors in the immutable
cache with packed edges and offset/length ranges; its build, both bounded source reviews, 18 focused checks and the ordered
49/22/11 campaigns pass at the same stages as S3. All eight GroupNorm replays and
the current-output decode witness pass. A separate serial S1 → S4 comparison gives
Delta medians **20.212 s → 20.562 s**, with peak RSS **380,164 KiB → 414,836 KiB**
(about **371 MiB → 405 MiB**). No time improvement is claimed for S4. These pairs
were collected separately from S1/S3, so they are not a direct S3/S4 timing test.

S3 broad coverage did not start: the LLVM Python environment lacks `psutil`,
which lit requires for per-test timeouts. S4 system Python provides that package, but its Python 3.14 default forkserver
then hit a sandbox socket restriction before tests ran. The local runner now
explicitly uses Unix fork workers. Neither harness failure is a test failure.
The earlier S3 debugger attempt was blocked by sandbox `ptrace` restrictions and
produced no inferior stack. An authorized S4 debugger run stopped its own child
after 60 seconds and captured `IntegerProjectionCache::same` → `identity` →
`intersect` within arithmetic storage-selector construction. That observation
identifies the current operation at the sampling instant; it does not confirm
that this run reached the historically reported `11 → 56` composition.

Receipts: `s3-source-manifest.json`, `s3-tool-manifest.json`, `focused-s3.*`,
`primary49-s3/`, `secondary22-s3/`, `prepared11-s3/`,
`s3-corpus-comparison.json`, `persistent-s3-boolean-equivalence.json`,
`groupnorm-replay-s3/`, `decode-replay-s3.*` and `paired-delta-s3/`.
The unsuccessful harness/debugger diagnostics are `sync-broad-s3.*` and
`prefill-stack-s3.*`. S4 receipts use corresponding `s4` names; `prefill-stack-s4.*` records the
successful stack capture (the timed debugger exits 124 after intentional SIGINT).
`check-boolean-association.py` reproduces the S3/S4 complete-IR normalization checks.

## Subsequent bounded construction changes

S5 skips a cross-child prefix query when its suffix would be a completion reaching
its own start at the identical retained port. This proof invokes no child callback.
All other queries preserve the previous prefix-first order, including missing
query capabilities and callbacks that can record construction errors. The
regression compares dense closure for all four presence assignments in three
accepted fixtures and checks a fourth fixture's precise refusal. Aggregate
counters record entries, aliases, skipped tails and actual prefix/suffix requests.
Both reviewers accept this bounded change; 18 focused checks pass. Three serial
S4 → S5 Delta pairs give medians **18.458 s → 18.311 s**. This is not completion
of the restricted-incoming service through sequences. The six emitted crossing
query profile records all contain zero work for this workload. They therefore
do not establish that this pruning reduces Delta's expensive incoming root;
attribution through the nested producer remains open.

S6 registers only final, immutable projection-cache outputs in an address-to-ID
directory. Caller-owned systems still undergo structural hashing/equality on
every request, because their contents can change at a fixed address. Reset clears
the directory before owners; copying/moving the cache is disabled. Tests cover
large integers, repeated owned queries, external reassignment, growth, independent
owners and reset. Both source reviews accept the bounded ownership change. The
build and **19/19** focused checks pass, followed by all 49 primary and 22
secondary production cases at logical and allocation stages. The prepared
campaign again has 10 logical successes, eight six-ID allocations and eight C++
outputs. All 82 production-case outcomes match S1; the only generated-metric
difference is Persistent GEMM's independently checked conjunction reassociation.
The eight GroupNorm replays and current-output decode witness replay pass.

Three serial S5 → S6 Delta pairs give medians **18.517 s → 18.523 s**, with median
peak RSS **416,976 KiB → 415,560 KiB**. No timing improvement is established.
Both prefill runs reach their 120-second watchdog (**120.037 s / 120.035 s**).
The cache's structural reuse tests pass, but these production measurements do not
establish reduced materialized control or resolved prefill construction.

Receipts: `s6-source-manifest.json`, `s6-tool-manifest.json`, `focused-s6.*`,
`primary49-s6/`, `secondary22-s6/`, `prepared11-s6/`,
`s6-corpus-comparison.json`, `persistent-s6-boolean-equivalence-repro.json`,
`groupnorm-replay-s6/`, `decode-replay-s6.*`, `paired-delta-s6/`,
`paired-prefill-s6/`, and `s5-s6-delta-construction.json`.

## Broad synchronization validation

The S6 broad run completed all 228 selected synchronization tests: **198 passed,
28 failed and two timed out**. It ran with one worker and a 120-second per-test
watchdog. `sync-broad-s6-lit.json` records individual results;
`broad-s6-input-manifest.json` pins fixtures and helpers. This is synchronization
coverage, not a passing full PTOAS suite.

Both retained baseline and S6 tools time out at 90 seconds on the exact
varying-parity rejection input from `sync_guarded_rotating_analysis` and the
first generated repeated-phase case `(period=2, outer trips=0, inner trips=0)`.
The original guarded check completes its preceding concrete equivalence checks;
the final nonadjacent case is unreached. Repeated-region's first RUN passes its
14 nested closures; later phase cases and the nested-allocation RUN are
unreached. These are bounded construction failures, not failed equivalence
assertions or capacity bounds.

Fresh direct comparisons reproduce identical output/diagnostics for storage
views, reuse recognition, arithmetic-program output, access regions, the
finite-visit assertion and missing persistent-refresh-set certificate. Both
tools fail arithmetic-extended at `unsupported_period`. The standalone arithmetic
recognizer has identical S1/S6 output, but no retained pre-S1 baseline binary is
available; its baseline attribution remains open. The initial finite-visit-types
probe supplied an unsupported extra input argument and is excluded; only the
`finite_visit_types-exact` supplemental receipt uses the actual RUN command.

The 27-fixture rerun uses retained baseline tools with the same S6 fixtures,
checkers and resource limits. For **25 fixtures**, the executed RUN lines and
complete diagnostics match after normalizing tool/temporary paths and removing
shell command traces. Conditional-region testing fails the same refusal
expectation (`prepared=True`, `unchanged=True`); only construction counters
differ. Its reported expressions decrease from 410 to 405 and selector comparisons
from 40 to 38. This does not make that assertion pass.

The remaining loop-session fixture fails on baseline but reaches the broad S6
watchdog. A subsequent serial pair isolates its exact `--regional-session-checks`
command: both return the identical `whole-region logical export did not complete`
diagnostic, at **83.522 s / 86.346 s**. This one pair attributes the observed
export failure; it is not a speedup measurement or proof of the broad watchdog's
cause. Composed-phases reaches the same whole-fixture 120-second watchdog during
RUN 10 after RUNs 8–9 succeed; its internal stalled query remains unidentified.
Rotating-region testing fails during logical insertion for coprime banks `(13,11)`
with default visits, with the same `std::bad_alloc` diagnostic under the 4 GiB
address-space limit on both tools;
its billion-visit comparison is unreached.

These observations attribute reached failure stages, not later RUN lines. The
rerun is not a baseline-source test suite. Receipts are under `attribution-s6/`,
`baseline-lit-s6*`, `broad-attribution-s6.json`, `attribution-normalized-s6/` and
`final-checkpoint-receipt.json`.

## Changed-code checks

The native builds use the repository C++17 configuration and at most two workers.
The earlier changed-code prefilter reports 67 matches: 65 multiline C++ conditions
whose bodies have explicit braces, and two Python process-use matches in the
fixed-protocol replay checker. The brace disposition receipt includes source
excerpts and balanced-condition checks. The Python checker uses an argument
vector, no shell, a resolved absolute executable path, a 30-second timeout and
checked return status. These matches are classified false positives under the
applicable brace and process-execution rules; the prefilter itself does not
report a clean exit. A real overlong line was fixed. `git diff --check` passes.
No full static-analyzer or device-environment compliance claim is made.
The executable-resolution hardening follows the S6 C++ checkpoint; its fixed
fixture, current-output replay and invalid-executable refusal are independently
recorded in `final-checker-verification.json`.
The pre-commit rerun reports 65 matches, all verified multiline conditions with
explicit braces, and no warnings (`compliance-commit.out` and
`compliance-commit-brace-disposition.json`).

## Caller ownership inventory

The inspected convenience interfaces have these supported callers. No interface
is deleted merely because it resembles another producer; deletion requires
migration of every supported caller and agreement of the independent oracles.

| Interface | Current callers | Ownership/deletion requirement |
|---|---|---|
| `analyze`, `minimumDemands`, `prepareLogical`, `attachAllocation` | Production session/module coordination and tests | Keep the request boundary, exact retained mathematics and fresh mutable plans. `minimumDemands` already wraps a request. |
| `analyzeNumericalRegion`, `analyzeArithmeticRegional`, `analyzeFiniteVisit`, `analyzeVaryingBoundary`, `analyzeFiniteExpansion` | Explicit backend audit tests in `pto-sync-input-test` and `NormalizedControlChecks` | They already accept requests, but audit entry points remain public. Isolate them without changing production selection or removing independent backend coverage. |
| `recognizeAllForms`, `certifyRegions`, `costRecords`, profiling reports | Explicit audit/diagnostic callers | Keep audit construction and copied snapshots separate from production cost claims. |
| Specialized guarded/numeric/arithmetic services | Production child/phase adapters and tests | Preserve full observed-binding/context checks and immutable mathematics; fresh target arenas own imported expressions. |
| `analyzeExplicitFunction`, `analyzeSequenceFunction`, `analyzeArithmeticFunction`, `analyzeArithmeticPeriodicFunction` | Production backend dispatch | Migrate retained producers before deleting orchestration; preserve generic mathematical kernels. |
| `analyzeNumericCandidates`, `prepareNumericCandidateExports`, `analyzeFiniteVisitCandidates` | Direct audit/test callers | Isolate explicit audit workloads while preserving independent backend coverage. |
| `prepareRotatingFunction`, `prepareBoundedLifetimeFunction`, `prepareArithmeticPeriodicFunction` | No in-tree callers found in tracked `include/`, `lib/`, `tools/` | Assess external API/export contracts before deletion; absence of in-tree callers alone is insufficient. |

## Open gates

The full restricted-incoming sequence service, general unbuilt-capability ranking,
changed-representation sequence upgrades, prefill transport/export and allocation,
Delta prefix reuse, combined GroupNorm output, arithmetic endpoint migration,
complete mapping-cost selection, and pending eventful-child contracts remain open.
No new runtime analyzer, general solver, admission cutoff or silent fallback was
introduced to close them.

The next selection change needs a read-only view of retained capabilities and
separate estimates for outstanding demands, queries, selectors and endpoint
qualification. Current request-keyed form ordering does not account for work
built by intervening requests. Numerical preflight and arithmetic recognition
remain candidate qualification work, and numerical-sequence adaptation depends
on previously retained numerical mathematics. Unknown costs must remain eligible
without violating stable ordering or dispatching that adapter before its producer.

Unchanged-coordinate endpoint upgrades already reuse parent crossings. Selective
reuse after a changed child representation needs dependency provenance: callback
owners, port/selector coordinates, storage partitions and alternate paths through
intermediate children can all affect a crossing. Unchanged crossing endpoints
alone do not certify an unchanged guard. The existing complete parent rebuild
remains the exact path until those dependencies can be tracked and checked.

The earlier partial broad run and seven-command comparison remain historical
evidence. Current attribution above does not make the suite green or cover RUN
lines skipped after an earlier failure. The three historical
`decode_build_swa_metadata` guard-equivalence cases remain unproved. Validation
and full milestone acceptance remain open.
