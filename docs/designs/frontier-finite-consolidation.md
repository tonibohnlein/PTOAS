# Finite analysis consolidation

Baseline: `1953d248c`. Implementation status: M1–M5 accepted within the contracts below.
The inherited repeated-phase timeout remains open; this is not an all-green gate.

## Contracts

One retained finite service owns either concrete numerical analysis or guarded
circuits. Concrete qualification includes activity, modeled accesses and
prerequisites. Regional export does not reconstruct concrete analysis symbolically.
Child intervals are contiguous on every execution and preserve indivisible macro
envelopes, original occurrence identities and the common storage interpretation.
Parents carry external frontiers; children do not drain or reset event resources.

Direct realization uses the existing publication policy. Per logical identity,
SET count is zero or one, and consumer-WAIT plus cleanup-WAIT count equals SET
count. Cleanup is in the owning invocation's terminal epilogue. Identities and
history bindings include enclosing occurrences. Grouping uses actual finalized
command activity, never just demand activity: complementary demands may have
simultaneously active early SETs. Unavailable history is a missing realization
capability, not permission to replay payloads or introduce private state.
Order equality requires the existing placement, readiness and invocation proofs.
Allocation certifies this finalized graph. T publications outstanding until
terminal cleanup require T IDs for that execution; there is no implicit reset.

## Milestones

1. One finite input/result and request lifecycle, retaining numerical and guarded
   kernels; migrate root/session ownership and remove duplicated preparation.
2. One regional adapter, lazy queries and boundaries, numerical concrete export;
   migrate sequence children and compare valid decompositions semantically.
3. One qualified direct-publication recipe construction for finite internal and
   crossing demands; per-publication identity/activity/history invariants.
4. Existing family normalization, allocation and materialization consume the
   finalized command graph; both combined and saved split output validated.
5. Final semantic/scaling/corpus campaign. Tests and counters land with their
   corresponding migrations, not only in this final milestone.

Every milestone requires source review and relevant runtime validation before
commit/advance. Use incremental builds and at most two aggregate heavy workers.
Preserve public pass options and archive compatibility. Keep distinct existing
protocols outside this migration and preserve mathematical results on export
failure. Do not claim unrelated periodic/prefix issues fixed.

## Validation

Independent concrete event closure/covers, all Boolean valuations on small cases,
flat versus composed original occurrence identities, all four event-kind queries,
actual boundary occurrence/absence, RAW/WAR/WAW/RMW and hardware protection,
simultaneous envelopes, conditional prerequisites, nested/skipped guards,
wrapping/index widths, empty children and enclosing visits. Check command order,
consumer/cleanup exclusivity, complementary early publications, terminal capacity,
transactional failures and threaded module-owned saved lowering.

Count work inspected as well as output: access/pair visits, residual work, query
records inspected/contributions, DAG visits/constructed/reachable nodes, cache
hits, copied entries/bytes, constructed/live scalar code, command sites and
executed inactive control. Scale concrete chains 32--4096; guarded few-writer
cases 16--1024; write-heavy cases 8--128 with W=sum(w_x*m_x) charged explicitly.
Different decompositions require equal semantics, not identical circuit sizes.
Use three serial interleaved timing pairs for performance claims and finish with
the full synchronization gate. Device qualification is separate.

## Deletion ledger

| Existing owner/adapter | Replacement | Remaining callers / deletion gate |
| --- | --- | --- |
| Root explicitAnalysis + MathematicalResult::explicitDemands | finite owner, concrete strategy | Removed in M1; callers use finiteDemands/concreteDemands() |
| MathematicalResult::finiteGuardedDemands | finite owner, guarded strategy | Removed in M1; ExpandedFinite remains a distinct coordinate adapter |
| Explicit/guarded root preparation dispatch | finite recipe request | Root dispatch replaced in M1; direct construction shared in M3 |
| Sequence explicit-child export reconstruction | common finite regional export | Removed in M2, including prefix compatibility; no parent-owned concrete cache remains |
| Explicit/guarded endpoint construction loops | common direct recipe builder | Duplicate publication/cleanup assembly removed in M3; demand enumeration remains producer-owned |
| Distinct prefix users and expanded-loop analysis | unchanged qualified protocols/adapters | outside this migration, caller inventory must preserve support |

## Evidence

Artifacts: `.local/finite-consolidation-20261016/`.
Baseline focused gate passed 11/11 in 60.48 seconds before source changes. The previous full gate
passed 262/263 synchronization tests; its repeated-phase watchdog remains open.

### M1 source review and validation

The root/session now shares `FiniteOccurrenceInput` and a tagged `FiniteAnalysis`.
Concrete results remain numerical. Guarded results retain their arena/state.
Original finite preparation uses one request; expanded-coordinate preparation
remains separate. Concrete prefix compatibility uses a borrowed strategy view.
The synthetic prefix fixture deliberately has no original-input owner and cannot
request finite preparation; production factories validate source provenance.

Architecture and compiler reviewers accepted this scope. The compiled focused
gate passed 19/19 in 81.59 seconds (`m1-focused.json` and `m1-focused.out`). The private PhaseIndex call identified by review was removed. Incremental
build of PTOASCompiler, pto-test-opt and pto-sync-input-test passed. The build's
covered-switch warning is in unchanged ProgramRecognition.cpp.

The changed-code check's line-length finding was fixed. Its nine remaining
G.FMT.11-CPP matches are multiline conditions with braced bodies (AnalysisRequest,
FiniteAnalysis, NormalizedControlChecks, SyncLogicalInsertionChecks and
WholeInvocationIdentityChecks); inspected and classified as regex false positives.
No production analyzer or device validation is claimed.

### M2 numerical regional export and child ownership

The finite service now exports concrete rows and stored numerical boundaries
through the existing regional interface. Only requested answers are wrapped in
shared expressions. Guarded exports retain the same finite owner. Original
contiguous slices share validated source facts; creating a slice copies only its
own roots. Legacy reference-based callers use an explicitly borrowed input, and
cannot export a detached prefix owner without an owned source context.

Sequence children retain this finite result instead of a separate by-value
explicit analysis. The parent-dependent concrete query/preparation adapter and
its recipe-template cache have been deleted. Immutable concrete recipes are
cached by the finite owner and instantiated with fresh mutable inventories.
Prefix collection and sequence prefix contributions use the retained owner;
expanded and numerical periodic coordinate adapters remain separate.

The new oracle tries every two-way contiguous partition, including empty
children, compares active demand pairs, all four event-kind queries and actual
storage boundary occurrences, and exercises callbacks after their caller releases
the result. Original occurrences, not expression or record identifiers, determine
agreement. Existing guarded, envelope, prefix and retry tests cover those clients.

Both reviewers accepted the source scope, conditional on runtime validation.
Review caught and removed whole-root copying per slice; physical-fragment cost
accounting was retained. The incremental compiler/tool build and direct partition
oracle passed. The 17 changed-code checker matches are braced conditions/loops
misclassified by its line-oriented G.FMT.11-CPP pattern; each was inspected. No
new compiler warning or production-analyzer result is claimed.

The M2 focused gate passed 32/32 in 121.68 seconds (`m2-focused.json` and
`m2-focused.out`), including guarded publication, numerical finite crossings,
regional/session retries, conditional upgrades, overlays and prefix clients.
This is a focused gate; the final synchronization campaign remains M5.

### M3 common direct-publication construction

Concrete finite, guarded finite and contextual sequence crossings now use
`prepareDirectPublicationFamily` in the existing publication service. It returns
one owned, fully qualified family and its actual command predicates before the
caller appends any plan inventory. Numerical grouped crossings retain their
compact builder. The three producers retain demand enumeration and allocation
coordinates, but their duplicate contextual SET/WAIT/cleanup assembly is removed.

The helper accepts one zero-displacement record, keeps member and enclosing
invocation identity, and shares the existing direct/early publication policy.
Terminal-choice evidence owns its predicate arena. Local barriers check consumer
availability without acquiring publication or cleanup. Sequence allocation still
uses uniform demand activity for selective publication and publication activity
for terminal-surviving early publication. No realization-order guarantee is
promoted by this refactor; the existing guarantee/placement checks remain owners.

Architecture review caught an initial availability-policy bypass. The corrected
helper accepts the producer's availability restriction at every required cut;
sequence passes its existing no-replay policy. The regression distinguishes a
pure guard the generic service can replay from the same guard restricted to
original availability, which requires early publication and cleanup.

Additional tests require two simultaneous early SETs for complementary future
demand guards, with exactly one consumer-or-cleanup WAIT for each identity.
They check absent sources, both producer routes, alias policies, two-ID success
and one-ID refusal. API tests cover invalid cuts, multi-member refusal, local
barriers, required-cut denial and retry with unchanged original IR. Existing
selected-repeat and missing-history oracles remain in the gate.

Both source reviewers accepted the corrected M3 scope. The incremental build
passed without new warnings; the focused gate passed 24/24 in 120.86 seconds
(`m3-focused.json`/`.out`), including saved contextual/mixed/cleanup output and
combined physical output. The nine changed-code G.FMT.11-CPP matches are
multiline braced conditions, reviewed as false positives. This consolidation
claims neither reduced demand count nor a planning/device speedup.

### M4 allocation/emission qualification

The finite pipeline regression runs concrete chains, local barriers, guarded
complementary publications, terminal cleanup, empty plans and repeated finite
bodies through ordinary split, saved/reparsed and combined output. It evaluates
original guard/trip bindings and compares payload closure, logical publication
matching and physical consumption-before-republication across the routes.
Terminal-publication fixtures distinguish a fixed protocol's live capacity from
possible alternative protocols. Device execution is not part of this host gate.

The empty-plan fixture exposed a saved-output refusal. The shared exact-empty
qualification now allows deferred requests with an empty finite certificate and
no recipe, commands, preparation, pending state or completion obligation to use
the ordinary empty insertion path. It creates no new archive schema or allocator.
Negative unit checks reject absent/mismatched certificates and nonempty state.
Physical empty-output qualification still excludes deferred physical state.

Both source reviewers accepted M4. The incremental compiler/tool build and
empty-plan negative checks passed. The focused gate passed 6/6 in 81.25 seconds
(`m4-focused.json`/`.out`). The new test checks 144 guard/trip valuations across
three physical routes, six C++ outputs and terminal-capacity cases of 2, 4 and 7
publications against six eligible IDs. C++ output is not device qualification.
The optimizer runs serially; the local test harness limits CPU affinity to one,
which also bounds the LLVM thread pool used by the compiler launcher. The two
build warnings are unchanged covered-switch defaults in SessionProducers.cpp
and SequenceBoundary.cpp. The one changed-code G.FMT.11-CPP match is a multiline
braced condition in PreparedInsertionChecks.cpp, inspected as a false positive.

### M5 qualification

The new finite qualification driver uses production pipe envelopes. Small
flat/composed equivalence checks cover every top-level two-way partition,
including empty children, and both three-child parenthesizations. They compare
active demands, all four event-kind reachability answers, presence and actual
boundary occurrences under every Boolean valuation. Nested branches, RMW and
simultaneous V/S scatter envelopes stay indivisible. This checks composition
against flat analysis; independent conflict correctness remains the responsibility
of the existing explicit graph/covers and command-causality oracles.

Queries are constructed before valuation rollback transactions, so query caches
cannot retain rolled-back expression IDs. Finite crossing and boundary keys
require ordinal zero and no enclosing visits before comparing original phases.
These checks complement the repeated-invocation trace coverage in M3/M4.

Scaling fixtures exercise concrete chains through 4096 payloads, guarded
few-writer collections through 1024 readers, and guarded write-heavy collections
through 128 writers. Both guarded writer-pair passes count inspected source and
pair positions separately from candidate callbacks. Residual phase/effect checks,
concrete incidence/witness visits, numerical queries, boundary entries, recipe
builds/cache hits, and actual root-copy entries are separate counters. The slice
gate constructs one child per original root and requires one copied entry each.
Counters measure these named operations, not every DAG traversal or total
compiler work; supplied effect/cell construction is outside the scan bound.

Concrete export and requested numerical answers must construct no guarded
analysis, and repeated preparation must reuse immutable recipes with separate
mutable inventories. Guarded pair counts are checked against the actual supplied
cell incidences and W, including write-heavy quadratic cases. Circuit-size checks
are per tested construction, not a partition-independent representation claim.
No planning or device speedup is claimed by this qualification milestone.

The new qualification tests passed 2/2 in 1.22 seconds: 57 valid partitions and
parenthesizations, with 13,596 semantic comparisons across four small cases.
The separate macro-envelope, deferred-callee and closed-child end-to-end gate
passed 3/3 in 4.13 seconds. At 4096 concrete payloads the scan inspected 10,240
incidences and 8189 hazards; 4096 numerical queries added zero expression nodes.
With 1024 guarded readers, the two scans inspected 2048 pairs and retained 2053
expression nodes. With 128 potential writers, they inspected 32,512 pairs and
retained 18,169 nodes. These are structural measurements of the tested forms;
they do not promise linear write-heavy or arbitrary guarded construction.

The primary PyPTO-lib campaign passed 49/49 logical compilations and 49/49
six-ID allocations with MayNotAlias. The input hashes and stage outcomes agree
with the historical rebase campaign at `3c166285004fc78e72f2493c397cb3633ff6f88f`
plus its recorded working diff. This is not a paired M1 baseline. Five modules
have changed logical arithmetic/command counts; whole-corpus trace equality
and planning speedup are not claimed. This campaign did not request C++ output
or device execution; M4 separately checked six C++ outputs.

Both reviewers accepted the final M5 source and focused qualification evidence.
The changed-code checker reports twelve multiline-condition brace matches and
two Python subprocess matches. The C++ bodies are braced. The Python harness
resolves its tool, uses an argument vector without a shell, sets a timeout and
checks the return code, as required by the loaded Python rule text; the checker's
blanket child-process prohibition does not match that rule. These findings were
reviewed explicitly, not counted as a clean analyzer run. The incremental build
completed without new warnings; existing covered-switch warnings remain.

The complete synchronization gate passed 265/266 in 724.74 seconds. Its sole
failure is the previously recorded `sync_repeated_region.pto` phased-repeat
structured-trace subprocess timeout at 90 seconds; the initial 17 nested closure
checks pass. The historical full gate passed 262/263 with the same failure.
No timeout or expected outcome was relaxed. This remaining repeated-phase
construction limitation is outside the finite consolidation scope and stays open.

The [machine-readable receipt](frontier-finite-consolidation-results.json)
retains source hashes, input hashes, actual scaling rows, campaign scope and
changed corpus metrics. Local raw logs remain under the evidence directory above.
The completed migration retains concrete numerical analysis through regional
export, shares guarded ownership and direct-publication construction, and removes
the superseded root/parent adapters in the deletion ledger. Distinct expanded-loop
and prefix protocols remain supported through their existing qualified paths.

Both reviewers accepted final M5 closure with the inherited failure retained.
All five finite consolidation milestones are complete within this scope.
