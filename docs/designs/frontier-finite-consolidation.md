# Finite analysis consolidation

Baseline: `1953d248c`. Implementation status: M1–M2 accepted; M3–M5 remain open.

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
| Root explicitAnalysis + MathematicalResult::explicitDemands | finite owner, concrete strategy | M1: session, guarantees, allocation, prefix compatibility, test adapters |
| MathematicalResult::finiteGuardedDemands | finite owner, guarded strategy | M1: original finite callers; ExpandedFinite remains a distinct coordinate adapter |
| Explicit/guarded root preparation dispatch | finite recipe request | M1 routes, M3 shared recipe builder |
| Sequence explicit-child export reconstruction | common finite regional export | M2, including prefix compatibility |
| Explicit/guarded endpoint construction loops | common direct recipe builder | M3 after command/cleanup oracles |
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
