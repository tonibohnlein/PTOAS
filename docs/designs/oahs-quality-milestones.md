# Synchronization quality milestones

Started 2026-09-22 at `b62b89de5`, following the comparison with research draft
0.35. The objective is to preserve pipeline parallelism under fixed payloads,
physical storage, original control and target event resources. Existing
InsertSync is a comparison, not an ordering specification.

## Milestones and acceptance

1. **Active: production publication-prefix certificate.** Establish a restricted
   certificate at actual command-word positions, including outward publications
   and neighboring key uses. Preserve the early release through later edits, or
   invalidate the certificate. Start from the current KDA release/acknowledgment
   witness; use hc_pre/RMSNorm for independent storage-lifetime checks and GEMM
   as a performance-qualified nonregression. A supplied protocol alone is not a
   constructor improvement. Require a linked production regression, independent
   complete payload-order comparisons, unchanged causal/reconstruction checks,
   and the larger prepared-kernel corpus. Record remaining native misses rather
   than promoting an experiment whose ordering gate fails.
2. **Common lifecycle view.** Consolidate physical identity, producing/reading
   occurrences, last physical users, next overwrite, legal gaps, participation
   and possible return support. Implement only the portion required by milestone
   1 first. Records prescribe obligations, not private event pairs, and do not
   become a second completion authority.
3. **Rearming at its actual deadline.** Keep storage acquisition deadlines
   separate from the next applicable publication of a reused event key. Select
   actual required returns before unnecessary private acknowledgments. Retain
   unresolved rearming obligations without granting anticipated receipt credit.
4. **Necessary recurring obligations before physical binding.** Use lifecycle
   and support records to avoid unnecessary channels during construction.
   Replace omission trials incrementally with direct certificates; retain
   explicit recurrence support and cold final validation.
5. **Open lifetimes and scoped ownership.** Generalize qualified successive
   physical uses across skipped children and re-entry. Reuse physical keys only
   with certified ownership lifetimes, matching and actual consumption paths.
   Include zero-use children, independent readers and generation-boundary
   negatives. Lexical nonoverlap alone is insufficient.

## Quality and validation contract

Distinguish required-order closure, actual emitted-command order, and the order
of a checked alternative under identical contracts. An extra relation is an
opportunity, not proof that the primitive vocabulary can avoid it. Compare full
relation sets; fewer relations or commands alone do not establish dominance.
Finite traces are regression evidence, not induction over arbitrary loops.

During construction, certify only a bounded placement/edit with complete causal
interfaces; do not add all-pairs graph search or a general final refinement pass.
An incomplete plan is not the comparison baseline for a no-added-order claim.
All original physical requirements and independent acceptance checks stay intact.

Run focused portable and native regressions first, then the prepared corpus and
the archived compatibility inputs. Pin source, input and driver identities and
record changed plans separately from failures. Use disk-backed artifacts outside
the source tree and obey the aggregate two-worker limit. Device tasks follow a
specific qualified candidate or unresolved timing-attribution question; preserve
authentic coupled execution and compare existing, pinned OAHS and candidate.

## Milestone 1 evidence and next action

The archived KDA path is M completion -> acknowledgment -> MTE1 release -> MTE2
load. The opt-in first-write/final-read plan previously removed 1,784 relations
and added 706; it is not accepted. Refresh this evidence at the starting commit.
RMSNorm's input storage also serves as reduction scratch, so conversion is not
necessarily its last physical read. hc_pre has an entry-prefix candidate on
distinct accumulator and input ranges. See `oahs-final-read-sources.md`,
`oahs-frontier-motion-context.md` and `oahs-sweep-followup.md`.

### First implementation: within one command word

`SelectedPublication.cpp` implements a default-on construction-time rule for
new ordinary completion publications. It uses the KDA witness's release-before-
unrelated-wait mechanism without recognizing a kernel or opcode sequence. It
does not yet improve the native KDA plan. The milestone remains active.

At ordinary binding, the constructor selects one earlier gap in the same word.
It may cross acquisitions on the publishing engine and commands on other
engines, but stops at another publication or fence on the publishing engine,
any ALL barrier, a matching physical-key endpoint, or a protected prefix.
Keeping each outward source publication on its original side of the edit is
essential: a required payload check alone cannot certify its exported order.

The structural ordering argument is local. SET observes the current completion
prefix but does not gate subsequent launches. Moving it before a source WAIT
removes that WAIT's completion prerequisite from SET. No source SET/fence is
crossed, other engine orders are unchanged, and matching event endpoints do not
cross. Every retained boundary path in the new fragment therefore already
exists in the old fragment. This is a restricted instance of the draft's
boundary-path certificate, not a general graph-inclusion algorithm. Its
statement assumes an unchanged surrounding graph; later greedy choices still
require independent complete-plan comparisons.

Coverage and protocol legality are separate. For every reachable original
occurrence of the shared word, replay to the exact proposed gap must establish
the motivating storage completion, an empty physical event, and actual
consumption knowledge at its publisher. A private ledger copy then undergoes
full fixed-plan analysis: protocol/resource checking must succeed, and no new
missing storage or retirement obligation may appear. A refused certificate
leaves the live ledger unchanged. Accepted motion retains endpoint IDs and
increments the ledger version; selected replay still checks finalized queries.
There is one proposed gap, no alternative-plan search or final motion sweep.

The ledger records the source-engine endpoint IDs preceding the protected
publication. Later insertion and return restoration stay behind that boundary;
each selected replay, including a retry after restoring returns, checks that
no new source endpoint entered the recorded prefix. Deletion may shrink it.
This protects the prefix **inside that word only**. It does not yet certify
edits to incoming state from earlier words, recurring-role coalescing, movement
across payloads, or a general all-publications construction invariant.

The initial implementation deliberately spends two fixed-plan analyses per
considered motion. `prefix_checks` and `prefix_analysis_sites` expose that cost;
`prefix_publications` counts accepted motions. Replace those solves with a
shared local boundary summary only after the quality rule is established.
`--no-publication-prefixes` in the diagnostic driver isolates the new rule.

`selected_publication_test.cpp` exercises actual construction, including a reused
physical key, and compares complete issue/completion relation sets using the
independent graph oracle. It removes the unwanted compute-completion -> refill-
issue relation without adding any relation. Negatives retain required compute
completion, refuse an intervening outward publication, source fence, ALL fence
and same-key crossing, and check insertion/restoration after certification.
The earlier source-gap policy's comparison explicitly disables this independent
new rule, so it continues to measure that policy alone.

### Current baseline correction and next implementation

At frozen `b62b89de5`, default KDA, dspark RMSNorm and hc_pre construct and
reconstruct. Both baseline and candidate now refuse the opt-in KDA
first-write/final-read experiment at cut 754: `split recurring receipt has no
certified return key`. The older 1,784-removed/706-added measurement above is
historical, not the current baseline. Do not relax key-rearming checks to
recover that experiment.

Next: extend the certificate from a word-local gap to a qualified source
boundary using the actual KDA publication/acknowledgment path. Track incoming
causal dependencies and outward publications across crossed words, including
invalidation after later edits. Keep source-time consumption evidence mandatory;
the cut-754 experimental failure is a separate prerequisite if that policy is
used. Require an actual default native plan improvement with complete relation
inclusion before claiming that the KDA witness is resolved. Recheck hc_pre,
physical RMSNorm lifetimes, GEMM, and the entire paired corpus at that boundary.

Artifacts, driver/input hashes, and final validation totals are recorded in
`HANDOFF.md` under the active quality work. Device timing remains pending an
improved native candidate; identical native plans establish no speedup claim.
