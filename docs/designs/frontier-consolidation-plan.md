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
| 2. Directional finite allocation | Implement Section 7's ordered SET/WAIT threshold sweep and interval coloring under explicit premises, including insufficient-capacity cases. Make logical-only allocation evidence optional; request stronger lane evidence only when useful. | Independent exact allocation oracle, linear reuse-query count, just-over-capacity regression; full compilation preserves allocation successes. | Pending |
| 3. Capability ownership | Separate mathematical provider, recognized form, conversion, composition and requested export. Upgrade endpoint capabilities on retained evidence. Add distinct logical-plan, closed-matching and completion-summary capabilities. | Owner/context/occurrence identities survive upgrades; failed upgrades retain mathematics; unchanged work is shared. | Pending |
| 4. Shared compact input | Delay fine cell memberships; preserve compact selections and borrowed memberships through regional adapters. Index/shared-own aliases, pointer/view origins, scalar/descriptor/control facts; compactly represent universal descriptor invalidation. | Select-chain, wide-access, view-depth and descriptor/call/depth scaling counters; same physical overlap and alias semantics. Coarse Section 6 input avoids fine atomization. | Pending |
| 5. Numerical composition | Project to actual parent ports with hierarchical interior queries; reduce crossings once; construct guarded boundaries with prefix/suffix passes; share immutable PhaseIndex facts. | Independent reachability/boundary oracle including macro envelopes and empty/guarded regions; bounded-port and sibling-region scaling families. | Pending |
| 6. General Section 6 fallback | Closed executable child interface and mode-tagged completion transfer on immediate-child CFG; two delta fixed points; qualified completion protocols, explicit loss and independent physical reuse. | Joins, backedges, while final-false visit, opaque/scalar prerequisites, child cleanup and repeated invocation tests; strict Section 8 dispatch policy. | Pending |
| 7. Exact repeated-guard integration | Derive certified incoming executed-source ordinals from retained demands, including relational sequence crossings. Use shared predicate histories, one invocation-owned recipe and one staged clone. | Public Gated Delta MayNotAlias logical emission succeeds with 29/29 exact regions; independent matching/closure and payload preservation; correct skipped-source, nested-reset and cleanup behavior. MayAlias gaps remain separately reported. | Pending |
| 8. Proof reuse and validation | Share lazy inactive-proof and premise contexts, replay mappings and safe Boolean conversion work. Close known guard-equivalence/oracle gaps and run pinned paired campaigns. | No unexplained acceptance loss or changed output; report eleven kernels and 783-input corpus by stage, wall time, memory and inner work. | Pending |

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
