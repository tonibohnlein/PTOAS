# Selection ownership and milestone 2 closure audit

This is a source audit of the construction-first milestone-loop changes. It is
not a validation receipt or a completed-milestone declaration. Build and test
receipts belong to `.local/milestone-loop/`; the construction checkpoint and
remaining protocol investigations remain separate.

## Implemented ownership

`FrontierAnalysis::analyze` owns production selection. It validates the session
and original region, reuses a complete retained answer when permitted, tries
Explicit first, then ranks remaining eligible exact candidates. Failed attempts
can retain mathematics or exports, so the remaining candidates are ranked again
after each attempt. Unknown work never rejects an eligible candidate.

`exactForms` and `arithmeticMethods` inspect retained qualification only. Cold
screening leaves original and expanded alternatives available with unknown
costs; it does not run numerical preflight, normalization or arithmetic
recognition to price an unused candidate. Form costs remain diagnostic
description estimates. Actual backend selection uses `rankExactCandidates`.

`previewRetainedBackendWork` resolves named, canonical-loop, regional-arithmetic,
expanded and identity owners. Its four capability stages are mathematics,
queries, selectors and synchronization. Root arithmetic additionally tracks the
shared generator dependency. Ready or unrequested stages have zero unbuilt
work. Refused stages retain their refusal status and are not reconstructed.
Hidden query prerequisites for selectors and identity adapters are included.
Known estimates order candidates by work alone; stable ties and unknown order
are retained. Dependency ordering keeps NumericalPeriodic before
NumericalSequence and Sequence before SequenceEndpoints.

| Retained data | Key and lifetime | Reuse boundary |
| --- | --- | --- |
| Complete selected answer | Original region, full five-bit request key, audit/certifying flag; owned by the session | Ready mathematical answers only, excluding completion and compact fallback. Exact key first, then compatible capability supersets with Explicit preference. Each returned answer gets the current request identity. |
| Exhausted request recipe | Full request key, audit flag, original caller stack, ordinary/identity-continuation namespace and retained-evidence generation | Reuses the immutable final outcome and diagnostics after complete exhaustion. No mutable plan is retained. New evidence invalidates the recipe. |
| Backend ranking recipe | Region, full request key, audit flag, complete candidate mask; retained stage/owner evidence | Repriced when retained evidence changes. An old unknown estimate cannot hide newly retained capabilities. |
| Form/arithmetic order | Region and request key, plus retained qualification identity | Append-only evidence snapshots keep old diagnostic rows valid. |
| Loop mathematics and exports | Canonical original loop and backend | Root/child adapters reuse the same owner where their coordinate contract agrees. |
| Arithmetic primitive counts | Immutable `ArithmeticProgram` address inside one session | Counts are request-independent. The owning recognized form outlives this cache; invalidate/reinitialize discards the session. No qualification is performed to populate it. |
| Arithmetic generator stages | Root `ArithmeticProgram` identity, checked by `belongsTo`, with DBM and integer owners in the session | Generator work is a separate additive estimate. Retained generators cost zero; missing reduction/conversion work remains estimated independently. Failed generator dependencies retain their refusal. |
| Specialized mathematics | Original context, observed constants/bindings, representation and declared limits | Separate specialization caches; unavailable observed constants also participate in matching. |
| Detached logical plans | Immutable retained mathematics/recipe | `prepareLogical` creates a fresh mutable plan; cached detached operations are not shared between callers. |

The five request bits are queries, selectors, synchronization, fallback mode
and stateful evaluation. Input, alias policy, original IR and fixed arithmetic
profiles are session context. Original IR mutation requires `invalidate()`;
changing profiles after arithmetic requests is rejected. Completion and compact
fallback answers deliberately do not populate the positive selection cache:
later exact mathematics must remain discoverable. Exhausted failed and fallback
walks instead use a separate generation-qualified recipe. Repeating the same
request and caller context performs no new candidate walk while retained evidence
is unchanged. Returning a cached recipe refreshes its diagnostic request identity.

Publication of a new original form, normalization/preflight result, arithmetic
generator, specialized mathematics, query/selector export, immutable endpoint
recipe or successful endpoint capability advances the evidence generation.
Intermediate retained stages count even when the enclosing backend fails.
Repeated use of an existing stage does not advance it. Direct audit and preparation
entry points participate in publication. Generation overflow clears exhausted
recipes before restarting the generation; input invalidation discards the session.
Ordinary recursive requests must still descend to an original child. Same-node
identity continuation uses a separate cache namespace and its existing restricted
candidate suffix. Backend failure memoization is unchanged.

The exhausted-walk changes passed source review and the 46-test focused gate
recorded in the [construction checkpoint](frontier-reviewed-construction-checkpoint.md).
The broader production refresh remains a separate gate. `checkProducerFailureStages` checks a real exhausted candidate walk,
unchanged diagnostics and fresh request identity, a direct recognition publication
that permits a retry, and mode/audit/capability/caller/continuation isolation.
Structural `request_walks` and `exhausted_walk_hits` counters report this work
independently of mathematical attempts.

## Sequence endpoint upgrade

`Shared/SequenceAnalysis.cpp::upgradeSequenceEndpoints` is an explicit stronger
view of retained sequence mathematics. It applies to an original sequence with
supported children and to an original conditional whose arm coordinates remain
unchanged. An already exported relational result also supports unchanged child
coordinates. Changed conditional or relational coordinates retain the ordinary
exact producer; this is not universal representation reuse.

When a child's original coordinate contract is unchanged, only owned preparation
callbacks are replaced; the original semantic callbacks and crossing roots are
retained. With all children unchanged, the new state retains the old state as
`upgradeSource`, shares crossing construction and rebuilds no expression nodes.
Preparation counters are reset rather than replayed as new work.

For a conditional, the original guarded semantic view and crossing roots remain
owned by the old state. Each upgraded arm's preparation callbacks are wrapped
with its exact original arm guard. Default preparation calls that owned wrapper;
resolver-based preparation obtains a fresh raw arm plan and applies its existing
mask once. A changed arm coordinate frame requests exact recomposition. The
conditional regression compares both preparation routes with a cold producer,
checks both branch values, and retains source and child-mathematics identities.

For a relational result, a typed preparation service owns the previously exported
arithmetic state and crossing state. Rebinding retains the exact relation owner,
selectors and crossing demands, and supplies a new immutable vector of child
preparation owners. The service does not own its containing relation, so this
ownership graph has no back-reference cycle. The original relation's child
mathematics remain unchanged. Missing initial exports have no service and keep
their original obligation. Upgrade receipts report reused relational results
and actual new relational composition calls separately.

The symbolic-sibling fixture upgrades the original sequence producer, asserts
zero relation composition and arena growth during the upgrade, then checks its
queries and minimum covers against an independently enumerated event graph.
Emitted logical commands are checked for full-identity matching in zero, one
and multiple-visit invocations. These checks passed the committed focused checkpoint gate; broader production
validation remains separate.

When a child representation changes, unchanged child mathematics is still
retained. The parent partition and transitive closure are reconstructed. This is
a real dependency: a changed intermediate child can change transitive reduction
between unchanged endpoints. The current state has no per-crossing dependency
certificate that would justify retaining arbitrary old reduced crossings.
Selective reuse within that closure would require such a certificate and an
independent equivalence oracle; it must not be inferred from unchanged endpoint
IDs alone.

Tests in `NormalizedControlChecks.cpp::checkSequenceEndpointRetry` cover changed
coordinates, unchanged-coordinate zero construction, retained mathematics,
fresh detached plans and unchanged source. The fixture seeds the expanded
child's actual query and selector capabilities; it must not demand an obsolete
cold ranking. The independent `check_sequence_endpoint_retry.py` checks command
closure for zero, partial and nonunit visits.

## Supported caller inventory and deletion criteria

This inventory distinguishes production orchestration from mathematical kernels.
The public visibility of a method alone does not make it superseded.

| API group | Supported in-tree callers | Ownership and deletion criterion |
| --- | --- | --- |
| `analyze`, `minimumDemands`, `prepareLogical`, `attachAllocation` | Logical insertion in `FrontierAnalysis.cpp`; recursive resolver in `SessionProducers.cpp`; test tools | Production request facade. Retain it as the selection/materialization boundary. |
| `analyzeNumericalRegion`, `analyzeFiniteVisit`, `analyzeVaryingBoundary`, `analyzeFiniteExpansion`, `analyzeArithmeticRegional` | Backend-specific session regressions in `pto-sync-input-test.cpp`, `NormalizedControlChecks.cpp` and related checks | Already request-based wrappers over `requestBackend`; retain explicit representation selection for diagnostics/tests. They do not perform the general candidate walk. |
| `analyzeExplicitFunction`, `analyzeArithmeticFunction`, `analyzeArithmeticPeriodicFunction` | Named arithmetic ownership/routing tests; existing public convenience access | Request wrappers selecting only the named backend. Production `SessionProducers.cpp` calls private `produce*Function` helpers, so wrappers cannot recurse into themselves. Generator-stage resume and whole-invocation evidence remain retained. |
| `auditSequenceFunction` | Numerical sharing/profile-freeze checks in `pto-sync-input-test.cpp` and concrete late-guard checks in `CompactBoundingPipelineChecks.cpp` | Renamed concrete composition oracle. Production uses `analyze`/`minimumDemands`, which may return an identity wrapper instead of constructing a full sequence. The private producer still shares retained concrete mathematics with this audit entry point. |
| `recognizeArithmetic`, `recognizeRegionalArithmetic` | Whole/regional producer preparation; arithmetic recognition diagnostic modes and regression tools | Qualification services. Move audit-only enumeration behind explicit audit naming if changed; do not delete the per-candidate qualification needed by production. |
| `finiteArithmeticRegion`, specialized guarded/numerical/arithmetic methods, guarded instantiation | Resolver callbacks in `SessionProducers.cpp`; specialization isolation tests | Production scoped services. Their richer context/binding keys are required; replacing them with a plain region-only request would lose semantics. |
| `recognizeAllForms`, `certifyRegions`, `auditNumericCandidates`, `auditNumericCandidateExports`, `auditFiniteVisitCandidates`, `auditedFiniteVisitDemands` | Certification/diagnostic pass and test tools | Explicit exhaustive audit workloads. Retain independent audit coverage and keep its work out of production selection measurements. |
| Demand getters, `explicitResult`, `result`, cost/profile accessors | Producers and test/diagnostic inspection | Read-only retained state. `result()` refreshes a diagnostic snapshot; callers must reacquire after mutating requests. Delete getters only after moving every supported inspection to an equivalent owned result. |

Removed orchestration includes the unused rotating preparation wrapper, bounded
whole-function preparation/state and arithmetic-periodic scalar preparation
wrapper/export state. Their in-tree callers had already moved to the request
and recipe path. Generic mathematical reducers and independent test oracles are
retained. This inventory establishes in-tree ownership; it is not an assertion
that downstream external users of public headers do not exist.

## Remaining obligations before full milestone acceptance

1. **Validate the new arithmetic stage split.** The source now separates
   generator construction from backend reduction/conversion estimates and
   records generator ownership in ranking evidence. It never subtracts guessed
   completed work from a whole estimate. New regressions use real DBM and integer
   generator stages to check zero retained generator work, unchanged remaining
   reducer cost, owner isolation and repeated previews. Run these together with
   periodic-to-general and general-to-periodic continuation checks.
2. **Missing export work remains an estimate.** Current structural export costs
   are heuristics. Keep them labelled as such; unavailable qualification stays
   unknown. A complete construction-work claim needs stage counters and paired
   serial production evidence, including materialization. Primitive-count cache
   tests must show one structural scan across repeated mixed requests and fresh
   scans after session reset.
3. **Validate the new facade separation.** Named explicit/arithmetic conveniences
   now use request wrappers and private producers. Concrete sequence consumers
   migrated to the explicitly named audit entry point. Run owner/retry,
   failed-export retention and production dispatch tests; no new unused sequence
   convenience wrapper was introduced. The public sequence-oracle rename is an
   intentional diagnostic API migration.
4. **Validate upgrade dependency boundaries.** Run changed-coordinate and
   unchanged-coordinate upgrade checks on the final source, plus the independent
   command-closure oracle. Full parent recomposition after changed coordinates
   is documented; any smaller recomposition claim needs new dependency evidence.
5. **Close the final request and profile gates.** Run explicit-success/no-later-
   form, repeat/no-walk, unseen-superset, weak/strong/weak, audit isolation,
   invalidation/reinitialization, failed-export retention, stable unknown/tie
   ordering and cold expanded fallback checks. Profile selection reasons,
   failure stage, reused/missing/refused outputs and retained unrequested work;
   selection snapshots are not additive construction counts.

Passing these selection gates does not certify GroupNorm/prefill/Delta/decode
allocation, successful kernel compilation, or device execution. Those outcomes
retain their own protocol-specific evidence and open items.
