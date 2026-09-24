# FrontierSynch Phase A: original-program analysis contract

This is the construction-facing interpretation of the working synchronization
draft v0.41 at `dc5c309`, Sections 3.1–3.5. `ProgramAnalysis` owns one imported original structure and
its lifetime, occurrence, and guarded-reader query services. The unchanged
shared `SyncInput` and source function must outlive it because physical uses
retain translated effect and IR identities. Its indexed
requirements and source subscriptions exist before construction traverses any
source. A subscription names an original position; it is not a saved selected
causal state or an emitted publication.

For each RAW, WAR, or WAW relationship, `decodeAt(deadline, index)` cheaply
retains the original source/deadline positions, least shared original owner,
effect-specific physical selectors, and guarded structural reader expressions.
`interpretAt` lazily joins that record with original occurrence status, shared
source and target histories, alternative may-origins, a write-delimited support
request, and first/last boundary answers. Support expansion remains a separate
`qualifySupport` query for a request under consideration. This is one construction-facing original
interpretation for a currently relevant request. An alternative origin has no
path guard yet, and a physical-bank candidate has no inferred predecessor
mapping. Those statuses remain unresolved; the underlying requirement stays
indexed. Several requirements can point to one structural reader frontier;
partitioning a physical footprint does not itself request an event.
Requirements are indexed by source/target engine direction, exposing possible
required returns without selecting them. Source subscriptions are installed
before traversal; histories are shared by use and support is expanded only
when a client requests qualification. `mayHaveIncomingGeneration` keeps the
no-prior-full-writer path separate from the alternative-origin may-set.

The typed `all`, `mayAfter`, `firstConflict`, and `lastRelevantUse` queries have
different strengths. `all` is a conservative physical may-set. `mayAfter`
traverses the designated original continuation and returns May, NoHit, or
Unknown. It withholds a negative proof when an unqualified repeated endpoint or
owner exit makes the dynamic interval ambiguous, distinguishes identical gaps,
and retains possible stop bypass even when another path has a matching access.
First/last answers carry guarded reader expressions and may accesses. A
structural expression is not an exact frontier for a requirement until its
episode and occurrence interval are qualified; WAW and unsupported selectors
remain Unknown. An exact original frontier still requires its predicate at the
proposed SET/WAIT gap. `guardAvailableAt` provides a conservative original-
operation check; selected command-word placement needs its exact gap check.
Repeated negative continuation answers stay Unknown until a checked
occurrence certificate is represented; callers cannot assert matching with a
Boolean flag. The fixed-visit query is cached by source, target and cell.

The current import does not prove complete overwrites from bounding geometry.
Consequently, older possible writers survive reloads, and a content-generation
certificate is unavailable on native input unless a separate effect-coverage
proof is supplied. The shared InsertSync translator is still the effect
completeness authority. D1 alternative origins are indexed but lack general
guarded source correspondence. D2 stores independent bank relations and
attaches a specific relation only to the translated effect that supplied it.
For one mandatory same-role use of an exact disjoint-bank permutation, it
establishes the predecessor distance within one counted-owner invocation only
when **every access incidence for both hazard roles on the queried cell** uses
that relation. A second fixed or differently selected access to the same cell
keeps the aggregated requirement's correspondence Unknown;
the recorded first ordinal with a local predecessor is the bank period, and
earlier ordinals require an explicit incoming case. This record is not a
nearest-physical-use proof, a completed production episode, or a release.
The first-period entry domain and cross-invocation predecessor remain unknown.
D4 retains original owner/child identities but does not transport arbitrary occurrence
frontiers through re-entry. These are **unfinished precision limits**, not
hardware restrictions or permissions to drop requirements.
The current support candidate uses the source/target writers for WAW, a unique
following writer for RAW, or a unique preceding writer for WAR. This is a
sufficient write-delimited interval candidate, not a whole-cell uniformity
rule. When uniqueness is unproved, the support request stays Unknown and its
physical requirement remains. `qualifiedBoundary` can promote a structural
reader frontier only after a complete generation interval, participating
reader, and exact fixed-visit occurrence have all been checked. This does not
yet establish D2/D4 recurrence frontiers.

`ProgramAnalysis::complete()` means preparation completed, not that every decoded
interpretation is exact. The constructor must treat Unknown as an unresolved
obligation.

The follow-up retains all translated source and target effect incidences through
shared lists in a coalesced demand and records its two engines and whole-operation completion
scope. Guarded reader frontiers can be expanded to original before/after
payload cuts, but each candidate separately records guard availability and
whether that cut is executable in the unchanged IR. Only the outer boundaries
of a multi-phase source operation are presently executable. When its earliest
sufficient phase boundary is internal, the source index subscribes the first
executable boundary after that whole operation and retains the earlier
analytical position so construction can account for the broadened prefix. The selected
command-word position and source snapshot remain Phase B facts.

`firstMayUse` and `lastMayUse` now reject starts/stops outside the requested
owner and return Unknown when traversal escapes that owner, retaining any may
accesses found before the escape. An uninterrupted producer-to-reuse physical
interval is recorded separately from full-generation establishment. No native
translated write currently has full-byte coverage evidence, so such an interval
does not silently kill older writers or qualify a generation-specific release.
The FrontierSynch importer also checks that each declared explicit memory effect of
a translated instruction is present among its shared phase effects. Missing
interfaces, unmapped resources, and modeled macro signatures remain unverified.
Passing this check is necessary, not sufficient, for the draft's full target
contract: implicit effects, instruction legality, synchronization primitives,
visibility, and invocation assumptions still need qualified profile evidence.
The original-value query also follows pure SSA dependencies of branch/loop
conditions and translated payload addresses to producer phases. It indexes a typed prerequisite at
the control site and subscribes the producer's executable source boundary.
Block arguments and unmodeled effectful intermediates retain an unresolved
incoming case. No indexed typed prerequisite is treated as already available.

Each subscribed source now identifies its exact deadline requirement by index.
The requirement builder coalesces duplicate incidences of one physical cell
and hazard kind at a target, while retaining distinct cells, kinds and possible
source origins. This is representation deduplication, not a claim that a selected
transfer has discharged any demand.

## LLVM/MLIR replacement audit

The repository pins LLVM/MLIR 19. Its `DataFlowSolver` and sparse/dense
analyses can replace plumbing for scalar/descriptor transfer and fixed-point
propagation **after** equivalent joins, loop-carried values, unknown handling,
and query work are demonstrated. `ValueBoundsConstraintSet` can strengthen
scalar bounds and address-slice proofs where PTO operations implement the
necessary value-bound interfaces. `DominanceInfo` already supports predicate
availability; `RegionBranchOpInterface` can help import structured successors
without another ad hoc traversal. `AliasAnalysis` can serve as a conservative
filter, but its value-level answer does not replace translated physical byte
intervals, may-alias witnesses, or complete-overwrite evidence. MLIR's effect
resource hierarchy also does not encode precise address and size regions.

Therefore the stable replacement boundary is **inside** Phase A's scalar,
control, and provenance implementations. Keep its physical-use identities,
guarded frontier result types, source/deadline index, and explicit Unknown
semantics. Replace one engine at a time and compare the complete Phase A query
answers on equivalent spelling, unrelated state, optional children, reloads,
and independent bank selectors. No MLIR utility should turn an uncertain may
fact into exact participation or selected completion.

## Acceptance still required

- Qualify definite-write coverage against exact target effects, including
  partial writes and read-modify-write cases.
- Generalize guarded first/last selectors and D1/D2/D4 correspondence without
  a product of independent loop states; retain exact endpoint predicate
  availability and child continuation.
- Add finite return/consequence and compatibility indexes only when the first
  constructor has a concrete consumer; keep their source positions subscribed
  before traversal.
- Validate a linked `frontier-synch` mode and compare the query answers against an
  independent fixture; focused source probes alone do not establish full
  first-pass generality or construction service.
