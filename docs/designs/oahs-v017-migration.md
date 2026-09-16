# OAHS migration to the v0.17 forward constructor

Specification: synchronization draft v0.17, source revision
`7cce28193b7220ae91d16b130dcff339d15220fa`, sections 7–10 and policies F1–F8.
The target is one selected-plan forward constructor through the existing
`algorithm=handoff` entry. `algorithm=existing` remains the comparison path.

## Preparation layer implemented in the first increment

The shared `MemoryDependentAnalyzer::storageCoordinates()` API qualifies a
single nonempty, nonoverflowing translated bounding interval. Known local
physical addresses share coordinates across allocation roots. GM offsets retain
their original root-relative coordinates under the production alias contract.
Unknown geometry and alternative addresses receive no coordinate qualification.
This query does not change legacy `MemAlias` decisions.

Native handoff import partitions qualified ranges at their endpoints. Each
canonical cell retains its domain, coordinate space, interval, and original
translated root identities (`NativeAnalysis::storageRoots`). A bounding interval
does not prove that an instruction definitely overwrites all its bytes; native
import still sets no `Access::definiteWrite` flags.

Unqualified footprints retain independent recurrence records. Pairwise
conservative witnesses retain every additional overlap reported by the shared
alias analyzer. An unknown access overlapping two disjoint intervals does not
merge those intervals. These witnesses are explicitly distinguished from
canonical atoms and cannot support strong updates. Distinct-footprint pair
queries remain potentially quadratic.

`StorageFrontierAnalysis` now exposes:

- `lifecycleAt(site, cell)`: incoming writers/readers, subsequent writers/readers,
  enclosing loops, participant engines, possible absence of a prior full write,
  and a possible access-free path from after the site to invocation exit.
- `describeRequirement(relationship)`: a reason set and occurrence qualification,
  separately from RAW/WAR/WAW. All physical relationships retain the overlap
  reason. Canonical slot overwrites can also receive the reuse reason.
- Readiness classification only when a unique full writer dominates an acyclic
  consumer occurrence with no intervening possible writer. Partial writes,
  branch bypasses, and repeated visits with unqualified correspondence do not
  receive that certificate. A mutable tile SSA handle supplies no certificate.
- Original observation identities and crossed-loop provenance. An existential
  loop witness does not establish an iteration distance or event matching.

The lifecycle queries use the whole original control graph. A read-only child
sees writers before it and overwrites after it, including enclosing backedges.
No region exit clears outstanding work or events. Marginal provenance is not a
complete completion-obligation list: strong overwrites may kill origins, while
all outstanding causal obligations remain independently tracked.

The current constructor's selection policy and ordinary transfer domain are
unchanged in this increment. Canonical storage is used by real native import;
the new lifecycle/provenance queries are preparation for the replacement driver.
No new planner mode, native resource credit, or operation whitelist is added.

## Remaining implementation sequence

1. Extend occurrence qualification with the draft's checked local slot/visit and
   first/last-use certificates as their constructor consumers are introduced.
   Keep uncertain relationships as physical requirements.
2. Implement the storage-indexed must-causal frontier: A/T/S/D ports, guaranteed
   reachability, optional access-class histories, may occupancy, and binding
   alternatives. Validate transfers and joins against matched exact reference
   semantics. The old operation-indexed domain is not this representation.
3. Implement F1–F7: all effects of a consumer together; known readiness/reuse
   before remaining physical overlap; provider-containment priority; saved early
   prefixes; source-position allocation; and one selected endpoint ledger with
   checkpoint invalidation. No competing whole-plan replay or route enumeration.
4. Implement F8 in the same constructor, retaining surrounding continuation,
   entry/backedge/exit obligations, zero-trip for and mandatory while-before.
5. Switch the handoff driver after the verification gates pass, then remove the
   obsolete candidate scoring, widening/recovery policies, and implicit interior
   ALL fallback. Keep explicit invocation retirement and native reconstruction.

The draft's joined-state simulation, loop induction, and selected-checkpoint
proof obligations remain explicit. Finite reference comparisons supplement
those obligations; they do not complete their proofs. Native phase qualification
remains disabled. Existing reference artifacts are retained.

## Regression gates

`test/oahs/lifecycle_test.cpp` checks canonical conflicts against the prior
conservative import on 300 seeded footprint populations; disjoint ranges linked
by an unknown access; different coordinate roots; overflow; retained provenance;
readiness rejection; and continuation across branches, loops, and while exits.
The native driver checks shared coordinate qualification, native root mappings,
canonical import, conservative readiness labels, and unchanged original IR.

Run the full standalone suite and the native driver/lit gate after rebuilding
all dependents of the changed public structures. Replay the 21 local prefill
files separately for semantic analysis and existing/default output preservation.
That sample is not the full production/PyPTO/pypto-lib denominator, and analysis
success does not establish handoff synthesis coverage or device correctness.
