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

The current constructor's selection policy and ordinary transfer domain remain
unchanged. Canonical storage is used by real native import; the new
lifecycle/provenance queries are preparation for the replacement driver.
No new planner mode, native resource credit, or operation whitelist is added.

## Must-causal frontier implemented in the second increment

`CausalFrontier.h` supplies the new ordinary state and its primitive operations
in the production C++ library. It owns an immutable copy of the imported
program; snapshots cannot be mixed between different programs. State contains:

- One shared closed causal relation over next-issue gates `A`, earlier-finish
  aggregates `T`, current publications `S`, and latest consumptions `D`.
- An optional successor bitset for each `(cell, source engine, read/write)`
  class. Absence and present history remain distinct. A payload queries all
  conflicting classes before replacing its touched signatures with `{T}`.
- May-empty/full key balance and alternatives for current static publication
  endpoints. These bindings do not invent a dynamic visit correspondence.

SET captures its actual source prefix without fencing later source issue. WAIT
gates its destination and replaces the consumption anchor. SET requires a real
path from the current consumption to its finish; the checker never inserts the
desired rearm edge. A named fence advances its engine's gate without consuming
keys. Failed primitives return the unchanged input snapshot and no credit.

Joins intersect already-closed relations, intersect histories only over paths
where each class is present, and union possible balances and publication
bindings. Non-must-live publication ports are removed from both relations and
histories. Unreachable input is separate from fresh invocation entry.

`checkCausalFrontier()` reuses original-control construction and command-word
validation. A finite worklist carries state through choices, counted loops,
mandatory while-before regions, and qualified observed control. It injects fresh
entry only at invocation entry. This is a strict fixed-plan check: it returns
the first unresolved payload/protocol prerequisite, and exports cut invariants
only after successful convergence. It does not replace M1's all-residual
diagnostic analysis or run construction/repair during verification.

The implemented contract is the ordinary issue-ordered asynchronous core with
consuming directed events and named prefix fences. Synchronous lanes, executed
ALL, invocation retirement, exclusive resources, selective phases, visibility,
and implicit/authored effect transfers require matching adapters. Unsupported
contracts are explicitly refused by this new service. Existing native import,
analysis, construction, and their supported contracts continue through the
current backend; the pass does not yet select this frontier. The new library
service is the foundation for F1–F7, not a completed replacement pass.

The differential bridge uses the **existing** pinned `v08/causal_interface.py`:
it is byte-identical to the v0.17 draft's `checks/compact/causal_interface.py`
(SHA-256 `543a7c796acf602d068d3b374c287b224104ddbc5d1d71a785224cc8db5162d1`).
No additional vendor sources or certificates are added. The bridge checks
primitive and join facts against every represented exact state, then checks
accepted structured plans against collecting fixed points without trip-count
unrolling. This is finite evidence under the matched contract, not the general
joined-domain or structured-qualification proof.

## Remaining implementation sequence

1. Extend occurrence qualification with the draft's checked local slot/visit and
   first/last-use certificates as their constructor consumers are introduced.
   Keep uncertain relationships as physical requirements.
2. Connect the new frontier to lifecycle reason/occurrence records and add
   qualified adapters for the native contracts before switching production
   acceptance. Preserve the distinction between static binding alternatives
   and qualified recurring correspondence.
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

`causal_frontier_test.cpp` tests prefix freshness, SET versus source fencing,
shared movement/relay credit, independent readers, consumption versus release,
stale acknowledgments, closed joins, optional publication ports, reservations,
original observation uniformity, and structured fixed points. The test-only
`causal_frontier_driver.cpp` links the actual production implementation.
`reference/causal_frontier_bridge.py` checks serialized states and fixed plans
against the unchanged exact reference, including endpoint-deletion mutations.

Run the full standalone suite and the native driver/lit gate after rebuilding
all dependents of the changed public structures. Replay the 21 local prefill
files separately for semantic analysis and existing/default output preservation.
That sample is not the full production/PyPTO/pypto-lib denominator, and analysis
success does not establish handoff synthesis coverage or device correctness.
