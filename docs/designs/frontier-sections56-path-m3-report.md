# M3 — Supplied-phase control and bounded prerequisite ownership

Baseline: `6e837006c212432f2801c7f88ba710b72c71dc4d` (accepted M2).
This is the working receipt for M3 of the
[path-completion plan](frontier-sections56-path-completion-plan.md).
Implementation and qualification are complete. Independent architecture and correctness reviews accept the stated scope. The
[machine-readable receipt](frontier-sections56-path-m3-results.json) records the
final gate, paired runs, tool/input hashes and limitations.

## Production changes

The existing phase recognizer now handles phase-periodic comparisons, Boolean
combinations and selections. Its scalar adapter preserves signed/unsigned
64-bit comparison semantics; unsupported narrow comparisons retain a refusal.
Absolute outer boundaries continue to use the existing exact slice construction.
Periodic atoms within mixed formulas are bound at their certified phase.

Binding candidates are discovered once from actual branch-control DAGs. Unused
Boolean calculations do not become new recognition premises. Each inherited
binding must also be invariant in every proper enclosing child loop. In
particular, `outer_phase && inner_iv < 2` retains its inner boundary split;
outer periodicity alone does not establish child invariance. Known selections
request only their selected arm. The rotating-child shortcut is used only when
there are no phase/control bindings requiring the contextual adapter.

The bounded recognizer retains its original same-visit prerequisite map instead
of reconstructing it during window construction. The immutable certificate
identifies the original input, loop, phase inventory and prerequisite scope.
Every `PhaseIndex` rebuild invalidates that scope identity. A foreign or rebuilt
scope is rejected before expression construction, including cached admission.

A lazily constructed reverse prerequisite index supports incident outgoing-edge
queries. It scans the supplied forward inventory once per immutable index;
subsequent requests inspect only the selected producer's edges. Recognition
retains incoming and outgoing inventories separately from the storage refresh
bound. These inventories are relative to the supplied index scope and do not
certify absence of obligations outside that scope.

## Existing constructions retained and qualified

| Contract | Production implementation | Qualification |
|---|---|---|
| All-access phase/storage classification | `PhasedRecognition`, `RepeatedStorageGeometry` | Repeated/phase/read-only/overlap oracles; original cuts and exact access membership |
| Numerical boundary ports | `ChainInterface`, repeated/sequence regional providers | Existing all-event and boundary-selector checks; no new relational adapter |
| Collective refresh | `GuardedRecognition`, `PhysicalRefresh` | Existing union-of-guaranteed-writers and explicit-window oracles |
| Difference-bound mathematics | `ArithmeticRegionalConstruction` and requested export cache | M2 production counters distinguish DBM and integer reductions; repeated requests retain conversions |
| Integer-exact octagonal primitives | `IntegerRelations` | Integer elimination oracle; general composition may leave the octagonal subclass |
| Difference-only arithmetic-to-periodic conversion | Existing `ArithmeticPeriodicConversion` | Non-distance, endpoint-domain, residue, empty and unavailable-export checks |
| Physical reservation separation | `Recognition::checkDisjoint` | Existing sorting and adjacent-overlap scan; no pairwise family matrix |

The reservation path still costs `O(S log S)` for sorting supplied intervals;
this receipt makes no linear ordered-input claim. Recognizing an arithmetic
fragment does not establish a small general-integer composition or selector.

## Boundary and scope limits

The retained local lifetime window and the separately certified global-query
transport window are distinct objects. Requesting global queries still checks
its additional mixing and physical-refresh premises. A local refresh bound is
not a readiness proof for an entry, an escaping result, or arbitrary history.

The positive boundary fixture imports a bounded child between concrete finite
children and reconstructs original prerequisites at the parent. Its seed reads
a disjoint tile, so storage overlap alone cannot establish its value-readiness
edge. Entry and exit event queries include all four event-kind combinations
and zero trips. Relevant escaping loop results and unsupported carried state
remain explicit refusals; no last-occurrence or zero-trip result map is invented.

No runtime evaluator, new analysis backend, extra completion barrier, changed
notification protocol or relaxed physical-ID rule is introduced.

## Qualification

The incremental native build passes using the existing LLVM toolchain. The
final focused gate passed 28/28 in 81.57 seconds, including the expanded
production-trace check. The phase-control check covers 28 unfolded event closures,
20 ordinary causal-reuse traces and 20 actual combined-output physical traces.
The bound-input test checks 48 composed entry/exit event-kind answers, retained
local-map ownership, foreign/rebuilt-scope rejection, and carried/escaping-result
refusals. Very large trace arguments leave the compiled phase representation
unchanged; the actual loops are not expanded.

An initial newly authored nested fixture contained nonadjacent same-pipe hazards:
external bank-zero writes surrounded an active bank-one visit. Its direct local
barrier strengthens payload order in both baseline and candidate. Their complete
payload closures agree; the different matching tuple encodings do not change it.
The order-equality fixture now omits those external writes to satisfy the draft's
adjacency premise, retaining the inner-threshold counterexample and both phase
polarities. This does not establish least realization for nonadjacent demands.

Three interleaved serial production pairs use the combined six-ID request:

| Fixture | Baseline outcome | Candidate outcome | Evidence |
|---|---|---|---|
| Period-two control | Logical preparation unavailable | Physical output succeeds | Candidate median 0.157 s; no speedup ratio |
| Period-three control | Logical preparation unavailable | Physical output succeeds | Candidate median 0.156 s; no speedup ratio |
| Mixed absolute/periodic control | Physical output succeeds | Physical output succeeds | Median paired candidate/baseline 0.992; neutral |
| Ordinary symbolic-inner repeat | Allocation export unavailable | Same refusal | No capacity lower bound or successful-allocation claim |

The mixed fixture selects the new sequence/phase representation instead of the
baseline arithmetic child. Its final arithmetic increases from 125 to 300 and
static SET/WAIT sites from 3/3 to 4/4. The independent traces pass, but this is a
code-size regression, not a control-size improvement. Candidate selection does
not compare the completed costs of all alternative child representations.

The compliance prefilter's one line-length finding is fixed. Its remaining 17
brace diagnostics were inspected: all flagged statements have braced bodies;
the regular expression misparses nested calls or multi-line conditions. No
suppression is added. The final changed-source compilation has no new warning;
unchanged covered-switch warnings occurred in the broad header rebuild. No
external static analyzer or device run is claimed.

The complete synchronization gate and 49-module campaign remain M5 obligations.
