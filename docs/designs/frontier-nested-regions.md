# Nested-region analysis implementation

This implements constructed subclasses of Section 5's repeated compact-region
result. It analyzes the selected order induced by the shared access model.
Conservative access ranges remain part of that model; no instruction-specific
footprint whitelist or kernel recognizer is added.

## Implemented layers

| Layer | Constructed interface | Current boundary |
|---|---|---|
| Hierarchical identities | Static payload type, local ordinal, enclosing visit coordinates; substitution and node-scoped analysis | Original MLIR region structure and cuts are retained |
| Invariant repetition | One body analysis, two-copy crossings, reduced inter-visit demands, weighted port queries, first/last selectors | Body control and access maps must be invariant in the outer visit; carried state needs another adapter |
| Explicit phases | One finite list of specialized bodies, complete-period composition and final-prefix selectors | Current producer specializes a guarded rotating inner loop; general compact child specialization is unavailable |
| Symbolic storage | Constant-stride visit ownership and shared read-only families, exact byte-owner and first/last selectors | Finite local footprints independent of inner occurrences; external symbolic crossing composition remains unavailable |
| Dispatch and reporting | Reuses existing numerical routes, then constructed repeated routes; records missing interfaces and charged expansion | Logical readiness and physical allocation readiness are separate |

The invariant construction can recur at arbitrary supported depth. That is an
inductive interface property, not a guarantee that every nest is recognized or
that circuit size is polynomial in nesting depth. A surrounding invariant loop
can use a phased child. Combining phase specialization with symbolic ownership,
or repeating an already symbolic-owned interface, still requires adapters.

See [phase composition](frontier-repeated-phases.md) and
[symbolic storage](frontier-repeated-storage.md) for the precise producer contracts.

## Dispatch and reuse

The existing sequence traversal fuses explicit runs and analyzes compact children
on their original region nodes. Child queries retain full occurrence identities;
composition does not replace a child by one atomic payload.

Numerical regional templates remain preferred where their inner bounds are
statically enumerable and the existing path can allocate physical IDs. The
nonexpanding bound check preserves folded constants and bounds specialized by
an enclosing inner induction variable. Symbolic bounds reach compact repetition
without first enumerating visits. Numerical expansion is reported separately as
`numeric_visits`; explicit phase construction as `phase_descriptions`.

A rejected repeated route does not certify that the program lies outside the
paper's class. Reports distinguish missing producers or adapters from failed
arithmetic/storage certificates. The fallback does not silently drop unexported
writers or reset persistent storage at re-entry.

## Logical insertion and physical allocation

Repeated recipes attach SET/WAIT and local barriers to the original payload
cuts. Matching records include enclosing visit coordinates and phase membership;
zero trips, absent inner work and partial periods suppress unmatched commands.
Existing same-pipe handling and shared hardware rules continue to apply.

New nested logical identities use metadata version 4. The allocator reports
`nested logical-plan physical allocation is not implemented yet` for these
plans. This work does not add scarcity repair or claim end-to-end compilation
for the new symbolic nests. Existing numerical plans retain their allocation path.

## Cost and validation

Cost is charged to phase descriptions, boundary ports, child query circuits and
emitted endpoint circuits. A P-port guarded closure takes cubic semiring
updates plus child-query construction; byte-selector construction takes
quadratic candidate comparisons plus their reference-query work.
Neither enumerates the runtime trip count. Large generated circuits can still
make the result impractical: the measured three-level phase stress fixture
emitted about 338,000 scalar preparation operations. Generic factoring and
invariant hoisting remain necessary performance work.

Focused validation compares both query answers and the closure of actually
emitted commands against independent finite unfoldings. It includes zero trips,
partial periods, nonzero loop lower bounds, nonunit steps, restarted banks and
three-level nesting. Large trace arguments verify unchanged emitted loop
structure. The interpreter has its normal finite limit; the named three-level
stress test explicitly requests a larger bounded limit.

## Final validation and selected corpus

Serial incremental compilation and linking passed with the local resource
safeguards. Focused checks passed:

- 120,960 invariant-repeat event-pair comparisons.
- 2,816 phased event queries and exact storage selectors.
- 14 invariant-repeat emitted-command closures.
- 32 phase emitted-command closures, including the complete three-level stress
  case (3,804,098 interpreted scalar visits under its explicit 8,000,000 ceiling).
- Byte-owner/selector checks with zero trips, holes, read/write boundaries,
  negative-origin rejection, bounded symbolic origins, and overflow rejection.
- Seven symbolic-storage production closures, with a separate assertion that
  sequence analysis selected repetition and constructed its storage interface.
- 32 existing sequence command closures and 15 allocation traces.

The corpus comparison used the same 326 inputs as the M2 baseline, consisting of
all 71 captured PyPTO/PyPTO-lib modules, 251 selected PTOAS modules with multiple
loops, and four development cases. It is a subset, not a full-corpus claim.
Each compiler invocation was serial, with a 15-second timeout and 1.5-GiB virtual
address-space ceiling; none timed out. Both runs used the shared may-not-alias
GM policy, and allocation was supplied physical IDs 0 through 5.

| Source | Modules | Logical plans | Allocated plans |
|---|---:|---:|---:|
| pypto-lib | 49 | 7 | 7 |
| pypto | 22 | 19 | 19 |
| ptoas | 251 | 24 | 23 |
| development | 4 | 2 | 2 |
| **Total** | **326** | **52** | **51** |

Every per-module logical/allocation status matches the M2 baseline. No new
version-four nested plan was selected in this subset; the newly constructed
routes currently have focused-fixture coverage, not demonstrated corpus gains.
The one allocation failure reports a sufficient palette of eight IDs, without
claiming that eight is minimal. Scarcity repair remains unimplemented.

The 274 logical failures retain route-specific reasons. Frequent missing inputs
or adapters include invariant control/descriptors, nonnegative constant loop
lower bounds and positive steps, carried loop results, child phase specialization,
and streaming or symbolic storage interfaces. Rejection by these implemented
recognizers does not establish that the draft's full theorem cannot apply.

Detailed local evidence is under `.local/nested-regions/`: `corpus-m2/` and
`corpus-final/` contain per-input reports and binary fingerprints;
`corpus-cases.json` records shared input fingerprints. Milestone
commit previews record reviewed paths and checks. The final reporting-only fix
corrects `queries_available` for failed analyses; it does not change production
analysis or the corpus compiler.
