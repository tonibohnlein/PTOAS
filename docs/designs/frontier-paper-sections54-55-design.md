# Practical demand analysis and regional composition

## Purpose and proposed direction

Sections 5.4 and 5.5 should explain how a compiler obtains minimum demands for
useful structured regions and combines those results into a program analysis.
The starting point is a lowered program with fixed physical storage, payload
assignments and pipe order. The compiler still has to determine which bytes
each operation accesses on each visit. It then constructs demand records that
describe the required occurrence pairs across all admitted executions.

Organize Section 5.4 around two principal routes: repeated buffer pipelines and
restricted arithmetic accesses. They are overlapping routes for regions; a
kernel may use both. Section 5.5 supplies their common composition interface.
Keep Sections 5.1–5.3 as the mathematical foundation. Retain bounded-lifetime
analysis as an explicitly scoped extension for irregular guards; its theorem
and clients need not disappear when the practical presentation changes.

Use the existing PTOAS implementation as the first integration. Reuse MLIR
facilities within it and evaluate an AIR adapter only against a concrete need.
This design proposes a paper organization and an implementation qualification
milestone. It does not replace the active
[implementation plan](frontier-family-pipeline-plan.md), introduce another
dispatcher, or change the selected synchronization protocol.

## Start with physical access descriptions

Consider a load that writes `x[i % 2]` and a computation that reads it. Once the
compiler establishes the physical identity and accessed portion of each slot,
it can derive readiness from load to computation and release from computation
to the load two iterations later. Knowing only that both operations mention
`x` does not establish those relationships. Different views can overlap, and
different allocation operations can denote the same reused physical bytes.

The opening of Section 5 should distinguish three inputs:

1. **Program facts:** operation semantics, physical allocations, descriptor
   state, loop bounds, conditions, pipe assignments and supplied prerequisites.
2. **Access descriptions:** the bytes read and written by each occurrence,
   represented by exact cells, rotating fragments or integer relations.
3. **Demand generators:** forward conflicting occurrence pairs, together with
   supplied non-memory prerequisites, whose closure with native order is the
   selected required order.

The finite and compact reduction results may take the second or third input as
given. A practical compiler route must also explain how it constructs that
input from the first, and charge that work. Operation effect interfaces and
alias information are useful inputs; a read/write classification alone does
not establish an exact physical footprint.

The implementation already makes an important distinction in
[SyncStorageEffects.h](../../include/PTO/Transforms/InsertSync/SyncStorageEffects.h):
an allocation enclosure and a declared access region describe the supplied
model, and materializing that model does not certify native instruction
footprint precision. Preserve this distinction in the paper-to-code mapping.
For each demonstration, record the alias assumptions and which operation
footprints have been qualified. If only a conservative access model is known,
report minimum demands for its selected order rather than claiming the
minimum for the original physical accesses.

## Section 5.4 outline

Suggested title: **Practical compact demand analysis**.

### Repeated buffer pipelines

Begin with the two-slot example above, then show why nesting matters: an inner
loop can restart its slot counter while the physical buffers survive into the
next outer visit. Extracting the inner body's records is a leaf analysis;
connecting visits is the composition problem developed in Section 5.5.

Present the route in this order:

1. **Recognize the access representation.** Establish exact disjoint buffer
   families and within-slot atoms. Normalize supported fragments to
   `X_l[(s_l*i + c) mod b_l]`, with compile-time slot count, stride and offset,
   a common stride within each family, and known access modes. Account for
   physical alias resolution and atomization before counting fragments.
2. **Extract generators.** Resolve each family's modular access order and
   obtain previous-writer, reader and next-overwrite relationships. Combine
   records from all families and compatible supplied prerequisites. A path
   through one buffer may make another buffer's direct demand redundant.
3. **Reduce the periodic representation.** Apply the existing quotient
   algorithms with separate start and completion events. Use the native-scan
   specialization only when its core-chain premises hold; retain the generic
   route for additional native relationships.
4. **Handle boundaries and produce exports.** Restrict record instances to
   present endpoints. Analyze prologues, tails and other changes of body as
   regions, or use an existing certificate that covers them. Construct the
   requested reachability and storage selectors separately.

The first theorem-level contract remains the existing common-stride,
fixed-body result. Periodic presence or invariant guarded specializations can
follow with their actual premises. Arbitrary iteration-dependent guards do not
inherit a periodic certificate merely because addresses use modulo two.

State the extractor's `O(A log(A+1))` arithmetic/comparison cost after normalized
exact fragments are available, including the separate per-family arithmetic
and bit-cost qualifications. Then state reduction and export costs in their
own representation sizes. A quotient whose size is independent of the trip
count does not imply that all boundary selectors or emitted code have the same
bound. In particular, report any slot enumeration in regional adapters.

### Restricted arithmetic accesses

Introduce this route with a finite nest whose bounds or guards change the
executed body, such as `0 <= j <= i < N` or a last-tile condition `i+1 < T`.
An access to a fixed-width interval at `base + 2048*i` has a simple arithmetic
description. An address loaded from `page_table[i]` requires additional facts
about that table; the affine index into the table does not determine its value.

Separate source extraction from recognition of the resulting relations:

1. Recover supported occurrence domains, access maps, order, native
   reachability and supplied prerequisites from the original IR. Preserve
   machine arithmetic, descriptor updates and the relevant control context.
2. Check every primitive against the declared arithmetic profile. The
   existing polynomial theorem fixes pipe count, numeric dimension, common
   residue period and, for bounded coefficients, the coefficient bound.
   Parameters and access auxiliaries count toward the dimension.
3. Construct conflict relations, bounded-witness reachability and cover
   subtraction using the Section 5.3 construction and established integer
   relation algorithms. Build only requested selectors and query exports.

Recognition linear in the normalized input size is a claim about step 2.
It does not make extraction, residue expansion, projection or Boolean
subtraction linear. General affine syntax is not a sufficient class test.
Use the polynomial theorem as a guarantee under fixed class parameters;
establish practical speed by measuring the complete route.

### Extensions and route selection

End with a short account of bounded-lifetime analysis: it supports irregular
guards when a refresh/lifetime certificate is available, with cost depending
on the represented window. Keep its proof and streaming variant available;
move detailed mechanics to an appendix only after checking their references.

A short table can compare the two principal routes and this extension by
accepted representation, certificate, output and cost parameters. Numerical
and symbolic quotient implementations are choices within a route. Small-count
expansion is a representation choice. Neither needs another main-text class.
Request selection and caching belong in Section 8.

## Section 5.5 outline

Suggested title: **Composing exact regional analyses**.

### What a parent needs from a child

A child that only writes `x` has no internal storage demand, but a following
reader still needs that writer's completion. Conversely, a child's last reader
must precede a later overwrite. Internal minimum demands therefore do not
constitute a sufficient regional interface.

Retain the existing interface, explaining each component through this need:

| Export | Role in the parent |
| --- | --- |
| First and last writer of each cell | Connect preceding and following storage lifetimes |
| First readers before the first writer, and last readers after the last writer, per pipe | Preserve entry readiness and exit release obligations |
| First and last payload on each pipe | Connect native start and completion chains |
| Reachability queries and boundary chain thresholds | Test whether crossing requirements follow from existing paths |
| Occurrence coordinates and presence predicates | Identify actual endpoints, including empty or conditional children |
| Endpoint recipes, when requested | Realize selected pairs at original executable cuts |

Storage partitions must be compatible, or reconciled exactly with that work
charged. Supplied cross-boundary prerequisites are separate input. Absence of a
boundary occurrence must be represented, not replaced by a whole-child wait.
The mathematical interface and executable endpoint support remain separate
capabilities of the same retained result.

### Sequential composition

First describe merging `A; B`. Add last-writer-to-entry-reader,
last-writer-to-first-writer and exit-reader-to-first-writer bridges, along with
native links and supplied prerequisites. Query paths through both children to
remove redundant crossings. Reference-interval locality preserves internal
covers when these are contiguous regions in the reference order.

Then explain the compression: boundary events lie on ordered native chains,
so threshold indices can represent their reachability. Keep internal queries
available through the hierarchy while exposing only the ports the parent
needs. State selector-construction and query costs beside the merge bound;
the storage-summary merge bound alone is not the cost of the full analysis.

### Repetition through the interface

Show repetition as another use of the same child exports. A loop-counter reset
does not reset physical storage or discharge pending readers. Connect the last
accesses of one visit to the first relevant accesses of subsequent visits.

Present invariant repetition first, including the conditions under which
numerical repeated squaring uses logarithmically many merges. Then present
certified periodic boundary behavior for varying visits. State which layouts,
selectors, guards and native interfaces repeat, and how the certificate is
obtained. Neither arbitrary nesting nor a finite storage allocation establishes
these conditions on its own. Empty visits and zero trip counts remain explicit.

Keep specialized overlays and re-entry constructions as referenced extensions
where needed. The main explanation should finish with the nested-buffer
example: analyze a body, retain its storage boundary, connect visits and
compose the prologue and tail without imposing a whole-region completion.

## Existing implementation and remaining qualification

The inspected workspace was clean at
`e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44`; the manuscript was read at
`68990fa5f6bc9f3e098ce3191fc4334e821459e7`. This inventory identifies existing
interfaces and selected call sites. It is not a fresh execution or proof audit.

Paths below are relative to `include/PTO/Transforms/FrontierSynch/` unless
otherwise stated; implementation files use the corresponding `lib/` tree.

| Existing component | Design consequence and remaining check |
| --- | --- |
| `Structured/RotatingExtraction.h` | Already exports period-one generators from exact rotating fragments. Qualify how the demonstration's original accesses reach this form. |
| `Structured/GuardedPeriodicWork.h` and `SessionCosts.cpp` | Cost estimation now shares core-chain versus generic quotient work. The earlier audit's stale-estimate finding must not be carried forward as current fact. Measure extraction and export work as well. |
| `Arithmetic/ArithmeticRecognition.h` | Already distinguishes primitive roles and fixed declared limits. Record source extraction and normalized class checking separately. |
| `Shared/RegionalAnalysis.h`, `Shared/ChainInterface.h` | Storage selectors and chain interfaces already exist. Identify which exporters retain compact storage and which enumerate slots or selector partitions. |
| `Shared/SequenceAnalysis.h`, `SessionProducers.cpp` | Endpoint upgrades already attempt reuse of retained sequence mathematics. Changed representations may still require exact recomposition; measure reuse without assuming every crossing survives. |
| `AnalysisRequest.h`, `ResultGuarantees.h` | Mathematical readiness, exports and realization guarantees are already distinct. Keep this ownership and report precision relative to the shared modeled effects. |
| `Shared/LogicalInsertion.h`, `Shared/PhysicalAllocation.h` | Existing downstream consumers remain the destination for the demand records. Preparation or allocation failure must not erase exact mathematics. |

`maxRegionalSlotVisits` currently limits selected regional exporters and
partition builders. This is an implementation boundary, not the definition of
the periodic mathematical class. Record refusal at the export stage when that
is where it occurs. A future compact exporter would need its own argument and
measurement; this design does not remove the existing checks.

The [current implementation narrative](frontier-family-pipeline-plan.md) and
its linked receipts supersede older status tables. They document substantial
implemented work and remaining limits. Reuse those tests and receipts, then
pin fresh results for the selected milestone rather than treating historical
successes as validation of this checkout.

## Reuse of MLIR and AIR

Use upstream MLIR control, SSA, rewriting and integer-expression facilities
where their semantics fit. Inventory existing relation operations before
replacing them: machine-width arithmetic, exact integer projection and
representation-size guarantees may require more than an API with a similar
name. Standard dependence and DAG-reduction algorithms remain prior machinery.
Our design obligation is to connect their premises to physical effects,
compact occurrences, regional interfaces and executable consuming flags.

AIR integration is an optional adapter experiment after this connection is
demonstrated. It would translate fixed operations, accesses, control and native
order into the existing analysis contract. Generic asynchronous completion
tokens do not by themselves establish the required physical access precision,
pipe semantics or consuming-ID protocol. Compare the same fixed payload and
storage problem before comparing whole compiler pipelines that transform it.

## First implementation qualification milestone

Use the existing prepared
[persistent GEMM input](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto)
and its [provenance](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/port.json).
It contains nested loops, two-stage storage, guarded prefetch, counter resets
and persistent physical buffers. Its declared GM disjointness and added
terminal completion are part of the comparison contract. The provenance does
not claim device qualification.

1. **Trace one existing production path.** Pin source, tool, input, alias policy
   and requested capabilities. Record which route analyzes each region and
   which adapter provides its boundaries. Begin by documenting existing
   behavior; introduce code only for a demonstrated missing obligation.
2. **Connect accesses to demands.** For the local buffers, show the physical
   maps, normalized fragments, generator records and reduced result. Account
   for GM effects and non-memory prerequisites explicitly. Preserve a record
   of every conservative enclosure or unresolved qualification.
3. **Exercise composition.** Follow one inner-loop seam and one outer reset,
   including entry, tail and an absent guarded prefetch. Verify that child
   queries and storage selectors suffice without flattening all visits.
4. **Validate small instances independently.** Derive concrete accesses and
   the split-event graph independently of the selected compact reducer;
   compare closure, covers and boundary selectors. Check zero, one and short
   trip counts, including counts below and above the buffer depth. Existing
   rotating, sequence and repeated-region tests supply starting points.
5. **Check executable output separately.** Verify endpoint availability,
   dynamic matching, adjacency conditions for order equality, cleanup and
   consumption-before-republication. Distinguish logical success, physical
   allocation, generated C++, device compilation and device execution.
6. **Measure the complete construction.** Vary trip count, number of static
   stages and nesting independently on controlled derivatives. Record access
   recovery, qualification, generator extraction, reduction, query/selector
   construction, composition, endpoint preparation and allocation. Attribute
   atom, residue, relation-piece and expression growth to the responsible
   stage. Audit mode is a different workload from production dispatch.

A second, small arithmetic fixture should demonstrate source extraction and
class checking on a nonperiodic iteration domain, using the existing arithmetic
tests as a base. This prevents the two-route presentation from resting entirely
on a periodic example. Full corpus expansion follows these focused receipts.

The milestone is complete when the retained mathematical result, requested
regional exports and executable outcome are accounted for on the same inputs,
with costs attributed to their stages. Successful reduction alone is not an
end-to-end result, and failure of one sufficient allocation is not a proof
that the target capacity cannot suffice.

## Manuscript revision sequence

First agree on this outline and the precise input contract. Next qualify the
existing implementation path and repair only demonstrated gaps. Then rewrite
5.4 and 5.5 around the resulting examples, preserving theorem hypotheses and
moving detailed extensions with their references intact. Finish by updating
Section 5's opening and Section 8's cost accounting to charge the same stages.
The central claim should be that supported compact access descriptions lead
to minimum demands that can be composed and, when endpoint and resource
conditions hold, realized by one executable synchronization program.
