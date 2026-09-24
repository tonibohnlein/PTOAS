# FrontierSynch first-pass port

## Source and boundary

Recipient: `codex/handoff-foundation`, foundation `e7537ad90`.
Donor: `codex/oahs-gemm-base` at
`371fdb344d2783b92d6c39424c507b2ce082e08c`.
Specification checked: synchronization draft revision 0.40, paper repository
`7e3f59c`, Section 3 (original storage analysis and D1–D4).

Port existing analyses in dependency order. Preserve their implementation and
record differences from the working draft before adapting their consumers.
The shared `SyncInput` remains the source of instruction effects for both
InsertSync modes. The analyses below belong to handoff Phase A.

## Stages

1. **Scalar and descriptor facts — source port present.** Copy
   `SyncSlotMapping.h` and `SyncTileDescriptorState.h` from the donor's
   `InsertSync` directory into `Handoff`. Only include guards and the descriptor
   header's local include path change. Existing type names are retained.
2. **Physical storage and original control — ported and connected.** Extract storage origins, physical
   uses, interval partitioning and original structured owner/site identities.
   Consume `SyncInput` records; isolate the donor's dependencies on its old
   `Program` representation. Connect scalar observations to actual address
   slices. Preserve conservative footprints when exact matching is unknown.
3. **Occurrences and composed reader frontiers.** Port original read summaries,
   guarded participation and child/re-entry correspondence (D1–D4). Distinguish
   independent participants from alternative ones. Retain original owners and
   executable predicate availability without multiplying independent selectors.
4. **Lifetimes and constructor-facing requirements.** Port generation/support
   intervals, distinct source milestones and consumer deadlines. Connect typed
   original-use queries and source subscriptions to the new constructor input.
   Selected completion remains the constructor's responsibility.

For each stage: extract the donor dependency closure, identify its real consumer,
check the draft contract, and record remaining representation limits. Do not
bring over a constructor dependency merely to make an analysis header compile.
The stage-2 physical/control import now runs in `frontiersynch::run` and consumes
the stage-1 scalar and descriptor analyses. Construction remains the next
independent layer after the staged analysis port.

## Stage 1 crosswalk and limits

| Ported service | Draft responsibility | Remaining boundary |
| --- | --- | --- |
| Checked scalar constants, folding and range queries | Original scalar facts with integer semantics | Supported nonnegative values, widths up to 64 bits; index treated as 64-bit. Unsupported forms return unknown. |
| Original counted-loop domain | Logical visit interpretation independent of raw induction spelling | Qualified nonnegative lower bound, positive step and overflow proof; this is a sufficient implementation restriction. |
| Address-observation dependency slice and finite recurrence | Scalar input to physical selection in D2 | Only relevant carried dependencies are explored. Default 256-state precision budget is independent of event capacity. This is bounded enumeration, not the draft's complete physical permutation decoder. |
| Descriptor state at original uses | Effective dimensions as original facts | Exact rank-two dimensions; equal branch values survive joins; loop-mutated state becomes unknown. Descriptor identity is distinct from storage aliasing. |

A scalar orbit does not establish disjoint banks, predecessor physical uses,
participation, generation identity or completion. The physical address map and
occurrence premises must be checked in later stages. A noninjective address map
cannot inherit the D2 permutation result. Independent observations receive
separate queries; callers must not create an unrelated global phase product.

Descriptor knowledge is a fact for later consumers, not an ACC ordering
exemption. Native access protection needs its separately scoped target contract.
Unknown scalar/descriptor results must preserve the original conservative
instruction effects supplied by `SyncInput`.

## Evidence

The port is checked against the pinned donor after reversing the two mechanical
path/guard changes. Focused header compilation is recorded in `HANDOFF.md`.
This establishes extraction fidelity, not full first-pass or native integration
acceptance. Constructor, corpus and device validation belong to later stages.

## Stage 2 extraction and draft check

The handoff entry now calls `importOriginalStructure(function, input, result)`.
The result owns original region/cell records and descriptor facts; it borrows
MLIR values, original operations and translated phase pointers from the unchanged
function and `SyncInput`. A failed control import leaves the output unchanged.

| Donor source | Extracted component | Adaptation |
| --- | --- | --- |
| `OAHS/Plan.h` | Original region, access and cell records | Extracted into `OriginalStructure.h`; no dependency on the old selected plan or target/event state. Phase records reference the shared translated instruction. |
| `OAHS/Native.cpp`, original-region import | Sequence, choice, for/while tree and stable original owners | Every translated phase is retained, including multiple phases at one source operation. Multi-block regions are refused explicitly. |
| `InsertSync/SyncOriginClosure.h` and `SyncOriginPropagation.h` | Structured SSA origin graph and delta fixed point | Read-only origin query; no rewrite or second translation of instruction effects. Carried/unknown geometry is conservatively widened in handoff-private copies. |
| `OAHS/Native.cpp`, physical extraction | Address-dependency slices, finite address sets, footprint grouping | Queries one selector at a time, retains its owner and addresses. No copied operation populations or selected occurrence dimension. |
| `MemoryDependentAnalyzer::storageCoordinates` | Checked local absolute / GM root-relative interval qualification | Extracted into the handoff importer; production alias behavior is untouched. |
| `OAHS/StorageWitnesses.h::appendCanonicalStorage` | Endpoint partition and pairwise conservative overlap witnesses | Uses the original record type and size-sized cell IDs. The unused older unpartitioned builder is omitted. Braces/formatting are mechanical changes. |

This implements Section 3's physical-fact/control foundation. Exact single
intervals in the same qualified coordinate space share canonical byte atoms.
An unknown footprint overlapping two disjoint exact intervals does not make
those intervals alias each other. A finite multi-address footprint remains a
may-set. Its scalar period supplies no D2 physical-permutation proof, no
predecessor reader, no event resource, and no completion credit.

### Preserved and temporary premises

- The shared translator remains the effect authority. Root closure enriches the
  provenance of represented effects; it does not recover effects omitted by
  translation. The donor's post-translation effect-refresh machinery is not
  part of this read-only analysis port.
- Region import requires verified single-block structured regions and retains
  every translated phase. Unsupported structure is a representation refusal,
  not a hardware limit. Original predicates remain available through owner
  anchors; executable guarded frontiers are stage 3.
- A physical address specialization requires a qualified allocation-root address
  slice and nonoverflowing translated extent/offsets. Views use the root's
  address and their translated offsets. Unsupported roots keep their original
  conservative effects. Extending address-root proofs belongs to the shared
  physical query, not a new opcode or kernel handler.
- Loop-carried or unknown origin geometry is widened while its possible root
  identities survive. This is an explicit sufficient restriction of this port;
  original-relative view/occurrence proofs must replace it before precision for
  those uses is claimed. No exact-cell or definite-write fact follows from a
  root identity alone.
- Imported writes retain `definiteWrite=false`. Bounding extents do not establish
  complete overwrite. Stage 4 must consume an effect-coverage certificate before
  applying provenance kills.
- Finite address relations are independent. The independent period-2/period-3
  example retains two relations; it does not generate a period-6 control graph.
  The scalar exploration limit remains the stage-1 precision budget.

### Work and validation boundary

Origin propagation uses the donor delta worklist. Canonical partition work is
charged to distinct footprint endpoints and output incidences; conservative
alias pairs can be quadratic in distinct footprints. Scalar exploration remains
bounded per queried footprint/owner and uses the shared scalar fact caches.
There is no event allocation, candidate-plan replay or dynamic-loop unrolling.

Focused current-source importer probes cover equivalent induction/carried
selectors plus unrelated carried state, independent selectors, partially unknown
addresses, GM aliases, and nested original control. A component probe checks
nontransitive overlap and cyclic origin propagation. Source IR is compared before
and after import, and every translated phase must appear once in the region tree.
Exact commands, inputs and build logs are kept with the artifact path in
`HANDOFF.md`. Full pass build, generality acceptance and synchronization-quality
claims remain outside this extraction's evidence.
