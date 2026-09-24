# FrontierSynch first-pass port

## Source and boundary

Recipient: `codex/handoff-foundation`, foundation `dbd99c242`.
Donor: `codex/oahs-gemm-base` at
`371fdb344d2783b92d6c39424c507b2ce082e08c`.
Historical source-port specification: synchronization draft revision 0.40, paper repository
`7e3f59c`, Section 3 (original storage analysis and D1–D4).

Historical comparison baseline: revision 0.42 at paper repository `267c435`,
implementation `3bbc58a67`, reviewed 2026-09-24. The [current v0.44 parity
inventory](frontier-synch-v0.44-parity.md) supersedes that baseline. The stages
below record the 0.40/0.41 extraction history; they do not claim implementation
of the later joint refinements. See the [0.42 gap inventory](frontier-synch-v0.42-gaps.md)
and [Phase A contract](frontier-synch-program-analysis.md).

Port existing analyses in dependency order. Preserve their implementation and
record differences from the working draft before adapting their consumers.
The shared `SyncInput` remains the source of instruction effects for both
InsertSync modes. The analyses below belong to FrontierSynch Phase A.

## Stages

1. **Scalar and descriptor facts — source port present.** Copy
   `SyncSlotMapping.h` and `SyncTileDescriptorState.h` from the donor's
   `InsertSync` directory into `FrontierSynch`. Only include guards and the descriptor
   header's local include path change. Existing type names are retained.
2. **Physical storage and original control — ported and connected.** Extract storage origins, physical
   uses, interval partitioning and original structured owner/site identities.
   Consume `SyncInput` records; isolate the donor's dependencies on its old
   `Program` representation. Connect scalar observations to actual address
   slices. Preserve conservative footprints when exact matching is unknown.
3. **Occurrences and composed reader frontiers — source port connected.** Port original read summaries,
   guarded participation and child/re-entry correspondence (D1–D4). Distinguish
   independent participants from alternative ones. Retain original owners and
   executable predicate availability without multiplying independent selectors.
4. **Lifetimes and constructor-facing requirements — source port connected.** Port generation/support
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

The FrontierSynch entry now calls `importOriginalStructure(function, input, result)`.
The result owns original region/cell records and descriptor facts; it borrows
MLIR values, original operations and translated phase pointers from the unchanged
function and `SyncInput`. A failed control import leaves the output unchanged.

| Donor source | Extracted component | Adaptation |
| --- | --- | --- |
| `OAHS/Plan.h` | Original region, access and cell records | Extracted into `OriginalStructure.h`; no dependency on the old selected plan or target/event state. Phase records reference the shared translated instruction. |
| `OAHS/Native.cpp`, original-region import | Sequence, choice, for/while tree and stable original owners | Every translated phase is retained, including multiple phases at one source operation. Multi-block regions are refused explicitly. |
| `InsertSync/SyncOriginClosure.h` and `SyncOriginPropagation.h` | Structured SSA origin graph and delta fixed point | Read-only origin query; no rewrite or second translation of instruction effects. Carried/unknown geometry is conservatively widened in FrontierSynch-private copies. |
| `OAHS/Native.cpp`, physical extraction | Address-dependency slices, finite address sets, footprint grouping | Queries one selector at a time, retains its owner and addresses. No copied operation populations or selected occurrence dimension. |
| `MemoryDependentAnalyzer::storageCoordinates` | Checked local absolute / GM root-relative interval qualification | Extracted into the FrontierSynch importer; production alias behavior is untouched. |
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

## Stage 3 occurrence and reader-frontier source port

`OriginalReadQueries` now consumes the Stage 2 original region tree and physical
accesses in `frontiersynch::run`. It ports the donor's interned, guarded first/last
reader expressions and sequence/choice/counted-loop composition. A read interval
returns `NoHit`, an exact conditional frontier, or `Unknown` with a reason.
Independent readers remain conjunctive; mutually exclusive readers retain their
original predicates. Write-delimited segments are structural intervals, not
proofs of an executed generation or complete overwrite.

`OccurrenceQueries` supplies a restricted D1 fixed-visit query, D2 bank
correspondence, and original child membership/skip/repeat facts for D4 clients.
The D2 certificate requires a qualified counted owner, one address per residue,
disjoint bank extents, and mandatory represented uses in each visit. A finite
noninjective or conditional relation retains its Stage 2 may-footprints but does
not acquire an exact predecessor-use distance. No event pool or selected command
participates in these queries.

The reader query checks whether its predicate is available at each read-site
leaf. That is narrower than executable availability at a future SET/WAIT gap;
Later endpoint qualification must check the actual proposed position. The
current original-child query retains source identities and members, but does not
transport selected credit or copied child interfaces. D1 guarded alternative
origins, general D2 permutations, and full D4 re-entry correspondence remain
unported. These are
representation limits, not hardware restrictions. Unknown results preserve the
original access obligations.

The normal FrontierSynch entry is the real consumer: it queries represented reader
cell/engine projections and bank relations, then reports counts before its
existing construction-unimplemented result. It does not claim that an analyzed
frontier is already a selected protocol. Stage 4 adds source milestones and
typed original requirements.

The serial current-source probe at
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage3/` checks two
optional siblings, a predicate unavailable at the first read, an exact fixed
visit, a nested child interval whose whole-loop participation is unknown, and
separate noninjective period-4 and exact period-3 address relations. The exact
relation survives the unrelated noninjective one. The focused build links
existing generated dialect/LLVM dependencies; it is not a full compiler build.
Query work is indexed by original read incidences and memoized structural
subtrees. The current entry still queries each represented cell/reader
projection separately; aggregate cost for large projection populations remains
to be measured. Generality review, full service, corpus and device evidence
remain pending.

## Stage 4 original lifetimes and requirement subscriptions

`OriginalLifetimes` adapts the donor `StorageFrontierAnalysis`'s four marginal
provenance flows (previous/next writers and readers) to `OriginalStructure` and
the Stage 3 control graph. Partial or unknown writes retain older origins; only
an independently proved full overwrite may kill them. The current importer does
not set `definiteWrite`, so this port makes no fresh-generation claim merely
because a translated operation writes a bounded footprint. Read-modify-write
effects retain both reader and writer origins.

The port indexes RAW, WAR and WAW original requirements by target deadline and
registers their source subscriptions before construction begins. Source-after
and target-before are stable original positions. They are not complete ordered
command-word gaps or selected SET/WAIT endpoints. A marginal source is not an
occurrence match, executable guard, established episode or completed transfer;
those fields remain explicitly unqualified. The future constructor must query
the current residual and causal frontier before selecting any communication.

Typed read-only queries expose `all` original accesses by physical cell/owner,
first/last original-use may frontiers within explicit starts and stops, and the
affected support interval between a represented producer and possible reuse.
The support interval retains readers, intervening reloads, bypass/re-entry
possibilities and **all** original requirements encountered on those paths,
including other cells. Its `complete` flag means the finite original graph was
traversed; it does not authorize a protocol or claim complete producer support
under future selected edits. Endpoint participation and source-prefix
preservation still require the Stage 3 guarded expressions and later selected
ledger checks. The current first/last use query returns marginal access sets;
it is not a guarded canonical frontier for arbitrary access classes. An owner
argument identifies the original scope but does not itself prove occurrence
matching or impose a lexical stop.

The FrontierSynch entry consumes the indexed requirements and reports their
population while retaining its construction-unimplemented result. A normal
source-port probe covers two optional readers before reuse, a reload between
them, source subscriptions, partial-write history retention, and the affected
support interval. Stage 3 probes continue to pass. Artifacts are at
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage4/`.

Remaining first-pass work before claiming the full draft contract: full-write
coverage certificates; guarded alternative origins and general D2/D4
correspondence; owner-qualified canonical first-conflict/last-use queries with
endpoint-gap predicate availability; and persistent support/source-prefix
dependencies through selected edits. These are explicit precision and
construction boundaries, not hardware limits or permission to drop original
requirements. The focused source build does not establish full native service,
corpus quality or generality acceptance.

## Coherent Phase A result after the source ports

The source ports were assembled through `ProgramAnalysis`, following the then-current
draft v0.41's three-way distinction, retained in v0.42: fixed original facts, complete physical
requirements, and placement opportunities. The result owns the immutable
original structure and shared query services. It indexes every requirement by
deadline and direction, subscribes every represented source position before
construction, and decodes an applicable request on demand. The decoded answer
retains the original physical-use owner, source and deadline positions,
translated effect identity, independent bank evidence, guarded reader frontiers
and explicit unresolved premises. Alternative origins, source/target histories
and fixed-visit correspondence are queried only when needed; eagerly copying or
searching them for every marginal edge can multiply work.

`All` is a conservative may summary; `MayAfter` uses a designated original
continuation; first/last results use a qualified reader interval or return
Unknown with retained may accesses. A source subscription, return opportunity,
or structural no-hit grants no selected completion.

This is an interface consolidation, not a claim that every draft derivation is
now exact. Imported full writes remain unproved; D1 alternative guards, broad
D2/D4 correspondence, write-frontier selection and endpoint-gap predicate
availability still need proofs. The counted reader rule now accepts a
nonconstant but iteration-invariant original condition when its value is
available before the loop; a condition recomputed from the loop induction
remains unresolved. [The Phase A contract](frontier-synch-program-analysis.md) records these
limits and the LLVM/MLIR replacement audit.
