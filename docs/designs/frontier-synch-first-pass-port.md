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
2. **Physical storage and original control.** Extract storage origins, physical
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
Wire the assembled Phase A into `frontiersynch::run` once its physical/control input
exists; this first source port is not yet invoked by that entry point.

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
