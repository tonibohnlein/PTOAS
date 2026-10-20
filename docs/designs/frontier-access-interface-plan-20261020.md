# A consistent access interface for synchronization

**Status:** implementation proposal, based on a read-only source inspection on
20 October 2026. This document does not implement the proposed changes.

**Workspace:** `workspaces/frontier-shared-input`, branch
`codex/frontier-shared-input`, base HEAD
`e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44`.
The inspected tree contains uncommitted compiler repairs and experiment tools;
HEAD alone does not reproduce it. The source fingerprints at the end identify
selected files inspected for this proposal.

## 1. Goal and recommended scope

The synchronization pass should consume one consistent description of each
memory access. That description must preserve useful symbolic geometry, say
which assumptions justify it, and provide a sound conservative interpretation
when some geometry is unavailable. Finite reduction, compact analysis, regional
composition, and completion fallback should agree about that interpretation.

The branch already contains most of this infrastructure. The project should
consolidate its contracts and fill demonstrated integration gaps. It should not
begin by replacing PTO buffer types or introducing another analysis backend.

The intended outcome for a production kernel is an explained result:

1. Exact demand analysis relative to qualified access descriptions, where supported.
2. Analysis of a declared conservative access model where precision is lost.
3. Conservative regional completion when the requested compact construction or
   executable export is unavailable.
4. A precise refusal when operation, control, or target semantics cannot support
   a sound plan.

An unknown address alone should not force an unexplained failure. Conversely,
knowing that an operation accesses GM does not establish its pipe, completion
semantics, guard availability, or a legal synchronization protocol.

### Non-goals for this project

- A new buffer dialect or compiler-wide MemRef migration.
- New periodic, arithmetic, or allocation algorithms.
- Runtime dependence tracking or runtime alias-based code versioning.
- More permissive native command semantics or new hardware protection assumptions.
- Increasing admission limits or unfolding a nest to force acceptance.
- Native/device qualification from host-side tests.

The [whole-kernel experiment work](frontier-pypto-region-survey-20261019.md)
provides motivating inputs. Its currently running extension should supply
baseline outcomes under both alias policies before production changes are
compared.

## 2. What the branch already provides

Paths below are relative to the worktree. These are existing components, not
proposed new modules.

| Component | Observed responsibility | Reuse decision |
|---|---|---|
| [`PTOAccessRegion.h`](../../include/PTO/IR/PTOAccessRegion.h), `lib/PTO/IR/PTOInterfaces/` | Access selections attached to `MemoryEffectOpInterface`; contextual valid-shape requirements; byte or coordinate selections; descriptor-only effects | Keep as the operation-side source of access meaning. Fix missing contracts here when justified. |
| [`SyncInput`](../../include/PTO/Transforms/InsertSync/SyncInput.h) | Owns translated instruction phases, memory records, declared effects, scalar facts, and descriptor state; supports fine-storage and coarse-completion input modes | Keep as the immutable session owner. Preserve the cheap coarse input path. |
| [`SyncAccessRegion`](../../include/PTO/Transforms/InsertSync/SyncAccessRegion.h) | Symbolic byte offsets, extents, base, scalar symbols, element width, and original iteration domains | Keep as the symbolic geometry representation. Unknown absolute GM addresses do not require discarding it. |
| [`SyncDescriptorState`](../../include/PTO/Transforms/InsertSync/SyncDescriptorState.h), [`SyncScalarFacts`](../../include/PTO/Transforms/InsertSync/SyncScalarFacts.h) | Descriptor facts at an access and scalar normalization for unchanged IR | Reuse their ownership and contextual reasoning rather than rebuilding a private walker per consumer. |
| [`SyncEffectRanges.cpp`](../../lib/PTO/Transforms/InsertSync/SyncEffectRanges.cpp) | Resolves declared selections; retains qualified allocation enclosures; preserves slot alternatives; refuses to treat cached loop-carried roots as all-visit bounds | Make this the central fine-input qualification/fallback boundary. |
| [`SyncStorageEffects`](../../include/PTO/Transforms/InsertSync/SyncStorageEffects.h) | Modeled effects, compact cell memberships, overlap/conflict queries, residual and uniform conflicts, cached independence checks | Keep as the common analysis-facing access service. Extend only missing contract information. |
| [`MemoryDependentAnalyzer.cpp`](../../lib/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.cpp) | Existing buffer/view alias checks, physical overlap across local allocation roots, GM policy | Reuse through the shared service; inventory legacy direct callers before consolidating them. |
| [`RegionalAnalysis`](../../include/PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h) | Original occurrence frames, access boundaries, deferred effects, storage selectors, query exports and symbolic storage certificates | Keep storage coverage separate from demand and executable capabilities. |
| [`RegionalSelectorPartition.cpp`](../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalSelectorPartition.cpp) | Shared production selector entry point; exact/absent/unavailable cell coverage; per-domain support and uniform cuts | Preserve the recent repair. Do not introduce a second selector fallback. |
| [`RepeatedReadOnlyStorage.cpp`](../../lib/PTO/Transforms/FrontierSynch/Shared/RepeatedReadOnlyStorage.cpp), `RepeatedStorageSupport.cpp` | Qualified residual-effect handling, replay crossing implication, and optional support geometry | Preserve the distinction between a conflict being redundant and its access disappearing from the interface. |
| [`CompletionStorage.cpp`](../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionStorage.cpp) | Coarse storage classes from spaces and proved local reservations | Reuse its wildcard treatment; integrate common domain facts without requiring fine geometry. |
| [`ResultGuarantees`](../../include/PTO/Transforms/FrontierSynch/ResultGuarantees.h), [`BoundingRegionalAnalysis`](../../include/PTO/Transforms/FrontierSynch/Shared/BoundingRegionalAnalysis.h) | Selected/input-order guarantees, reduction quality, realization guarantees and captured access contexts | Extend reporting here, preserving the meanings of existing fields. |

### Findings that affect the implementation plan

**Conservative completion already exists.** `CompletionStorage` merges proved
local reservations, maps unresolved reservations to all classes of their known
space, and maps unknown-space effects to all classes. GM is not treated as an
absolute local reservation. `CompletionRegion::buildCoarse` deliberately avoids
building fine storage geometry. The project must not turn this fast fallback
into a client of expensive exact access recovery.

**The finite analyses already handle residual conflicts.** `ExplicitAnalysis.cpp`
and `FiniteGuardedAnalysis.cpp` add unresolved relationships through shared
`residualConflict` queries. Numerical and arithmetic paths consume
`uniformConflict` information. Their semantics should be made consistent and
observable, not replaced by a second all-pairs pass.

**The selector defect has already been repaired.**
`regionalStorageSelectors()` treats provider callbacks as authoritative,
including an unavailable answer. Cell reconstruction requires coverage evidence.
`representedByCells` refers to exported boundary coverage, not just materialized
input geometry. Per-domain support permits useful local/output queries when
unrelated input domains remain unavailable.

**Atomic modes and atomic footprints are separate.** In the inspected
`TStoreOp::getEffects` fallback, AtomicAdd declares a destination read as well as
the store's effects. The reported Qwen gap therefore must be traced through
footprint construction and translation as well as mode declarations. Do not
describe it as simply “atomic stores have no read effect.”

**The input has two instruction views.** Translator stages and per-pipe macro
envelopes have different contracts. An envelope is not a sequence of native
payload stages. Access consolidation must preserve those distinctions.

## 3. Contract to establish

### 3.1 Storage identity and geometry

For an effect at an operation, retain:

- Its physical memory space, or explicit uncertainty about that space.
- Its canonical base/provenance where established.
- Its footprint representation and whether it is exact relative to the supplied
  operation contract or an enclosure.
- The original loop/control context in which that representation is valid.
- The assumptions used to establish it, including the alias policy.

Use existing `SyncStorageCell`, `SyncAccessRegion`, and `SyncSlotSelection`.
Do not replace a symbolic base-plus-offset map with an unknown access merely
because its runtime absolute address is unavailable.

Two restrictions matter:

1. Local allocation SSA identity does not distinguish storage after memory
   planning reuses addresses.
2. Distinct GM bases establish disjointness only under a valid proof or the
   explicit `may-not-alias` contract. Under `may-alias`, offsets relative to
   different unknown bases cannot simply be compared as absolute offsets.

A base or membership fact valid inside one traversal must not be promoted to
ownership across an enclosing reset. Repeated use of the same argument remains
potentially overlapping under either alias policy.

### 3.2 Access precision and fallback

Give each effect an unambiguous effective interpretation:

| State | Interpretation |
|---|---|
| Qualified declared selection | The supplied contract identifies the selected bytes, possibly symbolically. This is not independent proof of the native footprint. |
| Qualified enclosure | Actual accesses are contained in the described region or union. |
| Whole known storage domain | The access may touch any byte in that justified domain; no finite byte enumeration is required. |
| Unsupported operation semantics | A valid enclosing memory description or required execution semantics have not been established; diagnose explicitly. |

Keep declaration provenance and the reason for precision loss. Existing
`SyncAccessDescription` and `SyncAccessFallback` cover much of this information.
First add derived accessors and invariants over the existing fields. Add a
persisted field only if the required distinction cannot be derived reliably.
Do not introduce a second mutable copy of the access record.

The enclosing-domain hierarchy is:

```text
qualified symbolic selection
    → qualified allocation/view enclosure or union of alternatives
    → whole justified storage domain
    → all potentially accessible spaces, if the space itself is unknown
```

Each widening step needs a containment argument. A narrow view's valid shape is
not automatically an enclosure for an instruction that accesses padding.
Likewise, an initial loop-carried pointer is not an enclosure for its later
values. If neither bound is established, widen further.

Preserve read/write modes. The current representation may continue to use
separate read and write records for an RMW; no new mode enum is required.
Descriptor-state effects remain distinct from payload byte accesses, while
their scalar/control dependencies remain enforced.

An enclosure is a **may-access** fact. It does not prove that an operation
overwrites every byte, initializes the whole buffer, or kills earlier pending
facts. A reducer may compute the order of a deliberately conservative access
model, but it must enforce the added dependencies that make that model sound.

### 3.3 Query semantics

Keep existing result shapes where they already express the contract:

- `mayOverlap == false` requires evidence of disjointness; `true` permits
  overlap and is not a witness that overlap occurs.
- `mayConflict` also accounts for access modes; read/read needs no storage edge.
- `residualConflict` identifies possible conflicts not accounted for by the
  comparable materialized partition.
- `uniformConflict` supplies the admitted occurrence-uniform relationship.
  A negative answer must not be generalized into “no conflict”: another
  occurrence-dependent construction may still be needed.
- An exact empty access set is different from failed materialization.
- A storage selector's `nullopt` means unavailable. An empty selector answer
  means absence in its named model/context.

Do not turn pairwise may-alias relations into exclusive “alias classes” without
a construction that preserves all possible overlaps. Conservative transitive
merging is allowed but may lose precision; pairwise non-overlap is not a license
to omit a third buffer that overlaps both.

### 3.4 Four independent dimensions of a result

The report and interface must distinguish:

1. **Access precision:** exact supplied selection or conservative enclosure.
2. **Analysis guarantee:** covers/generators and equivalence/coverage relative
   to the selected modeled order.
3. **Export coverage:** which domains and queries a regional result can answer.
4. **Realization/resource guarantee:** what emitted code and allocation preserve.

Do not overload `exactSelectors`, `rangesMaterialized`, or successful allocation
to stand for all four. A conservative selected model can have complete selectors
for that model; an exact access model can have an unavailable selector export.

Keep the existing meaning of `ResultGuarantees::modeledOrder`: it is relative to
the unchanged modeled input. Any new precision report must identify its
relationship to supplied contracts without silently reinterpreting that field.

## 4. Integration into the existing pipeline

```text
PTO effect declarations + translated phases + existing buffer/view metadata
                              |
                 SyncInput and shared domain facts
                    /                         \
      fine symbolic access qualification      coarse space/reservation facts
                    |                         |
       SyncStorageEffects queries        CompletionStorage
                    |                         |
      finite / numerical / arithmetic     CompletionTransfer
                    |
       regional storage and query exports
                    |
       existing request-based composition and realization
                    |
       ResultGuarantees and existing physical allocation
```

This diagram is an ownership/consumer map, not a request to evaluate every route.
The existing dispatcher still stops at a complete acceptable result.

### Fine analyses

Route shared interpretation through `SyncStorageEffects`. Preserve specialized
mathematical kernels, hardware-protection checks, and source prerequisites.
Inventory direct `BaseMemInfo`, `MemAlias`, and raw-range consumers before
redirecting them: some are legacy insertion/resource clients with valid
contracts, not duplicate analysis code safe to delete.

A route requiring exact physical cells must refuse that particular premise when
only an enclosure or unresolved relationship is available. A route supporting
the selected conservative model may proceed with the corresponding guarantee.
Preserve the existing request quality policy; do not silently satisfy an exact
request with a weaker result.

### Regional composition

Retain `accessBoundary`, `deferredAccessBoundary`, domain support, and original
effect identities even when an effect contributes no internal demand.

Use `regionalStorageSelectors()` as the shared query boundary. Reuse
`selectorDomainSupportRanges()` and producer-certified uniform cuts rather than
sampling a byte and declaring a whole interval uniform.

The replay implication construction may prove that residual crossings are
already covered. That is a scoped order certificate, not proof of empty
footprints or a complete selector interface for the next parent.

### Conservative completion

Share cheap domain/provenance interpretation with the fine path, but keep
`SyncInputDetail::CoarseCompletion` independent of fine materialization,
arithmetic projection, and selector construction.

Do not require fine and coarse paths to produce identical partitions. Require
their descriptions to be sound and their differences in precision to be
explicit. Refining coarse GM classes by proven argument disjointness is optional
future work, justified only by measured excess in the experiment matrix.

### Ownership and caches

Keep access facts tied to unchanged IR and their owning `SyncInput`. A changed
alias policy, changed access abstraction, or changed source IR requires a new
qualified context and cache identity. Never mutate the model underneath retained
exact mathematics.

Reuse expression-owner/access-model/alias checks already present in regional
certificates. Coordinate adapters must translate presence and original loop
coordinates, and clear certificates they cannot transport.

Keep immutable analysis recipes separate from mutable prepared plans. The
project should not reopen the branch's completed plan-ownership consolidation.

## 5. Implementation sequence

Implement these as reviewable changes. Run each stage's focused gates before
starting a dependent migration. The experiment agent can finish its baseline
while stages A and the contract documentation are prepared; coordinate writes
to shared files before stage B.

### A. Freeze the baseline and inventory consumers

**Files:** shared access headers/implementations, finite/numerical/arithmetic
consumers, `CompletionStorage`, regional export helpers, existing test tools.

1. Record current source hashes and whole-kernel receipts under both alias modes.
2. Enumerate consumers of `rangesMaterialized`, `description`, `fallback`,
   `sharedProvenanceComplete`, `representedByCells`, and direct alias queries.
3. For each consumer, record whether it needs a footprint, an enclosure, a
   non-overlap proof, a boundary selector, or only a coarse storage class.
4. Identify conflicting interpretations and demonstrated precision loss.
5. Document the contract in the existing headers before changing behavior.

**Acceptance:** a caller inventory and a small set of concrete mismatches.
No timing or semantic change is attributed to this documentation stage.

### B. Establish one effective-access interpretation

**Primary files:** `SyncStorageEffects.h/.cpp`, `SyncEffectRanges.cpp`,
`SyncAccessRegion.h/.cpp`; operation contracts only where required.

1. Add derived queries or minimal typed metadata for effective domain, precision,
   and qualification reason.
2. Validate combinations: known-empty versus unresolved, symbolic versus
   materialized, qualified enclosure versus unknown provenance.
3. Centralize selection-to-enclosure fallback. Preserve original declarations,
   source operands, and useful descriptor/slot information.
4. Keep failures of one selection from hiding an unresolved effect of the same
   operation.
5. Add a deterministic test report keyed by original operation/effect identity;
   diagnostics must not construct additional mathematical exports.

**Acceptance:** exact cases retain their existing interpretation; unresolved
offsets receive a justified enclosure or explicit domain uncertainty. No
unknown state becomes empty, and no enclosure becomes a full-write certificate.

### C. Migrate dependence consumers and completion domain handling

**Primary files:** `MemoryDependentAnalyzer.cpp`, `SyncStorageEffects.cpp`,
`Finite/ExplicitAnalysis.cpp`, `Finite/FiniteGuardedAnalysis.cpp`,
`Structured/NumericTemplateStorage.cpp`, `Arithmetic/ArithmeticAccesses.cpp`,
`Shared/CompletionStorage.cpp`.

1. Migrate one consumer family at a time to the shared interpretation.
2. Retain existing residual and uniform-conflict mechanisms.
3. Preserve the fast path for fully comparable materialized effects.
4. Factor only genuinely shared space/reservation classification into a cheap
   helper; do not build fine inputs from the coarse path.
5. Keep instruction protection, scalar readiness, and supplied prerequisite
   handling outside generic alias reasoning.
6. Remove superseded per-consumer fallback logic once all its supported clients
   have migrated and the corresponding tests pass.

**Acceptance:** modeled required-order coverage is preserved under precision
loss; no extra producer construction occurs on the coarse-only path. Known
exact cases do not gain conflicts merely because their consumer was migrated.

### D. Integrate precision with regional exports and guarantees

**Primary files:** `RegionalAnalysis.h`, `RegionalSelectorPartition.cpp`,
finite providers, `SequenceAdapters.cpp`, `SequenceStorage.cpp`,
`RepeatedPhases.cpp`, `RepeatedRegionExports.cpp`, `ResultGuarantees.cpp`.

1. Preserve the existing domain-sensitive coverage certificates and authoritative
   callback behavior.
2. Identify the access model against which selectors are exact. Extend metadata
   only if the owner/context currently cannot express it.
3. Preserve useful local/output exports when a different domain is unavailable.
4. Keep conservative class summaries separate from exact byte selectors.
5. Aggregate precision reasons into existing result/ledger reporting without
   recomputing analyses.
6. Test owner, alias-policy, occurrence-frame, and guard changes explicitly.

**Acceptance:** attention-finalize's real readers/writer and absence probes stay
correct; replay preserves its crossing proof and unavailable reader exports;
missing GM queries never become empty. No readiness flag promotes a guarantee.

### E. Close one demonstrated access-recovery gap

Select this stage from the whole-kernel evidence, not from a general desire to
support more operations. Atomic output footprints are a current candidate.

**Likely files:** `PTOInterfaces/PTOSubViewAndCoreEffects.cpp`, relevant store
access helpers, translator effect mapping, and shared access recovery.

1. Trace one operation from declared modes through translated phases to its
   selected geometry and fallback.
2. Determine whether information exists but is lost, the shape contract needs
   a narrow extension, or only a sound enclosure is justified.
3. Implement the smallest supported correction and document its scope.
4. Preserve all modes across specialized helper early returns.
5. Re-run that kernel under both alias modes.

**Acceptance:** the reported access issue is resolved or accurately bounded.
Do not claim the Qwen root is exact merely because atomic footprint handling
improves; symbolic parent and executable interface obligations remain separate.

### F. Whole-kernel validation and deletion review

Use the experiment campaign already dispatched; do not create another driver.

1. Re-run the seven-kernel matrix and the captured indirect candidates under
   both policies.
2. Compare original baseline and migrated results, including modeled precision,
   coverage, exactness, excess prerequisites, retained size, and refusal reason.
3. Check that default request ordering and allocation policy are unchanged.
4. Delete confirmed obsolete adapters and document remaining compatibility
   consumers with their actual callers.
5. Update this plan's status with the accepted changes and reproducible receipt.

**Acceptance:** every attempted input has an explained outcome; no regression is
hidden as an unavailable field, timeout, or new assumption.

## 6. Verification plan

### Small semantic tests

| Case | Required check |
|---|---|
| Two overlapping views of one buffer | View identities do not hide overlap; disjoint subviews retain precision. |
| Different local allocation roots reusing bytes | Physical overlap is retained after memory planning. |
| Unknown offset into a bounded buffer | All possible concrete conflicts are covered by the enclosure. |
| Unresolved loop-carried pointer | Initial-root geometry is not used as a universal bound. |
| Different GM roots | Policy-dependent disjointness is recorded; same-root reuse remains checked. |
| Unknown read-only accesses | No read/read conflict; possible writers still interact with them. |
| Atomic update | Read and write modes survive import and conservative fallback. |
| Empty or skipped access | Proven emptiness is distinct from unsupported geometry or missing guards. |
| Partial write/padding | Allocation enclosure is not treated as initialization or definite overwrite. |
| Strided support with holes | Envelopes do not imply actual membership in holes. |
| Inner traversal followed by reset | Visit ownership does not escape its certified scope. |
| Missing boundary selector | Unavailable, not empty; unrelated valid domain queries remain usable. |
| Coarse-only request | No fine partition, arithmetic projection, or selector construction is invoked. |

For finite independent checks, let `H_exact` be the closure from concrete modeled
accesses and `H_coarse` the closure from an enlarged access description, under the
same reference/control and native semantics. Require:

```text
H_exact ⊆ H_coarse
```

Test closure inclusion, not inclusion of cover sets: adding order can make an
old cover redundant. Likewise, switching dispatcher routes need not produce
comparable emitted plans. Evaluate emitted coverage and excess separately.

Under `may-not-alias`, use executions satisfying its disjoint-base contract.
Use overlapping-argument tests only under a policy and kernel ABI that admit
them. A policy violation is not a synchronization regression.

### Existing regression assets to extend

- `sync_access_regions.pto`
- `sync_storage_effects_gm_alias.pto`
- `sync_storage_effects_reuse.pto`, `sync_storage_effects_views.pto`
- `sync_storage_bound_symbols.pto`
- `sync_descriptor_state.pto`, `sync_descriptor_gm_provenance.pto`
- `sync_completion_inputs.pto`, `sync_completion_scoped.pto`
- `sync_storage_export_coverage.pto`
- `sync_pypto_region_survey.pto`
- `sync_compact_persistent_contract.pto`, `sync_compact_replay_scaling.pto`

Reuse the independent access/closure/cover oracles and the shared production
query boundary. Production helper agreement is an integration check, not an
independent proof of extraction.

The current experiment work includes `assess_whole_kernel_access.py`,
`check_whole_kernel_access.py`, `check_whole_kernel_alias.py`, and
`capture_indirect_candidates.py`. These were present as uncommitted work during
inspection; coordinate with their owner and consume the finished receipts.
Their presence is not evidence that the campaign has passed.

### Cost and scaling gates

- Preserve compact K/replay behavior across the existing trip-count sweeps.
- Attribute cells, boundary atoms, symbolic nodes, and residual pair work
  separately; do not call a small demand graph a small storage interface.
- Preserve `uniformConflictWork` and existing stage-ledger counters.
- No mandatory whole-program pair enumeration to produce a diagnostic.
- No slot-count or byte-count expansion where the current symbolic path avoids it.
- Capture cold construction and repeated-request behavior separately.
- Use structural regressions rather than fragile wall-time thresholds.

## 7. MLIR reuse and the buffer-IR decision

The inspected branch already uses MLIR affine expressions, view interfaces,
memory-effect interfaces, and bufferization external models. The local LLVM 19
source provides `AliasAnalysis`, `memref.subview`, `memref.reinterpret_cast`,
and `memref.extract_strided_metadata`.

Use these facilities at existing boundaries:

- **View/descriptor import:** reuse metadata and view relationships where PTO
  types expose them; preserve target-specific layout maps.
- **Effects:** attach access selections to existing operation effect declarations.
- **Alias queries:** integrate a PTO-aware implementation or shared helper where
  needed; generic MLIR alias results do not automatically understand planned
  physical local-buffer reuse or the selected GM assumption.
- **Bufferization:** retain it as the tensor-to-buffer transformation layer.
  It does not supply asynchronous completion or consuming-ID semantics.

References: [MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/),
[AliasAnalysis](https://mlir.llvm.org/doxygen/classmlir_1_1AliasAnalysis.html),
[memory effects](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/),
and [bufferization](https://mlir.llvm.org/docs/Bufferization/).
Check APIs against the pinned LLVM checkout before implementation; current
online documentation may describe newer interfaces.

A larger IR change is justified only if the inventory demonstrates that a
necessary fact cannot be represented or preserved, for example:

- An operation loses its base/view relationship before synchronization.
- Descriptor mutation cannot be reconstructed soundly at an access.
- Physical layout or reuse information has no stable producer-owned contract.

In that case propose the smallest producer-side interface or attribute change
first, with ODS, verifier, lowering, bindings, and tests updated together where
affected. A wholesale MemRef conversion remains a separate project.

## 8. Execution, review, and completion criteria

This work shares a machine with the experiment agent. The aggregate limit is
two resource-intensive workers across all agents. Coordinate builds/tests; run
the narrowest relevant tests serially unless capacity is explicitly free.
Keep substantial artifacts on disk under the workspace.

Use the source/tool pins and environment established by the active checkpoint.
Inspection found that the worktree's `env.sh` contains a foreign-machine LLVM
path, while the checkpoint build cache names the local LLVM 19 tree. Do not
source that file blindly or reconfigure a working build. Resolve the intended
environment with the experiment owner before running commands.

Each implementation change should include:

1. The contract being changed and its concrete caller.
2. A semantic regression and expected guarantee.
3. Relevant source/tool hashes and executed commands.
4. Any cost or representation change.
5. The obsolete interpretation removed, or a named compatibility caller that
   still requires it.

Follow existing PTOAS secure-coding, C++17, header, build and changed-code
review requirements. This proposal itself changes only documentation.

### Definition of done

- Every admitted memory effect has an explicit modeled interpretation or a
  diagnosed unsupported semantic obligation.
- Fine and coarse consumers preserve the same underlying storage/provenance
  facts while stating their different precision.
- Unknown geometry never becomes empty or a non-alias proof.
- Missing export capability never destroys retained exact mathematics.
- Alias/context changes cannot reuse stale certificates.
- The production experiment matrix explains both successes and refusals.
- Known exact cases preserve their guarantees; conservative cases cover the
  reference requirements under their stated contracts.
- Superseded interpretations are removed when their callers migrate.

Whole-kernel exactness, native/device correctness, a minimum physical ID budget,
and a universal buffer analysis are not completion criteria for this project.
The deliverable is a consistent, auditable interface that lets the existing
algorithms use available precision and fall back soundly.

## 9. Evidence and source review record

The design is grounded in:

- [Compact demand repair](frontier-compact-demand-repair-20261019.md).
- [Expert and replay diagnosis](frontier-expert-replay-checkpoint-20261019.md).
- [PyPTO regional survey](frontier-pypto-region-survey-20261019.md).
- [Storage export and replay repair](frontier-storage-export-replay-20261019.md).
- [Existing family-pipeline plan](frontier-family-pipeline-plan.md), whose
  request, ownership, and allocation boundaries remain in force.

The source inspection covered the components and selected call sites listed
above. It was not a whole-program dead-code audit or independent proof audit.
No compiler build, experiment, or device run was performed while writing this
proposal.

Selected reviewed-file fingerprints follow. They identify the working-tree
snapshot used for the design, not a complete reproducible compiler build.

| File | SHA-256 |
|---|---|
| `include/PTO/Transforms/InsertSync/SyncStorageEffects.h` | `234c1ebd8483b456b232727053207fc6b0ab6445686fdc5af79916be0d453fe8` |
| `lib/PTO/Transforms/InsertSync/SyncInput.cpp` | `25b7affd004fc41d4d697db1747916740c602d2dc8c112427a5e6976ef5618dc` |
| `lib/PTO/Transforms/InsertSync/SyncEffectRanges.cpp` | `114d3bdcb5742ad745145cffca11a4c994376bd160578d0bca3fd0943499f2b9` |
| `lib/PTO/Transforms/InsertSync/SyncStorageEffects.cpp` | `8cc05e3057c1f035fef8d429658016bdc01ee6a356955935c47a6876626b36b0` |
| `include/PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h` | `3697f1c069efb4b3e83d407bb81ef8f3cdd0f6234475aa4e4ccde76fded265c5` |
| `lib/PTO/Transforms/FrontierSynch/Shared/RegionalSelectorPartition.cpp` | `7dd1c8c70c6a2f5dd795830f38eae3651c7968ae9c27071360e797e94200dc0d` |
| `lib/PTO/Transforms/FrontierSynch/Shared/CompletionStorage.cpp` | `de26a841a98588fa4347dcf323e980ded45cb477d98f31927c2852df6f77ab0e` |
| `lib/PTO/Transforms/FrontierSynch/Shared/RepeatedReadOnlyStorage.cpp` | `c07d01eb2caf5bbb219ea8fd9e6fdc43b7b408c9fcbea8706d578aba6f3e4fbc` |
| `lib/PTO/IR/PTOInterfaces/PTOSubViewAndCoreEffects.cpp` | `b399b0c4471306915ac0b8db158798c065d80eaa9df5542796e1e99adef61a5e` |
| `include/PTO/Transforms/FrontierSynch/ResultGuarantees.h` | `c8ad8fd579964ea5a3f9e0a854aee3453f23599b78d12f061bd37f587532ba6e` |
