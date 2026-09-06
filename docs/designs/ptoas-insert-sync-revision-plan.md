# InsertSync revision: correctness and pipeline parallelism first

Revision date: 2026-09-06. Implementation base: `tonibohnlein/PTOAS` at
`687ac2d876dafdf23a52af0ca360f272fe445522` (`codex/protocol-sync`).

## 1. Decision and precedence

Evolve the existing InsertSync production path. Retain its translator, broad
operation integration, structured traversal, dynamic-event lowering and codegen.
Do not replace it with the narrow ProtocolSync admission gate. General
correctness and pipeline-parallelism improvements are the first push; new
specialized pattern recognizers are the second push.

This deliberately supersedes the earlier ProtocolSync proposal's decision not
to evolve `InsertSyncAnalysis`. It does not supersede its useful distinctions
between effect completeness, generation identity, target legality, event lifetime,
independent verification and honest evidence. No old design document is silently
rewritten by this package.

The objective is to preserve the overlap permitted by required effects and
qualified target mechanisms, with the computation schedule and storage placement
fixed. Fewer signals are not an improvement when they introduce avoidable
blocking. “Maximum” is a design objective and an ordering contract, not a claim
that a globally optimal runtime schedule has been proved. Some target-level cuts
have incomparable effects; record those choices rather than hide them in counts.

### Non-negotiable separation

1. Required ordering: actual dataflow, physical reuse, descriptor/scalar
   prerequisites and target-qualified special effects.
2. Unlimited logical handoffs: distinct symbolic identities for dynamic
   occurrences, with correct participation. No hardware-key recycling constraint
   is allowed to invent a data dependency at this stage.
3. Finite realization: use the existing required ordering to establish safe key
   reuse; retain the baseline's publication/acquisition boundaries.
4. Scarcity recovery: explicitly add local backpressure/coalescing/serialization
   only when finite realization cannot be established under the supported model.
   An analysis limit is not a physical-scarcity proof. A zero-size usable event
   pool is not evidence that a serializing event recipe can help.

Actual buffer reuse remains required with infinitely many event IDs. Unlimited
IDs do not mean unlimited storage or permission to overwrite outstanding reads.

## 2. What is delivered now

This package starts implementation; it is not the whole revised algorithm.

| Item | Status |
| --- | --- |
| Remove blanket TLoad/TLoad WAW exemption | IMPLEMENTED |
| Checked physical-range arithmetic in existing overlap paths | IMPLEMENTED |
| Cross-pipe-first, same-pipe-second analysis inside existing InsertSync | IMPLEMENTED, OPT-IN |
| Header-only primitives compiled in local C++ tests | TESTED |
| Actual new `Run` body compiled with stubbed IR callbacks | TESTED; not an MLIR build |
| Native lit fixtures and native runner | INCLUDED, NOT_RUN |
| Scope/guard-aware completion certificates | PLANNED |
| First/last-use branch/loop motion replacement | PLANNED |
| Symbolic nested-slot generalization | PLANNED |
| Unlimited occurrence families separated from recurring realization | PLANNED |
| Proven finite-event allocator and new scarcity recovery | PLANNED |
| New pipeline pattern recognition | SECOND PUSH |

The new CLI flag is `--insert-sync-defer-same-pipe`, default false. Applying the
package activates the correctness fixes. It does not change the traversal policy
unless the new flag is passed. The full native suite and matched-corpus gate are
required before changing the staging default.

### Existing files changed

- `lib/PTO/Transforms/InsertSync/InsertSyncAnalysis.cpp`: remove the operation-name
  WAW exemption; retain the ordinary def/def alias query; optionally traverse
  twice, filtering dependency insertion by source/target pipe equality.
- `include/PTO/Transforms/InsertSync/InsertSyncAnalysis.h`: explicit stage state
  and a defaulted second argument to `Run`; existing calls retain combined order.
- `lib/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.cpp`: use checked
  base-plus-offset and conservative range overlap on nonrepresentable endpoints.
- `lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp`: register the experimental
  option in the existing linked pass translation unit and pass it to analysis.
- New `SyncPlanningPrimitives.h`: small C++17 utilities, no LLVM/MLIR dependency.

Neither computation nor allocation addresses are moved. `MoveSyncState`,
`RemoveRedundantSync`, dynamic-slot handling, explicit-sync skipping and the old
allocator remain in place in this first slice. Therefore this slice does not
claim to have repaired their previously identified limitations.

### Why these initial changes

The baseline's `isTLoadToTLoadWAWExempt` bypasses def/def checks based only on
operation names and MTE2 membership. Released Ascend documentation requires
ordering overlapping DataCopy destinations. Restoring the existing alias query
is not a blanket barrier-after-load policy: disjoint ranges still have no WAW,
and established handoffs can still discharge a dependency.

The current physical overlap helpers compute `start + size` and `root + offset`
without checking overflow. A wrapped end must not turn uncertain overlap into
false disjointness. `BaseMemInfo::allocateSize == 0` is treated as unknown extent,
not mathematical emptiness. This patch does not establish range validity for
an otherwise invalid address and does not fix all provenance handling.

For staging, the existing backward supply traversal can use a required reverse
handoff to establish a same-pipe hazard. If a barrier is inserted before that
handoff exists, it may drain unrelated work. Construct cross-pipe candidates
first; on the second walk reuse those candidates before inserting remaining
same-pipe cuts. Keep indices and sync IDs across the walks. Insert the function
exit drain once, after both walks, so it cannot mask a body hazard. BLOCKSYNC
keeps its original traversal even when the option is enabled.

The staged pass is not guaranteed to emit fewer events. Removing barriers as a
source of provisional coverage can expose additional necessary handoffs. The
native gate must inspect ordering, resources and actual kernel behavior.

## 3. Architecture to evolve toward

Keep a source-linked structured SyncIR; initially add read-only side records
rather than rebuilding every node. The final conceptual flow is:

```
PTOIRTranslator and existing macro/helper integration
  -> structured schedule + shared facts
  -> complete sparse requirements and storage transitions
  -> forward target-driven synthesis with backward frontier queries
  -> unlimited logical plan
  -> non-serializing finite realization
  -> explicitly attributed scarcity recovery, if needed
  -> SyncCodegen and independent emitted-IR checks
```

Facts and plan must be separate even if old before/after lists temporarily remain
as a compatibility output representation. Selected events must not change what
memory effects the compiler believes exist.

### Minimal shared records

```
PhysicalRegion:
    resource owner, address space, root provenance
    physical byte set / conservative upper bound / unknown
    slot set, actual selector, descriptor version

EffectOccurrence:
    physical phase, effect role, region
    guard, symbolic iteration tuple

Requirement:
    required property, source effect expression, target point
    participation and occurrence relation, witness

CompletionGuarantee:
    source completion frontier, destination lane, acquisition point
    guard, occurrence relation, target-contract witness

RegionTransfer:
    incoming/outgoing generation alternatives
    outstanding writers and readers by storage
    first-use/final-use publication opportunities
    descriptor and control forwarding

LogicalHandoff:
    immutable required relation
    publication/acquisition, participation, lifetime scope
    occurrence-indexed logical identity; no physical ID
```

Do not store one universal `done(region)` token. For a panel-consuming inner loop,
its panel-read completion and outstanding output stores are different facts.
Do not replace `alreadySync[P]` with a giant unshared history. Retain compact
prefix certificates where valid, shared guarded/loop nodes elsewhere, and cache
queries. A lane index is a projection of structured control, not a flattened
assertion that every phase executes once.

## 4. First-push work packages

Each work package names original real kernels and all their blockers. Reduced
fixtures accompany those kernels. A work package is complete only when the
unchanged original input runs through the revised native pipeline, not when one
small fixture is admitted. Prerequisite commits need not individually increase
admission, but a completed milestone must deliver an end-to-end result.

### R0 — freeze and establish the baseline

- Freeze compiler commit, LLVM build/version, target, frontend revisions, input
  hashes, alias/ABI contract and all commands. Use the same population throughout.
- Retain the driver population, first-per-seed sample and full corpus separately.
  The prior 14/213 ProtocolSync result is not an InsertSync baseline measurement.
- Record whether InsertSync actually ran or skipped because explicit sync existed.
  Record pre-pass failures without removing them from the denominator.
- Add stage traces for insertion, motion, removal and allocation. Existing
  `InsertSyncDebug` dumps are the starting point, not a new logging subsystem.
- Keep a pristine baseline executable and build the candidate separately. A test
  that accidentally invokes the stock compiler establishes nothing about the patch.

Acceptance: native compile/C++ results, correctness evidence and performance are
separate columns. Same external GM contracts in both arms; the new `pto.gm_alias`
attribute must not be assumed to control legacy alias analysis where it does not.

### R1 — repair demonstrated dependency omissions (started here)

Audit memory prerequisites before optimizing them away:
- RAW/WAR/WAW, including physical overlap of different allocation handles.
- TLoad/TLoad WAW; physically overlapping non-DPS scratch; same-static-operation
  recurrences; target-specific ACC/proxy relations.
- Scalar prerequisites ending at descriptors or address-selection actions.
- Bounds/overflow and conservative-versus-exact access precision.
- GM alias policy and completion-versus-publication qualifications.

Do not change all target exceptions indiscriminately. Each exemption needs a
source/lowering argument and adversarial tests. A confirmed omitted dependency
must be fixed even when the safe output has more synchronization. An unknown new
verifier result is not a confirmed defect and starts as a classified diagnostic.

Acceptance: original reproducer + reduced test, matching non-sync IR, safe output,
no lost independent-readiness frontier. Full `check-pto` is still required.

### R2 — phased general insertion (started here)

Use the existing forward iteration and backward `InsertSeqSync` queries. The
first opt-in step establishes cross-pipeline supply before same-pipe repair.
Next attach explicit source/target witnesses to inserted/omitted actions so the
reason for omission can be reconstructed independently.

Preserve the current fast path for a genuinely completed static prefix. Extend
its scope to guard and occurrence identity; a dynamic-slot acquisition cannot
be promoted to unconditional pipe-wide completion. Do not accumulate all prior
operations into each certificate.

Acceptance: native L1/L2/C1/C2 keeps separate publication; independent load and
consumer overlap in the reference execution model; mandatory same-pipe WAW is
still present when no handoff proves it; recurring and same-iteration identities
are not confused. Static action counts alone do not decide the result.

### R3 — replace unsafe/coarse motion incrementally

Targets: `MoveSyncState`, `SyncOperation` placement information and `SyncCodegen`.
Keep original logical endpoints separate from transformed codegen anchors.
Replace whole-pair constructions, not isolated set or wait edits:

- Prefix -> body: publish immediately after the required producer, under proved
  participation, acquire at first executing relevant consumer.
- Body -> suffix: publish after the last relevant source occurrence; acquire
  immediately before the suffix access that actually needs it.
- Zero trip: preserve true prefix/suffix hazards; no wait for absent body supply.
- Branch-selected producer: guard alternatives or prove exactly-one publication.
- Optional reader: retain conditional outstanding-reader state until real reuse.
- Cleanup: manage token lifetime separately; do not delay unrelated suffix work.

Every moved action needs SSA-operand availability, dominance, matching dynamic
predicate and placement legality. A condition computed after a producer cannot
be used by an early guarded signal without another certified construction.
Leaving a repeated wait behind a once-only signal is not a valid replacement
for blanket motion. Simply disabling MoveSyncState is NOT this work package.

Initially support a bounded construction but retain untouched legacy behavior
outside its proved domain during rollout. Report which domains received the new
construction; do not call the whole function newly verified when it did not.
A confirmed unsafe legacy path cannot silently remain a correctness fallback.

Acceptance: slow A / ready B / zero-trip A consumer does not block B's suffix;
first-use and last-use positions survive independent work; path participation
and progress are checked in addition to memory coverage.

### R4 — common provenance and descriptor/slot queries

Reuse the existing translator and ProtocolSync's qualified analysis utilities
through a shared query API. Do not switch all input admission to a new extractor
before its effects have been compared on the corpus.

- Keep logical handles, physical storage and content generations distinct.
- Preserve absolute versus root-relative addresses; separate physical owner
  identity from merely Vector/Cube kind when sections/queues are involved.
- Use tile layout and retained view strides; an allocation bound is not an exact
  instruction footprint. Disjoint conservative upper bounds can prove disjointness.
- Trace multi_tile_get's actual SSA selector, forwarded values and planned slot
  addresses. Do not infer capacity from the number of load operations.
- Track descriptor identity/version separately from payload effects. Symbolic
  scalar values need not have invented bounds; physical scalar readiness must
  terminate before the metadata consumer action.
- Preserve fixed macro/queue effects and hidden reservations. Unknown effectful
  operations do not become pure because they are inconvenient.

Acceptance: parity/reporting against translator first; refined decisions then
land independently, tied to original kernel blockers. No full-corpus admission
regression from an unconditional preparatory walk alone.

### R5 — compositional loop and choice transfers

The old copied-body construction may remain a supported implementation for its
proved subset. It is not a universal arbitrary-nesting proof. Evolve regions
into parameterized storage-specific transfers instead of adding a new
whole-function constructor per syntactic shape.

Use symbolic occurrence relations: same iteration, first/last relevant use,
carried instance tuple, bypass and next actual same-slot reuse. Conditional
refresh must retain a generation on skipped writes. Opposite branch arms in
different iterations are not mutually exclusive over an event lifetime.

A fixed-footprint recurring proof does not automatically support modulo slots.
For `j % 2` with inner lengths 3 and 2, the sequence is `0,1,0 | 0,1`.
The adjacent slot-0 use across the outer boundary must be protected. A persistent
slot protocol follows actual uses; its lifetime need not equal the innermost loop.

Acceptance fixtures include nested panel consumption, optional second reader,
conditional refresh, empty inner loops, odd/even ragged lengths, nonzero lower
bounds/nonunit steps, and loop-carried handles/selectors. General slots and
participation are correctness machinery; they do not wait for a new pattern
catalogue. Bounded unrolling is a differential oracle, not the production proof.

### R6 — finite realization and scarcity recovery

Split `SyncEventIdAllocation`'s proof, assignment and repair responsibilities.
Retain the unlimited logical plan for comparison. Ordinary allocation may not
move a signal/wait or reduce a true storage-capacity relation to a one-credit
cycle merely to get an ID.

- Use exact domain/core reservations and actual slot-event mapping.
- Prove consume-before-rearm from existing causal action paths, not lexical
  wait-before-set position. Logical handoffs have distinct dynamic identities.
- For an event ring, prove consumption of occurrence n before publication n+K
  using the participating occurrence relation, not an unrelated induction counter.
- Classify unsupported, analysis-limit, conservative interference, empty pool,
  realized assignment and supported pressure evidence separately.
- On real resource recovery, operate on the failed domain and smallest region:
  bounded producer lead, checked local coalescing, then a more serialized
  complete recipe. Recheck progress and memory after every change.
- Keep fixed exit drains and visibility operations separate from recovery counts.

A once-only seven-readiness burst may require backpressure with six IDs; it is
not evidence that every loop needs a serialized phase cycle. Never silently
convert an unimplemented selective/control analysis into a resource-fallback win.

Acceptance: original logical plan unchanged across allocation budgets; same-ID
collisions detected; unsound coalescing cycles rejected; added ordering and its
resource reason reported; unrelated channels unaffected.

## 5. Second push — new specialized protocol recognition

Start only after the first push delivers improvements on a representative set
of unchanged real kernels. “Second push” does not require every theoretical
control-flow form or every corpus row to have an optimal general solution.

Recognizers consume the shared facts and produce the same logical plan:
- regular slot-based input/output lifecycles;
- one generation consumed throughout an inner region;
- alternating prefetch with explicit initial generation and continuation;
- target-qualified operand/accumulator lifecycles.

Reuse native ReadyRelease construction; adapt old slot-bundle/hierarchical
recognizers from covering-performance and conditional completion merging from
hardware-graph. Do not import the global covering planner, whole-region anchoring
shortcuts, unvalidated target exemptions or normal latest-producer/earliest-
consumer sharing. Cross-pattern compatibility is about effects, participation
and resources, not simply disjoint operation sets.

Import patterns at the logical-plan level. Unchanged InsertSync currently skips
functions already containing explicit flags, so “emit one pattern then call
legacy InsertSync for the rest” is not a valid integration strategy.

## 6. Algorithm sketch after the first push

```
facts = analyze_shared_facts(existing_sync_ir)
requirements = derive_complete_effect_relations(facts)
plan = import_verified_fixed_supply(facts)

for target in structured_forward_order(existing_sync_ir):
    for missing_cross_lane_requirement at target:
        publication = backward_query_relevant_source_frontier(requirement)
        acquisition = latest_legal_acquisition(target, requirement)
        plan.add_logical_occurrence_handoff(publication, acquisition, relation)

for remaining_same_lane_requirement:
    if not established_by_existing_cross_lane_supply(requirement):
        plan.add_target_qualified_cut(requirement)

verify_memory_and_participation(plan)
remove_complete_redundant_handoffs_without_losing_required_supply(plan)
finite = realize_without_added_order(plan)
if finite is a supported resource failure:
    finite = recover_locally_and_record_added_order(plan)
verify_concrete_memory_tokens_progress_and_emit(finite)
```

Loops enter through their supported symbolic transfer rules, not by pretending
that the forward iteration visits every dynamic occurrence. Descriptor/scalar
and visibility properties remain typed requirements, not ordinary payload RAW
by default. The above functions describe future work; only R1 and the initial
R2 staging are implemented in this package.

## 7. Validation and rollout

Use sequential or at most two aggregate intensive workers. Rebuild every changed
header's dependent compiler/test consumer; stale test executables are not evidence.
Do not reset/stash user changes or push a branch as part of applying this package.

For each work package:
1. Original input hashes and documented contracts remain fixed.
2. Baseline/candidate compile, C++ emission, MLIR verification and skip reasons.
3. Safety and event-participation checks, with failures distinguished from unknown.
4. Explicit overlap witnesses and over-ordering mutations.
5. Native `check-pto`, plus the same frozen corpus populations.
6. Purpose-built device correctness tests on the exact candidate binary.
7. Device throughput measured separately from compile-inclusive wall time.

Per-row results: native success, skipped-explicit-sync, pre-pass failure,
unsupported semantics, proof/analysis limit, resource recovery, internal error.
Do not hide lost admissions in aggregate compiler timing. Keep confirmed invalid
inputs/failures in the ledger with their classification. A2 and A3 host success
is not A2 and A3 silicon qualification.

## 8. Sources and status boundaries

Source inspected at the pinned commit:
- `InsertSync/InsertSyncAnalysis.cpp`: current forward/backward scan, slot and
  WAW handling, `alreadySync` and `syncFinder`.
- `InsertSync/MemoryDependentAnalyzer.cpp`: physical/root-relative range handling.
- `InsertSync/MoveSyncState.cpp`: current loop/branch motion.
- `InsertSync/RemoveRedundantSync.cpp`: pipe-pair coverage and slot exclusions.
- `InsertSync/SyncEventIdAllocation.cpp`: current allocation/widening/fallback.
- `InsertSync/SyncCodegen.cpp`, `PTOInsertSync.cpp`, `SyncCommon.h`.
- ProtocolSync plan/amendment and the supplied c4f924 census. The old census is
  historical context, not a fresh measurement of either planner.

Released hardware references:
- CANN 8.5.0 DataCopy constraints:
  https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0103.html
- CANN 8.0.RC3 synchronization overview:
  https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/apiref/ascendcopapi/atlasascendc_api_07_0256.html

Development preview, not release qualification:
- https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.html
- https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.html

These establish the documented contracts used here, not successful execution of
this candidate. Same-address GM publication and topology-dependent cross-core
collectives retain their independent qualifications; event legality alone does
not establish either. No new numerical performance or full-corpus result is
claimed by this revision plan.
