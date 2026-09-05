# ProtocolSync obligation-engine amendment

## Status and precedence

This amendment records the design conclusion after the C.5 execution-lane,
C.6 lane-pattern, C.7 storage-track, Checkpoint E, and Checkpoint F
experiments. It revises the storage-generation and residual-obligation parts of
the original implementation plan and its first amendment. It does not make the
diagnostic C.6/C.7 candidates selectable.

The [C7.1 rebaseline](ptoas-protocol-sync-c71-rebaseline.md) now supplies the
394-row A2/A3-merged E/F rerun and the 199-program concrete audit, including a
clean-commit freeze at `1e911a4e0`; the separate 152-kernel
acceptance inputs remain unavailable. The user selected a fresh current-main
PyPTO/PyPTO-Lib acceptance population instead. Historical pending measurements below
should be read with that newer report. The user's agreed two-mode GM policy is
specified in the [GM alias contract](ptoas-protocol-sync-gm-alias-contract.md).

## Native-coverage revision: agreed implementation contract

Status: approved implementation direction, not a claim of completed native
coverage. This section revises the delivery sequence below.

The [local frontier baseline](ptoas-protocol-sync-local-frontier-baseline.md)
records the first production slice: conservative physical UB atoms, sparse
outstanding-access requirements, fixed-supply import, backward source-frontier
placement, and independent concrete all-pair checks. Its straight-line admission
boundary is explicit; it does not complete general generation or loop analysis.

Keep ProtocolSync's complete-world gate, allocator and cloned emission. Add a
general native baseline underneath optional lifecycle recognition:

```text
shared operation semantics and authoritative provenance
  -> physical atoms + structured memory/effect state
  -> canonical requirements + missing/invalid facts
  -> fixed supply + structured backward lane-frontier repair
  -> complete native world or atomic protocol-plus-residual alternative
  -> allocation + cloned emission + independent concrete verification
```

Lanes organize legal placement, not correctness requirements. Forward effect
analysis discovers requirements; a structured backward traversal, inspired by
InsertSync, proposes common completion frontiers. It retains path and loop
instances rather than flattening control into a DAG. Small certified handoff
recipes do not require the source loop to match a storage lifecycle pattern.

Separate actual requirements, missing/invalid semantic facts, and optional
recognition failures. Replace rejected-timeline synthetic residuals only when
canonical local RAW/WAR/WAW discovery replaces their conservative protection.
An in-place local access is not automatically an ACC conflict. Preserve
outstanding readers and writers as well as must/may reaching definitions;
logical generation replacement does not retire asynchronous effects.

The native planner checks existing transitive supply before inserting repair.
It groups obligations only when every claimed ID is covered by a legal,
target-qualified frontier. At choices, guaranteed synchronization coverage is
intersected across feasible arms while alternative memory states are retained.
Never wait unconditionally for a potentially absent signal. Loops need entry,
body, backedge and exit relations, including same-static-phase recurrences.
Capacity-one control handshakes may serialize execution but do not assert
capacity-one storage. Zero-trip and arbitrary-trip token invariants are required;
bounded unrolling is a test oracle, not the loop proof.

Direct-only constructs a complete baseline, including verified fixed supply.
Mixed selection preserves it when an optional protocol does not match or cannot
allocate. On the same semantic contract, direct-only admission must imply mixed
admission. Retain the deterministic structural cost comparison; it is not a
device-performance model. Do not add a generic body PIPE_ALL escape hatch.

Implementation gates, each with corresponding concrete-verifier support:

1. Commit diagnostics, rerun the clean 394-row C.6/C.7 and 199-program audits,
   and freeze a current-main frontend acceptance manifest with all collection
   failures and overlapping blockers.
2. Recover Exact/Conservative/Unknown access precision, descriptor shape state,
   physical-core and queue provenance, and both explicit GM alias contracts.
   Introduce canonical local effects and dual-run with F on the shared subset;
   classify synthetic-old and newly discovered requirements separately.
3. Deliver direct-only straight-line reuse, partial overlap and ordinary
   in-place local computation without lifecycle recognition. Import qualified
   existing events, barriers and macro supply before residual planning.
4. Deliver repeated same-iteration handoffs, carried requirements, choices,
   nested summaries and exits. Event reuse needs a target-consumption or
   happens-before proof; retain conservative interference otherwise.
5. Close GM visibility, communication, cache/fence, queue ownership and ACC/proxy
   blockers one qualified domain at a time. Existing certified supply is distinct
   from synthesizing arbitrary collectives. These rows remain in the goal.
6. Make OneShot and ReadyRelease certificates over canonical subsets, localize
   their shape restrictions, and allow unrelated work and compatible candidates.
   New stable/alternating L1 and accumulator lifecycle optimizations remain
   postponed; baseline ACC correctness does not.

The concrete verifier reconstructs outstanding effects and concrete event
generations without planner coverage lists or diagnostic tags. Report missing
semantics, invalid input, uncovered requirements with witnesses, event-lifetime
or deadlock failures, and analysis limits separately. Independent byte-set and
bounded path/iteration oracles and synchronization/address/guard/slot mutation
tests are acceptance gates, alongside host regressions and device qualification.

The delivery goal is native compilation of every valid collected current-main
A2/A3 acceptance kernel in both may-alias and assumed-disjoint-argument modes,
with legacy fallback disabled. Coverage takes priority over overlap; certified
serialization is allowed. Focused PyPTO/PyPTO-Lib provenance and contract fixes
are in scope, separately committed, without changing kernel intent. A5 is out
of scope. Use A3 device qualification and shared A2/A3 host/simulator regression;
do not imply A2 silicon evidence. Every new target claim requires authoritative
PTO-ISA, model, negative-test and focused device evidence. Missing facts or
qualification block completion rather than silently shrinking the population.

The earlier executive decision remains sound:

> Protocol-first, obligation-complete.

The experiments refine what *obligation-complete* requires. Storage timelines
and protocol recognizers cannot be the only source of correctness obligations.
They are proof templates over facts produced by an independent, path-sensitive
effect analysis.

The revised decision is:

> Build one canonical typed-obligation engine from storage atoms, execution
> lanes, structured control, iteration, and target effect domains. Let complete
> protocol recognizers discharge and compress those obligations. Let direct
> repair cover the supported remainder. If any effect or obligation is neither
> proved intrinsic nor covered by a verified mechanism, fail closed or use the
> attributed whole-function fallback.

## What the experiments established

The durable measurements and their limitations are in the
[evidence ledger](ptoas-protocol-sync-evidence-ledger.md). The design-relevant
results are:

| Result | Consequence |
|---|---|
| On `51fa6e338`, C.6 and C.7 reproduced every earlier semantic aggregate over 394 rows and 405 functions | The lane/track observations survived that E/F implementation; the corpus has not yet been rerun on `faeb8e71d`. |
| Exact interval projection represented 18,025 accesses as 3,492 atomic tracks, with zero mask/overlap biconditional failures over 1,276,033 access-pair relations | Atomic region-membership masks are a viable physical-storage representation for the exact interval subset. |
| All 66,839 existing linear raw pairs occurred in exactly one transition and had an exact common-track mask | Frontier grouping preserves the current raw-pair universe. |
| Only 54,217 memberships had independently checked linear frontier containment; 12,622 were guarded, multi-point, or otherwise non-linear | Zero pair omission is not yet a general frontier-coverage proof. |
| A separate lane/track lifecycle reconstruction found 177 strict local shapes in 76 corpus functions, while E admitted none | Lane/track structure is broader than E, but broader discovery is not permission to emit a protocol. |
| All four strict `ReadyRelease<1/2>` fixtures matched E on `51fa6e338`; `faeb8e71d` now rejects them for unsupported visibility while independent reconstruction remains diagnostic | E remains a useful complete proof template only for shapes passing its current visibility contract. |
| Branch/join tests retain exact atoms but produce no raw pairs across region boundaries; a join read is currently assigned the lexically last writer rather than a must/may-reaching definition | The missing control-sensitive memory-state analysis is a correctness gap, not a candidate-selection detail. |
| A dynamic subview is conservatively represented as the whole parent allocation without a precision-loss marker | The current projection audit is exact relative to supplied intervals, not necessarily exact relative to source address semantics. |
| The current target ablation changes only discharge status while transition identity remains stable | Storage/effect discovery and target mechanism selection should remain separate. |
| 46,382 exact overlapping read/read pairs exist, including 1,588 in ACC and 6,293 across execution lanes, while ordinary raw hazards correctly omit read/read | Ordinary memory hazards and target-specific proxy/resource ordering need separate effect domains. |
| Pre-`faeb8e71d` reanalysis recognized 5,868 fixed actions but imported zero into the planning selected world; the new concrete verifier reconstructs fixed synchronization supply | Connect concrete supply to stable obligation IDs and rerun the legacy-placement audit before claiming coverage. |
| ReadyRelease verifier mutations now reject deleted or moved actions, guarded placement, shifted physical ranges, and changed slot modulus | Independent reconstruction is practical, but diagnostics still need obligation-local witnesses rather than a whole-plan failure alone. |

## Revised semantic architecture

```text
shared operation semantics and preserved address/slot facts
                         |
                         v
        StorageDomain + AccessRegion + precision state
                         |
                         v
      region-membership atoms and access masks (spatial)
                         |
                         + execution lanes (where)
                         + structured control (which paths)
                         + iteration relations (which instances)
                         + effect domain (memory/visibility/proxy/etc.)
                         |
                         v
          MemoryStateFlow and canonical typed obligations
                         |
             +-----------+------------------+
             |                              |
             v                              v
 complete protocol certificates       direct candidates
             |                              |
             +--------------+---------------+
                            v
                  selected-world interpreter
                            v
                atomic allocation/materialization
                            v
             independent emitted-IR obligation verifier
```

This is not a return to the old undifferentiated dependency graph. A generic
edge does not say which bytes, path, iteration, completion property, visibility
property, or target mechanism is involved. The obligation graph is typed and
is derived only after those dimensions are explicit.

### 1. Storage domains, regions, and precision

Introduce an explicit precision contract:

```cpp
enum class SyncRegionPrecision : std::uint8_t {
  Exact,
  Conservative,
  Unknown,
};

struct SyncAccessRegion {
  SyncStorageDomainId domain;
  SyncAddressSet region;
  SyncRegionPrecision precision;
};
```

A storage domain identifies address space, physical allocation/alias scope,
and ownership scope. An exact access region may be represented by one interval,
several strided intervals, or a symbolic/Presburger set. Endpoint splitting
produces minimal disjoint region-membership atoms. An access is a mask over
those atoms.

The invariant is biconditional only for `Exact` regions:

```text
access regions overlap  iff  atom masks intersect
```

Conservative regions may share a conservative atom but cannot support a claim
of exact disjointness. Unknown regions remain explicit obligations or an
unsupported reason. A dynamic subview must never silently inherit an exact
whole-root range.

### 2. Path- and iteration-sensitive memory state

Add a sparse memory-state structure per storage atom, analogous in role to
MemorySSA but retaining PTO iteration and physical-effect facts:

```cpp
struct SyncMemoryStateNode {
  enum class Kind : std::uint8_t { Definition, Use, Phi };

  Kind kind;
  SyncStorageAtomId atom;
  SyncPhaseId phase;
  SyncControlGuard guard;
  SyncIterationRelation iteration;
  llvm::SmallVector<SyncMemoryStateNodeId, 2> reaching;
};
```

At a choice join, a use may have several may-reaching definitions and at most
one must-reaching definition. It is invalid to choose the lexically last writer
across mutually exclusive arms. For affine loop/slot cases, a reaching
generation is a relation `(writer, writer iteration)`, not just a generation
number attached to an operation.

Storage generations become derived views over memory-state nodes and atoms.
The existing `StorageTimelineAnalysis` may remain as a narrow protocol
recognizer, but it is no longer the canonical record of all storage effects.

### 3. Canonical typed obligations

Every obligation contains its evidence and required property:

```cpp
struct SyncObligation {
  SyncObligationId id;
  SyncObligationKind kind;
  SyncEffectDomain domain;

  SyncPhaseId source;
  SyncPhaseId target;
  llvm::SmallVector<SyncStorageAtomId, 4> atoms;
  llvm::SmallVector<SyncMemoryStateNodeId, 2> generations;

  SyncControlRelation control;
  SyncIterationRelation iteration;
  SyncRequiredProperty required;
  SyncProofStatus status;
};
```

The initial effect domains are separate:

- local memory RAW, WAR, and WAW completion;
- global publication/visibility;
- physical SSA completion;
- ordered/opaque effects;
- accumulator or proxy-resource ordering;
- fixed synchronization/queue supply;
- function and physical-section exit completion; and
- unresolved alias or semantic facts.

This separation lets the implementation proceed demand type by demand type
without allowing success in one domain to erase another. Read/read is not an
ordinary memory hazard. It becomes an obligation only when an independently
specified target proxy/resource rule says so.

### 4. Frontier compression is a proof, not discovery

Execution-lane traversal remains valuable. It proposes legal source and target
frontiers and identifies common cuts. Each grouped candidate must retain the
exact obligation IDs it claims and a certificate that expansion covers every
member under the original guards and iteration relation.

The following are valid uses of lane structure:

- find a shared publication/acquisition cut for several one-shot obligations;
- interval-stab same-lane completion obligations;
- place ReadyRelease publication, acquisition, final-use, and overwrite
  actions; and
- reduce redundant direct mechanisms after a complete selected-world check.

A frontier with no underlying obligation membership is diagnostic evidence,
not a selectable candidate.

### 5. Target rules discharge properties

The target model must answer queries such as:

```text
does program order establish required completion?
does PIPE_X barrier establish required completion?
does event P->Q establish completion, visibility, or only issue order?
does this proxy/resource class require RAR ordering?
does this fixed operation supply a certified edge or collective?
```

Storage facts do not change during target ablation. Only proof status,
available mechanisms, and cost change. Event-direction legality must remain
separate from GM publication, cross-core collective, and cache-visibility
semantics.

### 6. Protocol recognizers certify complete subgraphs

`OneShotPublish`, shared cuts, and `ReadyRelease<N>` remain prepared protocol
strategies. A recognizer consumes a set of canonical obligations and emits an
atomic certificate containing:

- exact obligation membership;
- storage atoms/generations and slot mapping;
- guarded publication/acquisition/final-use/overwrite frontiers;
- target properties supplied by every action;
- prologue/body/epilogue token transfer;
- zero-trip and steady-state invariants; and
- resource requirements before concrete allocation.

E should therefore be adapted, not discarded. Its strict `ReadyRelease<1/2>`
proof and verifier are the reference implementation of a complete protocol
certificate. Its current whole-function/channel-shaped discovery should
eventually consume the canonical memory-state and obligation records.

### 7. Behavior when no protocol pattern matches

Protocol recognition is an optimization and proof-compression layer, not the
definition of completeness.

If no pattern matches:

1. interpret the empty protocol world;
2. enumerate every canonical obligation in every effect domain;
3. prove intrinsic obligations through the target contract;
4. build direct candidates for the supported remainder;
5. re-run the selected-world interpreter;
6. materialize only if the world is complete; otherwise use the explicit
   fallback/fail policy.

Checkpoint F already supplies the right high-level boundary: a selected world
must be complete before mutation. It should be changed to consume the canonical
obligation engine rather than maintaining a separate, partially overlapping
hazard enumeration.

“Complete synchronization pass” must always be qualified as complete for a
declared semantic and target subset. Unknown operation semantics, imprecise
storage, unsupported control, visibility, cross-core, proxy, or fixed-protocol
facts are unsupported—not silently synchronized by a generic event.

### 8. Fixed synchronization is imported supply

Explicit events, barriers, queues, and certified macro-internal protocols must
be translated into typed completion/visibility/token facts. The importer must
verify direction, event identity/lifetime, guard, iteration, and target
contract. An unverified fixed operation remains an ordered semantic action and
does not discharge obligations.

This importer is required before comparing old placements by actual coverage.
Operation-name matches may remain a diagnostic convenience but cannot be used
as precision, recall, redundancy, or correctness evidence.

### 9. Independent verification

The emitted-IR verifier reconstructs:

- address regions and atom membership;
- reaching generations across paths/iterations;
- target completion, visibility, and proxy ordering;
- fixed and generated synchronization effects;
- protocol token state; and
- coverage of every canonical obligation.

It may share operation semantics, address recovery, and target contracts. It
must not share selected membership, frontier certificates, or planner token
proofs. A failure should name the first uncovered obligation plus its atom,
guard, iteration, and required property.

The current ReadyRelease verifier now detects action deletion/motion, nesting
under a guard, physical-range changes, allocation capacity changes, lane/event
corruption, and slot-modulus changes. Its next refinement is localized
obligation diagnostics and full region reconstruction rather than a Boolean
whole-plan verdict.

## Implementation sequence

### G0: preserve the diagnostic baseline

- Keep C.6/C.7 read-only and `selectable=no`.
- Preserve versioned corpus summaries and the exact claim boundaries.
- Add no stable/alternating L1 or accumulator recognizer yet.

### G1: repair access-region precision

- Add `Exact/Conservative/Unknown` provenance.
- Mark dynamic subviews and address-preserving-but-range-losing aliases
  conservatively.
- Make storage-domain identity include allocation/alias scope, not only address
  space and numeric offset.
- Extend the atom census to interval, strided, cross-root, and symbolic cases.

Gate: no access described as exact may have lost source address information;
the atom-mask biconditional passes for every exact access.

### G2: memory-state flow over structured control

- Build per-atom definition/use/phi links.
- Report must- and may-reaching writers at joins.
- Add loop-instance relations and modulo-slot reuse distance.
- Treat partial overwrites as atom-specific kills.

Gate: the adversarial branch/join and partial-overwrite suite has no lexical
last-writer substitutions and every feasible conflicting path yields an
obligation or a target proof.

### G3: unify canonical obligations with F

- Move local RAW/WAR/WAW, completion, reclamation, SSA, ordered, alias, and exit
  records into one typed obligation store.
- Adapt the F selected-world interpreter and direct-repair planner to consume
  it.
- Require exact obligation membership from every grouped frontier.

Gate: expanding all grouped candidates reproduces the canonical obligation set
with zero omissions and no unrelated coverage under guard/iteration semantics.

### G4: import fixed synchronization and audit old output

- Reuse the concrete verifier's explicit event/barrier/queue supply model.
- Relate that supply to canonical planning obligation IDs.
- Re-run the 199 available old emitted programs.
- Report covered, uncovered, and redundant obligations by stable phase/atom ID.

Gate: the selected-world interpreter explains old synchronization semantically;
no fixed action is counted merely because its operation name matches.

### G5: adapt protocol recognizers

- Make OneShot and E consume canonical obligations.
- Allow several independent protocol candidates when their storage/effect and
  resource contracts do not interfere.
- Keep each protocol's actions atomic through reverse deletion and allocation.

Gate: strict E fixtures remain exact matches, and independent lifecycle shapes
either receive a complete certificate or an explicit uncovered obligation.

### G6: qualify additional effect domains

Add domains one at a time with specification and device evidence:

1. local same-core completion;
2. fixed event/barrier supply;
3. global visibility/publication;
4. cross-core collectives;
5. accumulator/proxy-resource ordering; and
6. stable/alternating L1 ownership.

Each domain needs a target contract, negative tests, mutation tests, an
independent verifier rule, and a hardware qualification statement. A compiler
model or literature analogue alone is not hardware evidence.

## Experiments still required

The next experiments should be targeted at missing semantic facts rather than
another broad recognizer census:

1. provenance-precision mutation: dynamic/static subview offsets, strides,
   holes, cross-root equal numeric offsets, and post-planning aliases;
2. path-state mutation: both-arm/single-arm writes, may-uninitialized reads,
   partial kills, loop exits, and guarded overwrite/release;
3. fixed-supply import: delete, move, redirect, or reuse one old event and show
   the exact obligation that becomes uncovered;
4. target proxy query: use PTO-ISA documentation plus focused A3 tests to decide
   whether any ACC/proxy read/read pairs require ordering; and
5. ReadyRelease hardware gate: retain the already defined zero/one/odd/even,
   random-trip, reservation, repeated-launch, and manual/legacy comparison.

Stable/alternating L1 and accumulator protocol recognition remains postponed
until G1-G4 and the relevant G6 target contract pass. The current 177
independent lifecycle shapes are a discovery population for those later
experiments, not candidates to enable now.

## Grounding and claim discipline

The representation choices are supported by established compiler structures:
Triton ConSan region-membership atoms, LLVM MemorySSA and MemorySSA-backed DSE,
MLIR affine dependence analysis, Feautrier array dataflow, IREE resource
timepoints, OpenXLA allocation slices/live ranges, and accelerator event
verification work. Those sources support architecture and terminology; they do
not prove PTO hardware semantics.

PTO mechanism claims must come from the PTO-ISA/target contract and focused
hardware campaigns. The current C.6/C.7 measurements are host-side compiler
evidence only. In particular, a supported event-direction query is not proof
of GM publication, cross-core correctness, accumulator/proxy ordering, or
performance.
