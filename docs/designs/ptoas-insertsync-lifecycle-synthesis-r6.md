# InsertSync R6 — construct slot lifecycles before residual insertion

## Review basis and result

Reviewed integration base: `1509506a43a619fc0f888a49ead90aecce92b0fc` on
`tonibohnlein/PTOAS:codex/insertsync-revision-r1`, following R4 commit
`ca2647e71f0a069aa300c9ed9af3b498b35e741c`.

The supplied R4 results report 264 successful case/arm rows (528 PTO/C++
invocations) and no removed or guarded barriers. The reported R5 four-case run
moves no signals or waits. These are the integration campaign's results, not
native experiments repeated for this package. Matching scalar payload replay is
not a numerical or asynchronous correctness certificate. Zero inventory change
is not a device timing result.

The inspected integration still forwards the options and calls R4/R5 after
analysis, motion, cleanup, allocation and emission. No confirmed new integration
race/deadlock is established by this source review. The principal limitation is
in the design of our R4/R5 packages, not evidence that the integration agent
implemented them incorrectly:

1. They accept the old communication graph and hardware key assignment as fixed.
2. Their general completion bit means all prior occurrences of one static phase,
   which is conservative for overlapping generations.
3. The NativeGraph adapter requires a function-level core attribute and rejects
   physical sections. The unchanged historical GEMM input itself uses one
   `pto.section.cube` and no function-level core attribute. This is a concrete
   importer mismatch; its contribution to each measured zero-change row still
   needs the native pass-point diagnostic, not inference from counts alone.
4. They need enough whole-function post-emission modeling before making a local
   improvement. Moving a wait or a signal cannot split an over-broad handoff.

The review did not reproduce native tests or device results. The checked-in
`.ptoas-workspace.json` is machine-specific workspace metadata; cleanup of that
file is separate from this functional patch and is not performed here.

## Decision

Keep InsertSync as the production pass and keep R4/R5. Add one opt-in constructor
inside that pass, before ordinary dependency insertion:

```
existing IR and translated physical accesses
    -> R5 guarded structural projection (not a completion gate)
    -> complete exact-slot / LEFT+RIGHT bundle lifecycle
    -> exact supplied access relationships
    -> existing InsertSync residual insertion, motion and allocation
    -> non-serializing allocation for the new logical streams
    -> emit residuals plus atomic lifecycle actions
    -> reconstruct/check/commit
```

There is no new input annotation, dialect operation, caller promise, kernel-name
matcher, set-cover objective, or standalone synchronization pass. The new output
status/counters describe compiler work; the recognizer does not consume them.

## 1. Existing IR facts used now

- Physical addresses, memory space, allocation-sized conservative bounds, and
  aliases come from the existing `BaseMemInfo` translation. Equal physical
  intervals match even when their logical handles differ.
- A new optional argument to the shared R5 NativeGraph admits exactly one existing
  top-level `pto.section.cube` or `pto.section.vector`. It checks consistency with
  any existing function-level core annotation and rejects other physical work
  outside that context (the existing terminal ALL is retained).
- The function-level attribute is not synthesized. Protocol priming/draining stays
  inside the existing physical section when there is one. Ordinary R4/R5 callers
  retain their original constructor behavior unless they use this new entry.
- The same guarded CFG, loop entry/backedge/bypass and supported residue reasoning
  are reused before synchronization has been inserted. An unsynchronized function
  is not required to already have completion supply.
- Producer/consumer identity comes from physical lane and read/write sets. Bundle
  membership comes from LEFT/RIGHT address-space identity and common consumers,
  not function names, magic addresses or a tile-size pattern.
- Every potentially overlapping translated access is inspected. Partial, unknown,
  multi-address, or other-lane participation vetoes that optional candidate.

No new information is added to the input IR. Native `multi_tile_get` families with
runtime-selected event arrays, loop-carried handle permutations, arbitrary
mutable descriptors and mixed-core collectives remain future imports. The adapter is designed to consume the explicit separate-slot representations
present in the frozen controls/GEMM, but native matching has not been run here;
it must not be described as universal dynamic-slot support or an admission result.

## 2. Precise generations where the pattern proves them

A selected channel owns one exact physical slot, or a LEFT/RIGHT operand bundle.
Its abstract transfer is:

```
Free -> acquire -> write every member once -> Ready
Ready -> acquire -> one or more complete-bundle readers -> Free
```

The write/read episode may repeat or be skipped as a whole. The recognizer checks
the finite product of this state with the actual guarded CFG. Loops are not
unrolled to a numeric trip horizon. Each completed episode renames the dynamic
generation. Issuing into another physical slot cannot erase this slot's readiness.

This is a generation-specific protocol invariant, not R5's all-earlier-static-
occurrences completion bit and not a runtime generation counter. The generic R5
completion analysis is NOT globally rewritten by this patch.

Every consumer must read all members of a bundle, on one consumer lane. All
producer writes must use one producer lane. A consumer that reads just LEFT, an
additional reader on another lane, a missing consumer, or an ambiguous first/final
site prevents that specialized bundle. Independent readers are not serialized to
make a pattern fit. The first/final roles must be uniform at a static operation;
otherwise the more general guarded-boundary constructor is still needed.

The supported language includes many explicit readers, choices selecting closed
episodes, and nested repetitions with uniform sites. It is NOT a full arbitrary
consumer-region protocol: a dynamically repeated single read site that is first
on one occurrence and last on another needs a further boundary extension.

A write epoch orders storage accesses. An allocation-sized conservative write is
not assumed to initialize or kill every byte. Original memory and scalar
operations are unchanged.

## 3. Complete specialized constructions

The same core implements:

- AIV input slots: MTE2 -> V readiness, V -> MTE2 reclamation.
- AIV output slots: V -> MTE3 readiness, MTE3 -> V reclamation.
- AIC panel slots: MTE2 -> MTE1 readiness, MTE1 -> MTE2 reclamation, for supported
  explicit multi-reader episodes.
- AIC LEFT/RIGHT operand bundles: MTE1 -> M readiness, M -> MTE1 reclamation.

The native selector requires at least one represented path with a second
production after release. Statically one-shot storage stays with general insertion
rather than acquiring an unnecessary primed round trip.

For each selected slot/bundle, the complete recipe is:

```
start of physical function/section scope: publish initial Free
before first write:                     acquire Free
immediately after last producer:        publish Ready
before first relevant reader:           acquire Ready
immediately after final relevant reader:publish Free
end of physical scope:                  consume final Free
```

Initialization and cleanup span the true selected scope, not each syntactic
inner-loop invocation. Zero work consumes the initial credit at cleanup; a
partially produced live generation is rejected. Unrelated source-pipe work after
the final reader is not included merely to move publication to a region exit.

L0 bundling is allowed only when all consumers actually require both operands.
It is not independent-readiness coalescing. The constructor may generate MORE
pairs than a broad legacy handoff; there is no combined event/barrier score.

The patch constructs a new communication plan for the selected storage. It is
not an arbitrary event splitter and does not synthesize ACC or alternating
lookahead protocols. Required M->FIX and FIX->M relations remain residuals. MMAD's
intrinsic ACC rule still does not establish operand completion.

## 4. Integration and residual correctness boundary

`InsertSyncAnalysis` gains a read-only `lifecycleSupply_` callback. At ordinary
memory-dependency discovery, only an exact access pair on a certified member is
removed from `depVec`. All unrelated or partially matching entries remain.
`alreadySync` is NEVER updated from the lifecycle certificate.

The complete protocol is planned first but not inserted into the mutable legacy
lists. Legacy motion, redundant-pair cleanup and widening therefore cannot split
or relocate its actions. They run only on the residual plan. After residual
allocation, the new streams are assigned keys disjoint from every live residual
key in the corresponding directed domain. They use pool 0..5, without interpreting
that software pool as the hardware's total capacity.

This is a conservative combined resource realization, NOT a new optimal global
allocator. It does not reuse the new keys between distinct channels, move
frontiers to fit, or silently introduce a barrier. A resource failure discards the
clone and uses ordinary InsertSync on the original input. It is not a proof that
silicon serialization is inevitable. A residual body-wide fallback also abandons
the specialization. The original legacy fallback policy remains available outside
the accepted specialized path.

Original non-sync operation identities, attributes, result types and SSA operands
are checked before commit. New concrete channel actions are reconstructed from
actual keys/positions and fresh translated accesses, not from candidate coverage
bits. The checker validates per-slot production/acquisition/release and zero-work
transfer. A separate R5 causal check validates the new keys without relying on
residual acknowledgements. A further combined event check must succeed; an
unproved composition falls back without rejecting the input. Internal malformed
IR/graph errors remain hard failures.

Residual memory correctness still uses the existing production analyzer and its
semantic contracts. This is not a new independent whole-kernel safety proof.
Additional readers on selected storage cannot be omitted by the new recognizer,
but the shared translation is not independently device-qualified by this patch.

The accepted constructor does not subsequently feed its atomic events through
R4/R5 motion. On a failed optional construction the normal pipeline, including
requested R4/R5 options, remains available. This preserves protocol ownership.

## 5. What this addresses, and what it does not

It addresses the specific architectural objections:

- structural reasoning now participates BEFORE insertion/allocation;
- selected slot generations are distinguished by an actual protocol invariant;
- new ready/release streams and key assignments can be created;
- complete specialized patterns coexist with general residual insertion;
- existing physical sections are consumed instead of demanding new annotations.

It does not yet promise changed counts on any native benchmark. The original
GEMM may still expose unmodeled access footprints, conditional final-use forms,
resource conflicts or conservative combined-event checks. Those outcomes must be
retained in the fixed denominator with the constructor reason.

The buffer controls' MTE3 barriers may be residual GM-partition false conflicts.
A local-slot protocol is not a proof that distinct GM access occurrences are
disjoint. Keep the WAW fix and improve that query separately. The unchanged
Conv2D/helper, triangular scalar, and cross-core cases do not acquire missing
semantics simply because a pattern constructor exists.

## 6. Concrete development sequence after integration

1. Build the actual native adapter and run the supplied positive/negative/section
   fixtures. Fix compiler/API failures rather than downgrading positive tests.
2. Run the unchanged two-/three-buffer, four-use and historical GEMM inputs with
   synthesis off/on, holding target, alias contract, MMAD and traversal fixed.
   Record selected channels, skipped exact dependencies, failed realization
   reasons, and emitted/execute counts separately.
3. Inspect remaining GEMM MTE1/MTE2 barriers with their exact storage witnesses.
   Extend one actual missing first/final-use case, retaining the concrete input.
4. Import native multi-buffer selector families from existing addresses and SSA
   expressions; handle carried-handle permutations using existing yields. Do not
   add frontend promises or a synthetic global parity selector.
5. Add hierarchical/alternating lookahead recipes only as complete protocols with
   entry, steady state, final-use and bypass contracts. Reuse old recognition
   ideas, not set cover or whole-region placement shortcuts.
6. Improve joint non-serializing resource assignment and optional-candidate retry.
   New scarcity-induced ordering needs its own recorded justification.

Default options remain off until full native and device gates pass. No new kernel
rejection is accepted merely because this optimization cannot establish a model.

## Validation/provenance

The external R6 patch package's `STATUS.json` records its producer-side host-test
provenance. Repository acceptance additionally builds the native MLIR adapter and
runs the lifecycle lit fixtures. Reference executions are finite tests, not a
universal hardware simulation. The independent asynchronous interpreter separates
issue, completion, signal firing and wait consumption and checks both ordered-signal
and permissive-submission models.

Native source anchors were read at the base SHA above. Consult:
`PTOInsertSync.cpp`, `InsertSyncAnalysis.{h,cpp}`, `StorageFrontierAnalysis.cpp`,
`SyncEventIdAllocation.{h,cpp}`, and the unchanged `step3_swizzle.pto` fixture.
The supplied comparative assessment's requirements/completion and atomic-pattern
rules govern this continuation; the older ban on evolving InsertSync and its
covering-selection architecture are explicitly superseded.

Official reference: Ascend C SetFlag/WaitFlag(ISASI), functionality and constraints;
MLIR SCF dialect. No new hardware completion, visibility or cross-core contract is
introduced. The Ascend page is a development preview, not new device evidence.
