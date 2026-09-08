# Analysis-utilization assessment after the generation implementation

Historical assessment of the first generation implementation, before the shared-flow
follow-up. The [current design](../../../../docs/designs/ptoas-insertsync-buffer-generations.md)
and [implementation-status table](../../../../docs/designs/ptoas-insertsync-next-improvements-plan.md)
record which findings have since been addressed. The source descriptions below
refer to that earlier worktree snapshot.

Reviewed on 2026-09-08 against R8 `4c19cc1cba444f28ab9af2a022aae1e3c80079eb`
and the current uncommitted per-buffer generation changes. The supplied audit is
`/home/toni/work/pypto3_sync_more/ANALYSIS_UTILIZATION_R8.md`; it explicitly
excludes those uncommitted changes. This follow-up is source inspection, using
the already recorded native regression; no new build or benchmark was run.

The audit's central finding is supported by the native callers. The generation
change improves a committed GEMM plan, but it does not yet provide the shared,
open storage-flow service proposed by the audit. Calling it a complete shared
generation analysis would overstate the implementation.

## What changed, and what remains

| Question | Current implementation | Assessment |
| --- | --- | --- |
| Does early construction use generation facts? | `analyzeBufferGenerations()` computes reaching writes and next-access summaries; the latter drive publication, final release, and unused-generation return. | Yes, for admitted exact slots/bundles. This changes emitted GEMM. |
| Does it reuse R5's generation result? | It shares the guarded physical/control import, but runs a separate dataflow over `LifecycleSpec`. | No. R5 `generationFrontiers()` remains a separate producer. |
| Do open facts survive unsuccessful construction for native consumers? | The analysis result can retain facts on failure, but qualification rejects the channel; ordinary fallback receives no such result. | No useful fallback integration yet. |
| Does residual repair use the new analysis? | Accepted certificates reach `removeSuppliedDependencies()`; R8's mixed full-completion query remains active. | Yes, indirectly through a complete selected protocol, not through open reaching-write/final-reader queries. |
| Does R5 placement consume the new result? | No native placement caller receives `BufferGenerationAnalysis`. | Missing connection. |
| Are dynamic selector relationships imported? | `exactSlice()` requires one physical address. Local `allPairs()` still passes a default occurrence relation. | General selector/loop-carried correspondence remains unsupported. |
| Are loop summaries parameterized transfers? | The new code runs fixed points over the existing guarded graph. | No reusable transfer over arbitrary incoming symbolic generations. |
| Do all optimization flags compose? | Successful lifecycle/generation synthesis still returns before prefix pruning and R4/R5 refinement. | No; this must be explicit. |
| Is allocation joint over logical lifetimes? | Residual allocation runs first; lifecycle streams receive dedicated remaining keys. | No. Existing bounded retry remains. |
| Are independent checks preserved? | Actual emitted effects/events are reimported and reconstructed; retained completion exemptions are rechecked. | Yes; preserve this boundary. |

The new forward analysis stores **static writer-site alternatives** for each
guarded read. Repeated sites denote families of generations. It does not expose
an explicit source/target iteration-distance map. Matching episodes are checked
by the subsequent per-slot protocol state machine. These are useful guarantees
for the admitted domain, but not general occurrence correspondence.

There is also a concrete utilization distinction within the new result:
`nextAccess` drives construction; reaching definitions reject possible
uninitialized reads; exported `reads` are otherwise used for diagnostics, and
`possibleFinalReaders` is inspected by tests. Neither exported table currently
drives a separate native repair or placement decision.

## Source anchors

All paths below refer to the current checkout; the relevant R8 paths are also
present at the pinned commit.

- `include/PTO/Transforms/InsertSync/BufferGenerationAnalysis.h`:
  reaching-definition fixed point, continuation summaries, then protocol
  construction in the same entry point and result object.
- `lib/PTO/Transforms/InsertSync/LifecycleBoundarySynthesis.cpp`,
  `qualifyInsertSyncLifecycleBoundaries()`: reads the generation certificate and returns
  on incomplete qualification, before materializing guarded actions.
- `lib/PTO/Transforms/InsertSync/LifecycleSynthesis.cpp`, `makeChannel()`,
  `discover()`, and `removeSuppliedDependencies()`: exact-slot discovery,
  selected-channel filtering, and certificate/full-completion supply.
- `lib/PTO/Transforms/InsertSync/StorageFrontierAnalysis.cpp`, `allPairs()`:
  constant local byte regions, `compareAccesses(x, y, {}, false, budget)`, and
  `OrderedUnknown` witnesses. `checkPlacement()` separately consumes the R5
  snapshot.
- `lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp`: the Applied return precedes
  the ordinary path's optional pruning and refinement.
- `include/PTO/Transforms/InsertSync/LifecycleProtocol.h`,
  `allocateLifecycles()`: dedicated keys after residual reservations.

The audit also correctly distinguishes `RegionAtomBoundary` snapshots from
transfer functions, `GenerationFrontier` ordering epochs from complete value
versions, and the predecessor cone in `CompletionWitness` from a precise
failure-dependency set. Those stronger contracts cannot be obtained just by
connecting the existing fields.

## Demonstrated effect and next integration boundary

The [existing local regression](BUFFER_GENERATION_RESULTS.md) records six GEMM
channels committed on the first attempt: four L1 slots and two L0 bundles.

| Plan | Set/wait pairs | MTE2 | MTE1 | M | FIX | PIPE_ALL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R8 staged + MMAD | 44 | 3 | 2 | 4 | 1 | 1 |
| Current generation change | 56 | 0 | 0 | 4 | 1 | 1 |
| Hand-tuned | 53 | 0 | 0 | 0 | 0 | 0 |

The three tested equivalent parity expressions give the same new inventory.
This is a committed compiler-output change, not evidence of device speedup.
The [R8 device campaign](R8_DEVICE_RESULTS.md) predates it.

The next bounded integration should separate storage-flow discovery from token
construction, retaining immutable facts even when a closed protocol is
unavailable. Construction and placement should consume that shared result;
ordinary repair should receive qualified requirement/disjointness queries with
the original access identities. A final-reader fact alone cannot authorize
removing synchronization: the selected plan must also establish completion.

Acceptance should include a native case where protocol construction fails but
a retained storage-flow fact still changes a justified ordinary-repair decision.
Keep the existing GEMM parity tests and R8 control regressions. Explicitly bypass
post-refinement for selected protocols until ownership-aware movement and fresh
reconstruction are implemented.

Joint allocation is a real remaining capability, but the current GEMM candidate
already fits without retry. It should not become a prerequisite for examining
the remaining M/FIX obligations and exit drain. The accumulator's possible
live-in path still prevents its specialization; this describes the optional
analysis domain, not a demonstrated production synchronization defect.
