# ProtocolSync storage-track experiment

## Status and invariant

This document records the diagnostic-only experiment following the lane-pattern
study. It changes no candidate selection, materialization, verification, or
fallback behavior. Every storage transition is emitted with `selectable=no`.

The experiment tests this invariant:

> Execution lanes say where accesses execute; storage tracks say which
> contents and byte regions they affect; synchronization obligations arise
> only from combining both with control and iteration order.

The results support that decomposition. They do not establish a complete
synchronization pass or qualify a mechanism on hardware.

The [experiment and related-work evidence
ledger](ptoas-protocol-sync-evidence-ledger.md) consolidates the C.5-C.7
hypotheses, complete measurement tables, fixture matrix, claim boundaries,
reproduction artifacts, and source-by-source literature assessment. This
document remains the detailed C.7 construction and result record.

## Projection semantics

`StorageTrackProjection` partitions every exact physical local access by all
byte-interval boundaries in its address space. A storage track is one
non-overlapping atom:

```text
(address space, half-open byte interval)
```

It records every storage family and access occurrence that covers the atom.
Family identity is therefore not treated as physical disjointness: two logical
families may bind the same physical track. Partial overlap is not treated as an
equivalence relation; it becomes shared and unshared atoms.

Each occurrence retains its generation, phase, pipeline stage, execution lane,
program points, guard, iteration domain, access mode, and slot selector. An
affine multi-buffer access is conditionally attached to each physical track.
For example, the two depth-two tracks carry `slot(t)==0` and `slot(t)==1`
membership predicates plus the selector coefficient, offset, modulus, and loop.
An unslotted depth-one occurrence is unconditional.

Non-physical, unknown-range, missing-family, and invalid-interval accesses are
reported as unprojected rather than assigned an invented track. Adjacent
strided intervals remain distinct atoms; the experiment does not coarsen gaps
or over-approximate a strided view as a dense range.

## Transition-frontier construction

`StorageTransitionFrontiers` combines tracks with the existing control,
iteration, and execution-lane facts in this order:

1. Found ready and release lane frontiers become lifecycle transitions masked
   by the tracks common to their source and target stages.
2. `SameLaneCompletionCut` candidates become completion transitions covering
   all of their still-uncovered raw access pairs and common tracks.
3. Every remaining exact linear raw pair becomes a residual transition at its
   direct source-after and target-before points.

This priority is diagnostic coalescing, not selection. Each transition records
its track mask, raw-pair members, resource direction, target query, estimated
logical/action cost, old-pattern analogue, operation-name placement, and
Checkpoint E result.

The target query is deliberately separate from track construction. Cross-lane
transitions ask the A3 target contract for a directed event. Same-lane
transitions use the operation-level completion results: intrinsic ordering,
same-pipe barrier, unsupported target, or unsupported mechanism. A lane match
alone never proves completion.

Checkpoint E status comes from invoking the actual read-only ReadyRelease
planner, not from using channel admission as a proxy. One-shot and residual
transitions are `not-applicable`; recurring transitions report the planner's
exact admission or first rejection reason.

## Differential method

The current compiler was run on the earlier fixtures and the frozen 394-row A3
corpus with:

```text
ptoas --pto-arch=a3 --pto-level=level3 --emit-pto-ir \
  --enable-insert-sync --protocol-sync-analysis-only \
  --protocol-sync-dump=storage-tracks --protocol-sync-statistics \
  INPUT.pto -o /dev/null
```

The disk-backed runner used two workers. It wrote one record per transition
containing target-query result, old pattern/mechanism kind, actual E status and
reason, track/raw-pair cardinality, cost, and old selected placement comparison.

The corpus was rerun after rebasing onto E/F base `51fa6e338` on 2026-09-05.
All semantic aggregates were identical to the earlier `4975d7523` run. The
initial post-rebase storage-track time was 155,713 microseconds. With the later
projection, transition, lifecycle, and read/read audits enabled, it was 308,375
microseconds; these diagnostic timings are not a production cost model.

The worktree was later rebased through `7a09ed458`, including `faeb8e71d`.
Host tests pass there, but the 394-row corpus has not been rerun. The
measurements in this document remain tagged to `51fa6e338`, especially the E
and fixed-sync differentials changed by the incoming concrete verifier and
visibility checks.

Old dumps preserve resource, mechanism kind, and operation-name placement but
not occurrence-stable program-point IDs. Matches are consequently supporting
name-level evidence. Repeated operation names can make one old selection match
multiple current transitions; counts are not a one-to-one mechanism mapping.

## Fixture results on `51fa6e338`

The strict E fixtures gave the clearest positive differential on the measured
base:

- `ReadyRelease<1>` projects one 512-byte Vector track. Ready and release use
  the same track, and the storage differential matches E's directions,
  publication, acquisition, final use, next overwrite, capacity, and implicit
  physical slot.
- All three `ReadyRelease<2>` functions project two 512-byte Vector tracks with
  physical slots zero and one. Both selector offsets and the equivalent-selector
  case retain conditional membership for both slots. Every storage/E comparison
  matches.
- The partial-overlap fixture, although rejected by timeline construction,
  projects 48 strided atoms: 16 first-only, 16 shared, and 16 second-only. Its
  residual WAW transition is masked by exactly the 16 shared atoms.
- The old shared-event fixture retains its event placement. The Vector
  completion fixture retains the common `PIPE_V` cut over two tracks. The old
  MTE2 barrier fixture is classified as intrinsic, agreeing with the current
  target model rather than reproducing the prototype's barrier. The balanced
  choice retains its ready/release frontiers, while the actual E planner rejects
  it for unsupported control flow.
- Chained physical ranges `[0,512)`, `[256,768)`, and `[512,1024)` split into
  four exact atoms. The first and third remain disjoint even though all three
  are in one overlap-connected component. Whole/prefix/suffix ranges split into
  two atoms with the expected membership masks.
- Branch/join adversarial cases preserve their storage atoms but expose the
  current ordering gap: cross-region raw pairs are absent, and a join read can
  inherit the lexically last generation instead of a must/may-reaching writer
  set.
- A dynamic subview exposes a provenance-precision gap. Its one-row access is
  represented as the full `[0,2048)` parent range and is printed as exact and
  non-uncertain. The audit is therefore exact only relative to the intervals it
  receives.
- Capacity-four and capacity-six step-two selectors retain affine coefficient
  and modulus facts, but timeline/E rejection occurs before the general
  `N/gcd(a,N)` reuse helper is integrated into lifecycle analysis.

On `faeb8e71d`, the same `ReadyRelease<1/2>` storage shapes remain available to
the read-only analysis, while the E planner rejects them for unsupported
visibility. The old positive differential is retained only as versioned
evidence; it is not a current E-admission claim.

## Frozen-corpus results

All 394 rows completed without failure or timeout, covering 405 functions.

| Measure | Result |
|---|---:|
| Exact local accesses attempted | 18,151 |
| Projected accesses | 18,025 |
| Unprojected accesses | 126, all non-physical |
| Atomic storage tracks | 3,492 |
| Track occurrences | 29,770 |
| Tracks binding multiple logical families | 2,515 |
| Raw hazardous pairs | 66,839 |
| Raw pairs represented by transitions | 66,839 |
| Raw pairs unrepresented | 0 |
| Storage transitions | 37,261 |
| Logical cost | 37,261 |
| Steady-state action cost | 52,146 |
| Aggregate storage-track analysis time before added audits | 155,713 microseconds |
| Aggregate storage-track analysis time with all added audits | 308,375 microseconds |

The raw-pair coverage denominator is intentionally narrow: exact physical
overlap and a proven linear order in one block. The 100% result says no member
of that universe was lost between projection and transition construction. It
does not cover the 126 non-physical accesses, cross-block ordering, general
loop-carried hazards, accumulator RAR, visibility fences, or unknown aliases.

| Transition kind | Count | Target query | Actual Checkpoint E | Old selected placement comparison |
|---|---:|---|---|---|
| Completion cut | 1,020 | 1,020 pipe barrier | not applicable | 507 matches, 513 no selected match |
| Ready | 334 | 310 event, 24 pipe barrier | 97 one-shot/not applicable; 237 recurring/rejected | 199 matches, 135 no selected match |
| Release | 237 | 223 event, 14 pipe barrier | all 237 rejected | 26 matches, 211 no selected match |
| Residual | 35,670 | 14,784 event, 20,416 pipe barrier, 470 intrinsic | not applicable | 2,254 event matches, 32,946 no selected match, 470 intrinsic/non-selectable |

The 474 recurring transition rejections consist of 382 unsupported-control-flow
records, 72 unsupported-visibility records, and 20 incomplete-channel-set
records. No corpus recurring transition is admitted by the full E planner.
That corrects the earlier proxy interpretation: channel admission alone had
made 264 transitions appear E-admitted even though they fail later E gates.

All 37,261 emitted transitions have a supported result under the current A3
target query. This is a property of the recognizers and corpus: unsupported
records are not evidence of safe fallback, and target support is not device
qualification.

## Soundness, lifecycle, and effect audits

The projection audit reconstructs every access mask from the original supplied
intervals and checks the biconditional relation between byte overlap and common
track membership. Over the corpus it checked 18,025 accesses and 1,276,033
access-pair relations. The 158,767 overlapping and 1,117,266 disjoint
relations produced zero mask, missing-track, or spurious-shared-track
mismatches. There were 2,347 overlap components; the largest component and
largest access mask each contained 33 atoms.

The transition audit expanded grouped candidates into raw pair IDs. All 66,839
pairs occurred exactly once, none occurred more than once, and all complete
common-track masks matched. Linear frontier containment was checked for 54,217
memberships with zero failures. The remaining 12,622 guarded, multi-point, or
non-linear memberships are recorded separately and are not covered by that
proof.

An independent strict lifecycle reconstructor does not read E's selected
answer. It derives the producer, consumer, loop, capacity, physical slots,
lanes, publication, acquisition, final use, overwrite, and reuse from the
schedule and tracks, then compares with E. It exactly matches all four strict
E fixtures. In the corpus it reconstructed 177 shapes in 76 functions. E's
first rejection for those functions was schedule failure (31), unsupported
control flow (31), unsupported visibility (8), or incomplete channel set (6).
These are discovery records, not complete protocols.

The effect census found 46,382 overlapping read/read pairs, including 1,588 in
ACC and 6,293 across execution lanes. They correctly produce no ordinary
RAW/WAR/WAW raw pair. The target model has no independent proxy/resource RAR
query, so no accumulator synchronization claim follows from these counts.

## Old-emitted-IR coverage probe

On base `51fa6e338`, the full old archive contains synchronized PTO output for
199 programs. All 199 re-parse, covering 204 functions. ProtocolSync recognizes
5,868 fixed supplied protocol actions, but the empty planning selected world
imports none of them as completion, visibility, or token facts. They become
5,878 opaque semantic actions and leave 9,080 residual obligations.

Commit `faeb8e71d` adds a concrete emitted-IR verifier that reconstructs modeled
supply from fixed synchronization. Consequently the historical zero-import
result no longer describes the whole current verification path. The
199-program coverage audit must be rerun against that verifier before reporting
actual coverage or redundancy.

## Related compiler structures

The closest concrete analogue found in the literature scan is Triton's
[ConSan BufferRegion model](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md),
which represents exact byte regions and reasons about region membership and
visibility state. Its purpose is instrumentation, but its separation of byte
regions from execution events strongly resembles storage atoms plus transition
frontiers.

[LLVM MemorySSA](https://llvm.org/docs/MemorySSA.html) provides memory versions
and clobbering definitions over alias sets. It supports the need for a distinct
storage-state dimension, but it does not model PTO pipeline completion,
physical slots, or ready/release token protocols. MLIR's
[affine dependence analysis](https://mlir.llvm.org/doxygen/AffineAnalysis_8cpp_source.html)
and [SCF pipelining](https://mlir.llvm.org/doxygen/namespacemlir_1_1scf.html)
are closer analogues for iteration distance and prologue/body/epilogue order.

MLIR [One-Shot Bufferize](https://mlir.llvm.org/docs/Bufferization/) separates
tensor-level values from buffer aliasing and in-place ownership decisions.
[IREE Stream](https://iree.dev/reference/mlir-dialects/Stream/) and OpenXLA's
[buffer assignment](https://github.com/openxla/xla/blob/main/xla/service/buffer_assignment.h)
and [live ranges](https://github.com/openxla/xla/blob/main/xla/hlo/utils/hlo_live_range.h)
likewise combine logical values with resource lifetime and physical storage.
These are useful architectural precedents, not PTO synchronization proofs.

The May 2026 [AccelSync preprint](https://arxiv.org/abs/2605.07881) is directly
related to verification of accelerator synchronization coverage. Its
parameterized hardware event semantics and independent coverage check are
relevant to ProtocolSync verification; it is not a placement or synthesis
algorithm. This experiment does not rely on its claims and does not use it as
A3 hardware grounding.

## Conclusions

The central invariant is supported with one refinement: control and iteration
order must include an authoritative target completion query before a frontier
becomes even a feasible mechanism.

Storage tracks add information that execution lanes and Checkpoint E do not:

- exact partial-overlap masks;
- conditional physical-slot membership;
- physical aliasing across logical families;
- residual obligations retained after strict timeline/E rejection; and
- a common substrate on which event, barrier, intrinsic, and recurring
  protocol recognizers can be compared.

Checkpoint E still adds information the tracks do not: a complete atomic
ReadyRelease protocol, visibility/control restrictions, resource allocation,
token invariants, materialization, and independent verification. The intended
architecture is therefore not “tracks instead of patterns.” It is:

```text
storage tracks + execution lanes + control/iteration order
  -> explicit obligation frontiers
  -> complete protocol recognizers and residual cover
  -> atomic selection, allocation, materialization, and verification
```

Before any storage transition becomes selectable, the obligation universe must
be extended to cross-block and loop-carried order, unknown/non-physical storage,
visibility and cross-core demands, fixed synchronization, exits/tails, and the
deferred accumulator and stable/alternating L1 cases. The missing storage,
core, and iteration facts for those deferred protocols should be repaired
before adding recognizers for them.
