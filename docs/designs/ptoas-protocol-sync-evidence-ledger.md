# ProtocolSync experiment and related-work evidence ledger

## Purpose and status

This document is the consolidated evidence record for the read-only
ProtocolSync lane and storage experiments through Checkpoint C.7. It records
the questions asked, methods used, measurements obtained, conclusions that the
measurements support, claims that remain unsupported, and the relationship to
similar compiler structures.

The underlying implementation and all reported candidates are diagnostic only.
They do not change selection, allocation, materialization, verification,
fallback, or emitted synchronization. Every experimental candidate is marked
`selectable=no`.

The measurements were rebaselined on 2026-09-05 from the
`codex/protocol-sync` worktree. The E/F base is
`51fa6e338a048b9c49f96e2619bec683464d1ce6`; the local C.5 commit is
`ac476e6b8a9629cb120748c688782b279f886e50`, and the C.6/C.7 diagnostics were
uncommitted at measurement time. Both 394-row post-rebase runs completed
without failure or timeout. Every semantic aggregate matched the earlier
`4975d7523e1cdc88147049043ecc6780022ba718` run; only measured analysis time
changed.

The later soundness, lifecycle, target-ablation, effect-domain, and old-emitted
IR experiments use the same E/F base and frozen corpus. Their ignored
build-directory artifacts are supporting local evidence rather than repository
inputs; this checked-in ledger is the durable claim record.

The branch was subsequently rebased through remote commit `7a09ed458` on
2026-09-05, including the `faeb8e71d` concrete verifier and the later A2
one-shot commits, producing local C.5 commit `de108e8a9`. The C.6/C.7 worktree
was reapplied and host-validated, but the 394-row corpus was not rerun after
this second rebase. Therefore every corpus number below remains explicitly a
`51fa6e338` result. The incoming concrete emitted-IR verifier changes fixed-sync
supply and visibility behavior, so related coverage and E-admission claims must
be rebaselined before being treated as current-branch measurements.

## Document map

This ledger summarizes and connects the more detailed records:

- [semantic foundation](ptoas-protocol-sync-foundation.md): checkpoint
  boundaries, implementation map, retained CanonicalSync evidence, and rollout
  limits;
- [lane-pattern experiment](ptoas-protocol-sync-lane-pattern-experiment.md):
  raw access-pair retention, three C.6 recognizers, fixture comparisons, and the
  394-row lane-pattern corpus run;
- [storage-track experiment](ptoas-protocol-sync-storage-track-experiment.md):
  C.7 byte-region projection, storage-transition construction, detailed corpus
  measurements, and storage/E differential results; and
- [Checkpoint E](ptoas-protocol-sync-checkpoint-e.md): the strict
  `ReadyRelease<1/2>` admission, protocol, transaction, verifier, and device
  qualification contract; and
- [obligation-engine amendment](ptoas-protocol-sync-obligation-engine-amendment.md):
  the proposed revision that makes typed obligations and path-sensitive memory
  state the correctness substrate beneath protocol recognition.

The earlier implementation plan and amendment remain historical planning
inputs. Where they conflict with measured C.5-C.7 behavior, the measurements
and the fail-closed boundaries recorded here take precedence for future design
work.

## Terminology and independent dimensions

The experiments distinguish structures that were conflated in the earlier
pure dependency-graph approach:

| Term | Identity | Question answered | Does not establish |
|---|---|---|---|
| Operation or phase | Stable structured program point and semantic effect | What happens, and where in structured control? | Physical aliasing or target completion |
| Execution lane | `(physical core, pipeline)` | Where does an access execute? | Which bytes it affects or whether same-lane work is complete |
| Logical storage family | Allocation provenance and generation identity | Which logical contents are being produced or consumed? | Physical disjointness |
| Physical storage slot | Concrete member selected from a multi-buffer family | Which ping/pong allocation is selected for an iteration? | The selector's control validity by itself |
| Storage track | `(address space, exact atomic byte interval)` | Which physical region can two effects share? | Execution order, visibility, or protocol legality |
| Raw access pair | Exact overlapping endpoints with proven linear order | Which local RAW, WAR, or WAW hazard is in the experiment's narrow universe? | Cross-block or general loop-carried order |
| Frontier | Program point before or after which an obligation may be placed | Where can a transition be represented structurally? | Which mechanism is legal |
| Target query | A3 target-contract answer for the exact lane/resource relation | Intrinsic completion, pipe barrier, directed event, or unsupported? | Device qualification outside the modeled contract |
| Protocol | An atomic set of prime/body/drain actions and invariants | Does the whole recurring ownership transfer close correctly? | Hardware correctness until separately qualified |

The resulting invariant is:

> Execution lanes say where accesses execute; storage tracks say which
> contents and byte regions they affect; synchronization obligations arise
> only from combining both with control and iteration order.

The experiments refine the last clause: before an obligation becomes a
candidate mechanism, it also needs an authoritative target completion query.

## Experimental progression

| Checkpoint | Added evidence | Main question | Outcome |
|---|---|---|---|
| C.5 lane projection | Execution lanes and backward/forward ready/release frontiers | Does structured lane traversal retain useful placement information? | Yes, but lane order alone cannot decide completion or storage overlap. |
| C.6 lane patterns | Raw endpoints plus `SharedOneShotFrontier`, `SameLaneCompletionCut`, and `ChoiceBalancedRoundTrip` | Can narrow shapes survive strict timeline/channel rejection and reproduce old placements? | Yes for the tested shapes; the layer remains incomplete and diagnostic. |
| C.7 storage tracks | Exact byte atoms, physical-slot membership, aliasing families, and transition masks | Does a separate storage projection recover information lost by families, timelines, and lanes? | Yes; it retains partial overlap and residual obligations, while E remains the stricter protocol proof. |
| E differential | Actual read-only `ReadyRelease<1/2>` planner result | Do C.7 lifecycle frontiers agree with E where E admits? | Yes on all four strict fixtures; no recurring corpus shape passes all E gates. |

## Questions, hypotheses, and outcomes

### Q1: Is execution-lane projection useful beyond a dependency DAG?

Hypothesis: grouping operations by physical core and pipe, then traversing
structured control backward for acquisition and forward for release, can find
common placement cuts that a graph-only dependency cover obscures.

Outcome: supported for the tested shapes. Shared one-shot frontiers reproduced
all 22 old event placements at operation-name granularity. All 481 same-lane
completion cuts in rows admitted by the old implementation reproduced an old
selected barrier placement. Choice entry/exit structure also remained visible
inside rows whose whole-function old plan failed.

Boundary: the MTE2 and Vector same-lane fixtures have similar lane shape but
different target answers. MTE2 `tload`-to-`tload` WAW is currently modeled as
intrinsic; the Vector relation requires `PIPE_V`. Lane order is therefore
structural evidence, not a completion rule.

### Q2: Is a logical buffer family a sufficient storage identity?

Hypothesis: no. Physical overlap, subranges, and slot selection must be modeled
separately from logical generation identity.

Outcome: supported. The corpus contains 2,515 atomic tracks bound by more than
one logical family. The partial-overlap fixture splits into 48 atoms—16
first-only, 16 shared, and 16 second-only—rather than treating partial overlap
as equivalence. Depth-two fixtures produce two 512-byte tracks with conditional
`slot(t)==0/1` occurrence membership.

### Q3: Can the combined lane/track layer preserve the experiment's local
hazard universe?

Hypothesis: every exact physically overlapping access pair with proven linear
order in one block should appear in a lifecycle, completion, or residual
transition.

Outcome: supported for exactly that denominator. All 66,839 pairs were
represented: 30,821 by completion cuts, 348 by ready transitions, and 35,670
by one-pair residual transitions. Release transitions are derived
loop-lifecycle obligations and have no member in this deliberately
non-loop-carried raw-pair universe.

Boundary: this result says nothing about the 126 non-physical accesses,
unknown aliases, cross-block order, general loop-carried hazards, cross-core
visibility, accumulator RAR, fixed synchronization, or exits and tails.

### Q4: Are storage transitions equivalent to Checkpoint E protocols?

Hypothesis: no. Storage transitions should agree with E's frontiers when E
admits, while retaining useful obligations when E rejects.

Outcome: supported. The four strict `ReadyRelease<1/2>` fixtures agree on track
masks, physical slots, resource directions, publication, acquisition,
final-use, and next-overwrite positions. In the corpus, all 474 recurring
transition records were rejected by the actual E planner: 382 for unsupported
control flow, 72 for unsupported visibility, and 20 for an incomplete channel
set. None was admitted.

This corrected an earlier measurement error. Channel admission alone had made
264 transitions look E-admitted, but channel admission is only one E gate. The
reported status now comes from invoking the complete read-only E planner.

### Q5: Do target-supported diagnostic transitions imply a complete pass?

Hypothesis: no. Target feasibility, obligation coverage, complete protocols,
and device qualification are distinct gates.

Outcome: all 37,261 C.7 transitions receive a supported result from the current
A3 target query, but 35,670 are residual one-pair observations and no corpus
recurring transition is E-admitted. The result demonstrates target-query
coverage of this corpus, not a complete synchronization algorithm.

### Q6: Did the atom projection and grouping audits prove soundness?

Hypothesis: exact known intervals can be represented losslessly as atom masks,
and every raw pair grouped into a transition can be recovered exactly once.

Outcome: supported for the supplied exact intervals and the existing raw-pair
universe. The audit checked 18,025 access masks and 1,276,033 access-pair
relations: 158,767 overlapping and 1,117,266 disjoint. There were zero access
mask mismatches, zero overlapping pairs without a common track, and zero
disjoint pairs sharing a track. All 66,839 raw pairs appeared in exactly one
transition with the complete common-track mask.

Boundary: only 54,217 pair memberships had independently checked linear
frontier containment. The remaining 12,622 belong to guarded, multi-point, or
otherwise non-linear frontiers and still need a control/iteration proof. A
dynamic-subview adversarial fixture also showed that the legacy provenance may
replace a lost dynamic offset with the whole parent range without marking the
result imprecise. Thus `projection-audit status=exact` currently means exact
relative to the intervals supplied to C.7, not necessarily exact relative to
source address semantics.

### Q7: Can storage generations be reconstructed independently of E?

Hypothesis: strict recurring shapes can be reconstructed from storage tracks,
execution lanes, control, and slot relations before consulting E's selected
answer.

Outcome: supported for the deliberately narrow reconstructor. All four strict
`ReadyRelease<1/2>` fixtures matched E on capacity, loop, producer/consumer
lanes and phases, publication, acquisition, final use, next overwrite, reuse,
and physical slots. The corpus contained 177 reconstructed shapes in 76
functions. E rejected those functions first for schedule failure (31),
unsupported control flow (31), unsupported visibility (8), or incomplete
channel set (6).

Boundary: the 177 shapes are not selectable. The reconstructor requires one
guard-free loop, one write, one read, exact common track masks, known lanes, and
capacity one or two. Its broader discovery shows information beyond E's
whole-function gates; it does not prove the rest of each function is covered.

### Q8: Can old synchronized output be audited by actual coverage now?

Hypothesis: reanalyzing the old branch's emitted PTO IR will let the selected
world prove which current obligations each old event or barrier covers.

Outcome on base `51fa6e338`: rejected by a missing semantic bridge. All 199
available old emitted programs re-parsed successfully, covering 204 functions.
The shared semantic frontend recognized 5,868 fixed supplied protocol actions,
but the selected world imported zero of them as completion, visibility, or
token facts. They became 5,878 opaque semantic actions and the empty-world
interpreter reported 9,080 residual obligations.

Remote commit `faeb8e71d` subsequently added a concrete emitted-IR verifier that
reconstructs modeled supply from synchronization operations. The 199-program
audit has not yet been rerun through that verifier. The zero-import result is
therefore historical evidence for the planning selected world, not a claim
about the new concrete verifier.

The older operation-name comparison remains useful only as a locator: 553 of
3,361 selected mechanisms have at least one name-level transition match and
2,808 do not. It is not an actual coverage audit. Fixed synchronization must be
translated into verified selected-world supply, or a separate emitted-IR
coverage checker must be built, before coverage/redundancy claims are possible.

### Q9: Are read/read pairs ordinary storage hazards?

Hypothesis: no; ordinary memory RAW/WAR/WAW discovery should omit read/read,
while a separate target effect domain may impose proxy/resource ordering.

Outcome: the separation is supported. The exact projection contains 46,382
overlapping read/read pairs, including 1,588 in ACC and 6,293 across execution
lanes, while none enters the ordinary 66,839-pair RAW/WAR/WAW universe.

Boundary: the current target interface has no proxy/resource read/read query.
The experiment therefore does not establish which ACC or other read/read pairs
need synchronization. That decision requires PTO-ISA or hardware-model
semantics and focused device tests.

## Reproduction inputs and commands

The C.6 and C.7 corpus runs used the 394 A3 rows retained from the frozen
CanonicalSync campaign at `19674e06ff25be87badf0cb3595e014e443b558d`.
The local source archive is:

```text
build/protocol-sync-lane-c6-corpus/inputs/
  canonical-sync-394-corpus-d99520397.tar.gz
SHA256: 91b011b9225c1308d185885554752ba1d64198436c3af60c6d4d243cda2bbd2a
```

The compiler was invoked in analysis-only mode:

```text
ptoas --pto-arch=a3 --pto-level=level3 --emit-pto-ir \
  --enable-insert-sync --protocol-sync-analysis-only \
  --protocol-sync-dump=lane-frontiers --protocol-sync-statistics \
  INPUT.pto -o /dev/null

ptoas --pto-arch=a3 --pto-level=level3 --emit-pto-ir \
  --enable-insert-sync --protocol-sync-analysis-only \
  --protocol-sync-dump=storage-tracks --protocol-sync-statistics \
  INPUT.pto -o /dev/null
```

The disk-backed runner used at most two workers. The local supporting artifacts
are:

```text
build/protocol-sync-lane-c6-corpus/run_lane_pattern_experiment.py
build/protocol-sync-lane-c6-corpus/lane-pattern-results-post-rebase-51fa6e338/
  rows.jsonl functions.jsonl candidates.jsonl candidates.tsv summary.json
build/protocol-sync-lane-c6-corpus/storage-effect-v1-51fa6e338/
  rows.jsonl functions.jsonl candidates.jsonl candidates.tsv
  transitions.jsonl transitions.tsv summary.json
build/protocol-sync-lane-c6-corpus/evidence-audit-v1-51fa6e338/
  old-mechanism-coverage.tsv summary.json
build/protocol-sync-old-emitted/old-emitted-reanalysis-v1-19674e06-vs-51fa6e338/
  rows.jsonl summary.json
```

The build tree is intentionally ignored. The tables below preserve every
aggregate used to support the design conclusions; per-row and per-transition
records can be regenerated with the command and runner above while the frozen
corpus is available.

## C.6 lane-pattern measurements

All 394 rows and 405 functions completed without failure or timeout.

| Measure | Result |
|---|---:|
| Raw access endpoints retained | 18,151 |
| Raw hazardous pairs | 66,839 |
| Intrinsic completion pairs | 470 |
| Same-pipe barrier pairs | 51,261 |
| Cross-lane/not-applicable pairs | 15,108 |
| Unsupported target/mechanism pairs | 0 |
| Pattern candidates | 1,050 |
| Logical cost | 1,050 |
| Steady-state action cost | 1,096 |
| Aggregate lane-pattern analysis time | 187,178 microseconds |

| Candidate | Count | Target | E status | Old placement comparison |
|---|---:|---|---|---|
| Shared one-shot frontier | 22 | 22 supported | 22 not applicable | 10 shared-frontier and 12 direct-event matches |
| Same-lane completion cut | 1,020 | 1,020 supported | 1,020 not applicable | 507 matches, 513 no selected name match |
| Choice-balanced round trip | 8 | 8 supported | 8 rejected for multiple consumers | No old selected placement in the corresponding fail-closed rows |

The 513 unmatched completion cuts are not 513 proven disagreements. Most occur
inside rows for which the old all-or-nothing planner failed elsewhere. Another
26 name-level matches occur in rows whose old plan later failed for a different
function or demand. Because old dumps lack occurrence-stable program-point IDs,
all old-placement matches are supporting differential evidence rather than
one-to-one identities.

## C.7 storage-track measurements

All 394 rows and 405 functions again completed without failure or timeout.

| Measure | Result |
|---|---:|
| Exact local accesses attempted | 18,151 |
| Projected accesses | 18,025 |
| Unprojected accesses | 126, all non-physical |
| Atomic storage tracks | 3,492 |
| Track occurrences | 29,770 |
| Tracks bound by multiple logical families | 2,515 |
| Tracks with uncertain alias classification | 0 |
| Raw hazardous pairs | 66,839 |
| Raw pairs represented by transitions | 66,839 |
| Raw pairs unrepresented | 0 |
| Storage transitions | 37,261 |
| Logical cost | 37,261 |
| Steady-state action cost | 52,146 |
| Aggregate storage-track analysis time before added audits | 155,713 microseconds |
| Aggregate storage-track analysis time with projection, transition, lifecycle, and effect audits | 308,375 microseconds |

| Transition | Count | Raw-pair members | Track memberships | Maximum tracks | Action cost |
|---|---:|---:|---:|---:|---:|
| Completion | 1,020 | 30,821 | 4,224 | 11 | 1,020 |
| Ready | 334 | 348 | 334 | 1 | 668 |
| Release | 237 | 0 | 237 | 1 | 474 |
| Residual | 35,670 | 35,670 | 66,891 | 32 | 49,984 |

| Target result | Count |
|---|---:|
| Directed event | 15,317 |
| Same-pipe barrier | 21,474 |
| Intrinsic completion | 470 |
| Unsupported | 0 |

| Old selected placement comparison | Count |
|---|---:|
| Selected operation-name placement match | 2,986 |
| No selected operation-name placement match | 33,805 |
| Intrinsic and therefore not selectable by old mechanism | 470 |

The old comparison key is resource, mechanism kind, and operation-name
placement. Repeated operation names can make one old selection correspond to
multiple transition records. These counts must not be interpreted as a
precision or recall score.

### Post-rebase soundness and differential census

| Audit | Result |
|---|---:|
| Exact access masks | 18,025 |
| Access-pair relations checked | 1,276,033 |
| Overlapping / disjoint relations | 158,767 / 1,117,266 |
| Mask, overlap, or disjointness mismatches | 0 |
| Overlap components | 2,347 |
| Maximum atoms in one component / access | 33 / 33 |
| Raw-pair transition memberships | 66,839 |
| Pairs covered exactly once / multiply / omitted | 66,839 / 0 / 0 |
| Common-track mask mismatches | 0 |
| Independently checked linear frontier memberships / mismatches | 54,217 / 0 |
| Non-linear frontier memberships awaiting path proof | 12,622 |
| Strict independent lifecycle reconstructions | 177 in 76 functions |
| Exact E differential on strict fixtures | 4 matches, 0 mismatches |
| Overlapping read/read pairs | 46,382 |
| ACC / cross-lane read/read subset | 1,588 / 6,293 |

The target-ordering ablation held transition identity fixed at SHA256
`bbd457971e71898ed9ace22f8bda26b9137b4461618105c1d2bf642a03081456`.
With no target rules, all 37,261 transitions remained unresolved. Intrinsic
completion discharged 470; enabling same-pipe barriers raised the total to
21,944; enabling the currently modeled directed events raised it to 37,261.
This is a compiler-model feasibility ablation, not a hardware correctness
result.

## Fixture evidence

| Fixture | Observation | Supported conclusion |
|---|---|---|
| `ready_release_one` | One 512-byte Vector track; ready/release masks and all lifecycle points match E | C.7 preserves the complete depth-one storage differential |
| `ready_release_two_offset_zero` | Two 512-byte tracks with offset-zero conditional membership; E match | Physical slots and selector offset are retained |
| `ready_release_two_offset_one` | Same tracks with offset-one selector; E match | Slot identity is not hard-coded to one selector phase |
| `ready_release_two_equivalent_selectors` | Equivalent producer/consumer selectors map to the same two tracks; E match | Equivalent selectors preserve common physical membership |
| `partial_overlapping_writes` | 48 atoms, including exactly 16 shared atoms; residual WAW mask covers those 16 | Atomic tracks preserve partial overlap after timeline rejection |
| `reference_shared_event` | Ready transition retains old event placement | Tracks do not disturb a positive lane/event differential |
| `reference_old_mte2_barrier` | Two WAW pairs classify as intrinsic, not barrier | Current target semantics override prototype precedent |
| `reference_vector_completion_cut` | Common `PIPE_V` cut spans the expected tracks | Same-lane completion cuts can coalesce many raw pairs and regions |
| `reference_choice_round_trip` | Ready/release frontiers survive; actual E rejects control flow | Structural discovery is broader than complete-protocol admission |
| `chained_nontransitive_overlap` | `[0,512)`, `[256,768)`, and `[512,1024)` split into four atoms; first and third remain disjoint | Overlap components are construction containers, not alias equivalence classes |
| `whole_prefix_suffix` | The whole range maps to both atoms; prefix and suffix map to one each | Partial kills can be represented atom by atom, though generation flow is not yet atom-sensitive |
| `dynamic_subview_range_loss` | A dynamic one-row subview is reported as exact `[0,2048)` with `uncertain-alias=no` | Source-range precision is missing and must be repaired before selection |
| branch/join adversarial cases | Same/different/one-arm writers retain atoms but produce zero cross-region raw pairs; join state chooses a lexical generation | Path-sensitive must/may-reaching generations are still absent |
| modulo four/six with loop step two | Selector facts retain coefficient 2 and modulus 4/6; E and the independent recognizer reject unsupported capacity before deriving reuse | General `N/gcd(a,N)` slot recurrence exists in the helper but is not integrated into generation/protocol analysis |

## Claim register

| Claim | Status | Evidence | Important limitation |
|---|---|---|---|
| Execution lanes expose useful placement structure | Supported for tested fixtures/corpus | 22/22 shared-event and all 481 old-admitted-row barrier matches | Name-level comparison; not mechanism proof |
| Same-lane order implies completion | Rejected | MTE2 intrinsic versus Vector barrier fixtures | Must ask the target contract |
| Logical storage families imply physical disjointness | Rejected | 2,515 multi-family tracks | Exact local physical ranges only |
| Atomic tracks correctly retain partial overlap | Supported for exact intervals | 48-atom fixture and exact 16-track shared mask | Unknown and non-physical ranges are rejected, not approximated |
| C.7 preserves its narrow raw-pair universe | Supported | 66,839 represented, zero lost | Universe excludes cross-block and general loop-carried order |
| Atom masks implement overlap iff intersection | Supported relative to supplied exact intervals | 1,276,033 pair census, zero mismatches | Dynamic subview experiment shows supplied precision may itself be wrong |
| Grouped frontiers are generally coverage-sound | Not yet supported | 66,839 exact-once memberships; 54,217 linear checks | 12,622 non-linear memberships lack path/guard proof |
| Storage frontiers agree with E where E admits | Supported on four strict `51fa6e338` fixtures | Full publication/acquisition/final-use/overwrite and slot differential | `faeb8e71d` now rejects these fixtures for unsupported visibility; the corpus differential is not rebaselined |
| Independent track/lane reconstruction adds discovery beyond E | Supported diagnostically | 177 shapes in 76 E-rejected functions | Does not establish whole-function coverage or protocol legality |
| Current branch/join generations are sound | Rejected | Adversarial joins choose the lexical writer and omit cross-region raw pairs | Requires must/may-reaching memory-state flow |
| Old placements are covered by current obligations | Unknown | Pre-`faeb8e71d` audit of 199 emitted programs recognized 5,868 fixed actions | The new concrete verifier models fixed supply, but the audit has not been rerun against it |
| Read/read is an ordinary memory hazard | Rejected | 46,382 read/read overlaps are correctly absent from RAW/WAR/WAW | Proxy/ACC rules remain unmodeled |
| Storage tracks replace E | Rejected | 474 recurring transition records rejected by E | E supplies whole-protocol legality and verification |
| Current diagnostics form a complete sync pass | Rejected | 35,670 residuals plus unmodeled obligation classes | All records remain non-selectable |
| Current target support is hardware qualification | Rejected | Target queries are compiler-model results only | Requires PTO-ISA/hardware evidence and device campaigns |

## Related compiler and research structures

Sources were checked against primary project documentation on 2026-09-04.
They are architectural comparisons, not inherited correctness arguments.

| Work | Relevant structure | What transfers to ProtocolSync | What does not transfer |
|---|---|---|---|
| Triton [Concurrency Sanitizer and BufferRegion](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md) | Exact shared/tensor-memory address sets are split into physical region-membership atoms; visible read/write frontiers are tracked over state lanes; target hooks model barriers and asynchronous completion | Strongest direct precedent for exact membership atoms, region masks, visibility frontiers, target hooks, and phase-indexed synchronization state | It is runtime instrumentation, not PTO placement; its logical threads, memory spaces, and target mechanisms differ |
| [LLVM MemorySSA](https://llvm.org/docs/MemorySSA.html) | `MemoryDef`, `MemoryUse`, and `MemoryPhi` represent may-reaching memory versions and ordering constraints | Demonstrates the value of a separate memory-state representation and explicit control-flow joins | Versions are deliberately broad and depend on alias walkers; there is no PTO pipeline completion or ownership protocol |
| LLVM [MemorySSA-backed dead-store elimination](https://llvm.org/doxygen/DeadStoreElimination_8cpp_source.html) | Walks backward from a killing write to prior clobbering definitions, checks intervening reads and path coverage, and distinguishes complete from partial overwrite | Concrete implementation precedent for finding final-use and overwrite/reuse frontiers from memory-state links | Its goal is deletion under sequential LLVM semantics; partial-overwrite exploration is bounded and it has no asynchronous completion model |
| MLIR [One-Shot Bufferize](https://mlir.llvm.org/docs/Bufferization/) | Whole-function analysis builds alias/equivalence sets, detects read-after-write conflicts, and supports analysis-only conflict diagnostics before rewriting | Supports separating logical SSA values from physical buffer aliasing and keeping diagnostics read-only before transformation | It chooses in-place reuse or copies, not events/barriers or accelerator visibility |
| MLIR [affine dependence analysis](https://mlir.llvm.org/doxygen/AffineAnalysis_8cpp_source.html) | Computes memory dependences and dependence components across loop depths | A useful model for repairing iteration distance and loop-carried access relations | Applies to analyzable affine accesses and does not supply target synchronization semantics |
| Feautrier's [array dataflow analysis](https://doi.org/10.1007/BF01407931) | Computes the source statement and source iteration that supplies an array read in static-control programs | Direct precedent for making a storage generation an explicit `(writer, writer iteration)` relation rather than only an operation name | Exactness depends on analyzable control and affine access relations; it supplies neither target completion nor event selection |
| Lam's [software-pipelining work](https://suif.stanford.edu/papers/lam-sp.pdf) | Modulo scheduling and variable expansion distinguish simultaneously live values from different iterations | Classical basis for deriving physical-slot recurrence and for checking ping-pong or deeper rotating storage | Primarily concerns scalar/register lifetimes and does not establish safety for partially overlapping accelerator buffers |
| MLIR [SCF loop pipelining](https://mlir.llvm.org/doxygen/namespacemlir_1_1scf.html) | Applies a provided stage schedule and mechanically constructs prologue, steady state, and epilogue | Closely parallels ProtocolSync's need to distinguish prime/body/drain roles | It assumes the supplied schedule is valid; it does not discover synchronization or verify PTO mechanisms |
| [IREE Stream](https://iree.dev/reference/mlir-dialects/Stream/) | Makes resource operands, sizes, async execution, and completion timepoints explicit; resources are unavailable until their timepoints are reached | Supports representing resource identity and temporal availability as separate IR facts | Timepoints are higher-level dependencies, not same-pipe completion, visibility fences, or bounded PTO event protocols |
| OpenXLA [buffer assignment](https://github.com/openxla/xla/blob/main/xla/service/buffer_assignment.h), [HLO live ranges](https://github.com/openxla/xla/blob/main/xla/hlo/utils/hlo_live_range.h), and [memory-space assignment](https://github.com/openxla/xla/blob/main/xla/service/memory_space_assignment/memory_space_assignment.h) | Maps logical values to allocation slices, divides lifetimes into use segments and allocation sequences, and represents async-copy start/done availability | Supports the distinction between logical families, physical subranges, lifetime/overwrite order, and availability | It solves storage reuse and placement from a logical schedule, not cross-pipeline publication or completion |
| MLIR [GPU](https://mlir.llvm.org/docs/Dialects/GPU/), [NVGPU](https://mlir.llvm.org/docs/Dialects/NVGPU/), and [AIR](https://github.com/Xilinx/mlir-air/blob/main/docs/AIRComputeModel.md) | Expose asynchronous dependency tokens, token joins, copy groups, waits, barriers, and loop-carried token dependencies | Provide nearby IR vocabularies for PTO one-shot events, grouped waits, and recurring protocol actions | Synchronization is normally explicit or supplied by an earlier transform; these dialects do not infer it from post-allocation byte-region lifecycles |
| [AccelSync](https://arxiv.org/abs/2605.07881) | Defines a restricted accelerator-pipeline language and parameterized hardware event semantics, then checks whether cross-unit same-buffer write/read pairs are ordered by happens-before | Direct precedent for an independent, hardware-parameterized synchronization-coverage verifier and for separating program, synchronization, and barrier order | It is a May 2026 preprint about verification, not a placement algorithm; its soundness/completeness claims are bounded by its modeled language and hardware semantics |

The Triton comparison is especially close: its documented state lanes are
minimal exact physical region-membership atoms, and its visibility transfers
are masked by those atoms. That independently supports the C.7 decomposition.
It does not validate PTO's particular track construction, event directions,
same-pipe barriers, or recurring ownership protocol.

AccelSync is the closest formal verification comparison. ProtocolSync should
borrow the separation between placement and an independently reconstructed
coverage proof. It should not be described as evidence for synchronization
synthesis or as device qualification for PTO.

### Terminology refinement from the comparison

The literature makes three overloaded meanings of "lane" especially risky.
Future design text should distinguish:

- **execution lane**: `(physical core, pipeline)`;
- **storage domain**: address space, root physical allocation, and ownership
  scope;
- **storage atom** or **region-membership atom**: a minimal disjoint physical
  byte set whose membership in every known access region is identical;
- **storage track**: the ordered occurrence history for one storage atom,
  optionally parameterized by a symbolic physical-slot selector;
- **storage generation**: the contents of an atom from its defining write to
  the next killing overwrite; and
- **protocol lane**: the ReadyRelease token index assigned to a reusable
  physical slot.

The C.7 implementation currently uses `storage track` for the atomic interval
and attaches its occurrence history to that object. That is internally
consistent, but `buffer lane` should not become another synonym. Overlap is not
an equivalence relation. An overlap-connected component is only a construction
container and must be subdivided into exact membership atoms. For example,
`[0,4)`, `[2,6)`, and `[4,8)` require four atoms with memberships `A`, `A+B`,
`B+C`, and `C`; unioning the three whole ranges would lose the fact that the
first and third do not overlap.

For strided, symbolic, or non-contiguous regions, the conceptual unit is an
address set or Presburger relation rather than necessarily one interval.
Unknown accesses must retain an explicit conservative or unresolved state;
they must not be assigned an invented exact atom.

### Invariants suggested by the combined evidence

These are proposed proof obligations for subsequent experiments. They have not
all been established by C.7:

1. **Spatial exactness:** every exactly known access is the union of its atom
   mask, and two exact accesses overlap exactly when their masks intersect.
2. **Generation correctness:** every read has a must- or may-reaching
   generation on every feasible control path and iteration instance; joins
   preserve uncertainty rather than choosing one writer.
3. **Conflict completeness:** every feasible RAW, WAR, or WAW pair sharing an
   atom is either proved ordered and visible by target semantics or represented
   by an obligation.
4. **Frontier-preserving compression:** expanding any grouped frontier must
   reproduce coverage for every underlying hazardous pair under the original
   guards and iteration relation.
5. **Reuse safety:** the write that begins generation `g+1` cannot issue until
   all relevant uses of generation `g` on the same physical atom have
   completed.
6. **Mechanism separation:** storage analysis discovers obligations;
   execution-lane and target-completion rules decide whether intrinsic order,
   a barrier, an event, a queue rule, or ReadyRelease can discharge them.
7. **Effect-domain separation:** target-specific proxy or resource hazards,
   such as a possible accumulator RAR constraint, remain distinct from
   ordinary storage RAW/WAR/WAW while reusing frontier machinery where valid.

## Design implications

The experimental evidence supports the following architecture:

```text
structured control and iteration facts
          +
execution lanes + exact storage tracks
          |
          v
explicit typed obligations and legal frontiers
          |
          v
target completion and visibility queries
          |
          v
complete protocol recognizers + residual cover
          |
          v
atomic selection and resource allocation
          |
          v
materialization + independently reconstructed verification
```

Prepared patterns remain valuable as proofs for complete protocol families,
not as the only source of obligations. If no complete pattern matches, the
pass must not silently assume safety. It must either prove a complete residual
cover using independently legal mechanisms or fail/fall back according to the
explicit policy.

Demand kinds should remain separate at discovery time—publication,
acquisition, final use, overwrite, completion, visibility, cross-core, and
fixed synchronization—but selection must preserve atomic coupling where a
protocol requires it. In particular, ready and release halves of a recurring
ownership transfer cannot be independently accepted merely because their local
frontiers exist.

## Open work before selection

The following facts or proofs are still missing:

- cross-block and structured-region raw dependence order;
- general loop-carried dependence and authoritative iteration distance;
- unknown, dynamic, and non-physical storage identity;
- cross-core execution and visibility;
- MTE3-to-MTE2 same-address GM publication semantics;
- fixed synchronization and queue interactions;
- exit, tail, and zero-trip obligations outside strict E;
- event scarcity and global coalescing across typed demands;
- stable/alternating L1 ownership protocols; and
- accumulator RAR protocols.

Stable/alternating L1 and accumulator recognizers remain postponed until their
storage, core, and iteration facts are repaired. Adding pattern rules before
those facts exist would encode examples rather than prove protocols.

## Follow-up campaign disposition

The ordered campaign produced these outcomes:

1. **`51fa6e338` reproduction — complete.** C.6/C.7 completed 394/394 rows and
   reproduced every semantic aggregate; timing alone changed. A new
   `faeb8e71d` corpus rebaseline remains pending.
2. **Projection census — complete for supplied exact intervals.** Zero mask or
   overlap biconditional failures. Dynamic-subview precision remains a newly
   exposed upstream gap.
3. **Transition expansion — complete for raw-pair membership.** All 66,839
   pairs occur exactly once with exact masks. Non-linear frontier containment
   remains unproved for 12,622 memberships.
4. **Independent E reconstruction — complete on the measured base.** Four
   `51fa6e338` fixture matches and 177 diagnostic corpus reconstructions. On
   `faeb8e71d`, the fixtures remain independently reconstructible but E rejects
   them for unsupported visibility.
5. **Actual old-placement coverage — pending re-audit.** The pre-`faeb8e71d`
   selected world treated fixed events/barriers as opaque. The new concrete
   verifier models emitted synchronization supply, so the 199-program audit
   must be rerun before reporting coverage.
6. **Adversarial storage/control/slot suites — added.** They validate atom
   splitting and expose dynamic-range, path-generation, and capacity/reuse
   integration gaps rather than hiding them.
7. **Target ablation — complete for current target queries.** Transition
   identity is stable; only discharge status changes.
8. **Read/read effect separation — complete as a census.** Ordinary hazards
   exclude read/read; target proxy semantics are unavailable.
9. **Verifier mutation — extended.** The ReadyRelease verifier rejects missing
   or moved synchronization, guarded/nested placement, shifted physical ranges,
   capacity/lane/event corruption, and slot-modulus changes. Localized
   obligation diagnostics remain future work.
10. **Deferred recognizers — still postponed.** The experiments strengthened,
    rather than removed, the reasons to delay stable/alternating L1 and
    accumulator protocols.

The natural next experiments are now narrower:

1. add explicit `Exact/Conservative/Unknown` access-region precision and rerun
   the projection census;
2. implement per-atom must/may-reaching state at structured joins and loop
   backedges, then rerun the path and modulo suites;
3. connect concrete fixed-sync supply to canonical obligation IDs, then repeat
   the old-emitted coverage audit through the new verifier;
4. add obligation-local verifier witnesses and mutation tests for each effect
   domain; and
5. obtain PTO-ISA/hardware-model and focused device evidence for any proposed
   proxy/ACC read/read rule and for the existing ReadyRelease device gate.

The [obligation-engine amendment](ptoas-protocol-sync-obligation-engine-amendment.md)
turns these findings into a staged revision of the pass.

## Validation record

The diagnostic implementation associated with this ledger passed:

```text
cmake --build build --parallel 2

pto-protocol-sync-ready-release-test

llvm-lit -j 2 -v --filter protocol_sync build/test/lit

llvm-lit -j 2 build/test/lit
```

On `7a09ed458` plus rebased local C.5 `de108e8a9`, the full build and all 34
focused ProtocolSync lit tests passed. The one-shot, ReadyRelease, direct-repair,
and mixed C++ executables also passed every section. On the immediately
preceding `faeb8e71d` base, the full lit result was 1,891 passed, 1 unsupported,
and 0 failed out of 1,892 discovered tests; only the focused suite was rerun
after the two A2 one-shot commits landed. The scoped compliance checker reported
27 files, zero errors, and zero warnings. `git diff --check` was clean.

These are host/compiler validation results. No new device campaign was run for
C.6 or C.7, and none of the conclusions above should be relabeled as
PTO-ISA or hardware qualification.

## C7.1 rebaseline on the A2/A3 merged implementation

The [2026-09-05 C7.1 report](ptoas-protocol-sync-c71-rebaseline.md) supersedes
the pending corpus and concrete-audit statuses above. Both 394-row campaigns
and all 199 old emitted programs were rerun on `8c3a0b58d` plus the recorded
diagnostic patch. The actual patch, scripts, inputs, commands, per-row records,
raw diagnostics and hashes are retained in a disk-backed archive; compact
records are under `protocol-sync-evidence/c71/patch-frozen-8c3a0b58d/`.
That initial snapshot was patch-frozen. The subsequent clean rerun at
`1e911a4e00b62ad63846e531e88e11936a38ce16` completed both 394-row campaigns
and the 199-program audit without `--allow-dirty`; source stability passed.
All semantic aggregates reproduced, with differences only in timing fields.
Compact records are under `protocol-sync-evidence/c71/clean-1e911a4e0/`.
The raw archive is `build/protocol-sync-c71-1e911a4e0-evidence.tar.gz`,
11,749,756 bytes, SHA256
`1802eb8ac72e0d91f6b9a7fb3a0cbe268aa7842ee2be829ab4eedcb72ad9a461`.
This completes the clean diagnostic freeze, not the new frontend acceptance
campaign or target qualification.

All 40 comparable storage counters reproduced the earlier local `44e3bb474`
run. The new first-proof census divides 474 recurring rejections into 244
cardinality, 88 multiple-loop, 50 choice and 92 visibility records. All 126
unprojected accesses are local PyPTO-Lib queue-origin/consumer accesses.
Multi-family use-shape classification leaves asynchronous liveness unproved;
only 52 tracks are straight-line sequential-use candidates.

F accepts 17 of 204 concrete functions and rejects 187. Of the rejections,
171 stop in extraction or synchronization reconstruction and 16 reach the
residual-obligation stage. These are verifier-surface results, not a legacy
race count or exact coverage percentage. The full host suite passes 1,892
tests with one unsupported and no failures; all 35 ProtocolSync tests pass.

The 152-kernel acceptance report is retained separately with its source hash,
but its exact input manifest is still missing. The user has chosen a fresh
current-main PyPTO/PyPTO-Lib acceptance population instead of recovering those
historical inputs; collection and clean-revision admission runs remain pending.
The new acceptance runner
retains overlapping exposed blockers independently of strict admission.
The [GM alias contract](ptoas-protocol-sync-gm-alias-contract.md) records the
agreed default-safe / assumed-disjoint-argument modes for the next semantic
implementation; this diagnostic snapshot does not implement the mode switch.

## Native-coverage revision: explicit GM alias contract

The first semantic slice adds `--protocol-sync-gm-alias=may-alias` and
`--protocol-sync-gm-alias=assume-disjoint-arguments`. One shared caller contract
now feeds residual interpretation, OneShot, ReadyRelease and concrete
verification. Supported views, casts and SCF forwarding retain possible argument
roots; incomplete roots and same-root accesses do not gain disjointness.
The [contract report](ptoas-protocol-sync-gm-alias-contract.md) records the
CLI/IR precedence, transactional emission and exact limitations.

The host full suite passes 1,894 tests with one unsupported and zero failures.
The two new lit files exercise native direct/mixed/OneShot admission, stronger
may-alias re-verification, invalid/overridden contracts, read-only analysis,
rollback, branch/loop roots and integer-round-trip rejection. The campaign
runner now records A2/A3 and alias overrides independently and preserves each
kernel's overlapping diagnostics. Its 19 accounting tests pass.

This is not general may-alias repair, access-region precision, a canonical
local-obligation store, structured loop repair, or target publication
qualification. No new device claim or selectable lifecycle family is added.
Fresh-main frontend snapshots are pinned in the C7.1 report; their native
build, input collection and four-way acceptance campaign remain pending.
