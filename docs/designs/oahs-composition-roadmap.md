# OAHS composition experiments and implementation TODOs

Reviewed against `a6bbe1ff8` on 2026-09-19. The first AIC experiment is now
implemented and host-validated locally; unchecked items remain planned.

## Assessment and evidence boundary

The useful common mechanism is **completion support across storage domains**:
a transfer needed for one use can cover another use's reader completion,
preceding writer, and event-consumption knowledge before their own deadlines.
This is stronger than recognizing another bank loop or merging endpoints.
It must preserve each early publication and the original dependent access.

The supplied agent studies give plausible restricted constructions and valuable
negative examples. Their experiment totals are author-reported here. The four
new Downloads files (`WALKTHROUGH.md`, `ORIGINAL_INPUTS.md`, `SOURCES.md`, and
`inventory_audit.json`) were read and preserved with hashes under
`/home/toni/work/pypto3_sync_more/composition-review-20260919/`.
`review-provenance.json` records the audit: all 18 archived inventory file hashes,
329 numbered attention source excerpt lines, and the post-RMSNorm prepared-input
hash match the retained local corpus. This authenticates the source evidence;
it does not reproduce the reported protocol experiments.

The four executable experiment archives were still not found locally. Obtain
and hash them before importing or claiming reproduction of their tests:

- `oahs_compositional_next_steps.zip`
- `oahs_worked_compositions_examples.zip`
- `oahs_attention_composition_study.zip`
- `oahs_qwen_deepseek_prefill_walkthroughs.zip`

| Reported result | Appropriate use |
| --- | --- |
| Unequal 3/2-bank operands: 29 to 18 pairs, equal payload order | Synthetic construction target under regular, unskipped use |
| Post-RMSNorm projection: 102 to 81 pairs, equal payload order | Mechanism witness; 102 is deliberately constructed, not a native baseline |
| Archived attention AIC: 122 to 114 executed pairs | Investigate duplicate QK release on current output |
| Archived attention AIV: 84 relations removed, unchanged pair count | Investigate wait placement under the complete queue contract |
| Merge update: 4 to 2 pairs; quantization: 15 to 14 | Small local-effect regression candidates, not current compiler gains |

The attention study used the older `5ecb89fc7` corpus. Do not carry its excess
counts forward as current findings. The newer guarded attention device task is
already prepared; use its result alongside current native diagnostics.

## Rated shortlist and decision

Ratings are engineering judgments about the next experiment, not predicted
device speedups. Five means strongest priority. Effort/risk includes importer,
protocol, and occurrence qualification; a small reference proof does not imply
a small native patch.

| Candidate | Rating | Generality | Effort / qualification risk | Decision |
| --- | ---: | --- | --- | --- |
| AIC indirect QK release (C1) | 5/5 | High: a necessary provider covers another source | Medium / medium | Implemented locally: partial AIC 69 → 64 pairs; current-credit provider selection |
| Post-RMSNorm downstream return (C1) | 5/5 | High: retained value, reader/writer and key-consumption support | Medium / medium | Try next as the queue-free companion; regenerate current baseline first |
| Positive non-unit-step first-use/range facts (C4) | 4/5 | High: original scalar/control precision | Low--medium / low--medium | Separate bounded experiment; count actual residual/fence improvements |
| Merge update receipt reuse + seed negative (C5) | 4/5 | High: branch-sensitive source-prefix coverage | Low--medium / low--medium | Use as a small generalization test; may already be handled |
| AIV acquire after independent receive (C2) | 4/5 | High: deadline placement through a relay | Medium--high / high | Try after queue/slot semantics are qualified; potential overlap gain |
| Shared MAT input across sibling children (C4) | 4/5 | High: lifetime follows physical generation | Medium--high / high for A5 | Keep as the next region-composition target after target admission |
| Guard-stable co-use mapping and mixed routes (C3) | 4/5 | High within explicit participation premises | High / high | Stage behind a reproduced native gap; avoid a broad recognizer now |
| Quantization boundary and retained scale (C6) | 3/5 | High as a correctness/generalization test | Medium--high / high for A5 scratch | Later regression family; reported one-pair gain is not a priority driver |
| RoPE/cache multiple readers (C7) | 3/5 | High, but opportunity not yet localized | High / high | Park broad optimization; retain minimal staging as a control |
| Minimum-period selection alone (C3) | 2/5 | Restricted to regular unskipped schedules | Low for synthetic case / high if generalized | Portable proof/regression only until a matching native case exists |
| Opaque-prefix lower-bound theory | 2/5 for implementation | Potentially broad diagnostic value | Separate formal work | Send to paper/theory work; do not change acceptance or credit now |

First two experiments share one objective: avoid a duplicate completion path
without moving the necessary provider's endpoints. They should not start by
adding a new planner or general support-state subsystem. Diagnose whether the
missing information is provider coverage, retained history, occurrence matching,
or early recurring commitment; extend the responsible mechanism only.

Concrete latest-artifact observations from `guarded-attention-work/corpus/44.pto`:

- Lines 139 and 149 retain a direct M-to-MTE1 SET/WAIT around QK/PV reuse;
  lines 144--150 also carry M completion through MTE2 and probability readiness.
  This confirms two visible paths, not that deletion is already certified on
  every path or physical-key generation. The direct release can be installed
  during recurring preparation before ordinary provider selection runs.
- Lines 653--658 retain an MTE3-to-MTE2 acquisition before score pop. An MTE2
  barrier also precedes pop. Moving only one wait may not recover all ingress
  freedom; inspect the barrier witness and complete queue contract too.
- Static counts now split as AIC 69 pairs/7 named barriers and AIV 57 pairs/80
  named barriers, plus one ALL each. The archived 77/69 pair split is stale.

Treat the indirect AIC path as supporting evidence only while it is itself
required by the admitted contract. Do not preserve an unnecessary M-to-MTE2
serialization merely because it makes another channel redundant. If improving
that path removes the support, reassess the direct release rather than deleting
both. This dependency is part of the experiment's acceptance criteria.

The initial saved-output inspection above is superseded by the C1 native
experiment below. Continue with post-RMSNorm. Keep non-unit-step initialization as a separate
small follow-up. AIV is the next overlap-focused experiment once queue facts
are available. No new device speedup is inferred from the ratings or pair counts.

## What already exists

| Mechanism | Current implementation and remaining boundary |
| --- | --- |
| Reuse acquired completion | `SelectedGroups.cpp::consume` recomputes residuals after each actual acquisition and ranks provider coverage; identify why a specific desired source is unavailable before changing this policy |
| Reciprocal helper composition | `SelectedPlan.cpp::finish` tests removal of helper populations by engine pair using cold validation; useful existing behavior, but not direct avoidance of release channels or a free construction query |
| Guarded complete episodes | `CyclicFrontiers.cpp::qualifyGuardedBanks` admits exact cells with one writer/reader engine and unambiguous access-role boundaries; shared episodes are admitted only with no ordinary recurring requests and no authored synchronization |
| Optional specialization | Key exhaustion declines before ledger mutation; do not reimplement this as new work |
| First initialization | `NativeFirstUse.h` currently requires zero lower bound, unit step, and conjunctions of zero-equality tests |
| Retained completion | The causal frontier already transports acquired histories; a new phase summary must preserve this behavior rather than create another authoritative completion state |

The older `joinedCycle` release merge still needs an ordering certificate.
The new common-reader qualifier does not establish that every subsequent merge
or acquisition motion is order-preserving. Audit the complete pipeline.

## Sequence of work

1. Rebase the proposed opportunities onto current native output (C0).
2. Establish completion support at one attention AIC cut, then post-RMSNorm (C1).
3. Qualify AIV receive/reuse placement and preserve its tail hazard (C2).
4. Extend participating-use support and alternate routes only where C1/C2
   expose a missing fact (C3).
5. Use paired projections, merge/finalization, and quantization as distinct
   generalization tests (C4--C6); retain RoPE/staging as C7.

Coalescing hardening in the main TODO remains a prerequisite for claiming
general order preservation. Runtime optimization follows plan improvements,
with explicit work accounting throughout. Keep GEMM as a nonregression case.

## C0. Current-output attribution and corpus provenance

- [ ] Pin current source and prepared-input hashes. Read retained
  `projection-composition-work/projection-family-results.json` before repeating
  projection experiments; some helper excess may already be gone.
- [ ] Select post-RMSNorm, partial attention module 44, and single-block module
  48 as the initial distinct structures. Keep all six attention variants as
  regressions; verify claimed suffix-only equivalence before deduplicating work.
- [ ] At each targeted consumer record residuals, occurrence identities,
  requirement stage, available source coverage, selected transfers, and
  consumption evidence before and after acquisition. Classify missing imported
  semantics, lost correspondence, provider order, or genuinely missing coverage.
- [ ] Report static words separately from executed pairs, named barriers, and
  terminal ALL. State the exact reachability population in ordering comparisons.
- [ ] Verify model/target provenance. Keep refreshed Qwen A3 inputs separate
  from older DeepSeek A5 inputs and their target qualification.
- [ ] Retain the captured zero-trip DeepSeek RMSNorm loop as such; do not count
  its static words as executed steady-state overhead or repair its bounds.

Done when each next code change addresses a reproduced current residual or
placement defect. An already-covered case becomes a regression, not a new pass.

## C1. Required completion paths support other storage deadlines

- [ ] Represent support in existing requirement/frontier records: physical use,
  preceding reader frontier, participation, overwrite deadline, supporting
  transfer, source occurrence/scope, prefix coverage, and acquisition position.
- [ ] Distinguish structural conditional support from actual acquired credit.
  Select the supporting readiness/return cycle before the dependent fence
  decision; discharge demands only through selected commands or an explicitly
  supported and sealed recurring invariant. Reject circular justifications
  that omit all mutually supporting transfers.
- [ ] Post-RMSNorm: test whether reciprocal scratch release, followed by input
  readiness, already establishes retained normalization completion. Then test
  whether the required store-to-load return covers gamma's previous reader,
  writer, and readiness-key consumption before gamma overwrite/republication.
- [x] Attention AIC: reproduced the ordinary stage-ordering defect. The current
  imported path is `M -> FIX -> MTE2 -> MTE1`. Its required probability
  readiness now competes during Known selection when existing source coverage
  includes QK release. No new support-record type was needed for this case.
  Partial AIC 69 → 64 pairs; AIV 57 → 55 from the same general rule;
  single-block total 79 → 78. Recurring channels and named barriers unchanged.
  Full native causal/reconstruction checks pass, as do 126 bounded module-44
  paths with identical payload ordering and valid rearming. Device gain is
  unmeasured; details are in `../../../aic-completion-work/REPORT.md`.
- [ ] Keep input/gamma readiness separate and early. A missing private release
  may still require event retirement or rearming; account for both explicitly.

Acceptance: ordinary and guarded native fixtures pass final causal validation,
reconstruction, and independent emitted checks; no duplicate release is created
in the admitted case; no added payload order. Removing supporting readiness or
return must fail. An independent gamma reader must retain its own prerequisite.
For genuinely dependent in-place vector operations, retain required V fences.

## C2. AIV acquire output completion at its actual reuse deadline

- [ ] Establish receive-slot footprint, ownership/free lifetime, implicit
  resources, and cross-core queue semantics from the importer/lowering contract.
  A declared two-slot layout in the reference experiment is insufficient.
- [ ] If pop's destination is independent, keep output-release SET early and
  acquire it after pop but before score readiness carries completion to V.
  Check crossed selected words, acknowledgment timing, key occupancy, and all
  participating paths; no wait may cross its real access or protocol deadline.
- [ ] Retain a negative fixture for probability bytes `[65856,69952)` reused by
  tail maximum `[65856,65920)`. Deleting the return must expose the push-reader
  to tail-write hazard or missing rearming evidence.
- [ ] Compare under the full queue contract. Report which local-model order
  removals remain available after request/reply and collective edges are included.

Acceptance: receive may proceed earlier only with proved independent storage;
V's conflicting work remains gated, publication prefixes do not grow, and no
new event generation or payload relation is introduced. Pair counts may stay
equal; the improvement is placement and must ultimately be measured on device.

## C3. Support under different bank schedules and guarded participation

- [ ] Add the regular unequal-period construction as a bounded portable
  positive case: fixed fill order, one common reader, disjoint cyclic banks,
  fresh/qualified entry, no skipped episodes. Select the earliest-filled operand
  among the minimum-period operands. Preserve a previous receipt rather than
  move a later wait ahead of an earlier independent fill.
- [ ] Reject its use under arbitrary skips: original iterations 0 and 3 with
  3/2 banks reuse the three-bank family while the other selected bank is fresh.
- [ ] For guarded sharing, qualify an original co-use mapping such as
  `X_bank = f(gamma_bank)`, together with same-episode participation, reuse
  acquisition before gamma overwrite, and an actual downstream completion path
  covering gamma's last reader. Mapping alone is not a protocol certificate.
- [ ] Distinguish four interfaces: skipped complete episode; producer with zero
  readers; repeated readers without a producer; and a new physical use. Only
  the first is an identity on that bank's full interface. Preserve WAW on the
  zero-reader path and conjunctive readers on repeated-read paths.
- [ ] Compose direct `P -> Q` and update `P -> R -> Q` alternatives through a
  common outgoing assertion only when every arm realizes it. Preserve inactive
  route keys and consumption evidence; never union incompatible path credit.
- [ ] Partial attention: verify the reported active-interval guard reduction
  with signedness, division rounding, overflow, and original bound premises.
  Project it onto actual LEFT `[0,1,2,0,1,2,0,1]` and RIGHT
  `[0,1,0,0,1,0,0,1]` roles; RIGHT is not a global modulo-two schedule.
- [ ] Single-block attention: compose prologue/body/epilogue. Keep semantic row,
  operand-bank use, and queue generation distinct: PV0 can reuse QK2's bank,
  while QK3 completion must not be imported without a real requirement.

Acceptance: skipped/reentered episodes, nonidentity mappings, arbitrary valid
bank sequences, zero readers, mixed routes, and independent readers have both
positive and negative tests. No runtime history counter, global LCM expansion,
or product of all guards. Reference formulas such as `2N+r` are restricted
construction expectations, not assertions of minimality or native counts.

## C4. Paired projections: reload versus retained generation

- [ ] Qwen gate/up: extend original range/first-use proofs to positive non-unit
  steps where sound. Check reported `kb=0,2,...,18`, `inner=0,128`: tests against
  -1 or -64 are impossible; each active phase should have one initializer.
  Qualify the rotated block/phase guards, absent siblings, and ACC release
  across siblings. A reloaded activation creates a new physical use.
- [ ] DeepSeek KV/score: identify the same MAT X generation read in two children.
  Preserve readiness through both, release X after both reader frontiers, and
  release KV weights at their earlier last reader. Keep separate ACC buffers
  and operand staging lifetimes. Check target A5 import/lowering first.
- [ ] Add negatives for releasing X after only the first child, forgetting an
  actual reload, and advancing MAT release to an unrelated M completion.

Acceptance: generic range and cell/region lifetime rules; no model-name or
opcode-sequence recognizer. Accompany with down/q/kv projection regressions and
existing measured behavior; inspect current output before claiming any saving.

## C5. Partial merge and finalization: branch-sensitive receipt reuse

- [ ] Seed path: preserve Y readiness before metadata loads. Combining these
  publications must fail an ordering-quality test even when memory-safe.
- [ ] Update path: test whether the first required result receipt already
  covers earlier V-produced max/sum outputs; avoid their extra notifications
  using actual credit. The seed selector is an original invocation parameter.
- [ ] Finalization: test divide's joint input completion and any output-return
  support while preserving smaller views and writes within larger allocations.

Acceptance: distinct seed/update regressions, unchanged payloads, correct
physical alias extents, retained necessary V ordering, and no late readiness
coalescing. Do not interpret the reference 4-to-2 count as a compiler result.

## C6. Quantization: retained scale and aliased phase transition

- [ ] Qualify A5 scratch and reduction effects before applying local proofs;
  an operand listed as input is not necessarily read-only workspace.
- [ ] Preserve quantization scale `[256,512)` through the second pass. Require
  completion of scale store `[8448,8704)` before input overwrites
  `[8448,16640)`; transport retained-scale credit through the necessary return
  and input-readiness path.
- [ ] Track f32/i32/f16/i8 views as overlapping extents. Smaller writes cannot
  kill outstanding histories for untouched bytes.

Acceptance: eliminating an independently redundant scale notification passes;
moving the genuine store-return acquisition after the first load fails. Do not
generalize attention's independent-ingress placement to this aliasing case.

## C7. RoPE/cache and a minimal staging control

- [ ] Qwen RoPE/cache: enumerate independent final store readers, guarded
  outputs, retained inputs, and the next conflicting byte use. Reuse a return
  only when its actual path covers every due reader.
- [ ] DeepSeek qkv_rope_rows: qualify the shared 128-byte cosine/sine staging
  slot and repeated participation. Keep store-to-next-load handoff; semantic
  tensor names do not make the local slot disjoint. Avoid assuming scalar
  completion or rotation arithmetic from the kernel name.

## Quality model and computational cost

- [ ] Treat the proposed opaque-prefix ordering lower bound as a theory task
  for the paper agent. Specify permitted primitives, opaque operations,
  synchronous/native exceptions, control, and source issue order before using
  it as an oracle. The inclusion/attainability claims are not established here.
  Never add derived demand edges as actual selected completion credit.
- [ ] Distinguish raw access requirements, primitive-imposed ordering, and
  avoidable placement/resource ordering. Continue using device profiles:
  payload reachability alone missed compact MAT's same-pipe drain cost.
- [ ] Bound admission by represented access/region incidences, selector
  relations, selected support edges, and emitted words. Index common sources
  and deadlines; avoid repeatedly enumerating all requirement/provider pairs.
- [ ] Charge selected replay and final validation separately. Do not enlarge
  the existing finish-time helper omission into a general refiner. The desired
  next mechanism avoids unsupported private channels before key binding.
- [ ] Keep finite trace enumeration and all-pairs ordering checks in tests.
  Report sites, logical/physical channels, updates, replay visits, cold checks,
  wall time, and memory on native fixtures; no end-to-end linearity claim from
  a small descriptor or arithmetic rule.

All implementation milestones require unchanged independent final acceptance,
native reconstruction/lowering, focused negative tests, and the device-qualified
GEMM nonregression. Broader corpus and device campaigns follow a demonstrated
native plan improvement; this planning update runs neither.
