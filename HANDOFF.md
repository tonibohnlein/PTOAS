# OAHS current handoff

Updated: 2026-09-19

## Checkout

- Repository: `/home/toni/work/pypto3_sync_more/PTOAS-oahs-clean-m1`
- Branch: `codex/oahs-clean-m1`
- Base before the current milestone: `d6c5731279229f369579ab1dd69634f3a49bffa7`
- Current milestone: post-RMSNorm required output-return composition

Verify the branch, HEAD, and working tree before continuing. A newer user commit supersedes this record.

## Current local milestone: post-RMSNorm output-return composition

Base: `d6c5731279229f369579ab1dd69634f3a49bffa7`. This milestone is recorded
in the commit containing this handoff. The previous attention device task
remains pinned to the base commit.

Regeneration confirmed that the provider-selection change alone leaves
post-RMSNorm unchanged: gamma's private return is preinstalled by recurring
qualification. Declining all recurring channels removes that return but adds an
MTE2 barrier, so that diagnostic was not adopted.

`CyclicFrontiers.cpp::qualifyPipelineCycle` now prepares a restricted three-engine
write / in-place-work / final-reader cycle before ordinary repair. It preserves
the early input and gamma readiness words. The required store-to-load return
carries gamma reader/writer completion and readiness consumption, so gamma does
not receive its own release channel. The same actual credit preserves the
normalization result established through initial scratch release.

Admission requires exact cells, period one and straight same-occurrence middle
work; it declines unrelated owner protocols and fixed/authored words. It does
not generalize guarded or unequal-bank sharing. Physical key assignment and
final causal/reconstruction checks remain unchanged. There is no graph
expansion or new omission trial; genuine vector barriers remain.

Modules 22/23: **16 -> 10 static pairs**, **10 named barriers unchanged**, one
terminal ALL. Two outer entries with 20 iterations in each phase execute
**360 -> 246 pairs**, with all barriers unchanged. Forty-eight finite paths /
1,017,762 conflict checks pass; payload finish-to-issue relations are identical.
The graph remains 175 sites; selected updates 37 -> 22, replay visits
21,292 -> 11,773, recurring channels 2 -> 4, zero omission trials.

Validation: 20/20 portable suites; support/deletion/independent-reader/retained-
value regressions; native regression; both RMS variants construct, reconstruct
and lower. All 87 corpus modules construct/reconstruct: 84 byte-identical,
two RMS improvements, one RMS sample changes only event-key numbering. All six
attention plans are identical. GEMM remains 182/360/716 pairs, zero named
barriers, one terminal ALL, with emission/FileCheck/trace checks passing.

Details, scope and reproduction: `../rms-completion-work/REPORT.md` and its
paired native plans, logs and ordering comparison. Device benefit is unmeasured.
Next: qualify this mechanism on device separately
from the already-dispatched d6 suite; then return to AIV placement (C2) once its
receive-slot and tail-storage contracts are established.

## Committed milestone: probability readiness supports QK release

Base is the committed guarded-episode/hardening milestone `a6bbe1ff8`.
The committed constructor change is in `SelectedGroups.cpp::groups`.
It lets an independently required Overlap provider compete during Known
selection when its current selected prefix already covers a Known requirement.
Existing Known-source prefixes remain separate from that source's later
Overlap requirements; actual selection/replay supplies completion.

The module-44 QK-to-PV overwrite previously selected direct M-to-MTE1 release
before considering the required probability-readiness receipt. The latter
already carries QK completion through the current selected M-to-FIX,
FIX-to-MTE2 and MTE2-to-MTE1 path. It now discharges the reuse requirement
without creating the duplicate return. No recurring population, graph expansion,
new refiner, queue contract or hardware assumption was introduced.

Emitted static results:

- Partial attention modules 44–47: AIC **69 → 64 pairs**, AIV **57 → 55**;
  total **126 → 119**, named barriers unchanged at 87, two terminal ALL barriers.
- Single-block modules 48–49: **79 → 78 pairs**, 26 named barriers unchanged,
  two terminal ALL barriers. Analytical endpoint counts exceed emitted counts because
  complete identical guarded words share their emission.
- Shenggan GEMM unchanged: **182/360/716 pairs**, zero named barriers and one
  terminal ALL for one/two/four tiles.

Validation: 20/20 standalone suites; extended focused three-engine regression
including missing-support negatives and 0/1/2/4 visits; native regression
driver; all six attention constructions/reconstructions and A3 level3 C++
lowering; GEMM production emission, FileCheck and independent trace checks.
Module-44 finite comparison checks 126 paths / 90,699 conflicts, valid memory,
balance and rearming, and **identical payload finish-to-issue relations**.
Its imported local-effects model does not replace device/cross-core qualification.

AIC module 44 keeps 407 sites / three recurring channels / zero recurring
trials. Selected updates fall 66 → 61, replay-site evaluations
111,095 → 102,810. Device performance of this change is unmeasured.

Artifacts and reproducible diagnostics:
`../aic-completion-work/REPORT.md`, `corpus-summary.json`, `compare.py`,
`ordering-comparison.json`, paired baseline/candidate logs and plans,
all six emitted/lowered attention plans, and test logs.

### Remaining roadmap

The reviewed studies and rated choices are in
[`docs/designs/oahs-composition-roadmap.md`](docs/designs/oahs-composition-roadmap.md).
Downloaded source notes were checked against 18 archived hashes, 329 attention
excerpt lines and the post-RMSNorm input hash; evidence is preserved in
`../composition-review-20260919/`. Their reference-model counts are not native
compiler measurements.

Post-RMSNorm attribution and the required additional cycle mechanism are
recorded above. Keep pending attention device qualification separate from this
unmeasured local change. AIV receive placement (C2) still
requires queue-slot/tail-storage qualification; it was not implemented here.
Guarded use correspondence (C3) and the broader shared-release ordering
certificate remain open. Read retained projection results before repeating work.

## Guarded attention milestone

The constructor now derives complete guarded producer/reader episodes for exact
canonical cells before assigning physical event keys. Cells with the same
guarded first-reader episode keep separate early readiness publications and may
share one identical reader-release publication. The shared release uses the
earliest comparable reuse acquisition. This is admitted only when ordinary
recurring qualification produced no requests and no authored/fixed
synchronization is present.

The implementation does not reserve one recurring release channel per exact
cell and does not use omission trials for the new guarded channels. If the
specialized channel population cannot fit the directional key pool, it declines
before mutating the ledger and ordinary demand-driven construction continues.

Review hardening in the same milestone:

- local fences are selected only when their actual occurrence deadline is
  reached; future analytical copies are no longer permanently repaired from an
  unfinished checkpoint;
- first-use descriptors reject a true arm that can reach another listed
  decision before a tracked backedge or exit;
- the older nested shared-release merge requires both operands to be classified
  as storage releases. Its broader ordering certificate remains open work.

For attention modules `pypto_lib__prefill_fwd__44` through `__47`, the generated
plans have 126 SET/WAIT pairs, 87 named barriers, two terminal ALL barriers,
three cube recurring channels, zero vector recurring channels, and zero
recurring omission trials/analysis sites. Modules `__48` and `__49` remain at
79 pairs, 26 named barriers, two terminal ALL barriers, and their six existing
periodic channels on each participating function. All six construct,
reconstruct, and lower successfully.

The independent module-44 check covers 126 paths and 90,699 conflict checks. It
reports zero added payload-order relations, 347 removed path-local relations,
and absence of the designated previous-bank-compute to next-bank-fill edge.
This is host evidence only; device latency is pending.

The exact Shenggan GEMM regression remains unchanged at 182/360/716 pairs for
one/two/four tiles, zero named barriers, and one terminal ALL. The full
standalone OAHS suite passes 20/20.

Device instructions and the reviewed source patch are outside the repository:

- `/home/toni/work/pypto3_sync_more/guarded-attention-work/DEVICE_TASK.md`
- `/home/toni/work/pypto3_sync_more/guarded-attention-work/guarded-episodes.patch`
- patch SHA-256: `8ed4a0169ef66102f50e79c37b7ad93dc85b43ea96f9d1cc73b9aa60c28454f9`

## Delivered local synchronization plan

The current implementation composes three direct constructor mechanisms:

1. A finite physical-bank occurrence interface carries the outer MAT bank residue across the nested reader region. It relates each overwrite to the previous participating use of the same bank without expanding a product of nested first/middle/final modes.
2. Two operands in one qualified physical-bank episode retain separate early readiness transfers but share one exact storage-release return. This preserves early A extraction and removes the redundant second MAT release channel.
3. A first-use prefix qualifier recognizes the original conjunction `outer_k == 0 && inner_k == 0`. It splits one entry prefix, shares the remaining graph, and removes the impossible repeated ACC-initialization paths. It supplies no completion credit and retains conservative behavior for incomplete conjunctions, disjunctions, and unsupported loop forms.

Exact qualified cycles bypass whole-plan omission trials. Local fences are now
decided at their actual occurrence deadline; the hardening removed eager
batching over future analytical copies. Canonicalization still shares commands
in a genuinely shared word. The GEMM construction figures below are the
recorded GEMM milestone measurements, not a new whole-corpus cost bound.

## Exact Shenggan result

Generated artifact:

- `/home/toni/work/pypto3_sync_more/bank-occurrence-work/shared-release-first-use.pto`
- SHA-256: `3f1e19bf600e137685e0fdd4a49b5ea8ec4e19f1bf53f375618190ea2dfab8f2`
- construction log: `/home/toni/work/pypto3_sync_more/bank-occurrence-work/shared-release-first-use.log`

For identical concrete payloads:

| Tiles | Event pairs | Named barriers | Terminal ALL | Added / removed vs compact MAT | Added / removed vs manual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 182 | 0 | 1 | 0 / 446 | 0 / 16 |
| 2 | 360 | 0 | 1 | 0 / 928 | 0 / 32 |
| 4 | 716 | 0 | 1 | 0 / 1,892 | 0 / 64 |

The manual protocol has 167 event pairs for one tile. The 15-pair difference preserves separate early A/B readiness; the independent command-graph comparison finds the generated plan to be an ordering subset of the manual plan on all checked traces. Event count alone is not a performance estimate.

The one-tile direction counts are:

```text
M -> MTE1      66
MTE1 -> MTE2   18
MTE2 -> MTE1   32
MTE1 -> M      64
FIX -> M        1
M -> FIX        1
```

The trace checker validates one, two, and four output-tile entries. It checks memory conflicts, native ACC ordering, event occupancy, consumption before republication, final balance, same-bank prefetch, separate early A readiness, and the absence of current-bank compute ordering before different-bank preparation or parent DMA. Its injected terminal drain remains a discriminating negative control.

## Construction cost

The exact run reports:

```text
sites                         718
selected updates                3
replay site evaluations     21,846
recurring channels             10
recurring trials                0
recurring analysis sites        0
construction time          ~0.46 s in the recorded run
peak RSS                    ~112 MB
```

The first-use qualifier adds 67 analysis sites to the 651-site bank graph, but does not create a nested mode product. The expensive recurring omission population is gone. Construction time remains suitable for corpus and device qualification; repeated immutable structure construction remains a later optimization target.

## Validation completed

- standalone OAHS CTest: 20/20;
- portable bank occurrence tests: guarded and repeated entry, two and three banks, non-identity bank mapping, malformed boundaries, and exact shared-release structure;
- portable first-use tests: 145 concrete traces, zero trip, repeated entry, malformed boundaries, and genuinely repeating initialization negatives;
- native selected driver, including first-use conjunction and negative variants;
- exact native construction and reconstruction;
- exact Shenggan lit test: frontend lowering, FileCheck, and independent trace checker;
- `git diff --check`.

Sanitizers and device execution were not run locally.

## GEMM device qualification

The current 182-pair, zero-named-barrier GEMM plan is device-qualified. Across
180 samples it measured 229.306 us median at 299.7 TFLOPS and 0.921 MAC ratio,
compared with 230.548 us for the reconstructed manual protocol, 358.570 us for
compact MAT, 391.834 us for the earlier handoff, and 534.956 us for existing.
All 20 correctness runs passed with identical error ratios across the five
arms. The measured result is parity with the manual protocol, with a small
0.54% median advantage in this campaign.

Archive: `/opt/pypto/oahs-gemm-16564fa8a.tar.gz`, SHA-256
`9f71dc1fe129da538fe16ecebbd73d12410185b8a9176f91e512286e2c9dd25d`.

## Remaining work

1. Device-qualify guarded attention episode composition. Modules 44--47 require
   authentic runtime scheduler state; modules 48--49 are the primary runnable
   cases.
2. Attribute current post-RMSNorm/AIC residuals and select one native completion
   support experiment; follow the composition roadmap for AIV and other families.
3. Restrict broader recurring endpoint coalescing with an ordering certificate.
   Optional specialization now declines cleanly under key pressure.
4. Reduce repeated immutable control/storage construction after plan quality is settled.
