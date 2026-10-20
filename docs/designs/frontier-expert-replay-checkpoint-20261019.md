# Expert baselines and outer replay checkpoint — 2026-10-19

The compact-K repair remains accepted and unchanged. This checkpoint fixes a **test-probe absence assumption**, adds focused regressions and diagnosis, and stops at the unsupported replay interface. It does not change allocation, expert flags, demand selection, expansion limits, or fallback policy.

## Scope and pins

Source base: `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44`, branch `codex/frontier-shared-input`, with the accepted compact-K working changes and this checkpoint's diagnostic/test changes. The companion JSON pins the working source files, executables, prepared inputs, source snapshots, commands and receipts. A source revision alone does not identify this dirty working tree.

Target: A3; cube for Persistent, vector for GroupNorm/cumsum. GM policy: `may-not-alias`, supplied as disjoint argument allocations; repeated uses of the same argument can overlap. IDs remain `0,1,2,3,4,5`. Commands run serially; incremental builds use at most two workers. There is no device compilation or execution.

## A. Expert-plan witnesses

These are minimal **required-graph cover** witnesses, not just missing transitive pairs. Extraction starts from the prepared port's supplied f32 layouts and original operation coordinates, independently of PTOAS cells, generators or reducers. The command graph uses the existing independent matching/causality checker and explicitly includes supplied scalar-value readiness edges. It is not independent validation of that graph's native instruction-issue assumptions.

| Port | Minimal uncovered requirement | Exact modeled overlap | Authored synchronization |
|---|---|---|---|
| GroupNorm, block/subblock 0/0 | Completion of payload 4, `TMOV(bank[0], data)`, to start of payload 5, `TADD(accum, data, accum)`; event pair `(9,10)` | UB `[32928,37024)`, a complete 16×64 f32 data tile | ALL before the loop and after the four vector operations; none between this copy and add in prepared IR |
| Cumsum, two chunks of three scalar elements | Completion of payload 17, `TSETVAL(s[5])`, to start of payload 18, full-buffer `TSTORE(s)`; event pair `(35,36)` | UB `[4116,4120)`, one f32 value within output `[4096,8192)` | Final V→MTE3 SET/WAIT observes V work; no modeled S→MTE3 readiness path |

The GroupNorm witness does not depend on reduction scratch overapproximation, compact reduction aliases, unresolved GM aliasing, or a tail mask. The cumsum witness is likewise a four-byte concrete conflict. Cumsum's semantic derivative changes the chunk count 8→2 and per-chunk length 128→3 while retaining the 1024-element tile and final full-buffer store. It is not the original full-size execution.

The separate cumsum reuse witness is `(V,S,id=0)`: command 3 is the first WAIT at payload gap 2; command 4 is the next SET at gap 10. There is no consumption→republication path in the modeled graph, **even with the supplied scalar-value prerequisites included**. Textual live-flag occupancy and SET/WAIT counts alone do not establish this path. GroupNorm's checked expert flags pass the existing causal-reuse check.

### Source and lowering attribution

Both original snapshots are pinned to TileLang-Ascend `cbc67f2697e8ee18357adc447272691432b3948d` by their existing `port.json` files.

* GroupNorm source: `test/npu_validation/benchmarks/auto_sync/sources/tilelang-ascend/examples/cann-bench/group_norm/example_group_norm.py.txt:78–97`. For the selected FP32 specialization, source lines 93–96 copy/add/multiply/add, followed by ALL at line 97. `port_groupnorm.py:32–48` retains that ordering. The port fixes disjoint UB banks and compact column/row aliases; source memory planning is not reproduced as native compiled code.
* Cumsum source: `.../linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_cumsum.py.txt:37–50`. The V→S handoff is inside the outer chunk loop; the V→MTE3 handoff and full store are outside it. `port_vector.py:61–88` preserves this placement. Scalar array operations become PTO `tgetval`/`tsetval`, emitted as `.GetValue`/`.SetValue` by `lib/PTO/Transforms/PTOToEmitC/ScalarMisc/ScalarMiscOps.cpp:86–133`.
* Fresh C++ emission of each full-size expert port succeeds with synchronization insertion disabled. The emitted GroupNorm copy and add remain adjacent `TMOV`/`TADD` calls. Cumsum retains `.SetValue` followed eventually by V→MTE3. **C++ source generation is not native compilation.** This experiment does not recover either upstream TileLang's generated native program or its native compiler options.

**Classification:** the authored-source/prepared-port baseline boundary remains underqualified. No changed synchronization placement was found for these witnesses. The source's `TL_ASCEND_AUTO_SYNC=False` does not establish the behavior of helper lowering or the native CCE compiler. It is therefore unjustified to label either original kernel miscompiled, or to add blanket V hardware protection to the oracle.

Official [PipeBarrier documentation](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0271.html) requires synchronization for dependent V operations and treats scalar synchronization separately. The [Ascend C synchronization introduction](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha003/API/ascendcopapi/atlasascendc_api_07_0179.html) also describes compiler-inserted V and cross-pipe synchronization, subject to dependency information. These are reasons to qualify the compiler boundary, not evidence that a particular unexamined native compilation inserted the needed edge.

The missing native evidence is precise:

1. For GroupNorm, obtain the pinned TileLang lowering and native compilation options/output: does the copy/add lower with a V barrier, or another documented instruction-specific completion/readiness guarantee?
2. For cumsum, establish scalar UB-store visibility before the DMA through a documented instruction contract or an actually emitted S→MTE3 dependency.
3. For cumsum reuse, establish whether a WAIT targeting S constrains subsequent issuance of a SET on V, or whether lowering introduces an S→V acknowledgement. The current per-pipe event graph does not supply that command-issue edge. A blanket textual command-order edge is not an acceptable substitute.

Until then both expert ports are **unqualified comparison baselines under the supplied model**. Their protocols are unchanged. GroupNorm's generated-plan allocation refusal is a separate missing certificate (`matching source-tuple arity`), not an explanation of these expert witnesses or a capacity lower bound.

## B. Outer replay diagnosis

Derivative: two waves, K=2, kk=2, core 0, wrapped in an explicit epoch loop with 0/1/2/3 visits. Every epoch resets wave/K/kk and reuses the original buffers and the same GM arguments.

The ordinary production attempt retains the wave child as a phase/numerical regional composition: **124 types**, **107 storage cells**, **three symbolic GM effects**, query/selector capabilities present, no deferred boundary entries. Its K child retains a **28-type** phase interface. A separately pinned mathematics-only diagnostic request for original regions 4 (wave) and 11 (K) confirms these selected child representations; this is additional diagnostic work, not inspection cost or production timing.

Coordinate frames are original, not flattened execution ranks:

* Payloads carry `(epoch, wave)`, `(epoch, wave, k)`, or `(epoch, wave, k, kk)` according to their original loops.
* Inside the wave child, epoch is enclosing context. Its retained phase frame represents `wave = 4*n + phase`; K's phase frame represents `k = 2*m + phase`, with clipped startup/tail activity. kk is retained by the finite slice's original coordinates.
* The epoch parent would prepend its own visit coordinate, as `exportRepeatedRegion()` does. It does **not** reach that successful export; no wrong-coordinate answer was observed.

The exact failure chain is:

```
SequenceAnalysisState::repeatedChild(epoch)
  → originalRegion(body): accepted wave child and exact local interfaces
  → repeatInvariantRegion
  → repeatSymbolicStorageRegion
  → materializeRepeatedSymbolicStorage
  → selectorSupportRanges(body) = unavailable
```

Relevant code: `Shared/RepeatedRecognition.cpp:147–193`, `RepeatedRegionExports.cpp:51–70`, `RepeatedReadOnlyStorage.cpp`, `RepeatedFiniteStorage.cpp:16–103`, `RegionalSelectorPartition.cpp:15–77`, and `RepeatedPhases.cpp:169–193,592–624` under `lib/PTO/Transforms/FrontierSynch`.

This is **missing complete GM selector-support construction**, before querying byte membership or hitting a demonstrated atom-count limit. The old composite diagnostic mentioning the expansion limit did not distinguish these causes. New read-only debug messages do. For discharged residual GM maps, phased export deliberately refuses an exact selector in the same address domain. A local-selector capability is not a proof that every GM query is answerable.

The GM issue is real under the supplied access model: output address depends on wave/core but not epoch. The same `%C` rows are written again after a reset. The disjoint-writer-visit discharge in `Shared/Recognition.cpp:122–202` is valid within a wave traversal, but is not a certificate for those overlapping epoch visits. A full write/write requirement may already follow from other required paths; overlap alone does not prove a new nonnative cover must be emitted. It still cannot be discarded without a proof. A/B remain read-only under the supplied disjoint-argument assumption.

A forwarding-only repair is not justified: no retained exact output-byte selector or complete uniform GM partition is available to forward. The next construction would need to retain row-strided GM membership and first/last output writers through the phased wave interface, then certify cross-reset crossings with the original coordinates and guards. That requires a separate implementation/qualification step; this checkpoint does not invent that construction, raise limits, or replace child mathematics with fallback.

### Demonstrated checker correction

The test probe previously treated “no storageSelectors callback, no matching materialized cell” as an empty selector answer. An explicit GM-output query exposed this unsupported assumption. `CompactRegionContractChecks.cpp` now rejects same-domain unresolved/deferred effects in that fallback, using the existing alias policy. Existing production selector refusals are unchanged. The output-byte query `(GM,arg2,0)` now reports **unavailable**, rather than a false empty boundary. This is a probe defect, not a demonstrated production demand defect.

### Stage outcomes

| Case | Exact mathematics/interfaces | Logical insertion | Physical allocation | Generated output |
|---|---|---|---|---|
| Replay 0 | Empty result; checks pass | Fresh success | Fresh six-ID success; matching/reuse pass | Fresh C++ source success |
| Replay 1 | Local contracts pass; GM output selector unavailable | Fresh success | Fresh six-ID success; matching/reuse pass | Fresh C++ source success |
| Replay 2/3 | Child mathematics retained; complete root result unavailable even for a separate mathematics-only request | Existing ordinary fallback can emit conservative completion; **not counted as exact replay success** | Exact replay not attempted | Exact replay not attempted |
| GroupNorm generated plan | Earlier exact logical result retained | Earlier success | Earlier matching-arity certificate refusal; unchanged | Fresh expert-port C++ source emission succeeds; generated physical C++ not established |
| Cumsum generated plan | Earlier exact logical result retained | Earlier success | Earlier six-ID success | Fresh expert-port C++ source emission succeeds |

Earlier vector logical/allocation outcomes come from `.local/compact-demand-repair/accepted-generalization/results.json`; they are not new runs of those stages. Fresh commands and outcomes are pinned in this checkpoint's JSON. None is device qualification.

## C. Checks and reproducibility

Run from `workspaces/frontier-shared-input`, using the pinned source-snapshot build:

```sh
PROBE=.local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test
OPT=.local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-test-opt
.venv/bin/python test/lit/pto/Inputs/check_expert_protocol_witnesses.py --trace-tool "$PROBE" --prepared-root test/npu_validation/benchmarks/auto_sync/prepared --output .local/expert-replay-checkpoint/expert-witnesses
.venv/bin/python test/lit/pto/Inputs/check_compact_replay_checkpoint.py --trace-tool "$PROBE" --input test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto --output .local/expert-replay-checkpoint/replay-contracts
.venv/bin/python test/lit/pto/Inputs/check_compact_persistent_contract.py --optimizer "$OPT" --trace-tool "$PROBE" --input test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto --output .local/expert-replay-checkpoint/persistent-contract --static-stages
.venv/bin/python test/lit/pto/Inputs/profile_compact_persistent.py --optimizer "$OPT" --input test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto --output .local/expert-replay-checkpoint/scaling --sweep trips --repetitions 1 --require-compact
```

The replay regression checks zero/one explicit visits independently: one visit has 34 active payloads, 29 nonnative covers, 8,464 four-kind event-query comparisons (including absent potential occurrences), and 111 local-byte selector queries. Multiple visits deliberately report missing exact capability. They are not oracle-passing repaired cases. The GM-output negative regression protects the distinction between missing and empty selectors.

The expert regression extracts the two concrete overlaps and pins the cumsum reuse witness including scalar prerequisites. Both tests are in `sync_expert_replay_checkpoint.pto`. The focused lit gate passes **8/8**, including the accepted compact-K, static-stage, regional storage/sequence and allocation-profile checks. The full compact semantic rerun records **10 passing cases and the same one unresolved replay case**. The production K sweep (2/4/8/16/32, one serial sample each plus separate profiling) passes its structural gate: K≥4 retains the **332-type / 14-largest-finite-body** signature. The zero-inner-body case retains its two extra C→I prerequisites from existing MTE2 barriers. These samples do not establish a speedup.

No full corpus or device campaign was run. No timing ratio or native performance improvement is claimed. Scoped Ruff checks pass; changed-code compliance's syntactic brace/weak-pointer-lock warnings are inspected separately from a native build/test verdict.

## D. Paper consequences

| Established result | Remaining premise or limitation | Appropriate manuscript claim |
|---|---|---|
| Compact K survives composition and accepted finite checks/scaling | Supplied NZ layouts and GM disjoint-argument premise; measured derivative family | A demonstrated compact Section 5.4→5.5 path for the qualified pipeline, not arbitrary nested programs |
| One replay visit agrees on actual covers, four event kinds and local selectors | Repeated GM writer selectors/support are not exported | Exact composition requires the interfaces it consumes; child mathematics alone is insufficient |
| Repeated output accesses overlap across epoch resets | Complete cross-reset reduction/export not constructed | A reset is not a fresh storage lifetime or a disjointness certificate |
| Prepared expert commands fail the supplied event model | Upstream lowering, native compiler synchronization and S command-issue premises unresolved | Compare expert/generated plans only after qualifying the same payload/storage/native contract; no native defect conclusion |
| Missing GM selectors now refuse in the test probe | Native instruction footprints remain independently unqualified | An unavailable boundary answer must not be interpreted as absence |
| Zero-inner-body coverage remains correct with two extra C→I pairs | Existing barriers strengthen selected payload order | Coverage and minimum demands do not establish realization-order equality |
| Existing six-ID Persistent allocation succeeds | Feasibility of that protocol, not minimum capacity | Keep resource feasibility separate from Section 5 minimum-demand construction |
