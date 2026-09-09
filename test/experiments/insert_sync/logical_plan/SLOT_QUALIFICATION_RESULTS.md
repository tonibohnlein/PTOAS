# Slot qualification: bounded native evidence

This work checks occurrence-dependent slot conflicts and the reconstruction of
emitted guard bindings. It does **not** complete M2: the rebuilt twelve-case
population has two timeouts, an unsupported boundary construction, and an
unsuccessful event assignment. No hardware correctness or timing was tested.
The production work allowance is unchanged.

## Original twelve cases

The campaign started with the accepted M1 physical-address implementation at
`cdaae8d0f95c6dd6693b89f6fe22f262467bfb90`. Eight completed rows were preserved
when the user paused the work. The remaining four were then run on the same
frozen binary, without rerunning the first eight. Its recorded SHA-256 is
`930584475df644eef82e8a1e11e0a9a13055d0339b9c19dcbab5bbec232b2096`.

| Unchanged case | Strict construction | Evidence |
| --- | --- | --- |
| Rotating depth two | Applied | Exact occurrence requirements, payloads, event participation, and independent readiness |
| Rotating depth three | Timed out at 90 s | No completed construction proof |
| Equivalent depth-two mask | Applied | Same semantic checks as depth two |
| Explicit-base depth two | Applied | Same semantic checks as depth two |
| Shifted overlapping mapping | Unsupported boundary lowering | Exported requirements checked; no emitted construction |
| Reversed overlapping mapping | Allocation failure | Exported requirements checked; no emitted construction |
| Additional final reader | Timed out at 90 s | No completed construction proof |
| Common consumer | Applied | Both productions are required; no independent-readiness reduction claimed |
| Skipped first reader | Timed out at 90 s | No completed construction proof |
| Unknown selector | Applied | Conservative possible conflicts retained |
| Out-of-range selector | Applied | Conservative possible conflicts retained, including slot-zero fallback |
| Negative selector | Applied | Conservative possible conflicts retained, including slot-zero fallback |

Thus **7/12** cases have accepted strict constructions. Two further cases have
checked discovery only. Neither discovery nor a successful fallback counts as
constructor effectiveness. The three timed-out runs preceded the new atomic
discovery sidecar, so their incomplete processes provide no discovery result.

Raw artifacts are in the parent workspace under
`insertsync-builds/campaign/logical-plan/slots-m2/{initial12,remaining4}`.
Each row retains the input hash; `manifest.json` fixes the original twelve inputs.

## Rebuilt twelve-case campaign

After the bounded exact guard-query changes, all twelve original inputs were
rerun serially. Raw results and the exact binary/runner hashes are under
`insertsync-builds/campaign/logical-plan/slots-m2/exact-guards-original12`.
The runner correctly exits nonzero: this is not a passed milestone campaign.

**8/12 strict constructions now pass their emitted checks.** The additional
reader case completed in 87.65 seconds on this run, rather than timing out.
Rotating D3 and the skipped-first-reader case still reach the 90-second limit.
The shifted mapping still declines boundary lowering and the reversed mapping
still fails allocation. There are no dependency-oracle mismatches in any of
the twelve cases: all four unfinished constructions now have checked immutable
discovery sidecars, including the two timed-out processes.

This separates three results: twelve checked dependency discoveries, eight
checked emitted constructions, and an **unmet compile-cost gate**. The newly
completed extra-reader case remains too expensive and is close to the runner's
limit; one completion is not evidence of reliable performance admission.

## Focused native guard and overlap checks

Three small fixtures supplement the original population. They retain actual
native SSA selectors and four physical operations per loop iteration. Fixed
four-trip bounds keep these tests suitable for the mandatory serial gate.

| Fixture | Property challenged |
| --- | --- |
| `slot_guard_binding.pto` | Depth-two selectors and generated guard operands, with an otherwise unused dominating index argument |
| `slot_guard_known_parameter.pto` | The same loop under a condition using an original index parameter; both skipped and executing paths |
| `slot_partial_overlap.pto` | 512-byte VEC footprints with mappings `[0,512]` and `[256,768]`, producing genuine 256-byte overlap |

The final rebuilt focused run constructed all six selected inputs: these three fixtures
and the unknown/out-of-range/negative selector cases. The three positive
fixtures respectively exercised 224, 112, and 68 real local conflict occurrences
across their bounded replay scenarios. The partial-overlap fixture included 36
partially overlapping conflicts, two first readers that required the second
preload, and two that remained independent of it. Its requirements followed
physical byte overlap rather than equality of slot numbers.

The conservative negative cases retained respectively 351, 219, and 189 extra
possible-conflict occurrences in these scenarios. Those are deliberately
conservative obligations, not false claims of precise matching generations.

Three corruptions affect **only a newly emitted comparison operand**:

| Corruption | Fresh reconstruction rejection |
| --- | --- |
| Replace an induction-variable guard operand with the original modulo selector | Emitted publication lacks acquisition |
| Replace it with an otherwise unused dominating index argument | Emitted occurrence bindings differ from input |
| Replace it with an already represented original parameter | Emitted acquisition lacks publication |

For every corruption, the test requires unchanged original operation operands,
unchanged synchronization counts, rejection of the candidate, and byte-identical
preservation of the input function. A rejection based only on the original
payload snapshot does not pass these guard tests. Original operations are
captured from the candidate clone in the pre-emission observer.

The focused runner also compares the new native boundary-condition operations
against general exact integer subtraction. Cases include an original guard,
negative loop bounds, and a non-unit step. The hook checks repeated cached
queries, conditions at distinct ambient points, integer extrema, and residue
conditions. Its first run exposed a local-variable relation-space assertion on
the guarded fixture. The relation union now has a local-free space, while each
piece retains its exact floor definitions. The failed run remains archived.
All three hook inputs pass after rebuilding, including the guarded and
negative/non-unit cases.

The final mandatory `oahs_focused` gate passed in **61.52 seconds**. Its slot
subsuite made 13 serial native invocations: six strict constructions, three
generated-binding corruptions, three boundary-condition challenges, and one
traced early refusal. The latter confirms that unsupported physical facts are
reported without dereferencing an uninitialized order provider. The full gate
also passed the existing relation, occurrence, scalar, mapping, physical-address,
scalability, reconstruction, observer, constructor, retirement, and authored
synchronization checks.

Final artifacts are in build
`test-results/oahs/focused-ud5hobed`, with slot details in `slots/summary.json`
and `slots/commands.json`. The final native driver SHA-256 is
`c9f439eb75ea464c791c1f142098031fc00db1f8167ae92e49166f763d75cad9`.
The slot runner's `provenance.json` records its own hash, and every case records
the unchanged input hash. These are pre-commit worktree results. They accept
this bounded qualification checkpoint; they do not change the original
twelve-case coverage denominator or complete M2.

## Test boundaries and reproducibility

The requirement oracle interprets the original scalar selectors and actual
physical intervals, enumerates local RAW/WAR/WAW occurrences, and compares them
with exported symbolic relations using libisl. Actual emitted replay checks
payload identity, every represented local order, matching event consumption,
terminal completion, and the independent second-preload boundary. It does not
simulate hardware operation latency or numerical payload execution.

The driver writes a complete discovery sidecar atomically before expensive
planning. Completed runs compare it exactly with their final fact export. A
timeout can therefore retain checked discovery, while remaining a failed
strict-construction outcome. The runner records failures independently and
supports pausing between cases without erasing completed rows.

Use the established Python 3.12 environment and pinned libisl:

```sh
python test/experiments/insert_sync/logical_plan/check_slots.py \
  --driver "$BUILD/tools/pto-test-opt/pto-logical-sync-test" \
  --python-root "$BUILD/python" --output "$NEW_RESULTS/slots" --focused
```

The mandatory `oahs_focused` gate includes this bounded suite. A full twelve-case
rerun omits `--focused`; it must retain unavailable outcomes separately. New runs
record the driver and runner hashes before launching native work, so even an
aborted campaign retains its provenance. The original twelve-case population,
unchanged two-/three-buffer controls, compile-cost qualification, and subsequent
allocation/composition milestones remain separate acceptance gates.
