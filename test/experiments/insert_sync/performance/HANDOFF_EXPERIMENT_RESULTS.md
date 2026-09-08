# Forward supply / backward needs: native experiment

2026-09-08, `862ab8111` plus this experimental worktree change.
Native `pto-test-opt` SHA-256:
`7ba0ab0ab9f16a2071699556b6ab69777623062da10283d22c8a45b38c7ef823`.
The binary fingerprint was unchanged across the campaign.

## What is implemented and demonstrated

An explicit `--pto-experiment-handoff-planning` development pass consumes the
existing shared storage/frontier snapshot. Backward needs retain their original
consumer deadlines and follow actual forward completion through publications
and acquisitions, including transitive supply.

The current constructor handles linear, one-shot handoffs. It can advance a
publication, delay an acquisition or split a broad readiness handoff. Each
accepted trial preserves original payload and requirements, event validity and
return completion, and strictly removes modeled ordering without adding any.
Splitting requires a free key; the eight-key directed pool is not extended.

| Native fixture | Decision | Pairs before → after | Ordering change |
| --- | --- | ---: | --- |
| Independent A/B readiness | Split one broad handoff | 1 → 2 | Consuming A no longer requires B's load completion |
| Existing earlier-publication example | Advance release | 2 → 2 | Reloading A no longer waits for the unrelated second vector operation |
| Existing later-acquisition example | Delay acquisition | 2 → 2 | The first vector consumer no longer waits for the unrelated later load |
| Consumer needs both A and B | Keep bundled readiness | 1 → 1 | Complete joint requirement preserved |
| Additional relevant reader | Keep late release | 2 → 2 | Final reader still protected |
| Full directed event pool | Decline split | 8 → 8 | No key sharing or added serialization |

All these fixtures retain zero named barriers and one PIPE_ALL.
Pairs describe equal static set/wait inventories, not a performance score.

The independent concrete replay of the A/B result finds exactly one changed
prefix: at physical operation 2, the first `pto.tabs`, MTE2 completion changes
from prefix 1 (load B) to prefix 0 (load A). Payload hashes match and both event
inventories drain. This observation is separate from the native acceptance
proof; neither result measures runtime.

Native negatives also preserve invalid-event and unproved-dependency seeds
unchanged, and explicitly decline both zero-trip and nonzero loops. A second
run on the accepted split is unchanged and attempts no further rewrite.

## Full frozen benchmark population

This runs the experiment on existing emitted plans, not a new InsertSync
construction campaign. All eleven original fixtures were checked in manual,
combined and staged arms: **33 rows, 66 native parse/transform invocations, zero
failures, zero changed plans**. Canonical seed and trial PTO match byte for byte.

Every row declines this experiment's structured recurrence/choice boundary.
Thus the table establishes preservation under refusal; it does not establish
that backward needs were solved for these loop-bearing kernels.

| Fixture | Hand pairs | Hand named | Hand ALL | Automatic pairs before = after | Automatic named before = after | Automatic ALL before = after |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| One buffer | 6 | 0 | 1 | 6 | 0 | 0 |
| Two buffers | 12 | 0 | 1 | 12 | MTE3=2 | 0 |
| Three buffers | 18 | 0 | 1 | 18 | MTE3=3 | 0 |
| Four-use producer | 24 | 0 | 1 | 24 | MTE3=2 | 0 |
| GEMM | 53 | 0 | 0 | 56 | 0 | 0 |
| TopK | 10 | V=22 | 0 | 15 | V=14, MTE3=2 | 1 |
| Conv2D | 24 | 0 | 0 | 0 | M=8 | 1 |
| FlashAttention | 18 | 0 | 1 | 15 | 0 | 1 |
| Triangular inverse | 278 | V=105 | 0 | 277 | 0 | 1 |
| GDN | 16 | V=6 | 2 | 32 | V=13, MTE1=1 | 2 |
| KDA | 17 | V=7 | 2 | 35 | V=10, MTE1=1, MTE2=2, MTE3=1 | 2 |

Combined and staged have equal inventories. The automatic MTE3 fallback sites
in the buffering controls and TopK already execute zero times at the recorded
16-trip/group bounds; that earlier improvement is not attributed to this pass.
Likewise, the eight previously recorded GEMM traces already match the manual
cross-lane completion prefixes. There is no demonstrated remaining GEMM cut
to repair merely because its static inventory is 56 rather than 53.

FlashAttention's three arms use the existing exact generic-assembly bridge
for its GM-only pipe initializer. The custom printer omits the parser's
`local_slot_num` spelling; the original bytes and normalization hashes are
retained, and both arms receive identical operands/attributes. This is the
benchmark's existing parser bridge, not an effect-summary or queue change.

## Validation and reproduction

- Incremental native compiler, optimizer and matching Python targets: passed.
- 36 focused native lit tests: passed in 29.30 seconds.
- 24 accounting and boundary-observer tests: passed.
- Native positive/negative decision checks: passed, including repeat stability.
- Independent concrete A/B boundary replay: matching payload, one weaker prefix.
- Full emitted-plan campaign: 33/33 passed; all original ladder plans unchanged.
- Device correctness/progress and timing: **not run**.

```sh
cmake --build /disk/build --parallel 2 --target PTOASCompiler pto-test-opt PTOASPythonPackage
python /disk/llvm/bin/llvm-lit -j1 -sv \
  --filter='insert_sync_(handoff|shared_requirements|lifecycle|buffer_generation|generation|frontier|mmad|slot|loop_handle)|multi_tile_.*(slot|sync)' \
  /disk/build/test/lit/pto
python test/experiments/insert_sync/performance/run_handoff_experiment.py \
  --pto-test-opt /disk/build/tools/pto-test-opt/pto-test-opt \
  --python-root /disk/build/python \
  --campaign /disk/combined-structure-flow/controls-final \
  --campaign /disk/combined-structure-flow/kernels-final \
  --output /disk/new-handoff-experiment
```

Full native outputs, commands and diagnostics are under
`insertsync-builds/campaign/handoff-planning-experiment-final`; the independent
trace is in `handoff-planning-positive/boundary.json`. The initial campaign's
three parser failures remain separately recorded in
`handoff-planning-experiment`.

[Machine-readable results](HANDOFF_EXPERIMENT_RESULTS.json) retain all 33 rows,
input/output fingerprints, parser bridges, decisions and separate mechanisms.
The [annotated proposal](../../../../docs/designs/ptoas-insertsync-coherent-planning.md)
defines the remaining guarded/recurring planning and joint-allocation work.
