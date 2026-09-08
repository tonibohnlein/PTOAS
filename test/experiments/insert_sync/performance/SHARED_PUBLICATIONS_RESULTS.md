# Shared requirements: QK boundaries and Q-projection commands

The integrated InsertSync change improves the unchanged QK and Q-projection
outputs. It runs under `--insert-sync-buffer-generations`; the earlier standalone
one-shot experiment is disabled. Input payloads, allocations, views, ABI and
caller alias contracts are unchanged. Device correctness and timing are pending.

## Demonstrated behavior

| Target | Previous revised output | Current revised output | Preserved obligations / cost |
| --- | --- | --- | --- |
| QK first Q0 extract | Acquires Q0 and independent Q1 preload | Acquires Q0 alone on nonempty execution | Later Q1 reads and empty-path exit retirement remain protected; +5 scalar operations/launch |
| Q projection | 45 / 45 static sets/waits; 779 executed pairs at 32 iterations | 39 / 39 static sets/waits; 651 executed pairs | Four entire redundant readiness streams replaced by existing supply; separate release streams retained |

The original production InsertSync already has the earlier QK cut. This change
removes the previous revised path's extra preload dependency; it does not claim
an earlier cut than production. Current QK has zero `PIPE_ALL`, compared with
production's one exit drain. The current revision's Q-projection command count
remains above production (19/19 static sites, 389 executed pairs).

## Eight unchanged Qwen additions

Static sites; named barriers and `PIPE_ALL` are never combined with flags.

| Kernel | Original set / wait | Previous revised set / wait | Current set / wait | Named barriers (all three) | `PIPE_ALL` original / previous / current |
| --- | ---: | ---: | ---: | --- | ---: |
| rmsnorm | 40 / 40 | 40 / 40 | 40 / 40 | MTE3=3, V=29 | 1 / 1 / 1 |
| softmax | 15 / 15 | 15 / 15 | 15 / 15 | V=14 | 1 / 1 / 1 |
| online_softmax | 12 / 12 | 12 / 12 | 12 / 12 | V=20 | 1 / 1 / 1 |
| silu | 3 / 3 | 3 / 3 | 3 / 3 | V=6 | 1 / 1 / 1 |
| rope_kv_cache | 15 / 15 | 15 / 15 | 15 / 15 | MTE3=2, V=12 | 1 / 1 / 1 |
| qk_matmul | 21 / 21 | 21 / 21 | 22 / 21 | M=2 | 1 / 0 / 0 |
| sv_matmul | 25 / 25 | 25 / 25 | 25 / 25 | M=2 | 1 / 0 / 0 |
| q_proj | 19 / 19 | 45 / 45 | 39 / 39 | M=6 | 1 / 1 / 1 |

QK's 22 set sites / 21 wait sites are intentional: two mutually exclusive
publication sites use one existing event key. Exactly one executes, so every
replayed launch has balanced publications and acquisitions.

The five vector cases still decline the optional structural import at qualified
operation-summary boundaries (`texpands`, `tfillpad`, `tmax`, `tneg`, or
`tcolexpandmul`). Their production effect coverage is complete. These are
optimizer support limits, not evidence of missing production synchronization.
QK, SV and Q projection select four, five and ten lifecycle channels respectively.
Their BF16 / dynamic descriptor matrix operations remain outside the current
qualified MMAD rule; this change does not weaken that rule to remove M barriers.

## Original eleven fixtures

All 33 manual/combined/staged rows pass. Their complete recorded metrics are
unchanged from the preceding shared-generation implementation. Combined and
staged inventories agree in this campaign.

| Fixture | Hand set / wait | Hand named | Hand ALL | Current set / wait | Current named | Current ALL |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| one_buffer | 6 / 6 | 0 | 1 | 6 / 6 | 0 | 0 |
| two_buffer | 12 / 12 | 0 | 1 | 12 / 12 | MTE3=2 | 0 |
| three_buffer | 18 / 18 | 0 | 1 | 18 / 18 | MTE3=3 | 0 |
| four_use | 24 / 24 | 0 | 1 | 24 / 24 | MTE3=2 | 0 |
| historical_gemm | 53 / 53 | 0 | 0 | 56 / 56 | 0 | 0 |
| topk_128 | 10 / 10 | V=22 | 0 | 15 / 15 | MTE3=2, V=14 | 1 |
| conv2d_interior | 24 / 24 | 0 | 0 | 0 / 0 | M=8 | 1 |
| flash_attention_cube | 18 / 18 | 0 | 1 | 15 / 15 | 0 | 1 |
| triangular_inverse_16 | 278 / 278 | V=105 | 0 | 277 / 277 | 0 | 1 |
| gdn_wy | 16 / 16 | V=6 | 2 | 32 / 32 | MTE1=1, V=13 | 2 |
| kda_wy | 17 / 17 | V=7 | 2 | 35 / 35 | MTE1=1, MTE2=2, MTE3=1, V=10 | 2 |

The multi-buffer control and TopK MTE3 sites are guarded arithmetic fallbacks:
they execute zero barriers on the established 16-trip scenarios. GEMM retains
56/56 pairs and zero named barriers or `PIPE_ALL`. Fixed peer/queue protocols in
the relevant fixtures remain intact; this local campaign makes no launchability
or numerical-correctness claim for them.

## Executed mechanisms and scalar control

`Scalar operations` counts executed `arith.*` and `scf.*`, including yields.
The older replay field `scalar_steps` counts all interpreter steps, including
payload and synchronization; it must not be interpreted as scalar-only overhead.

| Target / trips | Original pairs | Previous revised pairs | Current pairs | Scalar operations original / previous / current |
| --- | ---: | ---: | ---: | ---: |
| qk_matmul / -1 | 5 | 5 | 5 | 29 / 29 / 34 |
| qk_matmul / 0 | 5 | 5 | 5 | 29 / 29 / 34 |
| qk_matmul / 1 | 21 | 21 | 21 | 49 / 49 / 54 |
| qk_matmul / 2 | 37 | 37 | 37 | 69 / 69 / 74 |
| qk_matmul / 16 | 261 | 261 | 261 | 349 / 349 / 354 |
| q_proj / 32 | 389 | 779 | 651 | 435 / 435 / 435 |

For positive QK trips, exactly one physical completion-prefix comparison changes
versus the previous revision: at physical operation 3, the first Q0 extract,
MTE2 prefix 1 becomes prefix 0. The second preload's later readers remain
protected. Negative/zero-trip executions retire both preloads and drain tokens.
The additional scalar work is two comparisons, two conditionals and one executed
yield per launch. Q projection has no changed payload completion prefix in the
32-iteration replay and preserves all 435 scalar operations.

The independent QK native test covers renaming, an additional reader, overlapping
panels, equivalent predicates with skipped readers, and two enclosing invocations
(54 concrete executions). Broken empty-path and duplicate publications fail.
Q-projection native checks cover zero, one, two and 32 iterations, renaming,
actual operand production/reclamation, and a deliberately broken plan with
balanced set/wait counts but missing readiness.

## Validation and reproducibility

- Incremental native `PTOASCompiler` and `pto-test-opt` targets build successfully.
- 22 focused native tests pass; 30 accounting and projection mutation tests pass.
- 33 old-fixture rows / 66 PTO+C++ emissions / 306 scalar replays pass.
- Eight additions / 32 PTO+C++ emissions / eight native diagnostic runs pass.
- Original/revised pass-entry payload, allocations, views and ABI match in all additions.
- Concrete reconstruction preserves required effects, event recurrence and exit
  retirement. Q-projection trial removal is rechecked after actual IR mutation.

The final comparison used a pre-commit worktree binary. Its SHA-256 is
`556d43a49f6be9224e24b25f28b9f02bdcac63be60220b07ddedd54f1008c655`.
The true original binary is from `7e2ec3e29420e297dcf5b3c59ba4d90841464821`, SHA-256
`0fa0a43290cb0a43f48755ded11fe9879323698b86ea8aa63cf127bb3844b7b3`.
The JSON retains input, runtime and optimizer hashes. Historical reports keep
their original hashes; they are not relabelled as this campaign.

Measured summed process time was 46.26 s for the old-fixture emissions and
26.68 s for the eight-addition campaign. These include Python startup, lowering,
C++ emission and diagnostics, and were run with at most two aggregate workers;
they are not isolated pass-time measurements or evidence of compiler speedup.
QK's native pass-only diagnostic invocation took 0.246 s; Q projection took
0.308 s. Publication search is bounded by 32 trials and existing analysis budgets.

Run the checked-in serial drivers with the matching rebuilt Python package:

```sh
python test/experiments/insert_sync/performance/run.py \
  --python-root "$BUILD/python" --buffer-generations --mmad-chains \
  --output "$RESULTS/controls"
python test/experiments/insert_sync/performance/run.py \
  --python-root "$BUILD/python" --buffer-generations --mmad-chains \
  --manifest test/experiments/insert_sync/performance/kernel-pairs-manifest.json \
  --output "$RESULTS/kernels"
python test/experiments/insert_sync/performance/run_qwen_additions.py \
  --original-python-root "$ORIGINAL_PYTHON" --revised-python-root "$BUILD/python" \
  --pto-test-opt "$BUILD/tools/pto-test-opt/pto-test-opt" --output "$RESULTS/qwen"
```

Use the build's Python ABI. The native tests `insert_sync_shared_publications.pto`
and `insert_sync_shared_readiness.pto` reproduce the target-specific behavioral
checks from the unchanged source kernels. Full local raw artifacts are under
`/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/shared-local-requirements`.

This implements the scoped integrated milestone, not a universal joint event
allocator or a full backward-demand solver for arbitrary control flow. It reuses
current finite assignments and preserves their endpoints. Device timing must
establish whether QK's one-time overlap gain outweighs added scalar control and
whether Q projection's reduced commands improve wall time.
