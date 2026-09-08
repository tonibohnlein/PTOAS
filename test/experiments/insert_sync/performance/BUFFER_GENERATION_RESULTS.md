# Shared buffer-generation regression results

2026-09-08; R8 `4c19cc1c` plus the uncommitted shared-generation implementation.
Native SHA-256: `69a256c345583146da05f51a53529ec7b0f0e1f588b1d23c5ecc6902de707e49`.

All 11 unchanged frozen fixtures pass in hand-tuned, combined and staged arms:
33 rows, 66 PTO/C++ compiler invocations, and 306 scalar payload replays.
Input hashes and manual/automatic payload parity are checked by the existing runner.
Automatic arms enable MMAD and `--insert-sync-buffer-generations`; staged also enables
`--insert-sync-defer-same-pipe`. Their static mechanism inventories match in every row.

These are static site counts. A pair is one set and one wait. Named barriers and
PIPE_ALL remain separate; these mechanisms are not combined into a score.

| Fixture | Hand pairs | Hand named barriers | Hand ALL | New pairs | New named barriers | New ALL |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| One buffer | 6 | 0 | 1 | 6 | 0 | 0 |
| Two buffers | 12 | 0 | 1 | 12 | MTE3=2 | 0 |
| Three buffers | 18 | 0 | 1 | 18 | MTE3=3 | 0 |
| Four-use producer | 24 | 0 | 1 | 24 | MTE3=2 | 0 |
| GEMM | 53 | 0 | 0 | 56 | 0 | 0 |
| TopK | 10 | V=22 | 0 | 15 | MTE3=2, V=14 | 1 |
| Conv2D | 24 | 0 | 0 | 0 | M=8 | 1 |
| FlashAttention | 18 | 0 | 1 | 15 | 0 | 1 |
| Triangular inverse | 278 | V=105 | 0 | 277 | 0 | 1 |
| GDN | 16 | V=6 | 2 | 32 | MTE1=1, V=13 | 2 |
| KDA | 17 | V=7 | 2 | 35 | MTE1=1, MTE2=2, MTE3=1, V=10 | 2 |

## What changed

| GEMM plan | Set/wait pairs | MTE2 | MTE1 | M | FIX | PIPE_ALL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hand-tuned | 53 | 0 | 0 | 0 | 0 | 0 |
| R8 staged + MMAD | 44 | 3 | 2 | 4 | 1 | 1 |
| First generation implementation | 56 | 0 | 0 | 4 | 1 | 1 |
| Shared-generation follow-up | 56 | 0 | 0 | 0 | 0 | 0 |

The new GEMM plan commits four L1 slots and two L0 bundles on its first attempt.
Guarded immediate-M predecessor facts reach the existing qualified MMAD rule;
17 retained MMAD source/target witnesses are rechecked on emitted IR. R8 mixed
completion supply remains active. Final completion cleanup removes three sites
left after insertion: two named barriers and the exit ALL. Other repairs are
avoided during insertion. This staged accounting explains why the cleanup
counter alone is smaller than the full before/after barrier difference.

The one/two/three-buffer and four-use controls each lose their exit PIPE_ALL after
the independent completion query proves their explicit drains sufficient. Their
other inventories are unchanged. The six remaining kernels retain the R8 inventories.
No automatic arm adds a body PIPE_ALL. GDN/KDA retain one exit ALL per physical
function; their two manual ALL sites are inside vector work loops.

Signed parity equality, signed parity inequality and unsigned parity equality all
produce the same accepted GEMM inventory. No kernel name, frozen address or loop
nesting shape selects the optimization. The open-flow accumulator and two-/three-slot
handle-rotation native fixtures exercise separate inputs. The rotations remove the
MTE2→MTE3 pair from the loop body while preserving wraparound repair. Their total
static pair inventory stays at four, and each adds a named MTE3 site; they demonstrate
changed dependency placement, not a measured runtime improvement.

## Validation and limits

- Incremental compiler/native-opt/Python runtime targets passed.
- 57 C++ core checks passed, including open flow, multi-lane readers, parameter substitution,
  summarized/unsummarized agreement, slot periods 2/3/5, exit-drain negatives, causal
  key-sharing negatives and a constrained allocation that fits after sharing.
- 34 focused native lit tests passed, including three GEMM expression variants,
  native selectors/handle rotations, MMAD negatives, zero-trip returns and option composition.
- 20 benchmark-accounting/comparison tests passed.
- Full frozen regression: 33/33 rows and 306 scalar replays passed.

Residual key compaction removes no key on these frozen fixtures. A native scarcity
benefit is not established by this campaign. Selected protocols remain fixed during
R4/R5 refinement; this composition currently supports proved residual barrier deletion.
General residual event movement with selected protocols, arbitrary parameterized
ownership transfers and broader helper/queue imports are remaining extensions.

Local compilation and scalar replay do not establish device numerical correctness,
asynchronous progress or wall time. Device validation of this new output remains pending.

## Reproduction and retained evidence

Use the matching Python interpreter and native/Python runtime:

```sh
python test/experiments/insert_sync/performance/run.py \
  --python-root "$BUILD/python" --mmad-chains --buffer-generations \
  --output /disk/results/generation-controls
python test/experiments/insert_sync/performance/run.py \
  --python-root "$BUILD/python" --mmad-chains --buffer-generations \
  --manifest test/experiments/insert_sync/performance/kernel-pairs-manifest.json \
  --output /disk/results/generation-kernels
```

The runner creates a serial MLIR context for each compiler subprocess.

Latest evidence:

```text
/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/shared-generation-flow/controls-verified/results.json
/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/shared-generation-flow/kernels-verified/results.json
/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/shared-generation-flow/slots/comparison.json
```

[Machine-readable inventories](BUFFER_GENERATION_RESULTS.json) include both automatic
arms, input identities, static placements hashes and per-scenario synchronization counts.

Earlier generation-only evidence remains in `campaign/buffer-generations/` with native
SHA-256 `918e6952287b66a0e2fa7f1442c3e3c7660404ac3981eee40978cc444629588b`.
The initial `shared-generation-flow/controls-regression` and `kernel-regression` directories
record a runner-level rejection of an unsupported threading CLI option. They contain
no successful compiler runs and are excluded from this table.
