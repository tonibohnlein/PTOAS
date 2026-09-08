# Full generated corpus: shared requirements and synchronization quality

Measured on 2026-09-08 at `3400426f393aac606c134663d5517218fe9a3dc6`.
Both original InsertSync and the current staged/MMAD/shared-generation path
compile **8,183 of 9,754 frozen rows** to PTO and C++. The same 1,571 rows fail
before InsertSync, with identical diagnostics. No new admission failures occur.
All 8,183 paired outputs pass MLIR verification and preserve the exact payload
projection, allocations, views and function interfaces.

The complete per-row mechanism table is [FULL_CORPUS_RESULTS.csv](FULL_CORPUS_RESULTS.csv).
The [compact evidence JSON](FULL_CORPUS_RESULTS.json) retains compiler/input
fingerprints, aggregate results, representative rows, diagnostics and replay
evidence. Device numerical correctness and timing were **not run**.

## Premise and implemented scope

The input already specifies payload order, tiling, buffers and prefetches.
Synchronization should answer two separate questions:

1. Which particular physical access occurrences require ordering, given their
   storage, generations, guards and enclosing iterations?
2. Does the combined synchronization plan already establish that requirement?

Input requirements remain meaningful when a protocol is rejected or a handoff
changes. Qualified intrinsic ordering, access disjointness, generation readiness,
full physical completion and event consumption remain distinct justifications.
Only missing ordering needs repair. Complete ready/release constructions can
implement recurring requirements; their publications and acquisitions should
avoid capturing unrelated work. Realization must preserve participation,
concrete key ownership, required ordering and any claimed boundary improvement.

The current implementation shares exact local-storage requirements between
ordinary repair and recurring construction. It also performs checked residual
publication refinement and removal of whole redundant readiness streams.
It is still a supported subset inside InsertSync: it does not provide universal
operation summaries, a general joint allocator, or a complete backward-demand
planner. An unsupported optimization retains ordinary insertion.

## Population and compilation

The full manifest contains **9,754 rows, 9,563 distinct input hashes and 213
source-entry seeds**. Many rows are related kernels from model expansion, not
independent algorithms. The established first-per-seed selection has 213 rows
and 186 input hashes; its fresh comparison remains 208/213 in both arms.

The frozen frontend inventory originally contained 349 entries: 213 collected,
102 collection failures, 17 requiring construction adapters and 17 declared
drafts. This campaign consumes the complete existing generated collection; it
does not rerun or relabel those earlier frontend outcomes.

| Full-corpus outcome | Original | Current |
| --- | ---: | ---: |
| PTO and C++ compile | 8,183 | 8,183 |
| `tci` lacks explicit temporary when memory planning is skipped | 752 | 752 |
| Narrowing `tcvt` lacks explicit temporary when memory planning is skipped | 715 | 715 |
| `tsel` temporary has incorrect element width for A2/A3 | 104 | 104 |

These are frozen-input/lowering contract failures, not evidence of an InsertSync
optimization failure. Correcting them requires a separately versioned input
population; this experiment does not silently repair its inputs.

## Mechanisms: rows with fewer, equal or more sites

Each cell counts successful paired **rows**, not instructions summed into a
performance score. Counts are static; guarded sites may execute differently.
Set and wait inventories remain separate in the CSV and agree in this corpus.

| Mechanism | Fewer sites | Same sites | More sites |
| --- | ---: | ---: | ---: |
| Set flags | 246 | 7,183 | 754 |
| Wait flags | 246 | 7,183 | 754 |
| `PIPE_ALL` | 544 | 7,639 | 0 |
| `PIPE_MTE2` barrier | 314 | 6,230 | 1,639 |
| `PIPE_MTE1` barrier | 241 | 7,942 | 0 |
| `PIPE_M` barrier | 295 | 7,888 | 0 |
| `PIPE_FIX` barrier | 102 | 8,081 | 0 |
| `PIPE_MTE3` barrier | 90 | 8,093 | 0 |
| `PIPE_V` barrier | 114 | 8,059 | 10 |

5,430 rows have identical complete inventories. This does not establish equal
placement or device performance. Conversely, a placement hash includes event
IDs and action ordering, so a changed hash alone does not prove changed blocking.

747 rows select lifecycles, spanning 100 entry seeds. Readiness sharing fires
in 166 rows from 59 entry seeds, removing 550 logical readiness streams and
923 set plus 923 wait sites from their constructed candidates. These counters
compare the accepted candidate before/after that transformation, **not original
InsertSync**. Publication advance fires in **zero** full-corpus rows. Its earlier
QK demonstration remains a separate benchmark result.

Of the 246 rows with fewer pairs than original, only two select lifecycles;
244 use the general path. Of the 544 rows losing `PIPE_ALL`, 497 select lifecycles
and 47 use general-path completion cleanup. Therefore, the whole original/current
delta must not be credited to the newest readiness transformation.

## Concrete examples

Each flags cell gives set/wait sites. Named barriers and `PIPE_ALL` are separate.
Case IDs and exact generated source paths are retained in the CSV/JSON.

| Kernel | Original flags | Current flags | Named original | Named current | ALL original/current |
| --- | ---: | ---: | --- | --- | ---: |
| `gemm_tile` | 8/8 | 14/14 | 0 | 0 | 1/1 |
| `multi_proj` / `proj` | 10/10 | 19/19 | M=4 | M=3 | 1/1 |
| `dspark_base_logits` | 35/35 | 51/51 | M=7 | M=7 | 1/1 |
| `proj_b_mm` (`e8fe..._007`) | 21/21 | 31/31 | FIX=1 | 0 | 1/0 |
| A8W8 `kv_proj_matmul` | 39/39 | 32/32 | 0 | 0 | 1/0 |
| A8W8 `q_proj_matmul` | 21/21 | 18/18 | 0 | 0 | 1/0 |
| `lm_head_matmul` (`53e5..._000`) | 94/94 | 94/94 | M=17, MTE1=2, FIX=1 | M=17, MTE1=2, MTE2=4, FIX=1 | 1/1 |

The two A8W8 projections come from `models/qwen3_14b/prefill_fwd_a8w8.py`,
entry `prefill_hidden_a8w8`, cases `a61baf44c0880ad5ad74_001` and `_082`.
Both retain four selected channels with freshly reconstructed obligations.
They are concrete improvements in mechanisms beyond the historical GEMM.

The small GEMMs demonstrate why higher pair counts need boundary analysis:

| Concrete execution | Original executed pairs | Current executed pairs | Earlier cross-lane completion prefixes | Later prefixes |
| --- | ---: | ---: | ---: | ---: |
| `gemm_tile`, four K iterations | 19 | 28 | 7 | 0 |
| `proj`, 64 iterations | 323 | 517 | 253 | 0 |

Both replays have identical physical payload traces and valid, drained event
participation. They establish less imposed cross-lane completion in these
executions, not a latency model or numerical correctness. `proj` also executes
127 rather than 128 M barriers; each retains one exit ALL. The first panel
consumer no longer acquires the independent second preload. Panel release and
operand acquisition can also occur without waiting for unrelated later work.
The measured arguments and the complete boundary differences remain in the raw
`boundaries.json`; representative differences are copied into the compact JSON.

All 1,639 rows with extra MTE2 barriers use the general path. Controls compiled
with buffer generations disabled reproduce the extra MTE2 sites in
`exp_gate_mm` and the large `lm_head_matmul`; these predate the latest shared
requirement refinement. The same control adds one MTE2 barrier to each small
GEMM, which the selected lifecycle path removes. This does not classify every
extra MTE2 dependency as either necessary or redundant.

Some mixed cube/vector `qk_pv` variants trade fewer pairs for more MTE2 and,
in ten rows, more V barriers. For example `ef61476d00115c75cd58_364` changes
55/55 to 49/49 pairs, MTE2 0 to 4 and V 14 to 16; its two ALL sites remain.
That is not an established performance improvement.

## What prevents wider use

Native pass-entry diagnostics were replayed for 205 of the 208 successful
representative rows. Their mechanism inventories all match the CLI outputs.
Three multi-function rows are excluded from this supplementary single-function
diagnostic adapter, not from compilation or payload verification.

| Representative native outcome | Rows |
| --- | ---: |
| Selected lifecycle construction commits | 23 |
| Import stops at an unmodeled non-payload effect | 59 |
| Physical operation lacks qualified frontier adapter | 33 |
| Construction succeeds far enough to hit the concrete graph node budget | 32 |
| Effect-coverage report retains ordinary translation | 25 |
| Fixed/helper/macro contract outside the constructor | 15 |
| No complete exact-slot lifecycle | 7 |
| Lifecycle recognition budget | 7 |
| Initial structural graph node budget | 4 |

All 59 non-payload-effect rows contain `pto.set_validshape`. The physical-adapter
group includes arithmetic, conversion, reduction and broadcast operations such
as `tmuls`, `tcvt`, `trowsum` and `texpands`; many occur together, so admitting
one opcode does not imply that its whole kernel becomes optimizable.

`exp_gate_mm` obtains generation/readership facts but its emitted combined
graph exceeds the 2,048-node importer limit. The large `lm_head_matmul` reaches
that limit even before synthesis. These are demonstrated representation/budget
obstacles, not missing kernel-pattern names.

1,671 successful full-corpus rows contain a function with
`effect_coverage="gap-legacy-retained"` in report mode. This check has a limited
semantic domain; the result does not establish that production InsertSync
misses a required device synchronization. It does mean these rows must not be
counted as fully supported by the optional structural optimizer.

## Next refinements and acceptance

1. **Qualify descriptor and physical-phase import.** Start with an unchanged
   `set_validshape` kernel and retain descriptor versions, consumers and
   lifetimes. Do not mark descriptor mutation pure or use a dynamic valid shape
   as a definite whole-allocation write. Follow with a small useful vector
   operation family and its complete effect/scratch contract. Require an actual
   emitted improvement, not merely removal of the first import gate.
2. **Reduce graph expansion and repeated proof work.** Use the existing loop and
   control summaries to retain occurrence relationships without repeatedly
   expanding the whole emitted protocol. Target `exp_gate_mm` and the larger
   `lm_head_matmul`; preserve bounded work and independent concrete checking.
   Raising the node cap alone is not the proposed fix.
3. **Improve command-efficient lowering without losing the demonstrated cuts.**
   Use `gemm_tile`, `proj` and the projection families to remove redundant
   priming, readiness or release actions only when requirements, participation
   and full-plan supply allow it. Preserve the earlier readiness/release points;
   do not merge streams merely to recover original pair counts.

Device timing should cover the two A8W8 projections and the two small GEMMs,
alongside the established manual-GEMM reference. These distinguish fewer
commands/drains from more commands with earlier boundaries. The ten mixed
`qk_pv` rows with additional V barriers deserve a separate exact-dependency
inspection. Broader device claims require the appropriate peer/harness contracts.

## Reproduction and experiment cost

Original is frozen production source `7e2ec3e29420e297dcf5b3c59ba4d90841464821`;
its compiler library hash is
`0fa0a43290cb0a43f48755ded11fe9879323698b86ea8aa63cf127bb3844b7b3`.
Current uses library hash
`556d43a49f6be9224e24b25f28b9f02bdcac63be60220b07ddedd54f1008c655`.
Both fingerprints are unchanged before/after the run. No compiler build was
performed for this campaign. The current binary was built before the source
commit, as recorded in the preceding benchmark; it is not relabelled as a fresh
post-commit rebuild.

Both arms use A3 and each manifest row's original compilation level. The
current flags are staged insertion, MMAD chains, buffer generations,
`gm-alias=assume-disjoint-arguments`, and `effect-coverage=report`. Original uses
its historical distinct-argument contract. The separate one-shot experiment
is disabled.

There are 39,016 compiler invocations in the full comparison: PTO and C++ for
every row in both arms, including failures. The serial warm-process runner
preloads one native runtime and forks an isolated process per invocation;
it refuses a multithreaded parent. On both 213-row arms, every successful PTO
and C++ file matches the fresh-process run byte-for-byte and all failures match.
Two serial arms obey the aggregate two-worker limit.

Summed full-corpus invocation wall time is 615.18 s original versus 966.52 s
current. This includes lowering, diagnostics and process overhead under the
campaign's concurrent workload; it is not an isolated InsertSync pass-time
measurement. The current path therefore has measurable compilation cost as
well as its selective output improvements.

The reused exact projection separately records synchronization-only branches,
comparisons and constants whose uses are exclusively synchronization control.
Constants contributing to payload/allocations remain compared. All 31 accounting
and projection tests pass, including guard and allocation mutations.

Raw inputs/manifests remain in the existing disk-backed snapshot:
`/home/toni/work/pypto3_sync_more/PTOAS/build/protocol-sync-native-corpus/static-p3a`.
Raw outputs, commands, diagnostics and replays are under:
`/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/shared-corpus-3400426f3`.
The new report does not copy that 390 MB generated-input collection into Git.

Run `compare_native.py --warm-process` with the full `manifest.tsv`, existing
input root, matching Python ABI/runtime and a new output directory per arm.
Add the current flags above using `--arm staged --buffer-generations
--mmad-chains --gm-alias assume-disjoint-arguments --effect-coverage report`.
Use `--arm main` with the frozen original runtime for the reference.
`summarize_corpus.py --manifest ... --inventory ... --original ... --current ...
--python-root ... --output ...` verifies and compares completed outputs without
recompiling them. Its optional `--pto-test-opt` collects supplementary native
diagnostics for a selected population.
