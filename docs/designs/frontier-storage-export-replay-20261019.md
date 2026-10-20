# Storage-export completeness and cross-reset replay — 2026-10-19

This follow-up starts from the expert/replay checkpoint and PyPTO survey. It does
not repeat capture, change allocation, raise limits, alter expert protocols, or
remove the zero-inner-body barriers. Results and pins are recorded in the
[companion JSON](frontier-storage-export-replay-20261019.json). This is host
validation of the supplied access model, not native instruction or device
qualification.

## Interface and validity argument

`RegionalAccessBoundary::representedByCells` now certifies the **exported**
boundary cells for that effect, including their occurrence guards. Materialized
input geometry alone is insufficient. The finite providers qualify their flags
against a merged, indexed inventory of the cells they actually export. This
retains numerical mathematics and does not rerun guarded demand reduction.

`regionalStorageSelectors()` is the shared production and test query boundary:

- A provider callback is authoritative, including `nullopt`.
- Otherwise the query checks every possibly relevant residual/deferred effect.
  Unknown membership or alias displacement yields unavailable.
- Complete cells yield exact, byte-masked selectors. Empty means proven absence.
  A residual in a disjoint domain, or a materialized set excluding the queried
  byte, cannot invalidate a valid absence answer.

A separate domain-support request can supply a complete envelope for one domain
without promising that unrelated domains are exported. Its certificate retains
expression owner, access-model identity and alias policy. Sequence and phase
composition consume compatible child certificates and rebuild the appropriate
coordinate projections. Support can contain holes: exact membership/selectors,
not envelope inclusion, establish accesses. Cuts require producer-certified
uniformity across the full atom, including predicates and absence.

The optional finite geometry attached to existing `RepeatedStorage` owners uses
their checked nonnegative/nonwrapping origin, translation stride and local byte
pieces. It refuses unknown finite origins/counts or unsupported residual domains.
No invocation, loop body or demand graph is unfolded. Row/interval partitions
remain charged and subject to the existing 256-atom interface limit. This is not
a claim that all physical boundary interfaces have size independent of visits.

The production call path `sequenceRegionalResult → sequenceStorageSelectors →
finiteStorageSelectors` reaches the former fallback when a sequence combines a
callable child export with a finite child. The new synthetic composition test
uses the actual finite guarded provider and this production path. This source
and regression finding is separate from the original survey-checker defect;
it does not establish a miscompiled original Qwen kernel.

## Cross-reset conflict reduction

A reset does not create fresh GM storage. A visit-owned certificate inside the
wave traversal is not reinterpreted as an epoch ownership certificate. The
existing phase adapter continues to turn composed families into general
membership evidence rather than exporting their old owner as an outer owner.

The retained wave output is row-strided. Enumerating all rows of two output
tiles can exceed the unchanged 256-atom limit. The alternative is a sufficient
**reduction proof using the existing local repeated closure**, not a larger
partition or additional serialization:

1. Construct the ordinary repeated boundary graph without the residual GM
   effects, retaining all child mathematics, presence, native edges and recipes.
2. Require certified first/last occurrence extrema for every omitted effect and
   every other effect that can conflict with it. Unknown GM maps are not admitted.
3. For every possibly conflicting source/target pair, ask the existing
   translation-invariant `RepeatedRegionState::relativeQuery` whether the last
   source completion reaches the first target start one visit later, whenever
   both extrema are present. Failure preserves refusal.
4. Include writer self-pairs. Intrinsic and per-pipe native order connects a
   first occurrence start to its last occurrence completion. The writer's
   self-pair therefore chains the proof across every positive gap. If the source
   is a reader, chain through the target writer instead. This relies on the
   enclosing invariant-repeat contract: guards, inner domains, maps and supplied
   prerequisites repeat unchanged; independently changing guards do not qualify.
5. Only after all implications succeed retain that unchanged crossing graph.
   Child-internal conflicts are unchanged. Adjacent and longer cross-visit GM
   conflicts are already in its closure, so no new cover or extra payload order
   is needed. This is equality for the supplied modeled order, not native access
   precision or realization-order equality.

The exported original access inventory and lazy byte selectors remain intact,
with the epoch visit coordinate prefixed to the selected original coordinates.
A missing selector stays unavailable even if a closure proof makes it unnecessary
for this crossing reduction. Allocation consumes the resulting protocol through
the existing implementation and has a separate outcome.

## Qualified results

The finite guarded producer previously skipped `independentOfOtherPhases()` cells.
That optimization was valid for its internal conflict construction, but not for
its regional boundaries. It now keeps the supplied materialized cells in the
existing shared scan. This introduces no new effect-recovery algorithm or guard
representation. A global selector-readiness bit is deliberately **not** used to
reject otherwise valid local or empty-region queries.

The attention witness now returns the correct GM output writer (occurrence 4)
and the two input readers (occurrences 0 and 1, MTE2) under its pinned finite
configuration. Separate out-of-range queries return exact absence. Its five
payloads, three actual covers, and all four event-query combinations still agree
with the independent oracle. No original Qwen root is promoted to exactness.

Replay uses two waves, K=2, kk=2, core 0, with an explicit epoch coordinate even
for one visit. Every epoch resets the original counters and reuses the same
buffers and GM arguments.

| Replay visits | Active payloads | Independent access comparisons | Actual = oracle covers | Four-kind query comparisons | Local selector queries | GM output probes | Matching/reuse, six IDs | C++ source |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 1, exact empty | pass | succeeds |
| 1 | 34 | 82 | 29 | 8,464 | 111 | 1,280 | pass | succeeds |
| 2 | 68 | 164 | 65 | 33,856 | 111 | 1,280 | pass | succeeds |
| 3 | 102 | 246 | 101 | 76,176 | 111 | 1,280 | pass | succeeds |

GM probes include interval starts, adjacent/interior bytes, last bytes and ends,
including holes between the strided rows. They are finite validation queries,
not a producer certificate inferred by sampling. The implementation's membership
argument still comes from the checked physical maps and owner inversion.

The oracle reads the original supplied NZ/ND layouts and byte accesses; it does
not consume compiler cells or generators. Actual retained regional records are
expanded into original occurrence pairs, predicated and deduplicated before
comparison with the independently constructed covers. The flat reducer is an
additional check, not independent evidence for extraction. First/last writers
and boundary readers use the original access sets.

Removing GM accesses **only in the independent attribution experiment** leaves
the required closure unchanged on these replay instances. Thus their repeated
GM writes overlap, but the required local-buffer order already covers those
conflicts. The production proof checks that property rather than assuming it.
No communication or predicate is deleted to meet a code-size target.

The emitted C→I payload prerequisites have no missing or extra pairs on these
four positive instances. This is finite validation. The implementation's general
`realization_order=covers-selected-order` classification is not upgraded to
uniform equality. The separate zero-inner-body case still has its original two
extra C→I pairs from the MTE2 barriers; those barriers and expert protocols are
unchanged.

### Representation scaling

These are ordinary production logical requests with
`prepare-allocation=false profile-stages=true`, not audit-mode requests.
Inspection of the selected retained owners calls no semantic callbacks.

| Varied dimension | Fixed dimensions | Retained repeated types | Largest numerical body | Shared arena nodes |
|---|---|---|---|---|
| Replay visits 2 / 3 / 8 | waves=2, K=2, kk=2 | 124 / 124 / 124 | 8 / 8 / 8 | 141 / 141 / 143 |
| K visits 4 / 8 / 16 | waves=2, epochs=3, kk=4 | 332 / 332 / 332 | 14 / 14 / 14 | 192 / 196 / 200 |
| kk stages 1 / 2 / 4 | waves=2, epochs=3, K=4 | 116 / 188 / 332 | 5 / 8 / 14 | 171 / 188 / 192 |

Zero and singleton invocation folding are separate shapes. The bounded kk
microstage description still grows with its supplied static size; this is not
claimed to be a compact symbolic implementation of arbitrary kk bounds. K and
epoch counts do not unfold the entire nest. Small arena growth is retained
expression/query work, not a proportional occurrence expansion. The structural
regression checks these retained sizes independently of wall time.

Stage receipts separate mathematical/interface diagnostics, ordinary logical
insertion, allocation-support preparation, allocation and C++ generation.
`check_compact_persistent_contract.py`'s `production_logical` measurement includes
allocation-support preparation; it is **not minimum-demand generation time**.
Profile timings include startup and diagnostics, are single samples on a shared
host, and establish no speedup. Device compilation/execution was not performed.

### Rechecked PyPTO regions and remaining failures

Existing captures/fixtures were reused; the survey and frontend capture were not
repeated. Kernel sources, frontend revisions and physical input hashes remain
those in the [survey receipt](frontier-pypto-region-survey-20261019.json).

| Region/input | Result after this repair | Remaining category |
|---|---|---|
| Qwen attention finite child | Local and modeled GM selectors, covers, queries and absence checks pass | Original symbolic work-loop/entry conditions remain unsupported |
| Qwen down K child | Compact retained analysis and independent checks preserved | Complete GM input-reader exports remain unavailable |
| Qwen down atomic output | Original unresolved modeled GM read/write intervals unchanged | Atomic-store access recovery; not an allocation or alias-disjointness proof |
| DeepSeek RMSNorm bounded instances | Access/closure/covers/local selectors preserved, including zero reduction visits | Root GM selector export still unavailable |
| Persistent replay output | Correct lazy output selectors and local-closure proof across resets | GM input-reader queries on A/B remain unavailable for nonzero visits |
| Original Qwen roots | Conservative completion still selected; 0/0/19 and 2/2/3 SET/WAIT/barrier sites | Exact root mathematics/exports and symbolic parent contracts remain separate gaps |
| Original RMSNorm root | Sequence; 8/8/13 sites; exact mathematics retained | Storage-selector request still unavailable |

All three original protocols still allocate with six IDs. The Qwen successes are
feasibility results for their conservative protocols. They are not exact
whole-kernel Section 5 results or proofs of a minimum ID budget. Allocation
algorithms and policies were not changed in this task.

The new replay route is intentionally sufficient. If residual conflicts are not
already covered, the ordinary finite-support route must still qualify within its
unchanged limit. General symbolic crossing synthesis, independently changing
replay guards, missing extrema/maps, and unclassified overlapping GM effects are
not silently admitted. Missing readonly input selectors remain missing exports;
they are never encoded as empty boundary lists.

## Reproduction and evidence

Source base: `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44` with the pre-existing
accepted working changes and this repair. This is a dirty-tree receipt, not a
clean-commit-only reproduction. The companion JSON pins source files, inputs,
LLVM revision, Python and the actual compiler executables. Raw commands/results
are retained under `.local/storage-export-replay/`.

Run from `workspaces/frontier-shared-input`:

```sh
build=.local/section8-refactor/source-pinned/candidate-build
.venv/bin/python test/lit/pto/Inputs/check_storage_export_coverage.py \
  --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test" \
  --output .local/storage-export-replay/reproduce-coverage
.venv/bin/python test/lit/pto/Inputs/check_compact_replay_checkpoint.py \
  --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test" \
  --input test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto \
  --output .local/storage-export-replay/reproduce-contracts
.venv/bin/python test/lit/pto/Inputs/check_compact_persistent_contract.py \
  --optimizer "$build/tools/pto-test-opt/pto-test-opt" \
  --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test" \
  --input test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto \
  --output .local/storage-export-replay/reproduce-pipeline --replay-only
.venv/bin/python test/lit/pto/Inputs/profile_compact_replay.py \
  --optimizer "$build/tools/pto-test-opt/pto-test-opt" \
  --input test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto \
  --output .local/storage-export-replay/reproduce-scaling
```

The CMake build uses the already selected source snapshot. Changed files were
copied there explicitly before incremental compilation. No LLVM rebuild was
performed. Native builds used at most two workers; campaigns/lit used one each,
with no more than two intensive workers in aggregate. Large receipts remain on
disk in the workspace, not `/tmp`.

The **fresh focused gate passes 17/17**. The full compact semantic campaign passes
**11/11**, including static one-/two-stage cases, skipped work and the unchanged
zero-inner-body extra-order case. The new `sync_storage_export_coverage.pto`
registers the production composition/absence and owner-geometry checks;
`sync_compact_replay_scaling.pto` registers replay emission/reuse and scaling.

An earlier focused run failed because the executable CLI selected a Python ABI
different from the workspace bindings. The complete gate was rerun with the
workspace venv on PATH; no expectation or timeout was relaxed. Intermediate
negative tests also caught an overbroad noninvariant-map proof attempt and a
global selector-readiness gate that rejected RMSNorm's valid empty local region.
The final implementation retains invariance qualification and domain-sensitive
availability. Those failed receipts are preserved rather than counted as passes.

C++ source generation succeeds for replay 0–3 with synchronization insertion and
memory replanning disabled. Its unchanged Python/native emitter artifact is
pinned separately from the rebuilt analysis/allocation tools. C++ output is not
device qualification. No full corpus, device compiler or device execution was
run for this follow-up.

Scoped Ruff and native builds pass. The changed-code prefilter's brace findings
are reviewed against balanced control conditions and braced bodies; its lock
warnings refer to `weak_ptr::lock`, not mutexes. The review receipt retains every
finding. No external SecK/EChecker or quantitative complexity analyzer was run.

## Paper consequences

| Established result | Remaining premise or limitation | Appropriate manuscript claim |
|---|---|---|
| Finite provider retains independent materialized GM cells and exports correct attention boundaries | Supplied modeled access/layout precision; symbolic parent contract still missing | Internal conflict independence does not establish absence from a regional storage interface |
| Production composition uses the common exact/absence/unavailable query boundary | Provider-owned effect/extrema and alias contracts remain trusted premises | Storage coverage is domain-specific; omitted or unresolved GM storage cannot become an empty answer |
| Replay 0–3 agrees on closure, actual covers, queries, output/local selectors and reuse | Invariant body/guards/maps, complete extrema, supplied GM argument disjointness | A demonstrated compact regional composition path across counter resets under these contracts |
| Independent attribution finds GM reset conflicts already in local closure | Other reset patterns may need real new crossings or a different supported interface | Overlap alone does not imply additional covers; certify coverage before constructing more relations |
| Retained K/replay sizes stay compact in the measured sweeps | Bounded kk description still has explicit finite cost; not arbitrary nested-loop support | Complexity is in the retained supplied description and requested interfaces, not executed visits alone |
| Emitted positive instances have no extra C→I pairs | General guarantee remains coverage; zero-inner barriers still add two pairs | Keep mathematical minimality, realization order and finite validation distinct |
| Selected protocols fit six IDs and emit C++ | No minimum-budget proof or device result; expert native premises unchanged | Physical feasibility and source generation are separate from exactness and native correctness |
