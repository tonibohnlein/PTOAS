# Constant maps and compact Qwen child import — 2026-10-20

**The original Qwen whole K-loop child now imports through the existing production completion path.** Mapping conversion, original-frame preparation, parent resource separation, whole-plan six-ID allocation and C++ generation succeed. IDs remain 0–3. Allocation algorithms, certificate production, logical child protocol and SCC transfer are unchanged.

Baseline: [range certificate](frontier-qwen-range-certificate-20261020.md). Pins, commands, cases and checks are in the [machine-readable receipt](frontier-constant-map-child-import-20261020.json). Disk-backed raw artifacts are in `.local/constant-child-map/`. The baseline is the immediate pre-normalization dirty source and binaries at HEAD `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44`, rather than HEAD alone. No commit or push was performed.

**Normalization contract.** A decoded record maps an admitted coordinate tuple `x` to `ids[offset(x)]`. Its original recipe establishes activity, selected record, matching identity and coordinate bindings. A tuple rule must have the expected original arity and pass `validPhysicalTupleRule(rule, ids.size())`. This bounds its offset over the entire admitted machine-coordinate domain, rather than sampled coordinates. A legacy cyclic rule selects its offset modulo the nonempty table size.

If every table entry equals `c`, every applicable lookup equals `c`. Replacing the table with `[c]` and the offset with zero therefore preserves its assignment for every original applicable coordinate tuple. The tuple rule retains its original coordinate count before the existing adapter appends enclosing coordinates. Legacy stride and phase become zero. The conversion does not supply new applicability: guards, selectors, cuts, namespace translation and occurrence frames remain in the unchanged recipe. Dropping unused arithmetic does not grant availability to an otherwise unavailable endpoint.

Validation precedes normalization. A malformed offset rule, foreign arity, invalid ID or overflowing extended arity still refuses, even for a constant table. Empty tables and noncontiguous nonconstant tables remain unsupported. Nonconstant contiguous tables retain their table and original function. The conversion never deletes a record or invokes another allocation. It does not generalize to every nonconstant table whose reachable subset happens to select a constant.

The narrow owner is `CompletionInlineChild::qualifyVisit`: its private `completionVisitMapping` helper in `Shared/CompletionInlineChild.cpp`, declared in `Shared/CompletionVisitMapping.h`, converts an already decoded assignment into the existing completion-palette frame. It validates the original rule, collapses constant tables, performs the established contiguous-table conversion, and then extends arity. The allocator/decoder and general map service are unchanged. Existing parent palette composition remains responsible for palette/namespace translation; bounded checks additionally establish that the actual Qwen physical commands retain the original decoded IDs.

The Qwen decoder still returns 138 record mappings and 258 table entries. There are 94 constant tables, including 28 repeated constant tables; normalization reduces the table-entry inventory to 206 and retains all 138 records. Those 28 records happen to be inactive under established proofs, but no inactivity fact is consumed by this repair.

**Tests and applicability.** Native mapping checks cover active mappings with singleton/two-/four-entry constant tables, IDs 0/3/5, coordinate arities 1/2/4, independently varied coordinate values up to `UINT64_MAX`, two enclosing coordinates, nonconstant cyclic and tuple maps, invalid arities/offsets/moduli/IDs, and unsupported irregular tables. The reference evaluates the original table lookup and compares the converted global palette offset. It uses no inactivity assumption.

The existing grouped-child test separately exercises a dynamic record selector, nonzero record namespace translation, enclosing visits, mixed frame checks and tuple-arity refusal through actual emission and allocation. The actual Qwen import exercises its 27 grouped endpoint sites and heterogeneous record arities. `sync_qwen_completion_import.pto` adds a production-path regression using one original symbolic plan, specialized only after insertion and allocation.

**Production stages.**

| Stage | Result / scope |
|---|---|
| Decoder → completion map | All 138 records convert; no coloring change or record deletion |
| Logical child preparation/import | Original whole K-loop invocation qualifies with its enclosing frame |
| Parent resource separation | Existing CFG gate accepts; one eventful inline child is retained |
| Whole-plan physical allocation | Existing six-ID pool succeeds, using 0–3 |
| C++ source generation | Succeeds with the pinned compatibility backend; insertion disabled there |
| Device compiler / execution | Not performed |

This is a whole invocation, not an open K iteration. Child summary diagnostics explicitly retain nonempty pending payload effects, `transparent=true`, and pending notification-completion work. Closed SET/WAIT matching is issuance closure; it does not claim completion of all payload accesses or WAIT events. Exact GM selectors remain unavailable where previously unqualified. No exact physical selector is fabricated from the coarse completion summary.

The parent is rebuilt with the imported child as an atomic region. Its unchanged storage transfer selects ALL cuts immediately before and after the K loop in the resulting production IR (in the pinned output, lines 169 and 3361). Those actual cuts, rather than obsolete cuts from the primitive fallback, support the existing resource-state traversal. No completion was added specifically to force resource acceptance. The traversal still handles outer-loop backedges without assuming a single invocation or a caller-specific K bound.

**Independent bounded results.** Both immediate-baseline and candidate whole plans are compiled from the original symbolic input before specialization. The independent access interpreter uses original operands, declared layouts and physical addresses, rather than compiler cells or generators. Its modeled conflict/native graph supplies required C→I order. A separate command graph checks acyclicity, matching identities, invocation confinement and actual allocated consuming-flag reuse, including tail commands and transitions across parent completions. Every applicable child endpoint is also compared with the original native decoder assignment; the checker supplies no replacement coloring.

The campaign covers all 20 named caller blocks at K=17; blocks 0/19 at K=0 through 17; zero outer visits; two and three outer-visit reset stresses; and K=32/64 scaling stresses. There are 59 executions per plan. Reset/multi-visit and large-K cases are explicitly outside the pinned launch/layout contract; they are structural synchronization tests, not native-valid kernel executions. K=0 does not initialize the accumulator, so its synchronization result is not a numerical kernel-correctness claim. Original expert protocols and zero-inner-body barriers are untouched.

For every checked execution, modeled required order is covered and the command graph is acyclic. No executed WAIT lacks its intended SET; SET identities remain distinct across resets and are consumed exactly once. The actual physical assignment passes directional capacity and causal WAIT→republication checks, including surrounding parent commands. These are finite checks under the existing layout/native model, rather than a uniform device proof.

| Execution | Previous excess C→I | Imported excess C→I | Removed / added C→I | Previous commands | Imported commands |
|---|---:|---:|---:|---|---|
| One visit, K=2 | 65 | 0 | 65 / 0 | 15 ALL | 18 SET + 18 WAIT + 3 ALL |
| Named caller, K=17 | 900 | 23 | 877 / 0 | 113 ALL | 186 SET + 186 WAIT + 10 barriers |
| Two visits, K=2 stress | 188 | 58 | 130 / 0 | 29 ALL | 36 SET + 36 WAIT + 5 ALL |
| Three visits, K=4 stress | 659 | 116 | 543 / 0 | 82 ALL | 126 SET + 126 WAIT + 7 ALL |
| One visit, K=32 stress | 1,805 | 0 | 1,805 / 0 | 210 ALL | 378 SET + 378 WAIT + 3 ALL |
| One visit, K=64 stress | 3,661 | 0 | 3,661 / 0 | 418 ALL | 762 SET + 762 WAIT + 3 ALL |

All 59 pairwise comparisons have zero added C→I pairs. Thus relation inclusion, not merely a smaller excess count, holds on these executions. At K=17 there remain 23 extra pairs, so realization-order equality is not claimed. Executed commands increase despite fewer broad completions. At zero K the imported conservative summary produces three barriers versus two before, with no additional payload prerequisite; those barriers were not removed to improve this result.

**Representation and costs.** One unchanged symbolic child serves every K: 171 families, 282 materialized endpoint sites, 27 grouped endpoint sites and 138 allocation records. Normalization visits the supplied map tables and does not enumerate K or outer visits. The existing compact-K/replay scaling regressions pass. The whole root now emits the retained child rather than primitive replacement: static SET/WAIT sites change from 0/0 to 121/128; barriers from 19 to 21 (9 ALL plus 12 local); arithmetic from 68 to 2,521 logically and 2,539 physically. Shared inserted control has 2,480 distinct reachable arithmetic nodes; guard roots reach 2,182 nodes with largest root 92, and selector roots 437 with largest root 10. This task makes the import executable; it does not claim compact-control optimality or a device speedup.

Descriptive single production runs measured logical insertion / allocation / C++ at 0.716 / 0.165 / 0.466 seconds for Qwen down. The immediate baseline logical run was 0.566 seconds. These are process-level observations, not paired performance evidence or isolated demand-generation times. Inclusive profiles are retained separately and are not summed.

All three original kernels pass logical insertion, physical allocation and C++ generation. Persistent GEMM and Qwen finalize remain byte-identical at logical and allocated IR. Qwen down changes intentionally to retain the K child. The structural zero-inner regression still has two extra prerequisites, none missing and valid reuse. Fresh final focused gate: 21/21 pass, including the new import regression and the previous certificate, reset, access/export, grouped frame/arity, completion, profiling and compact-K/replay tests. This is not a full-suite run.

**Reproduction.** Run from `workspaces/frontier-shared-input` using the established disk-backed snapshot build. Source, input, tools, LLVM and compatibility backend hashes, dirty-source manifest, complete subprocess commands, watchdogs and receipts are pinned in the JSON.

```sh
B=.local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt
.venv/bin/python test/lit/pto/Inputs/check_qwen_completion_import.py --trace-tool "$B/pto-sync-input-test" --optimizer "$B/pto-test-opt" --baseline-optimizer .local/constant-child-map/baseline/pto-test-opt --input test/lit/pto/Inputs/pypto_region_survey/qwen-down/input.pto --output .local/constant-child-map/reproduce --full
.venv/bin/python .local/constant-child-map/run_originals.py
```

Builds used at most two workers; campaigns/tests each used one, at most two in aggregate. Compiler watchdogs remain 35 seconds. The first build caught a signed/unsigned comparison in a test reference; that was corrected before qualification. Subsequent builds introduce no warnings. `git diff --check` passes; scoped compliance passes after manual review of braced multiline-control regex false positives. The whole dirty-worktree prefilter includes older findings and is not claimed clean. Dedicated security/complexity analyzers were unavailable.

| Established result | Remaining premise / limitation | Appropriate paper consequence |
|---|---|---|
| Compact child mathematics and coordinates survive completion import | Missing exact GM exports remain explicit; modeled footprint/alias contract unchanged | Section 5.5 results can be reused without inventing a complete physical storage interface |
| Constant-table normalization preserves every applicable assignment | Not a general arbitrary-table conversion or a new allocation theorem | Implementation repair at the retained-map/recipe boundary |
| Parent retains pending effects and checks resource separation on rebuilt CFG | Matching closure is distinct from payload/command completion; native ALL semantics remain a target premise | Qualified Section 8 executable completion fallback, separate from Section 6 conservative demand/token theory |
| Six-ID whole plan and actual-ID reuse pass bounded checks | No uniform native-footprint or device validation; no minimum-budget claim | Physical feasibility for the selected protocol under established certificates/model |
| 877 C→I pairs removed and none added at K=17; finite inclusion checks pass | 23 excess pairs remain, control/command counts increase | Measured bounded ordering improvement, not global maximal concurrency or latency improvement |
