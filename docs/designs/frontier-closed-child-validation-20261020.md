# Closed-child matching, grouped translation and physical refusal — 2026-10-20

The reusable adapter now binds the original enclosing frame and preserves grouped record selection. A qualified supplied two-bank child is imported successfully and removes ordering on the checked executions. The original Qwen down K invocation has independently checked logical matching, but **still lacks a uniform six-ID certificate**. Production retains its previous conservative plan. No allocation algorithm or SCC transfer was changed.

Baseline: [compact-child checkpoint](frontier-compact-completion-child-20261020.md). Full commands, hashes, assignments, witnesses and outcomes: [machine receipt](frontier-closed-child-validation-20261020.json). Raw artifacts are under `.local/closed-child-validation/`. Source HEAD is `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44` plus the pinned working sources; HEAD alone does not describe this dirty workspace.

## Implementation and validity argument

`Shared/SequenceAnalysis.cpp::prepareSequenceInsertion` supplies the retained `requiredOuterLoops` to the existing contextual preparation service. An explicitly wrong or absent contextual frame still refuses. No coordinates are inferred from an unrelated parent.

`Shared/CompletionInlineChild.cpp` checks the arity of **every** possible record label, including grouped endpoints, against the decoded certificate. Append translates every label by a checked namespace offset, adjusts the dynamic record-selector coordinate by that offset, and appends enclosing invocation coordinates. Singleton endpoints retain the previous constant construction. It does not substitute the first label for the dynamic selector.

Physical tuple rules have a different schema: coordinate zero is the source ordinal, followed by matching visit coordinates; the original-record selector is excluded. Namespace translation therefore does **not** shift tuple-rule terms or phases. Existing tuple arity, ownership, matching scope, endpoint confinement, cleanup and decoder gates remain mandatory. An arity-mismatched certificate is rejected.

This preserves the selected protocol: guards and cuts are unchanged, the namespace translation is injective, and both endpoints use the same original occurrence mapping and enclosing visit frame. The adapter imports a closed whole invocation; it does not independently close an open iteration. Missing GM selectors remain unavailable. Pending read/write effects remain in the existing conservative summary, with `transparent=true`; closed matching is not payload completion.

Changes are in the existing provider/export and completion services. `CompletionGroupedChildChecks.cpp` is a **test-only supplied protocol producer**, using the existing certificate schema and decoder. It is not a recognition backend or an allocation construction. The fixture needs no new production pattern recognition.

## Independent logical checks for original Qwen K

`check_closed_child_logic.py` prepares the original symbolic region 5 once using its enclosing frame, then evaluates that emitted recipe at bounded parameters. It does not reanalyze a specialized source to obtain an easier recipe. The independent interpreter constructs concrete accesses/conflicts under the stated relative-layout model, without using compiler cell partitions or generators.

Checks cover outer visits/K `(0,2), (1,0), (1,1), (1,2), (1,3), (1,4), (1,6), (2,2), (2,3), (3,4)`. Initialization/accumulation and tail branches are exercised. Every executed publication has exactly one intended WAIT in the same whole invocation, no unmatched WAIT occurs, identities distinguish visits and counter resets, and the command graph is acyclic. All checked child-local completion-to-start requirements are covered.

The matching check establishes no unmatched logical publications at entry/exit. It does **not** establish completed payloads, or that issued WAITs have completed before the next invocation republishes a physical flag. K=0 has no child notifications; that is a vacuous matching result, not numerical correctness of an uninitialized accumulator. Coverage here is the child, not all tail/store obligations outside it.

Multiple outer visits use `%arg3=40/60` as structural counter-reset tests. These are relative-layout stress cases; the original configured HIDDEN=5120 may restrict the caller to at most one such visit. Native ABI validity is not established for the stress parameters. Native instruction footprints and hardware issue semantics remain separately qualified.

## Six-ID refusal: precise scope

The original recipe has 171 families, 282 endpoint sites and 27 grouped endpoints. All **138 event records** have consistent endpoint arities and agree with the certificate coordinate counts. The default original-frame preparation now succeeds. The remaining refusal is lane-reuse evidence, not a record tuple-arity mismatch.

The current certificate describes 78 groups and 102 lanes. The decoder's strict-order route rejects nontransitive supplied evidence: direction 2→3 admits lanes 0→84→96 but lacks 0→96; direction 3→4 similarly has 4→36→45 without 4→45. Its existing sufficient greedy fallback first exhausts IDs 0–5 at lane 18 (group 6/local 2, direction 3→4, records 43–46); blocking predecessor lanes are 4,5,6,7,16,17.

Direction 4→3 has supplied lane-order width 12. **This is not the physical width of the actual command graph.** `Shared/RegionalEvents.cpp::regionalHandoffReuse` uses stronger payload/boundary conditions (including consumer-completion to producer-start between pipes), rather than the full command WAIT-to-SET relation. These sufficient relations can omit legal reuse. Taking transitive closure of guarded metadata without a proof is not a repair. A retained optional candidate exists; this diagnosis does not claim to exhaust every alternative certificate.

For each manageable concrete execution, the independent checker forms the handoff order `WAIT(h) reaches SET(g)`. It verifies acyclicity, complete matching, strictness and transitivity, constructs a minimum chain cover by bipartite matching, and verifies an equal-sized antichain. This applies the fixed finite matched-command-graph theorem (`event_id_reuse.tex`, lemma `lem:shared-minimum`), with directions separate and fresh-entry/no inherited notification assumptions. It is a theorem about this supplied finite graph, not arbitrary realizations of the same payload order.

| Execution | Minimum directional requirements |
|---|---|
| One visit, K=2 or 3 | 2→3: 2; 3→2: 2; 4→3: 4 |
| One visit, K=4 or 6 | Above, plus 3→4: 4 |
| Two unseparated visits, K=2 or 3 | 2→3: 2; 3→2: 4; 4→3: 8 |
| Three unseparated visits, K=4 | 2→3: 2; 3→2: 4; 3→4: 4; 4→3: 8 |

Concrete assignments for executions fitting six use IDs 0–3 and independently pass causal consumption-before-republication checks. No larger-pool assignment is used to force acceptance. Width eight concerns the fixed **unseparated child-only** execution; established parent completions can change that graph. Neither that witness nor the decoder refusal proves kernel infeasibility. The remaining obligation is a uniform certificate tied to actual child lifetimes and the permitted caller boundaries.

## Qualified adapter and ordering comparison

The supplied fixture has a finite two-bank child, a real preceding finite WAW child, and an outer invocation coordinate. Its two grouped labels are translated to 1/2 at a nonzero namespace offset while the source ordinal remains 0/1. The supplied existing-schema mapping uses those ordinals to select IDs 0/1. Independent access and command checks validate matching, causal reuse, pending effects, parent coverage and six-ID allocation for zero through three outer visits. A deliberately wrong tuple arity refuses the import while preserving primitive-parent coverage.

The baseline comparison uses the **same fixture helper and same conservative policy with the old production adapter**, not a different synchronization policy. The revised parent consumes the child's conservative pending summary and keeps the child's original SET/WAIT sites.

| Outer visits | Previous excess C→I | Revised excess C→I | Removed C→I | Added C→I | Previous / revised executed commands |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 2 / 2 |
| 1 | 4 | 1 | 3 | 0 | 7 / 8 |
| 2 | 8 | 2 | 6 | 0 | 12 / 14 |
| 3 | 12 | 3 | 9 | 0 | 17 / 20 |

These are actual relation-set comparisons: the revised C→I relation is contained in the old one on each checked execution. This is stronger than comparing excess counts, but remains bounded evidence. Static sites change from 0 SET/0 WAIT/6 barriers to 1 SET/1 WAIT/4 barriers (barrier totals include the unchanged local WAW barrier). More commands can preserve more payload concurrency.

The supplied child stays at one family, two records and two endpoint sites; appended outer coordinates do not enumerate trips. Candidate and baseline both allocate under the existing six-ID configuration and generate C++. These results qualify the adapter, not a uniform production certificate for Qwen K.

## Regression and whole-kernel results

Fresh focused gate: **15/15 pass**. An earlier run caught extra singleton arithmetic caused by unnecessarily constructing a selector addition. Restoring the existing singleton constant path fixed it; no scaling bound or watchdog was relaxed. The formatting-only follow-up rerun passes 1/1. Scoped compliance checks pass after reviewing braced-control regex false positives; the whole dirty-worktree checker still reports unrelated prior-checkpoint findings. `git diff --check` passes. The full compiler suite was not rerun.

Compact-K/replay tests retain access agreement, actual expanded regional covers, all four event queries, selectors, matching and reuse. Replay counts 1/2/3/8 retain 124 repeated types and largest numerical child 8, with arenas 125/141/141/143. K=4/8/16 retains 332 types and numerical child 14, arenas 192/196/200. Static stages remain a separately varying size dimension. The zero-inner-body regression still has exactly two extra C→I prerequisites and zero missing ones; its barriers were preserved.

| Original whole kernel | Alias policy | Logical / six-ID / C++ | Static SET / WAIT / barriers | Before/after |
|---|---|---|---|---|
| Persistent GEMM | may-alias | pass / pass / pass | 0 / 0 / 8 | byte-identical logical and allocated IR |
| Qwen down | may-not-alias | pass / pass / pass | 0 / 0 / 19 | byte-identical logical and allocated IR |
| Qwen finalize | may-not-alias | pass / pass / pass | 2 / 2 / 3 | byte-identical logical and allocated IR |

Candidate process timings (logical / allocation / C++) are respectively 0.315/0.166/0.669 s, 1.473/0.217/0.776 s, and 1.724/0.216/0.822 s. These are descriptive single runs, not speedup evidence; inclusive stage profiles are not summed. No extra original-kernel import is accepted because its physical gate remains unchanged.

## Reproduction and remaining obligation

Run from `workspaces/frontier-shared-input`; scripts cap interpretation at one CPU, use bounded watchdogs and keep artifacts on disk. Build with `--parallel 2` only when no other intensive job runs. Snapshot build/source and tool hashes are pinned in the receipt.

```sh
B=.local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt
.venv/bin/python test/lit/pto/Inputs/check_closed_child_logic.py --trace-tool "$B/pto-sync-input-test" --input test/lit/pto/Inputs/pypto_region_survey/qwen-down/input.pto --output .local/closed-child-validation/reproduce-logic
.venv/bin/python test/lit/pto/Inputs/check_grouped_child_adapter.py --trace-tool "$B/pto-sync-input-test" --optimizer "$B/pto-test-opt" --baseline-tool .local/closed-child-validation/baseline/pto-sync-input-test-fixture --input test/lit/pto/Inputs/completion_grouped_visit.pto --output .local/closed-child-validation/reproduce-adapter
.venv/bin/python .local/closed-child-validation/diagnose_certificate.py
.venv/bin/python .local/closed-child-validation/run_originals.py
```

Detailed gate command and all subprocess commands are retained in the receipts/logs. The turn-start old tools and old-adapter/same-fixture tool are separately hashed. This matters because unrelated earlier checkpoint edits were already present in the worktree.

The smallest next resource investigation is a uniform actual WAIT→SET reuse certificate for the original whole K invocation, checking whether existing periodic/ordered certificate interfaces can express it. That is separate work: no new allocation algorithm, synchronization, pool expansion or certificate bypass was introduced here. GroupNorm arity, atomic footprint and frontend compatibility remain outside scope.

| Established result | Remaining premise or limitation | Appropriate paper classification |
|---|---|---|
| Original compact K mathematics and bounded interface regressions retained | Model/layout qualification remains; no new exact GM recovery | Section 5.5 composition evidence unchanged |
| Conservative pending payload summaries retained | Matching closure is not payload completion | Section 6 demand abstraction preserved |
| Original-frame preparation and all grouped labels translated with dynamic selector | Physical/ownership gates still mandatory | Section 8 executable fallback implementation gaps repaired |
| Qualified fixture removes C→I prerequisites without additions in bounded tests | Supplied finite certificate; not uniform Qwen certification | Section 8 bounded executable composition evidence |
| Small single-visit Qwen graphs fit four IDs per direction | Existing uniform lane evidence insufficient; repeated child-only graphs can need eight | Separate Section 7 resource obligation |
| Original three kernels still compile and allocate identically | No native/device execution | Host compiler feasibility, not native correctness or device performance |
