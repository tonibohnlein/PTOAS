# Qwen caller qualification and uniform reuse evidence — 2026-10-20

**Outcome: the pinned caller is identified and its checked K child is feasible with four IDs per direction, but no uniform production import is certified.** The remaining problem is insufficient producer proof, together with caller facts that are not communicated to standalone synchronization analysis. No admitted fixed-plan obstruction was found. The certificate schema can describe the bounded assignments; it is not intrinsically limited to the refused mapping.

This continues the [closed-child checkpoint](frontier-closed-child-validation-20261020.md). The [machine receipt](frontier-qwen-caller-reuse-20261020.json) pins sources, binaries, commands, witnesses and raw artifacts under `.local/qwen-caller-reuse/`. Source HEAD remains `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44` plus manifested dirty sources. This task changes **tests and diagnostic code only**. Production allocation, SCC transfer, protocol, guards, cuts, frame binding and grouped-record translation are unchanged.

## Caller contract

The source is the captured PyPTO-lib revision `f3167ecbf8838ecff0e169ee8d854baafaa0236c`, rather than a restriction inferred from one execution. The generated orchestration preserves the following configuration as actual scalar assignments and launch arguments. These facts have a narrower scope than arbitrary calls of the exported PTO function.

| Premise | Source | Enforcement and scope |
|---|---|---|
| HIDDEN=5120; INTERMEDIATE=17408; chunks=256; four bands of 4352 | `models/qwen3_14b/constants.py:100–110`; `prefill_fwd.py:124,149–151` | Pinned model configuration |
| `%arg3=20`, `%arg4=17` | Generated `orchestration/prefill_fwd.cpp:129–130,758–759` | Caller initializes constants and packs them into the launch |
| Launch has 20 blocks; block index is 0–19 | Same orchestration, line 762; backend `get_block_idx/get_block_num` ABI | Launch requests 20; index range remains a runtime ABI premise, not device validation |
| Exactly one whole paired K child per block | Original outer loop: `hb=block_idx; hb<arg3; hb+=20` | Follows from the named caller and runtime index range |
| Eight paired visits; separate one-chunk tail | Original K loop: `0 .. 2*trunc(arg4/2), step 2` | At K=17, paired visits have `cb=0,2,…,14`; chunk 16 is outside the child |
| Inner L0 visits 0/128, with accumulator initialization only at the first active chunk | Original nested loops and comparisons | Original IR, not a new recognition assumption |
| Layer base and band base vary | Caller computes `layer_idx*17408`, `mlp_band*4352`; weight extent comes from the tensor | Address/input contracts, not notification-rearming evidence |
| Caller constants constrain the standalone synchronization request | Exported PTO signature and attributes | **Not established:** no scalar bounds, assertions or caller assumptions communicate arg3/arg4 constants |
| Parent resource reuse is separated | `CompletionRegion.cpp` resource-state traversal | Existing import gate requires already-selected completion or independently reserved palettes |
| Separate task dispatches start with admissible local notification state | Runtime/native invocation contract | Unqualified here; dependency task IDs are not a local flag-consumption proof |

The parent currently emits ALL boundaries, including before the output store and at return. Those are facts about the **current conservative plan**. A different imported-child graph must rerun the existing storage and resource checks; old boundaries cannot be assumed to survive unchanged. No boundary was added to qualify this task.

Closed child matching means the intended WAITs have issued. It neither empties the pending payload suffix nor establishes completion of the WAIT command events. The existing summary and parent gate retain those distinctions.

### Classification of the width-eight witnesses

Outer bounds 40/60 are **excluded by the pinned generated caller**, which passes 20. They are not rejected by an assertion in the standalone kernel: arbitrary standalone invocations remain unqualified. The stress cases also reach output columns outside the declared 5120-column view. They are structural reset tests, not evidence that this original caller needs eight IDs.

The four band launches and dynamic layer/window executions are separate task invocations, not multiple K-child visits within this kernel invocation. Their entry resource state still depends on the runtime/native task contract. Task dependency metadata alone is not promoted into a SET/WAIT reuse certificate.

Coordinates retain their existing meanings: outer `hb` distinguishes child invocations; paired `cb` and inner L0 ordinals reset inside each invocation. Physical coordinate zero is the source ordinal. The logical original-record selector is a different operand; no label-to-tuple substitution was introduced.

## Independent results from the original symbolic recipe

The recipe is analyzed/prepared once with its original enclosing frame, then the **emitted recipe** is evaluated at different parameters. No easier analysis is obtained by specializing and reanalyzing the input. The full sweep uses the previous native tool; the final diagnostic tool emits a **byte-identical symbolic recipe**, SHA-256 `df4d075c85cc542cf0d69434988cf89c00a9fe984f6dc61cbf27ff9ab46e0830`.

The 44-case sweep includes all 20 launched block indices at K=17; K=0 through 17 at block zero as explicitly labeled short-K derivatives; zero visits; an out-of-launch no-work block; two outer-reset stresses; and K=32/64 scaling stresses outside the named caller/layout. Dynamic layer/band bases are sampled at zero; this is not an exhaustive dynamic-layer/address ABI theorem.

Independent checks use original access operands/layouts, command matching identities and asymmetric command causality, rather than compiler cells or demand generators. They verify payload mapping, child-local required-order coverage, exactly one intended WAIT per SET, no unmatched WAIT, invocation confinement, distinct identities across resets, acyclicity and causal physical reuse when a six-ID assignment exists. Pending payload completion is not asserted. K=0/1 matching is vacuous for this paired child; it is not a numerical correctness claim for the accumulator or for the tail outside it.

| Scope | Directional minima `(2→3, 3→2, 3→4, 4→3)` | Conclusion |
|---|---|---|
| Every named block at K=17 | `(2,2,4,4)` | Bounded fixed-graph feasibility under six IDs |
| One visit, K=2/3 derivative | `(2,2,0,4)` | Same distinction; separate tail remains outside child |
| One visit, K=4 through 17 derivative | `(2,2,4,4)` | Initialization, accumulation and odd/even tails checked |
| Zero/no paired visits | `(0,0,0,0)` | No child notifications |
| Two unseparated visits, outer bound 40, K=2 | `(2,4,0,8)` | Outside named caller; fixed child-only stress graph |
| Three unseparated visits, outer bound 60, K=4 | `(2,4,4,8)` | Outside named caller; fixed child-only stress graph |
| One visit, K=32/64 | `(2,2,4,4)` | Structural scaling only; outside declared GM footprint |

Minima use the existing independent maximum-matching chain cover plus an equal-sized checked antichain of the strict order `WAIT(h) reaches SET(g)`. Applicable premises are a fixed finite acyclic matched command graph, independent directions and a fresh-entry resource scope; logical matching closure alone does not reset physical flags for another visit. Concrete assignments using IDs 0–3 pass causal consumption-before-republication. Stress cases requiring eight are retained as witnesses, not allocated from an enlarged pool.

## Why the uniform certificate refuses

The supplied certificate still has **78 groups and 102 lanes**. On K=17 only 26 lanes are active. Lane 18—the first exhausted lane in the existing sufficient greedy path—is inactive. Its records 43–46 cannot simply be deleted from production because the current producer has not proved the required uniform absence.

The original failure remains: some supplied successor lists are nontransitive, so the strict chain route refuses; its sufficient greedy fallback exhausts IDs 0–5. This is not a malformed tuple schema or a fixed-command-graph lower bound. No transitive closure was taken.

There are **37 ordered active-lane pairs** at K=17 whose actual WAIT→SET order is missing from supplied evidence. A second independent attribution checks the concrete consumer payload and subsequent producer payload: all 37 also satisfy the stronger consumer-completion→producer-start relation in the modeled required closure. Thus these particular misses are not explained by asymmetric command semantics alone. They expose incomplete uniform proof or envelope evidence.

An offline constant-lane map fits four IDs and passes causal reuse across all 39 named-caller/short-K/zero-visit cases. It uses existing tuple-rule forms. It is an **expressibility witness only**: it is not serialized as trusted producer evidence, installed in production, or claimed uniform for arbitrary K/callers.

### The first proof obligation is available but unproved

The new explicit diagnostic `test.contract_lane_pair` reproduces the producer's queries. For lane **12→24**, all 16 last-target/first-source combinations have an available query. None is discharged by existing cheap implication, constant-under-premise or threshold implication proofs. This separates an unavailable query export from an unproved sufficient implication.

One representative activity is

`min(T,0) < min(T,1) && min(T,1) < T`,

where `T` is the nonnegative paired trip count derived from the original signed bound. Under its derived range, this says `T≥2`. The corresponding query circuit becomes true on that domain. A local validity argument can simplify the min/select expressions using `T≥0` and the active interval. The producer currently carries the circuit, but does not discharge this numerical implication. The detailed typed DAGs are retained in `lane-12-24.json`.

`RegionExpressions::implies` deliberately performs no arithmetic propagation; `constantUnder` is a sufficient DAG specialization. `inactiveByThresholds` partitions direct **Input** comparisons. Here the critical comparisons concern the derived count/min expressions. An explicit diagnostic attempt with the existing threshold prover also returns unproved. No unknown result was treated as true.

This is enough to locate the gap, not enough to certify every inactive lane and all required reuse relations. Installing the finite coloring, assuming the caller's constants, or blindly closing the guarded successor relation would bypass missing evidence. No such production change was made.

The smallest justified follow-up is to carry/check the normalized count and phase-interval facts at the existing repetition/certificate boundary, then use them to prove lane inactivity and envelope implications. If qualification relies on the named caller, its scalar/launch contract must first reach that request explicitly. A caller-specific constant restriction must not be inferred from the function name or tensor geometry. A general proof abstraction or schema/contract extension needs its own validity argument; it was not smuggled into this adapter task.

## Validation, acceptance and scaling

Fresh focused gate: **16/16 pass**, 82.98 seconds, one test worker. It includes the new caller regression, absent-frame refusal, grouped selector/arity checks, qualified adapter, completion/access tests, compact K/replay and zero-inner behavior. Scoped compliance passes after review of two braced-control regex false positives; `git diff --check` passes. The whole dirty-worktree checker still reports prior-checkpoint findings; no whole-tree clean claim is made. Development analyzer/device qualification remains unavailable.

The qualified grouped fixture is compared freshly with the old production adapter under the same conservative policy. At 1/2/3 visits, it still removes 3/6/9 C→I pairs with **zero added pairs**; matching and six-ID causal reuse pass. This task does not change that adapter or its protocol.

| Original kernel | Logical / six-ID / C++ | Acceptance or output change |
|---|---|---|
| Persistent GEMM, may-alias | pass / pass / pass | None; byte-identical logical and allocated IR |
| Qwen down, may-not-alias | pass / pass / pass | None; K child still physically unqualified; root remains conservative |
| Qwen finalize, may-not-alias | pass / pass / pass | None; byte-identical logical and allocated IR |

Zero-inner retains exactly two extra prerequisites, zero missing requirements and valid reuse. Its barriers remain unchanged.

One symbolic retained certificate/recipe serves the sweep: 171 families, 282 materialized endpoint sites, 27 grouped endpoints, 78 groups/102 lanes. No trip-count enumeration is added to production certificate construction. Execution interpretation grows: K=17/32/64 produces 372/756/1524 commands and 186/378/762 handoffs. Those are dynamic trace sizes, not compiler representation growth. The focused compact-K/replay structural bounds also pass. No new uniform certificate was produced, so there is no claimed scaling result for a successful new certificate algorithm.

Preparation, allocation and C++ subprocess timings are in the receipt. Single runs are descriptive and inclusive timing columns are not summed. No speedup, native footprint, fresh-device-entry or device-execution result is claimed.

## Reproduction

Run serially from `workspaces/frontier-shared-input`. Artifacts stay on disk; the diagnostic scripts enforce one CPU, a 4-GiB address-space cap and 35-second compiler subprocess watchdogs. Do not run another intensive job alongside a two-worker build.

```sh
B=.local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt
.venv/bin/python test/lit/pto/Inputs/check_qwen_caller_reuse.py --trace-tool "$B/pto-sync-input-test" --input test/lit/pto/Inputs/pypto_region_survey/qwen-down/input.pto --output .local/qwen-caller-reuse/reproduce --full
.venv/bin/python .local/qwen-caller-reuse/attribute.py
.venv/bin/python .local/qwen-caller-reuse/run_originals.py
.venv/bin/python test/lit/pto/Inputs/check_grouped_child_adapter.py --trace-tool "$B/pto-sync-input-test" --optimizer "$B/pto-test-opt" --baseline-tool .local/closed-child-validation/baseline/pto-sync-input-test-fixture --input test/lit/pto/Inputs/completion_grouped_visit.pto --output .local/qwen-caller-reuse/adapter
```

`attribute.py` reads the pinned `full/` sweep, not the optional `reproduce/` destination. All exact subprocess and focused gate commands, build/source identities and dirty-source hashes are retained in the receipt and logs. The full sweep's previous tool and final diagnostic tool are separately pinned, with recipe byte identity checked.

| Established result | Remaining limitation | Appropriate paper consequence |
|---|---|---|
| Compact child representation and exact modeled demands/interfaces remain intact | Native footprint and alias premises unchanged | Section 5.5 evidence preserved |
| Conservative parent still carries pending effects and uses existing completions | Closed matching is not payload completion | Separate Section 6 demand abstraction from executable protocols |
| Named caller executes one paired child per launched block | Caller constraints absent from standalone request; runtime ABI remains a premise | No unrestricted caller/repetition claim |
| Matched bounded K=17 graphs fit four directional IDs | Producer cannot yet certify lane absence and reuse uniformly | Separate Section 7 proof obligation, not infeasibility |
| Frame/group adapter and six-ID decoder gates remain intact | No additional import accepted | Section 8 implementation preserved; no general token-to-SET/WAIT theorem established |
| Three originals compile, allocate and generate C++ unchanged | Host-only evidence | No native/device correctness or latency claim |

No theoretical token construction was edited. The precise final classification is **certificate producer evidence remains insufficient despite bounded feasible examples; standalone caller qualification is unresolved**. There is neither a uniform six-ID import nor a demonstrated obstruction for the named caller.
