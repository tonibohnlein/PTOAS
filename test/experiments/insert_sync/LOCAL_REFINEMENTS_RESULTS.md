# Local descriptor, graph and command refinements

Measured on 2026-09-08, on an **uncommitted incremental build** over
`ce472ef8cb26e65eacb39867ccf7f3a82acffb0f`. The comparison compiler is the
previous shared-generation implementation, `3400426f3`, not original InsertSync.
The [evidence JSON](LOCAL_REFINEMENTS_RESULTS.json) records both native-library
fingerprints, all 205 diagnostic rows, unchanged input hashes, mechanism counts,
and concrete replays. The [CSV](LOCAL_REFINEMENTS_RESULTS.csv) gives the per-row
comparison. Raw commands, IR and diagnostics remain under
`insertsync-builds/campaign/refinements-ce472ef8c` beside the checkout.

All **205/205 available single-function pass-entry cases** compile, verify and
preserve payloads, allocation expressions, views and function interfaces. This
reruns the existing representative diagnostic population, not all 9,754 rows.
The other eight representative rows were not newly tested: five previously
failed before InsertSync and three did not provide single-function diagnostics.
Device numerical correctness and runtime were **not run**.

## What changed

**Descriptors and physical operations.** The structural importer now interprets
direct local allocation and `set_validshape` as scalar descriptor state on its
existing guarded graph. A write qualifies as whole-slot production only when
every reaching descriptor version has the full physical extent. Branch joins
intersect that fact; reallocation resets it; loop backedges retain incoming
alternatives. Unknown or partial descriptors retain conservative accesses and
cannot qualify a complete whole-slot construction. A later proven full update
can restore qualification. Descriptor operations never establish lane completion.

The importer reuses scalar-prerequisite qualification and existing target
pipeline/memory-effect interfaces for arithmetic, conversion, reductions and
row/column expansion. Scratch reads and writes remain represented. Vector
`tfillpad` is admitted; MAT padding is not. Scalar prerequisites derived from
unmodeled asynchronous operations, forwarded mutable descriptor handles,
descriptor-derived scalar chains, unknown helpers and hidden resources retain
conservative rejection. This is qualified coverage, not universal descriptor or
operation support. Production InsertSync's effect contracts are not weakened.

**Graph representation and reuse.** Impossible finite-domain branches are
discarded before their bodies are copied. A constant two-trip loop has a first
and last occurrence, without an impossible middle recurrence; other loops keep
the existing guarded recurrence. The 2,048-node construction limit, 8,192-node
guard product limit and analysis-work limits are unchanged. Control-only ordered
reachability and immediate lane predecessors are shared between compatible
slice/bundle generation queries. Exact graph comparison prevents stale reuse;
plan-dependent completion is not cached as an immutable control fact.

**Command cost.** The existing checked lifecycle-supply cleanup can remove an
owned scope-entry publication and its exclusively initial acquisitions. A wait
site shared with recurring acquisitions is retained. Deletion requires the
complete event proof, retained requirements, unchanged guarded payload completion
prefixes, exit completion and fresh emitted reconstruction. It adds no guards,
moves no endpoints and changes no allocation. This extends the existing cleanup;
it does not introduce another planner or kernel-name recognizer.

## Previously recorded gates

| Previous gate | Rows | New outcome |
| --- | ---: | --- |
| Descriptor/non-payload import around `set_validshape` | 59 | 24 select lifecycles; 35 reach conservative recipe rejection |
| Missing qualified physical-operation adapter | 33 | 1 selects lifecycles; 11 reach recipe rejection; 21 remain unsupported |
| Concrete reconstruction node budget | 32 | All 32 select lifecycles |
| Initial structural node budget | 4 | All pass graph import; allocation retries still fall back |

Accepted lifecycle rows increase **23 → 80**. No row now stops at either
structural-node gate or the former descriptor gate. Six rows still exhaust the
later lifecycle-recognition work budget. Another 25 retain effect-coverage
rejection and 15 retain fixed synchronization/helper restrictions. These are
optional optimization outcomes, not claims of defects in production InsertSync.

For the large `lm_head_matmul`, four candidate channels now reach realization.
The dedicated allocator finds all eight keys occupied in MTE2→MTE1 or MTE1→M
and retries without committing. Its final summary says no complete lifecycle;
the retained attempt diagnostics identify the actual resource obstruction.
This is not proof that hardware serialization is unavoidable. The fallback
inventory remains 94/94, MTE2=4, MTE1=2, M=17, FIX=1, `PIPE_ALL`=1.

## Concrete changes on unchanged inputs

Each flags cell is **set / wait sites**. Named barriers and `PIPE_ALL` are
separate; the mechanisms are never combined into a performance score.

| Kernel | Previous flags | New flags | Named barriers, previous → new | `PIPE_ALL`, previous → new |
| --- | ---: | ---: | --- | ---: |
| `markov_logits` | 15 / 15 | 21 / 21 | M=1 → M=1 | 1 → 0 |
| `exp_gate_mm` | 30 / 30 | 53 / 53 | MTE2=1, MTE1=2, M=8, FIX=1 → M=6 | 1 → 0 |
| Small `gemm_tile` | 14 / 14 | 13 / 13 | none → none | 1 → 1 |
| Corpus `proj` | 19 / 19 | 17 / 17 | M=3 → M=3 | 1 → 1 |
| Qwen softmax | 15 / 15 | 15 / 15 | V=14 → V=14 | 1 → 0 |
| Qwen RoPE/KV-cache | 15 / 15 | 15 / 15 | V=12, MTE3=2 → same | 1 → 0 |
| Qwen online softmax | 12 / 12 | 20 / 20 | V=20 → V=20 | 1 → 1 |

`exp_gate_mm` commits 11 channels on its first attempt, including six consumer
regions and 13 guarded actions. Fewer barriers with more flags and conditions
does not establish a runtime improvement.

The small `gemm_tile` executes **28 → 27 pairs**; corpus `proj` executes
**517 → 515 pairs**, at their unchanged recorded bounds. Independent replay
finds identical cross-lane completion prefixes at every physical operation and
identical scalar-control counts. Their earlier readiness/release improvements
survive. These are distinct fixtures from Qwen `q_proj`, which remains 39/39
with M=6 and one exit drain.

Across the 205 rows, 99 inventories change. Forty-seven rows lose `PIPE_ALL`;
none gains it. Named barriers decrease on 84 rows for MTE2, 36 for MTE1, 77 for M,
and 42 for FIX; none increases. V and MTE3 inventories are unchanged. Five rows
lose set and wait sites, while 58 gain sets and 56 gain waits. Ten rows use the
initial-pair cleanup. These row counts include related generated instances.

## A remaining command regression

New operation coverage lets online softmax select two lifecycles. At the sampled
16 bound, it executes **96 → 188 pairs**, with the same 20 static V barriers and
exit drain. Independent replay at bounds 0, 1, 2 and 16 finds **no differing
cross-lane completion prefixes**; scalar control is unchanged. At bounds 0/1 it
executes 6 → 8 pairs, and at bound 2 it executes 12 → 20.

This is a demonstrated command-cost regression in those executions, not a
measured device slowdown. The implementation does **not** yet choose the cheaper
ordinary construction when a newly admitted lifecycle offers equivalent
completion boundaries. That is the next concrete lowering/plan-selection target;
removing the new operation adapter would only hide it. General joint event
allocation also remains unfinished, as `lm_head_matmul` demonstrates.

## Existing benchmark and validation

The original 11 fixtures retain all 33 manual/combined/staged inventories from
the previous report. All 66 PTO/C++ emissions and 306 scalar replays pass.
In particular, historical GEMM remains **56/56, no named barriers, no
`PIPE_ALL`**. Existing Conv2D device failures and FlashAttention device-build
restrictions are not resolved or relabeled by this local compilation result.

All eight Qwen additions compile with both original and new passes: 32 PTO/C++
emissions plus eight native diagnostics, with identical payload/allocation/view/
ABI contracts. QK remains 22/21 static sites with its guarded participation and
earlier first-panel publication; Q projection's six-pair readiness sharing is
preserved. RMSNorm, SiLU and SV inventories are unchanged from `3400426f3`.

Focused validation: 63 C++ generation/control/token checks, six descriptor
variants, 15 unchanged/renamed/empty/one/two-trip native loop cases, 13 existing
native lit tests, and 19 accounting tests pass. The four initial lit failures
were caused by the temporary serial-tool wrapper passing an extra argument to
Python helpers; correcting the wrapper made all four pass without test changes.

The new command/graph fixtures are frozen pass-entry inputs with hashes in
`test/lit/pto/Inputs/local_refinements/manifest.json`. Their baselines retain
the prior completion boundaries. Tests exercise concrete token participation
and command cost rather than trusting the planner's optimization counters.

The 205 native invocations took 61.5 seconds in aggregate versus 49.2 seconds
in the earlier campaign. These are process elapsed times from separate runs,
not a controlled compile-time comparison. More rows now execute construction
and verification. Cache reuse is checked by equivalent facts, fewer budgeted
operations and invalidation after changed control; no global speedup is claimed.
