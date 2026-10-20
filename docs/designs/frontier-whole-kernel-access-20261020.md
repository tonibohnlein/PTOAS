# Whole-kernel synchronization with incomplete accesses — 20 October 2026

The ordinary production path completes logical insertion for **11/14** original
kernel/policy combinations, physical allocation and C++ generation for **10/14**.
The existing explicit `completion-only=true` policy completes all three stages
for **14/14**, with substantially more payload order on the checked small cases.
These are different quality policies, not equivalent synchronization plans.

The [machine receipt](frontier-whole-kernel-access-20261020.json) pins inputs,
source/tool hashes, actual commands, stage profiles and finite checks. Durable
raw data is under `.local/whole-kernel-access/` in this worktree. No production
analysis, allocation algorithm, access importer, expert protocol, admission limit
or zero-inner-body barrier was changed. Only read-only diagnostics and tests
were changed. Native/device qualification is not established.

## Contracts and source audit

The source is `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44` plus the
manifested working tree on `codex/frontier-shared-input`. This is not a clean
commit receipt. Production files compared with the preceding storage-export
receipt have identical hashes. The source-pinned native build was reused;
only the diagnostic executable was incrementally rebuilt, without an LLVM build.

Target A3; physical pool 0–5; memory replanning and repeated synchronization
insertion are disabled during C++ generation. Both GM alias policies are tested.
Each native request has the same **35-second watchdog**, 4-GiB address-space cap
and one-CPU affinity. Campaigns use one worker each and never exceed two
intensive workers in aggregate. Timeout evidence is retained; no longer retry
budget or new compiler cutoff was introduced.

Relevant source boundaries:

- `InsertSync/SyncAccessRegion.cpp` and `SyncEffectRanges.cpp::resolveEffectRanges`
  recover modeled regions and retain available allocation bounds.
- `InsertSync/SyncStorageEffects.cpp` shares regions/cells and overlap queries;
  `MemoryDependentAnalyzer.cpp` applies physical overlap and GM root policy.
- `Shared/CompletionStorage.cpp` derives conservative storage classes;
  `CompletionRegion.cpp` supplies the structured graph, readiness and children;
  `CompletionTransfer.cpp` runs the existing SCC-local policy.
- `Shared/RegionalSelectorPartition.cpp` keeps exact selected-model answers,
  proved absence and unavailable domains distinct.
- `AnalysisRequest.cpp` and `FrontierAnalysis.cpp` retain mathematics and select
  the requested realization policy; `ResultGuarantees.cpp` reports its scope.

An unknown absolute GM base is compatible with an exact **relative** affine
access. `symbolic` in the diagnostic is a representation state, not automatically
an imprecise footprint. Same canonical GM roots retain their relative addresses.
Distinct roots may overlap under `may-alias`; offsets relative to different roots
are not comparable absolute addresses. `may-not-alias` requires the caller to
provide disjoint distinct root allocations. Derived views/addptrs and repeated
uses of one root are not fresh allocations. Local storage identity uses physical
ranges, including reuse by different SSA roots.

The four prepared ports explicitly record disjoint-GM contracts in `port.json`.
The PyPTO inputs have no independently established wider overlapping-buffer ABI;
we do not assert that arbitrary root overlap is legal for those model entries.
Their `may-alias` compilation is a more conservative policy test on the same
admitted disjoint executions, not permission to violate their ABI.

The importer already uses the requested enclosure hierarchy where proved:
exact declared region; enclosing allocation; union of known slot alternatives.
It checks provenance rather than trusting a carried pointer's cached initial
root. If no such bounded geometry is established, the effect stays unresolved.
The completion adapter maps a known-space unresolved effect to all classes in
that space; an unknown-space effect touches all classes. Overlapping local
reservations are merged. GM currently uses one coarse class even when argument
disjointness is supplied. Modes remain read/write; Qwen's atomic Add has both
GM read and write effects, neither silently omitted nor assumed commutative.

This is a conservative **transfer summary**, not a fabricated exact whole-GM
selector. It cannot solve unsupported scalar readiness, unknown operation
semantics, or a failed importer. A complete cell inventory for one domain never
makes missing GM answers empty. Exact reduction of a supplied enclosure is
minimal for that enclosure model, not necessarily for native instruction bytes.

## Access inventory

Counts are static modeled effects, not executed byte incidences. “Declared” is
provenance, not a native-footprint certificate. “Enclosure” includes unresolved
records whose fallback provenance is enclosure but for which no bound was built.

| Kernel | Declared-region records | Enclosure-provenance records | Unresolved | Qualification |
|---|---:|---:|---:|---|
| Persistent GEMM | 21 | 0 | 0 | Supplied NZ/GM layouts and accumulator protection |
| Elementwise | 13 | 0 | 0 | Supplied vector layouts; disjoint port ABI |
| GroupNorm | 58 | 35 | 0 | 35 allocation/scratch enclosures; scalar readiness separate |
| Chunked cumsum | 9 | 1 | 0 | One enclosure; scalar get/set readiness separate |
| Qwen down | 78 | 3 | 2 | Atomic GM read/write unresolved; local pipeline still modeled |
| Qwen finalize | 8 | 3 | 0 | Three local/scratch enclosures; finite GM body exported |
| DeepSeek RMSNorm | 60 | 31 | 0 | 31 local/scratch enclosures; root selectors still qualified separately |

## Original whole-kernel matrix

M/Q/S are separate exact **selected-model** mathematics, query and selector
capability requests. Readiness does not promise an answer for every GM address.
These diagnostic requests do not alter the ordinary production request. An
unavailable selector is not an empty answer and does not by itself make the
logical/physical kernel unusable. A retained exact answer is not silently
weakened when its executable export fails.

| Kernel | Alias | M / Q / S | Ordinary logical route | Physical, six IDs | C++ |
|---|---|---|---|---|---|
| Persistent GEMM | may-not-alias | ready / ready / ready | identity-sequence, compact phased/numerical children | pass | pass |
| Persistent GEMM | may-alias | unavailable / unavailable / unavailable | completion-transfer | pass | pass |
| Elementwise | may-not-alias | ready / ready / ready | sequence | pass | pass |
| Elementwise | may-alias | ready / ready / ready | refused | — | — |
| GroupNorm | may-not-alias | ready / ready / ready | sequence | refused | — |
| GroupNorm | may-alias | ready / ready / ready | refused | — | — |
| Chunked cumsum | may-not-alias | ready / ready / ready | sequence | pass | pass |
| Chunked cumsum | may-alias | ready / ready / ready | sequence | pass | pass |
| Qwen down | may-not-alias | unavailable / unavailable / unavailable | completion-transfer | pass | pass |
| Qwen down | may-alias | timeout / timeout / timeout | timeout | — | — |
| Qwen finalize | may-not-alias | unavailable / unavailable / unavailable | completion-transfer | pass | pass |
| Qwen finalize | may-alias | unavailable / unavailable / unavailable | completion-transfer | pass | pass |
| DeepSeek RMSNorm | may-not-alias | ready / ready / unavailable | sequence | pass | pass |
| DeepSeek RMSNorm | may-alias | ready / ready / ready | sequence | pass | pass |

Persistent/elementwise disjoint routes retain minimum demands for their supplied
declared models. GroupNorm, cumsum and RMSNorm reduce selected models containing
enclosures; may-alias also admits possible cross-root conflicts. None thereby
receives an independently certified physical-footprint precision guarantee.
Exact routes report `selected_order=equivalent` but realization remains
`covers-selected-order`. Completion reports `covering` and
`covers-modeled-order`; neither is an order-equality claim.

Concrete refusals:

- Elementwise and GroupNorm under may-alias retain sequence mathematics, but
  ordinary executable preparation fails. This is an export/realization gap,
  not proof of intractability or insufficient IDs. Automatic fallback does not
  overwrite those retained exact results.
- GroupNorm under may-not-alias reaches the separate allocator, which refuses:
  **“nested allocation requires a certified rule with matching source-tuple
  arity.”** This is missing allocation/coordinate evidence, not a checked
  six-ID capacity lower bound. No allocation repair is attempted here.
- Down under may-alias hits 35 seconds on each independently requested logical,
  mathematical, query and selector request. Since mathematical requests alone
  time out, allocation is not the cause of that watchdog. No failed-result
  profile establishes which relation operation dominates.
- RMSNorm's disjoint root selector-only request remains unavailable, while
  logical insertion, physical allocation and C++ generation succeed.

The explicit conservative policy is requested independently on the **unchanged
original inputs**, not substituted after a failed exact export. All 14 such
requests insert logical code, allocate with the existing pool and emit C++.

| Explicit completion-only policy | Static SET / WAIT / barriers |
|---|---:|
| Persistent GEMM | 0 / 0 / 10 |
| Elementwise | 0 / 0 / 6 |
| GroupNorm | 0 / 0 / 41 |
| Chunked cumsum | 0 / 0 / 8 |
| Qwen down | 0 / 0 / 32 |
| Qwen finalize | 0 / 0 / 6 |
| DeepSeek RMSNorm | 0 / 0 / 38 |

Both policies have these completion-only static inventories. The zero flag
counts make that policy's notification allocation vacuous; its cost is added
payload order. This is not a minimum-budget theorem or a resource repair of the
original protocol. Automatic fallback can retain more child work: down's
ordinary disjoint fallback has 19 static barriers versus 32 for the explicit
primitive completion control; finalize has 2/2 SET/WAIT and three barriers
versus six barriers in the primitive control.

## Independent finite validation

There are **44** whole-kernel bounded configuration/policy cases: zero, one and
two inner/reduction visits, two GEMM waves with counter resets, pipeline prefetch
and tail behavior, and an additional skipped-finalize case. Scalar inputs and
changed loop-prefix bounds are pinned derivatives; production originals are not
specialized. Bounds use new constants rather than changing constants also used
as divisors or geometry. Zero reduction cases are control/ordering stress tests,
not proofs that the numerical source algorithm admits a zero reduction.

The source-layout interpreters construct accesses without compiler partitions,
generators or selectors. They supply native start/completion chains, separately
modeled scalar readiness, and the existing accumulator-protection premise.
Unsupported data-loaded addresses refuse. The independent upper-graph checker
adds possible cross-root conflicts under may-alias without pretending their
relative offsets are absolute. Such a union need not be simultaneously
realizable by one physical pointer placement.

- Ordinary logical plans pass concrete prerequisite coverage and matching in
  **32** cases; 10 preparation refusals and two watchdogs remain explicit.
- **30** ordinary physical plans pass occupancy and causal
  consumption-before-republication; two bounded physical refusals remain.
- Explicit completion plans pass concrete coverage/matching in **44/44** and
  physical command checks in **44/44**. Flag-free reuse checks are identified
  as vacuous, not evidence about a difficult ID assignment.
- **13** qualified regional exports agree with the independent selected-model
  closure and expanded, predicated, deduplicated **actual retained covers**.
  This includes distinct physical versus possible-alias selected graphs.
  Flat reduction alone is not used to certify regional minimality.
- Requested selectors agree when exported; a mixed local/GM request can refuse
  because GM is unavailable. The earlier separate local tests remain passing.
  No unavailable mixed bundle is interpreted as a failed local selector or
  as an empty GM result.

Some cumsum scalar-loop occurrences are not resolved by the retained-event test
adapter; its leaf demand inventory is also unavailable. These are recorded as
**occurrence/export unqualified**. Their emitted plans still pass concrete
coverage, but this experiment does not certify their regional cover minimality.
GroupNorm's bounded probe reports nonconstant occurrence presence: its test
valuation is incomplete, rather than establishing a missing production
interface. Nonempty bounded finalize requests retain finite guarded mathematics
but refuse the requested export. Neither absence nor universal correctness is
inferred from these outcomes.

The preceding access-agreement and regional-survey oracles remain in the
focused gate, including actual retained covers, all four event-kind queries,
local/GM selectors, byte-boundary samples, and zero-inner extra barrier order.
New scalar/guard diagnostics do not make native footprints independently known.
Expert plans are unchanged and not used as correctness oracles. Scalar data
values used only for arithmetic are abstracted in these ordering checks; this
is not numerical-output testing.

## Approximation attribution

Numbers below are excess completion-to-start pairs in the **same disjoint
concrete execution**, not static site counts. The upper-GM column is a derived
independent graph merging GM membership, not another production run. Where it
matches possible-alias growth, that is only a finite observation.

| Bounded case | Extra C→I from upper GM graph | Ordinary may-not-alias excess | Explicit completion excess |
|---|---:|---:|---:|
| Persistent GEMM | 40 | 0 | 130 |
| Elementwise | 11 | 0 | 24 |
| GroupNorm | 10 | 2 | 203 |
| Chunked cumsum | 0 | 0 | 15 |
| Qwen down | 0 | 65 | 95 |
| Qwen finalize | 0 | unavailable | 2 |
| DeepSeek RMSNorm | 0 | 0 | 20 |

The remaining excess includes storage-summary coarsening, SCC-selected entry
completions and the ALL-barrier realization. This experiment does not falsely
separate those into exclusive numerical charges. The receipt records each
executed barrier gap and the next original payload type, locating the affected
inner/reduction/composition boundaries. Original ALL barriers remain in place;
there is no inserted flag-resource repair in the completion-only controls.

In particular, GM uncertainty alone adds no new requirements on these bounded
down/finalize/RMS/cumsum instances. More exact GM recovery therefore cannot
by itself explain away their measured completion order. For the GEMM zero-inner
case, the established two extra C→I pairs remain; they were not removed to meet
an equality target.

A synthetic pointer-ABI fixture additionally checks disjoint and overlapping
roots under may-alias: both ordinary and completion plans cover the concrete
requirements. It adds one prerequisite in the disjoint execution and none in
the overlapping one. The original ports' `port.json` requires disjoint roots;
an initial in-place-port experiment was excluded and superseded by this
explicitly overlap-permitting fixture. It is not original-kernel evidence.

## Construction size and costs

Single wall times include startup and subprocess polling; one other serial
campaign may run concurrently. They are descriptive, not paired speedup data.
M/Q/S are separate diagnostic workloads; allocation-support includes a new
logical compilation. **Do not add these columns.** Raw profiles retain exclusive
stage timings where available; missing decomposition is not filled with zero.

| Kernel / alias | Math | Query | Selector | Logical only | Logical + allocation support | Separate allocate | C++ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistent GEMM / may-not-alias | 0.165s | 0.165s | 0.165s | 0.417s | 0.666s | 0.165s | 0.516s |
| Persistent GEMM / may-alias | 0.115s | 0.115s | 0.115s | 0.366s | 0.366s | 0.165s | 0.416s |
| Elementwise / may-not-alias | 0.966s | 0.966s | 0.966s | 1.117s | 1.117s | 0.165s | 0.417s |
| Elementwise / may-alias | 0.466s | 0.466s | 0.466s | 0.666s refused | — | — | — |
| GroupNorm / may-not-alias | 0.416s | 0.466s | 0.466s | 0.666s | 0.666s | 0.165s refused | — |
| GroupNorm / may-alias | 0.516s | 0.516s | 0.516s | 1.067s refused | — | — | — |
| Chunked cumsum / may-not-alias | 0.215s | 0.215s | 0.215s | 0.416s | 0.416s | 0.165s | 0.567s |
| Chunked cumsum / may-alias | 0.215s | 0.215s | 0.215s | 0.416s | 0.469s | 0.215s | 0.566s |
| Qwen down / may-not-alias | 0.165s | 0.165s | 0.165s | 0.666s | 0.666s | 0.165s | 0.567s |
| Qwen down / may-alias | 35s timeout | 35s timeout | 35s timeout | 35s timeout | — | — | — |
| Qwen finalize / may-not-alias | 0.966s | 0.917s | 0.816s | 1.018s | 1.117s | 0.165s | 0.516s |
| Qwen finalize / may-alias | 0.465s | 0.466s | 0.416s | 0.566s | 0.666s | 0.165s | 0.574s |
| DeepSeek RMSNorm / may-not-alias | 0.065s | 0.065s | 0.065s | 0.265s | 0.265s | 0.165s | 0.516s |
| DeepSeek RMSNorm / may-alias | 0.065s | 0.065s | 0.065s | 0.366s | 0.316s | 0.165s | 0.516s |

For disjoint Persistent, exclusive producer-attempt work is about 0.081s and
logical preparation 0.004s. The separate allocation-support request spends
about 0.238s in allocation candidates, beyond its repeated analysis work. These
are ledger categories, not an end-to-end minimum-demand timing.

Production inventories are read-only. Owner maxima (not sums of shared arenas)
include Persistent 332 query types / 178 ports / 141 cells / 469 nodes;
elementwise 6/15/6/57; GroupNorm 61/60/33/115; cumsum 5/3067/2048/2153;
RMS disjoint 10/24/14/9. Cumsum's small command inventory therefore does not
imply a small storage interface. Completion owners do not supply an exact
query/selector arena; missing inventory counts remain unavailable. Their
reported completion contracts retain six storage classes and six inline
children for aliased Persistent, five classes and 18 inline children for
disjoint down, and two classes plus one eventful inline child for finalize.
The completion receipt does not identify every child's mathematical producer;
we do not rerun recognition to invent those missing retained-route names.

The fresh compact replay sweep preserves the prior structural bounds:
replays 1/2/3/8 retain 124 repeated types and an eight-site largest numerical
body; K=4/8/16 retains 332 types and a 14-site body. Shared node counts grow
125→143 and 192→200 respectively. Static kk=1/2/4 grows 116/188/332 types and
5/8/14 sites: compactness in K/replay is not uniform compactness in body size.
The 44 semantic derivatives also record per-policy preparation, allocation,
command counts and concrete excess. No broad timing improvement is claimed.

## Indirect candidates: capture versus analysis

Sources are pinned to clean PyPTO
`fb107da927f40a65e0e3ca7198e47f40d9e075d5` and pypto-lib
`f3167ecbf8838ecff0e169ee8d854baafaa0236c`. Capture specializes annotated or
meta-tensor signatures and calls `ir.compile(skip_ptoas=True)`. It never invokes
the device entry. Each attempt uses the same 35-second watchdog.

| Candidate | Capture / lowering outcome | Synchronization outcome |
|---|---|---|
| Qwen paged attention | Shape-only batch 1, one physical page; PYPTO planner, a2a3. Raw `attn_swpipe.pto` produced in 3.08s. It contains loaded sequence lengths and page-table indices feeding cache offsets. | PTOAS fails parsing `pto.load_scalar` (0.72s), before pre-sync IR capture. Root fallback and local-region analysis cannot be exercised through this incompatible opcode boundary. |
| DeepSeek sliding-window prefill | Original annotated `prefill_attention_swa_test`, default model configuration, a2a3. Specialization succeeds; frontend lowering fails in 2.92s on LegalizeTileCast FP32→FP8E4M3FN for A3, before a PTO input exists. Its sparse-attention source uses loaded SWA/compressed indices and a block table. | Not reached. This is a frontend/target conversion outcome, not a synchronization refusal. |

No addresses were replaced with invented affine expressions, no unsupported
scalar operation was renamed, and no unrelated frontend repair was started.
Regular local regions in these *new* captures are therefore not independently
qualified by this run. A conservative address set cannot repair an unimportable
operation or undefined native readiness contract.

## Remaining causes and next reusable step

| Cause | Evidence | Would conservative access descriptions resolve it? |
|---|---|---|
| Atomic/effect recovery | Down's GM atomic read/write unresolved | Existing whole-GM transfer covers storage hazards, but does not create exact selectors or native atomic qualification |
| Alias contract | Distinct roots may overlap; prepared ABIs require disjointness | Yes for conservative storage coverage, at extra order; never validates a contract-violating may-not-alias execution |
| Query/selector export | RMS disjoint root; nonempty finite/repeated GM bundles | Sometimes sufficient to use completion, but geometry coarsening does not supply an exact export |
| Independent checker valuation/frame | GroupNorm nonconstant presence; cumsum occurrence/leaf-inventory adapter | No: these checks remain unqualified until their test adapters supply the required frame/valuation evidence |
| Symbolic control/endpoints | Original Qwen parents; may-alias elementwise/GN executable failure; singleton derivative refusals | No: readiness, coordinates and historical availability remain separate |
| Logical realization | Exact selected order reports coverage rather than equality | No: modeled minimum demands alone do not qualify barrier placement/control order |
| Physical certificate | GroupNorm source-tuple arity refusal | No; no exact capacity lower bound or allocation repair established |
| Compilation cost | Down may-alias exact requests exceed 35s; cumsum interface is large | Explicit completion avoids the exact search, with a different guarantee; this is not an algorithmic speedup result |
| Native qualification | Scratch/ACC premises, indirect scalar opcode, A3 FP8 lowering, no device | No: these need their own supported semantic/lowering contracts |

The smallest useful next step is **provider-owned export of additional eligible
compact child fragments to the existing completion-child interface**. Existing
imports already preserve some exact finite work; qualified numerical/repeated
children should not need complete exact GM selectors merely to retain their
internal code. Start with one closed invocation, retain pending payload effects,
and prove notification closure and physical reuse separately. Do not import an
open loop iteration or transplant its visit-local disjointness proof across a
counter reset. Do not weaken a retained exact whole-region result implicitly.

This targets the measured additional completion order while retaining the
existing SCC solver and allocation service. It has stronger support here than
another general GM-address recognizer or allocation optimization. A separate
frontend/PTO scalar-op compatibility task is needed before either new indirect
candidate can qualify the existing fallback.

## Reproduction and validation

From this worktree, with `build=.local/section8-refactor/source-pinned/candidate-build`:

```sh
.venv/bin/python test/lit/pto/Inputs/assess_whole_kernel_access.py   --optimizer "$build/tools/pto-test-opt/pto-test-opt"   --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test"   --cli "$build/tools/ptoas/ptoas" --python .venv/bin/python   --timeout 35 --output .local/whole-kernel-access/reproduce-matrix
.venv/bin/python test/lit/pto/Inputs/check_whole_kernel_access.py   --optimizer "$build/tools/pto-test-opt/pto-test-opt"   --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test"   --timeout 35 --output .local/whole-kernel-access/reproduce-small
```

The receipt pins each actual command, input hash, diagnostic request and raw log.
The explicit original control is the same production command with
`completion-only=true`; separate allocation uses `eligible-ids=0,1,2,3,4,5`.
C++ generation uses `--enable-insert-sync=false --enable-plan-memory=false`.
Capture commands, signatures, original frontend source hashes and failure logs
are retained under `indirect/` and in the receipt. The independent alias and
reference/attribution scripts are also reusable without an external device.

The focused gate passed **23/23**; the final storage/alias/completion-input gate
passed **3/3** (two tests overlap), and the new synthetic-alias lit test passed
**1/1**. The completion-input oracle includes 400,000 overlap comparisons and
unknown-range/readiness checks. Scoped Ruff and diff checks pass; incremental
native builds introduce no warnings. The changed-code prefilter still reports
81 reviewed brace/weak-pointer regex false positives, with no unresolved
finding. External SecK/EChecker and quantitative complexity analyzers were not
run. No full-suite, device-compiler or device-execution claim is made.

Initial aborted derivative/probe runs are retained and excluded from acceptance;
no timeout was converted into a pass. Expert protocol discrepancies from earlier
checkpoints remain native/model questions, not revised oracles.

## Paper consequences

| Established result | Remaining premise or limitation | Appropriate manuscript claim |
|---|---|---|
| Useful whole-kernel code is generated without exact absolute GM addresses | Relative declarations/enclosures and modeled operation effects must be sound | Exact absolute address recovery is not a prerequisite for synchronization |
| 13 qualified finite exports agree on selected-model closure and actual covers | Native footprints, all root exports and all occurrence adapters are not qualified | Minimum-demand evidence is relative to the supplied selected access/alias model |
| Ordinary 11/14 logical, 10/14 physical/C++; explicit conservative 14/14 | Different guarantees and code; default exact-export failures remain | Report whole-kernel usability separately from exact mathematics and export completeness |
| All 44 explicit conservative finite plans cover concrete prerequisites | Finite layouts/control only; native completion/readiness assumptions retained | Conservative regional synchronization can remain useful under incomplete effects |
| Extra prerequisites exceed access-coarsening growth in measured cases | SCC-summary and ALL-realization charges not fully separated | Conservative demand coverage and realization-order equality are distinct objectives |
| K/replay representation remains compact in independent sweeps | Static body/port interfaces can still grow substantially | Charge supplied descriptions and requested interfaces separately from trip counts |
| Original six-ID plans and flag-free conservative alternatives allocate | No minimum capacity, native compilation or device result | Physical feasibility is a protocol-specific outcome, not exactness or hardware qualification |
| Both indirect candidates have explained pre-analysis outcomes | Scalar-op compatibility and target cast support missing | Do not claim indirect-address synchronization coverage before import/lowering succeeds |
