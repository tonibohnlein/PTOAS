# Production coverage and access repairs — 2026-10-20

The fresh seven-kernel matrix has **11/14 ordinary logical successes, 11/14
six-ID allocations and 11/14 C++ outputs**. The baseline had the same 11 logical
successes, but only 10 allocations/C++ outputs. **Explicit conservative
completion succeeds through all three stages for 14/14 cases**, before and after.
Neither result establishes native/device correctness or minimum physical budgets.

The reusable improvements are:

1. A scalar cyclic assignment is now accepted inside a nested family inventory
   when its entire physical tuple is exactly the source ordinal. This removes
   GroupNorm's disjoint-root physical refusal without changing its certificate,
   logical protocol, palette, guards or commands.
2. Qualified full, aligned f32 ACC-to-ND-GM atomic-add stores retain their existing
   relative rectangle, with **both destination read and write** effects. Qwen's
   two unresolved GM effects and its source-allocation enclosure become three
   declared accesses. Unknown bases remain unknown; this does not prove different
   GM roots disjoint or provide complete root selectors.

No SCC, demand-reduction, protocol-generation or allocation algorithm changed.
The Section 6 token research, expert protocols and established zero-inner-body
barriers were untouched.

## Pinned experiment

The source base is `147cb72edd014a0cc7950ae50e1d870a370a0253`, initially clean.
The [machine-readable receipt](frontier-production-coverage-20261020.json) pins
that state, the dirty production/test files, tracked patch, original input hashes,
frozen baseline tools, candidate tools and the separate compatibility C++ backend.
A revision alone is **not** the candidate source identity.

Raw artifacts are under `.local/production-coverage/`. Baseline executables were
copied to `baseline/tools/` before editing. The established Linux snapshot source
and build are `.local/section8-refactor/source-pinned/candidate-{source,build}`;
both modified production files match their built snapshot copies. The initial
source manifest found no snapshot mismatches. The workspace's old macOS metadata
was not used to select the toolchain.

All comparable compiler runs use A3, IDs `0,1,2,3,4,5`, **35-second watchdogs**,
one campaign worker, one-CPU affinity and a 4 GiB address-space cap. At most two
intensive workers ran in aggregate. The baseline and candidate are single
measurements, with other permitted work present: their times are descriptive,
**not paired speedup evidence**. No timeout was extended.

The matrix refresh made ordinary production requests and inspected their existing
profiles. It did not repeat the region-wide survey or ask missing exports merely
to populate the production matrix. Explicit completion was a separate request;
it was not used to relabel an ordinary failure as success.

## Original-input outcome matrix

S/W/B/A are static SET, WAIT, barrier and arithmetic operation counts in logical
output. Logical and allocation-support are separate **complete invocations**;
the latter includes mathematics and preparation again. Their process times must
not be added or subtracted to infer exclusive allocation-support cost. Physical
and C++ times include parsing/startup. A dash means the preceding stage refused.

Time columns are logical / allocation-support / physical / C++, in seconds.

| Kernel | Alias policy | Logical | Six-ID allocation | C++ | S/W/B/A | Process seconds |
|---|---|---|---|---|---|---|
| persistent_gemm | may-not-alias | success | success | success | 134 / 137 / 2 / 3502 | 0.416 / 0.666 / 0.165 / 0.666 |
| persistent_gemm | may-alias | success | success | success | 0 / 0 / 8 / 66 | 0.414 / 0.568 / 0.266 / 0.967 |
| elementwise_pipeline | may-not-alias | success | success | success | 7 / 7 / 1 / 172 | 2.570 / 2.269 / 0.266 / 0.817 |
| elementwise_pipeline | may-alias | failure | — | — | — | 1.317 / — / — / — |
| group_norm | may-not-alias | success | success | success | 17 / 16 / 24 / 477 | 1.267 / 1.269 / 0.266 / 0.816 |
| group_norm | may-alias | failure | — | — | — | 1.617 / — / — / — |
| gdn_chunk_cumsum | may-not-alias | success | success | success | 3 / 3 / 1 / 42 | 0.466 / 0.466 / 0.215 / 0.666 |
| gdn_chunk_cumsum | may-alias | success | success | success | 3 / 3 / 1 / 42 | 0.516 / 0.516 / 0.217 / 0.669 |
| qwen-down | may-not-alias | success | success | success | 121 / 128 / 21 / 2521 | 0.916 / 0.917 / 0.215 / 0.716 |
| qwen-down | may-alias | timeout | — | — | — | 35.009 / — / — / — |
| qwen-finalize | may-not-alias | success | success | success | 2 / 2 / 3 / 49 | 1.618 / 1.668 / 0.216 / 0.767 |
| qwen-finalize | may-alias | success | success | success | 2 / 2 / 3 / 49 | 0.917 / 0.867 / 0.370 / 1.121 |
| deepseek-rms | may-not-alias | success | success | success | 8 / 8 / 13 / 133 | 0.316 / 0.316 / 0.215 / 0.716 |
| deepseek-rms | may-alias | success | success | success | 8 / 8 / 13 / 133 | 0.416 / 0.416 / 0.215 / 0.666 |

All 14 explicit-completion configurations succeed at logical insertion,
allocation-support preparation, six-ID allocation and C++ generation. Their
individual counts, costs, commands and guarantees are in the receipt.

### Selected routes and access contracts

| Kernel | Retained/selected production construction | Access and export qualification |
|---|---|---|
| Persistent GEMM | Disjoint roots: identity-sequence with phased/numerical repetition and finite bodies. May-alias: completion transfer. | 21 declared records. Compact inner K and replay remain intact. Exact selected-model covers do not imply realization-order equality. |
| Elementwise | Disjoint roots: sequence with finite and repeated children. May-alias: retained sequence, including expanded finite-guarded child, but preparation refuses. | 13 declared records. Unknown inter-root displacement is not resolved by comparing relative offsets. |
| GroupNorm | Disjoint roots: sequence, finite children and repetition; now physically executable. May-alias: retained mathematics, missing child recipe. | 58 declared and 35 enclosure records; scratch and scalar readiness are separate contracts. |
| Cumsum | Sequence and finite/repeated interfaces. | Nine declared records and one enclosure. Native scalar visibility and prior expert-plan discrepancies remain separately qualified. |
| Qwen down | Completion transfer with the retained, contextually prepared compact K invocation under disjoint roots; may-alias ordinary request times out. | Atomic output now retains its exact relative rectangle under the admitted full-f32 footprint contract. No complete exact root GM interface is inferred. |
| Qwen attention finalize | Completion transfer with qualified finite children. | Eight declared and three enclosure records. Complete GM boundary export remains unqualified. |
| DeepSeek RMSNorm | Regional sequence with periodic and finite children. | 60 declared and 31 enclosure records. The previously unavailable root selector capability is not an empty answer; the ordinary endpoint request succeeds without that export. |

The recorded reduction label `covers` concerns the **selected modeled effects**.
For scratch enclosures it must not be described as physical minimum demands.
The ordinary exact-model routes report `covers-selected-order`, not equality.
Completion reports `covers-modeled-order`, with reduction unavailable. Successful
allocation does not upgrade either guarantee.

## First blockers and the repairs

### Scalar cyclic maps inside nested inventories

GroupNorm originally fails at its logical WAIT for original record 39 (grouped
family ID 6 is a different identity). Its scalar certificate has budget two,
stride one, phase zero, and **no tuple rule**. The endpoint has source ordinal
plus a record-label operand: after excluding that label, physical arity is one.

`Shared/PhysicalAllocation.cpp::preflightPieces` rejected every nonconstant
scalar map in a nested inventory. `planPhysicalEndpointMap` had the same
restriction on the combined route. Both now admit this precise one-coordinate
case. Wider nonconstant tuples still require explicit arity evidence; explicit
rules must still match the endpoint's full physical tuple exactly.

**Validity:** the already decoded assignment is
`ids[(stride * ordinal + phase) mod ids.size()]`. Coordinate zero of both existing
mapping implementations is that same source ordinal. With physical arity one,
there is no omitted visit coordinate and no reinterpretation of the logical
record label. The conversion therefore preserves the supplied assignment for
every admitted ordinal. It creates no new reuse relation or coloring. The
producer's existing resource certificate remains responsible for legal reuse
between invocations. The checks are independent of kernel names and dimensions.

The numerical and grouped mapping regression uses noncontiguous eligible IDs,
sparse/nonlinear labels and nonunit enclosing coordinates. It checks each emitted
ID and retains refusal when a second uncertified physical coordinate is added.
Other existing explicit tuple, namespace and arity-negative regressions pass.

### Atomic output rectangles

`PTOMatrixAccessEffects.cpp::addMatrixStoreAccessEffects` explicitly excluded
atomic stores despite recovering their destination view. The original Qwen
access dump already contained the relative map
`4 * (5120 * row + column + symbolic_column_offset)` and extents `128 x 256`.
Its default atomic effects correctly included destination read/write, but lost
that footprint during qualification.

The existing matrix-store descriptor path now accepts atomic-add **only for
f32-to-f32**, retaining its full aligned accumulator, supported ND view, row
stride, valid-extent and phase checks. It emits source read, destination read,
and destination write using the same checked rectangle. Unsupported conversions,
unknown valid extents and other layouts keep conservative effects.

This follows the available [TSTORE instruction documentation](https://pto-isa.gitcode.com/docs/isa/TSTORE/)
and [A2/A3 implementation](https://raw.githubusercontent.com/hw-native-sys/pto-isa/main/include/pto/npu/a2a3/TStore.hpp):
atomic mode surrounds the existing transfer; shape/stride/valid extents still
select its rectangle. These are documentation/source premises, consulted on this
date, **not device qualification of this compiler/toolchain**. No stronger native
completion, atomic ordering or command-issue semantics were introduced.

### Remaining ordinary failures

| Input/configuration | First missing obligation | Responsible component | Does broadening addresses repair it? |
|---|---|---|---|
| Elementwise, may-alias | Retained expanded finite-guarded child has queries/selectors but no executable original-coordinate adapter. | `FiniteGuardedAnalysis.cpp::prepareFiniteGuardedLogicalInsertion`; sequence preparation reports unavailable adapter at the requested cuts. | No. Explicit completion works; do not silently weaken retained exact mathematics. |
| GroupNorm, may-alias | Regional child has no endpoint recipe. | Bound sequence-preparation probe; recorded owner and inventory in `diagnosis/group-probe.json`. | No. Explicit completion works. |
| Qwen down, may-alias | Ordinary request exceeds 35 seconds. | No completed profile is emitted before termination. Prior relation-composition diagnosis remains prior evidence, not a fresh exclusive attribution. | Atomic recovery does not remove this timeout. Explicit completion works. |
| Qwen finalize / RMSNorm | Complete root GM selectors remain unavailable in the existing evidence. | Provider/domain coverage and regional export; no fresh survey was run. | Exact local queries remain useful. A conservative summary cannot become an exact selector. |

A refusal or timeout is not evidence of mathematical intractability. The most
useful next construction is a qualified original-coordinate executable adapter
for **already retained expanded finite results**, beginning with Elementwise.
It needs explicit ordinal/guard/invocation translation; increasing expansion
limits or immediately weakening the retained answer is not a repair.

## Instruction and conservative-access audit

| Instruction/interface | Established implementation contract | Remaining qualification |
|---|---|---|
| Atomic GM TSTORE | ODS attribute and lowering select atomic-add TSTORE; effects include destination RMW. Qualified f32 ACC/ND geometry now survives the importer. | Other atomic conversions/layouts are conservative; no atomic/device ordering experiment was performed. |
| TGETVAL / TSETVAL | `PTOMemoryAndElementwiseEffects.cpp` uses one-element linear read/write declarations when supported, otherwise existing buffer effects. PIPE_S scalar results use `SyncInput::resultAvailability`; `PhaseIndex` separately tracks nonnative value/readiness prerequisites. | A memory enclosure does not establish scalar readiness or native issue/completion behavior. Existing native/expert-protocol uncertainties are unchanged. |
| Reduction scratch | Existing target-dependent scratch helpers retain read/write modes. GroupNorm/RMSNorm use allocation enclosures where fine footprints are unqualified. | Enclosure soundness assumes scratch stays within the declared reservation. The host layout oracle does not prove every native scratch implementation. |
| addptr / tensor / partition views | `SyncAccessRegion.cpp` preserves root provenance, element scaling, offsets and strides where supported. Unknown absolute GM bases can retain exact relative geometry. | Different SSA names are not a disjointness proof; unrelated roots' relative offsets are not comparable absolute addresses. |
| Unresolved location | `SyncEffectRanges.cpp` retains proven allocation or alternative-slot bounds; `CompletionStorage.cpp` merges overlapping known local reservations. Unknown location in a known space touches every class in that space; unknown space touches every class. Modes are preserved. | This cannot repair a missing write, unknown operation semantics or unsupported readiness obligation. GM remains a coarse space class in completion even when some relative geometry is exact. |

The may-not-alias policy requires disjoint distinct GM roots as a caller contract;
it does not infer that fact from this benchmark. Same-root overlapping views
retain overlap under both policies. May-alias preserves possible cross-root
conflicts. Existing view/reuse/carried/slot/blocked access regressions pass 5/5.
The exact-answer / proven-absence / unavailable-selector distinction is unchanged.

## Independent validation and ordering

* The focused gate passes **23/23**, followed by **2/2** updated/new tests and
  **5/5** access tests: 29 distinct tests (the matrix-footprint test was rerun).
  No complete repository gate or device campaign was requested/run here.
* GroupNorm: ten fresh zero/one/two/four-trip and skipped-subblock cases cover both
  policies. Disjoint roots use ordinary production; may-alias uses explicit
  completion. Independent original operand/layout graphs check modeled coverage,
  acyclicity, payload identities, matching, actual six-ID mapping and causal
  consumption before republication. The may-alias plans also cover the possible
  cross-root-overlap upper model. This upper model does not claim that every
  possible overlap occurs simultaneously in one concrete layout.
* Three full original GroupNorm block indices (`0,1,3`, subblock zero) pass the
  same checks: 78 payloads, 96 commands, **two excess completion-to-start pairs**
  relative to the disjoint supplied layout model. Original logical output is
  byte-identical before/after; added and removed prerequisites are therefore
  both zero. Allocation is newly accepted; logical ordering was not optimized.
* On the four-trip block-zero execution, explicit completion has 250 excess
  pairs versus two for production, against the same disjoint reference. These
  are different explicit policies, not a before/after protocol improvement.
  Broad GM classes and selected completion cuts are confounded here; the entire
  difference must not be attributed solely to address uncertainty.
* Qwen: the **original symbolic retained recipe** and actual decoder map pass
  59 bounded executions per version. All 59 full prerequisite comparisons have
  **zero added and zero removed C→I pairs**. Matching, invocation identities,
  pending-payload contracts, acyclicity, capacity and causal reuse pass. Cases
  labeled short/zero derivatives or external reset stresses keep their prior
  scope; they do not establish new caller bounds or valid data initialization.
* Atomic geometry: an independent enumeration of a 16×16 destination at row/col
  offset one in a 64-stride view matches both destination read/write byte ranges
  under both policies. Full/unknown valid extents, f16/bf16 conversion exclusions,
  RMW modes and existing consumers are also checked. The oracle uses source
  shape/stride arithmetic, not compiler cells or generated demands.

There is an important remaining test-interface limit: the fresh GroupNorm
root-cover probes for trips 0–2 return **nonconstant occurrence presence**.
Thus no new independent root-cover/minimality claim is made from those probes.
Command coverage is independently checked; existing compact-K/replay tests check
actual retained covers and selectors. Neither replaces an unavailable GroupNorm
root cover valuation. The allocation repair leaves all retained mathematics
unchanged.

The access/layout oracles assume the declared footprint and scratch contracts;
they do not independently qualify native instructions. The command graph checks
use the established native pipe and flag-issue model. A graph consuming compiler
generators alone would not independently validate their extraction; these checks
construct requirements from original access operands instead.

## Scaling and cost

Fresh production-only replay profiles retain:

| Controlled derivative | Repeated query types | Largest numerical body | Arena nodes reported |
|---|---:|---:|---:|
| Replay visits 1 / 2 / 3 / 8 | 124 / 124 / 124 / 124 | 8 / 8 / 8 / 8 | 125 / 141 / 141 / 143 |
| K 4 / 8 / 16 | 332 / 332 / 332 | 14 / 14 / 14 | 192 / 196 / 200 |
| Static inner stages 1 / 2 / 4 | 116 / 188 / 332 | 5 / 8 / 14 | 171 / 188 / 192 |

Thus increasing trips does not enumerate proportional retained K/replay types;
increasing static stages legitimately enlarges the body. Small expression changes
are visible rather than called constant-size. Allocation is not requested in
these scaling profiles. Existing independent compact/replay and zero-inner-body
semantic regressions pass.

The receipt records exclusive ledger time by stage, retained provider rows,
ports/types, storage-cell/interface sizes and arena sizes, plus process times and
static scalar/command counts. Arena counts shared by multiple retained owners
are **not summed as distinct nodes**. Recovery/recognition/extraction inside a
producer are not separately timed where the current ledger lacks that split.
Qwen's watchdog emits no completed ledger: its 35 seconds are not attributed to
one stage. No inclusive timing columns are added together.

## Indirect compatibility boundary

No capture or broad survey was repeated. The pinned Qwen paged capture still
fails on obsolete `pto.load_scalar`. A clearly labeled diagnostic copy changes
only that spelling to existing `pto.load`, whose ODS signature is the same
pointer/index/scalar load. It reaches storage analysis with local physical
reservations and indirect GM effects. An explicit root completion request then
refuses **closed same-core execution**. The cross-core communication/control
contract is a different obligation from address containment; no synchronization
output, allocation or native qualification is claimed. The original capture and
frontend remain unchanged.

The pinned DeepSeek sliding-window capture still stops before synchronization at
A3 fp32→fp8 cast legalization. Recovering that path requires separate target/
frontend work, not an address recognizer or broader enclosure. Existing capture
commands, source/configuration hashes and stderr are retained by reference.

## Reproduction

From this worktree (all substantial outputs stay on disk):

```sh
build=.local/section8-refactor/source-pinned/candidate-build
.venv/bin/python test/lit/pto/Inputs/qualify_production_kernel_access.py \
  --optimizer "$build/tools/pto-test-opt/pto-test-opt" \
  --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test" \
  --cli "$build/tools/ptoas/ptoas" --python .venv/bin/python \
  --timeout 35 --output .local/production-coverage/reproduce-matrix
.venv/bin/python test/lit/pto/Inputs/check_group_norm_allocation.py \
  --optimizer "$build/tools/pto-test-opt/pto-test-opt" \
  --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test" \
  --input test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto \
  --output .local/production-coverage/reproduce-group
.venv/bin/python test/lit/pto/Inputs/check_qwen_completion_import.py \
  --optimizer "$build/tools/pto-test-opt/pto-test-opt" \
  --baseline-optimizer .local/production-coverage/baseline/tools/pto-test-opt \
  --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test" \
  --input test/lit/pto/Inputs/pypto_region_survey/qwen-down/input.pto \
  --output .local/production-coverage/reproduce-qwen --full
```

Exact executed commands and hashes, including the initial equivalent local matrix
driver, are in the receipt/raw matrices. The native build completed with two
inherited deprecated-`isa` warnings in unmodified
`PTODialectAndEntryVerification.cpp:265`. Python syntax and diff checks pass.
The compliance prefilter reports two brace errors at nested getter expressions;
both bodies visibly have braces. These are reviewed G.FMT.11-CPP regex false
positives, not suppressed findings or actual unbraced statements. No broader
static-analyzer or native conformance claim is made.

## Paper consequences and next priority

| Established result | Remaining premise/limitation | Appropriate manuscript claim |
|---|---|---|
| Existing compact regional mathematics survives; no algorithms replaced | Not every provider exports executable original-coordinate endpoints | Section 5 coverage is capability-specific; recognition refusal is not intractability. |
| Atomic f32 output footprint is recoverable from existing descriptors | Other layouts/conversions and native behavior remain separately qualified | Exact relative modeled geometry does not require a known absolute GM base. |
| All seven kernels have an executable explicit conservative policy under both aliases | Broader classes/completion can add order; missing semantic effects cannot be enclosed away | Conservative synchronization is useful despite incomplete address information, under stated effect/readiness contracts. |
| GroupNorm's supplied scalar assignment now materializes and bounded reuse checks pass | Not a minimum-budget theorem or a new allocator | This refusal was mapping interpretation, not demonstrated scarcity. |
| Qwen child import and all 59 bounded ordering comparisons remain valid | Closed notifications do not mean completed payloads; caller/native premises remain | Retained compact children can coexist with conservative parent synchronization. |
| Root cover valuation remains unavailable for GroupNorm | Independent command coverage does not prove minimum covers | Keep modeled reduction guarantees separate from independent validation coverage. |
| Qwen paged diagnostic reaches access input after a spelling-only conversion | Closed same-core execution and frontend compatibility remain unresolved | Do not claim complete indirect-kernel synchronization support. |

Prioritize the reusable expanded-finite original-coordinate endpoint adapter;
retain exact child mathematics when its export fails. Separately diagnose the
may-alias Qwen relation-construction cost with a bounded attributed experiment.
Neither task calls for a new allocation algorithm, a higher admission limit or
another recognition backend.
