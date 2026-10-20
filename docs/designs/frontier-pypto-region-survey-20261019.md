# PyPTO-lib regional coverage survey — 19 October 2026

Three representative kernels were lowered and surveyed without changing production
analysis or allocation. **Compact regional analysis works on the Qwen K pipeline
and DeepSeek reduction. Complete GM boundary export is the recurring gap.** The
original Qwen roots currently use conservative completion; successful allocation
must not be read as exact whole-kernel Section 5 coverage.

The [machine receipt](frontier-pypto-region-survey-20261019.json) contains commands,
source/tool hashes, region requests, retained inventories, timings and finite
checks. Substantial raw artifacts are under `.local/pypto-region-survey/` in this
worktree. They are local receipts, not independently downloadable logs.

## Inputs and reproducibility

- Native source HEAD: `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44`, branch
  `codex/frontier-shared-input`, with pre-existing workspace changes. HEAD alone
  is insufficient: the receipt hashes changed native sources and test tools.
- Frontend capture: PyPTO `fb107da927f40a65e0e3ca7198e47f40d9e075d5`,
  pypto-lib `f3167ecbf8838ecff0e169ee8d854baafaa0236c`, clean at capture.
  Existing frontend captures were reused; current PyPTO Python was not rerun.
- Target A3 (`a2a3` frontend), level 3, PYPTO memory planner. Native memory
  replanning disabled so supplied allocated addresses remain authoritative.
- Production requests use `gm-alias=may-not-alias`: argument-disjointness is an
  explicit assumption, not proved from SSA names. No selected kernel has a
  data-loaded indirect address. The oracle refuses unsupported indirect accesses.
- The additional region requests are diagnostic and may select another owner;
  they are not read-only inventories or production timings. Production inventory
  is explicitly `selected-retained-owners-only`, `no-semantic-callbacks`.

| Kernel | Original source | Significant configuration |
|---|---|---|
| Qwen `down_proj` | `models/qwen3_14b/prefill_fwd.py`, down_core near line 1422 | HIDDEN 5120, MLP 17408, band 4352, M tile 128, K chunk 256, L0 chunk 64, stage 2, atomic Add output |
| Qwen `attention_finalize_phase` | same source, finalize near line 480 | work stride 48 up to 1024, dynamic active-token guard, divide/cast/store |
| DeepSeek `rms_norm` | `models/deepseek_v4_pro/rmsnorm.py`, near line 31 | D 7168, token tile 8, reduce/apply tile 1792, stage 2 |

Raw captures are under `/home/toni/work/pypto3_sync_more/corpus-refresh-20260916/capture/`:
`pypto_lib__prefill_fwd/generated/kernels/aic/down_proj.pto`,
`pypto_lib__prefill_fwd/generated/kernels/aiv/attention_finalize_phase.pto`, and
`pypto_lib__rms_norm_test/generated/kernels/aiv/rms_norm.pto`.
Their SHA256 values and complete lowering commands are in the receipt. Source
hashes: Qwen `266b36de6ed41dd13ba8118b517ace2f6a1d3623eb373dacfd8bcfb1d6b077dc`;
DeepSeek `14945bd44a9a598740250f13438c9251834bf36ce48b0ea626537eab5ba5ed57`.

Fresh physical-storage IR was taken from `--mlir-print-ir-before=pto-insert-sync`
using the current CLI and original capture, then wrapped in an explicit A3
module. No payload edits were used to produce these inputs:

- `.local/pypto-region-survey/qwen-down/input.pto`
- `.local/pypto-region-survey/qwen-finalize/input.pto`
- `.local/pypto-region-survey/deepseek-rms/input.pto`

Commented copies are retained in
`test/lit/pto/Inputs/pypto_region_survey/`. Example reproduction from the worktree:

```sh
build=.local/section8-refactor/source-pinned/candidate-build
.venv/bin/python test/lit/pto/Inputs/survey_pypto_regions.py \
  --optimizer "$build/tools/pto-test-opt/pto-test-opt" \
  --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test" \
  --input-root test/lit/pto/Inputs/pypto_region_survey \
  --output .local/pypto-region-survey/reproduce-children \
  --cases qwen-down qwen-finalize --children --require-supported
.venv/bin/python test/lit/pto/Inputs/survey_pypto_regions.py \
  --optimizer "$build/tools/pto-test-opt/pto-test-opt" \
  --trace-tool "$build/tools/pto-test-opt/pto-sync-input-test" \
  --input-root test/lit/pto/Inputs/pypto_region_survey \
  --output .local/pypto-region-survey/reproduce-rms \
  --cases deepseek-rms --require-supported
```

## Region coverage

A = existing compact route; B = established mathematics with an export gap;
C = access/alias information incomplete; D = stronger scoped input contract or
additional construction needed. A region can have different classifications for
its internal order and storage export. Region IDs refer to the original survey
inputs; isolated derivatives have different IDs.

| Kernel / original region | Selected analysis and evidence | Classification / limitation |
|---|---|---|
| Down root r0; outer N grid r2, line 26 | Exact request unavailable; ordinary production completion-transfer | C+D: symbolic slice/control/entry obligations and unresolved atomic GM effects; do not attribute all failures to one missing selector |
| Down K-pairs r5, line 31 | Sequence of three qualified regional-repetition descriptions; numerical 28-site finite bodies and repeated boundary frontiers | A internally/local boundaries; B GM selectors unavailable. All three single-capability requests ready; blanket capability readiness does not qualify every GM address |
| Down even/odd L0 loops r13/r14, lines 57/85 | Included in the K parent's qualified finite/numerical bodies | A in that enclosing phase context; standalone requests unavailable. Missing standalone context is not evidence of a new intractability theorem |
| Down odd K tail r7 and tail L0 r16, lines 116/129 | Standalone exact requests unavailable | D: current scoped control/initialization and symbolic extraction premises not established |
| Down output/store r8, line 159 | Internal sequence request ready | C: atomic GM read/write descriptors have no resolved intervals. Internal mathematical readiness does not establish physical access completeness |
| Finalize root r0/work loop r2, line 21 | Exact root unavailable; conservative completion imports finite child | D for original symbolic work/index contract; bounded nonempty derivatives retain expanded-finite-guarded mathematics but B for discharged-effects storage exports |
| Finalize active-token guard r5, line 32; finite body r8, line 33 | Sequence with finite child; instantiated retained inventory says finite-guarded | A for local body; B for complete GM boundary domain. Local actual covers/queries/selectors validated; uncovered GM must be unavailable |
| RMS root r0 | Sequence mathematics; query-only and executable-only requests ready | A internal composition/executable route; B storage-selector-only request unavailable |
| RMS reduction r2, line 30 | Query-only: order-covered-periodic; selector-only: bounded-lifetime; executable-only: periodic | A under each qualified route. Do not claim the original periodic owner supplies the stronger export merely because another owner does |
| RMS reciprocal-rms seam r3, line 63 | Finite/sequence | A, with supplied scratch access enclosure qualification |
| RMS apply steady loop r4, line 72; epilogue r5, line 125 | Empty paired steady loop plus finite epilogue | A. Source apply loop has one tile; pairing leaves its work in the epilogue, not missing computation |

Useful implementation points: `Shared/RepeatedRegionExports.cpp`,
`Shared/SequenceAdapters.cpp`, `Shared/SequenceStorage.cpp`,
`Structured/NumericTemplateRegional.cpp`, and the shared finite service. The
region-level inventory records the actual producer/reducer when available;
`unavailable` instrumentation fields were not reconstructed by running another
recognizer.

## Independent small-instance checks

`pypto_region_oracle.py` reads original tile declarations, view offsets, strides,
operations and bounded control. It does not consume compiler cell partitions or
generators. The graph oracle builds native start/completion edges and byte-level
conflicts, computes closure, and checks alternate first edges in that independent DAG to
identify nonnative covers. The test then expands **actual retained regional demand records**, applies
predicates and coordinates, deduplicates, and compares those pairs with covers.
All four event-kind queries and first/last writer/reader boundaries are compared.

The existing flat reducer is an additional cross-check, not the independent
extraction oracle. The tests use its supplied hardware-protection contract for
matrix accumulation; this does not establish native instruction footprints.

| Instance | Active payloads | Actual = oracle covers | Four-event comparisons | Result |
|---|---:|---:|---:|---|
| Down K child, K=2 | 28 | 18 | 3,136 | pass; 71 access records checked separately |
| K=4 | 56 | 42 | 12,544 | pass |
| K=8 | 112 | 90 | 50,176 | pass |
| K=16 | 224 | 186 | 200,704 | pass |
| Finalize finite child, work=0 and 48 | 5 each | 3 each | 100 each | pass; work=0 has 11 access comparisons |
| RMS reduction prefix 0 | 12 | 10 | 576 | pass |
| RMS prefix 1 | 22 | 18 | 1,936 | pass; 55 access comparisons |
| RMS prefix 2 | 32 | 28 | 4,096 | pass; 83 access comparisons |

Child tests bind a known enclosing coordinate and preserve original contiguous
payloads/control. They do not prove that the unbound parent exports all those
coordinates. Test expression valuations bind only original scalar definitions
that have one value across the entire bounded execution, never a changing
predicate's last visit. These valuations do not confer production availability.

Unqualified cases are recorded rather than forced through recognition: whole
small down roots remain unavailable; whole bounded finalize loops cannot export
the discharged effects, and zero/skipped finalize diagnostic exports are also
unavailable. No unavailable answer is counted as empty or as a successful oracle
comparison. RMS prefix zero is a supported empty reduction with nonempty later
work. Its two positive prefixes stay inside the original D=7168 domain.

Access precision matters. RMS prefix 2 has 56 declared-region records and 27
allocation-enclosure scratch records. Finalize has 8 declared-region and 3 scratch
enclosure records. Agreement proves consistency with these supplied envelopes,
not exact physical minimum demands for undocumented native scratch behavior.
Down's atomic output GM read/write remains unresolved even on bounded roots.
Aliasing between different GM arguments is assumed only in the explicitly named
policy. Allocation and compiler success do not repair missing access evidence.

## Trip-count scaling and costs

The K child retains **84 query types, 105 ports and 398 shared arena nodes** at
K=2/4/8/16. Its finite numerical body has 28 sites; the three repetition summaries
have 38/39/38 generators and 12/13/12 reduced records. The numerical phase bodies
have 53/52/52 generators and 19/18/18 covers. Repeated rows refer to shared owners;
do not sum inclusive inventory fields as distinct nodes. Production logical
requests take 0.266/0.216/0.216/0.216 seconds in these single samples. The
all-pairs oracle expands with K and takes 0.166/0.366/1.017/3.420 seconds; that is
validation cost, not production analysis growth.

Finalize's instantiated child retains 5 types, 5 ports, 9 nodes at work 0/48.
This establishes coordinate-context stability, not trip-count scaling of its
unsupported parent loop. Expanded finite mathematics is not claimed compact.
RMS retains a 10-site reduction body across prefixes 0/1/2; parent boundary ports
are 12/21/24 and arena nodes 7/6/9. These are bounded-domain observations, not an
unbounded asymptotic guarantee.

Fresh serial wall times below include process startup; they are not exclusive
stage profiles, paired speedup measurements, or device times. Lowering includes
the existing pass run needed to capture pre-sync IR. Storage and phase probes
are separate diagnostics. Mathematical diagnostics exclude allocation, but may
construct requested analysis interfaces.

| Original kernel | Lower/capture | Storage diagnostic | Phase diagnostic | Exact root diagnostic | Ordinary logical, no allocation preparation | Allocation support (includes logical work) | Separate allocation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Down | .466s | .065s | .065s | .115s, unavailable | .616s | .216s | .166s |
| Finalize | .416s | .033s | .033s | .716s, unavailable | .917s | .766s | .166s |
| RMS | .416s | .065s | .066s | .065s, ready | .215s | .166s | .165s |

These timings are independent single invocations; columns are not additive.
Under conservative `may-alias`, down reached a 35-second watchdog. That is an
unresolved run, not a proof that the mathematical class is inapplicable. Finalize
and RMS completed that control request.

## Logical and physical outcomes

| Original kernel | Logical route | SET / WAIT / barriers | Six-ID allocation |
|---|---|---:|---|
| Down | conservative completion-transfer | 0 / 0 / 19 | succeeds for that conservative protocol |
| Finalize | completion-transfer preserving finite child | 2 / 2 / 3 | succeeds for that protocol |
| RMS | sequence | 8 / 8 / 13 | succeeds |

Allocation policy and algorithms are unchanged. Physical feasibility is not a
minimum budget proof, exact-demand certificate, or realization-order equality.
The inspected guarantees are coverage of the selected/modeled order, not equality
with it. This survey does not establish full emitted payload-order equality,
native correctness, C++ generation, device compilation or execution.

## One reusable improvement

**Make storage-domain coverage an explicit provider-owned export contract.** A
finite cell inventory or `rangesMaterialized` flag must not authorize an empty
selector for a GM domain absent from that inventory. Prefer exact lazy selectors
where supported; otherwise return unavailable while retaining mathematics.

The bounded attention child exposed a reproducer: at GM byte zero, the input
model selects output writer occurrence 4, input reader 0, and other input reader
1. The old survey finite-cell fallback returned empty sets for all three. Its
fallback trusted `representedByCells`, populated from `rangesMaterialized` by
`Finite/FiniteGuardedAnalysis.cpp:234`. Production's generic reconstruction in
`Shared/SequenceAdapters.cpp:314` relies on the same completeness distinction.
This identifies an interface-safety risk; it is **not** a demonstrated original
kernel miscompilation or a validated callback-specific failure.

Only the survey probe was hardened: an absent GM domain with a relevant residual
access now reports unavailable. The original failing probe, tool hash and inputs
are preserved in `.local/pypto-region-survey/gm-false-empty/`. No production
export or allocation was changed to make the survey pass. K and RMS also report
unavailable GM boundaries explicitly. A narrowly scoped follow-up should provide
and consume the coverage certificate through finite/sequence export, with this
nonempty GM witness as a negative-empty regression.

Separate gaps remain: atomic TSTORE GM effect recovery (C), and symbolic parent
entry/control scope (D). A GM selector repair alone is not evidence that those
roots become exact or compact.

## Scope of manuscript claims

| Established result | Remaining premise or limitation | Appropriate claim |
|---|---|---|
| K regional records match finite independent covers; retained size stable over four trip counts | enclosing context bound; incomplete GM export/native footprint qualification | compact minimum demands and local composition for the admitted supplied access model |
| RMS numerical/periodic and sequence interfaces agree on bounded instances | scratch enclosures; root selector export absent | qualified regional construction, not universal exact physical footprint analysis |
| Original three protocols allocate within six IDs | Qwen conservative routes; no budget minimum or order-equality proof | feasible assignment for selected protocols |
| Missing GM boundary domain distinguished from empty by survey probe | production domain-coverage contract still needed | unavailable selectors remain a capability gap, not semantic absence |

## Focused regression result

Fresh serial lit run: **9/9 passed in 49.27 seconds**, including the new survey,
compact Persistent contract, expert/replay checkpoint, allocation profiling,
sequence analysis/bridges, repeated regions/storage and finite allocation.
The new test performs six supported child cases and three RMS prefixes; relevant
GM answers must not contradict the independent nonempty witness. Ruff checks
passed for the three changed Python scripts. No complete-suite or device claim
is made. The tool-only incremental target was built with at most two workers.
