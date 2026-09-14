# Coverage witnesses and notification prerequisites

The starting revision is `61aefcc93aecbe4bfee921e941fea87a7ee8610e`.
`oahs-coverage-work/checkpoint-61aefcc93/checkpoint.json` in the sibling
workspace records the exact frozen manifest, binary hashes, input and ABI
identities, and original report provenance. Those earlier binary measurements
remain historical references; they are not clean-revision builds of this commit.

The separate baselines are qualified GEMM at 54 SET/54 WAIT, zero body barriers,
one terminal drain and seven lifetimes; the conservative frozen corpus at
253/363 admissions with 86 first publication refusals; and the eight-benchmark
conservative-profile report. The structured-reference refusal and absent device
validation remain incomplete gates. `existing` remains the default.

## What the 86 refusals establish

`audit_publication_refusals.py` consumes the pinned replay, verifies prepared
input and legacy output hashes, and writes every reported writer/reader pair
with native operations, roots/ranges, physical cells, alias profile, structural
relationship, legacy overlap decision, emitted legacy synchronization and
missing evidence. Several categories can occur in one input. The 86 rows
contain 68 distinct prepared inputs and 1,084 candidate pairs:

| Pair category | Count |
| --- | ---: |
| Cross-root alias contract insufficient | 701 |
| Range or causal evidence insufficient | 164 |
| Already nonblocking | 200 |
| Exact overlapping ordinary GM accesses | 7 |
| Exact overlapping macro phases | 12 |

These are candidate pairs, **not reconstructed causes of the first refusal**.
The native report currently enumerates a Cartesian writer/reader population.
Reverse-order and mutually exclusive pairs remain visible, with explicit
control annotations. Exact attribution to a failing consumer, cell and reaching
generation, and attribution of each emitted InsertSync command, remain open.
The classifier no longer treats exact offsets on distinct roots as proof of
overlap: those offsets are relative to different, possibly aliasing bases.

All 604 recorded GM accesses have complete origin sets. In 53 of the 56 PTOAS
refusals there is no same-root overlapping candidate. Those inputs use
`may-alias`; complete provenance cannot create a missing caller guarantee.
The 25 PyPTO and five pypto-lib rows already use their frozen disjoint-argument
contract. Many accesses still lack geometry: 392 tile, 143 scalar and four
macro accesses have unknown ranges. Static tile dimensions alone cannot turn
strided or dynamically positioned windows into constant contiguous intervals.
`bytes=0` in legacy diagnostics means unknown size, not an empty access.

This explains why the targeted provenance/range tests did not change the 86:
they exercise facts that are absent from these dominant cross-root cases, while
the real same-root store/reload cases also need publication qualification.
No blanket frontend `Out`/SSA noalias assumption is justified. A separately
generated validation caller does allocate distinct ordinary GM arguments, but
that guarantee is limited to that caller; FFTS and shared prefetch workspace
arguments are excluded. A future caller-qualified campaign must bind generated
caller, launcher, C++, PTO and argument identities and import explicit pairs.
It must remain separate from the frozen generic-ABI campaign.

## Implemented real-population increment

The three frozen pypto-lib tail-exchange rows 307–309 previously refused at an
authored remote notification with an unfinished local producer prefix. Native
translation now exposes a structural insertion cut before each notification.
The existing composition constructors propose the existing AIV scalar clean,
fence and named-pipeline completion recipes there, before checking the immutable
notification. Open-protocol residual construction includes that proposal during
invariant discovery; final verification checks only actual emitted commands.

This does not qualify MTE3→MTE2 publication. The three real inputs advance to
that publication refusal and are not new admissions. Removing or moving the
prerequisite fails verification; a fresh scalar or DMA write requires a fresh
prerequisite. A local reload after notification still needs its own publication
contract. Notification-containing functions preserve authored synchronization
as separate insertion/transfer nodes, so their structural cost may increase.
The new scans and transfers are accounted for; optional open analysis retains
its precharged work bound.

The [GM publication qualification task](gm-publication-device-task.md) remains
OPEN. Event legality alone supplies no credit. Broader provider migration and
indexed-slot/ordinal work remain subsequent milestones, not completed claims.

## Reproduction and review

Fresh increment artifacts use `composition-notification-final`,
`campaign-notification-final`, and `benchmarks-notification-final` under sibling
`oahs-coverage-work/`. Their reports identify the tested dirty revision and
binaries. Compilation time is telemetry. Corpus rows record original, emitted
and added static synchronization, distinguishing generated synchronization,
authored preservation and no-op; counts include terminal drains and are not
executed synchronization measurements.

Architect, algorithms/performance and correctness reviewers examined the audit
and notification construction. Review follow-ups cover invariant proposal
transfer, explicit scan accounting and fresh-generation mutation coverage.
Only local targets are built, with at most two aggregate workers. No device
execution is claimed.

Final local verification: 2,465,903 core assertions; 22 positive native cases,
191 mutations, nine expected native refusals and two frontend cases. Qualified
GEMM preserves 54/54 events, zero body barriers, one terminal drain and seven
lifetimes. The frozen replay remains **253/363, zero gains and zero losses**.
The original 86 publication refusals remain; three newly exposed ones bring the
current first-publication-refusal count to 89. All eight conservative benchmark
static synchronization, scalar guards, event keys and executed-model results
are unchanged. The benchmark harness exits nonzero for the preserved structured
GEMM refusal; that gate is not relabeled as passing.

`notification-increment-results.json` pins the fresh report hashes and compares
them with the immediate baselines. Benchmark compilation telemetry uses three
paired trials after one warmup; the serial corpus replay ran concurrently, so
these host times do not establish isolated compilation or device performance.
The complete causal witness audit and a gain in justified generic-ABI corpus
admissions remain unfinished acceptance criteria.

## Follow-up: capture the actual publication refusal

Construction now records the consumer node, optional macro-phase index and
physical cells where the failing MTE2 read observes unpublished MTE3 history.
This bounded payload is exported before optional diagnostic enumeration. An
exhausted or unmapped report cannot establish that a candidate is unrelated.
Native access records preserve pipeline and macro-phase identities, and a
missing operation mapping is explicit instead of silently becoming node zero.

The audit retains every original candidate, but separately counts those matching
the actual failing consumer phase and cell. This is stronger evidence than the
old Cartesian population. It still does not identify an exact reaching writer
generation: the construction state tracks may-write history by pipeline and
cell. No alias or publication credit follows from the diagnostic record.

The `campaign-failure-site-r1` replay records actual failures for all 86 original
refusal rows. Of 1,084 Cartesian pairs, 166 match the actual failing consumer
phase and cell: 126 cross-root alias questions, 27 unresolved range/reaching-write
cases, seven ordinary exact overlaps and six macro overlaps. Admissions remain
253/363 with no changes. `publication-failure-site-r1` preserves the full audit;
`composition-failure-site-r1` and `benchmarks-failure-site-r1` retain independent
native and conservative benchmark evidence for that diagnostic increment.

## Follow-up: ordinary providers share residual construction

Unguarded ordinary providers (Every participation without loop/word qualifiers)
now activate the same lifetime-plus-residual construction as the qualified
ACC/ownership cases. Their original obligations remain intact, ordinary
families in independent regions retain their cuts, and the complete symbolic
event population is allocated together. The migrated path does not delete
pipe-pair obligations. `families_residual` reports actual selected lifetimes
that used this construction.

Tests cover a vector lifetime beside an independent ordinary region on the
same pipe pair, early vector release, multi-reader storage, unrelated macros,
scalar visibility recipes, repeated/empty invocations and actual allocation
failure with baseline fallback. A single physical ID per direction can now
support the multi-reader family; the old path's reservation caused a refusal.
`selected_visibility_sites` counts retained static recipes separately from the
construction-visit telemetry.

Endpoint footprints alone proved insufficient for preserving ordinary
completion: the early-vector return receipt completes later, unselected vector
writes from the preceding visit. Construction now observes the verified
baseline's actual acquisition effects and can retain both directions of the
corresponding protocol group, within physical capacity. These observations
guide selection only; residual construction and fresh combined verification
must establish all credit again. Observation scans, map initialization and
capacity searches are charged, including failed attempts.

The intermediate `composition-ordinary-residual-r2` run restores the early-vector
and multi-reader barrier-free checks but exposes a UnitFlag traffic regression.
It is not a passing integration checkpoint. The generic migration remains under
development until qualified ownership and GEMM regressions pass together.
The subsequent construction retains conservative receipt groups and extends
the existing bounded cleanup to adjacent ordinary event pairs. Each optional
removal requires a complete proof against the immutable obligations. This
recovers the UnitFlag traffic benefit without weakening the multi-reader
barrier check; cleanup is not the residual synthesis algorithm.

Guarded/deferred and alternative acquisitions are not generally migrated.
Lifetimes whose cells overlap macro phases are excluded because the open walk
does not yet synthesize their macro prerequisites. This is a cell-specific
restriction, not a whole-function macro exclusion. Unsupported cases retain
their verified baseline, and native reconstruction remains mandatory.

The verified `ordinary-residual-r4` increment passes 2,430,457 core assertions,
22 native positives, 191 mutations, nine expected refusals and two frontend
cases. Qualified GEMM remains 54/54 with zero body barriers and one terminal
drain. All eight conservative benchmark synchronization, scalar guards, key
usage and executed-model metrics match `failure-site-r1`; the structured GEMM
refusal remains open. Timings use one paired trial without warmup while the
serial corpus replay runs concurrently, and are telemetry only.

The frozen replay remains 253/363 with no gains or losses. Reports in
`campaign-ordinary-residual-r4`, `publication-ordinary-residual-r4`,
`composition-ordinary-residual-r4` and `benchmarks-ordinary-residual-r4` record
dirty-source measurements based on `a5a5ad928`, not tests of a later commit.
The campaign pins the complete source diff during its run; this documentation
was completed afterward. `ordinary-residual-increment-results.json` records
the report identities and comparisons. Device execution remains `NOT_RUN`.

`retained_completion_groups` counts proposed receipt groups before optional
cleanup; it is not a final emitted-group count. `selected_visibility_sites`
describes the final static visibility population. Review found no correctness
blocker after the accounting fixes. Two coverage/diagnostic limitations remain:
an optional retained group can consume capacity needed by later residual keys,
causing verified-baseline fallback, and observation exhaustion can surface as
a later residual failure instead of retaining the original analysis reason.
These do not supply acceptance credit or bypass emitted verification.
