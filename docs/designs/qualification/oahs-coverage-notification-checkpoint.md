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
