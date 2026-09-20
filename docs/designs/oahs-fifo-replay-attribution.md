# FIFO contextual replay: attribution before optimization

Base: `9f30b9fd8`. The initial attribution below added opt-in diagnostics only.
The subsequent sibling-reuse implementation is recorded at the end of this note.
Neither follow-up changes the dispatched device snapshot.

## Diagnostic interface

Run `pto-oahs-selected-test --construct INPUT --trace-replay`. The ordinary
summary remains available. Additional `replay_component` lines describe the
original control graph's strongly connected components (a loop and all its
backedges belong to one such component). `replay_trace` records each contextual
solve, including initial solves and failed/retried solves rather than only
successful selected updates.

Each trace records the ledger version, construction cursor, edited canonical
words and all their reachable occurrence components. Restart boundaries separate
the changed-word minimum, the cached-fixed-point restriction and shared-word
widening. Counts separate reused sites, distinct evaluated sites, repeated site
evaluations, successor joins, changed joins and finalized requirement queries.
Per-component counts distinguish work in the active loop from other loops.

The elapsed time covers the contextual solve and its finalized checks, including
diagnostic bookkeeping, but excludes subsequent source refresh. It is not a
complete cost partition or an allocation/copy-byte measurement. Qualification,
proposal checks, recurring omission, helper trials and final certification keep
their existing separate counters. Trace storage/output is opt-in; these records
are not inputs to the constructor's decisions.

## Partial-attention findings

Input: the immutable MAT campaign's `pypto_lib__prefill_fwd__45/prepared.pto`,
which contains both AIC and AIV functions. Tracing preserves its selected PTO
byte-for-byte against the retained `9f30` hardening output.

| Measure | AIC | AIV |
| --- | ---: | ---: |
| Contextual solves | 62 | 126 |
| Selected updates | 61 | 125 |
| Site evaluations | 102,810 | 813,458 |
| Distinct sites, summed separately over each solve | 22,954 | 120,983 |
| Repeat evaluations within solves | 79,856 | 692,475 |
| Shared-word restart lowerings | 0 | 0 |

AIV has two large cyclic components: component 47 contains 868 sites and
accounts for 402,010 evaluations; component 54 contains 498 sites and accounts
for 407,064 evaluations. Together they account for 99.5% of replay evaluations.
The existing prefix cache works: edits in the second loop retain the first.
This is not evidence that all 813,458 evaluations are removable.

Crucially, the loops are alternative branches, not successive stages of a
single execution. Component 47 reaches the common exit through components
48/57/58, and cannot reach component 54. While the constructor is active in
component 47, it nevertheless evaluates component 54 another 206,115 times.
The current rule retains only a topological prefix, so an unchanged sibling
later in that order lies in the recomputed suffix.

Checking every edited word occurrence sharpens that number: 64 edits touch only
component 47 and spend **202,944 evaluations in component 54**, which is outside
their control-successor closure. One earlier edit touches both 47 and 54 and
must not receive that reuse treatment. Across all components, 204,161 AIV
evaluations lie outside the changed-word control closure. This distinction is
why the trace records all changed components rather than only the earliest one.

This identifies a candidate for reuse beyond a prefix. It does **not** authorize
dropping the sibling from analysis: actual incoming state, edited occurrences
and shared endpoint aggregates must still be accounted for. The sum of work
outside changed-component control reachability is a diagnostic opportunity,
not a certified speedup or a safe skip set.

## Certificate requirements identified by attribution

1. Derive affected components from **all occurrences** of edited words and their
   control successors; preserve initial/cold solves and unsupported cases.
2. Reuse an unaffected component only with unchanged selected words and the
   same incoming semantic interface. Do not seed a changed loop with old facts.
3. Preserve every endpoint's complete occurrence aggregate. If a shared word
   spans reused/recomputed components, either retain a conservative fallback or
   cache/rebuild independently qualified component contributions. An old global
   endpoint aggregate must not retain stale contributions from a changed arm.
4. Check the same ledger against cold replay at every cut: incoming, pre-issue
   and outgoing causal state, occurrence history, consumption evidence and all
   endpoint aggregates. Also compare refusal behavior and source coverage.
5. Scale unrelated prefixes, alternative siblings and sequential loop episodes
   separately. A sequential successor genuinely depends on its predecessor;
   lexical nonoverlap alone cannot justify reuse.

Keep all mandatory validation. If native commands remain identical, acceptance
needs local equivalence and compile-cost evidence, not new device timing.

## Reproduction artifacts

Local disk-backed artifacts:
`/home/toni/work/pypto3_sync_more/fifo-replay-work/`.
`run.py` serially regenerates partial attention, down projection, LM head,
post-RMSNorm and Shenggan controls, checks archived plan identity, reconciles
trace counters, and writes `summary.json`. Each case retains its plan and raw
trace. These artifacts are separate from the immutable dispatched device bundle.

The portable selected-update regression checks trace opt-in behavior, unchanged
commands, per-component accounting and reconciliation with total replay work.
The existing replay tests retain complete incremental/cold state comparisons,
including shared observations, invalid edits and helper restoration.

All five native control modules construct/reconstruct successfully and retain
byte-identical PTO against their archived `9f30` outputs. Trace sums reconcile
for all six functions. These attribution-only checks did not establish a speedup;
the subsequent implementation and measurement are recorded below.
The rebuilt portable suite passes 23/23 tests, and the rebuilt native diagnostic
test suite exits successfully. Logs are `core-tests.log` and `native-tests.log`
in the artifact directory.

## Implemented: reuse unchanged sibling components

`SelectedReplay.cpp::reusableComponents()` now retains unchanged components
outside the old topological prefix. It consumes the existing `Control` graph,
component membership, canonical words and word-occurrence index; it introduces
no second causal state or kernel-specific rule.

### Why the reused states and endpoint aggregates remain valid

The cache must be a successful whole-graph contextual fixed point for the same
immutable program. A missing/failed/partial cache cannot supply sibling reuse.
An unexplained ledger-version change also forces recomputation.

Start with every component containing a reachable occurrence of an edited word.
Close this set under:

1. Original control successors, including paths through joins and later loops.
2. All occurrences of every nonempty command word touched by the set.

The complement is predecessor-closed and has unchanged transfer equations.
Pending payload transfer does not depend on the construction cursor or whether
a payload has been finalized. Consequently its incoming interface and least
solution are unchanged. Copying the complete cached cut states is justified by
that structural certificate; a mere textual or lexical separation is not used.

The second closure rule is deliberately conservative. Every endpoint's complete
aggregate belongs entirely to the retained region or entirely to the recomputed
region. The constructor never copies a global endpoint aggregate containing
stale contributions from an edited arm. A shared word can therefore force an
otherwise independent sibling to be recomputed, even when that shared word was
not itself edited. Empty words contribute no endpoint aggregate.

Recomputed components start from bottom, with only actual retained predecessor
outputs as boundary inputs. All finalized payload requirements are still checked
at convergence, and the final cold certificate and native reconstruction remain
mandatory. The ordinary non-contextual replay path is unchanged.

`--prefix-replay` on the diagnostic driver selects the old prefix-only rule for
controlled host comparisons; production defaults to sibling reuse. Tracing's
`resume` still describes the old prefix boundary. `sibling_components` reports
additional components reused beyond it.

### Work accounting and limits

The invalidation walk queues each dirty component once and scans its original
sites/edges once. Each reached nonempty canonical word expands its occurrence
list once, in addition to the edited-word seed scans. Boolean workspaces have
component/word size. `invalidation_sites`, `invalidation_edges` and
`shared_word_occurrences` charge this work separately from causal evaluations;
cached-state copying and source refresh still contribute to overall elapsed time.

On partial AIV, replay decreases from **813,458 to 608,848 evaluations (25.2%)**
at the same 125 selected updates. The dependency walk adds 87,175 site visits,
90,830 edge visits and separately counted word-occurrence visits. These cheap
structural visits are not interchangeable with causal transfer evaluations.
AIC stays at 102,810 replay evaluations. Selected attention PTO is byte-identical.
This does not solve the remaining repeated fixed point inside the edited loop.

### Validation and artifacts

The portable tests compare cached, prefix-only and cold replay for the same
ledger, including every cut's incoming/pre-issue/outgoing causal state, original
occurrence history, consumption evidence and every endpoint aggregate. Fixtures
cover alternative versus sequential loops, a non-edited shared word, edits to
both branches, matched event generations, deletion, an invalid acquisition and
recovery after failure. Scaling varies loop body length (2/8/32), sibling count
(2/4/8), and unrelated prefix length (0/16). Existing source-coverage, shared
observation and helper-restoration comparisons remain enabled.

The rebuilt portable suite passes 23/23; native diagnostic tests pass. All
**88/88 corpus modules** construct/reconstruct and retain byte-identical selected
PTO against the archived `9f30` plans, including projections, GEMM and attention.
Full-corpus identity and matched host timing are recorded separately in
`/home/toni/work/pypto3_sync_more/sibling-replay-work/` by `validate.py`.
The archived prefix-only diagnostic binary remains in
`fifo-replay-work/driver-prefix`. No device rerun is needed to measure a
compilation-only change with identical selected plans.

Three serial host rounds used the same diagnostic executable and input, pinned
to the same CPU, alternating prefix-only and sibling-reuse order. Tracing was
off. All six outputs match the baseline plan. Wall time includes parsing,
construction, reconstruction and PTO output for both AIC/AIV functions; it is not
end-to-end device compilation or kernel execution time.

| Variant | Wall seconds, three rounds | Median |
| --- | --- | ---: |
| Prefix-only | 22.458, 24.568, 25.718 | 24.568 s |
| Sibling reuse | 18.001, 18.304, 20.007 | 18.304 s |

Paired reductions range from 19.8% to 25.5%; the ratio of medians is 0.745.
This small local campaign supports the compilation improvement on this input,
not a general speedup bound. The deterministic replay reduction is 25.2%, with
the added dependency-walk work included in the measured time. AIC has unchanged
replay work, and no separate AIC speedup is claimed.


## Paired evaluation of the full supported corpus

A fresh serial sweep compared the current executable's default sibling reuse
with `--prefix-replay` on all **88 previously accepted corpus modules** (97
function instances). Both modes passed construction and reconstruction on every
module: **176/176 module runs**, with identical selected PTO between modes and
against the archived `9f30` plans. This tests the supported corpus; it does not
claim new admission for previously unsupported inputs.

Exactly 12 function instances reduce replay, 85 retain their counts, and none
increase. Related generated variants are counted separately in this inventory;
they are not independent performance samples.

| Function family / manifest rows | Instances | Prefix evaluations per instance | Sibling evaluations per instance | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Partial attention AIV, prefill44–47 | 4 | 813,458 | 608,848 | 25.2% |
| Single-block attention AIV, prefill48–49 | 2 | 64,501 | 49,973 | 22.5% |
| Final RMSNorm, prefill15 | 1 | 5,907 | 5,307 | 10.2% |
| QKV projection, qkv_proj0–2 | 3 | 786 | 679 | 13.6% |
| RMSNorm rows, rms_norm0 | 1 | 3,417 | 3,140 | 8.1% |
| Top-k, topk0 | 1 | 2,349 | 1,783 | 24.1% |

Qwen down/gate/up/KV/Q/LM projections, Shenggan GEMM, post-RMSNorm and attention
AIC retain their replay counts and exact selected plans. This is evidence that
the reuse rule transfers through shared control/word structure; it has no
kernel-name admission rule.

### Complete work inventory

These are unweighted sums over this corpus, which includes related variants.
They are neither model invocation weights nor a bound on arbitrary programs.

| Counter | Prefix-only | Sibling reuse |
| --- | ---: | ---: |
| Causal replay evaluations | 4,085,803 | 3,236,543 |
| Selected updates | 1,786 | 1,786 |
| Contextual solves | 1,432 | 1,432 |
| Invalidation site visits | 0 | 529,295 |
| Invalidation edge visits | 0 | 550,344 |
| Shared-word occurrence visits | 0 | 81,370 |
| Additional reused components, summed over solves | 0 | 3,625 |
| Proposal-check sites | 58,394 | 58,394 |
| Recurring omission analysis sites | 21,452 | 21,452 |
| Final helper-trial sites | 44,969 | 44,969 |
| Final certification sites | 87,511 | 87,511 |
| Loop-entry candidate analysis sites | 42,047 | 42,047 |
| Loop-entry preparation sites | 8,372 | 8,372 |

Replay falls by 849,260 evaluations (20.8%), but the added structural walk is
real work and is not subtracted as if every visit had equal cost. Components
with no additional reusable sibling can pay that walk without reducing replay.
State copying, source refresh and other uninstrumented costs remain included
in elapsed time, not fully partitioned by this table.

### Timing and reproduction

The sweep ran one pair per module, serially on CPU0, alternating arm order,
with tracing disabled. Summed diagnostic wall time was 145.082 seconds for
prefix-only and 130.301 seconds for sibling reuse (10.2% less). These single-run
sums include parse/construct/reconstruct/output and harness overhead, have no
confidence interval, and contain noisy small cases. Use them as a corpus cost
diagnostic, not a general speedup claim or per-kernel latency measurement. The
separate three-round partial-attention experiment above remains the repeated
measurement for that input.

Artifacts are under
`/home/toni/work/pypto3_sync_more/sibling-replay-work/paired-corpus/`:
`evaluate.py` (serial reproducer), `results.json` (commands, input/output hashes,
raw function counters and timings), `summarize.py`, `summary.json`, and per-arm
plans/logs. Inputs are from the immutable MAT source package and the accepted
case list in `mat-release-work/corpus.json`. The evaluated driver SHA-256 is
`e9b7feae7aa3f20639d470452cdd490f5614f33a62da4ee871ceb8be1c412c00`.
The exact driver invocation is `taskset -c 0 pto-oahs-selected-test --construct
INPUT`, adding `--prefix-replay` for the control.

No device work is needed to establish the compiler-cost benefit of this change:
all selected plans are identical. The outstanding device placement experiments
retain their original immutable source snapshot.
