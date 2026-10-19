> Qualification scope: two-trip normalization is opt-in via
> `normalize-two-trip-loops=true`. It defaults to false because default-enabled
> normalization exposed a greater-than-120-second arithmetic-extraction path in
> gate-up projections. The down-projection results below enable it explicitly;
> they do not establish default corpus improvement.

# down_proj: two-trip normalization and remaining exact-path gaps

Baseline: `2df046db8cca11cbb0f65f1dffccd1764347f8ad`.
Input: primary PyPTO-lib Qwen3 prefill `down_proj`, prepared corpus case 0034,
SHA-256 `053f39b6ca12070c1c87f3b970c5ab10c23ee6e6cb5878d7dab1d7a67f59aad2`.
Both plans use `gm-alias=may-not-alias`. The candidate is this working-tree
normalization change. This is a partial integration improvement, not complete
exact qualification of down_proj.

## What Section 6 permits and what this case exposes

The current `approximate_demands.tex`, subsection “Executable completion at a
selected cut”, explicitly permits ALL-BARRIER for common completion. It also
states that this does not consume outstanding flags. The SCC transfer is not
intended to compute the exact minimum order. Its child-preservation contract
requires an executable child with closed matching; retained mathematics alone
is insufficient.

The original down_proj output used completion-transfer and 19 static ALL sites.
It preserved eligible finite children but failed to retain its main reduction
as one selectively synchronized child. This is a production integration gap,
not evidence that the SCC algorithm itself computes the wrong transfer.

## Implemented normalization

The public insertion path now unrolls original leaf `scf.for` loops with exactly
two constant trips before recognition captures anchors or callee seals. It uses
MLIR's unroller and canonicalizer within the existing module transaction. Failed
preparation cannot publish partially normalized source IR. Library callers keep
original coordinates by default; analysis-only continues to inspect original IR.

Qualification requires index induction, nonnegative bounds, and a representable
doubled step, matching the LLVM 19 utility's premises. Integer-width, negative,
dynamic and other trip-count cases retain their original form. Only original
leaf loops are selected: newly exposed parents are not recursively expanded.
Functions with manual event synchronization or physical sections are preserved.
Completion-only requests retain their existing policy.

This removes the three `0..256 step128` loops in down_proj and folds their
induction-dependent conditions. The outer core-distribution and pair-reduction
loops remain. It introduces no new synchronization algorithm or kernel-specific
recognizer.

## State after normalization alone

After normalization, the existing bounded-lifetime recognizer accepts the pair
reduction with refresh span **R=1**. Its mathematical, query and storage-selector
requests succeed. The enclosing body sequence also succeeds.

The root then fails at outer repetition. Its diagnostics include a runtime bank
ordinal origin, unavailable constant-stride evolving-storage families, and
restricted-arithmetic class rejection. The outer lower bound comes from the
runtime core index; address expressions include clamped offsets. A diagnostic
specialization of that origin to zero still fails bank geometry/slot-formula
qualification. Therefore replacing only the numerical-origin check is not a
sufficient repair, and the diagnostic specialization is not a production fix.
No nonnegative/no-wrap or scalar parameter bounds have been silently assumed.

At this prerequisite checkpoint there was also a Section 6 capability gap: the bounded provider did not
export a completion child. That checkpoint’s eventful inline importer qualified
concrete finite fragments; it could not import this whole bounded loop merely
because the loop had an exact mathematical result. Fallback descended into
smaller finite children. The [retained-child milestones](frontier-retained-consumption-plan.md)
subsequently add the whole-loop capability; their separate qualification reports
the order and control-code tradeoff. Such an extension must qualify closed matching for a
whole loop invocation and certify repeated resource use, or leave the capability
unavailable. An ALL barrier cannot manufacture notification closure.

## Checked order and resource effects

[Machine-readable results](frontier-down-proj-two-trip-results.json) retain the
raw allocated IR inventories and six execution comparisons. Plans are compiled
from the unspecialized input. Separate oracle inputs are specialized and fully
unrolled. An all-conflict/native graph agrees with the compiler's numerical event
queries. Effects and hardware-protection interpretation share the importer;
this is not an independent hardware model.

Original static payload labels survive cloning. Their executed order agrees
before and after normalization. Each candidate trace also passes logical
matching and physical consumption-before-republication checks with eligible
IDs 0–5. The emitted candidate actually uses IDs 0 and 1.

The paper metric is `A = |(H_plan minus H_required)_CI|`.

| Execution | Payloads | ALL before → after | Extra CI before → after |
|---|---:|---:|---:|
| Empty outer loop | 0 | 1 → 1 | 0 → 0 |
| Tail only, K=1 | 15 | 9 → 5 | 14 → 7 |
| Pair only, K=2 | 29 | 15 → 7 | 65 → 32 |
| Pair and tail, K=3 | 43 | 22 → 10 | 88 → 48 |
| Two pairs and tail, K=5 | 71 | 35 → 15 | 204 → 131 |
| K=17 | 239 | 113 → 45 | 900 → 629 |

No required CI pairs are missing in these executions. This is finite evidence,
not a uniform proof of order equality. The candidate still reports conservative
completion and **does not** realize the full kernel's exact minimum order.

Static raw sites change from 19 ALL and no flags to 13 ALL plus 15 SET and 15 WAIT
sites. For K=17, total dynamic synchronization operations increase from 113 to
215 while excess payload order decreases. Raw interpreted arithmetic visits
change from 407 to 343 and guard visits from 69 to 163. These counts precede
post-insertion canonicalization; no claim of smaller machine code or device
latency follows from them. Compilation speed is not this change's objective.

## Validation

`sync_frontier_two_trip.pto` checks a reduced cube pipeline through the public
pass, independent all-event order equality, original payload labels, carried
values, folded branches, zero/short/repeated visits, and physical ID reuse.
Scalar probes cover nondivisible two-trip intervals, unsupported negative and
integer-width bounds, dynamic bounds, doubled-step overflow, original-leaf
selection, and preserved manual synchronization.

The focused 16-test run passed 15 tests; the family pipeline test hit the wrong
Python interpreter through the CLI shebang. After selecting the existing venv
and incrementally relinking the CLI libraries, that complete test passed. An
additional focused rerun checks the expanded normalization edge cases. Earlier
test-development failures (negative bounds reaching LLVM's assertion and an
incorrect integer-loop test syntax) were repaired rather than relaxing the
semantic checks. Final rerun results are recorded in the local receipts.

Receipts: `.local/kernel-walkthrough-down-proj/comparison/`, including
`normalization-checked/results.json`, `compare-normalization-checked.py`,
`normalization-tests.json`, `normalization-rerun.json`, and
`normalization-final-focused.json`. No full-suite, device-compiler or device
execution claim is made by this prerequisite receipt. The subsequent
[retained-regional qualification](frontier-retained-consumption-report.md)
records the integrated implementation, independent reviews, and final gate.

The public pass defaults `normalize-two-trip-loops=false`. The dedicated
normalization tests explicitly enable it and check labeled payloads, iter_args,
exact event closure and physical reuse. Original-coordinate family tests retain
their original default commands. Default-enabled gate-up projections exposed an
expensive arithmetic construction path, so broad default admission remains open.
