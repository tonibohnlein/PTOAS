# Semantic correction sequence

Baseline: `a9ae8cb05`. Audit:
`/home/toni/work/pypto3_sync_more/region-prefill-review-20260922/REVIEW.md`.
The objective is to replace recognition gates with shared semantic analyses,
preserving causal semantics, independent checking and demonstrated overlap.
This sequence refines the remaining M4/M5 work; completed M1–M3 mechanisms stay.

| Dependency order | Audit findings | Shared representation and invariant | Acceptance |
| --- | --- | --- | --- |
| 1. Effective access contracts | A1–A3 | Reaching tile descriptor dimensions plus translated physical output coordinates/layout. Ordinary ACC accumulation ordering is access-local, independent of input dtype and K equality; it grants no operation completion. | Real Qwen down-projection and DeepSeek KV/score projection; type-independent admission, K tail, equivalent view, unrelated and relevant descriptor changes; operand/FIX negatives. |
| 2. Physical-use relations | C1–C8, scalar part of B1/B2 | Address/selector dependency slices over original scalar semantics. Represent independent recurrence relations and exact or may-footprints separately; compiler exploration budgets are independent of event capacity. | Carried versus induction-derived slots, equivalent scalar expressions, unrelated recurrence/predicates, partial FIFO knowledge; preserve overlaps and integer semantics. |
| 3. Generation and support intervals | A2 whole-cell restriction, D1–D7, E4 | Extend lifecycle/occurrence records with participating generations, entry/reload/outside-reader/final-use obligations and affected producer corridors. A protocol must preserve support for every residual repair in its interval. | Initialization and epilogue, separate reader episodes, unrelated producer work and exclusive accesses, overlapping readers; retain the fence-relocation negative. |
| 4. Composed endpoint requirements | B1–B4, C6/C8 | Union first-consumer, first-write and final-reader requirements by original owner and exact ordered gap, with original participation predicates. Materialization policy cannot erase an independent requirement. | Equivalent first-use predicates, coincident roles/single visits, nested unrelated suffixes, prior refinements; participation-change negatives. |
| 5. Construction and binding | E1–E3 | Reuse source/deadline, prefix and support certificates for ordinary storage relays. Bind jointly required protocols under occurrence-scoped ownership and actual consumption/rearming at each deadline. | Non-FIFO equivalent dependency, disjoint reusable keys, residual deadlines; missing support/ownership/rearming negatives. |

Start from the BF16 Qwen `prefill_fwd/down_proj` lifetime: an initialization
followed by K-axis accumulation into the same 128x256 F32 ACC allocation. The
missing fact is the documented output-access contract, not an additional BF16
protocol. DeepSeek's two 128x64 accumulators are an independent witness. The
remaining initialization barriers are a separate participation question.

For each correction, reproduce through native import and construction, compare
complete payload-order sets, event resources and compilation work separately,
and run the relevant corpus. Investigate changed plans; byte identity is not an
acceptance target. Preserve independent causal and emitted-order checks. Keep
policy experiments distinct from semantic repairs. No kernel-name recognizers,
dtype allowlists, global period products or arbitrary subset searches.

The audit's later architectural findings fit the same dependencies: region
transfers and occurrence relations (H1/H7) support steps 2–4; generation and
cross-cell completion support (H2/H3/H8) support steps 3/5; exact transactional
binding (H4) belongs to step 5; the common gap/deadline interface (H5) belongs to
step 4. Separate facts, proofs and policy (H6) throughout. Add structured refusal
facts (H9) with their real consumers; investigate dependency-granularity cost
(H10) with a measured recomputation witness, not an unmeasured cache redesign.

## First correction boundary

Remove unsupported dtype and identical-K restrictions. Resolve effective valid
dimensions at each access through a shared descriptor analysis, and physical
identity through existing translated footprint coordinates rather than SSA or
allocation spelling. Keep the existing whole-cell compatibility certificate
until step 3 supplies an interval-scoped replacement; do not silently weaken it.
Unresolved descriptor state or physical layout remains unknown, not disjoint or
ordered. General selector arithmetic belongs to step 2, not an ACC-specific
expression recognizer. Draft changes should formulate established mechanisms
and their hypotheses, rather than claim universal coverage from these witnesses.

## ACC correction gate inventory and review boundary

The dtype/K contract correction is implemented. General ACC/lifecycle analysis
is not complete. The following sufficient restrictions remain unfinished
analysis limitations, not hardware restrictions or optimization-benefit policy.

| Gate or fact | Obligation and evidence | Status / representation sensitivity / replacement |
| --- | --- | --- |
| Ordinary A3 MMAD, unspecified phase, accumulating consumer | The cited Mmad access contract; phase modes and fresh initialization have different obligations. | Necessary contract scope for this rule. No dtype allowlist. Ordinary instruction legality remains the frontend/lowering premise. |
| Known effective dimensions, supported range, floor-product threshold | Reaching metadata must establish the documented threshold and compatible output coverage. Tests include partial/nonmultiple valid dimensions and the exact threshold. | Threshold/range are contract premises. Requiring concrete dimensions is sufficient, not necessary: symbolic range proofs belong to shared scalar analysis (step 2). |
| Descriptor definitions, assignments and equal branch joins | Metadata updates affect descriptor state, not all aliased storage handles. Relevant unknown assignments refuse credit; unrelated or later updates do not poison earlier uses. | New shared semantic analysis. Descriptor reads feeding scalar expressions, dynamic region-result/argument forwarding and loops with unresolved reaching assignments remain unfinished limitations; replace with shared scalar facts and descriptor-state transfer relations. |
| Shared translated coordinates plus exact address/layout proof | Bounding may-footprints cannot identify one actual accumulator. Address-preserving reshape/bitcast and identity subviews preserve the base; offset/stride uncertainty does not. | Sufficient interface limitation. Nonidentity boxed subviews and structured forwarding remain unknown; step 2 must extend shared origin/geometry facts rather than add accepted view patterns. Identity and unknown-offset views have positive/negative native fixtures. |
| Footprint matches the tile extent; equal output layout/coverage | Excludes queue envelopes, widened views and overlapping but different accumulator identities. Checked arithmetic precedes footprint multiplication. | Sufficient certificate, not a new layout whitelist. Strided/fragmented geometry needs the step-2 physical representation. |
| Every M access on a canonical ACC atom is compatible | Current causal core consumes one cell property and a consumer-role flag; without a per-generation relation it would otherwise exempt incompatible accesses. | Retained whole-cell limitation, explicitly unfinished. Step 3 replaces it with interval/generation certificates before this gate can be removed. |
| No credit for operand release, FIX readiness/completion or event consumption | Unchanged pending histories and independent causal checker. Native tests delete required M-to-FIX and M-to-MTE1 acquisitions and require rejection. | Mandatory invariant, unaffected by representation or unrelated work. |

Separate generality review is required before claiming general completion.
Review inputs: the gate inventory above; `SyncTileDescriptorState.h`,
`SyncAccumulatorOrdering.h`, native import; the 22 native ACC variants and
access-scope negative; and the pinned campaign in
`/home/toni/work/pypto3_sync_more/oahs-semantic-contract-work/`.
The broader audit objective remains active after this first deliverable.

## First-deliverable evidence

The separate read-only generality review accepted this bounded correction and
found no demonstrated soundness blocker. It identified a conservative missing
case: a `get_validshape` → `set_validshape` round trip loses known dimensions
because immutable constant evaluation does not consume descriptor-read facts.
This belongs in the shared scalar/use interface, alongside the explicit
geometry and generation limitations above. The review does not accept the
broader objective as complete.

Validation: 27 portable suites, both native suites, 88 corpus cases, 19
compatibility cases and six targeted KDA/hc_pre/RMSNorm plans pass. The native
suite covers 22 contract/representation variants plus operand/FIX access-scope
negatives. Six targeted plans are unchanged. Sixteen corpus plans, one
compatibility plan and the two original prefill witnesses change only through
PIPE_M barrier removal. Exact fence-erasure certificates preserve payload,
control and event commands, establishing no added ordering for every original
trace; native independent checking establishes represented safety. Full
issue/completion relation comparisons on 14 prefill traces independently find
no added relations.

| Original prefill witness | Static M barriers | SET / WAIT (unchanged) | Constructor updates | Replay evaluations |
| --- | --- | --- | --- | --- |
| Qwen down projection | 12 → 6 | 69 / 69 | 17 → 11 | 16950 → 11316 |
| DeepSeek KV/score projection | 16 → 8 | 72 / 74 | 17 → 9 | 11202 → 5970 |

Static guarded SET/WAIT counts need not match; the independent dynamic checks
remain required. Resource counts, complete ordering and compilation work are
reported separately. No device performance measurement or claim is made.
The draft's target illustration now formulates the access-local safety argument
and retained whole-cell limitation. Next implementation step is physical-use
relations, not a new dtype/view/kernel acceptance list.
