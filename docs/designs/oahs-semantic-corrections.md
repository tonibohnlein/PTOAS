# Semantic correction sequence

Active baseline: GEMM parity commit `16564fa8a`, merged with upstream master
`66bd855ed` in `388e127fd`, followed by shared InsertSync import in `d882ede3e`.
Audit: `/home/toni/work/pypto3_sync_more/region-prefill-review-20260922/REVIEW.md`.
The audit inspected a later implementation. Reproduce each restriction on this
branch before implementing its replacement; later M1–M3 mechanisms are not
implicitly present here. The objective is shared semantic analysis while
preserving causal checks and demonstrated overlap.

| Dependency order | Audit findings | Shared representation and invariant | Acceptance |
| --- | --- | --- | --- |
| 1. Effective access contracts | A1–A3 | Reaching tile descriptor dimensions plus translated physical output coordinates/layout. Ordinary ACC accumulation ordering is access-local, independent of input dtype and K equality; it grants no operation completion. | Real Qwen down-projection and DeepSeek KV/score projection; type-independent admission, K tail, equivalent view, unrelated and relevant descriptor changes; operand/FIX negatives. |
| 2. Physical-use relations | C1–C8, scalar part of B1/B2 | Address/selector dependency slices over original scalar semantics. Represent independent recurrence relations and exact or may-footprints separately; compiler exploration budgets are independent of event capacity. | Carried versus induction-derived slots, equivalent scalar expressions, unrelated recurrence/predicates, partial FIFO knowledge; preserve overlaps and integer semantics. |
| 3. Generation and support intervals | A2 whole-cell restriction, D1–D7, E4 | Extend lifecycle/occurrence records with participating generations, entry/reload/outside-reader/final-use obligations and affected producer corridors. A protocol must preserve support for every residual repair in its interval. | Initialization and epilogue, separate reader episodes, unrelated producer work and exclusive accesses, overlapping readers; retain the fence-relocation negative. |
| 4. Composed endpoint requirements | B1–B4, C6/C8 | Union first-consumer, first-write and final-reader requirements by original owner and exact ordered gap, with original participation predicates. Materialization policy cannot erase an independent requirement. | Equivalent first-use predicates, coincident roles/single visits, nested unrelated suffixes, prior refinements; participation-change negatives. |
| 5. Construction and binding | E1–E3 | Reuse source/deadline, prefix and support certificates for ordinary storage relays. Bind jointly required protocols under occurrence-scoped ownership and actual consumption/rearming at each deadline. | Non-FIFO equivalent dependency, disjoint reusable keys, residual deadlines; missing support/ownership/rearming negatives. |

On this baseline, the later general publication-prefix certificate, shared
occurrence service and explicit rearming-obligation records have not been
ported. Before broadening endpoint composition or binding (steps 3–5), establish
the corresponding ordering, participation and neighboring-use certificates in
the shared interfaces. Passing the safety checker alone does not prove that a
change preserves pipeline overlap. Port useful later mechanisms selectively;
do not reintroduce their recognizer restrictions to recover benchmark results.

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
`SyncAccumulatorOrdering.h`, native import; the 22 ported ACC variants, two argument-descriptor join variants, and
access-scope negative; and the pinned campaign in
`/home/toni/work/pypto3_sync_more/oahs-semantic-contract-work/`.
The broader audit objective remains active after this first deliverable.

## Backport evidence

The ACC implementation and 22 native variants are ported from `b506cc19e`.
Its original campaign is historical evidence; validation on this older
constructor is recorded separately in `HANDOFF.md`. Steps 2–5 remain planned,
not implemented by this backport. The prior campaign paths above document the
source commit, not a claim that its results already reproduce here.

The backport additionally seeds function-argument descriptor state before branch
joins. Without this, a conditional assignment to an argument could be omitted
from the join, resurrecting its original static dimensions. Native fixtures
cover both an unknown conditional assignment (no ACC credit) and an equal known
assignment (credit preserved). This is a conservative-state correction, not a
new synchronization policy.
