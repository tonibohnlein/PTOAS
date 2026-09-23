# OAHS refactor: shared facts, composed lifetimes and sound construction

Approved consolidated implementation sequence, 2026-09-23. This supersedes the
restart checkpoints and historical milestone queues below. Active branch:
`codex/oahs-gemm-base`; starting HEAD `d5718272e`. Original audit:
`/home/toni/work/pypto3_sync_more/region-prefill-review-20260922/REVIEW.md`.

## Objective and acceptance boundary

Recover the general algorithm, not the old branch commit by commit. Preserve
GEMM enclosing-bank construction and complete readiness/release support while
replacing representation-sensitive admission with shared semantic analyses.
The causal frontier is the only authority for guaranteed completion. Original
physical facts, possible returns and source subscriptions grant no credit.

Soundness, matching, ownership and rearming are mandatory. Currently supported
inputs must continue compiling through checked OAHS construction. Recorded
temporary ordering, resource and compilation-cost regressions are permitted;
final acceptance recovers demonstrated GEMM/MAT overlap and useful sharing
through the general path. Historical event counts and byte identity are not
optimization targets. Explain remaining corpus tradeoffs individually.

Keep the existing atomic optional-recurring decline and bounded local
original-control retry. Report their reason and complete failed-attempt work;
a retry is service evidence, not success of the declined mechanism. Peer/queue
kernels do not take the observation retry. Do not add the old conservative
serializer/dispatcher or completed-plan deletion/subset search.

## Continuous implementation and review loop

User-directed execution policy (2026-09-23): continue through the complete plan.
A stage or commit checkpoint is not authorization to stop. Each coherent step:

1. Implement the shared mechanism and its positive, negative and composition tests.
2. Run relevant checks with the aggregate two-worker limit.
3. Obtain explicit acceptance from three independent reviewers: architecture/design,
   correctness, and performance. At this refactoring stage, performance review
   detects unnecessary asymptotic complexity and avoidable repeated work; small
   timing differences and tuning are not acceptance gates. Keep measured ordering,
   resources and compiler work as evidence.
4. Resolve blocking findings and obtain rereview of the final patch.
5. Record the scope, evidence, limitations and all three verdicts; commit that step.
6. Proceed directly to the next dependency-ordered step.

Review acceptance applies to the named increment, not to unfinished stages or
unmeasured device claims. Temporary regressions remain visible in the ledger.
Do not commit a rejected step or substitute the implementer's own verdict for a
reviewer's. Only an actual external blocker or a user pause interrupts this loop.

## Shared interfaces and invariants

| Interface | Meaning |
| --- | --- |
| Physical uses | Original accesses, descriptor state, selector dependencies, exact/may footprints and independent occurrence relations. |
| Generation families | Producing episodes, participating readers, next conflicts, initialization, reload, bypass and continuation obligations. Finite descriptions, not unrolled histories. |
| Endpoint requirements | Original owner, role, participation, stable ordered gap, required source history and deadline. |
| Support obligations | Completion/consumption needed by a deadline, candidate actual transfers and support actually established. |
| Ordered packets | Exact command words, logical matching, ownership intervals and physical bindings. Check and commit the same words. |
| Certificates | Proved, disproved under represented facts, or unknown, with failed premise and affected interface. Budget exhaustion is unknown. |

Extend existing records, with a real consumer for every new query. Keep facts,
certificates and deterministic selection policy separate. Establish occurrence
identities, ordered gaps and exact packet materialization early; enable richer
policies only after their support contracts exist. Region exit implies no drain.
The ACC exception never releases matrix operands, completes FIX or consumes events.

## Stage 0 — Close the current checkpoint and pin migration evidence

Already adapted: shared InsertSync import (`dbe56f7e6`), ACC contract
(`37554ef9b`), hardening (`be14229f2`) and physical/child occurrence foundation
(`53a8458a4`). Do not port them twice or claim the last foundation is a complete
lifecycle/occurrence service.

- Finish the interrupted portable checks and complete ordering comparison of the
  changed DeepSeek CSA compatibility plan before widening control refinement.
- Pin supported inputs, binary/plan identities, resources and construction work,
  including discarded attempts. Keep one temporary-regression ledger.
- Protect exact-fit starvation, atomic rollback and exact check/commit order.
  Consolidate consistent existing materialization into a shared transaction.
- Verify the missing shared merge-sort effect correction (`94c8b0f1`) against
  current lowering and adapt its shared effects/tests only. Castptr origins are
  already ported; do not restore an OAHS instruction whitelist or FFTS gate.

Exit: reproducible evidence, classified CSA change, explicit inherited failures
and protected transaction invariants. No new device claim.

## Stage 1 — Complete shared physical-use and occurrence consumers

Build on `53a8458a4`, adapting common lifecycle/source/receipt queries from
`55e70638`. Preserve original owners, participating child entries/body/exits,
endpoint positions and shared suffixes through refinement. Physical footprint,
occurrence correspondence and event feasibility remain separate outcomes.
Independent relations and partial facts survive unrelated unknown relations or
infeasible optional materialization. Reuse immutable original analyses.

Separate further scalar/footprint improvements from activation of richer
control refinement. Each query replaces an existing consumer restriction;
never hide the old recognizers behind a new class.

Exit: carried/IV selectors, equivalent expressions/views and unrelated state
preserve applicable facts; event capacity does not bound physical knowledge;
refinement preserves an independently valid nested readiness/release protocol.

## Stage 2 — Generation families and affected-interval support

Start with one generation read by two children before reuse. Vary reloads,
skipped children, outside readers, unrelated producer work and overwrite after
the parent. Adapt mechanisms/tests from `315459c2` and `2b1fe121`, not their
whole-cell/whole-producer recognizers.

Represent initialization, read/write episodes, reload, bypass, scratch reuse
and epilogue obligations. Replace global vetoes with certificates covering every
residual repair and exported obligation whose placement could change in the
affected interval. Preserve the X/Y fence-relocation negative. Scope exclusive
resource interference to that interface too. Open entry/exit obligations are
explicit; full contextual replay remains the execution/checking mechanism.

Replace whole-cell ACC eligibility only when the access/production-episode
relation preserves obligations from older incompatible accesses.

Exit: unrelated work outside a proved interval stops vetoing admission; reload,
overlap and missing-support negatives remain protected.

## Stage 3 — Compose endpoint roles and certify placement

Collect first-consumer, first-write and final-reader requirements before owner
refinement. Preserve distinct ordered gaps when roles coincide. Participation
comes from original bounds, step and control, including zero/single visits,
skips, repeated entries and nested shared suffixes. Query matched endpoints,
same generation or next participating use rather than require one metadata mode.

Adapt joint collection (`58ce63c9`) with common correspondence (`55e70638`).
Adapt exact-gap/publication-prefix certificates (`edcdcece`, `01aa6e5a`) covering
intermediate work, outward publications, matching occurrences and neighboring
event uses. Revalidate after relevant edits. Unknown/budget exhaustion retains
separate boundaries; it never justifies motion. Discovery is independent of
benefit heuristics; materialization may remain demand-driven.

Exit: endpoint roles coexist without losing participation or positions; no
sharing relies on uncertified boundary movement.

## Stage 4 — Select complete support and bind ordered packets

Select complete readiness/release obligations before ordinary residual repair.
Discover required transfers whose existing source histories and endpoints cover
additional completion or consumption needs. Preserve separate readiness cuts;
never enlarge prefixes or move a return to manufacture coverage.

Use one packet qualification/binding path for direct, recurring, relay,
shared-return and helper clients. Apply proposed words privately in exact order:
the first actual receipt may establish second-leg credit. Check realizability
before suppressing another transfer; an unavailable direct direction need not
reject an otherwise supported packet. Check matching, occupancy, active/dormant
ownership and both neighboring event uses at actual deadlines, replacing global
key-count proxies. FIFO provenance is not a relay premise; intermediate work
and outward publication checks remain mandatory.

Adapt required-provider/shared-return mechanisms (`3f547429`, `e8ed8596`,
`9cadd35c`) and ownership/relay tests (`7f22b091`, unported `b62b89de`,
`74dcd195`, `89f5b2d6`, `9f2063c2`). Add deadline-driven rearming using both
`0b25d581` and `e5fe6147`: conditional balance, stale receipts, earlier deadlines
and later edits are part of the mechanism, not optional follow-ups.

Decline failed optional proposals atomically. Keep independent groups only when
support proves separation. No arbitrary subset or final helper-deletion search.

Exit: every construction entry uses the same support and binding contracts;
possible returns never substitute for acquired credit.

## Stage 5 — Consolidate and recover quality

Remove superseded paths as replacements pass acceptance. Restore the mechanism
portfolio: GEMM/MAT overlap, retained generations, cross-cell returns, RMSNorm,
attention, CSA refinement and scarce-key cases. Follow Qwen/DeepSeek prefill from
physical fact to candidate, selected transfer, emitted order and coupled device
execution; do not assume every kernel must speed up.

Do not recover counts with broad release merging, speculative future fences,
completed-plan helper deletion or the old experimental option matrix. Reference
gains must come through the shared constructor. Obtain an independent generality
review and update the draft's construction rules and proof obligations alongside
established mechanisms; finite tests are not general proofs.

## Validation, cost and working discipline

At every default-path checkpoint, run focused positive/negative, equivalence and
composition tests through real import/construction, then supported-input
construction/reconstruction with unchanged independent checks. Compare complete
payload-order sets on changed plans; report commands/resources and work
separately. Distinguish mechanism selection, optional decline, retry and authored
exclusion. Keep historical campaigns separate from current-head evidence.

The generality matrix includes independent/coupled selectors, arithmetic/views,
partial FIFO facts, relevant/unrelated descriptors and predicates, initialization,
reload, outside readers, skipped/repeated children, coincident endpoint roles,
cross-cell returns, stale consumption, dormant ownership and packet-supported
second-leg reuse. Final device tests use authentic coupled execution.

Extract established plan-equivalent cost reductions when their consumer exists:
next-provider-only selection (`42b051bc`), empty classification/state-copy
avoidance (`0b25d581`), and measured unchanged-query hoisting. New caches or
reusable region execution require a recomputation witness and invalidation/cold
equivalence tests. Open region interfaces are required now; new execution engines
are not. Keep local resource-intensive work within the aggregate two-worker cap.

This document is the canonical sequence; TODO tracks tasks; HANDOFF records the
current checkpoint and exact evidence. Old M1–M5 labels remain historical bounded
achievements, distinct from active refactor stages 0–5. The [donor ledger](oahs-donor-ledger.md) records
adapted portions, excluded portions, actual consumers and branch validation.

## Temporary-regression ledger at the starting checkpoint

| Witness | Current evidence / missing capability | Owner / recovery |
| --- | --- | --- |
| DeepSeek CSA compatibility case 1 | Construction/reconstruction pass. Finite explicit-command comparisons show +20/-20 relations for one active outer visit (+40/-40 for two); added A-extraction-to-external-receive order. No peer-progress conclusion. | Stage 0 classification; stages 1–3 if correspondence/support is missing. |
| GEMM resources | 330/652/1296 pairs vs historical 182/360/716; hardening removed ordering, no new device timing. | Stages 3–5 certified sharing and binding, not broad merge restoration. |
| Fence-heavy replay work | RMSNorm/TopK/route_sort work grew after deadline-local fencing. | Measured plan-equivalent work track; preserve actual-deadline repair. |
| Qwen topk_select, kernel_softmax_prepare | Starting baseline had two refusals. Current service passes 18/19: softmax_prepare uses a visible local observation retry, not a successful refined protocol. | Retain topk refusal and softmax refined-path failure; service coverage is distinct from mechanism recovery. |
| Physical refinement limits | One compatible period materialized per owner; other independent relations remain physical facts. | Stages 1–3 consumers; no event-derived analysis limit. |

## Progress after the first consolidated increment

Stage 0 host closure and a Stage 1 common-query increment are implemented.
22 portable suites, all three native drivers, 88 corpus cases, GEMM and the
recorded targeted/prefill witnesses pass. Previously passing emitted plans are
unchanged. Compatibility service coverage improves to 18/19 through a visible
local retry; refined softmax support remains open. CSA's +20/-20 one-visit order
change is classified and retained as a recovery witness. Full details and exact
artifact pins are in HANDOFF. This does not complete Stage 1 or stages 2–5.

## Accepted implementation steps

### A — Shared facts and exact recurring materialization

Base: `d5718272e`. Architecture/design reviewer: **ACCEPT**; correctness reviewer:
**ACCEPT**; performance reviewer: **ACCEPT** (2026-09-23). Each independently
reviewed the implementation and recorded host evidence without running parallel
builds or tests. Their acceptance covers this increment, not full Stage 1.

Architecture verified one causal authority, shared physical queries, immutable
correspondence and identical checked/committed materialization. Correctness
verified all-path pairing, no partial credit on exhausted queries, intervening
key occurrences, packet ordering and the pinned merge-sort output contract.
Performance verified unchanged previously passing plans, separate query-work
accounting and no unexplained new timing regression. Single host timings do not
establish a speedup. The inherited mixed CSA order and fallback-only softmax
service improvement remain open. Detailed evidence is linked from HANDOFF.

Architecture's nonblocking finding: `RequirementFrontier::lifecycleRelease`
duplicates the shared use record and has no consumer; remove with the next
shared-query consumer change. Continue Stage 1 immediately after this commit.

### B — One matching query for endpoint sets

Base: `b6958b0e4`. Architecture/design, correctness and performance: **ACCEPT**.
Canonical endpoint sets now share source-identity pairing across recurring,
alternative-source and loop-entry clients. Removed both superseded balance
monitors, their unused graph copy and the unused duplicate release field.
Architecture's cleanup request was resolved before acceptance. The independent
up-to-four-site token oracle and a joining-source/shared-suffix workload cover
matching, original pairs, repeated entries, same-word order and budget refusal.

22 portable suites, all three native drivers and the pinned supported-input
campaign pass with every plan unchanged from Step A. Results and counters:
`../oahs-gemm-base-builds/refactor-step-b/`. Source-identity state growth is
explicit; future budget refusal stays unknown. Performance acceptance follows
the user's asymptotic-complexity criterion. The observed older/newer campaign
timing difference was not reproduced by serial alternating binaries. Continue
with shared post-payload boundaries and generation/support obligations.

### C — Shared after-payload publication boundaries

Base: `3ee150030`. Architecture/design, correctness and performance: **ACCEPT**.
One immutable all-occurrence boundary certificate replaces the recurring mode
and observation-atom equality scans. Its gate inventory is in `oahs-analysis.md`.
Native equivalence fixtures explicitly exercise boundary work and retain both
per-bank directions without retry. 22 portable suites, all three native drivers
and the complete pinned supported-input checks pass with unchanged plans.
GEMM: 87 boundary visits, 74,466 correspondence visits, unchanged 21,846 replay
visits. These are separate analyses, not acquired credit. The checker has one
reviewed nested-call bracing-regex false positive; the body has braces.
Continue into shared physical-use succession and scoped generation/support;
this boundary fact alone does not establish either.

### D — Shared physical-use succession

Base: `54cee6cda`. Architecture/design, correctness and performance: **ACCEPT**.
The existing storage analysis owns memoized nearest physical uses in either
direction, including explicit open boundaries. The recurring client no longer
walks its own access graph. Unknown is distinct from a proved empty frontier;
may-writes and RMW stop traversal without killing older writer provenance.
A two-child/reload/outside-reader test protects this distinction and query reuse.
Native equivalent-representation tests require the new query to be exercised.

22 portable suites initially and four focused suites after review amendments
pass, as do all three native drivers and every previously supported pinned
input. All plans equal Step C. GEMM uses 2,792 physical-use visits; ordering,
resources and replay are unchanged. Evidence: `../oahs-gemm-base-builds/refactor-step-d/`.
This completes the shared succession replacement, not generation/support
selection. Next: replace child-owner recognition with generation requirements
and prove support for affected remaining repairs before admitting new cycles.

## Historical audit detail and restart checkpoints

The material below records earlier reasoning/evidence. Its numbering and active
status do not supersede the consolidated stages above.

# Original semantic correction sequence

Active branch: `codex/oahs-gemm-base`, baseline `37554ef9b`. The historical
GEMM snapshot is rebased as `f08b28194` atop upstream `66bd855ed`, followed by
shared InsertSync instruction import `dbe56f7e6` and ACC correction `37554ef9b`.
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
`SyncAccumulatorOrdering.h`, native import; the 22 ported ACC variants, two supported argument-shape variants, and
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
joins. The originally added argument-mutation fixtures were invalid: this dialect
permits `set_validshape` only on locally bound dynamic descriptors. They have been
replaced with supported static/unknown argument-shape cases. Existing local-tile
variants cover equal and unknown branch updates. No instruction verifier is
relaxed, and the invalid fixtures are not claimed as native join evidence.

## Approved continuation after the GEMM restart

Facts, certificates and selection policy remain separate. Extend the existing
physical/control/lifecycle records; do not add a second planner or completion
ledger. A possible return is not acquired credit. Unknown proof results never
establish completion, disjointness or event availability.

### Checkpoint 1: hardened, measured baseline (implemented)

Validate the ports and correct invalid fixtures before adding analysis precision.
Port the first-use separation premise from `f0d2a07db`; remove speculative fences
in distinct future command words and uncertified recurring endpoint motion.
Share identical boundaries only. Failed optional recurring construction gets
one fresh ordinary OAHS attempt from original inputs, never partial-ledger reuse.
Report the failed attempt's reason and work explicitly. No new conservative
constructor or completed-plan subset/deletion search is introduced.

Exit evidence: focused positive/negative tests, native reconstruction, supported
input results, and complete ordering comparisons for changed witnesses. Event
counts and compilation work are separate measurements. No new device claim.

### Checkpoint 2: physical use plus stable occurrence consumers (foundation implemented; acceptance open)

Derive each address/selector from its original dependency slice. Preserve
integer semantics, independent relations and partial may-footprints. Compiler
analysis budgets are independent of key capacity. Include the common query for
original owner, participating child occurrence, entry/body/exits, original
ordered endpoint positions and next relevant use before enabling more refined
control. Semantic requirements must survive another analysis's refinement.

Adapt dependency-slice code/tests from `dc13183ac` and occurrence/lifecycle
indexing from `55e706384`. Do not copy the WIP frontend integration or retain the
single-owner mode grammar as the general contract. Exercise carried versus IV
selectors, expressions/views, unrelated state, independent/coupled selectors,
ambiguous overlaps and the old CSA lost-child-return regression.

The current increment implements dependency-sliced finite address relations,
separate per-period physical-use records, conservative unions, and paired child
entry/body/exit occurrences. It retains original owner membership when one
refined entry has a narrower reachable path. A local event-resource failure can
atomically decline optional observation materialization; external protocols
cannot use that retry. This establishes no event credit and does not compose
storage generations. The existing constructor still realizes one compatible
period per owner. Ordered endpoint gaps, next relevant physical use, partial
FIFO knowledge and the CSA ordering review remain checkpoint-2 obligations.
The draft should distinguish a finite relation from its optional control/event
realization; a proof of the generalized composition awaits those interfaces.

### Checkpoint 3: generation-scoped support and composed endpoints

Represent producer episodes, participating readers, next conflicting uses and
entry/reload/bypass/exit obligations as views over existing facts. Collect all
first-consumer/first-write/final-reader roles with exact gaps and original
participation before refinement. Include retained inputs with two child readers,
reloads, skipped children, outside readers and next overwrites beyond the parent,
then connect these to a Qwen or DeepSeek prefill lifetime.

Replace broad ACC/storage/engine eligibility only after scoped certificates
cover their obligations. Preserve the X/Y residual-fence-relocation negative.
Port reader-region tests and producer-support reasoning, not their whole-engine
admission gates. Open interfaces preserve state; region exits imply no drain.

### Checkpoint 4: common realization and useful required receipts

Use one internal packet contract: requirements/gaps, participation and support,
physical binding, exact staged words, validation, commitment of those same words.
Apply actual transfers in order; only their checked execution grants credit.
Keep complete readiness/release selection ahead of residual repair. Share an
independently required return only when its existing endpoints cover the added
obligation. Generalize structured source coverage along with ordinary coverage.

Adapt ownership/restoration fixes (`7f22b091f`, `b62b89de5`), required-return
sharing, two-sided relay checks and restricted publication certificates
(`edcdcecee`, `01aa6e5a5`) when the common consumers need them. Deferred
acknowledgment policies and replay-cache extensions are later, witness-driven
work. Remove obsolete paths when the replacement meets their responsibilities.

### Migration and final acceptance

Temporary ordering, resource and compile-time regressions are explicit ledger
entries, never new admission filters. Soundness, matching, ownership and rearming
remain mandatory. Preserve enclosing-cycle and bank composition throughout.
Any future conservative fallback remains restricted to qualified local kernels;
peer kernels are outside that fallback contract.

At each default checkpoint run focused checks and the supported corpus. Record
complete added/removed payload-order sets, resources, work and host time
separately. Final acceptance restores demonstrated GEMM/MAT and targeted gains
through shared mechanisms, explains other corpus tradeoffs, and measures actual
coupled device execution. Tests do not substitute for general proof; amend the
draft when an established mechanism needs formulation or new proof obligations.

The TODO tracks work, this document defines sequence and invariants, and the
HANDOFF records exact current evidence and the next action. Historical results
from the old branch are donor evidence, never validation of this checkout.
