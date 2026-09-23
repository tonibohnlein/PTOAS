# OAHS current handoff

## Current iteration — Step D physical-use succession

Step C committed `54cee6cda`. The existing storage analysis now owns cached
nearest-use frontiers, including read/write/RMW and explicit open boundaries.
The recurring qualifier consumes this query and its private traversal is removed.
Incomplete facts cause explicit refusal. This is succession, not generation
completion or acquired causal credit; scoped support is the next responsibility.

All three reviewers ACCEPT the final source. The initial 22 portable suites and
four focused suites after the incomplete-result/RMW amendments pass. All three
native drivers and the pinned supported-input campaign pass: 88/88 corpus,
18/19 compatibility (unchanged topk refusal), GEMM, two prefills and three targets.
Every plan/outcome equals Step C. Artifacts: `../oahs-gemm-base-builds/refactor-step-d/`.
GEMM charges 2,792 physical-use visits, with unchanged replay/resources. Two
compliance bracing reports are nested-condition regex false positives; both
bodies are braced. A test-only line wrap follows the campaign. Do not mark
full generation support complete from this query alone. Continue immediately
into composed generation admission and affected producer-support obligations.

## Current iteration — Step C publication boundaries

Step B committed as `3ee150030`. Step C replaces the two private recurring
publication scans with `Control::publicationAfter`: first legal boundary,
no crossed payload, every original source-word occurrence, distinct source and
boundary words, and exact equality with the shared matching pairs. The gate
inventory and its unknown cases are documented in `oahs-analysis.md`.

22 portable suites pass. Real native equivalence tests now assert query work,
no optional retry, and retained per-bank readiness/release for carried/IV and
equivalent scalar selectors, a view, unrelated carried state and unrelated
control. All three native drivers and the pinned corpus campaign pass; every emitted
plan remains identical to Step B. Artifacts: `../oahs-gemm-base-builds/refactor-step-c/`.
GEMM charges 87 boundary-scan and 74,466 correspondence visits, with unchanged
21,846 replay visits and 16 recurring channels. All three reviewers ACCEPT.
The result is a position/participation proof, not a cell/generation certificate.

## Current iteration — Step B endpoint-set matching

Step A committed as `b6958b0e4`. Step B replaces recurring, alternative-source
and loop-entry matching with the common endpoint-set query. Both obsolete
monitors and the unused lookahead graph copy are removed. The independent token
oracle now tests common matching, including cycles and same-word ordered pairs.
The 32-source/shared-40-site-suffix test charges source identities explicitly.

22/22 portable suites and all three native drivers pass. Pinned Step B results:
`../oahs-gemm-base-builds/refactor-step-b/`. All 88 corpus, 18/19 compatibility,
GEMM, two prefills and three targeted outcomes and plans equal Step A. No new
ordering or resource change is hidden in those comparisons.

Occurrence-query visits: corpus 173,563 → 236,014; compatibility 148,729 →
222,673; GEMM 0 → 24,459. The removed balance traversals were uncounted, so
this is not all newly added work. Nonetheless the campaign reports constructor
time corpus 52.06s → 67.34s and GEMM 0.825s → 0.973s. The largest corpus tail
also slows while issuing no occurrence query. Serial alternating pinned runs find comparable times: corpus case 50 is
about 14.1s on both binaries; GEMM about 1.2s on both. The older campaign
timings do not isolate a change-induced slowdown. Do not claim a speedup.
The bounded query may conservatively refuse a large alternative-source/suffix
population; preserve that limitation until a measured representation change.

All three reviewers explicitly ACCEPT Step B. Per user steering, performance
acceptance addresses unnecessary asymptotic complexity and repeated work, not
small timing differences. The only final source changes after the campaign add
braces to two existing one-line conditionals. Continue immediately
with common publication-boundary queries after this step is committed.

## Active checkpoint — common lifecycle/occurrence consumers (2026-09-23)

Branch `codex/oahs-gemm-base`, starting HEAD `d5718272e`. Foundation changes have explicit ACCEPT verdicts from all three independent reviewers.
They are being committed as Step A; this does not complete Stage 1.
The user requires an implementation → three-reviewer acceptance → commit loop,
continuing through all stages without stopping at intermediate checkpoints.
The canonical [stages 0–5](docs/designs/oahs-semantic-corrections.md) and
[36-donor ledger](docs/designs/oahs-donor-ledger.md) are persistent; TODO tracks
that sequence. Stage 0's host closure is complete. Stage 1 remains active; the
common-query increment below is implemented and host-validated, not the whole
semantic refactor.

### Implemented in this increment

- One `OrderedPacket` materialization path constructs recurring trial words and
  committed words, preserving fixed prefixes, intra-word order and request IDs.
  It does not yet implement arbitrary gaps or the full Stage 4 binder.
- `RequirementFrontiers` owns shared physical-use roles, source-release candidates
  and separate successor/return deadlines. Ordinary entry placement, recurring
  release queries and binding provenance consume those records. The superseded
  private recurring release lookup and old indexing implementation are removed.
- `Control::correspondence` records every source/receipt occurrence pair over
  original control, or returns explicit unknown/disproved results. Binding checks
  all corresponding intervals and every occurrence of an intervening key use.
  Immutable queries are memoized; budget exhaustion grants no partial credit.
- Shared merge-sort effects exclude executed-count writes for `exhausted=false`.
  Verified the emitted template argument and local A2/A3 and A5 PTO-ISA paths at
  `0c112d61f41342bd0867ce1080c29f1590d72484`. Shared effects/translation tests cover
  those architectures; native construction retains its A3 contract. No opcode
  admission gate or target expansion was introduced.
- The new first-use test wrongly required a single exit. Its corrected assertion
  checks both the copied zero-trip exit and original shared-suffix exit, with the
  same original boundary identity. Production occurrence behavior was unchanged.
- The draft's endpoint-qualification section now describes bounded occurrence
  pairing and one ordered proposal for checking/commitment, including the limits.
  Existing edits to its references and GEMM supplement were preserved.

### Current evidence and its limits

Evidence: `../oahs-gemm-base-builds/refactor-stage0/` and
`../oahs-gemm-base-builds/refactor-stage1/`. The latter contains the supported-input
manifest, original source hashes, pinned candidate binaries, plans, native logs
and `summary.json`. All prepared-source hashes match the prior physical-use run.

- **22/22 portable suites pass**, including correspondence, budget exhaustion,
  hidden intervening key use, two-reader deadlines, reload provenance, independent
  bank/child composition and unchanged independent causal references.
- **All three native drivers pass**, including the new shared merge-sort effects
  and existing equivalent scalar/view/independent-period cases.
- **88/88 corpus**, **GEMM**, three targeted kernels and both Qwen/DeepSeek
  prefill witnesses construct/reconstruct; their emitted plans are identical to
  the recorded physical-foundation baseline. The previously passing compatibility
  plans are unchanged too. GEMM retains 330/652/1296 pairs and the narrower refill
  ordering from hardening; no new device timing is claimed.
- Compatibility service coverage is **18/19**, formerly 17/19. The newly accepted
  `kernel_softmax_prepare` is explicitly **an observation decline and checked
  original-control retry**. The new occurrence check refuses an invalid refined
  binding before it can become a later byte-completion failure. This is not a
  successful refined protocol, a merge-sort gain or resolved generation support.
  Qwen `topk_select` remains the inherited refusal.
- Corpus replay stays at 448,848 visits and GEMM at 21,846. New immutable occurrence
  queries add 173,563 corpus visits, reported separately. The recorded single host
  run totals about 52.1s of constructor time vs 52.8s previously; this is not a
  controlled speedup claim. Discarded query work is exported too.
- No sanitizer or device campaign was run. A focused-link script initially read
  the compile manifest before CMake regeneration, omitting the new sources; it
  now refreshes that list first. No project-wide rebuild was required.

### Classified CSA regression and recovery obligation

The previously unreviewed CSA compatibility change is **mixed**, not event-key
renaming: +20/-20 payload start/completion relations for one all-active outer
visit, +40/-40 for two; alternating participation gives +8/-8 and +16/-16.
The A extraction moves before the B-readiness exchange; its immediate return
then carries A completion to a later MTE2 external receive. The vector function
is byte-identical. Current changes retain this prior physical-foundation plan.

`refactor-stage0/csa-order.py` and `csa-order.json` retain exact plan hashes and
nine finite complete local-order comparisons. External set/wait ports are
included, but peer progress and hidden lowering effects are not modeled. The
added and removed orders are reported separately. Stages 2–4 must recover this
through scoped support/placement/rearming, not by discarding better physical facts.

### Remaining boundary and exact next action

Continue Stage 1 by connecting shared source/receipt correspondence to the
remaining placement consumers and inventorying partial physical/FIFO facts.
Do not widen control refinement until every affected original child interface
remains available. The existing single-owner recurrence grammar, private enclosing
observation qualifier and one-period materialization restriction remain explicit
limitations. Generation-scoped support, joint endpoint roles, prefix certificates,
arbitrary ordered gaps and general deadline binding remain stages 2–4.

The new records describe candidate physical support, never completion. Existing
relationship-fallback omission trials remain; replace their responsibility with
actual support at Stage 4, without adding another completed-plan search. Final
reference recovery, coupled device work and independent generality review remain
open. See the canonical regression ledger and the draft for proof obligations.

### Changed-code review

The fast changed-code checker has nine reviewed `G.FMT.11-CPP` false positives:
its line regex mistakes nested-call parentheses and continued conditions for
unbraced bodies. Every reported body has braces. Exact findings and line-by-line
review are in `refactor-stage1/compliance.log` and `compliance-review.md`. The real
125-column initializer finding was fixed. No suppressions or checker weakening
were added; `git diff --check` passes. Compiler builds introduce no warnings.

## Historical physical-foundation checkpoint (before this increment)

Branch `codex/oahs-gemm-base`. The preceding hardening checkpoint is committed
as `be14229f2`; this checkpoint adds a general physical-use and occurrence
foundation. The complete dependency order and acceptance boundary are in
[semantic corrections](docs/designs/oahs-semantic-corrections.md).

- Native scalar analysis follows each relevant address's original dependency
  slice. Carried and induction-derived selectors, dialect-folded scalar
  expressions and address-preserving views can yield finite physical footprints.
  Unrelated carried values and event-pool size do not bound the physical fact.
  Independent periods remain separate `PhysicalUseRelation` records. The
  current control constructor materializes one compatible period per owner;
  an incompatible relation contributes its conservative may-footprint.
- `ObservedLoopOccurrence` pairs each child copy's entry, body, members and
  possible exits. First-use refinement recomputes the reachable child paths
  while preserving the original owner's complete membership. The owner/reachable
  distinction was required to retain the established GEMM recurring protocol.
- If optional native observation materialization cannot bind events, a local
  kernel gets one fresh OAHS attempt on original control, preserving the
  physical facts. Shared peer/queue protocols do not take this retry. Failed
  admission work and reason remain visible in the report.

Recorded evidence: all three native drivers pass, including independent
period-2/period-3, coupled period-6 and 17-bank cases. The pinned campaign
constructs and reconstructs **88/88 corpus**, **GEMM**, three targeted kernels
and two Qwen/DeepSeek prefill witnesses; **17/19 compatibility** pass, retaining
the two inherited refusals. Corpus and GEMM plans match `be14229f2`; one
DeepSeek CSA compatibility plan changes and needs a complete ordering review.
Results and per-case commands are in
`/home/toni/work/pypto3_sync_more/oahs-gemm-base-builds/physical-use/`.
These are host results, not new device measurements.

At the user's request, final validation was stopped. The portable build/test
run was interrupted (exit 130) after the last occurrence-test additions, and
no final changed-code check or complete ordering comparison of the changed CSA
plan was run. Treat checkpoint 2 as **an implemented foundation with acceptance
open**. The next action is to inspect that CSA ordering and finish the focused
portable checks before using these facts to widen construction. Later work must
add generation/support intervals, exact endpoint gaps/participation and a common
packet/binding contract; none is supplied by these records alone.

## Historical checkpoint — harden the retained GEMM constructor

- Worktree: `/home/toni/work/pypto3_sync_more/PTOAS-oahs-gemm-base`.
- Branch: `codex/oahs-gemm-base`; committed baseline: `37554ef9b`.
- Linear upstream base: `66bd855ed`; GEMM snapshot: `f08b28194`
  (historical source `16564fa8a`).
- Included ports: shared InsertSync instruction translation `dbe56f7e6`,
  documented ACC access-order correction `37554ef9b`.
- Current task: implement the dependency-ordered continuation in
  [semantic corrections](docs/designs/oahs-semantic-corrections.md).
  Do not merge the old WIP refactor wholesale or restart again.

Workspace environment backup: `../oahs-gemm-base-builds/workspace-local-backup/env.local.sh`.
The tracked `env.sh` contains historical machine paths. Local configuration was
moved out of the source tree; line endings were normalized in `d5718272e`.
Use at most two resource-intensive workers across all local commands.

### Hardening result and exact evidence

The first continuation checkpoint is implemented. The algorithm still uses the
retained enclosing-bank constructor and unchanged independent causal checker.

- First-use refinement rejects decisions not separated by a tracked transition.
- Recurring sharing preserves exact endpoint positions; broad one-sided and
  two-sided release motion is removed pending a contextual ordering certificate.
- Local fences are decided at the current occurrence's deadline. Shared emitted
  words still share commands; separate future words receive no speculative fence.
- One failed optional recurring attempt is discarded completely before a fresh
  ordinary OAHS attempt from the same input/fixed words. Its reason and work are
  exported in `declinedRecurring`; no new serializer or subset search is used.
- The two invalid argument-mutation fixtures were replaced with legal static and
  unknown argument-shape cases. The dialect disallows `set_validshape` on function
  arguments; local-tile branch-join variants remain covered. No verifier changed.

Artifact directory:
`/home/toni/work/pypto3_sync_more/oahs-gemm-base-builds/hardening/`.
Focused build scripts live in its sibling `migration/`; the link includes the
shared SlotAffineAnalysis dependency. Baseline binaries are pinned in
`hardening/baseline/`. Validation results:

- **21/21 portable suites**, including native-core first-use, deadline-fence,
  six-visit distinct-release and optional-resource rollback witnesses.
- **All three native drivers pass**, including 24 supported ACC variants and
  unchanged operand-release/FIX negatives. The initial baseline selected driver
  failed only at the invalid argument-mutation fixture; see the retained logs.
- **88/88 corpus** construct and reconstruct on baseline and candidate.
- **17/19 compatibility** on both. Existing failures: Qwen `topk_select` and
  `kernel_softmax_prepare`, unresolved original byte completion. No new failures.
- **KDA, RMSNorm and hc_pre**, both original Qwen/DeepSeek prefill witnesses and
  **GEMM** construct and reconstruct. These five targeted/prefill plans are
  unchanged; GEMM changes as described below.
- Nine corpus plans, two compatibility plans and the separate GEMM witness
  change. Five changes are certified bijective key renamings with otherwise
  identical ordered commands/control. Concrete complete start/completion order
  comparisons on the remaining changed functions add **zero** relations.
  These finite traces are not universal ordering proofs.
- GEMM removes **56/120/248** payload relations for 1/2/4 tiles. The independent
  physical/event oracle retains all previous overlap checks and now requires
  **14/28/56** A-refill boundaries free of the later B-reader prerequisite.

Current GEMM event populations are **330/652/1296 pairs**, compared with the
historical **182/360/716**. No named barriers, one terminal ALL. This is an
explicit resource/possible latency regression, not a new device timing claim.
Native replay work is unchanged for GEMM (21,846 visits) but grows on several
fence-heavy kernels: final RMSNorm 2,499 -> 5,907; row RMSNorm 1,643 -> 3,417;
TopK 1,111 -> 2,349; route_sort 8,028 -> 17,582. These are recovery obligations,
not permission to restore speculative fences or broaden release prefixes.

Evidence: `summary.json`, `ordering.json`, `portable-final.log`,
`pto-oahs-selected-test-final.log`, `gemm/trace-final.log`. The campaign summary's
initial selected-unit rc=1 refers to the invalid fixture; the final unit result
is recorded separately. Source changes after that campaign only correct test
fixtures, strengthen test oracles, and update documentation.

The changed-code prefilter's four bracing reports are false positives on compound
conditions: all four bodies have braces. Its local `env.local.sh` header report
is outside the source change. The real added-line length issues were fixed.
No sanitizer or device campaign was run.

### Next checkpoint and regression ledger

Implement shared physical-use facts together with stable child occurrence
queries. Keep the API invariant explicit before enabling broader refinement.
Do not import the old WIP integration wholesale. Remaining inherited compatibility
failures need the corresponding original-control/support fixes, not a serializer.

| Witness | Remaining issue | Responsible checkpoint / recovery |
| --- | --- | --- |
| Shenggan GEMM | More release channels/commands despite strictly narrower checked ordering | Common packet/required-return coverage; retain independent A-refill check and remeasure device time |
| final/row RMSNorm, TopK, route_sort | More contextual replay after removing speculative future fences | Shared immutable preparation / measured replay work; preserve deadline-local decisions |
| Qwen topk_select, kernel_softmax_prepare | Baseline and candidate completion refusal | Occurrence/support interfaces; restore native construction/reconstruction |
| Qwen/DeepSeek prefill | Isolated GEMM improvements still not established in coupled prefill | Physical-use and generation composition, then authentic device comparison |


## Historical GEMM snapshot and campaign evidence



Updated: 2026-09-18

## Checkout

- Repository: `/home/toni/work/pypto3_sync_more/PTOAS-oahs-clean-m1`
- Branch: `codex/oahs-clean-m1`
- Base HEAD: `8d08bbe6d8043822e9dd950bf4be1934ea75cd48`
- Working tree: uncommitted bank-occurrence, shared-release, first-use, replay-cost, tests, and documentation changes

Verify the branch, HEAD, and working tree before continuing. A newer user commit supersedes this record.

## Delivered local synchronization plan

The current implementation composes three direct constructor mechanisms:

1. A finite physical-bank occurrence interface carries the outer MAT bank residue across the nested reader region. It relates each overwrite to the previous participating use of the same bank without expanding a product of nested first/middle/final modes.
2. Two operands in one qualified physical-bank episode retain separate early readiness transfers but share one exact storage-release return. This preserves early A extraction and removes the redundant second MAT release channel.
3. A first-use prefix qualifier recognizes the original conjunction `outer_k == 0 && inner_k == 0`. It splits one entry prefix, shares the remaining graph, and removes the impossible repeated ACC-initialization paths. It supplies no completion credit and retains conservative behavior for incomplete conjunctions, disjunctions, and unsupported loop forms.

Exact qualified cycles bypass whole-plan omission trials. Guarded copies of one original local-fence decision are batched into one selected update and one emitted command word. These changes avoid the expensive candidate-analysis/refinement path used by earlier experiments.

## Exact Shenggan result

Generated artifact:

- `/home/toni/work/pypto3_sync_more/bank-occurrence-work/shared-release-first-use.pto`
- SHA-256: `3f1e19bf600e137685e0fdd4a49b5ea8ec4e19f1bf53f375618190ea2dfab8f2`
- construction log: `/home/toni/work/pypto3_sync_more/bank-occurrence-work/shared-release-first-use.log`

For identical concrete payloads:

| Tiles | Event pairs | Named barriers | Terminal ALL | Added / removed vs compact MAT | Added / removed vs manual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 182 | 0 | 1 | 0 / 446 | 0 / 16 |
| 2 | 360 | 0 | 1 | 0 / 928 | 0 / 32 |
| 4 | 716 | 0 | 1 | 0 / 1,892 | 0 / 64 |

The manual protocol has 167 event pairs for one tile. The 15-pair difference preserves separate early A/B readiness; the independent command-graph comparison finds the generated plan to be an ordering subset of the manual plan on all checked traces. Event count alone is not a performance estimate.

The one-tile direction counts are:

```text
M -> MTE1      66
MTE1 -> MTE2   18
MTE2 -> MTE1   32
MTE1 -> M      64
FIX -> M        1
M -> FIX        1
```

The trace checker validates one, two, and four output-tile entries. It checks memory conflicts, native ACC ordering, event occupancy, consumption before republication, final balance, same-bank prefetch, separate early A readiness, and the absence of current-bank compute ordering before different-bank preparation or parent DMA. Its injected terminal drain remains a discriminating negative control.

## Construction cost

The exact run reports:

```text
sites                         718
selected updates                3
replay site evaluations     21,846
recurring channels             10
recurring trials                0
recurring analysis sites        0
construction time          ~0.46 s in the recorded run
peak RSS                    ~112 MB
```

The first-use qualifier adds 67 analysis sites to the 651-site bank graph, but does not create a nested mode product. The expensive recurring omission population is gone. Construction time remains suitable for corpus and device qualification; repeated immutable structure construction remains a later optimization target.

## Validation completed

- standalone OAHS CTest: 20/20;
- portable bank occurrence tests: guarded and repeated entry, two and three banks, non-identity bank mapping, malformed boundaries, and exact shared-release structure;
- portable first-use tests: 145 concrete traces, zero trip, repeated entry, malformed boundaries, and genuinely repeating initialization negatives;
- native selected driver, including first-use conjunction and negative variants;
- exact native construction and reconstruction;
- exact Shenggan lit test: frontend lowering, FileCheck, and independent trace checker;
- `git diff --check`.

Sanitizers and device execution were not run locally.

## Device qualification required

The local result matches or improves the manual plan's checked payload ordering, but device latency is not established. The next device task should compare:

- the current generated 182-pair, zero-named-barrier arm;
- the reconstructed manual arm;
- the prior 200-pair bank-qualified arm;
- compact MAT;
- existing and the frozen previous handoff where useful.

Use the project reference compiler flags obtained from `_kernel_compile_flags`, verify `-O2` in both device compiler invocations, retain numerical checks, and report MAC/MTE/scalar profile data. Do not infer performance from event or barrier counts.

## Remaining work

1. Measure the current GEMM plan on device before changing its protocol again.
2. Redesign guarded attention occurrence correspondence from remaining obligations; the scalar bank orbit does not change the six representative attention plans.
3. Restrict broader recurring endpoint coalescing with an ordering certificate and make optional specialization decline cleanly under key pressure.
4. Reduce repeated immutable control/storage construction after plan quality is settled.
