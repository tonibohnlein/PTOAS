# Boundary-preserving residual construction

This increment preserves incoming readiness providers through the existing
lifetime/residual transaction and rejects optional combinations whose incoming
completion becomes broader. It removes the designated GEMM, softmax and QK
regressions by retaining the previous verified option where composition still
has a quality tradeoff. It does **not** claim that those lifetime combinations
now compose efficiently. Qualified GEMM retains seven lifetimes and 54 SET/54
WAIT sites under both hardware profiles.

## Construction and verification

Incoming providers retain the original completion demand: source prefix,
publication/acquisition cuts, cell witnesses, participation and loop generation
scope. Retained identities include source, observer, old physical key and
owning loop. Physical key reuse alone no longer merges entry generations with
recurring or differently scoped protocols.

Readiness, lifetime priming/release and event acknowledgment are separate.
The constructor preserves supported forward First/NonEmpty providers. It does
not automatically copy their reverse acknowledgment: another actual handoff
may already carry the consumption receipt. Missing obligations are reconstructed
after applying the proposed and retained commands' effects. Ordinary readiness
retention matches a definite writer/read and the actual publication/acquisition
key. Exact cuts remain usable even when loop prefix origins are unknown;
equivalence between different cuts requires a known common source origin.

The open verifier supports a narrow First/NonEmpty population. All endpoints
share the same nonempty decision at their common structural owner; the nonempty
case peels the first visit before subsequent-visit fixed-point analysis. Fresh
entry-demand reconstruction and the existing entry-word projection constrain
the first-consumer cut. Unsupported participation remains conservative.
Nested loop decisions are rebound for each outer visit. Native extraction
excludes `unsignedCmp` loops from signed first/nonempty and next-iteration
facts. No signedness reinterpretation supplies a participation guarantee.

Immutable byte, resource, visibility and event-reuse obligations still feed
global allocation and fresh emitted verification. Existing bounded retry
without optional completion groups remains available; no pipe-pair deletion,
ACC exclusion weakening or MMAD operand-release credit was added.

Before selecting a migrated entry-provider candidate, a bounded comparison of
the baseline and candidate's completion observations screens for additional
incoming writer completion at original consumers. Unresolved comparisons retain
the verified baseline. This is a **coarse nonregression screen**, not a
generation-level boundary certificate: abstract writer bits merge generations
and varying visits can cause conservative rejection. Concrete generation replay
and mutation tests independently establish the reported tested boundaries.
The screen does not certify every release or acknowledgment boundary.

## Pinned final local evidence

All paths below are relative to the parent workspace. Runs use the dirty
implementation above `0b54eff6f71699b5be9fc8d2e564b197d367d9c1` and preserve
earlier authorized changes. The tracked source diff hash at measurement is
`1fcf089c762777c4833e82bc402cee12551be33160486502aa8ae750ef6d7087`.
The final compiler library SHA-256 is
`90f7b58241b981d80c351e9c1b9f6454c5e752128a9a2ccbfe41fa5f19c69742`.
Later documentation, test-only edits and a source-comment correction do not
relabel these compiler runs. The final focused core run includes the additional
entry acquisition-count assertion; the matrix/native/corpus binaries predate
only those nonfunctional compiler edits.

Artifacts include input/output hashes, binaries, commands, alias contracts,
hardware profiles, scalar guards and physical keys:

- `oahs-coverage-work/quality-matrix-boundary-final-r1`: controlled matrix,
  family witnesses, endpoint inventories and five-launch generation replay.
- `oahs-coverage-work/benchmarks-boundary-final-r1`: all eight benchmarks,
  existing/composition/precision arms and subsequent C++ emission.
- `oahs-coverage-work/boundary-observations-final-r1`: comparison with the
  pre-regression `oahs-regression-0b54eff6f-GLbQqv/eight-case` outputs.
- `oahs-coverage-work/boundary-immediate-comparison-final-r1`: comparison with
  the immediately preceding `benchmarks-quality-final-r1` increment.
- `oahs-coverage-work/boundary-witnesses-final-r1`: fresh softmax/QK rejection
  witnesses. These diagnostic recompilations are separate from timing samples.
- `oahs-coverage-work/composition-boundary-final-r1`: native and mutation gates.
- `oahs-coverage-work/campaign-boundary-final-r1` and
  `publication-flow-boundary-final-r1`: frozen corpus and bounded writer paths.

The previous increment's measurements in
[oahs-synchronization-quality.md](oahs-synchronization-quality.md) remain
historical evidence, including their mixed quality results.

## Controlled GEMM matrix

The prepared original input and qualified input have identical payload,
physical assignments and A/B/C alias guarantees. Only scalar metadata changes:
argument 3 is [0, 2147483520], multiple 128; argument 4 is
[256, 2147483136], multiple 256; argument 5 is [512, 2147482624], multiple 512.
Launches and profiles match the previous matrix. Original input SHA-256 is
`97076a5d31bd18eb5c0bf50316c6240c47d15a3a4da19a3c5b4d27b70ae70ce9`.

| Input / planner | Profile | Discovered / selected | SET | WAIT | Named barriers |
| --- | --- | ---: | ---: | ---: | ---: |
| Original / precision composition | Conservative | 5 / 0 | 56 | 59 | 21 |
| Original / precision composition | Qualified MMAD | 5 / 0 | 56 | 59 | 10 |
| Qualified / precision composition | Conservative | 7 / 7 | 54 | 54 | 8 |
| Qualified / precision composition | Qualified MMAD | 7 / 7 | 54 | 54 | 0 |
| Qualified / InsertSync | Existing | — | 44 | 44 | 21 |
| Original / InsertSync, additional control | Existing | — | 44 | 44 | 21 |

Every arm has one additional terminal ALL site. Original composition rejects
the optional combination at `persistent-boundary`: additional incoming
completion at node 54, the first extraction. It retains the original guarded
plan. Qualified construction does not depend on MMAD credit for its lifetime
handoffs: conservative mode adds eight static M barriers, preserving the
qualified handoff observations and boundaries checked by the native gate.

| Launch | Original conservative SET / WAIT / named | Qualified conservative SET / WAIT / named | Qualified MMAD SET / WAIT / named |
| --- | --- | --- | --- |
| Empty grid | 0 / 0 / 0 | 7 / 7 / 0 | 7 / 7 / 0 |
| One panel | 27 / 27 / 12 | 31 / 31 / 4 | 31 / 31 / 0 |
| Two panels | 52 / 52 / 19 | 53 / 53 / 8 | 53 / 53 / 0 |
| Three panels | 73 / 73 / 29 | 75 / 75 / 12 | 75 / 75 / 0 |
| Distributed tiles | 104 / 104 / 38 | 99 / 99 / 16 | 99 / 99 / 0 |

Each execution also has one terminal ALL. Every composition matrix replay
establishes generation consumption, consumption receipt before rearm and final
payload retirement in the concrete observer. This is not device validation.
Existing controls retain their previously recorded receipt-proof gaps in that
observer; balanced token counts alone do not resolve those gaps.

The original 56 SET/59 WAIT sites are not 56 pairs. FIX→M EVENT_ID1 has one
publication and four static acquisition sites; three are unreachable guarded
alternatives in the recorded original program. The generation inventory records
the surviving first-tile acquisition and acknowledgment across tile reuse.
Qualified FIX→M EVENT_ID0 instead uses priming, per-tile acquisition/release
and final cleanup. Buffering release keys similarly have distinct next-visit
and exit-cleanup acquisitions. The endpoint artifacts retain the exact guards,
loop owners, keys, per-generation consumption and retirement evidence.

## Static sites, executed commands and ordering

All 24 benchmark compiler runs and C++ emission runs pass. The precision arm
has the following static sites; one terminal ALL is additional in every row.

| Case | Immediate prior SET / WAIT / named | Current SET / WAIT / named |
| --- | --- | --- |
| One buffer | 4 / 6 / 0 | 4 / 6 / 0 |
| Two buffers | 8 / 10 / 0 | 8 / 10 / 0 |
| Three buffers | 12 / 15 / 0 | 12 / 15 / 0 |
| Four-use | 20 / 22 / 0 | 20 / 22 / 0 |
| Softmax | 17 / 17 / 19 | 17 / 17 / 21 |
| QK | 21 / 21 / 2 | 18 / 18 / 4 |
| Q projection | 17 / 17 / 6 | 17 / 17 / 6 |
| Original GEMM | 61 / 61 / 10 | 56 / 59 / 21 |

Softmax rejects two proposed lifetimes at node 210 (first final divide); QK
rejects four at node 49 (first extraction). Both retain their verified ordinary
options. At n=16, softmax returns from 157 to 129 executed SETs and WAITs,
while named barriers return from 243 to 259. QK returns from 276 to 243 SETs
and WAITs, while named barriers return from 32 to 64. These are explicit
tradeoffs, not a claim that event reductions compensate for barriers.

The recurring traffic came from replacing guarded readiness with recurring
closed exchanges and from unconditional lifetime priming/cleanup. Softmax now
has eight loop SET sites executed 15 times plus nine outside sites instead of
ten loop sites plus seven outside. QK has 15 loop sites executed 16 times plus
three outside instead of 17 loop sites plus four outside. The empty original
GEMM launch again executes zero events rather than five priming/cleanup pairs.

All tested cross-lane prefix observations now match the pre-regression baseline
for all eight cases. In particular A's first extraction no longer requires B's
later preload, and softmax's empty-loop first final divide/conversion/store
again require only the first three initial loads. Buffering and Q projection
boundaries are unchanged.

Against the immediate prior mixed-quality plan, restoring the prior option also
loses some earlier release boundaries. At QK n=16 there are 16 improved and 30
worsened prefix observations; distributed original GEMM has 3 improved and 33
worsened. These are the reverse of the earlier increment's tradeoffs. They do
not cancel, and this increment does not claim dominance over both options.
Softmax's empty-loop scenarios improve three observations, with no changed
observations in its nonempty scenarios. The remaining implementation work is
to combine the guarded readiness boundary with the useful release improvement,
instead of choosing between these verified plans.

Precision compilation times are 0.406–0.458 seconds in one paired trial without
warmup; ratios to existing are 0.945–1.075. One other serial validation worker
ran concurrently. Timing is telemetry, not a stable performance estimate or
an acceptance threshold.

## Generic and coverage validation

The focused core oracle separates payload issue/completion and independently
checks hazards, actual prefix edges, token consumption and rearm causality.
The new three-cell fixture tests independent preloads, shared A readiness,
zero-trip continuation and repeated loop visits at trips 0, 1 and 3. It checks
that B's later completion is not a predecessor of the first two A consumers.
An executed-mechanism hook additionally checks that the first A consumer
acquires readiness once for both one and three visits, zero times for zero
visits, and that the second consumer performs no separate readiness acquisition.
Continuation readiness and unrelated recurring storage are counted separately.
Changing B's write to overwrite A makes the old plan fail verification.
Existing generic early-vector tests preserve last-reader release before
unrelated work; nested/choice entry tests cover skipped paths, different outer
visits and shared-prefix storage. The fixture accepts a verified fallback;
it is not evidence that every tested program selects persistent lifetimes.

One former missing-ack mutation is now a positive non-implication test: another
actual return already supplies the receipt. The independent oracle checks its
repeated invocations and rearm edges. Missing providers, wrong owners/keys,
reordered choice words, moved first acquisitions and fresh-write mutations
continue to reject. Native signed/unsignedCmp tests check that signed scalar
guards are not inferred for an unsigned loop contract.

Four focused CTest groups pass, including 2,438,598 core assertions, six bounded
writer-flow tests, endpoint tests and coverage-witness tests. Native validation
passes 24 positive cases, 191 mutations (including the qualified GEMM mutation
population), nine expected refusals and two frontend cases.

Frozen coverage remains 253/363 admissions (79 PTOAS, 7 PyPTO, 167 pypto-lib),
with no gains or losses against the immediate baseline. Generated sync,
authored-protocol preservation and no-op success remain separate fields.
All 86 publication refusals remain. No alias promise or GM publication
capability was introduced.

Independent attribution now materializes bounded structural CFG paths and
backedge owner IDs for matching writer candidates. Of 166 matching candidates,
39 have a forward path, 113 have a path through a named backedge and 14 cannot
precede the consumer within one invocation. No report exhausts its allowance.
These are candidate paths, not feasible branch proofs, last-write generations,
cross-invocation exclusions or causal dependencies. Device qualification remains
separate and unperformed.

Architect, algorithms/performance and correctness reviewers found no remaining
blocking issue. The architect's additional entry execution-count assertion was
added and the four focused groups rerun successfully. Reviews performed no
builds; local intensive work stayed within two aggregate workers.

Existing remains the default. Correctness gates pass and the designated
regressions are removed, but general efficient composition of these rejected
families remains open. The structured-reference refusal and device gates remain
open as well.
