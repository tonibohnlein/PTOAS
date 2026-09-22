# OAHS current handoff

## Active checkpoint — harden the retained GEMM constructor

- Worktree: `/home/toni/work/pypto3_sync_more/PTOAS-oahs-gemm-base`.
- Branch: `codex/oahs-gemm-base`; committed baseline: `37554ef9b`.
- Linear upstream base: `66bd855ed`; GEMM snapshot: `f08b28194`
  (historical source `16564fa8a`).
- Included ports: shared InsertSync instruction translation `dbe56f7e6`,
  documented ACC access-order correction `37554ef9b`.
- Current task: implement the dependency-ordered continuation in
  [semantic corrections](docs/designs/oahs-semantic-corrections.md).
  Do not merge the old WIP refactor wholesale or restart again.

Source `env.local.sh`; the tracked `env.sh` contains historical machine paths.
Preserve `.codex/CLAUDE.md` line-ending noise and local workspace configuration.
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
