# OAHS current handoff

## Active baseline decision — 2026-09-22

The user selected the source snapshot pinned by the Shenggan GEMM parity
campaign as the baseline for further development:

- Commit: `16564fa8ae7282631f20ce112c10bab0cf69ed36`.
- Branch: `codex/oahs-gemm-base`.
- Worktree: `/home/toni/work/pypto3_sync_more/PTOAS-oahs-gemm-base`.
- Rebased counterpart: `f2d24ece1`; its OAHS code and portable tests match.

Code remains at that snapshot. No new build or validation has been run here;
device parity is historical evidence recorded in the later campaign report.
The current implementation and the main-based scaffold remain in their separate
worktrees. The scaffold's build is stopped.

Next: retain the demonstrated bank/readiness/release mechanics, review later
correctness fixes separately, and first port the shared InsertSync semantic
integration from `5e0a72772` without importing its unrelated constructor changes.
Do not resume the zero-based rewrite. Keep aggregate local build/test workers
at two or fewer. Source this worktree's `env.sh` for its isolated configuration.

The checkout notes and validation below are historical notes from this snapshot.

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
