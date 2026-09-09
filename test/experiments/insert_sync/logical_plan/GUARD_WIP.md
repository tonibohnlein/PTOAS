# Guard generalization — paused WIP

Paused at the user's request on 2026-09-09. This is **not an accepted guard
milestone**. M3/GEMM/allocation development has not started. The compiler default
and the logical work allowance are unchanged.

The working implementation adds relation-derived bound differences and small
parameter residues, original-definition-domain arithmetic qualification, and a
pre-emission bound on recursive guard expansion. Containment has sufficient
rational proofs and negative-only integer sample checks, retaining exact
subtraction for unresolved cases. Guard selection also avoids repeating old
candidates and uses intersection to detect whether a clause contributes.

## Recorded checks

All artifacts remain outside the source tree under
`insertsync-builds/campaign/logical-plan/guards-01/` in the parent workspace.
These are worktree builds, not a new device campaign.

| Check | Recorded result |
| --- | --- |
| Incremental native compiler and test-driver builds | Passed, including final guard-selection edit |
| Native MLIR/libisl relation parity | 75 passed (`relations4`) |
| Occurrence qualification | 28 passed; 1,936 softmax phase-pair comparisons (`occurrences-final`) |
| Known-local requirements | Five inputs passed (`requirements`) |
| Boundary/replay observer tests | Eight passed |
| Guard expansion limit | Six checks passed, including exponential-size and excessive-depth refusal |
| M1/M2 checkpoint population | Four unchanged inputs preserve mechanisms, payload, scalar replay and boundaries in `acceptance2` |
| Structured skipped/empty checks | 24 observations passed in `acceptance2` |
| Two-buffer | Strict PTO construction, C++ emission and nine replay/boundary observations passed in `acceptance2` |
| Three-buffer | No accepted output; `acceptance2` timed out at 90 s, latest `cover-probe` at 120 s |
| Latest two-buffer native positive | Passed in 74.5 s, 35,749,849 work units (`guard-negatives/none`) |
| Changed residue / removed nonnegative guard | Both rejected with original input preserved (`guard-negatives`) |
| Changed slot-distance guard | Test run interrupted at user request; no result claimed |
| Complete final acceptance / device tests | Not completed / not run |

The `acceptance2` run predates the final candidate-reuse/intersection cost edit.
Its four checkpoint comparisons and buffering observations must be rerun for
final milestone acceptance. The latest native positive and two negative tests
include that edit. Source reviewers found no blocking defect, but their final
acceptance was conditional on completing the test gate.

The earlier `acceptance` run exposed a test-observer error: arithmetic private
to a nested synchronization guard was counted as payload. The observer now
handles that case and has an escaping-result negative test. This was not a
demonstrated payload mutation by the compiler.

## Remaining work

1. Resolve three-buffer compile cost and establish complete native construction.
   The latest trace reaches guard preparation and subsequent relation queries;
   it records an exact difference of 99/14 pieces taking about 17 seconds.
   Raising work budgets is not a correctness or performance result.
2. Finish the slot-distance corruption check and rerun the full focused gate on
   the final source, including C++ emission, payload/boundary preservation,
   negative/empty execution, and existing reconstruction/fallback checks.
3. Record compile and scalar-control costs separately from synchronization.
   Intermediate two-buffer output has 8 sets / 10 waits, two MTE3 sites and no
   PIPE_ALL. At 16 trips it executes 61 matched pairs and 15 MTE3 barriers,
   versus upstream's 68 pairs, 16 MTE3 barriers and one PIPE_ALL. Scalar/control
   commands increase substantially. These are not device timing results.
4. Obtain the three final review acceptances before calling this milestone done.

The test runner intentionally requires successful three-buffer construction;
it is not currently green. Do not replace that effectiveness gate with fallback
or relabel this WIP as completed guard generalization. The research worktree,
including its original unfinished changes, remains preserved.
