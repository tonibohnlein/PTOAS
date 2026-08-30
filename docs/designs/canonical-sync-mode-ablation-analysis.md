# CanonicalSync mode-ablation findings

## Executive result

The current action-first CanonicalSync compiles and freshly verifies all 29
inputs with zero body `PIPE_ALL`. It emits 840 body synchronization
instructions versus InsertSync's 1,057, a reduction of 217 (20.5%). This is a
substantially different result from the earlier refined binary used by the
device agent.

The current branch already contains the experimental infrastructure needed for
the next round:

- selected-mechanism provenance and detailed descriptors;
- independent derived-family masks;
- direct versus direct-pair candidate modes;
- singleton versus pair-lookahead selectors; and
- action-first versus serialization-first objectives.

No new pass infrastructure is required before the first device validation.
The missing piece was a controlled same-revision benchmark harness and a
correctness-first device protocol.

## Aggregate compile/static results

Body counts exclude the automatic final tail `PIPE_ALL`.

| Mode | Compiled | Set | Wait | Targeted barrier | Body PIPE_ALL | Body total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| InsertSync | 29/29 | 446 | 446 | 165 | 0 | 1057 |
| full catalog, no pairs | 29/29 | 302 | 302 | 272 | 0 | 876 |
| full catalog, pairs, singleton greedy | 29/29 | 291 | 291 | 258 | 0 | 840 |
| full catalog, pairs, pair lookahead | 29/29 | 291 | 291 | 258 | 0 | 840 |
| full catalog, serialization-first | 29/29 | 337 | 337 | 221 | 107 | 1002 |

The restricted correctness basis (`D`) and basis plus ownership (`D+O`) are
not general production modes: they compiled only 7/29 and 8/29 inputs. They
fail closed because several kernels need certified lift/factoring mechanisms.

## What pairs do and do not do

Enabling transitive pair coverage while keeping singleton greedy changed 16
outputs and reduced aggregate body synchronization from 876 to 840. Across the
final plans, 78 pairs remained active in 15 cases and supplied 338 extra demand
rows. Therefore pair composition is useful.

Pair lookahead itself changed no output relative to singleton greedy on the
same pair-enabled catalog. It paid additional selection work without improving
this corpus. The production selector should not be justified by these results;
the pair catalog matters, but the two-mechanism greedy move does not yet.

Pair coverage is also not sufficient:

- direct and direct-pair both fail closed on historical GEMM;
- adding ownership synthesis makes GEMM compile;
- pair-enabled and pair-disabled ownership modes both emit 52/52 events and no
  barriers; and
- the selected GEMM ownership plan has zero active pairs.

This supports the proposed model:

```text
direct correctness atoms
+ certified unary lifts and factoring/fusion mechanisms
+ a bounded transitive pair layer
```

Pairs compose existing recipes. Ownership and lifecycle rules synthesize new
physical recipes, which is the essential capability for GEMM.

## Historical GEMM

| Mode | Set | Wait | Targeted barrier | Body PIPE_ALL | Body total |
| --- | ---: | ---: | ---: | ---: | ---: |
| InsertSync | 44 | 44 | 21 | 0 | 109 |
| ownership-only synthesis | 52 | 52 | 0 | 0 | 104 |
| full action-first | 46 | 46 | 4 | 0 | 96 |
| full serialization-first | 50 | 50 | 6 | 0 | 106 |

The old device-tested refined plan was 59/59 events plus 46 targeted barriers
and failed on device. The current full plan is 46/46 plus four targeted
barriers. Its allocator also uses up to six IDs in the busiest directed domain
rather than collapsing nearly everything onto ID0. This is promising static
evidence, not a correctness claim; silicon validation remains mandatory.

The ownership-only and full plans pose a real performance-model question. The
full plan has eight fewer static actions but introduces four loop-carry drains.
The ownership-only plan is event-only. Device timing should determine whether
the smaller plan or the less-serializing plan wins.

## Objective model and PIPE_ALL

The action-first objective produced no body `PIPE_ALL` in any case.

Serialization-first reduced the reported aggregate serialization breadth from
315,380 to 191,407, but it selected the localized fallback in two cases:

- case 10 `rope_kv_cache`: 62 body `PIPE_ALL`;
- case 25 `prefill_c4_kv_score_proj`: 45 body `PIPE_ALL`.

In both cases event pressure was far above eight and the shared repair work
budget was already exhausted before a repair trial ran. This is a useful
negative control and also reinforces the pending repair-termination/reporting
review. Serialization-first must remain diagnostic until its fallback behavior
is corrected and validated.

The action-first objective still has seven static regressions versus InsertSync
even though it wins in aggregate. Static action count cannot decide whether its
additional targeted barriers are cheaper or more expensive than the removed
events. Device and in-core measurements are required.

## Is the performance branch slower because of PIPE_ALL?

Not established. In the older static campaign, the performance branch emitted
35 body `PIPE_ALL` barriers in 17/29 cases. Four of those cases had a larger
unweighted static plan than InsertSync, but 11 had a smaller plan. Cases with
no `PIPE_ALL` also included both wins and regressions. The global drain can be
very costly when dynamically executed in a hot loop, but placement and the
dependency frontier matter more than the static count alone.

A credible test needs:

1. performance-branch CanonicalSync versus InsertSync from the same compiler
   revision;
2. dynamic `PIPE_ALL` execution counts, not only static occurrences;
3. normalized non-sync equivalence;
4. paired silicon timing; and
5. profiles attributing drain stalls and lost pipe overlap to each barrier.

Losses in no-`PIPE_ALL` cases would prove that `PIPE_ALL` is not the complete
explanation. Large within-case drain stalls in loss cases, absent in matched
controls, would support it as a contributor or cause.

## Compile-time note

The harness recorded whole-process wall time while running at most two compiler
processes concurrently. It is suitable for detecting timeouts and large
outliers, but it is not a clean isolated latency comparison because filesystem
caches and paired contention differ by mode order. The device task therefore
requires a separate serial, balanced compile-time measurement. The local run
already shows that pair lookahead does more work than pair-enabled singleton
selection while emitting identical plans.

## Recommended next action

Publish or archive exact compiler commit
`bf89e4e89e3f5640d0e490846e09e60934a4c8b0`, then execute the device task
supplied separately from the repository. Correctness of current GEMM on the
devices that exposed `507015` is the first gate. Only correctness-closed arms
should proceed to timing and profiling.
