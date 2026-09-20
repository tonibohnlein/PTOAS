# First-consumer readiness placement

2026-09-20. Baseline: `495fb9cbda7649f15a7fc09d6b1d91ffb7737d54`.
Local evidence: workspace sibling `first-consumer-work/`.

## Mechanism

Publish an invariant input's readiness at its original producer prefix; acquire
it once at the first operation that actually reads it. Earlier independent
operations on the receiving pipe remain free to issue. Later iterations retain
that acquired completion. No payload, allocation, native completion contract or
fence-deletion policy changes.

Native admission requires a leaf counted loop with proved positive constant
bounds/step, a straight unconditional prefix, an input not written in the body,
and a producing pipe that does not execute in the reader region. It additionally
requires later source-pipe work on the straight incoming corridor: this milestone
creates first-visit vocabulary where it can avoid a broader publication, rather
than for command-count reduction alone. Unknown/exclusive storage declines.
Nonzero lower bounds and positive non-unit steps are supported.

Only the prefix through the qualifying consumer and its following publication
anchor is copied analytically. Other words and physical operations stay shared.
If the copy reaches the loop terminator, its transition to the repeated header
is forward, not a backedge. The emitted first-visit predicate is the original
`iv < lower + step`, independently read back by native reconstruction. No dynamic
iteration unrolling, history counter, first/tail product or new runtime state is
introduced. Down adds five sites; sibling projection phases add separate small
prefixes.

The ordinary source query cannot use a first-only publication for an unconditional
shared acquisition word. The existing entry-frontier query can instead select a
qualified first-input cut, including when construction visits a later analytical
occurrence first. Coverage still comes from the actual saved publication state;
source classes must be invariant, participation must balance, and the selected
protocol must pass the existing all-path event/rearming trial before insertion.
The subsequent actual replay establishes the credit.

Completed entry protocols may reuse physical keys across sibling regions. This
requires actual empty-token and consumption knowledge at the new publication,
no old endpoint for that key in the new reader region, and acceptance of the
existing all-path protocol trial. Lexical exit alone releases nothing. This
removes the permanent-reservation failure exposed by KV and gate/up; it does not
add a search over complete plans or remove independently necessary returns.

## Checked plan changes

The finite oracle checks identical instructions, physical effects and original
control contexts, memory conflicts, matching, consumption before republication,
and the complete payload finish-to-issue relation. These are local protocol
checks, not numerical device execution or predicted stalls.

| Checked execution | Baseline event pairs | Candidate | Relations removed | Added |
| --- | ---: | ---: | ---: | ---: |
| Down, 17 chunks, one tile | 359 | 343 | 184 | 0 |
| Down, 17 chunks, two tiles | 712 | 680 | 368 | 0 |
| Gate/up, one active phase | 418 | 398 | 230 | 0 |
| Gate/up, both phases | 830 | 790 | 460 | 0 |
| KV, one active phase | 418 | 398 | 230 | 0 |
| Q/out AIC, one tile | 418 | 398 | 230 | 0 |
| LM head, bounded one tile | 393 | 363 | 235 | 0 |

Across 37 paths, 511,187 local conflicts pass. All checked named and terminal
fence populations are unchanged. Direct removed prerequisites include the later
A1/B1 load completions before B0 extraction; transitive removals reach downstream
matrix issues. The earlier A extraction does not acquire B readiness. Source
ordering is improved without joining the A/B publications.

The odd down tail's producer is last on its source corridor and is deliberately
outside this opportunity filter. Its command-count-only reduction remains parked.
The second child's A1 entry publication still includes B1; that is a separate
remaining placement issue. No claim of an optimal projection plan is made.

## Validation and limits

- 22 portable suites pass, including the first-consumer command-graph positives
  and unsafe/order-broadening mutations.
- Native construction/reconstruction passes with nonzero lower bound, non-unit
  step and repeated outer entry. Admission negatives cover empty or unknown trip
  count, regenerated input, guarded consumer and an active producer pipe.
- Production down lit now requires early MAT readiness. A KV sibling-region lit
  regression also exercises physical key reuse and guarded first acquisition.
- Down and KV lower to C++ successfully. The lowering was run on one CPU affinity
  to bound LLVM's automatic thread pool, which reads that affinity.
- GEMM retains 200/394/782 pairs, zero named fences, one terminal ALL and all
  existing ordering/rearming checks. Its emitted plan is unchanged except for a
  trailing newline.
- All 87 original corpus modules plus Shenggan construct/reconstruct: 11
  projection outputs change, 77 remain unchanged (GEMM differs only in trailing
  whitespace). All six attention outputs and post-RMSNorm are byte-identical.
  Results are in `first-consumer-work/corpus.json`.
  The eight separately supplied A5 reference inputs are outside this A3 campaign;
  their layout/typed-contract import failures are recorded, not qualifications.

No device measurements of this candidate exist yet. Compare candidate, the pinned
495fb9cbd handoff and existing InsertSync on identical original inputs. Pipeline
latency and available overlap counters, not pair count, decide adoption. See
[the device task](oahs-first-consumer-device-task.md).
