# Share a required return before allocating private channels

2026-09-21, based on `21f95f9b7`. This is a construction and host-validation
result; no device latency result is claimed.

## Selected interface

For a closed producer cohort, keep separate X/Y readiness frontiers. If the
required X return follows Y's final reader phase, and X's next overwrite
precedes Y's in the same straight producer corridor, select X's existing return
as support for both cells. Do not allocate a private Y return or move X's wait.

The reader-region qualifier uses its existing physical-cell candidates,
`Control::straight`, canonical-word occurrences and `nearestRoles`. It does not
introduce another causal analysis. The new support query requires:

- One writer occurrence per cell, the same producer and reader engines, and a
  closed producer cohort under the existing global support restriction.
- One body return publication plus the original invocation prime, and one
  overwrite acquisition plus the original exit drain. Body publication words
  have unique analytical occurrences.
- Strictly ordered, distinct overwrite sites: X before Y.
- Every path to X's return crosses Y's final-reader boundary in that physical
  generation. A writer or invocation boundary kills this correspondence.

The selected return retains its original endpoints and records the union of
supported cells. Strict overwrite order prevents circular support; a surviving
earlier-deadline return may carry an already composed support set. Readiness
endpoints remain separate. Equal return frontiers continue to use the existing
exact-frontier grouping.

This selection happens before physical-key allocation. The one existing
mandatory proposal solve checks the exact words that will be committed. It must
prove all affected producer requirements, including zero-reader WAW, and all
event matching and consumption-before-republication obligations. Failure rejects
the optional proposal atomically. `repairFreeProducers` remains authoritative;
there is no hypothetical return credit, new omission trial, or weaker checker.

## Why the deadline matters

Y readiness is acquired before Y's final reader frontier. X's later actual
publication observes that receipt and Y's reader completion. Its acquisition
before the next X overwrite therefore carries Y-reader completion, Y's old
writer completion, and consumption of Y readiness. Since Y's overwrite and
next readiness publication follow X's overwrite, this supplies their actual
storage and event-reuse deadlines.

The private Y acquisition would supply an earlier subset of a receipt already
acquired at X. Removing that private transfer preserves the selected fragment's
completion interface. Publications are nonblocking: deleting the private SET
does not advance or retard a payload gate. No existing or outward channel is
reordered. Complete-plan equality is independently checked on the cases below;
this is not a general optimality claim for later ordinary construction.

If Y is overwritten first, using X's later reader prefix would require an
earlier X wait. The qualifier declines. A supplied mutation explicitly proves
that such a plan can be safe while adding payload ordering.

## Discriminating portable evidence

Both recurring omission and final helper trials are disabled in the positive
and deadline-negative construction comparisons.

- Positive: four private channels become three composed channels, retaining
  both readiness publications and acquisitions. Complete issue/finish relation
  sets are equal on empty, short, varying-length and repeated-entry traces.
- The actual trace contains Y-reader-finish → next-Y-overwrite and
  old-Y-writer-finish → next-Y-overwrite.
- Moving Y readiness consumption after X's publication, then erasing all
  payload effects, isolates missing rearming: balanced and acyclic event words
  fail consumption-before-republication. Both independent graph semantics and
  the production checker reject it.
- Reversed overwrite order keeps four channels. Forcing X's acquisition ahead
  of Y remains safe but adds four complete payload relations, including the
  forbidden last-X-read → earlier-Y-overwrite relation.
- With two forward keys and only one reverse key, composition constructs
  successfully. Private-return admission rejects the capacity requirement.
- Reload, independent reader, later Y reader, and unqualified possibly-empty
  child cases do not share. Removing either readiness or the supporting return
  fails both checkers.
- Three outward-publication cases place a real third-engine receipt before,
  between and after the reader frontiers. Equality includes the third engine's
  payloads, not just the producer/reader subgraph.

Twenty paired finite paths have equal complete payload-order sets, including
a three-input chain that composes two returns into one surviving owner. Finite
paths are regression evidence; mandatory whole-graph causal checking and the
qualified participation interface remain required.

## Native witness

`test/lit/pto/oahs_shared_reader_return.pto` has X/Y loads followed by X, Y,
then X reader children. Genuine vector barriers remain.

| Metric, two parent entries | Private returns | Shared return |
| --- | ---: | ---: |
| Payloads | 18 | 18 |
| Checked local conflicts | 165 | 165 |
| Complete payload relations | 602 | 602 |
| Executed SET/WAIT pairs | 10 | 7 |
| Reserved forward / reverse keys | 2 / 2 | 2 / 1 |
| Selected updates | 4 | 4 |
| Replay-site evaluations | 324 | 324 |
| Mandatory proposal evaluations | 72 | 72 |
| Final certificate evaluations | 68 | 68 |

There are zero added or removed payload relations. The constructor records one
new structural query, with 63 site visits. Recurring and helper trial counts are
both zero. Work in the original qualifier is already included in
`qualification_microseconds`; the new query and site counters identify the
additional propagation rather than hiding it in replay savings. As with the
existing qualification counters, these describe construction, not separate
native read-only admission probes.

Seven paired native variants all construct/reconstruct with equal complete
payload-order sets. Reverse deadlines, reload, a later Y reader and a statically
empty Y child retain their existing protocols. The one-visit case goes from six
to four executed pairs; four parent entries go from eighteen to thirteen. The
positive, one-visit and four-entry cases each reserve one fewer reverse key.

## Corpus and suites

All 23 portable suites and both native test executables pass. The final added
three-input chain also passes the focused cyclic suite. The new native FileCheck
and independent ordering regression pass; the diagnostic ablation reproduces
the frozen pre-change executable's witness output byte-for-byte.

The paired corpus completes 176 constructions across 88 modules / 97 functions
per arm. All 88 candidate plans are byte-identical to `21f95f9b7`, including its
new FIFO plans; no real corpus plan changes in this milestone.

| Aggregate constructor work, per arm | Before | After |
| --- | ---: | ---: |
| Selected updates | 1,782 | 1,782 |
| Replay-site evaluations | 3,234,019 | 3,234,019 |
| Mandatory proposal evaluations | 58,394 | 58,394 |
| Selected endpoints | 5,235 | 5,235 |
| Recurring omission trials | 65 | 65 |
| Final helper trials | 55 | 55 |

No corpus input reaches a new sharing propagation: both sharing-query and
sharing-site counters are zero. This does not mean the opportunity filters are
free. Phase and wall timings are retained in the raw records; the paired sweep
overlapped other local tests and is not a controlled compile-speed benchmark.
There is no compiler-speedup claim or new corpus device-timing task.

The diagnostic driver accepts `--no-reader-return-sharing` as a reproducible
ablation. This is not a new native algorithm mode: ordinary `algorithm=handoff`
uses composition. Checking and reconstruction cannot be disabled by this flag.

Reproduce the native ordering comparison with the diagnostic driver on the
fixture, with and without that flag, followed by:

```sh
python3 test/benchmarks/retained_children/check.py private.pto shared.pto
```

Local artifacts, frozen pre-change executable, commands and logs live in
`../return-sharing-work/`. The paired corpus compares the frozen executable
from `21f95f9b7` against this implementation, rather than older MAT/FIFO plans.
`corpus/work-summary.json` contains the totals; `native-variants/results.json`
records all seven paired native cases and their exact commands.

## Remaining scope

Multiple body return occurrences, ambiguous shared observation words,
independent reader engines, unqualified participation and producer reloads keep
the existing fallback. Global producer support is not weakened. General
first/last observation composition, moving-frontier certification and scoped
recurring-key reuse remain separate tasks. No new device campaign is implied
by a host event-count reduction.
