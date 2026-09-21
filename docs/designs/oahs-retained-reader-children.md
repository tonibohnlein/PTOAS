# Retained input across reader children

Update: the subsequent [multi-input cohort extension](oahs-retained-producer-cohort.md)
now qualifies a restricted complete-producer case. The single-cell limit below
describes this initial milestone and its counterexample.

Local follow-up to `d6e9b5365`, 2026-09-20. No device result is claimed.

## First-pass fact and construction decision

The existing reader-region qualifier previously proposed one readiness
acquisition and one release per leaf child. With one producer followed by two
children reading the same generation, the resulting one-publication/two-wait
proposal failed participation and ordinary construction took over.

The qualifier now composes those existing region records using the same
nearest-role propagation as guarded bank episodes. Each record retains its
original entry, qualified acquisition and exit. Writers separate generations;
reader boundaries preserve them. Native sibling exit/entry anchors may coincide,
so the query includes a neighboring role at the queried cut itself.

This is an immutable view over original effects, control and first-consumer
facts. It supplies endpoint candidates, not completion, key assignments or a
second causal state. The ordinary proposal balance, resource admission, full
causal replay, final check and native reconstruction still apply.

```text
load X
publish readiness
acquire readiness at first reader child
child A reads X
child B reads the same X
publish release after child B
unrelated work on the reader engine
...
acquire release before next overwrite of X
```

A real reload between children starts another readiness/release episode.
Independent reader engines are not collapsed into one final-reader guarantee.
Mixed first/retained or last/non-last paths decline unless the existing original
frontiers suffice; no runtime history counter or new guard is introduced.

## An ordering limit discovered during implementation

A safe retained-input cycle is not enough to promise globally narrower ordering.
The initial probe included `P: write X; P: write Y`, with only X read by the two
children. Ordinary construction fenced before X, also covering old Y. Completing
the X cycle removed that repair, leaving a Y fence after the new X write. That
fence added `finish(new X) -> issue(new Y)` (two payload-vertex relations in the
one-episode witness). Memory and event safety alone would accept this tradeoff.

The new cross-child admission therefore requires a single producer write cell
and excludes other exclusive producer accesses. Read-only nonexclusive sources,
including the GM input, remain allowed. Existing separate-generation reader-region
cycles retain their admission. This is a deliberate initial restriction, not a
claim that multi-input retained kernels are solved. A future extension needs to
account for the placement of the other actual repairs, not simply remove this
check or move their fences speculatively.

## Validation

- The linked portable constructor selects two logical roles directly, without
  recurring omission trials. Eight traces cover empty/short children and varying
  lengths on successive parent entries. Independent full payload reachability
  comparisons against ordinary construction remove 62 relations and add none.
- Moving the return to the first child's exit is rejected by both causal replay
  and the independent graph oracle. Removing either complete supporting channel
  is rejected. Reload and independent-reader cases retain their real obligations.
  The uncovered producer-write counterexample declines this specialization.
- `test/lit/pto/oahs_retained_reader_children.pto` reproduces the miss through
  native A3 import. Readiness now serves both children; release precedes the
  unrelated final V operation. Static SET/WAIT counts fall from 5/5 to 3/3;
  three V barriers remain, while the MTE2 repair is discharged by the real cycle.
  Construction, reconstruction and the pass/FileCheck regression pass.
- All 23 portable suites and the native diagnostic suite pass.
- All 88 corpus modules construct/reconstruct successfully and remain byte-identical
  to the committed sibling-replay baseline, including GEMM, attention, projections
  and post-RMSNorm. No speedup is claimed for those unchanged plans.

The 62-relation figure belongs to the portable traces, not a native timing or
whole-corpus ordering comparison. Native plan differences are checked separately.
Artifacts, including the before/after plans and serial corpus reproducer, are in
`/home/toni/work/pypto3_sync_more/retained-generation-work/`.

## Work and next step

The role propagation adds two constant-height graph traversals per eligible cell,
O(N+E) each, beyond existing cell/region incidence qualification. It creates no
new observation sites, branch products or repeated complete-plan searches.
Qualification time remains in the existing qualification counter; selected replay
and final checks retain their separate counters. This is not a global linear-time
claim for all qualification work.

Next: find a real retained-input benchmark admitted by this scope, or certify the
other producer repairs needed to extend it to multiple write cells. Keep the
reload and fence-motion counterexamples as admission tests. Do not launch another
broad device timing campaign for unchanged corpus outputs.
