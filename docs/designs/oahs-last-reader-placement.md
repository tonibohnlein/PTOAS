# Release at the last read inside a child

Local implementation on `2cc458cbe`, 2026-09-21. The dispatched device campaign
remains on its frozen revision. No new device result is claimed here.

## Mechanism

For a retained input, the reader-region cycle previously published its return
at child exit. In this example that prefix unnecessarily includes the final
unrelated Q operation:

```text
P writes X
child loop:
    Q reads X
    Q does trailing work
P overwrites X on the next parent visit
```

The selected return can instead be published after the final participating read
of X, before the trailing work. Earlier Q operations remain in the ordinary
source prefix. The change removes only the final trailing operation from this
particular completion path; it does not make prefix completion selective.

The first-pass extension shares the existing physical cell effects, reader
regions, canonical command words and original-control graph:

1. Native admission finds an invariant cross-engine input with a last read
   followed by work on the same reader engine.
2. `refineLastVisit` exposes final/nonfinal observations at the selected
   original post-read anchors. It copies a straight body once analytically,
   shares payload identities and all other command words, and adds no runtime
   history. Nonfinal visits backedge; the final visit exits.
3. The reader-region qualifier uses nearest-role propagation to establish that
   no further read lies between a candidate source and child exit. A balanced
   entry-to-source query proves exactly one publication opportunity per entry.
4. Only the last reader child of the physical generation uses this release.
   The complete readiness/return proposal still passes staged protocol and
   resource admission. Actual selected transfers establish completion and
   consumption knowledge; the observation supplies neither.

The native guard is `upper - iv <= 1`, using existing `LoopHasNext` emission
and independent read-back. The first native scope requires index-typed,
nonnegative constant lower bounds, a larger constant upper bound, unit step,
a straight leaf child inside a parent loop, and an inactive producer engine.
Those bounds make the signed subtraction representable. Unknown/empty bounds,
non-unit steps, guarded reads, regeneration and existing refinements retain
their previous handling. This is a general control/storage qualification, with
no kernel or opcode-sequence recognizer and no new constructor mode.

The original child entry/exit remain lifecycle boundaries. A separate
publication field preserves the earlier physical completion boundary; moving
the publication does not change the generation or the next overwrite deadline.

## Validation

- Seven portable traces cover zero parent entries, one/many child visits and
  varying lengths on successive entries. Complete payload order loses 52
  relations, adds none, and the final trailing read no longer gates the next
  producer write. The exit-placement baseline fails that forbidden-edge test.
- Both checkers reject missing readiness/return support and a mutation that
  publishes on every body visit. A later read of the same input retains the
  exit fallback. Unknown/empty metadata, guarded bodies and repeated refinement
  are rejected. The trace checker also verifies final/nonfinal atoms against
  independently counted remaining physical reads.
- Nine native admission variants cover the positive case, a one-trip child,
  empty/unknown bounds, non-unit step, a guarded reader, regeneration, a later
  read of the same input and negative lower bounds. Positive variants construct
  and reconstruct. The FileCheck fixture pins the guarded SET before trailing
  vector work, with its genuine vector barriers retained.
- The native fixture has 14 payloads and 127 checked local conflicts. Compared
  with a separately built control omitting only the new native qualification
  call, full order shrinks **378 to 374**, with **four removed and zero added**
  relations. Executed event pairs remain **five**. The control is a development
  variant, not an additional production mode or a device measurement.
- All 23 portable suites, both native diagnostic executables and four
  native/FileCheck fixtures pass. All 88 corpus modules (97 functions)
  construct/reconstruct with byte-identical plans to `2cc458cbe`.

## Cost and limits

The witness grows from 22 to 25 analysis sites, selected updates from two to
three, and replay evaluations from 89 to 130. Proposal and final checks each
remain at 34 evaluations. There are no recurring-omission or final-helper
trials. The larger graph and split command words have a real construction cost;
this is a placement improvement, not a compiler-speed improvement.

Native discovery scans candidate bodies and their physical effects. Qualified
bodies add one analytical copy, without a product of unrelated loop modes.
Canonical-anchor uniqueness currently uses a graph scan per selected anchor.
The core's extra nearest-role and balance traversals run only when a final-visit
candidate exists. Ordinary readers without that vocabulary do not run those
extra traversals. Existing work counters include qualification, graph growth,
proposal checking, replay and final checking; they do not separately count each
new query's individual operations.

The generated guards can have runtime cost. Fewer ordering relations with equal
event counts do not prove a device speedup. Combining final-visit observations
with first-consumer/bank refinements, non-unit steps and guarded participation
is future work. The general frontier-motion certificate remains separate.

Local artifacts: `../last-reader-work/` relative to the repository root.
See `HANDOFF.md` for final integration results.
