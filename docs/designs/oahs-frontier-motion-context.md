# Common-frontier motion: the outward-publication test

Local work on `d6e9b5365`, 2026-09-20. This is a supplied-protocol regression
and a refinement of the certificate requirements. It does not establish that
the current native constructor emits the broader plan. No production placement
rule or dispatched device snapshot changes in this step.

## The distinction to preserve

One episode has these original payloads, on disjoint A, B and z storage:

```text
Q writes z
P loads A
P loads B
Q reads A and B
R reads z
```

Separate readiness can use the following command word before Q's combined read:

```text
WAIT A-ready       // source immediately after P loads A
SET z-ready        // outward Q -> R publication
WAIT B-ready       // source immediately after P loads B
Q reads A and B
```

The outward publication carries Q's z writer and the A receipt. It need not
carry completion of the later B load. Replacing both readiness roles by one
publication after B and one acquisition at this same cut yields:

```text
WAIT AB-ready
SET z-ready
Q reads A and B
```

R's z reader now waits for B. The common consumer cut, participation and memory
safety all match, but the outward publication's meaning has changed. Distinct
positions within that command word matter.

Move the outward SET after both original waits and this particular obstruction
disappears: the separate and merged protocols have identical complete payload
ordering in the checked cases. That positive case is deliberately paired with
the negative one; it is not a general contextual theorem.

## Executable evidence

`test/oahs/selected_placement_test.cpp::commonFrontierContext()` constructs both
protocols explicitly. The independent `GraphOracle.h` now optionally exports
the complete strict launch/finish relation, rather than only checking selected
forbidden edges or comparing relation counts. The new tests compare the sets.
Desired memory edges are still never inserted into that graph.

Repeated episodes reuse the same storage and keys. Actual Q-reader -> P returns
protect A/B overwrites and forward-key reuse. Actual R-reader -> Q returns
protect z overwrites and z-readiness reuse. Initial uses need no fabricated
completion; final releases are explicitly drained.

| Episodes | Outward SET after both waits: separate / merged relations | Outward SET between waits: separate / merged relations |
| ---: | ---: | ---: |
| 1 | 29 / 29 | 25 / 29 |
| 2 | 148 / 148 | 140 / 148 |
| 4 | 686 / 686 | 670 / 686 |

All twelve supplied plans pass both the production causal checker and the
independent memory, balance, acyclicity and rearming checks. With the SET after
both waits, the complete payload relations are equal. With it between waits,
the merged relation is a strict superset, adding B completion -> R issue in
every episode (four additional launch/finish relations per episode).

For two/four episodes, removing the complete Q -> P return channel keeps token
counts balanced but fails both memory ordering and rearming. The production
checker also rejects it. This guards against declaring the positive composition
valid by silently assuming a future return or ignoring physical-key reuse.

These are small ordinary-core protocol fixtures, not native compiler output,
queue/cross-core qualification, device timings or proofs for arbitrary loops.
Local logs: `/home/toni/work/pypto3_sync_more/frontier-motion-work/`.

## Linked constructor probe

A separate small probe calls the actual periodic qualifier and constructor on
an A/B common-reader cycle plus a Q -> R z transfer, using two cell-number
arrangements. Both arrangements construct successfully with frontier motion
on/off. Before omission trials, no-motion produces six roles and motion four;
the final selected population is four channels in both modes.

Crucially, the qualifier lists the Q -> R role before the two P -> Q readiness
roles. Binding therefore does not place that export between the two readiness
acquisitions as the supplied counterexample does. This probe **does not reproduce
the ordering defect**. Role counts are not an ordering proof, but the inspected
order explains why a blanket ban on every outward publication at a common cut
would be too coarse. Retain the distinction between before/between/after the
actual endpoint positions. Source and output are `probe.cpp` and `probe.log` in
the artifact directory.

The rebuilt portable suite passes **23/23 tests**. No native corpus regeneration
or device timing is needed for these test-only and documentation changes.

## Consequence for the next production certificate

`CyclicFrontiers.cpp` currently qualifies a generic merge using matching
occurrence descriptors, one common frontier and `Control::straight` ordering of
the other frontier. Those facts do not describe intervening command-word exports.
The current qualifier runs before physical endpoint binding, so it cannot simply
read an already complete selected command word as its certificate.

The next bounded production step must establish:

1. An actual proposal whose separate roles expose this placement issue, including
   their insertion order and occurrence mapping. This regression does not prove
   that the present constructor orders its endpoints like the supplied witness.
2. The exact gap and exported prefix for every outward publication affected by
   a proposed move, including other required roles installed in that word.
3. Both neighboring physical-key uses and the consumption evidence supporting
   the selected bindings. The six finite positive cases do not supply that
   general certificate.
4. Preservation under subsequent endpoint insertion, or invalidation/rejection
   of the certificate when a later edit introduces a relevant export.

Use the existing requirement/source/word views for these facts. Do not introduce
another completion authority, assume that all SETs can precede all WAITs, or
substitute a safety-only changed-plan check for an ordering certificate. Until
this is proved, keep no-motion grouping available as the conservative control.
