# Exact periodic publication succession: query checkpoint

This checkpoint supplies the exact successor query needed by event-reuse
analysis. It does not yet change native construction, allocation or emitted
reconstruction, and does not complete ordinary buffering or its compile-cost
gate. No device result is attributed to this change.

The admitted population is one common finite integer interval with shared
parameter guards, fixed enclosing coordinates and publication atoms carrying
phase, actual within-iteration rank and residue. Parameter-only existential
constraints are retained. Nonunit intervals, iteration-dependent witnesses,
variable enclosing coordinates and coefficients outside int64 are unsupported.
An empty common set returns empty before qualifying unused atoms.

Sort by residue and schedule rank, remove duplicate atoms, and connect cyclic
neighbours. Intersect each endpoint with the original interval: clipped or
empty executions gain no terminal successor. Complexity is O(K log K) plus
O(K * template size) output, independent of period magnitude. Balanced-map
qualification, sorting, copies and output are charged. Integer arithmetic
remains exact, including negation of INT64_MIN phase IDs.

The caller must still establish that these atoms describe ALL actual selected
publications and that ranks, parameters and invocation coordinates have their
original meaning. The returned relation supplies a reuse obligation; it does
not establish completion or safe event reuse. Fresh emitted events require
independent population qualification. Unsupported shapes retain the general
exact query when this helper is integrated.

## Validation

The final native relation suite passed **255 checks**, including 72 new
periodic-query cases and their references. Small cases compare full results
with isl schedule lexmin and the existing native firstTargets query. Negative
bounds, symbolic and contradictory guards, parameter-only existential locals,
multiple residues, duplicates, tied ranks, clipped/singleton/empty domains,
wide coefficients and budget exhaustion without partial output are covered.

At 8/32/128/512 publication atoms, both period 1 and period 2147483647 produce
8/32/128/512 nonempty pieces and consume 598/2714/12234/54538 work units.
Every scaling piece is checked against its expected exact relation, including
the wrap edge; the reference check does not build a quadratic union comparison.
These are query-population bounds, not end-to-end compilation measurements.

The initial general-native INT64-period comparison timed out at 45 seconds.
That failure is retained in relations-2; the compact result passes the exact
isl comparison. The automatic suite runs the general-native comparison only
on the small-period semantic population. Earlier test-only failures and their
fixes remain in the numbered campaign directories.

Build: cmake --build insertsync-builds/PTOAS-oahs-clean --parallel 2
--target pto-logical-relations-test (from the parent workspace).
The serial check uses check_relations.py with the built driver and the pinned
/usr/lib/gcc/x86_64-redhat-linux/15/libisl.so.23 runtime.
Artifacts: insertsync-builds/campaign/logical-plan/overnight-20260910/relations-7
in the parent workspace. This is a worktree build based on 08a60761a.
Native driver SHA-256: 1150d429bc39b691f17b82f520af07d063175b668a915d89b7a74eb3e79cc581
