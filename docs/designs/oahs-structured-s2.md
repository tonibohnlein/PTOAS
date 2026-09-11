# OAHS S2: first/last loop-boundary transfers

## Status and base

This is a candidate follow-up to the exact delivered S1 package, itself based
on PTOAS `44b311519cb2194215700566f4a7de75f3d151cf`. It extends the existing
`planner=structured` implementation on `codex/oahs-upstream`. It does not add a
pass, change the default, invoke a legacy seed, or call the relational planner.

During preparation, S1 landed as `d2107dd1b7e7b7a09db0746cf7bfcdd6b2a8f2cd`.
All existing files touched by S2 and its interface prerequisites match that
commit byte-for-byte. S2 therefore applies directly on that source as well as
on the exact delivered S1 payload. The integration commit's reported S1 native
results are separate evidence, not S2 validation.

The extended portable production core is tested. S2 native adapter compilation,
unchanged-QK construction, native mutations, device execution and compile-cost
acceptance remain to be performed on the pinned LLVM/MLIR 19.1.7 build.

## S2's exact boundary

The admitted schedule is:

```
exactly-once prelude
one zero-based, unit-step periodic loop
exactly-once epilogue
```

The loop may execute zero times. Prelude and epilogue payloads still execute in
that case. Boundary payloads must be in the function's original block; a loop
with boundary payloads must be a direct, unconditional child of that block.
S1's periodic body, modulo/mask qualification and physical alias contracts are
unchanged. There is no loop unrolling, Presburger relation, predicate synthesis,
backtracking allocator or analysis-work quota in the added boundary algorithm.

Not yet supported: arbitrary nested/sequential loop composition, dynamic
skipped-reader predicates outside S1's periodic fragment, memory-valued loop
forwarding, cross-core protocols, macros/queues, dynamic event keys, or an
unavailable bound at an early publication. An unsupported bound does not cause
the publication to be moved later. UnitFlag/MMAD elision and retirement policy
are unchanged. This is not the complete nested-GEMM milestone.

## Representation

`Atom::segment` identifies Prelude, Body or Epilogue. Body occurrences remain
`t = D*k + r`, clipped by `0 <= t < N`. A boundary atom occurs exactly once;
its identity is not an invented recurring epoch.

`Requirement` stays immutable. Body/body requirements retain S1 distances.
Boundary requirement endpoints determine Once-to-First, Last-to-Once or
Once-to-Once correspondence. Boundary distances are zero only as an encoding
convention, not a claim that a final reader is in iteration zero.

`Action::participation` distinguishes:

| Kind | Meaning |
| --- | --- |
| Every | S1 recurring participation, or an unconditional one-shot endpoint |
| First | Execute at the first occurrence of this body atom |
| Last | Execute at the last occurrence of this body atom |
| IfBody | Execute outside the loop iff the counterpart residue occurs |

No new source-IR annotations are required. Emission uses these internal facts;
reconstruction recovers their meaning from actual operands and predicates.

## Construction

1. Select the recurring body's logical plan with S1's existing staircase and
   residual same-pipe repair. This selection is split from allocation, not
   copied into a new solver.
2. Build a first-occurrence view. It includes the prelude and each body's first
   occurrence in execution order. Seed completion from actual body barriers
   and distance-zero body handoffs. Distance-one handoffs cannot supply the
   first iteration.
3. Apply the same latest-source/prefix-advance rule to prelude and first-reader
   requirements. A first acquisition remains known on its destination lane;
   repeated readers do not consume the same one-bit notification repeatedly.
4. Add first-only same-pipe barriers for missing entry requirements. Do not
   execute them every iteration merely because the target is in a loop.
5. For epilogue requirements, construct last-occurrence views qualified for all
   possible trip counts. Retain incomparable last-reader residues and lanes.
   Add last-reader handoffs or residual epilogue barriers only where needed.
6. Assign boundary and recurring resources from the same target pool, then
   emit and freshly verify the complete result.

A QK-style early publication remains after the first Q preload. It is not
moved behind the second Q preload or the first K load merely to cross a loop
boundary. A prior acquisition already required by the body may supply another
boundary requirement; no extra broader acquisition is introduced for that.

## Why the first/last views are valid

### First occurrences

If the first occurrence of body atom q executes, all earlier first occurrences
on the same admitted periodic schedule execute as well. Thus every first-view
path used to cover a requirement at q has its intermediate occurrences. An
entry acquisition's knowledge persists to later q occurrences because a wait
orders subsequent issue on its destination lane.

A periodic body edge with positive epoch distance is deliberately absent from
this view: its source does not exist at the first target occurrence.

### Last occurrences

For N >= 0 write `N = D*K + s`, with `0 <= s < D`.

* K=0: only atoms with `r < s` occur; their last occurrence is also their first.
* K>=1: every atom occurs. Its last epoch is K when `r < s`, otherwise K-1.

Membership and relative last-epoch order change only when s crosses a
represented residue. The implementation checks these intervals, not every
integer s and not a finite selection of K values. The K>=1 calculation is
symbolic: K cancels from every relative-last comparison. Native N<=0 is the
empty-body case.

A periodic handoff `p[k-d] -> q[k]` supplies **last p** at **last q** only when
`lastEpoch(q)-lastEpoch(p) >= d`. Otherwise no such last-view completion edge is
added. The existing two-layer Cuts machinery continues to require a real
completion mechanism; issue order alone is not a completion proof.

A first-only barrier contributes persistent PRELUDE knowledge to a last view.
It must not complete body work issued after that first barrier. This distinction
is explicit in `BoundaryCuts::barrier()`.

A Last-to-Once publication follows the actual final reader, and its acquired
prefix includes that lane's preceding accesses. Different consumer lanes are
not collapsed into a single static "last reader".

### Zero trips

First/last pairs have matching existence conditions. No token is published for
a nonexistent body counterpart. A prelude-to-epilogue dependency is checked on
the empty path, independently of all body handoffs. If needed it receives its
own direct handoff or barrier. The terminal PIPE_ALL remains separate and does
not consume event notifications.

## Direct lowering

For a body atom at residue r and period D:

```
Prelude -> First:
    set after prelude producer if N > r
    wait before body reader if iv == r

Last -> Epilogue:
    set after body reader if N - iv <= D
    wait before epilogue consumer if N > r
```

An explicit per-residue guard is retained when one original operation has
multiple represented atoms. Inside the admitted loop, `0 <= iv < N` makes
`N-iv` representable in signed 64-bit index arithmetic. The emitter does not
form potentially overflowing `iv+D`. Prelude guard operands must properly
dominate the selected publication; this is checked before mutation.

The native reconstructor recognizes those exact forms, checks the original IV,
upper-bound identity, period, residue, actual source/destination cuts and action
ordering. It rejects changed first/last/existence predicates. It also compares
recovered boundaries with selected boundaries, so a safe but later publication
is not accepted as an equivalent lowering.

## Resource policy and its limitation

A boundary key owns exactly one matched one-shot pair in a directed domain.
It is disjoint from recurring keys. The allocator reserves those actual keys
before running the existing recurring assignment on the remaining pool. Both
populations are selected before this resource decision.

This intentionally does NOT implement boundary/recurring key sharing. A
failure may reflect this conservative assignment policy, not hardware
infeasibility. There is no acknowledgement, boundary widening, body drain or
silent fallback to force a fit. The default target/reservation profile is S1's.

## Complexity and integration

The added analysis is structurally terminating, not quota-terminated. Let A be
the number of represented atoms, R the number of distinct body residues, and H
the selected handoff count. There are at most `2*(R+1)` last views; none is built
when there is no epilogue. Each uses S1's finite dense closure (O(A^3)) and
incremental edge updates (O(A^2)). Retained-view memory is O(R*A^2). This is a
polynomial first implementation, not a claim of optimal asymptotics or a passed
native <=2x timing gate. Large structural populations still warrant profiling.

The changed production files are the existing core header, core implementation
and native adapter. Existing core/native test targets are extended. No new
planner switch, library, dialect, solver, test executable or build target is
introduced.

## Acceptance

Keep the unchanged one/two/three-buffer CLI gate. Add the **unchanged QK input**
from the existing hash-checked population. Require its first Q-panel extraction
to acquire MTE2 only through the first preload, not through the second preload
or the first K load. Do not replace it with the smaller examples below if it
fails. Record the actual blocker and keep S2 native acceptance open.

The additional native boundary fixtures exercise multiple preloads, periodic
readers, final release, zero/negative/short/long trips, mixed boundary/recurring
keys and equivalent signed/unsigned remainders. Mutations change actual first,
last and existence guards, move a guarded publication, remove endpoints and
retirement, or corrupt keys. Original payload/ABI/views/allocations are compared
and failure must leave the original function untouched.

The portable tests also contain a **hand-transcribed QK-shaped storage model**.
It is NOT native extraction or a native QK compilation. Its finite asynchronous
oracle separates issue/completion and set enqueue/fire, checks every older
conflict, progress and causal key reuse. Finite tests challenge the symbolic
argument above; they do not replace it or constitute device evidence.

After native correctness, run the existing repeated paired compilation campaign
with identical settings. Report sets, waits, barriers, terminal drain, scalar
work, key pressure, changed readiness and release cuts, and compilation time
separately. Device qualification remains independent.

## Next scope

S3 should generalize the same boundary transfers through nested invocations and
supported nonperiodic choices, retaining incoming definitions and outstanding
reader sets. Do not restart the planner or reinstate Presburger. Qualify S1/S2
first: this document does not authorize weakening a failed native test.
