# Shared requirements and guarded publication cuts

This change integrates exact local-storage requirements into InsertSync and uses
them to repair an unnecessarily broad QK publication. It is enabled by the
existing experimental `--insert-sync-buffer-generations` option. The unchanged
Qwen QK kernel is the effectiveness case; no kernel name, shape or particular
instruction sequence is tested by the compiler.

## Shared input and resolution

`LocalStorageRequirements` snapshots the translated access records by value,
before lifecycle selection or ordinary dependency traversal. It retains exact
physical slices, all participating readers/writers, hazard kinds, original
operation/access identities, guarded control occurrences and the existing
storage-flow result. A requirement applies to all feasible ordered occurrences
of its source and target; it never guesses a distance-one relationship between
nested loop invocations. Unknown overlap or correspondence remains conservative.

A projection is qualified only if every overlapping participant names the same
exact slice. Partial, unknown and multi-address access groups retain the existing
legacy path. This is a boundary on the migrated query, not an admission condition
for production compilation.

`ResolveLocalRequirements()` is the decision path for qualified groups. It asks
what order is required, then resolves it using the current traversal supply,
selected lifecycle relationships, or the existing qualified intrinsic rules.
Only the remaining obligations request direct repair. Those groups return before
the legacy sequence of independent decisions; both paths cannot own the decision.
MMAD ordering remains an ACC-specific rule and does not establish M completion.

Lifecycle construction reads the same slice projections instead of independently
rescanning physical uses/definitions for each candidate. Its complete generation
and token qualification remains separate from the input requirements. Failed
one-shot or recurring recognition does not discard requirements: in particular,
both QK preload requirements remain available even though their panels receive
no primed ready/free lifecycle.

This is not a claim that all generation analyses have become one cached result.
The existing qualified generation engine still checks member bundles, and dynamic
slot/GM cases retain their established queries. The migration replaces exact
local requirement discovery and resolution, not every semantic adapter at once.

## QK's actual boundary and the empty path

The previous revised output has this source cut:

```text
load Q0
load Q1
publish Q_ready
acquire Q_ready
for block:
    load K
    publish K_ready
    extract Q0       // unnecessarily acquires completion of Q1
    acquire K_ready
    extract K
    ...
    extract Q1
```

The first Q0 extract requires Q0. Later Q1 extracts still require Q1, but the
existing K readiness stream already supplies its completion to those operations.
The publication cannot simply move earlier: when the loop is empty, the old
publication/acquisition also retires Q1 before return.

The accepted complete replacement uses the same concrete event key and wait:

```text
load Q0
if lower < upper: publish Q_ready
load Q1
if lower >= upper: publish Q_ready
acquire Q_ready
for block: ...
```

Exactly one publication executes on each original execution. No wait is repeated
per loop iteration, no additional key is needed, and the empty path keeps its
original source cut. The conditions use available loop bounds without arithmetic
that can overflow. Skipped readers are not inferred from nonempty loop bounds:
if a nonempty loop can skip all useful handoffs, the trial must prove that path
as well or retain the original publication.

## Planning and realization checks

The placement query consumes the actual combined emitted plan, with both direct
and recurring actions. Only explicitly owned residual publications are editable;
selected protocol actions stay indivisible. It considers earlier producer
boundaries from the shared requirements and can qualify an existing loop's
nonempty/empty alternatives. The work is deterministic and bounded (at most 32
trials and eight producer boundaries per publication, within the existing budget).

Every trial must preserve all affected shared requirements and independently
recovered conservative conflicts, including GM-facing effects. It reconstructs
the actual guards/events, proves their consumption/rearm behavior, and checks
retirement at reachable exits. It cannot use the old publication as its own
completion witness. It must also remove a concrete cross-lane completion
requirement at a payload operation; merely changing textual position is not a win.

The normal rewrite crosses no synchronization, region boundary or other lane's
physical operation. On each execution its source cut is earlier or identical,
with the same acquisition and event participation. It therefore introduces no
additional payload-ordering edge in the supported completion model. Allocation
has already fixed the keys and no scarcity recovery is added by this rewrite.
Actual device scheduling and instruction cost still need measurement.

Payload operations, attributes, types and SSA wiring are preserved. The final
lifecycle clone is verified after this placement. The older generic event mover
does not subsequently own these publications when buffer-generation mode is
active; residual barrier refinement and the qualified cleanup remain active.

## Q projection is a different lowering problem

Q projection still selects four L1 slots and six separate L0 slots. Each RIGHT
slot is used twice per outer iteration, with different LEFT slots in the two
halves. Each individual LEFT slot is used once. The native ready/free streams
therefore have different publication/consumption occurrences.

For example, the RIGHT slot at address zero is paired with LEFT at 6144 in the
first half and LEFT at 2048 in the second. The first matrix reader must release
RIGHT before its second extraction. Merging all those physical members into one
whole bundle would require data that has not yet been produced at the first
consumer. LEFT readiness streams with different consumers also cannot be merged
by simply publishing after the later extraction: that recaptures independent
work at the earlier consumer.

The implemented lowering instead replaces only the complete redundant LEFT
readiness streams. The retained RIGHT readiness publications already include
those LEFT productions, and their acquisitions precede the same matrix consumers.
The LEFT release streams remain separate because their reuse occurrences differ.
This removes four logical readiness streams, containing six static sets and six
static waits: **45/45 becomes 39/39**, with the same barriers and scalar control.

The decision consumes the combined concrete plan and both the immutable local
obligations and independently extracted memory requirements. Removing a stream
must leave every requirement sourced on its physical producer lane supplied by
strong full-phase completion. It must preserve every represented payload
completion prefix, event consumption/rearm, and exit retirement. This is a
conservative sufficient check; a stream with only generation-specific alternate
supply stays unchanged when full-phase completion cannot establish the claim.

All publications and acquisitions of the selected readiness key are replaced
as a unit. Priming, release and drain actions of the separate Free stream remain.
After simulated removal succeeds, the actual actions are detached transactionally;
fresh translation and graph reconstruction must reproduce the proof, guarded
payload identities and unchanged completion prefixes before the actions are
destroyed. Failure restores all original actions in order. Concrete IDs and
remaining endpoints do not change, and no later allocator broadens them.

For 32 outer iterations, executed pairs fall from 779 to 651, with 435 executed scalar
operations in either revised output. Original InsertSync still executes 389 pairs
(and has 19/19 static sites). This is a reduction of the revised constructor's
command overhead, not parity with the original pass or a measured speedup.

## Validation and limits

The native regression uses the unchanged QK input, a renamed function, an added
reader, physical overlap, equivalent guarded forms that can skip all readers,
and two enclosing invocations. Concrete replay includes negative/zero/positive
trip counts and checks actual token participation and the required Q0/Q1 cuts.
Broken empty-path and duplicate publications must fail that independent replay.
Q-projection tests check physical LEFT/RIGHT production and reclamation against
actual completion prefixes for zero, one, two and 32 iterations. Removing both
ends of the remaining readiness streams keeps counts balanced but must fail
that independent operand-readiness check.

The benchmark projection separately accounts for synchronization-only scalar
control. It still compares every payload operation, allocation expression, view,
ABI and payload guard. Mutation tests ensure a changed payload guard cannot be
hidden as synchronization control.

See [the recorded native results](../../test/experiments/insert_sync/performance/SHARED_PUBLICATIONS_RESULTS.md)
for inventories, concrete executed mechanisms, scalar overhead and compile work.
These host checks are not device numerical correctness or wall-time measurements.
