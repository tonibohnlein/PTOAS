# Lifetime hardening after e74a01f2

Base: `e74a01f27b82e03fbb46188084616bcc9a8f32fa`.
This change adds no hardware premise, changes no original operation or address,
and does not change the InsertSync default. Whole-compilation timing is
reported as telemetry; there is no fixed ratio acceptance threshold.

## Two views of an operation

`Node::effects` remains the authoritative scheduling-obligation population.
It retains the current ACC read/read resource exclusion. `Node::byteEffects`
is an optional, lowering-derived producer/reader projection used only by
lifetime discovery. An absent projection conservatively uses `effects`.

For example, a FIX read of ACC has a real read and a writer-like exclusion
obligation. It is a reader for storage-lifetime discovery, but its original
exclusion still participates in construction and verification. Nothing clears
an ACC conflict solely because the discovery projection calls it a read.

The native GEMM exposed that the current open constructor cannot yet combine
that stronger exclusion with the newly visible ACC lifetime. Such summaries
are therefore discovered but are not eligible for persistent replacement.
They remain on the verified ordinary/typed path, while unrelated storage
lifetimes are still selected. This avoids a whole-plan fallback and makes the
remaining composition work explicit.

The native adapter derives both views from the same imported memory effects.
Projection validation rejects invented writes, changed reads, or a discarded
write not justified by the existing read/exclusion representation. All original
read/write and exclusion demands remain in fresh open/closed verification.
Exact validated UnitFlag credit keeps the previous discharge boundary; its
already-handled ACC cell is not given a redundant event-lifetime proposal.

This is an incremental representation split, not a replacement of every old
writer-like resource encoding with a new general resource algebra.

## Bounded indexing

`LifetimeSummaryIndex` replaces the two linear signature searches and the
all-summary producer-envelope pair scan. Exact signatures retain lifetime,
entry/exit, lanes, frontiers, and participation flags. Producer envelopes are
considered only within equal lifetime/reader/parent signatures.

Signature construction, lookup/insertion work reservations, and witness copies
consume the discovery allowance before execution. Exhaustion discards the
optional result; partially merged populations are not accepted. Complexity of
the new index is O(S log S) for S fixed-width summaries plus witness volume.
The implementation reserves conservative ordered-map work; it does not claim
that every other existing compiler stage has been made linear or budgeted.

## UnitFlag physical sections

Ownership validation traverses exactly `inventory.physical.lifetimeScope`
when that already-qualified scope is a one-region operation distinct from the
function. This admits a qualified cube section without making unknown region
operations transparent. Whole-function semantic import and surroundings audits
remain unchanged. Ownership validation still runs when credit is disabled.

Producer and store transitions must also agree on the exact ACC tile geometry,
not merely base address and byte extent. The currently qualified store layout is
checked explicitly. Equal-byte but differently shaped ownership transitions are
rejected.

When a counted loop is the final payload operation in the physical scope, its
guarded loop-exit acquisitions bind to the synthetic terminal cut represented by
the generated retirement barrier. Guard recovery and fresh reconstruction both
recognize this anchor without adding it to the original-operation population.

## Coexistence and rollback

The supplied restricted-coexistence proposal started from the verified current
plan. Native qualification showed that pair-wide replacement could leave
residual physical demands uncovered, including the early-vector and shared-ring
regressions. That proposal is therefore not selected here.

Persistent lifetimes currently retain guarded, deferred, child-return, and
structured-episode plans unchanged. Eligible plans build one conservative
all-`Every` residual transaction; if its complete open-protocol check fails,
the previously verified precise plan remains the result. Joint guarded/open
inference and general pre-allocation family selection remain future work.

Pair-wide removal remains limited to whole-program lifetime summaries. Local
families remove only the fallback at their verified cuts. Residual
requirements, all event lifetimes, progress and rearm must still pass the
existing complete open check. Allocation failure restores commands,
stripped-pair state and occupied keys. There are at most eight family trials,
and snapshots are charged beforehand.

## Selected statistics

Prior-stage cycle/key counts are retained under `preLifetime*` fields, not
fabricated from removed instructions. A selected persistent plan is recounted
from actual mechanisms: SET/WAIT sites, named barriers and directed event keys.
`selectedCountsValid` distinguishes these measurements from untouched baseline
paths. Surviving closed cycles are inferred from actual commands. If that
accounting cannot be established, `selectedCycleCountKnown` is false rather
than substituting a removed-command count. This accounting failure does not
invalidate a separately proved open protocol.

## Tests and evidence boundaries

The standalone C++ test compiles the actual new production index/projection
headers. It covers exact and incompatible signatures, randomized aggregation,
summary scaling, bounded rollback and ACC access-role/exclusion separation.
These tests do not compile the complete composition constructor or MLIR adapter.

`lifetime_revision_test.inc`, included by `composition_core_test.cpp`, exercises
actual lifetime discovery/construction and corruption checks once built in the
native tree. `check_lifetime_revision_native.py` tests top-level and
section-wrapped UnitFlag with credit on/off, missing return and malformed
contracts. A lit fixture covers the public pass on a section. Those native and
complete-core tests were supplied but not executed in the patch-preparation
environment.

No new historical-GEMM synchronization count, compiler ratio, numerical result,
or device speedup is claimed. Re-run the exact prepared/ABI-qualified GEMM,
retaining alias and hardware premises, and record complete generated plans,
selected family witnesses, dynamic counts, publication/release boundaries and
retirement separately. Full native and device qualification remains required.

## Integration qualification in PTOAS-oahs-clean

After applying and adapting the patch, the six focused OAHS CTest gates pass
serially. The dedicated native ownership campaign passes top-level and
section-wrapped positive cases with credit enabled and disabled, and rejects
missing/partial transitions, malformed entry ownership and equal-byte
mismatched geometry. The public pass also accepts a physical section whose
final payload operation is a counted loop in both credit modes.

The exact final compiler replay admits **253/363** frozen inputs and refuses
110 without crashes or timeouts. Coverage is unchanged by this hardening:
86 refusals require the unqualified MTE3-to-MTE2 GM publication, followed by
seven remote-signal alias refusals and smaller explicit operation/context
contract populations. This is compatibility evidence, not device or numerical
qualification.
