# Qualified final-read sources in ordinary construction

## Current status (2026-09-22)

The combined first-write/final-read experiment now constructs and reconstructs
KDA after the event-ownership and neighboring-use fixes. Its quality gate still
fails: 1,784 payload relations removed and 706 added versus the committed plan.
It remains disabled by default. The investigation sections below record earlier
checkpoints; see [the latest sweep follow-up](oahs-sweep-followup.md) for current
failures, validation and artifacts.

## Mechanism

`SelectedOptions::finalReadSources` (native test driver: `--final-read-sources`)
exposes a last-read publication before an original anchor's shared command word.
It is an opt-in extension of the existing constructor, disabled by default.

The native qualifier uses the existing imported physical cells, previous-reader
frontiers and original loop control. It requires a real subsequent conflicting
write: an invariant input with no release deadline does not justify refinement.
The storage analysis is shared with first-write qualification. Initial admission requires a nonempty constant-bound positive-step leaf
loop, an unconditional read prefix, no regeneration of the input, one reader
engine, and an inactive producer engine. Every conditional suffix path is
included in the read/write check. The original step determines the final guard
`upper - iv <= step`; the native qualifier checks increment/difference arithmetic.
A one-visit loop is explicitly represented. Already-refined owners are retained,
not overwritten by this importer.

`refineReaderVisits` supplies the final analytical continuation. The new mode
keeps original payloads and their ordinary words shared and inserts only an
independent final-only endpoint gap. It reuses `beforeSharedWord` and the native
emitter's existing ordering of those gaps. The gap may precede a scalar/control
anchor as well as a payload, but must immediately precede the same original
anchor. No runtime flag, payload rescheduling or assumed completion is added.

Ordinary source selection now checks:

1. The actual incoming source-prefix history covers every motivating read class.
2. A two-state original-graph walk pairs every publication with exactly one
   consumer, including repeated entries and bypasses. No motivating source read
   is regenerated while the transfer is live.
3. The chosen key is empty and its previous consumption is known **at the gap**,
   on every represented occurrence. Existing key uses cannot overlap the new
   interval. Reserved recurring/closed keys are excluded.
4. After the new acquisition, selected reverse publications/acquisitions must
   establish consumption before the next publication. This includes the next
   execution of the new source itself. Target-time emptiness is not source credit.

The candidate carries the ledger version and is rechecked before binding.
The dedicated gap's publication prefix is protected by endpoint identity in the
ledger: later prepend/after/restoration edits cannot insert a receiving command
before it. All existing acknowledgment endpoints are retained by this binding.
Actual selected replay supplies the receipt's completion; no hypothetical return
is credited. No new completed-plan deletion trial is introduced.

When qualified final gaps exist, the option uses whole-original-graph contextual
states. The control view indexes those gaps once; kernels without one do not
switch replay mode or scan the full graph for every provider. Saved ordinary sources
also check all-occurrence publication/acquisition matching under this refinement;
a source matching only one analytical visit cannot bind a shared emitted word.
Work is recorded as `finalReadQuerySites` and `finalReadPublications`.

## Validation

The linked constructor test `ReplayTestAccess::finalReadSource` covers single
and multiple visits, one and repeated region entries, and virgin and reused keys.
Sixteen finite constructed-plan comparisons pass the independent command graph:
no added payload relations, and strict reduction for multiple visits. Removing
actual return support rejects rearming; adding a later read rejects freshness.
A later word-start insertion must remain after the protected publication.

Native tests cover a top-level step-64 reader with a conditional suffix, a single
visit, and later-read, empty-loop and no-overwrite negatives. Native emission/reconstruction passes. A matched native explicit-command
comparison removes **4 relations and adds none** with the conditional suffix
active (6 payloads: 66 → 62 relations); the inactive path is unchanged (4
payloads: 28 relations). This is a host ordering result, not device timing.
The witness retains one acknowledgment in both arms. Its analytical graph grows
from 20 to 27 sites and selected replay evaluations from 29 to 191 (95 recorded
final-source query visits). This is a placement improvement with construction
cost, not a constructor-speedup claim. Endpoint counts, including barriers, are
10 versus 9; they are not dynamic event-pair counts. All 25
portable suites pass. The earlier supplied-plan acknowledgment/release witness
remains separately identified as such.

## KDA boundary

With first-write and final-read sources enabled, KDA selects one final-read
publication, then declines at cut 330: `recurring forward role lacks its
consumption path`. The exact later key and occurrence still need attribution. Possible causes are
lost incidental consumption support and shared-occurrence matching; this run
does not distinguish them. Keeping the original acknowledgment is not by itself
a certificate for every later publisher.
This is a later binding/support boundary, not permission to broaden the early
source or assume rearming. No accepted full KDA output or device gain is claimed.

The previous complete first-write plan still removes 1,512 payload relations and
adds 634; this mechanism has not yet closed that full-plan quality gate.

Artifacts live in `../kda-first-write-work/`: `final-read-core-test.log`,
`final-read-all-tests.log`, `final-read-native-tests.log`, `kda.final-read.log`,
and the default/experimental corpus directories. `final-read-native-order-0.json`
and `final-read-native-order-1.json` pin the native comparison.

The final corrected experimental sweep (`final-read-experiment-corpus-qualified/summary.json`)
passes **88/88 modules**. No corpus input selects a final-read publication, and
all 88 plans remain identical. The default sweep also passes 88/88 with identical
plans. Thus the current corpus demonstrates no placement benefit from this
extension; the positive evidence is the linked and native witnesses above.

Byte identity is a default regression diagnostic, **not**
a requirement for experimental plans: changed plans need safety and complete
ordering comparison, followed by useful matched device measurements.
