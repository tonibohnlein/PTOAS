# KDA phase support and first-write rearming investigation

## Current status (2026-09-22)

The combined first-write/final-read experiment now constructs and reconstructs
KDA after the event-ownership and neighboring-use fixes. Its quality gate still
fails: 1,784 payload relations removed and 706 added versus the committed plan.
It remains disabled by default. The investigation sections below record earlier
checkpoints; see [the latest sweep follow-up](oahs-sweep-followup.md) for current
failures, validation and artifacts.

Investigated after choice-frontier commit `7c48f4ab3`. No new production
qualification is enabled by this note. The first-write experiment remains a
separate, disabled working-tree change.

## MAT cycle admission: a concrete missing interface

The imported KDA AIC graph has MAT cells 47–49 written by MTE2 in the prologue,
repeated body and tail. `qualifyReaderRegionCycles` rejects any cell with an
acyclic writer. It also requires every reader to belong to a qualified leaf
reader region. KDA has straight phase readers as well as child readers.
Consequently, removing the acyclic-writer veto alone cannot supply a complete
protocol.

The recorded default-plan residuals identify ten MTE2 fence sites involving
cells 47–49 (83, 130, 150, 190, 237, 257, 297, 344, 364, 522) and one involving
cell 50 (396). These are analytical cut identities, not dynamic counts. They
are leads for protocol construction, not proof that eleven fences are removable.
The earlier all-MTE2-barrier deletion fails native verification.

A general extension needs a per-generation sequence of:

1. First write, requiring no fictitious earlier reader release.
2. Readiness publication and first participating reader deadline.
3. Last physical reader, including straight phase readers and child summaries.
4. Next conflicting write and its actual acquired return.
5. Final consumption/drain, including zero-reader paths where WAW remains.

Use the existing cell/control role views to derive these relationships. Do not
couple all cells to one new global history product or silently relax mixed writer
engines. Preserve separate bank publication prefixes and the producer-support
safeguard against moving another residual fence after a new write. An initial
regression should combine a straight prologue, a repeated reader phase and a
straight tail, with a zero-reader/WAW negative.

## First-write experiment: a later key use invalidates earlier rearming

Reproduced using the frozen pre-commit experiment driver on the normalized KDA
input (`choice-final-driver --explain INPUT --first-write-consumers`). It fails
construction; it is not emitted code.

Canonical word 428 is shared by analytical sites 428 (cyclic) and 730 (peeled
first visit). It precedes operation 121, an MTE1 move reading MAT cell 50 and
writing LEFT cells 54–55. The selected ledger contains:

```text
428: SET M→MTE1 key0; WAIT M→MTE1 key0
     SET MTE1→M key0; WAIT MTE1→M key0
432: SET M→MTE1 key0
434: WAIT M→MTE1 key0
... back to 428 ...
```

The last selected pair is endpoints 322/323; words 432 and 434 also have first
visit aliases (734 and 736). The reverse exchange at 428 acknowledges its own
consumption. It does not acknowledge the later consumption at 434. Next-visit
publication at 428 therefore needs additional actual support.

The reduced event sequence fails consumption-before-republication, and passes
three visits with a reverse transfer after the later consumption. This is a
finite protocol witness, **not** a proof that blindly inserting that return
repairs all of native KDA or preserves its ordering. It demonstrates why an
already closed exchange is not a perpetual certificate for subsequent uses of
its key.

Before the linked repair below, the diagnostic implementation only staged publications in
`firstWriteWords`. The later insertion crosses ordinary shared-word key uses as
well. The required extension is a key-neighbor certificate over all affected
analytical occurrences: new publication, new consumption, and next publication.
It should establish an actual return at the new deadline or retain a certified
alternative binding. It must preserve the selected early source gap. Do not use
one occurrence's emptiness, move publication late, or assume a future return.

First reduce this to a linked constructor regression. Require missing-return
refusal, repeated entries, a first-only receipt sharing later publication words,
and preservation of bank-A/bank-B payload order. Then validate the staged native
protocol. Avoid repeating whole-graph trial solves for each possible helper;
the current experimental 12 checks / 152,103 site evaluations are diagnostic
scaffolding, not the desired production complexity.

## Evidence and next action

Local directory: `../kda-first-write-work/`.

- `kda.default.explain.txt`: original physical effects, cell topology and fences.
- `kda.first-write-current.explain.txt`: current failed experiment and exact
  selected endpoint population; expected process exit 1.
- `check-first-write-return-boundary.py`: reduced missing-return/actual-return
  check, both expected outcomes pass.
- `choice-isolated-tests.log`: all 25 portable suites pass without experimental
  edits. `choice-isolated-native-tests.log` passes; isolated KDA output is
  byte-identical to the checked choice candidate.

The investigation initially selected the linked shared-word/new-consumption regression
and occurrence-qualified key-neighbor repair. That is a smaller prerequisite
than broadening MAT phase admission, and protects early placement in other
refined loops too. The MAT phase interface can then build on that support.
No device speedup, complete MAT-cycle implementation, or accepted full-KDA
first-write plan is claimed.

## Linked repair milestone (working tree)

`ReplayTestAccess::sharedWordNewConsumption` now exercises the production
constructor's `edge` path with an accepted initial reciprocal exchange and a new
early publication/later acquisition. The first visit and repeated visit share
all five body words. Before the repair the linked test fails with `latest
consumption does not precede this publication`, matching the diagnosed boundary.

The new `consumptionBeforeNextPublication` query begins after **every** analytical
occurrence of the new acquisition. It follows original successors to the next
publication of that physical key, including a next execution of the proposed
publication. Only an actual selected reverse publication and its later acquisition
supply return evidence. Reverse publications before the new consumption do not
count. At joins, pending reverse-key facts intersect; path-specific facts cannot
be combined. The map only loses facts on revisits rather than enumerating a
product of histories. Each cut is processed initially and again only when its
pending-key set loses an element; the bound depends on the admitted directional
key population, not the number of control paths. More general relay support
remains unproved by this query.

If this sufficient certificate is unavailable, the constructor selects a direct
return after the new acquisition. The reverse key needs source-time emptiness
and consumption knowledge at every occurrence; failure leaves endpoints and the
publication gap unchanged. The forward publication is never moved. Both return
endpoints are recorded as an ordinary acknowledgment so existing return
composition/restoration can track them. Normal whole-ledger replay and final
checking still validate all forward and reverse generations; the structural
query is not a replacement for those checks.

The query is enabled only with `firstWriteConsumers`, which defaults to false.
Within that experiment it applies to repeated split transfers involving shared words or the
existing first-write prefix vocabulary. It is based on control/endpoint roles,
not KDA names. It does not enable native first-write observation qualification.
The old experimental full-protocol trial loop has been removed.

Native KDA with the opt-in first-write experiment now passes the old cut-428
failure but declines later at cut 720: `no reusable key or nonrecursive consumption
acknowledgment`. Thus the experiment still must remain disabled. The new run
records 25 neighboring-use queries / 7,524 site visits versus the old run's
12 trial solves / 152,103 analysis-site evaluations. These runs reach different
construction points; this is query-work attribution, not a matched total-time
speedup. There is no accepted full-KDA first-write plan yet.

Current milestone logs: `shared-rearm-before.log`, `shared-rearm-after.log`,
`shared-rearm-boundary.log`, `shared-rearm-native-suite.log`,
`kda.shared-rearm.log`, under the same local artifact directory.

### Admission correction from the corpus

The initial unrestricted shared-word rule changed four default partial-attention
AIV plans (rows 44–47), adding conservative returns. All 88 modules still passed
construction/reconstruction, but that was not a demonstrated quality improvement.
The final implementation gates the new repair with the existing disabled
`firstWriteConsumers` option. Its proof remains generic; promotion to all shared
words requires its own ordering evidence. The diagnostic `--explain` path now
passes the same experiment option to construction as to native import.

The initial broad-admission corpus is retained in `shared-rearm-corpus/` as
negative admission evidence, not the final default-plan result.

Final admission validation: all 25 portable suites pass again, as does the native
selected suite. The four affected attention modules were reconstructed with the
gated implementation and are byte-identical to their saved committed baselines;
all report zero repair queries. The other 84 initial corpus modules had no repair
queries or plan changes. This is an 88-module scan plus four targeted reruns, not
an assertion that a second complete 88-module campaign was executed.
`shared-rearm-gated-corpus/summary.json` records the final driver/input hashes.
The first-write bank fixture also passes four paths / 16 deadlines with zero
added payload order. No experiment changes were committed or pushed.

### Cut 720 allocation diagnosis (2026-09-21)

A standalone probe linked against the current native importer and constructor
reproduces the opt-in failure. Cut **720 is the source publication**, not the
consumer: it aliases word 418. The acquisition deadline is cut **746** (word
444), operation 129, the MTE1 `tmov` from MAT address 8192 to RIGHT address
32768. Its outstanding readiness requirements are cells 42 and 43 from MTE2.
The relevant body load is operation 118 (`tload` into `%115`); `%131` denotes
that MAT storage at the reader. Other original operations also alias these
cells, so static cell identity alone is not an occurrence certificate.

At both source occurrences (418 and 720), the six MTE2→MTE1 keys have:

| Key | Occupancy | Publisher knows previous consumption | Allocation restriction |
| --- | --- | --- | --- |
| 0 | Empty | Yes, in both gate and prefix | Globally closed/recurring reservation |
| 1 | Empty | No | Last consumption endpoint 313 at cut 400 |
| 2 | Full | Yes for the preceding consumption | Current publication 401, acquisition 430 |
| 3 | Full | Yes for the preceding consumption | Current publication 405, acquisition 434 |
| 4 | Full | Yes for the preceding consumption | Current publication 409, acquisition 438 |
| 5 | Full | Yes for the preceding consumption | Current publication 414, acquisition 442 |

Keys 2–5 are empty by the consumer, but that later state cannot justify their
publication at 720. All six pass the existing interval query from 720 to 746;
this alone does not establish source-time availability or reservation ownership.
Key 0 passes `canPublishAt` at every source occurrence. Its recorded recurring
publications are at 93/200/307 and acquisitions at 605/624/666. Those are
analytical sites, not a numeric execution timeline.

The exact cut-400 word is:

```
310 SET  MTE1→MTE2 key 1
311 WAIT MTE1→MTE2 key 1
312 SET  MTE2→MTE1 key 1
313 WAIT MTE2→MTE1 key 1
```

The existing return therefore does **not** acknowledge endpoint 313. The final
snapshot confirms no consumption knowledge at MTE2. F7 cannot place its ordinary
repair here because `straight(400,746)` is false. `straight(720,746)` is true.
This is a control/occurrence admission limit, not proof that all bindings are
physically impossible.

Private full-graph analyses of the *partial selected ledger* tested:

- Forward SET at 720 and WAIT at 746 using reserved key 0: complete analysis,
  zero protocol/phase-resource obligations or diagnostics.
- The same endpoints using key 1 without a return: 12 protocol obligations.
- Key 1 plus a reverse SET immediately after endpoint 313 and reverse WAIT
  before the unchanged forward publication at 720: zero protocol/phase-resource
  obligations or diagnostics, independently for reverse keys 1 through 5.

These trials preserve the publication cut and validate the actual shared words.
They do not validate compiler reservation ownership, a finished KDA construction,
or no-added payload ordering. Remaining payload residuals are expected because
construction stopped before completing the function. No trial was committed to
the production ledger and no production policy was changed in this investigation.

The immediate general opportunity is **certified reuse of an inactive recurring
reservation**: key 0 already has source-time credit and needs no new return.
Its certificate must account for the reservation's owning scope and both
neighboring uses on every participating path, including repeated entry and any
future endpoints promised by that owner. Do not simply ignore `closedKeys`.
The alternative is a qualified cross-control acknowledgment for key 1, using the
actual cut-400 consumption and retaining the early gap. It needs occurrence and
participation support beyond F7's straight corridor, and an ordering comparison.
Neither requires late publication or hypothetical rearming credit.

Reproduction artifacts are in `../kda-first-write-work/`: `cut720-probe.cpp`,
`build-cut720.py`, `cut720-build.log`, and `cut720-state.log`. The probe links the
current experimental archive, not the frozen sweep executable. These are host
investigation results; the first-write option remains disabled by default.

The probe also checks the motivating deadline explicitly: the unchanged partial
ledger has two residuals at cut 746 for cells 42/43. Both key-0 binding and every
qualified staged key-1-return variant reduce those to zero; key 1 without the
return retains both. Thus the successful trials establish the requested local
readiness as well as event legality, not merely a balanced endpoint population.

### Inactive recurring-reservation certificate

Ordinary binding now has an opt-in fallback after unreserved source-time keys
are exhausted and before F7 acknowledgment repair. It borrows an existing
recurring key without clearing its reservation. Admission requires:

1. Actual empty/consumption-known source state at every publication occurrence.
2. Fully materialized recurring ownership: channel publication/acquisition words
   equal the active owner endpoint population. Fixed endpoints, removed uses,
   lazy closed bindings, and entry-protocol reservations decline.
3. A two-state original-control walk: each proposed publication has one proposed
   acquisition; no old key use occurs while the borrowed token is live; no exit
   leaves it live. Shared words and repeated entries remain distinct occurrences.
4. An actual selected reverse receipt after the new consumption and before every
   next publication, including an owning protocol's next entry. No helper is
   added to make this borrowing certificate pass.

The certificate is recomputed against the current ledger immediately before
insertion. Normal contextual replay and final checking remain mandatory. Later
borrows inspect earlier borrowed endpoints as ordinary neighboring uses; the
owner is never globally marked finished. This is not general event recoloring.
The feature stays within `firstWriteConsumers` pending complete native evidence.

The linked shared-word fixture limits the forward pool to its reserved key,
executes the owning exchange again on later visits, and borrows the gap between
owner uses. It checks preserved early placement and zero added acknowledgments,
complete finite ordering inclusion for 1/2/4 visits, and rejects incomplete owner
materialization, lazy entry ownership, interval overlap, and missing return
support. The key-1 cross-control repair remains an alternative investigation,
not an automatic fallback added by this change.

Native result: the production constructor borrows key 0 at word 418/occurrence
720 (SET endpoint 340) and consumes it at word 444/occurrence 746 (WAIT endpoint
341), with no acknowledgment for this transfer. The next failure is cut 724,
consumer 750. The new probe confirms key 0 is full at 724, keys 2–5 are also
full, and key 1 remains empty without knowledge of consumption 313 at cut 400.
This is a separate, overlapping transfer; reusing key 0 again there would be
incorrect. The retained key-1 cross-control repair is the concrete next option.

Validation so far: 25/25 portable suites pass, including reserved borrowing over
1/2/4 visits and negative ownership/neighbor/rearming checks. The native build
passes. `kda.reservation.log` records the production progress and
`reservation-state.log` the selected endpoints and next source-time states.
The total run performs more work because construction progresses farther;
this is not a matched compile-time speedup claim.

Final host validation: native selected suite exits 0; default KDA reconstruction
exits 0 and emitted PTO is byte-identical to the pre-change default snapshot
(`kda.shared-gated-default.pto`). The full corpus was not rerun for this
experiment-only addition. No new device task or default promotion is warranted
until the full experimental KDA plan is accepted and ordering-compared.

### Cross-control acknowledgment qualification

The next opt-in repair matches the actual last-consumption endpoint at every
occurrence of the selected source word. A two-state original-control traversal
pairs a reverse publication immediately after that consumption with its
acquisition at the source gap. It rejects bypassed/unmatched visits, intervening
reverse-key uses, and intervening forward republications. The reverse publisher
must already have empty/consumption-known credit in the complete endpoint
aggregate. These structural queries are charged to `splitRearmingQueries/Sites`.

The return and forward transfer are checked together: the forward receipt can
carry consumption of the return back to its publisher. Checking the reverse
half alone prematurely rejects this actual cycle. No speculative causal fact is
installed; ordinary replay checks the complete selected command population.
The linked branch fixture tests repeated entries, bypass/interference/missing
return negatives, and full finite ordering inclusion versus late publication.

At the subsequent cut 728, all six forward keys really are full at the original
source gap. F7 now matches an old consumption's analytical occurrence in the
current peeled corridor before applying its ordinary repair. It does not use the
repeated-body representative's numeric position as that occurrence. This may
move that later, capacity-constrained publication; the no-added-order comparison
is required before claiming any KDA quality improvement. Cut 724 remains early.

Construction then reached final acceptance, but native reconstruction exposed
production codegen's signature deduplication of distinct same-word generations.
OAHS now requests exact command-word emission (including repeated SET/WAITs),
while the existing pass retains its default emission behavior. The native unit
regression requires three same-key SET/WAIT pairs to survive in their exact order.

### Completed native plan and failed quality gate

With exact-word emission, full KDA construction **and native reconstruction pass**.
The cross-control return preserves cut 724; occurrence-aware F7 also handles the
later capacity-constrained transfer. Native selected tests and all 25 portable
suites pass, including the finite branch fixture (1/2/4 entries, alternating
branch choices, missing return and reverse-key interference negatives).

The required comparison against committed `7c48f4ab3` is **not accepted**:

| Finite KDA case | Committed plan | Experimental plan |
| --- | ---: | ---: |
| Payloads | 1,727 | 1,727 |
| Strict explicit payload relations | 5,953,677 | 5,952,799 |
| Relations removed | | 1,512 |
| Relations added | | **634** |

Bindings are `%arg16=67, %arg17=67, %arg18=67, %arg19=0, %arg20=5`.
`kda.cross-return-order.json` records exact input/output hashes and examples.
The first added completion-to-issue example is matrix accumulation payload 12
(completion vertex 25) to MTE2 load payload 14 (issue vertex 28). Thus fewer total
relations does not establish inclusion, and full safety acceptance does not
meet the plan-quality requirement. This is the explicit local graph, not a
peer/queue visibility or hardware-latency claim.

An attribution-only experiment kept already-required helpers out of the existing
engine-pair deletion trial; a second also pinned the newly selected helper pairs.
Neither enabled successful helper composition (zero `rearmingComposed`), and the
second increased acknowledgments. Both edits were reverted. Their frozen outputs
and logs are `kda.protected-helper*` / `required-helper*`; they are not implemented
policy. Do not add an endpoint-subset deletion search to force the quality gate.

The cross-control repair, reservation borrowing and exact emission are retained
as tested mechanisms; the first-write experiment remains disabled by default.
There is now a complete experimental native plan to diagnose, rather than an
allocation failure. Next work must explain/remove the added compute-to-load
paths through actual selected returns and exported completion, while preserving
necessary rearming and the useful early gaps. No new device comparison task is
issued: the user's prerequisite of a useful change with **no added local payload
ordering** failed. The existing committed-revision sweep remains a separate task.

Artifacts: `kda.cross-return.pto`, `kda.cross-return.log`,
`kda.cross-return-order.json`, `cross-return-test.log`,
`cross-return-core-tests.log`, `cross-return-final-native-tests.log`, and
`cross-return-final-build.log` in `../kda-first-write-work/`. The default KDA
output is byte-identical to `kda.choice-isolated.pto` after exact emission.

Maintenance boundary: a cross-control helper is deliberately not registered with
the existing common-cut online helper-restoration list. That interface restores
a removed WAIT after its SET in the same word; the new pair occupies different
words. Its endpoints remain explicit and are still considered by the existing
cold final helper trial. Extending online composition requires stable restoration
anchors for both endpoints, not reusing the common-cut assumption.

Default-corpus check: all **88/88** historical modules construct/reconstruct and
retain byte-identical plans against saved committed baselines. Results are in
`cross-return-default-corpus/summary.json`; the corpus was run with two explicit
workers. This validates the default emission change's existing coverage, not the
experimental KDA ordering gate, which still fails.


### Added-order attribution: keep rearming, narrow the exported release

The first added edge has now been traced through actual emitted commands:

```
M accumulation (payload 12)
  -> M→MTE1 key 1 acknowledgment (before payload 13)
  -> MTE1→MTE2 key 2 MAT release
  -> MTE2 load (payload 14)
```

The committed plan has no such path. The trace is in
`../kda-first-write-work/trace-added-path.py` and `added-path.txt`.

A new linked supplied-plan regression, `releaseBeforeAcknowledgment()`, keeps
all acknowledgment endpoints and moves only the final-read release before the
acknowledgment. For two, three and four reader visits, both the causal checker
and independent graph oracle accept matching/rearming. Complete payload order
is a strict subset of the broad-release control; the penultimate compute no
longer gates the overwrite. Removing the first acknowledgment fails actual key
rearming. This demonstrates a feasible boundary, **not native construction of
that boundary**, and does not establish the full KDA no-added gate.

Two diagnostic implementation attempts were reverted:

- A source-inactive control corridor does not admit this source: ordinary read
  occurrences repeat, while the exit release participates once. Its native plan
  was byte-identical (`kda.source-corridor.pto`). Do not weaken participation to
  make that corridor pass.
- Allowing online helper composition across third-engine publications, with
  replay and whole-batch restoration on immediate failure, later failed at cut
  330: `recurring forward role lacks its consumption path`. Successful replay
  at the earlier edit does not certify the future allocation interface. Logs:
  `kda.online-return.log`, `online-return-build.log`. This is not retained policy.

The next implementation needs a last-participating-read source usable by
ordinary construction, including its exact pre-acknowledgment word gap. The
current native last-reader path excludes this top-level step-64 child with a
conditional matrix suffix; the joint path also looks for trailing reader-pipe
payloads, while this obstruction is an inserted receiving endpoint. Preserve
those distinct admission facts and original final-visit arithmetic. Keep the
actual acknowledgment until a replacement proves every affected key deadline.

Validation for the retained diagnostic regression: 25/25 portable suites pass
(`release-ack-core-tests.log`). No new accepted KDA plan or device task results
from this investigation; the full result remains 1,512 removed / 634 added.
