# M1: general residual analysis

Base: `985af99082e7e6fc8eb68085107ac4dac52f8200` on `codex/oahs-clean-m1`.
Specification: *Order-Preserving Synchronization with Reusable Events*, v0.5,
sections 5--7 and 9.5. This milestone implements the general analysis service,
not the later `BackwardCuts`, `CoverByPrefixes`, or episode refinements.

## Interface and status

`PTO/Transforms/OAHS/Analysis.h` provides:

```cpp
auto before = oahs::analyze(program);                  // empty command population
auto after = oahs::analyze(program, candidateCommands);
if (!after.complete) { /* inspect diagnostics: missing contract or bad input */ }
else if (!after.verified()) { /* inspect residual/protocol/retirement obligations */ }
```

`complete` means a supported fixed input/plan was analyzed to convergence.
It never means synchronization is sufficient. `verified()` also requires empty
residual, protocol, retirement, and diagnostic populations. `verify()` uses this
same service, without requesting copies of boundary states. No previously cached
constructor demands are trusted. `captureStates=false` changes reporting and
copy cost, not analysis semantics, precision, or acceptance.

The old declaration-only `Result analyze(program)` is renamed `validateProgram`.
Existing callers that only wanted the structural/declaration check must adopt
that name. This deliberately avoids calling a declaration check residual analysis.

The native overload `analyzeHandoffSync(function, NativeAnalysis &)` exposes the
imported `Program`, the report, and a mapping from physical-phase IDs to ORIGINAL
MLIR operations. It analyzes unsynchronized input using the existing native
import contract and never inserts commands. Success means analysis completed;
residual requirements are expected. The one-argument native API calls the same
service and discards the report. Missing native operation contracts still fail
import explicitly. Phase pointers are valid only while the caller preserves IR. The public service
uses the existing fresh-invocation/retirement contract; M1 adds no arbitrary
nonquiescent calling convention or external live-token import.

## What the report preserves

- Original producer and consumer phase IDs, cell, actual read/write roles, and
  separate RAW, WAR, WAW and exclusive-resource witnesses. A read-modify-write
  can contribute more than one kind. Every consumer is inspected, not just the
  first failing consumer. Several readers remain separate prerequisites.
- Per-observer pending classes and live-event coverage remainders at every
  represented physical cut and invocation exit: incoming, before payload issue,
  and outgoing. This is the CURRENT cut vocabulary; arbitrary scalar/region-entry
  insertion points and additional executable predicates are not added by M1.
- Logical possible occupancy, must-valid receipts, latest-consumption knowledge,
  and carried acknowledgment facts. No wall-clock occupancy claim is made.
- Parent-linked static branch/loop contexts, with distinct loop owners and while
  before/after contexts. They are scope/provenance descriptions, NOT executable
  predicates, path-sensitive partitions, or exact occurrence-distance claims.
- Endpoint precondition failures, causal rearm failures, unconsumed invocation
  state, and payload retirement obligations. Failures mean not established under
  the abstraction; they are not invariably concrete counterexamples.
- Separate malformed-input and unsupported-semantic diagnostics. Unimplemented
  resource, visibility, authored-event or internal-phase effects are not silently
  interpreted as byte completion or as empty effects.

For original source metadata use `program.operations[requirement.demand.producer]`
and its `original` / `phase` fields. Cell geometry/provenance remains available in
`program.cells`; the native report supplies the MLIR phase pointers. A report's
context/key indices and snapshots belong to that analyzed program and candidate.
They must be discarded after relevant edits. No cached-receipt reuse API is added.

## One interpreter; provisional facts are not established completion

The existing static worklist and primitive transfers remain the implementation.
Construction still uses a private, explicitly speculative first-obligation query
when proposing repairs. This query is not the public `AnalysisResult` and cannot
accept emitted code. It remains useful for building a complete memory plan before
checking whether its genuine release packets already justify event recurrence.

The reporting/checking path does the following:

1. Solve the finite worklist with a provisional certificate for each endpoint.
2. Check every endpoint precondition at stabilized incoming states. Revoke each
   failing endpoint's certificate, conservatively forget that key's balance and
   receipt, and invalidate saved acknowledgments of its latest consumption.
3. Recompute states from entry, so downstream publications cannot retain payload
   completion or consumption facts derived from a revoked certificate.
4. Repeat until no new certificate is revoked. Then collect ALL residual and
   protocol obligations and optionally export boundary states.

A revoked endpoint does not disappear from the candidate. It remains a reported
protocol obligation and makes `verified()` false. Only its ANALYSIS CREDIT is
suppressed. In the reference continuation it leaves pending payload unchanged;
its key is possibly empty/full, its receipt invalid, and its consumption evidence
unknown. Other keys may still carry independently established facts. Continuing
past an invalid protocol is diagnostic analysis of the original reference control,
not a claim that the invalid candidate can execute.

This is conservative at base precision. A primitive invalid on one merged path
loses its certificate for the static site. Later path refinement may recover a
more precise fact, but neither a recognized episode nor a new runtime guard is
required by the base service.

For E event-command sites there are at most E+1 certification solves: each
repeating pass permanently revokes at least one previously enabled site. Each
solve uses the finite change-driven worklist; runtime trip counts are not
expanded. There is NO empirical iteration/work cutoff. Valid supported plans
need one certification pass. Constructor's existing finite repair progress and
actual hardware-key limits are unchanged.

## State invariants and proof boundary

The intended concrete-to-abstract relation is the one proposed in v0.5 Appendix D:

* `pending[q][a]` overapproximates earlier dynamic visits of physical phase a
  whose completion is not established at q. Issuing a fresh visit adds its class
  to every observer and every valid old remainder. It does not kill older readers
  or complete the previous write. Absence of a bit does not prove that a occurred.
* A valid event remainder overapproximates classes not certified by its actual
  publication. Publication removes preceding source-engine classes from this
  remainder, not from the publisher's pending state. Acquisition intersects the
  observer's pending state with a valid remainder. SET does not gate later issue.
* Occupancy is may-balance along reference control. Receipt validity and consumed
  knowledge are must-facts. Fresh consumption invalidates earlier acknowledgment
  generation facts in every saved receipt; a barrier does not consume an event.
  Initially consumption knowledge is vacuous because no earlier use exists.
* Site joins union may-facts and intersect must-facts. Unreachable is separate from
  fresh quiescent input. All preconditions use stabilized invariants. For retains
  bypass; while exits after its before-region. No region-entry reset is introduced.

Certificate revocation does not supply a new ordering fact: it replaces an
unjustified transfer by uncertainty and recomputes dependents. Its finite progress
is straightforward; the complete generation-sensitive concretization and hardware
simulation proof remain outstanding. These implementation invariants, local tests,
and finite graph checks do not constitute that full proof. The accepted core's
progress argument still relies on its original forward-reference prefix contract;
unmodeled hardware queue or macro prerequisites require separate qualification.

## Tests and cost

`oahs-analysis-test` exercises the v0.5 section 7.5 multiple-writer program with
no optional summaries, nested while/for/choice, all residuals, fan-in, independent
readers, self recurrence, fresh writes, branch-local missing publication,
unsafe rearming, no laundering of an invalid receipt through another key,
acknowledgment versus reader release, independent valid transitive receipts,
retirement versus event cleanup, invalid inputs, and unsupported semantic fields.
It compares fixed-plan acceptance with the existing first-failure check and checks
uncovered conflicts against the independent concrete launch/finish graph. That
graph NEVER inserts desired conflicts or rearm obligations as actual edges.

The native test driver now checks that analysis returns real RAW witnesses, native
phase mappings, branch contexts and retirement obligations while preserving IR.
The source patch alone does not establish that these native tests have run.

Existing baseline, structured and scaling tests remain. The analysis report can
contain quadratically many conservative producer/consumer witnesses; it is an
explicit audit population, not a near-linear compression theorem. State capture
adds snapshot-copy/storage cost. The final checker disables snapshot copying.
This milestone adds no incremental cache, sparse dominance compression, early
prefix search, slot-history arithmetic, kernel recognizer or planner mode.

## Remaining scope

M1 establishes the residual-analysis interface and its conservative proof-credit
boundary for CURRENTLY ADMITTED semantics. It is not completion of the entire
production-population semantic-accounting milestone. Macro phases/private events,
explicit visibility, authored communication and queue/ownership protocols, mixed
physical contexts and additional target/operation variants remain explicit native
or core refusals. Value/control availability must still be qualified by import;
the new report does not invent asynchronous SSA-value semantics.

The next algorithmic milestone is backward original-cut queries and separately
typed prospective prefix coverage, consuming these states and obligations. Full
native build, corpus accounting and device validation remain separate gates.
