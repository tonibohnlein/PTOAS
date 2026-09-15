# M2: backward cuts and prospective prefix coverage

Prerequisite: the exact M1 package `oahs_m1_985af990`, based on PTOAS
`985af99082e7e6fc8eb68085107ac4dac52f8200`. This is an incremental patch ON M1,
not a replacement for it. No new remote branch state is assumed or modified.
Normative design: v0.5, sections 6.2, 6.4, 7.2 and 7.4. The concrete query
organization and deterministic selection policy below are implementation choices.

## Public interface

`PTO/Transforms/OAHS/Prefixes.h` provides:

```cpp
// The query OWNS copies: caller-side changes cannot silently stale its facts.
oahs::PrefixQuery query(program, currentCommands);
const auto &analysis = query.analysis();             // certified M1 analysis
const auto cuts = query.backwardCuts(consumer);
const auto one = query.inspectPrefix(source, publicationCut, consumer);
const auto cover = query.coverByPrefixes(consumer);
```

An overload `PrefixQuery(program)` uses an empty actual-command population.
`complete` means the fixed query completed, not that a physical protocol exists.
`matchingEstablished` means a reference publication/acquisition pairing has been
proved for the represented control. `routeAvailable` means a mechanism and an
eligible nonreserved key (or a supported consumer-cut same-pipe fence) exist; it
is NOT an allocation or a proof that the key can be reused. `selectable()` also
requires at least one residual component to be covered. `coversAll()` describes
a JOINT PROSPECTIVE cover; it is NOT final synchronization verification.

Prospective prefixes have no event key, occupancy state exported to the program,
or consumption credit. They never mutate the candidate, payload, predicates,
reservations, M1 outstanding state, or existing event receipts. Returned records
are observations/proposals, not trusted inputs to `verify`.

## Cut and context vocabulary

A publication cut is BEFORE all existing commands at the corresponding original
physical phase. An acquisition is AFTER those commands and BEFORE that phase's
payload. This order is important: an acquisition already at a publication cut
cannot retroactively become knowledge of the earlier source snapshot. The query
can instead use a later physical cut after that acquisition.

The current M1/native physical-cut vocabulary is retained. M2 does not introduce
arbitrary scalar/region-boundary cuts, intra-command insertion offsets, new
executable predicates, or opaque macro interior cuts. The native emitter and
reconstructor still determine where those physical cuts actually occur.

`Control.h` now builds the original static control graph for both M1 Transfer
and the queries. It retains sequence edges, both choice arms, counted-loop
bypasses/backedges, and while-before execution and its condition-false exit.
Context IDs and original phase IDs remain stable. Explicit backedge-owner labels
are metadata, not a new transition relation. Phase IDs need not be in lexical
order; candidate tie-breaking uses an independently recorded original lexical
rank. No runtime occurrence graph is built.

## BackwardCuts

Reverse reachability starts at the consumer and visits `(static site, has crossed
an original backedge)` pairs. It records candidate physical cuts, their M1 contexts,
and whether some backward path reaches them without or through a backedge. Loop
owners are retained as witnesses. This is at most twice the static-site population.

All ordinary backward-reachable cuts remain available. Source lanes are not
restricted to the original writer: a relay lane with genuinely acquired completion
may supply the prefix. A backward path is only a candidate-discovery fact, not
participation or a previous-generation certificate. In particular, a source cut
found on the next textual line through a backedge is not silently interpreted as
the right prior visit.

The initial source index is intentionally general reverse reachability followed
by actual prefix filtering. It is not episode discovery or a kernel-shape recognizer.
More selective provenance indexing may improve its cost without changing meaning.

## Forward coverage and freshness

M1's certified analysis supplies the source's incoming pending set at publication.
For source lane p, the seed is the same memory algebra as a real publication:

    U_seed = N_p \ source_lane_classes(p).

The query then conservatively adds every original phase that may issue after
capture and before the corresponding acquisition. The result U describes what
that immutable prefix does NOT establish. A fresh visit is added even if its
static phase class also names an earlier source visit.

Coverage for a residual requirement is established only when its producer class
is absent from U. Several independent components remain a conjunction. For a
selected set of prefixes, the joint remainder is their intersection; remaining
requirements are checked against that intersection. A relay can cover components
from more than one source only if its actual source snapshot has acquired them.
An invalid M1 endpoint supplies no such acquired fact.

The query does not simulate new acquisitions into existing source state. Thus it
cannot justify one new candidate using speculative credit from another. This is
conservative; actual realization can establish additional sharing on reanalysis.

### Reference matching certificate

Each prospective pair is checked by a finite two-bit balance monitor over the
same static graph. It starts empty; publication sets full and acquisition sets
empty; joins union possible balances. After convergence, every publication must
have empty input, every acquisition must have full input, and the invocation exit
must be empty. The two endpoints at a common cut are ordered publication first.

This catches branch bypass, repeated source without a target, repeated acquisition,
unseeded initial use, and unused final publication. Source and target inside a
common repeated body can be certified without unrolling. A richer first/next/last
or slot-dependent protocol can remain unproved; no correspondence is invented.

A reference pairing is NOT a consumption-before-republication path on the source
engine. Real event reuse still needs M1's protocol state and final verification.
The test suite deliberately exhibits paired virtual traces without causal rearm.

### Local justification and costs

If the source and target alternate on all represented reference traversals, each
acquisition names exactly one preceding source capture. The M1 source remainder
overapproximates work not certified there. Union with all possibly intervening
access classes preserves this overapproximation; stopping before the target avoids
including the target's own current effect. This gives the local class-level
coverage implication, conditional on M1's completion/domain contract.

Matching and freshness are computed separately. The balance worklist has two bits
per static site; freshness is a reachability scan stopped at acquisition. An N-bit
receipt is NOT propagated through every site for every candidate. With S sites,
E static edges, N phases and R residuals at the consumer, a single candidate uses
O(S+E+N+R) work and O(S+N) scratch, excluding the certified M1 analysis. Consumer
requirements are indexed once, not found by rescanning all global residuals for
every candidate. The returned covered-component list adds O(R) result storage.

Enumerating up to P*N cut/lane candidates and retaining their remainder vectors
still costs polynomial time and can require quadratic report space in N. The
constructor still reanalyzes complete candidates. This is NOT a linear-time whole
pass, a global optimum, or completion of the full abstract simulation proof.
There are no analysis-work, recursion-iteration, or candidate-attempt cutoffs.

## CoverByPrefixes and construction integration

Candidate seeds that leave every residual component pending are skipped: a saved
remainder only grows before acquisition, so they cannot help. Other candidates
retain participation/route failures as diagnostic observations. Among selectable
candidates the set-cover heuristic chooses greatest NEW residual coverage, then
earlier original lexical cut, then lane. It terminates because every selection
removes at least one remaining requirement. Counts, ordering freedom and event
pressure are not claimed interchangeable; the heuristic is not a dominance theorem.

The production constructor now consults these candidates for a missing cross-pipe
completion. A candidate must cover every currently selected source component; it
may additionally cover other sources. A genuinely certified relay is allowed.
The packet is then allocated and checked by the existing construction path.
No public planner or optimization mode is added.

Queries are rebuilt after every candidate edit. Existing packets keep their
ordinary finite repair slots. Each slot can be created once, its publication can
move to the consumer once, and it can gain one acknowledgment. The slot remains
indexed by the original missing-source obligation even if its publisher is a
relay. No new retry cycle or arbitrary attempt allowance is introduced.

Early publications are rendered at their actual cut before existing commands,
even when phase IDs are not lexical. A new common-cut packet is appended there;
it may cover more than the conservative pre-command query seed. Every complete
candidate is reanalyzed and verified, so neither the seed nor the coverage report
substitutes for actual emitted semantics. Existing same-lane repair still chooses
its supported consumer fence directly; M2 does not implement general optimization
among incomparable same-lane fence placements.

If no precise proposal is established, ordinary consumer-cut realization remains
available. Actual key scarcity or failed protocol repair can still require a checked
conservative result. Absence of a budget is not a claim that every early placement
is feasible. No legacy backend fallback is introduced.

## Tests and evidence

`oahs-prefix-test` checks early cuts through choices and repeated control, independent
fan-in, real relay sharing, stale/fresh snapshots, snapshot timing relative to an
existing acquisition, independent readers, original nonlexical phase IDs, absent
branch sources, zero-trip bypass, repeated endpoint imbalance, while-before entry,
reserved/missing keys, same-engine fences, invalid input and immutable sessions.
It includes the v0.5 multiple-writer case and nested variants without summaries.

The independent graph oracle now also has optional analysis-only prefix observation
vertices. They receive edges from prior source-command finishes, have NO outgoing
edges, and never enter an engine's actual finish list. Claimed coverage is checked
at those immutable source observations, before adding a hypothetical handoff. Thus
hypothetical receipt feedback cannot make an incorrect source snapshot appear true.
Separate checks materialize virtual packets and inspect reference balance, actual
coverage and acyclicity; causal rearm is explicitly not promised by these queries.

A frozen exact M1 interpreter compares full analysis snapshots, residual/protocol
reports and acceptance with the extracted shared-control implementation. It is
test-only. Native tests exercise query nonmutation, original phase mapping and
actual early source placement inside a branch. These native sources require the
full configured MLIR/PTOAS build and are NOT claimed executed by this package.

## Remaining scope

M2 is ordinary source-cut discovery and prospective coverage for CURRENTLY ADMITTED
semantics. It adds no macro, visibility, authored-communication, ownership/queue,
new target or operation-completeness contracts. It does not synthesize new guards,
slot correspondence, rank/epoch families or persistent lifetime schemas. It reports
unproved preceding-visit matching rather than guessing it. M3's richer event
realization and later optional precision can consume this interface; episodes are
not prerequisites. Native compilation, corpus admission and device qualification
remain separate gates.
