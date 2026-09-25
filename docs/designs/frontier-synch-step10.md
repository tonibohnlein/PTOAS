# Phase A step 10: exact structural boundaries and interval participation

**Integration update (2026-09-25): accepted at this component's scoped v0.44 gate.**
See [current integration evidence and corrections](frontier-synch-steps7-13-review.md).
Patch-delivery validation/status statements below describe the original submission.


## Review status and base

Review candidate for `codex/handoff-foundation`, based on
`4122dd1531fbdb2859bd7b27772dba56930be393` (integrated steps 2–6).
Specification: draft v0.44, Section 3.6 / D3, Lemma 5.22,
Proposition 5.23, and Appendix I.2 / Lemma I.2.

The standard-library production core was compiled and tested with GCC and
Clang, including AddressSanitizer/UndefinedBehaviorSanitizer. The real MLIR
adapter and public-interface fixtures have **not been compiled or run** in the
preparation environment: a native LLVM/MLIR/PTOAS build was unavailable. No
independent reviewer has accepted this increment. This is not a claim of final
Step 10 acceptance, complete Phase A parity, or a working constructor.

## Implementation and consumers

`ExactFrontierCore.h` implements the reference D3 sequence, choice and
qualified-counted equations, width-correct interval intersection/conditional
extrema, comparison normalization, and structural interval slicing.
`OriginalExactQueries.h` applies that core to the original structure, fixed
selector and shared scalar service. It uses `OriginalReadQueries`' predicate
and frontier arena; it does not allocate another obligation universe.

The public `ProgramAnalysis::exactFrontiers(interval, readOnly)` returns the
structural first/last roots and no-hit condition. Its `firstConflict` and
`lastRelevantUse` interval overloads independently qualify the chosen direction
and all its legal original cuts. The existing requirement-facing boundary
adapter and qualified support-boundary path call the same service. Generation
and source/target pairing remain separate D1/D2/D4 answers. In particular, a
structural Exact result is not a certificate for a matched event generation.

The complete original interval is the cache key: original version, owner,
selector, occurrence interpretation, both cuts, stopping-access inclusion and
continuation. `readOnly=true` additionally requires one reader engine and no
intervening write to the cell. Different reader engines are queried separately;
one engine's last access cannot discharge another engine's obligations.

The stopping flag retains its Step 2 semantics. A query stopping before an
access includes that access only when `includeStoppingAccess=true`. A default
legacy source-to-deadline interval excludes its stopping payload; a NoHit for
that interval does not remove the still-required deadline obligation. No
boundary adapter silently widens an interval to include that payload.

`All` is a complete original-effect summary used only for exclusions. Positive
may intersections are descended into and qualified. Writes on other engines
still split a read-only episode. A write-only child cannot disappear through a
reader filter and reconnect the reads on either side. Proved constant arms and
qualified zero-trip loops contribute only their actual original executions.

## D3 and the interval fragment

Sequence selects the first nonempty child and the last nonempty child, retaining
shared prefix/suffix nonemptiness expressions. Choice retains the original
guard identity. Counted repetition requires qualified original bounds, positive
step, final increment and invariant body participation. It uses the original
induction variable, not a new visit counter. Unknown participation is not
replaced by an unconditional last-iteration access.

For I.2, the adapter finds one exact selected read site in a zero-based,
unit-step loop, with no writes to the cell. It derives the conjunction of its
original control conditions. Invariant lower bounds and exclusive upper bounds
produce `ell = max(0, lower...)`, `h = min(N, upper...)`. Original invariant
Boolean operands are retained even when they are operands of an `andi` rather
than conditions of separate `scf.if` operations. Operand reversal, complementary
arms, conjunctions and complemented disjunctions use deterministic comparison
normalization, not valuation enumeration.

The immutable interval recipe retains the original SSA references and integer
interpretation. Its nonempty condition is `ell < h`. First selects `ell`; last
selects `h - 1` **only in the nonempty arm**. The interval atom itself performs
that conditional evaluation, so embedding it in an eager Boolean conjunction
does not speculatively decrement unsigned zero or signed MIN. Inclusive bounds
and equality (`j == k`, hence `[k,k+1)`) require an independent width-correct
proof for `+1`; failure retains Unknown rather than wrapping the bound.

Only admitted invariant scalar values are used. Every resulting endpoint is
checked again through the shared original-value service. Available,
NeedsCompletion, NotObservableHere and Unresolved remain distinct, including
named independent prerequisites and rejection of circular enabling control.
An unavailable later condition can invalidate the last frontier without
invalidating the first frontier. Bound evaluation does not inspect event state
or acquire completion.

## Effects and supported interval limits

The exact leaf test inherits the existing exact-cell read-incidence contract;
this change does not independently prove every native read footprint. Unknown
ranges, overlap-witness cells, varying/non-singleton physical address
alternatives and unresolved aliases do not receive a fixed-cell certificate.
For a write-only selector, this implementation requires a qualified full-cell
write witness instead of treating a may-allocation interval as a definite hit.
Partial-write overlap proofs more precise than that witness remain a precision
limitation, not a claim that partial writes have no conflicts.

The I.2 implementation is the stated one-site interval fragment, not arbitrary
symbolic last-use discovery. Non-interval varying conditions, several
independently varying reader sites, and unsupported nested participation retain
the original requirements. No finite unrolling proves an unbounded result.

The structural slice index handles whole structured children, body intervals,
legal payload cuts, and transparent sequence wrappers. A cut pair crossing a
split inside an unfinished child or requiring AfterBackedge transport retains
its missing D4 premise. This candidate does not implement Steps 7–9. Such
transport must be connected to their qualified records, rather than guessed
from lexical nesting. Selector qualifications without an implemented meaning
remain unresolved. These integration limits must be reviewed against the
complete parity inventory; they are not automatically draft-open problems.

## Evidence and review gates

Run the standalone production-core tests from a checkout with this patch:

```sh
cmake -S test/frontier-synch-step10 -B build-step10 -DCMAKE_BUILD_TYPE=Release
cmake --build build-step10 --parallel 2
ctest --test-dir build-step10 --output-on-failure
cmake -S test/frontier-synch-step10 -B build-step10-asan -DSTEP10_SANITIZE=ON
cmake --build build-step10-asan --parallel 2
ctest --test-dir build-step10-asan --output-on-failure
```

Actual local results: 1,560 finite frontier executions and 194,607 interval
domains passed, together with comparison normalization, structural-cut tests
and empty/overflow regressions. A separate 10,000-optional-reader stress test
formed 59,998 predicate nodes and 69,997 frontier nodes without enumerating
its valuations. These are core-formation and explicit query-output checks, not
a native compiler complexity or runtime claim. Assertions remain enabled in
Release configurations.

The main project adds `pto-frontier-exact-core-test` and
`pto-frontier-exact-test`. The latter uses actual verified MLIR scalar/control
programs and explicitly declared translated-effect fixtures, through the public
ProgramAnalysis interface. It checks optional and independent engines,
read/write selectors, stopping inclusion, invariant repetition, clipped/empty
intervals, singleton equality, last participation before the final iteration,
unavailable and asynchronous conditions, legal cut sides, unchanged original
IR and obligation identities. These are not full native PTO importer fixtures.

```sh
cmake --build build --target pto-frontier-exact-core-test pto-frontier-exact-test
PTOAS_BUILD_DIR="$PWD/build" python3 test/frontier-synch-step10/run_native.py
```

`frontiers.mlir` registers the same assertion runner with lit. Missing binaries
are an error, not a skipped successful test. Full native build/lit, existing-mode
regressions, real PTO read-footprint premises, supported transport integration,
and independent source/acceptance review remain required. The candidate must
not be recorded as accepted until those checks and any resulting fixes pass.

Relevant inventory areas: common boundaries/D3 in PA-040–PA-056, interval and
spatial qualification in PA-057–PA-059, original-value outcomes in PA-032–PA-035,
and full original-query keys in PA-014/PA-080. This note is a delta for review,
not replacement status for those entire families. Descriptor formation,
subscription closure and integrated development-kernel evidence remain
Steps 12–14; directional covering remains Step 11.
