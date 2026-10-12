# Shared early publication and terminal cleanup

Finite guarded insertion, sequence crossings and circuit endpoint preparation
use the same `prepareGuardedPublication` / `appendPublicationCleanup` mechanism.
This changes endpoint realization, without changing demand generation or the
Section 5.4 algorithms. No kernel names or instruction families select it.

For a retained cross-pipe demand, let `g` be its exact activity predicate and
`s` the presence of its source occurrence. Ordinary lowering first tries the
selected SET guard at the source cut. If that guard is unavailable, the shared
lowering can prepare these commands:

| Cut | Command | Activity |
| --- | --- | --- |
| Immediately after the source | SET | `s` |
| Immediately before the consumer | WAIT | `g` |
| Function epilogue | WAIT with the same identity | `s && !g` |

The consumer and cleanup alternatives use the same family, record and matching
tuple as the publication. Activity implies both payload occurrences exist.
Consequently an executed SET has exactly one matching WAIT. A skipped source
produces neither a SET nor a cleanup WAIT. Cleanup follows all original payload
work; it is never placed at a conditional join or an analysis boundary.

## Capability contract

The current shared construction certifies one publication per function
invocation from original structured branches or exact regional selectors. It
requires a single-block, single-exit function, source and consumer cuts with at
most one selected occurrence, an available consumer guard, and an available
complementary guard and identity at the return cut. Branch presence is built
outermost first with select-based
masking, so an inactive arm's poison does not become an active command guard.

The existing contextual expression emitter checks availability at each cut.
Deterministic scalar arithmetic can be replayed under its existing rules;
memory reads, region-valued computations and nondeterministic operations cannot
be cloned merely to make a source predicate available. Their original SSA
results can be used when they dominate the consumer or epilogue.

An intervening runtime loop does not prevent this construction when both
endpoints are outside it. An endpoint inside a loop is admitted when its emitted
predicate implies one fixed coordinate for every enclosing counted loop. Each
selected coordinate must be invariant across all those loops. The shared qualifier checks mandatory
equalities through conjunctions and select-based masks without enumerating
Boolean cases. Both insertion and physical allocation repeat this check. A
regional producer also supplies exact source existence in invocation coordinates,
so a last-visit publication can be consumed at return without a history buffer.
Repeated publication, an unavailable branch-private outcome, or an unavailable
terminal matching coordinate returns an explicit endpoint obligation.
Mathematical results remain retained and detached
preparation is discarded on failure.

Physical-context projection transfers every alternative cut, invocation loop
and original SSA predicate, then requalifies its proof against the receiving
function. Detached predicate definitions remain owned by the prepared plan.
Recognized physical sections are transparent only after implicit section
predicates have been materialized. Serialized allocation rechecks that a
cleanup inside a context-only section still belongs to the terminal suffix,
with no original payload or SET following it.

## Allocation lifetime

`RegionalAllocationGroup::terminalCleanup` records that the ordinary consumer
is not an upper bound for the publication's lifetime. The group's activation
includes every SET, including surplus publications. Regional conflict and lane
reuse proofs cannot reuse that palette after its consumer bound. Earlier
palettes can still precede it when the ordinary proof establishes safe handoff.

Palette coalescing preserves these groups without flattening their terminal
alternative into an ordinary bound. Repetition does not lift terminal cleanup
to a per-visit acquisition. This is a conservative allocation rule; it can
require more IDs than a finer conditional lifetime proof. Allocation failure
retains the logical plan, and scarce-resource repair remains separate.

## Draft correspondence and validation

The working draft's `direct_realization.tex` conditional-direct construction
uses source-qualified complementary cleanup, and `combined_algorithm.tex`
requires its endpoint and terminal contracts. Those passages were rechecked
while implementing this shared mechanism.

`sync_guarded_publication.pto` tests finite and sequence producers under both GM
alias policies, conditional/else/nested sources, empty and nonempty intervening
loops, and late region-valued guards. It independently derives byte conflicts,
compares payload reachability, checks exact SET/WAIT matching, and verifies
physical ID reuse against causal consumption. Two late publications require two
IDs even if their ordinary consumers appear ordered. Unavailable terminal
outcomes retain mathematical queries without preparing a plan. Nested and
non-unit-step selected consumers and last-visit sources are checked, including
rejection after a serialized visit filter is removed. A separate check covers
inactive source-arm poison.

The optimized build and replay receipts are under
`.local/three-paths/guarded-publication/`; `manifest.json` pins the base revision,
uncommitted source hashes, compiler commands, draft hashes and binary hashes.

The optimized `-O2 -UNDEBUG` build passed. The focused lit run passed all 14
tests, including the shared mechanism's 684 independently checked traces,
sequence composition, public compilation, finite allocation, bounded allocation
and macro-envelope regressions. The selected-visit checks include empty and
nonempty loops with negative lower bounds, crossing zero, and non-unit steps.
Public cube and A3/A5 vector section checks exercise logical insertion and
physical allocation. Cube checks also cover rollback when an original
branch-private outcome is unavailable.
An older guarded-rotating negative test timed
out; its varying-guard fixture also times out with the pinned M2 baseline
binary. That test is excluded from the 14-test result, and its separate
baseline/candidate receipts are retained. This is focused validation, not a
full-suite or corpus campaign.

The unchanged causal-convolution prefill input now passes the public logical
pass under both GM alias policies:

| GM policy | Logical pass | Elapsed seconds | Peak RSS KiB | Six-ID allocation |
| --- | --- | ---: | ---: | --- |
| MayNotAlias | Ready | 105.34 | 1,036,260 | Allocation export unavailable |
| MayAlias | Ready | 92.53 | 1,047,536 | Allocation export unavailable |

These are validation replays, not a controlled performance comparison. The
MayNotAlias replay overlapped the serial lit run within the aggregate two-worker
limit; MayAlias ran afterward. Both used a 180-second external watchdog and a
4-GiB address-space limit. Those limits are measurement safeguards, not class
criteria.

Every original PTO operation count is preserved in the output. Under
MayNotAlias, the plan contains 59 SETs and 84 WAITs: 25 surplus-publication
cleanup alternatives. Both allocation attempts report that a supported finite
or uniform allocation export is required and the adapter is not implemented.
This establishes logical preparation, not event scarcity, physical allocation
success or C++ emission for prefill. The logical plans remain available for
subsequent allocation work.

The [prefill manual event-reuse investigation](frontier-prefill-event-reuse.md)
records its two-bank feedback and barriers for future allocation work. It
distinguishes explicit pipeline throttling from additional ordering that still
needs to be established by comparison with the exact event graph.
