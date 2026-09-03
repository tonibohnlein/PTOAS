# CanonicalSync mechanism effectiveness

This document is the durable ledger for CanonicalSync mechanism families. It
records what each family materializes, which correctness proof admits it, how
it interacts with event allocation, and what an ablation has actually shown.
Roadmap and experiment logistics remain in
[`ptoas-canonical-sync-planning-todo.md`](ptoas-canonical-sync-planning-todo.md).

## Measurement vocabulary

The following quantities are intentionally reported separately:

| Quantity | Meaning |
| --- | --- |
| Residual demand | A correctness obligation not already discharged by immutable supply |
| Candidate mechanism | One selectable, coverage-authenticated physical synchronization recipe |
| Selected logical mechanism | One candidate retained by selection; this is the quantity minimized by the current unit-weight objective |
| Emitted synchronization instruction | A concrete barrier, set, wait, cache operation, or fence |
| Event-ID pressure | The maximum number of interfering live event generations in one directed event domain |
| Device performance | Runtime measured on hardware; it cannot be inferred from static mechanism counts |

A selected barrier normally emits one instruction. A selected non-recurring
event mechanism emits a set and a wait. Recurring and visibility protocols may
emit more operations. Consequently, a percentage change in selected logical
mechanisms is not automatically the same percentage change in emitted
instructions or device runtime.

## Non-recurring shared event frontier

Status: implemented, enabled by default, and independently ablatable.

The mechanism starts from ordinary direct event windows in one concrete,
once-only block. A direct event for a demand from source pipeline `P` to target
pipeline `Q` has this physical shape:

```text
P: source ------ Set<P,Q>
Q:                 Wait<P,Q> ------ target
```

The set captures completion of the qualifying `P` prefix. The wait transfers
that completion frontier to the following `Q` suffix. The physical cut can
therefore cover more than the demand that originally motivated it.

For compatible direct event windows, frontier synthesis proposes one new cut:

```text
P: produce A   produce B   Set<P,Q>
Q:                         Wait<P,Q>   consume A   consume B
```

The set is placed after the latest included source, and the wait is placed
before the earliest included target. The candidate is legal only when the set
must precede the wait. This one set/wait pair can replace several narrower
pairs because it captures all included producer prefixes and guards all
included consumer suffixes.

Candidate discovery groups windows only when they agree on:

- physical core and directed event domain;
- control guard and action region;
- recurrence owner and mechanism kind; and
- one concrete block with ordered `after(source)` and `before(target)` points.

Normal optimization admits only non-recurring direct events and requires at
least two distinct motivating demands. Each proposed physical cut is evaluated
by the ordinary regional coverage engine, flat scoreboard, and bounded oracle.
The selector sees one authenticated event candidate, not a bundle of the
original mechanisms. Recurring lifecycles are excluded because their priming,
circulation, release, and drain must be proved as a complete protocol.

The default can be disabled for an otherwise identical planning run with:

```text
--enable-canonical-sync \
--canonical-sync-disable-shared-event-frontiers
```

### Measured effectiveness

The matched host ablation after commit `625374167` used one compiler binary and
the frozen 394-row corpus. The disabled arm changed only this candidate family.

| Result | Frontier off | Frontier on | Change |
| --- | ---: | ---: | ---: |
| Admitted rows | 199 | 199 | 0 |
| Fail-closed rows | 195 | 195 | 0 |
| Proposed shared frontiers | 0 | 577 | +577 |
| Selected shared frontiers | 0 | 84 | +84 across 50 verified functions |
| Selected logical mechanisms | 3,462 | 3,345 | -117 (-3.38%) |
| Coverage worlds | 10,383 | 10,960 | +577 (+5.56%) |
| Sparse incidence entries | 1,463,135 | 1,583,100 | +8.20% |
| Indexed boundary-phase tests | 1,937,469 | 2,138,996 | +10.40% |
| Bounded-oracle state operations | 1,607,871 | 1,629,302 | +1.33% |

The precise conclusion is therefore:

> Shared event-frontier synthesis reduced selected logical mechanisms by
> 3.38% on this corpus without changing admission.

This is not yet evidence of a 3.38% reduction in all emitted synchronization
instructions, event-ID pressure, or device runtime. The ablation also shows
that the extra candidates have a measurable compile-work cost. A device ON/OFF
comparison is still required to determine whether the broader cuts improve or
hurt execution time; placing the set after more producer work or the wait
before more target work can reduce pipeline overlap even while reducing static
mechanisms.

## Relationship to event-ID scarcity

Shared frontiers help scarcity in two places.

First, proactive frontier candidates can prevent scarcity. If selection uses
one shared event instead of several coexecuting events in the same directed
domain, allocation sees fewer event generations. That can reduce the number of
simultaneously required IDs from the compiler pool `0..5`. It does not always
reduce peak ID pressure: mutually exclusive or otherwise noninterfering direct
events may already reuse an ID safely.

Second, the post-selection scarcity coalescer reuses the same frontier-discovery
implementation. When allocation fails in a directed domain, it:

1. collects compatible selected event mechanisms from that failed domain;
2. discovers legal latest-source/earliest-target frontiers;
3. chooses the frontier containing the most selected members;
4. replaces those members with one coalesced allocation group; and
5. retries event allocation.

The scarcity path also supports eligible non-boundary, unguarded recurring
events, but it never mixes ordinary and recurring mechanism classes. A
recurring coalesced group retains its reverse-release lifecycle. If coalescing
does not make allocation feasible, the allocator next tries a serialized
ready/release repair and then a recurring-release pool; otherwise it fails
closed. No internal `PIPE_ALL` is introduced.

The current 394-row frontier ablation did not separately measure peak live IDs
or attribute a scarcity repair to this mechanism, and equal admission in the
two arms means it did not recover an additional row in that experiment. Its
scarcity benefit is therefore structurally implemented and independently
verified, but not yet quantified as a corpus-level ID-pressure improvement.

## Existing same-pipeline frontier sharing

Same-pipeline demands already obtain the useful frontier behavior without a
second synthesized family. A direct barrier is placed at a demand's target
endpoint, and singleton coverage records every other demand interval hit by
that same barrier. Selection then solves the resulting sparse cover.

In the same 394-row run, 2,732 unique barrier candidates were available and
1,763 were selected, avoiding 969 candidates (35.47%). Of the selected
barriers, 1,685 covered multiple demands across 141 verified functions. This
is why a duplicate same-pipeline frontier generator was rejected for now.

## Required evidence for future mechanism families

Every new family should add one entry to this ledger containing:

1. its physical recipe and hardware preconditions;
2. its eligibility and independent verification rules;
3. a default-on/default-off ablation with identical demand policy;
4. static mechanisms, emitted instructions, and event pressure separately;
5. compile-work deltas; and
6. device timing when launchable cases exercise the mechanism.
