# PTOAS ProtocolSync Checkpoint E

## Scope

Checkpoint E adds recurring local-buffer ownership protocols after the
Checkpoint D one-shot foundation. The first implementation milestone is the
strict `ReadyRelease<1>` vertical slice. It is opt-in with:

```text
--protocol-sync-ready-release
--protocol-sync-fallback=legacy|fail
--protocol-sync-dump=plan
--protocol-sync-statistics
```

It is not a production replacement for InsertSync. The implementation remains
A3-only and device-unqualified until the purpose-built repeated-launch campaign
passes. `ReadyRelease<2>` is the next milestone and is not enabled by this
revision.

## Admission contract

The initial planner admits one complete local lifecycle in one non-nested
`scf.for` only when all of these facts are proven:

- exactly one physical local storage timeline and one channel;
- one producer and one consumer phase on distinct pipes of the same core;
- authoritative capacity one and same-slot reuse distance one;
- exact publication, acquisition, final-use, and next-overwrite frontiers;
- matching ready RAW and release WAR results from the legacy shadow oracle;
- legal A3 events in both producer-to-consumer and consumer-to-producer
  directions;
- no branch, physical section, queue, fixed synchronization, opaque semantic
  action, scalar GM visibility, or mixed read/write GM family.

Any missing fact rejects the candidate. Unsupported functions use whole-function
legacy fallback unless fail-closed mode is requested. Internal planner or
verifier failures never fall back.

## Atomic protocol

Logical lane selection precedes event-ID allocation. The single-lane token
state begins as `Free=1, Ready=0`, and the planner certifies zero, one, odd,
even, and inductive steady-state transfers before allocation.

For producer pipe `P` and consumer pipe `Q`, the emitted protocol is:

```text
before loop:       SetFlag<Q,P>(release)
before producer:   WaitFlag<Q,P>(release)
after producer:    SetFlag<P,Q>(ready)
before consumer:   WaitFlag<P,Q>(ready)
after consumer:    SetFlag<Q,P>(release)
after loop:        WaitFlag<Q,P>(release)
```

The compiler pool is `0..5`; imported reservations are excluded independently
for the ready and release domains. The loop body never receives `PIPE_ALL`.

## Transaction and verification

The pass analyzes and mutates a disposable module clone. It commits function
bodies only after every function succeeds and the complete module verifies.
The emitted-IR verifier does not consume the selected plan: it reconstructs
the capacity-one storage lifecycle from frozen accesses and stages, then checks
all six roles, placement, directions, IDs, reservations, and absence of a body
barrier. Focused fault injection covers missing actions, wrong IDs/directions,
misplacement, bad lane tags, and a fabricated body `PIPE_ALL`.

## Next milestone

`ReadyRelease<2>` will reuse the atomic protocol but add two independently
allocated logical lanes, selection from the preserved slot SSA value, both
initial offsets, non-contiguous IDs, and independent selector reconstruction.
It requires the dedicated A3 zero/one/odd/even/random trip-count device gate
before qualification.
