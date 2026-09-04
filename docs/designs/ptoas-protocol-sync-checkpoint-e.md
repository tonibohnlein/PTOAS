# PTOAS ProtocolSync Checkpoint E

## Scope

Checkpoint E adds recurring local-buffer ownership protocols after the
Checkpoint D one-shot foundation. The implementation admits strict
`ReadyRelease<1>` and `ReadyRelease<2>` vertical slices. It is opt-in with:

```text
--protocol-sync-ready-release
--protocol-sync-fallback=legacy|fail
--protocol-sync-dump=plan
--protocol-sync-statistics
```

It is not a production replacement for InsertSync. The implementation remains
A3-only and device-unqualified until the purpose-built repeated-launch campaign
passes.

## Admission contract

The initial planner admits one complete local lifecycle in one non-nested
`scf.for` only when all of these facts are proven:

- exactly one physical local storage timeline and one channel;
- one producer and one consumer phase on distinct pipes of the same core;
- authoritative capacity one from a `pto.alloc_tile` root, or capacity two
  from a `pto.alloc_multi_tile` result type; the latter also requires an exact
  unit-stride `(iv + offset) mod 2` selector and reuse distance two;
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

Logical lane selection precedes event-ID allocation. Every lane begins as
`Free=1, Ready=0`, and the planner certifies zero, one, odd, even, and inductive
steady-state transfers before allocation. Capacity two supports initial
selector offsets zero and one.

For producer pipe `P` and consumer pipe `Q`, the emitted protocol is:

```text
before loop:       SetFlag<Q,P>(release[lane]) for every lane
before producer:   WaitFlag<Q,P>(release[slot])
after producer:    SetFlag<P,Q>(ready[slot])
before consumer:   WaitFlag<P,Q>(ready[slot])
after consumer:    SetFlag<Q,P>(release[slot])
after loop:        WaitFlag<Q,P>(release[lane]) for every lane
```

Capacity one uses static event operations. Capacity two selects the allocated
IDs with dynamic event operations derived from the preserved slot SSA value.
The compiler pool is `0..5`; imported reservations are excluded independently
for the ready and release domains. IDs need not be contiguous. The loop body
never receives `PIPE_ALL`.

## Transaction and verification

The pass analyzes and mutates a disposable module clone. It commits function
bodies only after every function succeeds and the complete module verifies.
The emitted-IR verifier does not consume the selected plan: it reconstructs the
storage capacity from the allocation root type, checks it against the frozen
family record, and reconstructs the slot selector and lifecycle from frozen
accesses and stages. It then checks every prime/body/drain role, dynamic
lane-to-ID selection, placement, directions, IDs, reservations, and absence of
a body barrier. A selector describes which slot an access uses; it does not
prove storage capacity. Other local roots remain unknown-capacity until a typed
queue or allocation descriptor supplies an authoritative value.
Focused fault injection covers missing actions, wrong IDs/directions,
misplacement, selector corruption, bad lane tags, and a fabricated body
`PIPE_ALL`.

## Qualification gate

The exact `ReadyRelease<2>` revision requires the dedicated A3 zero/one/two,
odd/even, varied/random trip-count, both-offset, non-contiguous-ID, and repeated
launch device gate before qualification. Host verification or prior
CanonicalSync evidence is not a hardware certificate.
