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

The diagnostic `--protocol-sync-dump=storage-tracks` mode also compares a
selected plan with the read-only storage projection. It requires matching
track masks, physical slots, directions, publication, acquisition, final use,
and next overwrite. This is a structural differential over shared schedule
facts, not an independent hardware proof. The complete fixture and corpus
results, including the correction from channel-admission proxy status to the
actual E planner result, are recorded in the [experiment evidence
ledger](ptoas-protocol-sync-evidence-ledger.md).

It is not a production replacement for InsertSync. A2 and A3 use the same
`Npu2201A2A3` capability profile. The shared profile is simulator-qualified for
A2/A3, and the purpose-built repeated-launch campaign additionally qualifies
the recurring implementation on A3 silicon.

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
- legal NPU 2201 A2/A3 events in both producer-to-consumer and consumer-to-producer
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
storage capacity and physical slot addresses from the allocation, checks them
against the frozen family record, and reconstructs the slot selector and
lifecycle from frozen accesses and stages. It then checks every
prime/body/drain role, dynamic lane-to-ID selection, placement, directions,
IDs, reservations, slot modulus, and absence of a body barrier. A selector
describes which slot an access uses; it does not prove storage capacity. Other
local roots remain unknown-capacity until a typed queue or allocation
descriptor supplies an authoritative value. Focused fault injection covers
missing actions, moved signals/waits, nesting an action under a guard, wrong
IDs/directions, shifted physical ranges, changed allocation capacity or slot
modulus, selector corruption, bad lane tags, and a fabricated body `PIPE_ALL`.

A separate C.7 reconstructor derives the strict lifecycle without reading E's
answer and matches all four `ReadyRelease<1/2>` fixtures on capacity, lanes,
phases, slots, and all four frontiers. It discovers 177 similarly narrow shapes
in 76 corpus functions that E rejects for other whole-function gates. This is
useful discovery evidence, not an expansion of E's admission contract.

## Qualification gate

The exact `ReadyRelease<2>` revision requires the dedicated zero/one/two,
odd/even, varied/random trip-count, both-offset, non-contiguous-ID, and repeated
launch gate before qualification. That evidence is simulator qualification for
the shared A2/A3 profile and silicon qualification when run on A3 hardware.
Host verification or prior CanonicalSync evidence is not a hardware certificate.
