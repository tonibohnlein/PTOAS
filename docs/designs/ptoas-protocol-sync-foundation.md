# PTOAS ProtocolSync semantic foundation

## Decision

ProtocolSync is a new synchronization planner developed from PTOAS `main`, not
an extension of the mutable InsertSync planner and not a continuation of the
CanonicalSync demand-cover implementation. Its governing principle is:

> Protocol-first, obligation-complete.

The implementation starts from base
`75e4a224d45bb81b7101df97edd1e4a98c0e1b9d`. The frozen research reference is
`codex/canonical-sync-hardware-graph` at
`19674e06ff25be87badf0cb3595e014e443b558d`; it remains an oracle and source of
narrowly reusable contracts, not a branch to cherry-pick wholesale.

## Checkpoint A boundary

The first checkpoint is diagnostic only:

- shared operation summaries record physical phases, memory effects, storage
  provenance, slots, queue roles, fixed supply, and event reservations;
- `StructuredSyncIR` records immutable regions, phases, accesses, and program
  points with stable pre-order IDs; phase cores come from normalized physical
  sections, with function/pipeline fallback only when no section is present;
- the existing InsertSync translator remains the production implementation;
- a shadow adapter compares the new schedule with the legacy flattened
  structure, legacy core class/pipeline, definitions/uses, macro phases and
  completion, slot/depth facts, hidden reservations, and queue semantics;
- the analysis pass is disabled by default and never emits synchronization.

The pass runs after memory planning and reserved-buffer/identity-move cleanup,
but before `PTOResolveBufferSelect`. At this point planner-assigned physical
addresses and `pto.multi_tile_get` selector/depth information must coexist.

Use:

```text
--protocol-sync-analysis-only
--protocol-sync-dump=schedule
--protocol-sync-statistics
```

The dump and statistics options require analysis-only mode. Registering the
pass or compiling without these flags does not run a preparatory walk and does
not change the default pipeline.

Statistics are emitted as one JSON record per function. The schema includes
work counters and microsecond timers for all later stages from the outset;
storage-generation, channel, planning, allocation, materialization, and
verification fields remain zero while those stages are outside Checkpoint A.

During this shadow checkpoint, the semantic context imports storage provenance
from the legacy translator. ProtocolSync records are independent of legacy
types, and a later reviewed change will replace this transitional source with a
shared provenance provider before legacy InsertSync consumes the new frontend.

## Checkpoint A implementation map

- `ProtocolSync/SyncSemantics.h` and its implementation own shared operation,
  phase, access, slot, queue, completion, failure, and statistics records.
- `InsertSync/LegacySyncIRAdapter.h` owns the only boundary that exposes legacy
  SyncIR types; it constructs the transitional provenance context and performs
  shadow comparison.
- `ProtocolSync/StructuredSyncIR.h` and its implementation own construction,
  freeze-time validation, and deterministic schedule dumping.
- `ProtocolSync/PTOProtocolSync.cpp` owns the analysis-only pass and its JSON
  diagnostics.
- `Passes.td`, the PTOAS CLI, and the main lowering pipeline provide opt-in
  wiring at the audited pre-`PTOResolveBufferSelect` pass point.
- Focused lit tests cover IR/C++ output identity, physical interval plus slot
  preservation, structured control, queue depth, macro reservations, and
  deterministic rejection of an unsupported effectful operation.

## Research evidence retained

The four archived CanonicalSync campaigns remain external evidence:

- `d99520397`, SHA256
  `ab105c5729f0bf556a8b3b1bf72d8f41d5b3924ab986c943629ee9ec163e30f3`:
  opt-in registration must not affect the default pipeline; unsupported input
  must not crash or mutate.
- `0ec958f1e`, SHA256
  `ee99830218e5f7e5796524a29aec43963559828a1ad74208117b41ddd15c78cd`:
  collect stage/work counters from the first diagnostic implementation.
- `137e6282`, SHA256
  `aba7ba1bb735aa1b351d41b409c9193dbf40c2f304e1eb23396275ce11d2a9f9`:
  optional protocol failure must not reject an otherwise repairable function.
- `19674e06`, SHA256
  `d327cd32ebf132117b22594e0719b805f2fb80e03b5423a2411aae1e180d6218`:
  shared frontiers reduced static pairs on a tested A3 subset, but did not
  prove scarcity or performance.

The archives are not committed into PTOAS.

## Hardware and rollout limits

- Generated cross-core AIC/AIV synchronization is unsupported initially.
- Event-direction legality does not establish MTE3-to-MTE2 same-address GM
  publication; that capability remains fail-closed.
- A3 is selectively device-qualified. A2 remains device-unqualified.
- Existing function/section drains are production policy and are not removed
  by the diagnostic pass.
- No ReadyRelease protocol is enabled until a purpose-built A3 fixture proves
  logical lane depth, selector evolution, reuse distance, and repeated device
  execution.
- Future mutation must occur on a clone and commit only after independent
  semantic and MLIR verification.

## Next gates

Checkpoint B must demonstrate, on official one-buffer and double-buffer
fixtures, both planned physical storage identity and retained logical
slot/queue identity. Only then may storage timelines and access-derived channel
diagnostics be added. Protocol lowering and event allocation remain outside
this checkpoint.
