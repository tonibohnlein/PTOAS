# Shared production synchronization semantics

Both InsertSync constructors consume `PTOIRTranslator` physical records. The
handoff adapter obtains semantic accounting from `describeSemantics()` instead
of maintaining an independent ordinary-op registration list. The report is
read-only and includes every original operation, its category, translated
phases, and any missing contract. Run it again after an analysis refreshes the
translated storage origins.

## Ordinary contract

The ordinary compatibility contract is production's `OpPipeInterface` plus
`MemoryEffectOpInterface`: one physical pipeline and explicitly declared byte
reads/writes. Shared validation requires a translated phase with the declared
pipeline, a mapped physical record for every effect, and a declaration for every
translated effect. Valueless effects, non-default resources, missing effect
interfaces, and effects lost during translation are gaps. An empty translated
list by itself establishes nothing.

An operation using this contract needs no handoff registration. For example,
ordinary vector arithmetic, matrix extracts, matrix multiplication and
accumulation, and ordinary FIX stores use their existing interfaces. Whole
operation completion remains the model for ordinary accumulation; no block
permission or UnitFlag completion credit is enabled.

`SinglePhaseSyncOpInterface` is an optional lowering-owned restriction for
variant-bearing operations. It can withhold the ordinary contract, as for
special load/store forms. Non-default accumulation/store phase attributes also
require a resource contract. These restrictions are applied by shared semantic
extraction, independently of the constructor.

This reuses the production abstraction as an explicit compatibility premise.
It is not a claim that every hardware lowering has independently been proven
complete. New lowering behavior must update its existing effect/pipeline or
macro/resource description; absence of a constructor opcode case is expected.

## Descriptor effects

The `tileDescriptorEffect` annotation distinguishes synchronous tile-handle
metadata from payload storage. `set_validshape` and `get_validshape` supply it
from their effect implementations, not from a caller-controlled IR attribute.
They lower to `SetValidShape`, `GetValidRow`, and `GetValidCol` member accesses.
The shared report accounts for them in original issue order; they create no
asynchronous payload phase. Payload and descriptor operations are preserved.

These effects retain MLIR's default resource and their original value/read/write
roles, so generic compiler passes retain conservative dependencies with tile
commands. Synchronization analysis refines their meaning using the effect
annotation. There is no descriptor-opcode dispatch in the handoff adapter.

## Other semantics and compatibility

Macro models retain their production phase population, but internal completion
and private-event lifetime composition remain reported gaps. Authored events,
visibility, calls, unrepresented ownership effects, and unsupported structured
control are likewise explicit gaps. Native target/core admission is a separate
check. These rows remain in coverage totals even when ordinary analysis grows.

Shared extraction also repairs missing atomic destination reads and queue
payload/storage effects for `existing`. These corrections can change its
synchronization where it previously missed a hazard. Default and explicit
`existing` remain equivalent. Unsupported protocol descriptions leave the
legacy translator behavior intact; handoff refuses incomplete descriptions. Handoff requires a complete report, then uses the existing constructor,
allocation, emission, and reconstructed verification. Origin closure and local
unknown-address widening retain their previously established handoff behavior.

## Inspection and regression gate

`pto-oahs-native-test --analyze INPUT` reports all shared semantic gaps and runs
native residual analysis when shared extraction is complete. It checks that the
original function remains identical. This is analysis coverage, not synthesis
coverage. Run the same original input independently through `algorithm=existing`
and `algorithm=handoff` to compare emitted mechanisms; failed synthesis is not
a successful coverage row.

Regressions exercise operations with no handoff-specific declaration, the
ordinary cube chain, missing translated effects, pipeline disagreement,
variant refusal, analysis preservation, and transactional rejection. The local
prefill population is evaluated with the original payload and contracts.

## Preserved A3 tile queues and atomic stores

`getSyncProtocolModel()` is owned by the PTO IR/lowering implementation, next
to the queue effects and verifiers. Both constructors use its physical effects
through the shared translator. The handoff constructor has no queue opcode
cases. Native reports retain the original handle, local slot binding, contract
identity, and cross-core flag reservation.

The initial queue contract is `a3-gm-tile-fifo-v1/pto-isa-0c112d61`, based on
pto-isa revision `0c112d61f41342bd0867ce1080c29f1590d72484`,
`include/pto/npu/a2a3/{TPush,TPop,TFree}.hpp`, and PTOAS's
`PTOToEmitC/ScalarMisc/AsyncSession.cpp` and `buildTPipeToken()`:

- Tile push reads its local tile and writes the GM ring on FIX or MTE3.
- Tile pop waits for peer readiness, loads GM into local slots through MTE2,
  and returns GM-slot credit on MTE2. The local slot envelope spans all possible
  indices. Different declarations bound to overlapping slots therefore alias.
- For bidirectional pipes, VEC uses the first local base and MAT uses the peer
  base, matching EmitC's constructor operands. Dynamic local bases retain
  unknown-range coverage. Pops rebinding existing non-declaration handles
  conservatively widen the affected local address space, including old aliases.
- Tile free is a no-op in this source profile. It does not release local reads.
- Peer flags occupy a separate cross-core namespace. Qualified tile paths use
  no private directional local keys, and EmitC leaves UnitFlag disabled.

The original queue operations, operands, attributes, and order are preserved
and checked during reconstruction. Peer matching, peer progress, and the
runtime's cross-core drain remain prerequisites of the original program; the
local checker grants no peer-event completion credit and does not prove global
queue liveness. This is a source-qualified compatibility contract, not device
qualification or an independent proof of the runtime protocol.

Global-entry TALLOC/commit/release, A5 queues, odd-split/explicit-subblock
overloads absent from the pinned source, and phase/quantized epilogues need
their own complete descriptions. They remain explicit gaps, not ordinary empty
byte effects. Scalar ownership effects are not silently treated as payload.

Atomic-add TSTORE has a source read and destination read/write. Its lowering
scopes atomic mode setup/reset around the normal store command; the original
operation retains that sequence. This adds no cross-core completion guarantee.

Coverage must inspect the production boundary: frontend pipe normalization,
pipe-init validation, and reserved-buffer resolution precede InsertSync.
Running InsertSync directly on raw frontend queue syntax measures a different
stage. Report analysis admission separately from constructed plans and device
execution; neither implies full PyPTO/pypto-lib population coverage.

## Preserved A3 hard collectives

The lowering-owned protocol model also describes hard `pto.syncall` for A3.
The source contract is `a3-hard-collective-v1/pto-isa-0c112d61`, from the same
pinned source's `npu/a2a3/SyncAll.hpp::SYNCALL_IMPL`, `common/type.hpp`, and
PTOAS's `PTOToEmitC/SyncComm/Sync.cpp`. This is the normal explicit-instruction
lowering; the source's `__PTO_AUTO__` build mode suppresses the intrinsic and
is not a qualified cross-core protocol implementation here.

Hard collectives contain a local ALL drain followed by the original device
handshake. They access no payload or scratch bytes and use no directional
local keys. AIV-only uses cross-core flag 14, AIC-only uses 11, and mixed uses
11 through 13. The report retains the participant group, flag range, and
local-drain behavior. No synthetic payload phase is introduced.

Both constructors preserve the collective. Handoff reconstruction checks its
original position, attributes, and control along with the other original IR.
The current analysis deliberately grants no completion credit from its drain
or peer handshake. Local residuals and generated event occupancy continue
across it; an intrinsic ALL does not consume a local notification. This can
retain redundant local synchronization, and is not an optimality claim.

The original program/runtime must provide matching participating cores,
compatible control, progress, and noninterfering cross-core flag ownership.
The local analysis neither proves those global properties nor introduces a
new GM visibility guarantee. The reserved cross-core flags do not reduce the
unrelated directional local-key pool.

Soft collectives remain explicit gaps: their scratch accesses, cache/visibility
operations, and private local-event lifetimes need a different complete model.
A5 collectives also need a separate source qualification. Neither borrows the
hard A3 collective's empty payload-effect population.
