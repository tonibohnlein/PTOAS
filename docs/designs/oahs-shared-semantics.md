# Shared production synchronization semantics

Both InsertSync constructors consume `PTOIRTranslator::Build()` physical
records. OAHS imports those nodes directly; it does not independently qualify
instruction behavior. An operation without a translated synchronization node
contributes no synchronization effects and remains in the original IR. No
explicit "no effect" registration is required for this default.

## Ordinary translation contract

The authoritative input is existing InsertSync's pipeline, memory-effect and
physical-storage translation. A pipeline operation without a memory-effect
interface can produce an empty-effect node; unmapped effects are handled by the
shared translator. OAHS neither invents effects for skipped instructions nor
uses an instruction whitelist to admit them. This default concerns only sync
analysis: it does not mark operations pure for other MLIR transformations.

`describeSemantics()` remains a read-only audit. It reports missing declarations,
unmapped effects, pipeline inconsistencies and unsupported protocol descriptions.
Its result is **not** required by native construction. `SinglePhaseSyncOpInterface`
qualifications are diagnostic information,
not a second registration requirement for handoff.

The causal checker validates the represented physical dependencies and generated
event protocol. It does not establish that instruction registrations completely
describe hardware behavior. Translation errors, unsupported native target/control
representations, and failed generated-plan validation still cause failure. In
particular, the current native adapter requires one translated phase per original
instruction: multiple phases need phase-aware emission anchors, and must never
be reduced by silently taking only the first phase.

Shared registration corrections belong to the operation/translator layer and
apply to both algorithms. New instructions need no OAHS-specific declaration.

## Descriptor effects

Pure `ViewLikeOpInterface` operations such as `pto.treshape` use the shared
translator's storage provenance and create no physical byte-access phase.
Their source and result must both have translated storage records. A descriptor
pipeline declaration does not turn such a view into an asynchronous payload
operation. Reads and writes through either handle retain alias dependencies;
the view itself establishes no completion. This matches existing InsertSync's
alias translation.

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

Shared macro/protocol descriptions remain translator inputs. Optional metadata
is retained for reports and qualified placement refinements, without becoming
an admission list. If a description is incomplete, structured origin refresh
uses the same ordinary-effect fallback as the translator. It does not introduce
an additional protocol-completeness rejection.

Registered atomic destination reads and queue payload/storage effects apply to
both `existing` and `handoff`. The native adapter then uses the
[selected constructor](oahs-selected-plan.md), shared emission and causal-frontier
reconstruction. [Storage import](oahs-analysis.md) closes loop-carried origins
and conservatively represents unknown local addresses.

## Scalar division effects

`TDivSOp` declares a read only on its tile input and a write on its destination.
Its custom syntax accepts both tile/scalar and scalar/tile order. The scalar
SSA value is not a memory access; any load producing that value retains its
own effects. This correction is in the shared operation interface, with no
translator or constructor exception for scalar operands. Native tests check
both orders, exact effects and retained tile RAW/WAR synchronization.

## Authored local events

The public InsertSync pass applies the same authored-event exclusion before
selecting either algorithm. Functions containing SET/WAIT or record/wait events
are preserved without further automatic insertion. This exclusion does not
validate the authored protocol or import its event ownership into OAHS.

Original cross-core and visibility instructions retain the shared translator's
behavior. Preserving them supplies no inferred peer progress or local completion.

## Inspection and regression gate

`pto-oahs-native-test --analyze INPUT` reports all shared semantic gaps and runs
native residual analysis when shared extraction is complete. It checks that the
original function remains identical. This is analysis coverage, not synthesis
coverage. Run the same original input independently through `algorithm=existing`
and `algorithm=handoff` to compare emitted mechanisms; failed synthesis is not
a successful coverage row.

Regressions exercise operations with no handoff-specific declaration, the
ordinary cube chain, missing translated effects, pipeline disagreement,
audit/construction separation, analysis preservation, and transactional rejection. The local
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
their own descriptions to improve shared translation. Missing descriptions remain
visible in the audit without adding an OAHS admission condition. Construction
uses the phases and physical effects supplied by the shared translator.

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

## Qualified local accumulator access ordering

`SyncAccumulatorOrdering.h` describes an access-order exception for ordinary
A3 `TMATMUL` / `TMATMUL_ACC` lowering to `mad`, using actual valid M/N/K
sizes. The [A2/A3-supported Mmad documentation](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html)
states that consecutive K-axis accumulations need PIPE_M below
`(m / 16) * (n / 16) == 10`, while sufficiently large shapes have native
write/read ordering. This is a development documentation source, checked on
2026-09-17; the lowering correspondence is the A2/A3 `TMatmul.hpp` path in
PTO ISA revision `0c112d61f41342bd0867ce1080c29f1590d72484`, also used by the
shared source-qualified contracts above. This is not new device qualification.

Admission is deliberately narrower: ordinary unspecified AccPhase, FP16 inputs
and FP32 ACC, positive full valid dimensions divisible by 16 and at most 4095,
matching K and result dimensions, a constant destination address, and an exact
canonical ACC atom. All M accesses to that atom must have the same qualified
physical destination, dimensions and layout. Unknown/mixed accesses keep their
normal requirements. A function containing mutable valid-shape updates does not
receive this exception. Small/dynamic sizes, partial valid shapes, mixed layouts,
other matrix variants and UnitFlag/phase modes retain conservative behavior.

Only an **accumulating consumer** may use this rule. A subsequent fresh matrix
initialization still needs its ordinary prerequisite. The rule suppresses the
local ACC access requirement in both the causal frontier and cold compact
analysis; it does not modify pending operations, launch/completion reachability,
publication contents or operand effects. Consequently neither operand release
nor M-to-FIX readiness follows from this exception. Existing actual transfers
can still establish completion and remove requirements through the normal path.
The `existing` comparison algorithm is unchanged.
