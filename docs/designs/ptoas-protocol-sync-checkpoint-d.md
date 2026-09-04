# ProtocolSync Checkpoint D: narrow one-shot emission

## Purpose

Checkpoint D is the first ProtocolSync revision that mutates IR. It is
intentionally a correctness milestone, not the general protocol planner.

The emitted subset is limited to a device-qualified A3 NPU 2201, same-core, exactly-once,
strictly linear sequence of exact physical phases. Every adjacent phase pair is
completion-ordered. This deliberately total phase chain makes the first emitted
planner obligation-complete before the generation-aware residual-obligation
interpreter exists.

## Supported plan

For adjacent phases `A` and `B`:

- equal non-scalar pipeline: emit `PipeBarrier<P>` immediately before `B`;
- equal scalar pipeline: use documented intrinsic scalar order and emit nothing;
- different target-legal pipelines: emit `SetFlag<P,Q>` after `A` and the
  matching `WaitFlag<P,Q>` before `B`;
- otherwise: reject the function without mutating it.

Every function containing physical phases receives one final `PIPE_ALL` drain.
When phases execute inside one explicit Cube or Vector physical section, the
barrier is emitted inside that section before its terminator. Otherwise it is
emitted before the function return.

## Admission restrictions

Checkpoint D rejects:

- loops, choices, guarded stages, and multi-phase macros;
- mixed AIC/AIV functions or phases spanning different physical sections;
- existing synchronization, queue actions, or other ordered semantic actions;
- unsupported or incomplete schedule facts;
- recurring, rejected, or unauthenticated local channels;
- scalar-GM access, ordered/read-write GM effects, and GM reads after any
  possible earlier GM write;
- unsupported event directions and event-ID exhaustion.

A2 uses the same NPU 2201 family but remains emission-disabled until a distinct
A2 device campaign qualifies this mutating pass. Direct pass use also requires
an explicit `pto.target_arch = "a3"`; the target resolver never relies on the
repository's historical attr-less A3 default. The `ptoas` CLI additionally
requires the user to spell `--pto-arch=a3`, so the driver's historical default
cannot qualify emission. Checkpoint D rejects every `pto.device-spec`, including
underscore-form A3 names, until an exact profile has its own device campaign;
legacy `Ascend910*` and A2 `Ascend910B*` names are not treated as A3 evidence.

The directed-event table is the intersection of the NPU 2201 hardware matrix
and the CANN 9.0 `HardEvent` ABI used by lowering. In particular,
`MTE1_FIX`, `MTE3_FIX`, and `FIX_MTE1` are not exposed by that ABI and are not
legal Checkpoint-D candidates even where a hardware overview describes a raw
combination or no current application scenario.

The GM restriction is intentional. Pipeline completion alone is not a proof of
scalar DataCache publication or of every DMA store/load visibility case.

## Allocation

Each selected event generation in one directed event domain receives a distinct
compiler event ID after subtracting hidden macro reservations. Lexical
`WaitFlag`-before-later-`SetFlag` order is not accepted as a consumption proof.
Different directed event domains may use the same numeric ID.

## Atomic mutation

ProtocolSync:

1. clones the complete input module;
2. plans, materializes, and verifies every defined function only in that staged
   module;
3. independently reconstructs every tagged protocol from the frozen schedule;
4. validates direction, event ID, reservations, placement, exact phase-chain
   completeness, channel coverage, and tail drain;
5. runs the MLIR verifier on each staging module and on the complete staged
   input module;
6. commits all cloned function bodies only after every function succeeds.

Any unsupported function or verification failure leaves the entire original
module unchanged and instructs the user to run legacy InsertSync.

## Non-goals

Checkpoint D does not implement:

- `ReadyRelease<1/2>` or any loop protocol;
- residual one-shot repair for irregular schedules;
- branch token transfer;
- existing-synchronization composition;
- visibility protocols;
- generated cross-core collectives;
- A5 lowering;
- cost-based synchronization minimization.

Those remain later checkpoints. The strict total chain should be relaxed only
when the residual-obligation interpreter can prove that omitted adjacent
boundaries are unnecessary.
