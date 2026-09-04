# ProtocolSync Checkpoint D: narrow one-shot emission

## Purpose

Checkpoint D is the first ProtocolSync revision that mutates IR. It is
intentionally a correctness milestone, not the general protocol planner.

The emitted subset is limited to explicit A2 or A3 NPU 2201 profiles,
same-core, exactly-once, strictly linear sequences of exact physical phases.
The profiles have distinct identities even though this checkpoint uses the
same conservative barrier, directed-event, and compiler-event-ID tables for
both. A3 one-shot emission passed the Checkpoint-D device gate. A2 one-shot
emission is an opt-in simulator candidate until its separate campaign passes;
it has no A2-silicon qualification. Every adjacent phase pair is
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
barrier is emitted as the final operation in that terminator-free section.
Otherwise it is emitted before the function return.

Physical-core ownership follows the nearest explicit Cube or Vector section.
For a flat function without a physical section, its explicit `pto.kernel_kind`
resolves core-ambiguous scalar and data-movement pipelines. The legacy shadow's
pipe-derived Vector classification may be refined only when that authoritative
context and the ProtocolSync phase both identify the Cube core.

## Admission restrictions

Checkpoint D rejects:

- loops, choices, guarded stages, and multi-phase macros;
- mixed AIC/AIV functions, more than one physical section, or phases spanning
  different section ownership contexts;
- existing synchronization, queue actions, or other ordered semantic actions;
- unsupported or incomplete schedule facts;
- recurring, rejected, or unauthenticated local channels;
- scalar-GM access, ordered/read-write GM effects, and GM reads after any
  possible earlier GM write;
- unsupported event directions and event-ID exhaustion.

Direct pass use requires an explicit `pto.target_arch = "a2"` or
`pto.target_arch = "a3"`; the target resolver never relies on the repository's
historical attr-less A3 default. The `ptoas` CLI additionally requires the user
to spell `--pto-arch=a2` or `--pto-arch=a3`, so the driver's historical default
cannot qualify emission. Plans retain the resolved target identity and cannot
be allocated or transactionally materialized under the other profile.
Checkpoint D rejects every `pto.device-spec`, including underscore-form A3
names, until an exact device profile has its own qualification record; legacy
`Ascend910*` and A2 `Ascend910B*` names are not used to infer either explicit
ProtocolSync profile.

A2 enablement is deliberately narrower than shared primitive legality.
Checkpoint D one-shot plans may use the A2 profile, but Checkpoint E
`ReadyRelease<1/2>` remains A3-only until recurring event generations are
separately validated on a faithful A2 simulator. Simulator evidence can qualify
the opt-in A2 compiler path for the tested one-shot subset; it cannot establish
silicon performance or silicon qualification.

The A2 and A3 directed-event tables are the same explicit intersection of the
NPU 2201 hardware matrix and the CANN 9.0 `HardEvent` ABI used by lowering. In
particular, `MTE1_FIX`, `MTE3_FIX`, and `FIX_MTE1` are not exposed by that ABI
and are not legal Checkpoint-D candidates even where a hardware overview
describes a raw combination or no current application scenario.

The GM restriction is intentional. Pipeline completion alone is not a proof of
scalar DataCache publication or of every DMA store/load visibility case.

## Allocation

Each selected event generation in one directed event domain receives a distinct
compiler event ID after subtracting hidden macro reservations. Lexical
`WaitFlag`-before-later-`SetFlag` order is not accepted as a consumption proof.
Different directed event domains may use the same numeric ID.

## Atomic mutation

`--protocol-sync-fallback=legacy|fail` controls anticipated unsupported or
resource-infeasible plans. The rollout default is `legacy`, which applies
InsertSync to the pristine staged function. `fail` is the strict admission and
diagnostic mode. Unsupported targets and internal planner, materializer, or
verifier failures never fall back.

ProtocolSync:

1. clones the complete input module;
2. plans, materializes, and verifies every defined function only in that staged
   module;
3. independently reconstructs every required protocol from the frozen schedule
   without consuming planner channel, protocol-kind, direction, or ID claims;
4. enumerates all fixed synchronization operations and rejects untagged or
   unexplained actions;
5. validates direction, event ID, reservations, exact frontier placement,
   exact phase-chain completeness, pair uniqueness, and tail drain;
6. independently verifies each materialized function in place, then runs the
   MLIR verifier once on the complete staged input module;
7. commits all cloned function bodies only after every function succeeds.

Strict-mode rejection or verification failure leaves the entire original
module unchanged. Statistics are buffered until module commit, so a staged
mutation in a rejected module is reported as `rolled-back`, never
`materialized`.

Statistics separately report attempted/admitted/rejected plans, selected
one-shot protocols, directed event pairs, same-pipe barriers, tail drains,
event-domain count and pressure, and the maximum allocated compiler event ID.
The selected-action counters describe planner decisions; final `status`
distinguishes committed results from module rollback. The `producer` field uses
the stable amendment categories `protocol-plan`, `legacy-fallback-unsupported`,
`legacy-fallback-resource-infeasible`, `fail-closed-policy`, and
`internal-error` (plus `analysis-only` when emission is disabled).

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
