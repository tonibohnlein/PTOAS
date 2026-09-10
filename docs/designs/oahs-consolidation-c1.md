# OAHS consolidation C1: direct guards, fitting-pool allocation, and delta completion

## Status and base

Manual merge of C1 into `codex/oahs-upstream` at
`d55a7c3968c07593a90da6685b63a38330085d7a`, retaining the upstream allocator
and guard-lowering behavior where it conflicts with the candidate patch.

The native C++ integration was compiled with the pinned LLVM/MLIR 19.1.7
build. Compact-update, endpoint, one-buffer, retirement, and scalability checks
passed. The full focused gate remains blocked by the unavailable system `libisl`
runtime; the three-buffer logical compilation also exceeds the local 90-second
harness limit, and device correctness/performance remain unqualified.

The portable production helpers compile and pass under GCC and Clang with
ASan/UBSan. Finite relation-update tests pass. These results do not substitute
for the full ordinary-buffer or device acceptance gates.

## Synthesis of the two reviews

The supplied review, `Pasted markdown(6).md`, and the preceding review of the
same head agree on the organizing algorithm:

    occurrence requirements -> completion-prefix advances -> logical handoffs
        -> valid finite events -> fresh emitted verification

They identify different aspects of the same cost problem:

| Finding | Decision in this patch | Remaining work |
|---|---|---|
| Compact facts are expanded and later rediscovered | Replace a common guard-search path with one relation-row-derived formula | Preserve compact maps/predicate provenance from predecessor construction itself |
| Guard lowering searches combinations of unrelated proposals | Try one bounded formula first; accept only complete equivalence | Broader executable guard forms and shared Boolean expression DAGs |
| Occupied low keys are tried before available keys | Dedicated-first when the entire directed-domain population fits the existing pool | Joint allocation/backtracking and bounded optional sharing |
| Completion additions replay every reached path | Seed only new entrances plus previous unfinished work | Guarded-prefix/antichain representation throughout completion queries |
| Shared integer algebra is correctness-critical | Add no elimination/projection rules; retain independent differential tests | Continue mathematical differential testing and isolate existing algebra |
| Reconstruction is not an independent mathematical implementation | Keep fresh emitted imports and existing mutation checks | Do not claim shared algebra is independently proved by reconstruction |

The attached review's main recommendations occur at lines 164-216 and 247-331.
The original constructor plan separately calls for dedicated keys before sharing
and retention of fresh emitted verification. No new planner, dialect annotation,
public option, effect whitelist, or target capability is introduced here.

## 1. Direct guard lowering

`BoundaryLowering::prepare()` keeps its cheap whole-ambient check, then tries
`prepareDirect()` before the existing candidate-cover search.

### Recognized forms

The direct adapter derives a conjunction/disjunction from the actual endpoint
set's rows. It recognizes comparisons of:

- an available original induction variable against a constant;
- an available original integer parameter against a constant;
- `upper_bound - iv` or `iv - lower_bound` against a constant;
- a loop/parameter scalar's residue, exposed by a unit equality with one local.

The arithmetic matcher in `LogicalSyncCompactForms.h` recognizes equal or
opposite affine coefficient vectors and safely computes the comparison bound.
It does not scale inequalities, perform projection, solve a dependence, or
pretend unrelated occurrence coordinates are equal. INT64_MIN/MAX behavior is
covered by the production-helper tests.

The adapter never treats a constrained existential as a freely usable floor.
A row such as `x - m*q - r = 0` proposes `x mod m = r`; other constraints on q
remain part of the original wanted set. They can only disappear from emitted
control if the final complete-domain proof establishes equivalence.

Rows syntactically present in every ambient alternative need no new test.
Unknown rows may be left out of the PROPOSAL only. Both

    proposed contains wanted
    wanted contains proposed

must return Proved before the direct path is accepted. A broad interval envelope
alone is insufficient. Thus an omitted hole, predicate, guard, invocation
constraint, or additional quotient restriction cannot authorize an event.

### Machine semantics and cost

Every scalar must dominate the original insertion point. Comparisons must fit
the actual integer type. Difference expressions need an existing full-ambient
range proof before emitting subtraction. A remainder requires a full-ambient
nonnegative numerator and a positive representable divisor. Negative-numerator
cases that lack this qualification fall back to existing guard lowering.

`LoopResidue` is one additional lowering atom, not a loop-pattern recognizer.
It refers to the original IV through the existing occurrence universe. It is
not a new PTO operation/attribute and does not change the payload schedule.

The optional direct proposal gets at most 50,000 existing query work units per
attempt and at most min(500,000, one eighth of remaining work at lowerer
creation) across the lowerer. Actual work is charged back to the unchanged
parent allowance. Refusal/exhaustion of this optional proposal does not poison
the fallback. These counters are not wall-time guarantees for MLIR primitives.

Accepted direct forms return before candidate-cover search. The old search is
not run to rediscover the same accepted expression. Existing code-size limits
and fresh reconstruction remain mandatory. A direct formula may still emit
more or fewer scalar tests than the old heuristic: measure, do not assume.

Trace line:

    logical direct_guards attempts ... accepted ... work ...

### Deliberately unfinished

This is not general Presburger-to-code generation, nor the final compact
predicate representation propagated from the first planning stage. It uses
existing normalized endpoint sets as the integration point. Unsupported forms
retain the current general path. No claim is made that ordinary buffering now
stays compact through the complete pipeline.

## 2. Fitting-pool assignment

Before assigning keys, count logical streams per directed pipe pair.

When the complete population fits in the already qualified eight-key pool, try
unused keys before occupied keys. Prove that stream's own functionality and
consumption-before-rearm once, not once per unused numeric key. A dedicated key
does not make recurring use automatically valid.

When the population exceeds the pool, keep the existing ascending/sharing-first
order. A globally unused-first heuristic could occupy every key before a later
stream needs one, even though merging two earlier compatible streams would have
freed it. Preserving the old scarce-domain policy avoids introducing that new
order-dependent failure mode. It does not make the old policy complete.

Only occupied-key trials union the new matching relation with the existing
family and prove the combined participation/reuse obligations. No trial moves,
widens, or inserts a logical handoff. Failures retain the existing status and
clone rollback semantics.

Trace line:

    logical allocation dedicated ... sharing_trials ...

This policy can use more IDs and can change which complementary static endpoints
coalesce. Therefore static set/wait sites need not be byte-identical. Required
ordering, runtime participation and readiness/release boundaries remain the
acceptance contract. Report key pressure and mechanism counts separately.

This patch does not implement dynamic key functions, target reservations,
backtracking, optional post-feasibility coalescing, or scarcity serialization.
The inherited hardcoded pool is unchanged, not newly justified here.

## 3. Exact incremental completion additions

Let O be qualified inclusive same-lane issue order, H the old handoffs, A added
handoffs, R already reached completion endpoints, and D old unfinished work.
Use relational composition in execution order. New seeds are:

    S = O ; A  union  R ; O ; A
    R' = R union S
    D' = D union S

Continue D' through the new full transition relation O ; (H union A).

Why this suffices: a newly enabled path either begins with a new handoff, or has
an old prefix before its first new handoff. R covers established old prefixes;
D preserves old paths that have not finished propagating. After crossing the
first added handoff, normal propagation explores all subsequent old/new
handoffs. The update does not make source order into completion by itself.

The implementation performs this per existing source scope, retaining full
relations with guards, parameters, and invocation coordinates. It updates a
copy and commits only after every needed query succeeds. Removing/moving a
handoff still uses `withHandoffs()` and resets plan-dependent proof state.

This replaces `state.pending = state.reached`. It does NOT replace general
completion relations with guarded-prefix antichains and does not eliminate all
materialized subtraction. That larger representation change is the next
algorithmic task, not an implied capability of this patch.

## 4. Tests included

`check_compact_updates.py` compiles the actual production header's tests, runs a
finite delta-vs-fresh-search comparison, and optionally drives the existing
native relation executable through eleven completion-update scenarios. The
existing `oahs_focused` runner now passes that native driver, making those
scenarios mandatory when the native gate is run.

Portable helper coverage includes exact/opposite rows, comparison bound overflow,
INT64_MIN coefficients, extreme residue constants, coupled/nonunit refusal,
and exhaustive fitting/scarce key-order checks for pools through 12 keys.
The production pool itself remains eight.

Finite update tests exercise 4,320 warmed/partially explored graph states and
12,961 assertions, using a separately implemented queue-based fresh search.
They check additions after saturation, unfinished old work, repeated additions,
and empty additions. These finite tests are not proofs over arbitrary symbolic
loops or a hardware event interpreter.

The supplied native requests exercise explicit and callback issue-order
providers, fresh tails after old saturation, retained pending work, new source
paths, empty/repeated additions, distinct invocation identities, and a rejected
incompatible-space update that must preserve earlier state. The native requests
were GENERATED, NOT EXECUTED in the preparation runtime.

All existing physical-slot, generated-guard, event-cut, mutation, retirement,
and relation tests remain in the focused gate. No negative test was removed.

## 5. Acceptance on the pinned PTOAS build

1. Apply on the pinned base after the source/hash check. Rebuild affected native
   targets with the existing two-worker ceiling. Run `oahs_focused`.
2. Run unchanged online softmax, Q projection, QK, one buffer, the twelve slot
   fixtures, and ordinary two/three buffering. Do not substitute rotating D3 for
   the ordinary three-buffer input. Keep refusals separate from emitted success.
3. Verify direct-path receipts: some ordinary-buffer endpoint preparations must
   actually report direct acceptance. If not, record their exact unsupported
   row forms; do not count portable matching tests as native effectiveness.
4. Preserve payload/allocation/view/ABI and actual event matching. Compare
   readiness and reclamation boundaries, static sets/waits, barriers by pipe,
   terminal PIPE_ALL, scalar/control work and key pressure independently.
5. Run repeated paired whole-compilation measurements against ordinary
   InsertSync on the same optimized build/hardware, with five paired samples,
   unchanged options/input hashes, and no competing work. Report guard,
   allocation, reconstruction and overall time. The <=2x gate stays binding.
6. Keep device correctness/progress and device timing separate and pinned to
   the resulting exact commit. No old device result qualifies this patch.

If native tests fail, this is a candidate patch failure, not justification to
weaken reconstruction, increase the allowance, hide fallback, or change the
fixture. Keep this branch and resolve the demonstrated failure.

## 6. Next implementation after C1 qualification

Finish compact guarded predecessor/next-use maps and completion antichains for
an explicitly closed affine/periodic fragment. Preserve predicate expressions
as those maps are built, rather than recovering them after projections.

At merges, retain incomparable guarded facts. Multiple readers and partially
overlapping writes are not generally one predecessor function. Fall back to
the existing exact relations when composition leaves the admitted fragment.
Replace the superseded common-case computation rather than run two independent
planners. Broader allocation/MMAD/GEMM work remains behind buffering acceptance.
