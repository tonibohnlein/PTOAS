# Structured OAHS S5: guard admission and checked local refinement

## Base and decision

Apply after the exact S4 `OAHS_S4_Ordinals_Initialization` package
(patch SHA-256 `4d7c1d68c7aec920d439fbc30b55131d2da49cd3aadbe91c7935c9d5afe0d731`),
which applies after S3 commit `60f131ab29fb0db610aaca7a1a0ef75f191458ff`.
This is a review/hardening increment, not another solver, pass, selector, target
profile, or relaxation of the physical-access contract. The default remains
`existing`. The native MLIR adapter and device execution still need qualification.

S4 is the right architectural step: internal ordinal coordinates preserve the
original payload loops; first-iteration effects have a separate finite phase;
empty loops do not execute fictional initializers; and all seven original
population inputs are back in the native gate. Keep those changes. S5 fixes a
source-level guard-admission defect and adds narrowly scoped refinements.

## 1. A common period is not a truth partition

S4's `PeriodicScalar::computePeriod` can establish the effective period of both
operands of an integer comparison without introducing their comparison's truth
boundaries. `NativeFacts::build` then evaluates the guard at the left endpoint
of a residue interval and uses that truth value for the entire interval.

A concrete witness is the original loop `iv = 6*k` with

```
left  = iv mod 4
right = iv mod 12
active = left != right
```

Both operands have effective period two. The active values at ordinal residues
zero and one are respectively false and true: `(0 != 0)` and `(2 != 6)`.
Neither operand is a literal, so S4's literal-threshold code contributes no
comparison cut. The whole `[0,2)` interval can therefore be classified inactive.
When all payloads are in the true arm, the importer can erase those payload
occurrences from its synchronization model without altering the original IR.
A snapshot or a second physical translation does not detect that omission: both
still see unchanged physical operations. This is a source-level correctness
finding, not an observed native/device failure in the preparation environment.

S5 refuses non-Boolean value/value comparisons with a nontrivial effective
period until an exact partition is implemented. Boolean comparisons remain
supported because their operands already establish truth-change cuts. A
period-one phase remains constant. Existing residue/literal comparisons and
qualified first-iteration affine eq/ne predicates keep their S4 handling.

There is no enumeration over a numeric modulus, no use of a residue as a full
induction value, and no new fallback. Two native refusal fixtures pin both the
co-periodic witness and a period-three value compared to a nonliteral
period-one selector. They assert the guard-admission diagnostic, not just any
failure. Portable tests check the arithmetic witnesses; they do not claim to
execute the MLIR importer.

## 2. Post-barrier handoff refinement

`refinePeriodicHandoffs` takes an already allocated and verified local periodic
plan. It makes exactly one reverse sweep over the originally selected handoffs.
For each candidate, it removes that pair and rebuilds checking from the actual
remaining actions and the original immutable requirements. A deletion is
accepted only if both memory/resource completion and event consumption-before-
rearm still pass. The initial input is also checked, including malformed
endpoints on an empty demand ledger.

Surviving publications, acquisitions, keys, relative command order, and all
barriers are unchanged. The routine does not recolor, move waits, broaden
publications, add acknowledgments, or delete barriers. It does not repeat the
sweep to an optimization fixed point. Work is one original-plan check plus at
most H trial checks for H handoffs; each check uses the existing finite model.
This is not a claim of near-linear compilation or a minimum-event optimum.

The native adapter calls this only for a complete, unwrapped, periodic body.
Boundary/startup models and re-entrant invocation models retain their S4 plans.
In particular, refining the local part of an S3 plan independently could break
its cross-invocation proof and its stable-shape reallocation contract. S5 does
not do that. There is no new public switch.

The hand-transcribed `four_use` physical-conflict model still initially selects
19 pairs and two named barriers. The new routine deletes one pair, preserving
18 pairs and the same two barriers/keys. With the separate terminal drain,
that corresponds to 41 versus 39 emitted synchronization instructions. It is
an abstract core result, not native kernel or device evidence. The native gate
now checks at most 18 sets/18 waits and at most two named barriers for unchanged
`four_use`, alongside preservation and token tests. A native mismatch is a
real gate failure to investigate, not permission to reduce the fixture.

A negative control has a reverse handoff needed only to acknowledge a recurring
publication. Its memory demand remains covered after removing that handoff,
but its event protocol does not. S5 keeps the handoff. Random physical-conflict
models check deletion, survivor subsequences, unchanged keys/barriers, and the
existing separate finite asynchronous oracle.

## 3. Preserve verifier failure severity

S4 reclassified every failed final invocation verification as allocation
failure. This could mask a missing requirement or an invalid progress proof and
cause the native adapter to try its alternative allocation policy.

S5 records typed `InvocationFailure` values for missing completion, cyclic
progress, and event recycling. `InvocationResult::constructionStatus` maps only
`InvalidPlan` with `EventReuse` to `AllocationFailure`; all other statuses are
preserved. Structural failures with no discriminator likewise retain their
original status. Native `InvalidPlan -> InternalError` propagation is unchanged.
Tests exercise actual missing-completion and event-reuse results and explicitly
check the progress classification. There is no string-based error parsing.

## 4. Build exit views only for exit obligations

Bulk region construction and coverage now build last-occurrence views only when
an original requirement targets an epilogue atom. An unrelated epilogue no
longer triggers all residue-tail closures by itself. This does not omit
requirements or alter mixed-event checking.

The public `supplies` query deliberately keeps building the needed exit views,
even if the requested requirement is absent from the model's ledger. A
regression checks a missing prelude-to-epilogue requirement on an empty ledger;
returning true vacuously would be unsound. The same query with a valid handoff
must still return true.

Synthetic no-hazard static-size measurements isolate this bookkeeping change;
they are not the <=2x end-to-end compiler campaign. Dense first/last tables,
common-period restrictions, and global atom-pair enumeration are not claimed
to be solved by this patch.

## 5. Diagnostics and qualification

The native trace now reports actual refinement trial/removal counts. Its
hard-coded `presburger_queries 0` text is removed; no replacement pretends to be
an instrumented relation-call counter. A genuine forbidden-entry/dependency
regression remains future work.

Portable S4 and S5 tests are source-pinned. The S5 core/header/test compile under
Clang 17 (-O2) and GCC 14.2 (-O1), both with ASan/UBSan and strict warnings.
The preparation run preserves S4's local and invocation populations and adds
162 refinement-model checks, 423 deletion trials, and 68 accepted deletions.
Those are model/test counts, not device kernels or distinct benchmarks.

The new native adapter lines and two refusal fixtures were NOT built/executed
here. The all-seven native gate, repeated paired compilation, and exact-commit
device correctness/progress/performance remain required. Neither this document
nor passing the shared mathematical checker is an independent hardware proof.
Do not promote the default planner or weaken the unchanged corpus for S5.
