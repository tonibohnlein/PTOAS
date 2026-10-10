# Regional proof and composition gaps

This work follows Sections 5.4 and 8 of the working-tree synchronization draft.
It extends shared proofs and regional adapters; it does not add kernel-specific
recognizers, implicit frontend assumptions, streaming emission or scarcity repair.

The starting revision is `d5d65f79d33f599f89164ba516adc564ae88c1f4`.
The independently inspected TileLang ledger contains 240 original regions, with
230 exact-demand results under MayNotAlias and 218 under MayAlias. The two-policy
audit, source anchors and diagnostic evidence are in
`.local/section8-refactor/step2-current-d5d65f79d/`.

## Reference and implementation order

Paper hashes at the first milestone:

- `sections/minimum_demands.tex`:
  `ff90d2f2fb202ff06f58ed7cec8df0d1d760eaed7a0ef8d83819a36e635c3da8`
- `sections/combined_algorithm.tex`:
  `3ad4b4fe36fccf2f1792eb60e6b443d9c1663cb4c78fecd976587d0c74dfac4f`

These match the previous audit. Recheck the draft before subsequent milestones.

1. Reuse shared modular recurrence proofs in regional dispatch and phase views.
2. Recover runtime arithmetic with explicit machine semantics and proven ranges.
3. Generalize refresh certificates over overlapping physical storage atoms.
4. Complete parameterized fixed-occurrence interfaces and alias-aware crossings.
5. Validate every original region and report actual constructions and obligations.

Each milestone requires architecture and correctness review before its commit.
Unproved prerequisites, construction limits and unavailable interfaces remain
obligations; none establishes exclusion from every paper class.

## Baseline distinction

The current production CLI intentionally stops after mathematical analysis with
the previously requested not-implemented diagnostic. Consequently it cannot be
used to infer an end-to-end compilation regression from the regional audit.

At the starting revision, the retained library insertion path produces logical
synchronization for PyPTO GEMM, and the allocator succeeds with IDs 0 through 5.
Optimized baseline binaries and logs are pinned under
`.local/section8-refactor/regional-gaps/baseline/`. C++ generation uses the pinned
code-generation-only emitter, with synchronization and memory planning disabled.

## Modular recurrence proof boundary

The shared scalar evaluator already proves constant-seeded additive modular
recurrences. The regional index now retains that same proof as an immutable
description in the original loop's zero-based ordinal. It records the seed,
stride and modulus only after proving each machine update cannot wrap.

Regional admission asks whether relevant carried arguments have such proofs.
The original relevance and scalar-prerequisite traversal remain unchanged.
Relevant escaping loop results still require a separate exit/zero-trip
interpretation and are refused. Unknown seeds, unsupported recurrences and
unproved machine updates cannot become independent execution parameters.

Phase specialization uses the certified period `modulus / gcd(modulus, stride)`
and evaluates representatives with 128-bit modular arithmetic. It does not
substitute a potentially overflowing closed-form product into affine constants.
Proofs belong to one index/input context and are cleared when that index is rebuilt.

The tests compare small bank sequences with independently stepped updates and
large representatives with a wider integer reference. Command-closure checks
use independently authored byte conflicts, including nonunit original loop
coordinates, zero/partial trips, proper slot orbits and nested recurrence resets.
An additional probe makes the inner domain depend on the carried bank, requires
actual phase descriptions, and inserts that same sequence result before checking
its command closure. Other cases exercise the normal dispatcher independently.

### First milestone validation

The optimized candidate uses `-O2 -UNDEBUG` and C++17, matching the pinned
baseline. The source mirror's changed C++ and build inputs were checked against
the reviewed working tree before validation.

All 22 serial corpus requests completed: MayNotAlias improved from 230/240 to
232/240 original regions; MayAlias remained 218/240. There was no lost region or
new corpus timeout. PyPTO GEMM now certifies all 17 regions under MayNotAlias:
its independent K loop selects regional repetition, and its containing sequence
selects regional sequence. Both provide queries and selectors. This does not
resolve their MayAlias storage-composition obligations.

The candidate's full library preparation, allocation with IDs 0 through 5 and
C++ generation all succeeded for PyPTO GEMM. Generated C++ is byte-for-byte
identical to the optimized baseline. These are compiler checks, not device runs.

Artifacts, binary and source hashes, exact commands, timings and every regional
outcome are under `.local/section8-refactor/regional-gaps/m1/`.
The new recurrence and command-closure checks passed. Existing tests that checked
general repetition were updated to permit either valid repetition provider;
phase-specific coverage remains mandatory in the explicit probe. Library tests
use the retained library adapter because the public CLI intentionally stops at
Step 2.

The composed-phase, periodic-inner-length, arithmetic-recurrence, carried
relevance, guarded-rotating and sequence checks passed. The broader phase-owned
storage and rotating library suites each reached a 180-second external watchdog
on both baseline and candidate. Those suites are not reported as passing; the
paired limits neither establish a new regression nor classify any input.

The changed-code prefilter reports eight brace findings. Each is a reviewed false
positive: its regular expression backtracks to a nested call's closing
parenthesis, while the actual balanced control condition has a braced body.
The raw scanner result and independent brace adjudication are retained separately.

## Occurrence-local scalar range contexts

The shared scalar evaluator can now refine integer ranges from guaranteed
ancestor `scf.if` arms at an executed occurrence. Signed comparisons, swapped
operands, false-arm inversions, true conjunctions and false disjunctions retain
their machine widths. Unsigned intervals crossing the sign bit provide no
signed refinement. Unsupported predicates provide no facts. Contradictory
supported facts discard the refinements instead of manufacturing expressions.

Facts and both successful/failed range caches belong to one evaluator context.
Range-only queries inherit that immutable context without publishing placeholder
symbols into expression caches. Refinement follows operation admission: a
positive result does not prove that its producing addition was nonwrapping.
Arithmetic access normalization uses the consuming phase's context. Presence
guards and loop-domain formulas use their scalar definition contexts, so they
cannot assume the occurrence whose presence they are reconstructing.

The prefill loop is outside the earlier `length > 0` prefetch arm. That sibling
condition does not constrain its runtime bound. With literal 64-bit index
semantics, start=0 and end=-5 give unsigned chunk count `2^62-1`; ordinal `2^61`
is an executed visit and multiplying it by four wraps to `INT64_MIN`. The
remaining-length subtraction then wraps to `INT64_MAX-4`. No implicit valid
sequence-length contract can remove that case. This remains a machine-arithmetic
representation obligation, not evidence that no tractable class applies.

Focused checks cover both index widths, branch/root isolation with identical SSA
values, signed extrema, unsigned sign boundaries, Boolean alternatives,
contradictory paths, guarded chunk arithmetic, both cache query orders, and the
literal prefill witness. An internal arithmetic-builder check distinguishes
occurrence-valid unsigned division from its unavailable control-domain formula.

The optimized build, all 40 range cases, the interleaved cache and arithmetic
builder checks, and six focused existing arithmetic/recurrence checks pass.
The final serial two-policy corpus audit preserves all 480 regional outcomes:
232/240 under MayNotAlias and 218/240 under MayAlias, with no new timeout.
No command fragments or allocation are constructed by those demand requests.
Artifacts and source/binary hashes are under
`.local/section8-refactor/regional-gaps/m2/`. The changed-code prefilter reports
seven brace matches; independent balanced-condition inspection confirms that
all seven actual control bodies are braced. Raw findings and adjudication are
retained separately.

## Outstanding proof obligations

Prefill requires literal signed/unsigned reasoning from loaded sequence endpoints
through chunk bounds and tail arithmetic. No ordering of endpoints is assumed.
Gated delta requires a complete physical refresh/span certificate including every
potential write, guard and scalar prerequisite. Its per-iteration floating-point
guard is not an integer-arithmetic predicate.

MayAlias composition must retain complete shared modeled overlap requirements.
Missing symbolic-byte selectors or fixed-occurrence relation adapters must not
discard child demands. A bounded-window demand circuit supplies local rows;
those rows must never be promoted to whole-loop reachability without a proof.
