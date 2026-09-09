# Exact recurring-query cost checkpoint

This increment reduces the cost of exact occurrence calculations. It does not
complete the ordinary-buffer compilation target, expand target semantics, or
claim device qualification beyond the separately recorded cdaae8d0f campaign.
The constructor budget and emitted guard limits are unchanged.

## Algorithms

Materialized difference and Boolean containment now share one qualified
first-violated-constraint partition engine. Difference retains breadthwise
materialization. Boolean containment traverses the same exact partitions with
an explicit depth-first stack and stops only at a feasible witness outside
EVERY relevant right-hand disjunct. Full coverage requires exhausting every
branch. Total floor definitions remain separate from membership constraints;
arbitrary existential witnesses are never complemented as if they were floor
functions. Inequality implication work is deferred until equality branches
have been exhausted. Matrix alignment, copies, partition storage and stack
work are charged before allocation where their dimensions are known.

This avoids constructing an entire uncovered union for a Boolean answer. It
is not a linear-time theorem for general Presburger containment. In the
2/4/8/16-dimensional orthant tests, the Boolean query emits one partition
piece while materialized difference emits 2/4/8/16. Hard coverage cases can
still explore many branches, and MLIR operations are not bounded in wall time
by the query-work counter.

The native periodic adapter also normalizes individual integer cells exactly.
It uses GCD normalization, opposite-row equalities, unit-equality substitution,
and integer-exact elimination where one whole inequality frontier is unit.
A protected quotient and an explicit divisibility anchor preserve periodic
occurrences. Unsupported coupled/nonunit elimination retains the original
cell. Empty results require an exact contradiction; exhaustion supplies no
partial result. Caps bound rows, columns, local count and coefficient width.

Complete rewritten-or-original cells form an equivalent whole population.
The separately constructed interval envelope remains only a proposal. Both
containment directions must establish its equality with that whole population.
This avoids re-expanding eliminated existential witnesses during qualification;
it does not drop their semantics. Fresh emitted-IR reconstruction repeats the
qualified computation against actual event occurrences.

## Validation and limits

The final relation suite passes 320 native/libisl checks. The occurrence suite
passes 63 cases and retains 1,936 unchanged-softmax phase-pair checks. Tests
cover hidden next-use domains at periods 2, 3 and 2,147,483,647, negative
quotients, implicit odd congruences, free and one-sided locals, indivisible
equalities, empty and mixed populations, unavailable elimination, output-growth
refusal and work exhaustion. Phase-specific interval mismatches must fail the
whole-population equality gate. All three hidden-next cell queries use 3,778
work units independently of period; the largest full adapter query uses 4,650.

One earlier test attempt failed because the Python oracle generated invalid
ISL syntax. A later large-period adapter attempt timed out at 60 seconds while
rechecking unsimplified existential cells. Both failures are retained. The
final test keeps the large-period adapter enabled and passes after exact
whole-population normalization. An implicit odd-domain admission failure was
also fixed by recovering exact opposite-row equalities before quotient choice.

The four established kernels pass strict construction, C++ emission, immutable
payload/allocation/view/ABI projections, scalar replay and required boundary
checks. Static inventories remain one-buffer 4/4, softmax 12/12 plus V20,
Q projection 23/21 plus M5, and QK 17/17 plus M2, each with terminal PIPE_ALL.

Diagnostic probes before the final implicit-congruence test fix record:

| Input | Result | Invocation seconds | Work |
| --- | --- | ---: | ---: |
| Two buffer, accepted adapter baseline | applied | 28.32 | 20,524,351 |
| Two buffer, exact-query worktree | applied | 21.71 | 17,286,091 |
| Three buffer, exact-query worktree | applied | 117.17 | 54,496,922 |

The two-buffer emitted IR is byte-identical to the accepted adapter baseline.
All 20 periodic successor queries apply, each below 6 ms in that diagnostic.
These are isolated, unpaired probes, not the required five-pair performance
gate. Three-buffer succeeds with fresh reconstruction in the longer 120-second
diagnostic, after the earlier 90-second probe timed out. That timeout change
is a measurement limit, not an increased compiler allowance or accepted cost.

Two-buffer still spends about 8.03 s in guard preparation and 8.31 s in
reconstruction. Three-buffer spends 22.76 s in guards, 40.17 s in allocation
and 35.81 s in reconstruction. Materialized subtraction remains expensive in
completion-frontier expansion. Primitive timers are nested and must not be
summed. Boolean fallback time now appears under containment plus shared
partition/qualification timers; fewer public subtraction calls alone do not
establish a speedup.

The <=2x existing-compilation gate remains OPEN. No allocation/MMAD/GEMM
milestone is accepted by these results. Next exact candidates are direct
relation-derived guard proposals and demand-driven completion-frontier work,
subject to the same independent checks and three-reviewer gate.

Artifacts are under the parent workspace's
`insertsync-builds/campaign/logical-plan/overnight-20260910`:
`relations-boolean-2`, `occurrences-cell-4`, `exact-cost-four`, and
`native-exact-cost-1/2`. Source is a worktree based on e2a02e6ce, built with
Release -O1 and assertions using the pinned MLIR19 build. The separate
153-test default/shared campaign passed on recorded pre-change binaries with
unchanged hashes; it is not attributed to this pending algorithm build.

The automatic `oahs_focused` CTest gate passed in 72.35 seconds at
`test-results/oahs/focused-ftu392ps`. It ran concurrently with the serial
four-kernel correctness campaign; that duration is validation evidence, not
a controlled performance comparison. Binary hashes are retained in
`exact-cost-binary-hashes.json`. The final native driver SHA-256 is
7a2cd2846432d40816243c4eebb1558e35c805bf5acacd179147c2ec3ddb81cd .
