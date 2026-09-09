# Exact boundary-query checkpoint

This checkpoint removes repeated boundary-query work and qualifies a small
native slot/guard gate. It does not complete the multibuffering milestone or
the requirement that buffering compile within twice ordinary InsertSync.
The production work allowance remains unchanged. No approximation, legacy
seed, or weaker reconstruction check was introduced.

The algorithms/performance review is in
[oahs-exact-query-cost-review.md](../../../../docs/designs/oahs-exact-query-cost-review.md).
Its next substantive recommendation is compact exact recurrence maps; the
changes here are the smaller, separately testable prerequisite.

## Computation changed

- Boundary conditions are interned by original point and every test field.
  The lowerer owns stable relation nodes; candidate lists borrow them instead
  of repeatedly copying domains. Exhausted queries are never cached as facts.
- Ordinary integer atoms use exact complements of their membership row.
  Floor/modulo definitions stay in each piece, while the union's public space
  has no locals. Integer negation and strict-bound adjustment use APInt.
  Imported general predicate relations retain general subtraction.
- Loop-difference expressions and full-ambient arithmetic ranges are cached.
  Extrema over the changing uncovered endpoint domain are still recomputed.
- Each stream projects its acquisition domain once and enumerates represented
  phases. Unknown coordinates retain the full population, charging that dense
  output before materialization.

The cache is local to one immutable pre-emission occurrence universe. It does
not retain completion receipts across a changed plan or supply constructor
facts to fresh emitted verification.

## Verification

The compiler library and four affected native tools built with two workers.
The final `oahs_focused` CTest passed in **61.52 seconds**. Its artifacts are in
`insertsync-builds/PTOAS-oahs-clean/test-results/oahs/focused-ud5hobed` in the
parent workspace. The native driver's SHA-256 and all commands are recorded
there and in the slot runner's provenance.

The gate includes six strict native slot inputs, three generated-only guard
corruptions, and three boundary-query fixtures, including a negative loop
bound and non-unit step. See [SLOT_QUALIFICATION_RESULTS.md](SLOT_QUALIFICATION_RESULTS.md)
for their physical-access and event evidence.

The boundary hook compares direct atom complements with general exact integer
subtraction, including integer extrema and moduli two/three. It challenges
the same atom at different points in one cache. Warm lookup populations of
8, 32, and 128 require stable borrowed identities, no new domain/range builds,
no new cache entries, and exactly one charged unit per lookup.

Endpoint tests vary 1/4/8 represented streams and 8/32/128 available phases.
Sparse query work must remain independent of unrelated phases. A wildcard
must retain all phases and refuse when its budget cannot pay for enumeration.
The native fanout campaign additionally checks actual production counters:
one acquisition projection and intersection, despite increasing later payload
phases. These are bounded algorithm-population checks, not a universal
linear-time claim.

The first modulus challenge caught an assertion caused by constructing a union
space with local variables. That failed artifact is retained under
`slots-m2/focused1`. The corrected implementation keeps locals in disjuncts;
the complete rebuilt gate above then passed. An initial trace-parser mismatch
was also fixed before final acceptance; failed runs were not counted as passes.

## Preserved native results

The unchanged four-kernel runner passed against the rebuilt compiler, including
payload/allocation/ABI projections, scalar replay, boundary observations,
C++ emission, and reconstruction challenges. Raw results are under
`insertsync-builds/campaign/logical-plan/slots-m2/exact-guards-four1`.

| Kernel | Sets / waits | Body barriers | Terminal `PIPE_ALL` |
| --- | ---: | --- | ---: |
| One buffer | 4 / 4 | None | 1 |
| Online softmax | 12 / 12 | V: 20 | 1 |
| Q projection | 23 / 21 | M: 5 | 1 |
| QK | 17 / 17 | M: 2 | 1 |

These inventories preserve the prior accepted checkpoint. Single PTO-emission
invocations were approximately 0.49, 5.41, 12.98, and 2.49 seconds respectively.
They are not repeated paired measurements or evidence of a speedup.

The broader current-source slot campaign completed all twelve probes: eight
strict constructions passed emitted checks, two declined lowering/allocation,
and two timed out. All twelve dependency discoveries were checked, including
saved sidecars from unfinished construction. Rotating D2 still takes about
48 seconds, and the newly completed additional-reader case takes 87.65 seconds.
This is a failed full campaign, not buffering acceptance. Discovery-only evidence
never counts as an emitted construction or rollback proof. The exact recurrence
and full buffering performance gates remain open.
