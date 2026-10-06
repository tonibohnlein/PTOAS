# Compact repeated regions: explicit phase composition (M3)

The q-phase route preserves the original loops. It builds q specialized child
interfaces, composes one complete period, and computes its guarded port closure
once. Its cost depends on those phase descriptions and ports, not the runtime
trip count. The supported explicit phase count is at most 256; this is a
representation limit and does not establish complexity polynomial in log(q).

For T visits, the query uses ceil(T/q) periods and excludes nonexistent events
in the final prefix. Boundary selectors instead combine floor(T/q) complete
periods with the first T mod q phase views. A phase without an access leaves the
last matching access from the preceding complete period intact. Zero trips
produce absent selectors and disabled endpoints.

Endpoint recipes remain attached to original IR cuts. Repeated member identities
use the period coordinate; per-type frame divisors and phase predicates bind it
to the original loop ordinal. Both SET and WAIT guards exclude a counterpart in
the discarded final suffix. Enclosing sequence/repeat interfaces retain these
coordinate maps and guards.

## Producer contract and current implementation

The public phase constructor accepts exact regional phase views sharing one
expression arena. The implemented producer adapter specializes a recognized
compact guarded-rotating inner loop inside an outer loop. Other surrounding
operations must be payload-free. A copied analysis is specialized before its
callbacks are created: caller-supplied query coordinates are never substituted.

Relevant constant bank counts determine q by checked LCM. Nonnegative index IVs
and positive constant steps are required. Add/multiply normalization is admitted
for power-of-two periods, where low bits commute with fixed-width wrapping.
Nonpower periods do not receive an implicit no-overflow assumption. Signed
remainder is admitted only for the proven nonnegative outer IV. Outer-dependent
bounds, guards, descriptor rebinding, and unavailable phase specialization remain
explicit unmet interfaces. Every modeled writer must remain exported across
outer re-entry; an inner-loop discharge alone cannot justify dropping it.

The existing flat and successful numeric allocation paths remain available.
This milestone does not supply generic evolving storage-family selectors or
finite-ID allocation for the compact symbolic repeat.

## Validation targets

The native phase oracle independently unfolds tiny q=2/3 regions with guarded
and absent phases, compares all payload-event reachability answers, and checks
physical first/last selectors. The executable command oracle covers empty,
partial, and complete periods; zero inner trips; enclosing q=1 frames; nonzero
lower bounds and nonunit steps; and missing writer re-entry interfaces. Large
trace arguments check that emitted loop structure remains compact. Actual test
results belong in the milestone validation record, not in this design contract.

The absence of trip expansion does not imply small scalar endpoint code. Nested
query composition can produce large guard circuits, especially when external
work queries a phased region through another repetition. The combined executable
stress case is retained with an explicit, bounded test-only scalar interpretation
ceiling; the normal interpreter ceiling and event/depth limits stay unchanged.
This validates its emitted command relation without claiming that its scalar
code size is satisfactory for production kernels.
