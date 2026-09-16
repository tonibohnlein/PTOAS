# M6: incremental replay, immutable source indexes, and resource alternatives

Prerequisite: the exact packaged M4 + M5 source stack (`oahs_m5_on_m4.zip`).
This is the first executable M6 scaling/quality increment, not completion of native
coverage, phase qualification, device evaluation, or the compact-domain proof.
The normative architecture is the 43-page *Storage-frontier analysis and certified
event construction* draft and the post-M3 roadmap. There is still one constructor.

## Fixed-plan replay

`Replay.h` adds a move-only `ReplaySession` owning one immutable original `Program`.
Its public `analyze(commands, options)` accepts a complete actual command population.
It reports the same semantic result as cold `analyze(program, commands, options)`.
`complete` means analysis completed; only `verified()` establishes synchronization.
No supplied coverage, state, or certificate is trusted. The session is not thread-safe.

The session validates the owned program once and checks every candidate's commands.
It stores one last complete report and one last **unsuppressed provisional** compact
fixed point. Public reports and prior caller values are copies, never borrowed mutable
state. Identical commands and reporting options can reuse the report; invalid commands
do not replace the checkpoint. Changing reporting options is not a precision change.
Changing any original effects, control, observations, reservations, target or invocation
contract requires a new session. `clear()` discards cached computations.

For a changed candidate, compare the full ordered word at every cut. Let D be all
original control sites forward-reachable from a changed cut. D includes loop backedges,
so a later lexical edit can invalidate an earlier header and its whole reachable loop.
All D input states restart at unreachable bottom. U = sites minus D is predecessor-
closed; retain its fixed-point states and replay its boundary contributions into D.
Then solve D to convergence. States use copy-on-write storage so retaining a prefix does
not deep-copy every old state for every proposal.

### Why this does not keep stale completion

The original program and graph are fixed. No changed command, and no path from a
changed site, reaches U. Thus all equations defining U are unchanged and have the same
least solution. Solving D from bottom with U's fixed boundary contributions computes
the same least fixed point as the complete system. This argument depends on the same
monotone finite transfer contract as cold M4 analysis; it does not prove that contract
from hardware. In particular, **do not seed D with its old joined state**: deletion or
motion can remove completion, acknowledgment, and receipt provenance.

The checkpoint's key-to-index map must agree exactly. A changed set or first-use layout
of keys triggers cold replay. The engine/event correspondence cannot be inferred from
an equal vector length or a key number at one endpoint.

Endpoint certificates are discovered afresh for each candidate. Only the first,
unsuppressed solve can reuse a checkpoint. Every subsequent certificate-revocation
pass starts cold from the original entry. The finite M1 revocation procedure therefore
cannot preserve invalid credit through a different relay key or retain old suppression
when an endpoint becomes legal again. Successful revocation results are not fed back as
provisional seeds. The final `verify()` still runs the independent cold analysis.

A phase-bearing program uses the supplied M5 collecting domain. Changed phase plans
receive full phase replay. An exactly identical complete phase report can be cached;
ordinary compact state is never substituted for phase/resource state. No native phase
assumption is enabled, and no full collecting state is merged with another history.

## Constructor integration

`BundleQuery` uses one replay session for its trial population. Discharged/introduced
requirements are always compared to its immutable base plan, NOT to the previous
trial. A rejected trial never commits anything to the caller's plan. Captured states
remain a reporting option; final legality is unchanged.

Private construction proposals also reuse an unsuppressed fixed point through a
separate `ProposalReplay`. This retains the pre-existing ability to build a partial
recurrence before every event precondition is proved. Its facts cannot become a public
`AnalysisResult` or an acceptance certificate. Public trial replay and final cold
verification remain distinct. Their work is counted separately.

`ConstructionOptions::incrementalReplay=false` is a developer differential-test
ablation. It changes computation reuse, not the declared program or candidate policy.
No public pass/planner mode is added.

## Immutable prefix indexes

`PrefixQuery` remains tied to one owned program and one actual command population.
It now indexes canonical observation members and consumer requirements once, and lazily
caches backward cut reachability, source remainders, and source/target balance/freshness
queries. The latter monitor is independent of the source engine; several lane proposals
can share that traversal without sharing completion credit. Ordered reports preserve
the previous per-cut residual ordering, including phase collector reports.

No entry is reused after the query's actual plan changes: construct a new query instead.
A saved prefix is still not an event; only a valid actual acquisition establishes credit.
The caches use full structural keys, not an unchecked hash/fingerprint. Their populations
are finite and grow with the requested cuts/pairs. Querying every consumer can still be
expensive; this is not a universally linear storage or time claim.

## Broader finite resource policy

The constructor keeps shortest routes first and unused directional keys preferred.
It can additionally try every already-used eligible key and one representative unused
key per direction. That unused-name symmetry uses the CURRENT fresh-entry, direction-
pool eligibility contract: no authored use of an otherwise unused key is omitted.
Future key-specific contracts must revisit this reduction.

After a preferred complete construction attempt fails, the relay/original stages can
restart from the original candidate and enumerate all simple eligible engine routes.
This includes engines with no payload effects, and a route can be longer than an
available direct direction. Key assignments are visited lazily, not materialized as a
Cartesian product. A protocol-correct positive candidate on a shorter route ends
exploration of longer routes for that particular proposal. This is an explicit finite
quality policy, not complete resource synthesis or proof that the skipped routes are
inferior in whole-program payload order.

A private pending recurrence is NOT enough to prune longer routes as certified. To
avoid expanding every route while an ordinary recurrence is merely unfinished, the
preferred attempt runs first; expansion restarts only after that attempt actually
fails. Restricted storage stages retain their endpoint/pair restrictions; later stages
widen them. Failed endpoint/key commitments are rolled back before every new attempt.

For an unproved consumption-before-republication requirement, first try a different
eligible key by full actual replay. A rekey is committed only if that complete plan is
already verified. Otherwise try a real return path, including multi-hop paths through
eligible engines, at the actual acquisition sites. Every reply publication/acquisition
and key belongs to the same plan and verification. Commit a reply only for an identified
resolved consumption obligation with no new memory/resource/protocol obligations.
Storage release and readiness consumption are never conflated; the reply may precede
the following reader and then supplies no release of that reader.

Each forward leg can relocate once and acquire one reply route. There are at most
N*P motivating repair slots, with at most P-1 forward legs and at most P-1 reply legs
per forward leg; all route/key alternatives for an attempt are finite. A rekey commit
already completes a verified plan. The finite named preferred/expanded/recovery passes
therefore do not introduce retry cycles or a numerical work allowance. The route/key
choice population remains combinatorial in general; no polynomial bound is claimed.

If expanded choices fail, restart the retained pre-M6 resource policy from the original
program, rather than keeping failed new commitments. This `BaselineRecovery` is the
same handoff constructor's original policy, NOT legacy InsertSync or another backend.
It can choose the already-checked coarse realization where the target permits it.
This makes the prior constructive acceptance available while allowing explicit gains.
All final output is still cold verified. `resourceAlternatives=false` is a developer
comparison policy, not a public planner selection.

## Work and quality evidence

`ReplayStats` labels cold, incremental, unchanged, key-layout, phase-full, disabled and
invalid cases. Invalidated sites refer to the initial provisional solve; certificate
revocation can add full passes afterward. `reusedSites` counts retained reachable
states for changed plans, or the static-site population of a reused whole report. It
is not always complementary to `invalidatedSites` because unreachable bottom exists.

`ConstructionWork` separates private proposal evaluations from public trial evaluations
and records rekey/route/reply trials. `routedReplies` counts chosen reply routes,
INCLUDING a direct one-leg reply. These counters omit import, prefix-query initialization,
final verification, and inner bit operations, so they are NOT total compiler work.
Stage totals and final totals retain discarded attempts. Final command/handoff fields
still describe only the returned plan.

`oahs-m6-benchmark` measures complete synthetic host transactions and construction
ablations. Fixed-plan hot timings include session setup. Complete report comparison is
outside the timing interval. Cold controls use plain public `analyze`, not an artificially
slow cache wrapper. Five alternating fixed-plan samples and three alternating constructor
samples are recorded. No timing threshold is asserted by CTest, and no event-count
reduction is claimed to imply device acceleration.

The tests compare every semantic report field (excluding work counters) with cold
reanalysis over edit streams; they also use the independently constructed finite graph
oracle and the unchanged v08/v010 reference bridges. Narrow core route/rekey cases use
all-path reference collection, not a finite iteration bound. Phase cold controls remain
separate from the compact-replay measurements. Vendor source and certificates are not
changed by this milestone.

## Scope left for later M6 work and production qualification

This increment does not implement collecting-phase incremental solving, arbitrary local
phase decomposition, compact-domain coarsening, open-region replacement, general global
key reallocation, or exhaustive non-simple relay protocols. It does not prove a better
payload order for every input. The added routes can yield incomparable plans; the paired
reference remains the quality oracle for its supported contracts.

No native lowering or target assumptions change. M4/M5 native compilation, full operation-
semantic coverage, and actual device validation remain outstanding external gates. Native
phase credit stays disabled; service order, block geometry/layouts, visibility, queues,
retirement and hardware agreement are not inferred from test-only reference success.
