# M3a: actual packet bundles and joint replay

Normative specification: `main(20260915-194549).pdf`, identical to the PDF in
`synchronization_theory_acm_v0_8.zip`, sections 5, 8.4, 10 and 11.2.
This change extends the existing handoff constructor; it adds no planner mode,
semantic admission rule, kernel recognizer, or numerical analysis-work budget.

## Query contract

`PTO/Transforms/OAHS/Bundles.h` supplies an owned immutable `BundleQuery` over
original `Program` and current `Commands`. `evaluate(trialCommands)` checks the
complete replacement command population against the SAME original program. It
reports the certified M1 analysis, all-source/all-cell residual differences,
protocol and retirement differences, exact changed cut words, and physical key
and command use. Boundary states are available unless captureStates=false.
Malformed/reserved commands give incomplete analysis. Completed replay can still
have obligations. Only `analysis.verified()` accepts a plan. No supplied source
label, prospective cover, graph edge, or cached credit is trusted.

A helper can have zero memory credit yet resolve a protocol obligation. A return
can cover both WAW and WAR while leaving later unrelated source work pending.
There is no independent weight cached for a command or hazard kind. Protocol
report differences compare (cut, direction, key, kind) with multiplicity: inserting
commands can change offsets. The complete report retains exact endpoint offsets.
These report differences are diagnostics/selection information, not proofs of
open-interface equivalence or of added-order neutrality.

## Production policy

A repair slot contains an ordered bundle of directional legs, rather than exactly
one forward handoff. The first leg may publish at several original physical cuts;
all of them feed one acquisition. Later legs acquire and republish at the same
consumer cut in their actual causal order. Target lanes without payload memory
are retained. Discovery enumerates all shortest eligible topology paths. Keys are
chosen deterministically, preferring unused keys; reuse remains a proposal subject
to the same complete protocol analysis. This finite policy is not exhaustive over
all key assignments or longer routes.

M2 individual-pair candidates remain valid proposals. In addition, a backward
predecessor frontier proposes the immediately preceding physical cut on each arm.
It does NOT assume that these sites are exclusive, exactly participating, or cover
a preceding generation. Whole-bundle replay establishes those properties or
rejects the proposal. Discovery does not cross a backedge to invent an occurrence
mapping. A branch's final relevant write may need a not-yet-supported region-exit
cut; ordinary common-cut construction remains available in that case. No dummy
payloads or executable guards are inserted.

Every cross-engine trial is rendered with all prior bundles, reserved/assigned
keys, actual acknowledgments and terminal retirement before it is evaluated.
Selection uses certified residual credit at the consumer across ALL sources, then
newly acquired unrelated phase completions at that consumer, then static endpoint
count. This is an explicit local quality heuristic, not a global order dominance
proof or latency model. The M3b paired reference measures actual added order.
The same-engine fence path remains the ordinary residual repair after relevant
cross-engine transfers have been considered. BundleQuery can independently replay
and compare local/coarse fence alternatives; this is not a complete optimizer over
incomparable fence positions.

As before, partially built recurrences can have unproved endpoints. The private
construction-only transfer can supply a pending repair step when no certified
trial currently makes memory progress. Such provisional information is NEVER
BundleEvaluation credit or public AnalysisResult established completion. The
remaining obligations survive; actual acknowledgments are added only in response
to rearm failure, after existing storage returns are present. Helper edits are
replayed as complete actual candidates too. Final public verification is mandatory.

## Finite progress, no work allowance

There is one slot per (consumer, motivating source), at most N*P. A chosen bundle
has at most P-1 shortest-route legs. Each leg can relocate its publication once
and gain at most one direct acknowledgment. A frontier has at most N original
cuts. Candidate enumeration is finite; an unsuccessful or unchanged repair
terminates or uses the explicitly checked conservative realization. No loop can
retry a slot without consuming one of these finite changes. The prior simple
3NP edit bound is replaced by O(NP^2) successful slot/leg edits. The bound is on
edits, not total running time: full-candidate replay and candidate populations can
still be expensive. Real key scarcity is not an analysis budget.

## Acceptance cases and remaining limits

The tests require actual construction of (1) a payload-free two-hop route with
no direct direction and no ALL, and (2) branch-alternative publications before
unrelated trailing work with one common acquisition. The existing joint WAW/WAR
return must retain early readiness and leave unrelated work pending. Tests cover
helper-only protocol credit, missing branch publication, stale snapshots, immutable
query sessions, reserved routes, and repeated/branching generated inputs.

Native FileCheck coverage is supplied for branch-alternative publication at
existing physical cuts. Full native compilation and device execution must be run
in a configured PTOAS checkout; standalone results do not establish either.

This milestone does not add arbitrary control-edge cuts, slot correspondence,
new predicates, macros, authored transfers, visibility, resource interlocks or
extra target assumptions. It does not prove compact-domain simulation. A failed
route search is explicitly reported as construction-policy failure, not global
target infeasibility. No source theory files are silently rewritten by this patch.

## Native emission identity

Reconstruction additionally requires exact per-cut ordered command words before
accepting the result. The test hook inserts a memory-safe extra terminal ALL and
requires transactional rejection. This separates protocol identity/quality from
merely passing the hazard checker. Shared codegen still has legacy coalescing
behavior; if a future candidate requires a word it cannot preserve, native
emission rejects rather than silently changing that word. The adapter must then
be extended explicitly; this gate is not claimed to prove full native coverage.
