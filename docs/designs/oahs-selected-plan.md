# Selected-plan construction: v0.18 handoff implementation

Specification: the synchronization draft v0.18, policies F1–F8.
See [shared semantic extraction](oahs-shared-semantics.md) and
[storage and fixed-plan analysis](oahs-analysis.md) for the input contracts.
`algorithm=handoff` uses this constructor. `algorithm=existing` remains the default
and comparison path; no additional pass mode or legacy fallback is introduced.

## Entry points and state

`constructSelectedPlan(program, fixedWords)` is a production C++ service. It
returns selected commands only after cold validation of the original program.
A failure retains diagnostic records but clears executable commands and the
certificate. `fixedWords` preserve their order and observation identity; reserved
keys remain unavailable. Unsupported typed effects are refused, not erased.

The three objects remain separate:

- `StorageFrontierAnalysis` owns immutable original storage/provenance queries.
- `CausalFrontier` owns the shared must-causal A/T/S/D state and event generations.
- `selected::Ledger` plus source, decision, checkpoint and update records own the
  selected construction. Stable endpoint IDs survive earlier word insertion.

A saved source does not publish anything. Its version and actual original cut
must agree with the current checkpoint map before it supplies a placement query.
The source's prefix is compared with the consumer's current occurrence record;
a later access cannot inherit an older receipt merely by sharing its class.

## Policy map

| Rule | Implementation |
| --- | --- |
| F1 | `SelectedControl.cpp`: original graph, SCC order and original-cut frames; `SelectedPlan.cpp::run` visits each task once. |
| F2 | `CausalFrontier::inspect`, `SelectedGroups.cpp::reasons/consume`: every effect including RMW, no hypothetical acquisition. |
| F3 | `SelectedGroups.cpp::groups/sourceGroup`: known readiness and reuse together, then remaining overlaps; group only one source stream and one target frontier. |
| F4 | `coverage/groups`: strict containment among required providers, then stable source ordering; `consume` rechecks the whole residual after real acquisitions. |
| F5 | `Control::after`, saved `SelectedSource` records and ledger placement preserve separate early source positions. |
| F6 | `consume`: cross-engine repairs first, named fence only for the remaining same-engine residual, then complete payload check. |
| F7 | `SelectedAllocation.cpp`: source-time key checks, ledger intervals, nonrecursive consumption acknowledgment and one shortest eligible route. |
| F8 | `SelectedControl/SelectedReplay`: open choice/loop interfaces and actual fixed-point validation; `CyclicFrontiers.cpp`: qualified recurring requirement frontiers and deterministic role allocation. |

Ordinary endpoint `request` IDs index `decisions`. For a qualified cyclic result,
`channels` instead contains the role requests and endpoint `request` IDs index
that table. Neither table supplies completion to the checker.

## Selected updates

Updates reuse a predecessor-closed unchanged prefix. Each materialized endpoint
invalidates the saved
selected map from the component of its earliest changed word onward and
recomputes every finalized site from there to the consumer from bottom. It
rechecks affected payloads and both sides of each selected key use. No old
ledger-version fact seeds a changed traversal. A failed update stops construction;
there is no alternative-plan scoring, recoloring, retry search or legacy fallback.

Components before the earliest changed word are reused verbatim. The
construction order is a topological order of the strongly connected components,
so those components have unchanged equations and unchanged inputs and therefore
the same least solution. A cyclic component is reused only from a completed fixed
point, never from the hypothesis-seeded traversal of the active component, and
the active component is always recomputed. Invalidation includes every reachable
site sharing an edited observation word, even when its canonical site is later
in control order or unreachable. A reused endpoint aggregate must have all its
occurrences in the reused prefix; otherwise the boundary moves earlier.
Reused endpoint snapshots are copied
with their cuts. `replaySiteEvaluations` therefore counts only recomputed sites;
`selected_update_test.cpp` checks that each update evaluates at most the sites
from its earliest changed word to the consumer.

The frontier's per-primitive closure is incremental for the same reason: the
retained relation is already transitively closed and a primitive adds three
fresh vertices with no edge back into it, so each old row gains only the fresh
vertices it reaches through its gate, prefix, or matched publication. The
differential bridges compare the resulting facts with the exact reference.

Acyclic forward advancement reuses an unchanged predecessor map only when its
ledger version is unchanged. `selectedUpdates`, `replaySiteEvaluations`, normal
`forwardSiteEvaluations`, finalized-query counts and changed cuts distinguish the
work. `elapsedMicroseconds` includes portable model/control/storage preparation,
policy construction and final compact validation. `preparationMicroseconds` is a
subset; native import/emission and external reference checking are excluded.

## Structured construction and proof boundary

Choices keep their original successors and share one word per original
observation. A nonparticipating source is never awaited through a branch-local
publication: when an early correspondence is unavailable, the selected policy
uses the actual current-cut closed protocol, or refuses.

For unrefined reducible loops, construction provisionally retains body access
classes at the header and uses natural-loop exit summaries for surrounding
readers. It does not execute a failed future payload. These construction-only
summaries neither change the program graph nor establish a final certificate.
Completed loop components are checked on the **original** edges with a joined
fixed point from the actual incoming interface. Child exits are open; only the
invocation exit checks token balance and retirement. While-before is not treated
as an optional body. Irreducible/multiple-entry loop interfaces are refused.

Recurring common-cut channels keep distinct stable forward and acknowledgment
roles during a loop component. Compiler role reservations are released after the
component, but physical consumption/occupancy state is not reset. These
conservative protocols can require more keys and add more order than a qualified
readiness/release recurrence.

The cyclic qualifier admits only the draft's isolated two-role exact-slot case:
one pure full write and one read per visit, disjoint slots, one original period,
available original first/steady/final observations, no other payload effects,
no surrounding reuse and no retirement demand. It verifies role alternation,
prior-reader and next-writer facts on the original finite graph. It returns
**requirements and original endpoint sets**, not event commands or key numbers.
The constructor binds those requests in stable cell/role order to the lowest
unreserved eligible keys and validates the selected ledger from fresh entry to
all original exits. There is no dynamic trip-count enumeration or kernel-name
dispatch. The native importer does not yet supply the full-write/slot certificate
required to use this precise specialization; native loops use the generic path
or a classified refusal. A different loop cannot borrow this qualifier's proof.

The must join deliberately loses some disjunctions. The update test contains a
safe pair of concrete branch continuations that the joined state refuses; exact
collection is a development reference, not a production fallback.

## Live native adapter

The selected native entry uses the original SCF graph, including all original
instruction cuts, choices, zero-trip alternatives and loop backedges. It retains leaf loops directly in original control. This representation is selected before construction, without retry:
native import does not yet supply the occurrence/full-write certificates needed
by the selected cyclic specialization. The portable qualified-observation API
remains available. The analysis-only
report can expose a normalized first/tail observation representation.

The original Qwen3 RMSNorm and post-RMSNorm kernels exercise this representation
in native regression coverage. Their unrelated reduction loops require
conservative body hypotheses without inferred first/tail correspondence.

The frontier now represents the existing synchronous-payload contract by putting
the next launch gate after that payload's completion. It does not turn SET into
a source-side barrier. Terminal ALL is allowed only at the original invocation
exit and records retirement without consuming notifications or granting interior
completion/rearming credit. Mixed retired/nonretired inputs retain both a may
continuation restriction and a must exit requirement.

`runHandoffSync`, used by `algorithm=handoff`, invokes `constructSelectedPlan`
and validates reconstructed commands with `checkCausalFrontier`. Both mutation
hooks share this exact implementation, including import, private-copy transaction,
SyncCodegen, and exact command/guard/payload read-back. The optional selected
report describes construction; LogicalResult also includes reconstruction.
Shared fixed-plan diagnostic services remain available independently of
construction.

Preserved queue, atomic-store and hard-collective operations continue through
lowering-owned shared extraction and unchanged original IR. No peer, UnitFlag,
implicit resource, cache visibility or intrinsic-drain completion is invented.
General authored/internal local-event models, selective phases, visibility and
exclusive-resource adapters remain explicit refusals where not represented.
Bounding geometry still does not imply a definite full write; native known
readiness remains conservatively classified under the existing importer.

`pto-oahs-selected-test` tests ordinary/branch/loop construction, retained A3
queue and hard-collective inputs, and transactional mutations through this hook.
Its `--construct INPUT` path writes synchronized IR only after success and reports
construction/reconstruction and work counters separately. It is a developer test
entry, not another production pass mode.

## Validation and remaining qualification

The accompanying portable tests construct from unsynchronized input. The exact
reference bridge checks 108 selected programs, including 100 deterministic
straight-line/choice/loop/nested inputs and four cyclic slot populations. Every
exported causal fact is checked against every reached exact state. Eight selected
examples are also checked by the unchanged paired-order collector; four
straight-line examples additionally use an independent full-history graph.
Cyclic endpoint-deletion tests cover 120 original-observation endpoint mutations.
The reference files remain byte-identical to the pinned repository versions.

Standalone tests build the production library. Native integration builds the
selected and overflow regression drivers. The native drivers are registered in
`check-pto`. Real-kernel replay is reproducible with:

```sh
python test/oahs/replay_prefill.py --build /path/to/native-build \
  --output /path/to/new/report-directory --timeout 60
```

Run with the Python version configured for the native build. This captures all
21 pinned prefill families (23 functions), preserves module-level failures and
timeouts, compares default/explicit `existing`, and lowers successful live handoff
output without a second synchronization insertion. Static inventories and host
timings are diagnostics, not device correctness or performance evidence. This
cohort does not enumerate all PyPTO/pypto-lib kernels.

The live switch preserves shared contracts and conservative F8 control handling.
Precise native occurrence certificates, additional typed adapters, full population
coverage, and device qualification remain separate work. Report all refusals/timeouts and
whole-pass cost rather than treating this switch as production replacement
qualification.
