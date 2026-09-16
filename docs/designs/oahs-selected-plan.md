# Selected-plan construction: v0.18 handoff implementation

Base: PTOAS `42428352acde3f6e26afe55c4227d5568201eaf4`.
Specification: the supplied v0.18 draft and `OAHS_HANDOFF_42428352a.md`.
No public pass option or live-driver cutover is included.

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

This is a dependency-complete **cold-update baseline**, not an incremental
invalidation implementation. Each materialized endpoint invalidates the saved
selected map and recomputes every finalized predecessor from original entry.
It rechecks affected payloads and both sides of each selected key use. No old
ledger-version fact seeds a changed traversal. A failed update stops construction;
there is no alternative-plan scoring, recoloring, retry search or legacy fallback.

Acyclic forward advancement reuses an unchanged predecessor map only when its
ledger version is unchanged. This optimization is not advertised as incremental
selected-edit propagation. `selectedUpdates`, `replaySiteEvaluations`, normal
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

## Native adapter and gated integration

The selected native entry uses the original SCF graph, including all original
instruction cuts, choices, zero-trip alternatives and loop backedges. It does
not expand leaf loops into the historical backend's first/tail observation
quotient. This representation is selected before construction, without retry:
native import does not yet supply the occurrence/full-write certificates needed
by the selected cyclic specialization. The portable qualified-observation API
remains available. Shared semantic extraction and the live historical driver's
representation are unchanged.

This choice was tested on the actual Qwen3 prefill captures: the initially
supplied native path refused RMSNorm and post-RMSNorm during selected updates
after leaf-loop expansion. Keep these original kernels in regression coverage;
do not substitute simplified bodies or claim precise native slot recurrence.

The frontier now represents the existing synchronous-payload contract by putting
the next launch gate after that payload's completion. It does not turn SET into
a source-side barrier. Terminal ALL is allowed only at the original invocation
exit and records retirement without consuming notifications or granting interior
completion/rearming credit. Mixed retired/nonretired inputs retain both a may
continuation restriction and a must exit requirement.

`testing::runSelectedHandoffSyncWithMutation` uses the same shared importer,
private-copy transaction, SyncCodegen and exact command/guard/payload read-back as
the live driver, with the new constructor and compact checker as callbacks.
`runHandoffSync` continues to call the historical constructor and checker. The
new hook never falls back to them. The optional report describes construction;
the returned LogicalResult additionally includes native reconstruction.

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

## Validation and remaining cutover gates

The accompanying portable tests construct from unsynchronized input. The exact
reference bridge checks 108 selected programs, including 100 deterministic
straight-line/choice/loop/nested inputs and four cyclic slot populations. Every
exported causal fact is checked against every reached exact state. Eight selected
examples are also checked by the unchanged paired-order collector; four
straight-line examples additionally use an independent full-history graph.
Cyclic endpoint-deletion tests cover 120 original-observation endpoint mutations.
The reference files remain byte-identical to the pinned repository versions.

The package's original validation used a reduced portable build. Full-checkout
integration additionally builds the entire standalone production library and
the native selected/overflow drivers. The native drivers are registered in
`check-pto`. Real-kernel replay is reproducible with:

```sh
python test/oahs/replay_prefill.py --build /path/to/native-build \
  --output /path/to/new/report-directory --timeout 60
```

Run with the Python version configured for the native build. This captures all
21 pinned prefill families (23 functions), preserves module-level failures and
timeouts, compares default/explicit `existing`, and lowers successful selected
output without a second synchronization insertion. Static inventories and host
timings are diagnostics, not device correctness or performance evidence. This
cohort does not enumerate all PyPTO/pypto-lib kernels.

Before live cutover: run those native gates, compare the entire admitted
population with import/construction/reconstruction/refusal rows, qualify missing
native occurrence and typed adapters without weakening their contracts, and
review whole-pass time/memory. Only then remove the historical constructor and
its unused ordinary backend. Neither that removal nor reference cleanup is part
of this patch.
