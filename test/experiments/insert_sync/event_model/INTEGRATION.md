# Placement and native migration

**v0.2 remains the same isolated experiment.** Its patches add/update only
`test/experiments/insert_sync/event_model`. They do not change native InsertSync.
The old commit below is a historical anchor, not a freshly checked branch head.
Use the existing experiment child/worktree and preserve the other agent's work.
See `REVISION.md` for the implemented review fixes.


## Recommendation

Create a **short-lived feature branch/worktree from the current working InsertSync
branch**, not another clean-main autosynchronization rewrite. The reviewed anchor
is `862ab811124d2e67a7f2cde992461983163f6a7a`. The other agent's uncommitted changes
are not in that anchor and must not be reset, overwritten or implicitly included.

Suggested commands, run by the repository owner from their checkout:

```sh
git fetch origin codex/insertsync-revision-r1
git worktree add -b codex/insertsync-event-model \
  ../PTOAS-event-model 862ab811124d2e67a7f2cde992461983163f6a7a
cd ../PTOAS-event-model
git apply --check /path/to/ascend-event-lab.patch
git apply /path/to/ascend-event-lab.patch
cd test/experiments/insert_sync/event_model
python -m unittest discover -s tests -v
python run_campaign.py --output /tmp/ascend-event-model-results
```

Rebase the small experiment branch onto the agent's agreed integration commit
when ready. No remote branch, commit or push was performed by this delivery.
The patch only adds this experiment directory. It touches no production CMake,
passes, flags, existing tests, or existing source files. Patch application was
checked in a clean temporary repository, not a complete native PTOAS checkout.

## Why a separate worktree, but not a separate compiler architecture?

The new code is an executable specification/reference for a narrow decision, not
a production replacement. It should be easy to rerun, compare and delete. Isolating
it prevents two agents from editing the same production planning path while we
establish the algorithm's behavior. Its placement in the same repository retains
input provenance and makes a native exporter/test straightforward.

A successful native migration must **replace one existing decision path** using
the current common requirement interfaces. It must not add another permanent pass
that constructs different private semantics and then competes with InsertSync.

## The missing native bridge (not implemented in this package)

Add a read-only exporter immediately before synchronization selection, using the
existing translator and qualified analyses. Do not create a second PTO text parser
or infer memory effects merely from an operation name. Required fields:

| Existing fact | Exported meaning |
| --- | --- |
| Original phase/operation identity and region path | Stable statement identity and complete iteration tuple |
| Physical context and OpPipe/translated phase | Actual same-core lane, with unsupported cases explicit |
| Existing control/loop/selector arithmetic | Domain, schedule, and proven parameter conditions |
| BaseMemInfo/view/layout/subset facts | Actual physical byte relation, not a guessed allocation footprint |
| Definite versus may-write information | Whether a previous value can really be killed |
| Fixed resource/intrinsic/visibility contracts | Explicit supported semantics or exclusion, never ordinary payload |

The JSON files are a facts interchange for the experiment, not new source-program
annotations or extra caller promises. A native exporter must document every range,
alias and arithmetic assumption. Distinct pointer roots are not automatically
disjoint. The current scalar replay supplies useful dynamic occurrence identities,
but its hashed tensor descriptors are **not** this complete effect export.

## Keep discovery independent from current repair

At the reviewed head `SyncRequirement::Kind` mixes disjointness, intrinsic order,
lifecycle membership, full completion and direct repair explanations. Preserve
its existing behavior while migrating one family, but separate the underlying
RAW/WAR/WAW property and occurrence relationship from its current discharge.
Do not use `alreadySync` to decide which semantic requirements exist.

For the migrated family:

```
immutable source facts -> requirement relation
                                |
logical direct/recurring plan -> resolution and frontier decision
                                |
                     target realization and fresh reconstruction
```

Neither the exporter nor experiment should add a mandatory completeness gate to
the production pass. Unknown semantics remain with the established conservative
path, with attribution. A prototype rejection is not a discovered device bug.

## Native milestones

1. **Read-only parity:** export the unchanged QK, Q projection, buffering controls
   and GEMM at the actual pass entry. Capture commit/input hashes and compare
   dependencies with the existing translator. Do not rewrite inputs to fit examples.
2. **One real planning decision:** use the prefix-frontier query for the looping
   QK readiness relationship reported by the engineering agent. Retain separate
   requirements for the other preload/consumers. Restore them correctly on retry.
3. **Participation:** lower symbolic H/H-inverse using existing first/last/empty
   code and stable original identities. Prove guards are available at insertion
   points. Reconstruct every concrete event after emission; no planner certificate
   or proposed modulo expression substitutes for that check.
4. **Resources:** use the symbolic reuse condition or native causal analysis for
   candidate assignments across direct and recurring streams. Preserve target
   reservations; a failed search does not justify moving endpoints or serializing.
5. **Quality:** verify the claimed removal of unnecessary blocking on the actual
   emitted kernel, not just a lower site count. For Q projection, combine only
   compatible boundaries. Preserve MMAD and GM-disjointness improvements.
6. **Device:** independent numerical/progress tests and isolated device timing.
   The reference model's interleavings and symbolic proof are not device evidence.

## What to reuse, and what to replace

Reuse native effect extraction, selector normalization, alias policy, target
ordering rules, structured boundary code, allocation reservations, diagnostics,
clone/verify/commit and independent re-extraction. The prototype demonstrates the
coordination rule between them.

For one supported local-storage family, replace its old ad hoc readiness merge
or move decision with the shared requirement/frontier decision. Once the native
consumer works, remove the superseded decision logic; retain this small reference
as a regression oracle, not a third production generation engine.

## Acceptance boundaries

- This package succeeds if its explicitly modeled examples are proved and its
  negative controls distinguish missing synchronization from unnecessary ordering.
- It does **not** claim the unchanged native QK or Q-projection case is fixed.
- The native milestone succeeds only when the unchanged looping input commits a
  justified improvement and current working controls/GEMM remain valid.
- Zero named barriers, 53/56 static pairs, and host test counts are never treated
  as interchangeable with pipeline quality or device runtime.
