# InsertSync follow-up: evidence, admission, and redundant completion barriers

Base: `60db1026a1de8f1bd0154f9bdcb21e0f6d944a7b`,
`tonibohnlein/PTOAS`, `codex/insertsync-revision-r1`.
Prepared 2026-09-06. This is a source patch, not an upstream commit or a device qualification.

This assessment accompanied the supplied patch before native integration.
Subsequent native build, regression and benchmark results are recorded in
[FOLLOWUP_V2_VALIDATION.md](../../test/experiments/insert_sync/FOLLOWUP_V2_VALIDATION.md).

## 1. Assessment

I agree with the corrected assessment. Concrete defects in compiler reasoning,
missing analysis models, injected-fault detection, native emission, and device
failures are different evidence classes. They must not be conflated.

| Finding | What the existing evidence supports | What it does not establish |
| --- | --- | --- |
| Load/load WAW exemption | Source-level omission under the stated ordering contract; the original overlapping issue667 example reproduces omission in native output | A reproduced device failure in the 213-row corpus |
| Wrapped address arithmetic | A concrete arithmetic weakness, repaired and tested at boundaries | An ordinary corpus failure caused by overflow |
| Shifted slot maps / zero-trip handle provenance | Unsound disjointness arguments with focused regressions | That every legacy kernel or lifecycle is incorrect |
| Missing helper/queue/resource models | A completeness gap in the newly introduced check | An incorrect established lowering, absent further evidence |
| Auditor mutation detection | The checker notices specific deliberately injected faults in its domain | A fault originally emitted by InsertSync |
| Safe but earlier acquisition / later publication | An unnecessary-ordering concern, when demonstrated | A race or deadlock merely because more waiting occurs |

A missing summary may ultimately expose a real omission. It must not be erased,
called pure, or promoted to a correctness certificate. But its mere absence is
not a reason to replace established translation with a new mandatory admission
model. The compatibility policy below makes that distinction explicit.

### `df89ee02ca1c`: keep

This is the useful native baseline step after the initial patch. The branch
records 208/213 native PTO and C++ emission successes for main, R1 combined and
R1 staged; five inputs fail before InsertSync. It checks pass participation,
freezes native-library fingerprints and retains per-input output differences.
Correcting issue667's obsolete no-barrier expectation is justified by its
actual overlapping destinations. None of this proves whole-corpus safety or a
runtime speedup. These are recorded upstream results, not executions performed
while producing this package.

### `60db1026a1de`: retain fixes, revise the rollout

Keep explicit GM contracts/root tracing, compatible physical slot partitions,
zero-trip and changing-handle provenance, pass-local configuration and diagnostic
auditing. Do not revert restored WAW checks or checked range arithmetic.

Revise the mandatory coverage gate. The branch records 165/213 emissions after
that change: 43 additional coverage-gate rejections plus five pre-pass failures.
Those 43 rows are not 43 discovered device bugs. Eight formerly positive lit
fixtures were changed to expected compiler failures; green negative tests are
not recovered autosynchronization capability.

The auditor's recorded nine `verified-local` outputs and 156 `unsupported`
outputs are different from complete hardware verification. In particular,
GM visibility and unmodeled internal protocols are outside that certificate.

## 2. What this patch implements

### A. Restore diagnostic-first completeness checking

New option: `--insert-sync-effect-coverage=report|strict`, default `report`.
The equivalent pass option is `effect-coverage=...`.

* Report mode runs the existing read-only completeness query, retains its
  source diagnostic as a remark, records
  `pto.insert_sync.effect_coverage = "gap-legacy-retained"`, and continues the
  established translator/planner. It never labels that gap safe.
* A complete check records `"complete"`. This is translator-coverage status,
  not scheduling, progress, publication or device correctness.
* Strict mode preserves the previous fail-closed completeness behavior.
* An explicitly present helper effect array with malformed arity/type or an
  invalid effect spelling remains an error in BOTH modes.
* Unknown failures without a diagnostic remain errors. The diagnostic handler
  surrounds only the known read-only query, not translation, planning,
  allocation or code generation. Exceptions and assertions are not caught.

The checker currently reports the first completeness gap, not a complete census
of every gap in a function. This patch does not introduce an operation whitelist
to silence it or invent ownership/resource summaries.

Eight original bodies regain positive automatic-insertion RUN lines. Their
strict rejection checks remain as explicitly requested diagnostics. Deleted
placement checks for issue564, issue622 and the queue frontend are reinstated
from `df89ee02`, rather than replaced with weaker "some output exists" checks.
The historically implicit disjoint-argument convention of those placement
comparisons is now explicit on the RUN command, not silently inferred by the
compiler. Original computation/addresses are not changed.

The supplied package did not run these tests natively. Native integration has
since passed the restored placement checks; see the linked validation report.

### B. Harden the diagnostic auditor

1. A known multi-phase macro or hidden-event operation is not attributed to its
   advertised single pipe. This narrow auditor returns `unsupported` until the
   complete phase/resource model is implemented.
2. For literal finite loops, bind the actual induction value and evaluate the
   supported scalar subset: constants, index casts, comparisons, select,
   add/sub/mul, bitwise operations and unsigned remainder. Arithmetic overflow,
   unsupported expressions and unknown inputs remain unknown.
3. Record when a path uses an uninterpreted predicate. A failed such path is
   `unsupported` with a feasibility reason, not a demonstrated invalid token.
   Continue checking exact paths so an exact failing path can still be reported.
4. Success over every enumerated (possibly conservative) path remains a local
   success; no symbolic-loop induction is inferred from finite enumeration.

The literal-loop limits are not enlarged. The newly bound index values use the
existing PTOAS A2/A3 64-bit index convention. The code is NOT a general path
satisfiability solver and does not interpret symbolic trip counts.

The two focused native fixtures exercise a first-iteration-only wait and a
correlated EQ/NE predicate whose feasibility the checker cannot establish.
Both passed during subsequent native integration, as recorded in the validation report.

### C. Implement a non-serializing synchronization reduction

New opt-in option: `--insert-sync-prune-completed-barriers`.

After event assignment and emission, a small completion-prefix interpreter
identifies named barriers whose ENTIRE source prefix is already complete through
existing acquisitions or an earlier named barrier. It removes only those
barriers. It does not remove event pairs, retime a signal or wait, change an ID,
add a handshake, weaken an alias contract, or touch PIPE_ALL.

Example:

```
MTE2: load A; set MTE2->V
V:    wait MTE2->V; consume A; set V->MTE2
MTE2: wait V->MTE2; barrier MTE2; overwrite A
```

The return handoff already establishes completion of the earlier MTE2 prefix.
The named MTE2 barrier is redundant. If new MTE2 work occurs after the return
wait, the barrier is retained unless that NEW prefix is also proved complete.

The state separates physical issue epochs, acquired completion prefixes and
causal action clocks. A source signal does not automatically advance that
source's completion knowledge. Initial sentinel epochs preserve unknown incoming
work. Repeated hardware keys require causal consumption-before-rearm, not merely
textual wait-before-set order. A later unproved event lifecycle discards every
proposed removal without modifying IR.

Initial refinement domain: a single A2/A3 vector-core block, static flags, no
structured regions, and the explicitly supported ordinary tload/tstore/tabs/tadd
physical operations. Metadata-only instructions may intervene. Macros, unknown
operations, dynamic events and control-flow regions leave the function UNCHANGED.
This is an optional optimization boundary, never a compiler admission boundary.
Production integration also requires the translator coverage query to complete.

The refinement uses completion, not precise access overlap, so it does not
re-run all alias analysis. It preserves the ordering of every retained original
physical and event action in its model. That reduces synchronization instructions
without sacrificing pipeline overlap. It does NOT claim new overlap or measured
speedups. It does not yet address broad loop/branch motion; that is the next step.

## 3. What is intentionally unchanged

* The WAW correction and overflow helper from the imported patch.
* GM contract interpretation, slot partition certificate and changing-handle fix.
* InsertSync's operation translator, default staged/combined choice and codegen.
* Existing explicit-sync bypass and atomic-helper handling.
* MoveSyncState, event-pair redundancy deletion and the existing allocator.
* Existing mandatory exit policy, GM publication and cross-core qualifications.
* No new pattern recognizers and no global set-cover planner.

In particular, this patch does not certify the existing allocator or pretend
that a logical static channel with no assigned ID is already an unlimited
family of dynamic event identities.

## 4. Next implementation step: real-kernel boundary placement

After building and running this follow-up, choose an original corpus kernel
where the stage dumps show a wait moved before independent work or a publication
moved past the last relevant source. Use that kernel, not just a synthetic loop,
as the unit of the next change.

### R3a — retain witnesses and before/after evidence

Keep the original logical source and target separate from the mutable codegen
anchors. For each moved/deleted action, record its source-prefix, acquisition,
guard and relevant loop instance. Use existing stage dumps; no new mandatory
semantic frontend is required. Keep complete real inputs and their alias
contracts fixed.

### R3b — one complete first-use/last-use construction

Implement the supported once-only prefix/body or body/suffix handoff as a whole:

* publication immediately after the required source, with bounds/predicate SSA
  availability checked;
* acquisition at the first executing relevant consumer, not automatically at
  loop entry;
* a distinct nonempty/empty transfer, preserving true bypass hazards;
* publication after the last relevant reader, not after unrelated source-lane
  work or unrelated output stores;
* cleanup separate from payload prerequisites, so it does not block an
  independent suffix merely because it is syntactically after the loop.

Do not simply disable motion. A once-only signal and a wait on every iteration
are not a correct replacement. Do not use a serialized whole-loop hub to make
the protocol easy to balance. Outside the new construction's supported domain,
retain clearly attributed legacy behavior; do not claim it newly verified.

Acceptance examples include the slow-A/ready-B/zero-trip case and a panel whose
last reader can release it while an unrelated output store remains outstanding.
Both need a corresponding unchanged real corpus case before the milestone is
called complete. Test positive executions and over-ordering mutations separately
from race/token mutations.

### R3c / R6 — simplify and realize without changing the baseline

Broaden complete-handoff deletion only with both completion coverage and token
lifetime arguments. Keep unlimited logical occurrence identities as the baseline.
Finite assignment may exploit required acknowledgements and proven mutual
exclusion, but may not retime publication/acquisition. Only an explicitly
attributed resource-recovery stage can introduce extra ordering. An analysis
limit or conservative interference graph is not a silicon scarcity theorem.

New lifecycle pattern recognition remains the second push. Shared slots, effects,
control participation and basic handshakes are general correctness machinery and
must not wait for a shape-specific recognizer.

## 5. Required native comparison

Run the unchanged 213-row/186-hash population in distinct directories:

1. Frozen `df89ee02` baseline (retain its original contract behavior separately).
2. Frozen `60db1026` for explicit comparison of the hard-gate regression.
3. Follow-up, coverage=report, pruning off.
4. Same follow-up and contract, pruning on.
5. Strict coverage as a diagnostic ablation, NOT as the admission headline.

Run the relevant combined and staged configurations, with compiler-library
fingerprints. Report PTO/C++ emission, actual pass participation, compatibility
gaps, supported/unsupported audit, static action counts, removed-barrier counts,
new failures and timeouts separately. Do not change caller noalias assumptions
between arms and attribute their effect to the optimizer.

Run all check-pto tests and restored positive regressions. A failure in an old
positive test is a regression to explain, not permission to convert it to an
expected rejection. A confirmed unsafe omission needs a fix, even if safe output
requires more synchronization. Device correctness and device-only timing remain
separate required gates before a performance claim.

## 6. Sources and validation boundary

Pinned code and recorded campaign:

* https://github.com/tonibohnlein/PTOAS/commit/df89ee02ca1cafad1ac8a75f6d9c55b354ae4d84
* https://github.com/tonibohnlein/PTOAS/commit/60db1026a1de8f1bd0154f9bdcb21e0f6d944a7b
* `test/experiments/insert_sync/README.md` at 60db1026.
* `SyncEffectCoverage.cpp`, `SyncAudit.cpp`, `SyncAuditPaths.cpp`,
  `SyncGMAlias.cpp`, `SyncSlotMapping.h`, `PTOIRTranslator.cpp` at that base.
* User-supplied “InsertSync versus ProtocolSync” assessment and R1 revision plan.

Hardware completion contracts consulted (development-preview provenance retained):

* https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.html
* https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.html

The ordered-source and permissive nonblocking-source graph variants in the local
tests are declared abstract models, not two hardware qualifications. This package
contains no native PTOAS binary, full repository checkout, SDK, frozen 213-row
corpus, device result, or measured device speedup. Its original `STATUS.json`
belongs to the external patch package; the linked validation report records
the subsequent work performed in this repository.
