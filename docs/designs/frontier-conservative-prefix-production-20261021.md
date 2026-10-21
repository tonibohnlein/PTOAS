# Standalone conservative prefix services through production

This checkpoint migrates the accepted current-cut implementation and its joint
pruning into the compiler library. It does not add retained-child composition,
change default dispatch or implement a new allocator. Starting revision:
`7facb6ea859ca06399f02e8438fd8aaa6aebad64`, plus the pinned uncommitted pruning
changes. The complete starting diff and source/tool hashes are under
`.local/conservative-prefix-production-20261021/`, including `start.json` and
`baseline-source/`. A fresh baseline focused gate passes 34/34 on that final
pruning tree, before migration. Earlier reports retain their historical scope.

## Ownership and callers

| Stage | Previous live caller/owner | Shared implementation and callers |
|---|---|---|
| Access classes | Test-only completion-dataflow serializer over `CompletionStorage` | `PrefixInput` uses the same immutable reservation classes and original `SyncInput` effects. `PrefixPreparation` consumes the typed graph. |
| Structured control | JSON SCF tree reparsed by the test bridge | `PrefixInput` builds one finite graph using MLIR SCF for/if, with one operation-to-site index; no trip-count unfolding. |
| Fixed point | JSON adapter → `SourcePipeServices` | Existing `solveSourcePipeServices`, unchanged. |
| Proposals/delivery check | Test bridge → `ServicePruning` | Existing proposal and joint certificate, unchanged, called directly with typed masks. |
| Admission/native/control/visibility | `CurrentCutContracts.h` / `CurrentCutNativeOrder.h` in tools | `PrefixContracts.h`, `PrefixNativeOrder.h` and `PrefixQualification.cpp`; the old implementation headers are removed. Existing `PhaseIndex`, `CompletionReadiness`, effect and target contracts remain authoritative. |
| Command/role recipe | JSON `commandPlan` / `roleIds` | `PrefixRecipe` constructs the same ordered acknowledgment quartets and constant roles; both the live service and diagnostic serializer use it. Logical-only research serialization stays test-only. |
| Physical preparation/certificate/emission | `CurrentCutServiceChecks.cpp` | `PrefixPreparation.cpp`, private `PrefixPreparation.h`, `PrefixServices.cpp`; existing physical endpoint-map planner/emitter and `FunctionStaging`. |
| Public integration | Experimental flags only | Module pass `pto-frontier-analysis` explicitly delegates to `insertConservativePrefix`; experimental flags are thin wrappers over that same service. |
| JSON | Inter-stage production-prototype dependency | Terminal diagnostics and abstract oracle inputs only; never parsed to select masks, roles, cuts or physical operands in the live service. |

The diagnostic dataflow exporter still serializes arbitrary research inputs;
it does not become a production contract. Its physical command serialization
uses the shared typed recipe. No production library includes tool headers.
The complete fixed point stays unchanged while an optional candidate owns a
separate mask; failed refinement keeps the original. Prepared commands, cuts,
constant maps and the separate structural protocol check stay transaction-local.
Changing a mask triggers recipe construction and physical checking again; no
certificate is cached independently of the representation it checked.

## Explicit entry point and guarantees

```bash
pto-test-opt --pto-frontier-analysis='conservative-prefix=true physical-output=true gm-alias=may-not-alias' input.pto
pto-test-opt --pto-frontier-analysis='conservative-prefix=true physical-output=true prune-prefix-services=true gm-alias=may-not-alias' input.pto
```

`allocation-ids` supplies the existing eligible pool (default 0..5). The
library validates distinct native IDs, intersects each direction with the
existing reserved-tail contract, and requires its baseline forward/return
roles. Synthetic `test.current_cut_ids` is consumed only by the test wrapper.
Opposite service orientations have different F/R roles in their common actual
direction. Pruning retains the baseline mapping without recoloring.

Both option forms report conservative modeled-storage coverage and
`covers-selected-prefix-order`. Neither is minimum-demand analysis or an
order-equality proof. The acknowledgment protocol may add order relative to
the smaller logical prefix plan. Local barriers/native discharges and the
whole-invocation return ALL have their original meaning. Empty flags at
invocation entry and exclusive role ownership are required; closing issued
notifications does not complete pending payloads. The return ALL separately
supplies payload completion. The A3 invocation-completion premise is the existing one; the target local-pipe
barrier table intentionally does not describe ALL. Invocation completion is excluded from the
paper's per-payload command-site bound.

`prune-prefix-services` requires `conservative-prefix`. The route requires
physical output and allocation preparation and excludes exact audit,
arithmetic-profile, SCC completion, deferred-output, coordinate-allocation and
loop-normalization policies. It never runs after a missing exact export and
never silently changes alias policy. Default production selection is untouched.
Manual/open protocols and calls remain unqualified. Original control, guard
calculations, occurrence frames and resets remain in their original IR.

Refusals distinguish access/operation admission, scalar-GM visibility,
control/readiness, cuts/directions, resource reservation, delivery/protocol
certification and IR emission. A resource refusal is insufficiency of this
sufficient constant-role construction, not proof that every protocol is
infeasible. More IDs do not establish scalar visibility, native footprints,
readiness or legal command cuts. May-not-alias still requires the caller's
distinct-root disjointness premise; unknown GM coordinates are not absolute
addresses and reservation classes remain conservative.

The module pass holds exclusive symbol-table ownership and stages the entire
request. The function service uses existing `FunctionStaging` inside that
owned module. A failure in a later function leaves every original function
unchanged. The prepared list is complete and protocol-checked before any
staged IR emission; verification precedes commit. No live sibling inspection
from a function pass is introduced.

## Validation and measurements

The final tree passes a fresh **35/35 focused tests** in 277.08 seconds,
without a timeout. This includes the previous 34 cases and the new production
policy test. The standalone admission/cut/visibility policy is unchanged.
Native serial builds have no new warnings. The implementation uses existing
MLIR SCF, effect interfaces, dominance, symbol ownership, staging and physical
endpoint services. No full repository/device campaign or sanitizer run is claimed.

Experimental and production outputs agree for 18 semantic fixtures, 10 static/
trip scaling cases and original Delta, after ignoring terminal newlines only.
`MlirOptMain` appends one extra trailing newline; no text inside the IR is
normalized. All 36 fixture/mode outputs also agree with the pinned preceding
pruning archive under that same comparison. Fresh migrated experimental Delta
outputs are **byte-identical** to the immediate baseline.

Actual production commands pass independent declared-layout access/control,
matching, acyclicity and causal physical-reuse checks. The 32 Delta scenarios
reuse independently traced original SSA/control plus retained reservation
model effects; they do not independently prove native footprint extraction or
execute full numerical Delta. All checked Delta payload prerequisite relations
remain identical despite fewer commands. The disjoint-buffer fixture retains
strict order containment: 16 event-reachability pairs removed, none added;
completion-to-start excess falls 11→7 and executed commands 65→49.

| Qualified original Delta, may-not-alias | Baseline prefix | Certified pruning |
|---|---:|---:|
| Source-pipe OUT facts | 1,144 | 1,144 |
| Cross services | 218 | 129 |
| Static SET / WAIT | 436 / 436 | 258 / 258 |
| Barriers, including return ALL | 61 | 61 |
| Total static commands | **933** | **577** |
| Arithmetic operations | 1,846 | 1,134 |
| Production IR bytes | 152,684 | 98,702 |
| C++ bytes | 77,343 | 60,001 |

The extra production IR byte relative to the experimental receipt is its
terminal newline. Both plans retain IDs 0/1 per used direction in the default
six-ID pool; no minimum budget is claimed. The new production entry also
preserves Delta's scalar-GM visibility refusal under may-alias, with both six
and eight explicitly eligible IDs. No allocation-capacity explanation replaces
that missing visibility premise.

The new policy regression covers two-role pools 0/1 and 3/5, insufficient and
reserved pools, unsupported target, incompatible options, ignored test-only
pool annotations, read-only inputs under both alias policies and a potentially
conflicting scalar-GM read. A late second-function refusal follows successful
staged preparation/emission of the first; the failure dump contains the original
module without partial commands or published success diagnostics.

The three existing original-kernel checks—Persistent GEMM (may-alias), Qwen
down projection (may-not-alias), and Qwen finalize (may-not-alias)—pass default
logical insertion, allocation preparation, six-ID allocation and C++ generation.
All **12 corresponding artifacts are byte-identical** to the pinned immediate
baseline. This is a default-path regression check, not another corpus survey.
Both Delta routes plus nondefault-pool, scalar-control and disjoint-buffer
production outputs also generate C++. The C++ compatibility backend and its
libraries are pinned separately; the new pass runs in the freshly built native
compiler/pass driver. No device compilation or execution is performed.

Static stages 1/2/4/8/16 and trip counts 1/256 retain the previous fact, service
and proposal bounds with no trip-count expansion. For the pruned plans:

| Stages | Proposal nodes | Service tests | Deleted | Joint OUT facts | Transmissions |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 2 | 0 | 40 | 52 |
| 2 | 9 | 3 | 1 | 46 | 58 |
| 4 | 11 | 5 | 3 | 58 | 70 |
| 8 | 15 | 9 | 7 | 82 | 94 |
| 16 | 23 | 17 | 15 | 130 | 142 |

The 256-trip cases are compile-only derivatives. Bounded execution checks use
small valid instances. Nine **abstract diagnostic** cases vary pipe counts
2/3/4 and class counts 1/4/8 independently, without claiming those arbitrary
pipe numbers constitute a target-qualified emitted program. The receipt records
prepared modes, facts, inspections, transmissions, role tables and costs.
Expected solver/debt work is parameterized by h and k; ordered pair maps and
short-mask validation add implementation lookup/scanning costs. Fixed-h/k
linear scaling does not establish the paper's ideal bound uniformly in k.

One fresh descriptive production sample separates costs (milliseconds):

| Stage | Baseline prefix | Certified pruning |
|---|---:|---:|
| Access recovery | 1.820 | 1.796 |
| Phase index | 0.190 | 0.189 |
| Admission/readiness | 0.061 | 0.062 |
| Class summaries/index | 0.019 | 0.019 |
| Structured graph | 0.012 | 0.011 |
| Baseline solve | 0.106 | 0.106 |
| Protected paths | — | 0.009 |
| Proposals | — | 0.020 |
| Joint verification | — | 0.307 |
| Physical preparation, inclusive | 0.284 | 0.112 |
| Protocol check, subset | 0.012 | 0.007 |
| IR emission/verification | 1.466 | 0.942 |
| Compiler subprocess | 164.094 | 164.943 |
| C++ generation subprocess | 465.545 | 466.464 |

These are single serial observations, not interleaved speedup measurements.
Staging, parser/printing and process startup are included in subprocess cost,
not fully separated by the stage timers. JSON diagnostics remain inspection
cost, rather than an inter-stage representation. A failed early test used the
local-pipe capability table for ALL; that table intentionally does not model
invocation ALL. The final code retains the original A3 invocation contract.
Another early comparison exposed only the documented extra terminal newline.
Both failed intermediate receipts are preserved; the final full gate passes.

Ruff F checks and `git diff --check` pass. The scoped compliance prefilter
reports 41 brace-rule matches, individually reviewed as false positives with
braced bodies (nested calls/multiline conditions confuse its regex); the
review ledger records no unresolved finding. No suppression was added.

Timing definitions differ from the JSON prototype: class preparation now
includes the typed site summaries/index, and graph preparation traverses the
original SCF directly. There is no live baseline/candidate JSON-plan
serialization or reparsing stage. Diagnostic statistics serialization remains
separate from the mathematical solver and is not an input to it. Inclusive
physical preparation includes native/map/protocol subsets and must not be
added to them. Isolated timings are descriptive, not speedup evidence.

## Next checkpoint: retained children, not admitted here

The existing Qwen down-projection whole K invocation is the next candidate.
The constant-map child-import report establishes a retained symbolic recipe
with 171 families, 282 endpoint sites, 27 grouped endpoint sites and 138
allocation records. The decoder uses IDs 0–3, with original enclosing frames,
visit guards, grouped record labels and heterogeneous tuple arities. It has
closed issued notifications and nonempty pending payload effects. Exact GM
selectors remain unavailable; completion composition uses sound coarse access
summaries instead. These facts are not standalone prefix admission.

The pinned decoder uses two IDs in directions M→MTE1 and MTE1→M, and
four in MTE1→MTE2 and MTE2→MTE1 (numeric pipes 2/3/4). Thus two
parent roles plus this single child's four-ID palettes would exactly fill a
six-ID direction in the latter pair, before any other reservations/users.
For example, child IDs 0–3 and parent roles 4/5 would be disjoint in
the heavy pair, if eligibility and every other owner permit them. The current
standalone baseline mapping is not changed to prepare for that possibility.
That arithmetic is a follow-up budget check, not current acceptance.
`CompletionInlineChild::resourceFootprint()` already exposes directional ID
masks; `classifiedSummary()` and `roots()/input()` expose sound effects and
original sources. First-publication/last-consumption guards and per-pipe entry
cuts are not yet an exported provider contract. The existing decoder proves
within-visit use, while current SCC integration uses actual parent completion
cuts for cross-visit separation.

The follow-up must obtain a provider-owned **per-pipe** sound access summary,
participating pipes, visit entry/readiness cuts, first publications/last
consumptions, internal palette assignment and ownership, coordinates and exact
visit guards. It must prove each entry wrapper precedes the child on every
participating pipe. A scalar cut existing in the IR does not establish that
multi-pipe property. Omitted GM domains must remain unavailable as exact
selectors, without losing their conservative effects.

Reserve child palettes separately from parent acknowledgment roles and other
child families in each actual direction. Check `sum(child palettes) + parent
roles <= eligible direction pool`, including reserved tails. Empty matching
at a child exit does not imply `old WAIT -> next visit SET`. The Section 7
entry acknowledgment must supply that causal path using supported directions,
compatible guards and occurrence frames. Existing resource-separating ALL
cuts in the SCC import are not automatically present after replacing that
parent. Parent and child protocols require a fresh joint physical check;
pruning cannot touch arbitrary child commands. No such import is enabled now.

## Future automatic selection policy

After a whole-region request has exhausted qualified exact candidates **without
retaining an exact mathematical result**, an explicitly permitted conservative
quality policy could request this standalone route before primitive SCC
completion. Eligibility must use the same access/native/control/ownership
contracts; a retained exact result with a missing export must remain retained.
Compare already completed conservative alternatives only when their
coverage/quality contracts agree; do not run every backend merely to compare
code size. Expose selected conservative coverage, separate physical reuse,
return completion and possible added order. This is a proposed policy, not a
behavior change in this checkpoint.

## Paper consequences

| Construction | Implemented evidence | Remaining premise / scope |
|---|---|---|
| Section 6 source-pipe analysis | Existing finite fixed point, original SCF, sound class summaries; no runtime-trip unfolding | Supplied operation effects/alias contracts; modeled-effect coverage, not native footprint proof |
| Joint service deletion | One fixed candidate, protected scalar/native paths and existing joint debt certificate | No refresh-relay pruning, optimality claim or arbitrary child-command deletion |
| Logical prefix realization | Same current cut/control and original occurrence identity | Ready admitted control and legal pipe cuts; no historical guard synthesis |
| Section 7 physical realization | Same ordered acknowledgment quartets, distinct actual-direction roles and structural rearming check | Standalone empty/exclusive roles and target command semantics; may add order |
| Invocation | Closed notifications checked independently of final ALL | Return completion is a separate target premise and command count |
| Bounded checks | Actual emitted code and independent access/control/command graphs | Host-model evidence, not a uniform native/device proof |
| Child palettes | Concrete next contract identified | No retained-child composition in this checkpoint |

No allocation algorithm, hardware exemption, expert protocol or paper-theory
source is changed. There is no device compilation or execution.

## Reproduction and receipts

After synchronizing the task's changed files into the established Linux source
snapshot, build `pto-sync-input-test` and `pto-test-opt` with `--parallel 1`.
`gate-final-command.json` pins the complete 35-test lit invocation (`-j1`).
The report's first two commands exercise unpruned/pruned production directly.
`run_evidence.py` in the disk-backed campaign records the three original default
comparisons and fresh Delta/C++ samples using fixed 35-second compiler
watchdogs. `baseline-source/`, `baseline.diff`, `start.json`, `paper/`,
`accepted-lit/`, final source snapshots and source/tool/artifact hashes preserve
the dirty starting point and final evidence. The previous fixture outputs are
explicitly reused from the preceding accepted pruning archive; fresh output
comparisons and the new full gate are separate results.

[Machine-readable receipt](frontier-conservative-prefix-production-20261021.json).
Changes remain uncommitted; no push was performed.
