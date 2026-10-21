# Jointly certified experimental service pruning

Starting revision: `7facb6ea859ca06399f02e8438fd8aaa6aebad64`. Pruning stays
optional in the experimental current-cut path; production dispatch, allocation,
alias assumptions, native exemptions and original payload/control are unchanged.
Pins and fresh artifacts: `.local/service-pruning-20261021/`.

## Eligibility and proposed construction

The actual baseline uses `CompletionStorage` reservations, original SCF and
`solveSourcePipeServices`. The exporter supplies no refreshes: every consumer
selects every potentially earlier conflicting source pipe from IN. The general
solver supports fixed class-wide KILL/GEN, but such inputs can depend on relay
coverage through refresh operations. This pruning adapter initially refuses
refresh-bearing inputs rather than applying the full-baseline local proof to
them. A fixed candidate's verifier checks delivery of raw access facts without
assuming that refresh or acknowledgment clears them.

Existing `CurrentCutContracts`, `PhaseIndex`, result availability and
`CompletionReadiness` supply the non-storage qualification. The baseline admits
only synchronous scalar payload results, supported original SCF, and no extra
completion-readiness cuts, carried values, calls or existing manual protocols.
Every scalar phase already has a qualified native serialization contract.
Keep all scalar-incident services, every self-pipe barrier/native discharge and
all native SSA/control edges. This deliberately coarse protected subgraph is
derived from phase/availability contracts, not empty storage effects. Any
unqualified non-storage prerequisite prevents pruning, leaving the baseline.
The whole-invocation return ALL remains unchanged; closed notifications are
not an assertion that payloads completed before that return completion.

One proposal scan partitions the supplied CFG into straight-line blocks. Each
retained cross-pipe acquire establishes a source/destination anchor. Track all
source class/mode accesses after it, propose a later deletion only if none
conflicts with that consumer, and never clear/create an anchor on deletion.
Forget anchors at branches, joins and loop boundaries. Independent receivers
have separate state.

One final delta worklist tracks `(class, mode, source, destination)` debts.
Retained prefix services clear only their destination's incoming debts; GEN
restores current accesses after testing for uncovered incoming conflicts.
Check the full candidate and the protected supporting subgraph together.
Rejection retains the complete baseline. No greedy repeated certification or
trip-count expansion is introduced.

## Physical validity argument

Remove an entire `FSET, FWAIT, RSET, RWAIT` service at its original common cut.
Keep the baseline direction/role mapping, guards, cuts and all remaining
commands in order. For successive uses of a forward role, its first FWAIT
precedes RSET, then RWAIT, then the next FSET on the source pipe. For successive
reverse-role uses, RWAIT precedes the next FSET/FWAIT and hence the next RSET.
Different forward pairs have different directional roles; forward and reverse
roles sharing a direction have distinct IDs. This establishes causal rearming
for every remaining service sequence, including repeated visits, without
adding synchronization or treating loop boundaries as resets. Entry/return
ownership is the baseline's standalone contract. A structural protocol check
must confirm these premises for the selected generated program, in addition to
independent bounded checks on actual emitted commands.

Deleting whole services makes the candidate an ordered command subsequence.
Native order and retained scalar/control dependencies are unchanged. Each
command-semantic edge in the candidate was present or implied in the baseline;
therefore payload order cannot increase. Storage coverage follows from the
joint delivery invariant, independently of local proposal completeness.

## Implementation and retained contracts

`--current-cut-pruned` is an explicit option of the experimental test driver.
`--current-cut-output` remains the unpruned baseline. Neither option changes
production dispatch. The baseline mathematics, original source-pipe fixed point
and its selected masks stay retained. The candidate owns a separate delivery
mask and uses the baseline role assignment. Joint rejection selects the
baseline plan; a failed prepared-protocol check also retries the baseline
before any source IR mutation.

| Obligation | Existing evidence and preservation |
|---|---|
| Storage | Same `CompletionStorage` classes and original phase access modes; no new alias or footprint premise. No refresh supplied by this exporter. |
| SSA and control | Existing availability, `PhaseIndex` and `CompletionReadiness`; all admitted payload-result prerequisites are qualified native scalar paths. Original calculations, control and coordinates remain. |
| Supporting services | Every scalar-incident service is retained, independently of storage modes. Delta protects 113 such services and retains 154 native prerequisite records. |
| Self-pipe order | All selected self deliveries remain; existing scalar serialization discharges the same 15 selected scalar barriers. No hardware exemption changes. |
| Invocation | Standalone empty-role entry ownership and final ALL remain the baseline contract. No open child import or caller restriction is inferred. |
| Physical reuse | Full acknowledgment quartets, original constant role maps, directional eligibility and eligible-ID checks. Prepared commands receive a separate structural protocol certificate. |
| Unsupported evidence | Additional readiness cuts, carried values, calls, hidden/manual protocols and unsupported phases keep their existing admission refusals. An unqualified supporting path prevents pruning. |

[ServicePruning.cpp](../../lib/PTO/Transforms/FrontierSynch/Shared/ServicePruning.cpp)
owns local proposals and the fixed-candidate delivery check. The existing
[service bridge](../../tools/pto-test-opt/SourcePipeServiceChecks.cpp) supplies
its already available graph and masks. The existing
[current-cut adapter](../../tools/pto-test-opt/CurrentCutServiceChecks.cpp)
owns phase-derived protection, unchanged role maps and prepared-protocol
checking. No new operation registry, analysis backend, allocator, barrier
repair or runtime state is introduced.

The diagnostic JSON entry point accepts explicit fixed candidates for negative
certificate tests. These diagnostic overrides are not forwarded from MLIR or
available to the live emitter. A failed candidate never mutates the retained
baseline analysis. Storage coverage of a candidate does not authorize deleting
a protected non-storage service.

## Alias-policy regression

The operation-admission lit driver now compares `--current-cut-contracts` under
both policies. Read-only inputs qualify under either policy. A scalar reader
and a potentially overlapping writer through distinct GM roots qualify under
`may-not-alias` only with that policy's explicit caller disjointness contract;
`may-alias` reports `gm-read-visibility`, and emission refuses for that reason.
No cache freshness is inferred from flags or SSA names. A separate empty-ID-pool
case has no operation/visibility issue but fails existing role reservation.
Thus missing visibility evidence and insufficient event capacity remain distinct.

## Actual emitted-code validation

The expanded campaign passes **34/34** tests. Final ordered-subsequence and
container-lifetime hardening and independent raw-service checks receive a
separate final **2/2** directly affected test run. No fresh all-34 campaign is
claimed after these narrow final changes. Builds,
tests and source-generation commands run serially. Existing compact-K/replay,
zero-inner-body, child-import and scalar/accumulator regressions are retained.
No admission limit or watchdog is increased.

Eighteen small emitted-code cases cover skipped/taken anchors, conditional and
intervening conflicting writes, three participating pipes, empty loops,
repeated visits, counter resets and actual scalar-loaded control values. Their
required accesses come from the independent declared-layout interpreter rather
than compiler generators. Both programs are checked for original payload and
control behavior, modeled coverage, matching, acyclicity and causal physical
reuse. All candidate payload orders are contained in their baselines.

A whole-state reference solver checks the native delta certificate on fixed
candidates. An additional structural check rejects reordered service masks even
when their storage-delivery sets cover the requirements. The first version of
that new negative compared storage coverage alone; its preserved failure receipt
and corrected final rerun distinguish those two obligations explicitly. Two individually safe deletions pass separately but their union
fails the joint storage check. An empty-effect case passes storage-only checking
and fails when its separately supplied non-storage path is removed. Refresh
inputs are explicitly declined. The retained-anchor regression ensures that a
deleted service never creates an anchor. Certificate state/transmission bounds
are checked independently for the diagnostic cases.

The Delta checker parses actual emitted commands, verifies complete intended
forward/reverse pairs at each original cut, compares the candidate's services
with the baseline's service subsequence, and verifies unchanged directional
role IDs. Bounded command graphs check the resulting causal reuse. Original
SSA/control dependencies are traced independently from source text, including
frame compatibility across resets. A payload/control comparison removes only
individually unreferenced scalar constants and consistently renames SSA tokens;
negative checks ensure that a changed used constant or predicate is detected.

Thirty-two Delta control scenarios include the configured two outer visits,
K=0..3, both initialization-branch outcomes, all bounded softplus masks and
prefetch/tail choices. Zero/one outer counts are explicitly labeled structural
derivatives. Branch outcomes are supplied, not derived by executing full
numeric Delta. Reservation conflicts are extracted independently of compiler
cell IDs and demand generators, but consume retained reservation descriptions.
They therefore validate coverage/realization of the supplied model, not native
footprint recovery. Scalar cache/entry and target contracts remain required.

## Results on identical inputs and access classes

| Original Delta, qualified `may-not-alias` | Baseline | Candidate |
|---|---:|---:|
| Original payload sites | 78 | 78 |
| Source-pipe OUT facts | 1,144 | 1,144 |
| Cross-pipe services | 218 | 129 |
| SET sites | 436 | 258 |
| WAIT sites | 436 | 258 |
| Barriers, including final ALL | 61 | 61 |
| Total static commands | 933 | **577** |
| Physical IR arithmetic operations | 1,846 | 1,134 |
| Physical IR bytes | 152,683 | 98,701 |
| Generated C++ bytes | 77,343 | 60,001 |

The unpruned physical output is byte-identical to the pinned `7facb6ea8`
baseline. Exactly 89 complete services are removed. The scalar arithmetic
reduction includes 712 unused constants formerly constructed by endpoint-map
materialization; original control computations remain unchanged. Both programs
use the existing six-ID configuration with IDs 0 and 1 per used direction.
Pruning establishes no smaller minimum budget and performs no recoloring.

In the checked Delta scenarios **all payload reachability pairs and excess
completion-to-start counts are unchanged**. For example, the configured outer
count with K=3, `initial=True` and all optional softplus outcomes false executes
2,328 versus 1,348 commands. Retained scalar-related paths still imply the same
payload order. This is a command/control-construction improvement, not a
measured Delta concurrency improvement.

The disjoint-intervening-access fixture gives a strict ordering improvement.
A source pipe delivers X, then touches disjoint Y; a later reader of X no longer
acquires Y's unrelated completion. On the checked four-visit execution:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Executed commands | 65 | 49 |
| Excess completion-to-start pairs | 11 | 7 |
| Removed start/completion reachability pairs | — | 16 |
| Added payload reachability pairs | — | 0 |

This comparison establishes relation containment, rather than inferring it
from lower excess counts. All local and return barriers are retained.
The older X14 826/326 figures used different access/admission inputs and are
not a comparison against this 933-command baseline.

## Construction work and cost

The same original Delta description has three storage classes, four pipes,
89 CFG nodes and 94 edges. Proposal generation tests 218 services, inspects
480 class entries, performs 201 class updates and clears 387 entries. The
single joint check retains 2,262 debt facts, inspects 2,025 incoming facts and
makes 2,406 fact-edge transmissions, with no violation.

A serial descriptive sample separates the stages below (milliseconds; the
receipt pins each measured source/tool version). These are not interleaved
speedup measurements:

| Stage | Baseline | Candidate |
|---|---:|---:|
| Access recovery | 2.416 | 1.608 |
| Phase index | 0.741 | 0.192 |
| Contract/readiness admission | 0.110 | 0.068 |
| Baseline source-pipe solve | 0.127 | 0.083 |
| Baseline plan serialization | 6.449 | 4.269 |
| Protected-path inventory | — | 0.044 |
| Pruning graph/mask preparation | — | 0.030 |
| Local proposals | — | 0.020 |
| Joint verification | — | 0.189 |
| Candidate plan construction | — | 1.215 |
| Physical-map preparation, inclusive | 0.885 | 0.141 |
| Prepared protocol check, subset of map preparation | 0.012 | 0.008 |
| Emission and IR verification | 2.209 | 1.072 |
| Whole compiler subprocess | 265.305 | 64.086 |
| C++ generation subprocess | 666.056 | 464.840 |

Do not add inclusive/subset columns. Other descriptor, CFG, native-discharge and
map-construction counters/timers remain in the receipt. Process startup and
printing are included only in whole subprocess time. Candidate construction
still builds and retains the baseline plan for fallback; serialization costs
more than the proposal scan. These single samples establish no compiler or
device speedup.

| Static repeated stages | Proposal nodes | Service tests | Deleted services | Certificate OUT facts | Fact transmissions |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 2 | 0 | 40 | 52 |
| 2 | 9 | 3 | 1 | 46 | 58 |
| 4 | 11 | 5 | 3 | 58 | 70 |
| 8 | 15 | 9 | 7 | 82 | 94 |
| 16 | 23 | 17 | 15 | 130 | 142 |

Trip counts 1 and 256 produce identical retained counts at each static size.
The 256-trip cases are compilation derivatives, not 256-trip oracle executions.
Fixed-class/fixed-pipe growth is linear on this subclass. Runtime trips are
not unfolded. The implementation additionally pays short mask-validation and
ordered pair-map lookup costs; these measurements do not establish the draft's
expected proposal bound uniformly as the number of pipes varies.

## Remaining scope and paper consequences

| Established result | Remaining premise or limit | Appropriate claim |
|---|---|---|
| One candidate is jointly checked against raw storage debts and explicit protected services. | Sound supplied accesses, baseline admission and native/control contracts remain required. | Optional deletion preserves conservative coverage; no minimum-demand claim. |
| Prepared and actual services retain matching/rearming structure. | Standalone ownership, native command semantics and coherent scalar inputs remain premises. | Fixed-role physical realization is certified separately from logical delivery. |
| Delta static/executed commands shrink; bounded payload order is identical. | Scalar-related services remain deliberately protected. | Less synchronization code; no demonstrated Delta ordering or device speedup. |
| Disjoint-buffer fixture loses four excess completion prerequisites without adding others. | A bounded declared-layout example, not arbitrary-kernel optimality. | The same reusable local rule can improve ordering under its supported structure. |
| Static-size growth is controlled and retained counts do not grow with trip counts. | No general optimal-pruning, general guarded-history or variable-pipe complexity claim. | Compact optional implementation of local proposals plus one fixed-candidate check. |

Asynchronous readiness, arbitrary retained-child protocols and refresh-based
relay pruning need additional supporting contracts. Refining the deliberately
coarse scalar-protection subgraph would require another proof; storage
independence alone is insufficient. Source JSON references are acquired after protection-metadata insertion,
independently of DenseMap bucket capacity. No paper theory file, production dispatcher,
allocation algorithm, expert protocol or native hardware exemption is changed.
No device compilation/execution is available or implied.

## Reproduction

After synchronizing changed files into the established Linux source snapshot,
build the existing test target with `--parallel 1`. The receipt records the
full 34-test filter and final two-test follow-up, source/tool hashes, output
hashes and commands. For direct checks from the workspace:

```bash
.local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --gm-alias=may-not-alias --current-cut-pruned \
  test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto
.venv/bin/python test/lit/pto/Inputs/check_service_pruning.py \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --output .local/service-pruning-reproduction \
  --delta test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto
```

[Machine-readable receipt](frontier-service-pruning-20261021.json).
