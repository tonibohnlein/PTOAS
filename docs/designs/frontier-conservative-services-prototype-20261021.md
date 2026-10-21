# Source-pipe services and qualified current-cut lowering

The projected C++ analysis is integrated at the existing experimental test
boundary. It preserves the origin solver's IN/OUT projection, selected services,
logical/physical command trees and role maps. A restricted, genuinely recurrent
A3 vector invocation now lowers those services to physical PTO IR and C++ through
the existing staging and physical-map services. This does **not** enable a new
production backend or establish whole-Delta realization.

The starting worktree was clean at `29cd96a2af94fd9d9b5c5711561a0222df67b37a`.
The [receipt](frontier-conservative-services-prototype-20261021.json) pins changed
contents, build products, inputs, research references and commands. Artifacts
are under `.local/conservative-services-20261021/`. No device was used. Existing
35-second per-process watchdogs and the aggregate two-worker ceiling were kept.
The completed lossless-cast/Delta probe from the acceptance checkpoint was
reused, rather than repeating the kernel survey.

## Checkpoint 1: unchanged services, cheaper analysis

[SourcePipeServices](../../include/PTO/Transforms/FrontierSynch/Shared/SourcePipeServices.h)
provides a target-independent C++ solver; its implementation lives in the
existing Shared directory. It retains `(class, mode, pipe)` facts in hash sets,
not origin inventories. A new OUT node/fact is queued once; each outgoing edge
receives it once. Incoming killed facts stay in IN, and selection reads IN.
GEN remains present even when that executed node refreshes the same class.
No refresh is inferred on Delta. Origin-specific kills and distance/origin
queries are outside this representation's contract.

The validity argument is projection of the finite least fixed point. Mapping
`(class, mode, origin)` to `(class, mode, pipe(origin))` commutes with union,
GEN and class-wide KILL. It therefore commutes with each iteration from empty
states, including joins, backedges and empty visits. The projected fixed point
and each selected conflicting pipe set are identical. This preserves the
existing conservative service policy; it does not improve its precision.

For h classes, k pipes and V/E CFG vertices/edges, the solver uses expected
`O((1+hk)(V+E) + descriptor preparation)` hash work and
`O((1+hk)V+E)` state/queue/graph entries. There are at most 2hk facts per cut.
This is not a bitset claim or an end-to-end compilation bound. Access recovery,
class construction, CFG/descriptor preparation, solver work and plan generation
are recorded separately. Counters verify that edge transmissions equal
`sum(|OUT[v]| * outdegree(v))`, rather than merely checking retained sizes.

| Supplied Delta inventory | Origin OUT facts | Projected OUT facts | Edge transmissions | Origin pairs avoided |
|---|---:|---:|---:|---:|
| Original coarse classes (3) | 13,376 | 1,144 | 1,209 | 5,497 |
| Pinned research classes (30) | 39,336 | 10,736 | 11,346 | 1,975 |

The second row consumes the already supplied X08 inventory as a comparison
fixture. Class refinement is not integrated. All node states, services, both
once-generated structured plans and role-ID maps agree on both Delta inputs,
20 X10 synthetic fixtures and five scaling sizes. The replayed X09 candidate
retains its certificate, 822-command tree and role IDs; deletion is not
implemented in the new provider. The independent Python origin solver remains
unchanged.

At fixed two classes/two pipes, 16/32/64/128/256 static sites retain
136/264/520/1,032/2,056 projected OUT facts, versus
1,088/4,224/16,640/66,048/263,168 origin facts. The graph and representation do
not unfold runtime trip counts. Separate negatives reject invalid modes,
class arities, refreshed classes and occurrence indices. Supplied-refresh tests
check the algebraic interface, not newly inferred native overwrite proofs.

## Native qualification and Delta's remaining boundary

The test-only [ordered-cut adapter](../../tools/pto-test-opt/CurrentCutServiceChecks.cpp)
consumes the generated structured physical plan. It does not reconstruct a
second service policy from the facts. All original loops, branches, SSA values
and payload operations remain in place. Commands are inserted immediately
before their original consumer operation and inherit its current lexical
control. No earlier branch history or guessed source payload is introduced.

| Premise | Original Delta token/state region | Qualified controlled invocation |
|---|---|---|
| Payload correspondence | 78 original pipe-envelope sites, four pipes; original coordinates retained by input export | Single-phase load/add/expand primitives, original sites and per-pipe order retained |
| Shared legal cut | Abstract plan available; scalar service and compound instruction contracts not generally admitted | Original consumer operation is the shared insertion cut; empty/skipped visits execute no service |
| Current control | Original bounds/branch DAG retained, including loaded scalars; no new readiness theorem | Entry arguments/constants/IV arithmetic; original comparisons execute before their SCF branch |
| Readiness | Scalar memory readiness and enclosing/control obligations remain separate | Existing PhaseIndex/CompletionReadiness qualification must succeed without additional cuts; no carried results |
| Native commands | Unpruned service sets request PIPE_S local barriers; existing target and insertion contracts forbid them | A3 AIV capability table admits MTE2↔V and V/MTE2 local barriers; return uses existing ALL completion semantics |
| Resource ownership | Abstract role maps are not a native whole-Delta resource certificate | Command-free invocation, no hidden events under common physical preflight; explicit eligible pool and distinct directional roles |
| Region boundaries | No imported-command interface or generalized child closure is synthesized here | Initially empty owned notification palette; each service closes matching, and final ALL drains invocation work before return |

A concrete token-loop witness is original Delta site 20, line 153:
`pto.tgetval` on PIPE_S has selected pipes `{S,V,MTE2}`. Its unpruned plan
requests an S local barrier. A smaller corresponding whole-input witness is
site 9, line 121. `getSyncBarrierAvailability(a3,AIV,PIPE_S)` is false, and
LogicalInsertion explicitly rejects scalar barriers. Dropping those services
would need another justified native-protection/pruning argument and is outside
the first checkpoint. This is not evidence of insufficient flag capacity or
mathematical class nonmembership.

The experimental emitter also deliberately refuses other Delta instructions
outside its scoped primitive/cut contract, rather than silently importing
compound or scalar effects. Its ordinary refusal is
`current-cut: unsupported command/compound/control contract`; the independent
service inventory identifies the scalar-barrier obstruction behind any proposed
broader admission. The analysis result and existing production completion route
remain available. No test capability table is installed into production.

## Ordered emission and resource argument

X06's logical plan places all selected publications, a possible local barrier
and acquisitions at the current cut. X07's physical plan instead retains the
ordered sequence FSET, FWAIT, RSET, RWAIT for each selected cross-pipe service,
then its local barrier. The existing general endpoint materializer groups all
SETs before WAITs at a cut, so feeding X07 through that batching contract would
change the protocol. The small experimental adapter retains ordered commands
inside a FunctionStaging transaction and uses the common PhysicalRecordAllocation,
planPhysicalEndpointMap, emitPhysicalEndpointMap and emitPhysicalCommand services.
No allocation algorithm is changed or bypassed: the new producer supplies the
specified X07 role assignment and the existing mapping planner validates and
re-expresses its constant maps.

For consecutive executed services p→q, the acknowledgment supplies
`FWAIT_i → RSET_i → RWAIT_i → FSET_next` and
`RWAIT_i → FSET_next → FWAIT_next → RSET_next`. Native WAIT gating on the
relevant pipe supplies the inter-service steps. Distinct forward/reverse roles
in each direction therefore provide consumption before republication. Skips,
empty visits and counter resets do not reset this argument: next means the
next executed service. Each service's identity is its consumer site, enclosing
occurrence tuple and source pipe. Constant physical maps do not replace that
logical identity.

Roles use at most two **eligible IDs per direction** in this isolated scope.
The tests map role ordinals to IDs 3 and 5, not an assumed globally free 0/1.
The InsertSync `reservedEventIdNum` table was inspected: its entries are counts
subtracted from capacity, not directional capacities. Its V↔S reservations are
not used by this non-scalar scope. Common preflight rejects resident/hidden
notification users, and the adapter rejects command-bearing children instead of
flattening them. A different target, insufficient pool, or unsupported boundary
retains refusal. This does not authorize palette sharing with arbitrary callers.

Closed matching alone does not finish payloads or commands. The final ALL is a
whole-invocation completion treatment, separate from flag consumption. No
cleanup is moved to an arbitrary child exit and no exact child protocol changes.

## Independent emitted-code checks and cost

24 bounded native cases cover zero/one/several inner and outer visits, skipped
loads, taken/empty branch arms, nested resets, two independent storage classes
and a completely empty invocation. The checker interprets source byte ranges
independently of compiler cells and generators. It verifies payload correspondence,
required-order and selected-prefix coverage, command-graph acyclicity, full
payload-event relation differences, matching, eligible directions and causal
consumption before republication. It compares actual emitted commands with the
once-generated static plan, including guards through original control execution,
service order and the mapped role IDs. Supplied non-storage readiness in this
scope consists of ready entry values and pure IV/control arithmetic; it does not
include an unproved asynchronous scalar condition.

In the two-class all-taken example, 20 payloads and 67 commands add 18
completion-to-start prerequisites over the declared byte-conflict reference.
The first-load-only example has 16 payloads, 47 commands and 14 excess pairs.
There are no removed selected-prefix event pairs in any case. The tested
physical relations happen to equal their selected prefix relations; this is
finite evidence only. X07 can add order through acknowledgments on other
programs, as its retained research evidence demonstrates.

| Static add stages | Flags | Barriers including return | Added ID arithmetic | Original control tests |
|---:|---:|---:|---:|---:|
| 1 | 8 | 3 | 16 | 1 |
| 2 | 12 | 4 | 24 | 1 |
| 4 | 20 | 6 | 40 | 1 |
| 8 | 36 | 10 | 72 | 1 |
| 16 | 68 | 18 | 136 | 1 |

At each size, changing inner trips from 1 to 256 preserves these static counts;
only the literal bound changes. No new guard expressions or historical state
are constructed. C++ generation succeeds for the nested conditional example
(3,554 bytes). This uses the pinned retained compatibility C++ backend, not a
newly built device compiler. Native-map preparation, access recovery, phase
indexing, class/CFG preparation, solving and emission/verification have separate
diagnostic timings. These single observations are descriptive, not speedup
measurements. Coverage and compactness imply neither a small approximation
factor nor a runtime benefit.

## Regression and preservation evidence

The focused gate passed 28/28 tests, including the prior 26 acceptance tests.
The two prototype tests were refreshed after adding the final two-class/empty
checks and diagnostic instrumentation. Compact-K/replay, zero-inner-body
barriers, access coverage, unavailable selectors and completion children remain
covered. No production dispatch, SCC transfer, allocation policy, exact
recognizer or expert protocol changes.

The production pto-test-opt binary is byte-identical to the accepted starting
binary (`1fc013d5…`). Across nine preserved original kernel/policy cases,
32/36 logical/support/allocated/C++ artifacts are byte-identical. The three cases
with receipts pinning that same starting optimizer (lossless cast both aliases
and Delta may-alias) preserve all 12 artifacts byte-for-byte. Five of six
Elementwise/GroupNorm/Persistent cases from the older checkpoint also preserve
all four artifacts. Persistent may-not-alias differs from that older-tool receipt
in Boolean conjunction construction order. A scope-aware comparison verifies
identical logical, allocation-support and physical IR after comparing only
non-escaping associative AND chains; raw files and the differing C++ remain
archived, with no golden changes. This is not a new synchronization policy.

Ruff F checks, the native build and whitespace checks pass. The compliance
prefilter's twelve brace findings were inspected: every body is braced; its
regex misreads nested/multiline conditions. Actual overlength additions were
fixed. The receipt retains the raw findings and individual dispositions. No
full external static analyzer or device validation is claimed.

## Reproduction

Use the prepared Linux source snapshot/build; the worktree's old macOS workspace
metadata is historical and was not used to select a new toolchain.

```sh
cmake --build .local/section8-refactor/source-pinned/candidate-build \
  --target pto-sync-input-test pto-test-opt --parallel 2
.venv/bin/python test/lit/pto/Inputs/check_source_pipe_services.py \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --input test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto \
  --output .local/conservative-services-20261021/projected
.venv/bin/python test/lit/pto/Inputs/check_current_cut_services.py \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --python .venv/bin/python \
  --cli .local/section8-refactor/source-pinned/candidate-build/tools/ptoas/ptoas \
  --output .local/conservative-services-20261021/native
.venv/bin/python .local/conservative-services-20261021/replay_x10.py
```

The receipt pins the local X10 replay driver and its research inputs. That
replay consumes existing refined classes and a proposed deletion solely to
check unchanged results; it does not integrate either optimization.

## Paper consequences and remaining scope

| Established result | Remaining premise or limitation | Appropriate claim |
|---|---|---|
| Source-pipe least fixed point equals origin projection | Fixed classes/pipes, class-wide supplied KILL; access recovery charged separately | Compact conservative service selection need not construct origin pairs |
| Generated X06/X07 plans and role maps preserved | Native cuts, control and commands require independent qualification | Analysis-cost improvement preserves the selected policy |
| Real recurrent A3 non-scalar invocation emits physical IR and C++ | Isolated ownership, ready control, supported primitives and directional commands; no device validation | A scoped implementation bridge exists, not general token realization |
| Independent byte/command checks establish bounded coverage and reuse | Local/vector layout and native command semantics remain supplied contracts | Host-model validation, not universal native correctness |
| Two-class examples expose extra prerequisites | Whole-prefix publication and acknowledgments can serialize substantially | No minimum-demand, approximation-factor or runtime-speed claim |
| Delta projected analysis succeeds | Unpruned scalar services and additional scalar/compound/control contracts block native emission | A realization boundary, not mathematical intractability or proven flag scarcity |

Next work should qualify a specific scalar protection/cut contract or extend a
command-bearing boundary interface, while preserving the analyzed service policy
unless a separate deletion proof is supplied. Service pruning, refresh inference,
new recognizers, global dispatch and production enablement remain deferred.
