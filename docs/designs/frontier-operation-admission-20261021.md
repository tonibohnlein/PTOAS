# Contract-based operation admission for experimental current-cut realization

The experimental adapter now consumes existing effect, phase, readiness and
ownership contracts instead of a payload-operation name list. Scalar GM pointer
loads can retain their runtime values and control uses. The original Delta input
now emits physical PTO IR under the existing may-not-alias contract, with its 78 payload sites and
original structured control intact. This does not enable production dispatch or
qualify native numerical execution of Delta.

The [receipt](frontier-operation-admission-20261021.json) pins the actual starting
dirty tree, final sources/tools and commands. Work continues at
`29cd96a2af94fd9d9b5c5711561a0222df67b37a`; that revision alone does not describe
the changes. The preceding source-pipe and scalar-native checkpoints remain
uncommitted and are preserved. Fresh evidence is under
`.local/operation-admission-20261021/`. Earlier campaign directories were not
reused as output destinations. All work is host-only.

## Contract boundary

| Requirement | Existing provider | Admission/realization consumer |
|---|---|---|
| Read/write semantics | MLIR `MemoryEffectOpInterface`, contextual declarations retained by `SyncInput::effectsFor` | Every declared non-descriptor effect must be a represented read/write. An empty translated inventory is not evidence that an operation is pure. |
| Storage identity and coverage | `SyncInput::buffers`, phase use/def records, existing `CompletionStorage` reservations | Every declared storage operand must have all its translated alternatives represented in the appropriate phase mode. Unresolved geometry uses existing conservative classes. |
| Pipe and events | `OpPipeInterface`, `SyncInput` pipe envelopes, `PhaseIndex` | Original pipe agrees with one noncompound phase. Multi-phase/envelope cuts remain unavailable. |
| Pure and descriptor work | MLIR `isMemoryEffectFree`, PTO `hasOnlyDescriptorEffects` | Preserve original values without creating fictitious memory payloads. `alloc_tile` retains its explicit declarative exception because it lacks a purity interface; this does not exempt other unknown operations. |
| Legal cut and control | Original single-phase operation under supported SCF | Ordered services remain immediately before the original payload. For/if/yield stay supported; loop-carried values, calls and other regions need additional contracts. |
| Result/control readiness | `resultAvailability`, `PhaseIndex`, `CompletionReadiness` | Native synchronous scalar results can feed payloads, branches and loop bounds. Additional completion cuts and non-scalar payload results remain explicitly unavailable. |
| Scalar GM read visibility | Existing cache-maintenance/lowering contract and `SyncStorageEffects::independentOfOtherPhases` | Supported GM reads are invocation-read-only under the supplied alias policy. A possible conflicting writer needs further cache/visibility evidence; event availability is insufficient. |
| Native order | Existing scalar protection and prototype native-order query | Every scalar phase must qualify, including earlier phases in a mixed scalar sequence. Consumer qualification alone is insufficient. |
| Events, target and ownership | `preflightPhysicalOutput`, existing macro models, capability and directional reservation APIs | Hidden/manual protocols, unsupported directions and incompatible palettes retain their existing checks. |

[CurrentCutContracts.h](../../tools/pto-test-opt/CurrentCutContracts.h) is a small
test-only adapter over these providers. It does not register another set of
operation semantics. The payload name list and broad arith-dialect exemption
were removed from
[CurrentCutServiceChecks.cpp](../../tools/pto-test-opt/CurrentCutServiceChecks.cpp).
Diagnostics identify the operation, source location and unmet contract.
`--current-cut-contracts` reports all discoverable operation/control issues
without emitting code. Physical preflight and completed command-direction/map
checks remain separate; an empty diagnostic inventory is not an allocation
certificate.

Known storage with unresolved addresses is covered by existing classes: unknown
GM addresses use the known GM domain rather than distinct SSA-root classes;
unknown local reservations touch all relevant classes in their space; an unknown
space uses the established all-class behavior. Unknown read/write semantics or
an unrepresented declaration still refuses. A broad address class cannot repair
an omitted effect, unsupported completion, illegal cut or missing readiness.
It also cannot establish cache freshness. The repository's
[memory-consistency design](ptoas-memory-consistency-design.md) keeps GM cache
maintenance explicit; the pointer-load lowering emits a plain subscript without
automatic invalidation. The official [cache-control documentation](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0177.html)
also distinguishes cache maintenance from ordinary synchronization.

## Scalar pointer-load qualification

`PTOLoadOp` declares PIPE_S through `OpPipeInterface`. Its effect implementation
uses `addLinearAccess` and falls back to a read of the pointer when exact geometry
is unavailable. The noncompound scalar-result contract in `SyncInput` states
that the scalar access completes before its result becomes an ordinary value.
Existing lowering emits `PTOAS__PTR_LOAD`; it does not replace the runtime load
with a constant. These facts already existed and were blocked by the name gate.

The prototype native-order query now admits the relevant A3/AIV GM PTO-pointer
form with an integer, index or floating scalar result. Vector-valued loads are
not admitted by that rule. GETVAL/SETVAL retain their existing vector-tile scope.
Other scalar forms require their own existing/supplied semantic evidence.

This is separate from visibility of the location read. The final admission
requires invocation-read-only GM storage, proved by the existing alias/effect
provider, plus the source model's coherent initialized-input entry premise.
That entry premise is not proved by this pass. With no possible GM writes,
a single effect-inventory scan discharges the check. Otherwise, the existing
cached independence query is used for each scalar GM read. Delta's three input
reads qualify under may-not-alias; that policy requires the caller to supply
disjoint distinct GM roots. It is not inferred from their SSA names.
Potentially overlapping writers remain an explicit `gm-read-visibility` refusal.
The current sufficient check does not prove temporal noninterference or
guard-dependent cache freshness, even when such a stronger proof might exist.
No cache-maintenance commands or new caller bounds are invented.

The admission argument uses the staged inventory: each payload has one original
phase; every declared storage alternative appears with its read/write mode;
each scalar phase has the scoped synchronous-result/native-order contract;
and required value/control readiness has no unsupported completion cut. The
separate visibility check excludes potentially conflicting invocation-local GM
writes from admitted scalar reads. These facts qualify the existing protocol
without modifying its selected obligations or treating missing information as
absence. They rely on sound provider declarations and the stated entry premise.

The supplied scalar-serialization premise applies to the mixed admitted
sequence. Since **every** scalar phase is qualified before realization, the
previous executed scalar completion precedes the next scalar start, including
across unlike opcodes, skipped branches, loop boundaries and resets. Native
completion order covers the rest of the scalar prefix. No scalar barrier is
emitted, and no cross-pipe obligation is deleted.

The source-pipe fixed point, selected services, acknowledgment roles and cuts
are unchanged. FSET/FWAIT/RSET/RWAIT order, directional reservations, eligible
IDs and whole-invocation entry/return ownership remain enforced. This is still
the standalone experimental protocol, not general physical child import.

The accumulator-chain rule is also preserved: matrix shape threshold,
initialization/reset, accumulator identity, interference and visit scope remain
in the existing providers. In particular, a FIX-pipe accumulator access that
interrupts a protected matrix chain is not silently exempted. No universal
matrix/FIX completion-to-start rule is introduced.

## Bounded checks and their dependencies

The new lit driver executes original pointer loads against an independent
declared-layout memory model. Loaded values drive the original index casts,
loops and branches. Values are supplied to the bounded command interpreter only
after lowering, as observations of executed loads. They never become
compiler-time bounds, replacement payloads or guard qualifications.

Eight positive cases include negative, zero and short loaded counts, mixed
load/GETVAL/SETVAL sequences, repeated outer visits, counter resets, guarded
vector consumers, and a subtraction primitive absent from the old name list.
The memory model observes changed scalar contents rather than a constant stub.
Sixty-four bounded byte-interval checks verify coverage by the retained reservation
summaries, preserving mode and domain. GM summaries are conservative; these are
not claims of exact native GM selectors or native instruction footprints.

For those cases, actual emitted commands pass payload/control correspondence,
modeled prerequisite coverage, full event-relation comparison with the selected
native-aware plan, closed matching, acyclicity, eligible-direction checks and
causal consumption-before-republication. The observer and event-graph oracle
do not call the C++ native capability predicate. They share the stated scalar
hardware premise and declared layout, which remain explicit assumptions.

Negative checks reject vector-valued loads, an operation lacking a consumed
purity/effect-phase contract, loop-carried values, and two GM-writer variants
without a cache-visibility certificate. These are valid parsed
inputs with contract diagnostics, not accidental parser failures. Removing the
four-command V-to-S service from the actual mixed load/GETVAL example exposes
the required vector-completion-to-scalar-start edge. The preceding vector-to-S
and vector-local-barrier negatives remain in the retained gate. Earlier coherent
host-memory transfer examples are retained as model-only evidence, not final
native-admission successes: completion ordering alone did not qualify their GM
cache visibility.

## Delta result and practical inventory

All 259 original operations are inspected once for admission, using one shared
input/phase preparation. All 152 declared effects are checked. No operation or
control admission issue remains under may-not-alias. Three existing independence
queries establish the scalar GM reads' read-only condition. Under may-alias,
that sufficient certificate is unavailable; this is not a capacity obstruction
or proof that the original execution needs cache maintenance. The actual emitted SCF
tree and payload order match the original inventory, and every static command
block matches the once-selected physical plan. There are 218 selected cross-pipe
services, 75 selected self-prefix obligations, 15 native scalar discharges,
872 SET/WAIT sites and 61 barriers including return ALL: **933 commands**.
IDs 0 and 1 are used within the existing six-ID configuration; this establishes
a qualified assignment for the selected protocol, not a minimum budget.

Five bounded control-abstraction scenarios check empty invocation, zero token
iterations, initialization/no-initialization, both softplus branches and a
two-token prefetch/tail case. The original outer activation guard is retained.
Zero/one outer counts are structural derivatives of the configured two-iteration
outer loop; the two-outer case retains that configured count. These tests do not
establish new caller bounds or claim that every branch mask is numerically feasible.
The checker parses the actual emitted command blocks, unfolds only the reference
execution, and checks 3,841 reservation-conflict pairs plus matching, acyclicity
and causal reuse. Its effects come from the existing reservation descriptions,
not compiler cell memberships or generated demands. Therefore this independently
checks propagation/realization of those effects, not their extraction from native
instructions. Independently traced original SSA/control dependencies are added
to payload and command events, with enclosing-frame compatibility checked for
each selected scalar definition. Branch outcomes are supplied control
scenarios, **not** numerical execution of the full Delta memory computation.

The preceding `pto.load` refusal was an unused existing effect/phase/readiness
provider. Removing the name gate also allows ordinary single-phase vector
operations whose contracts already qualify. Remaining limits are distinct:

| Layer | Current result or remaining premise |
|---|---|
| Operation/control admission | No unmet contract in the pinned Delta inventory. Unsupported forms still have explicit refusals. |
| Effect coverage | Conservative allocation/space summaries remain; interface declarations are not a native footprint proof. |
| Logical/physical realization | Actual command blocks and bounded control scenarios pass; the standalone ownership and target premises remain required. |
| Numerical kernel behavior | Not established by the reservation/control checker. |
| Device behavior | No device compilation or execution is available or claimed. |

## Scaling, accounting and preservation

| Static vector stages | Admission operation visits | Effect visits | OUT facts | Native discharges | Ordered sites before return | Physical PTO bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21 | 7 | 41 | 2 | 13 | 2,852 |
| 2 | 22 | 10 | 46 | 2 | 18 | 3,590 |
| 4 | 24 | 16 | 56 | 2 | 28 | 5,066 |
| 8 | 28 | 28 | 76 | 2 | 48 | 8,018 |

Runtime trip counts are loaded from memory, so the compiler sees no specialized
count. The named 1/256 scaling samples compile identical original IR and retain
byte-identical output for each stage count; they are not executions of 256 trips.
Independent bounded executions cover the counts listed above. Production
representations are not unfolded to validate them. Access/phase preparation,
contract/readiness checking, class/CFG construction, solving, serialization,
native-discharge queries, physical-map construction and emission have separate
counters or timers. Narrow native/map timers are subsets of the existing
inclusive preparation timer and must not be added to it. Single timings are
descriptive, not speedup evidence.

Visibility admission records one inventory scan and, when possible GM writes
exist, one cached independence request per scalar GM read (three for Delta).
Those requests can inspect other effects; the counters do not establish a
uniform linear bound for arbitrary aliasing inputs. Read-only fixtures use the
no-GM-write fast path. Admission does not rerun access recovery per operation.

The fresh final focused gate passes **33/33** after the visibility qualification
and all source changes. Earlier subset receipts are retained as intermediate
evidence. C++ generation succeeds
for the load-driven example (2,186 bytes) and original Delta physical IR
(77,343 bytes), using the pinned existing compatibility CLI/native libraries.
This is source generation, not device compilation.

The production optimizer and allocation/hardware providers are unchanged.
Nine existing kernel/alias cases succeed at all four retained stages. Against
the preceding scalar checkpoint, **32/36** artifacts are byte-identical. The
four differences are Persistent GEMM may-not-alias conjunction ordering under
the same optimizer binary. A scoped, nonescaping Boolean-AND comparison proves
equality of its logical, saved and physical PTO artifacts; no golden was
updated. Its C++ is recorded as different, without claiming a separate C++
equivalence proof. This is existing output-order variability, not evidence of
a production algorithm change or a measured speedup. All previous raw receipts
remain unchanged; the new preservation runner uses a fresh destination.

Ruff F checks (Python 3.12 target), native builds and `git diff --check` pass.
Brace-prefilter findings are reviewed individually with source excerpts; their
complete conditions have braced bodies. The long test literal was split while
checking byte equality of generated input. Full static-analyzer, sanitizer and
device qualification are not claimed.

The receipt records these results, unchanged hardware/allocation files and final
source/tool identities. Reproduction uses the same configured Linux
snapshot as the preceding checkpoint; synchronize changed source/test files
before rebuilding with `--parallel 1`. Tests also run serially, leaving the
other aggregate machine worker available for paper work. Ordinary production
dispatch, allocation algorithms and unrelated exact-analysis routes are unchanged.
New compiler invocations use the existing 35-second watchdog; the inherited
structured-trace helper retains its existing 120-second watchdog. No watchdog
or admission limit was raised.

The brace prefilter exits nonzero on 25 regex matches. Each was inspected against
the complete condition and has a braced body; the receipt records these as
false positives with excerpts. This is a reviewed prefilter result, not a clean
analyzer run.

For direct reproduction after synchronizing and rebuilding the pinned Linux
source snapshot, run these commands from the workspace (the receipt records
the complete final gate filter and other commands):

```bash
.local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --gm-alias=may-not-alias --current-cut-contracts \
  test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto
.local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --gm-alias=may-not-alias --current-cut-output \
  test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto
.venv/bin/python test/lit/pto/Inputs/check_operation_current_cut_services.py \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --output .local/operation-admission-reproduction
```

## Paper consequences

| Established result | Remaining premise/limit | Appropriate claim |
|---|---|---|
| Existing interfaces support contract-based admission and scalar load-to-control realization. | Effect and target contracts must actually describe the instruction. | The model construction has an executable path for these admitted operation contracts. |
| Native scalar order discharges selected self-prefixes across mixed scalar operations. | No cross-pipe visibility follows from serialization alone. | Obligation selection and command realization are separate stages. |
| Original Delta emits an unpruned standalone physical plan; static blocks and bounded control scenarios pass. | Native footprints and full numerical/device execution remain unqualified. | Host-model implementation coverage, not universal native correctness or performance. |
| Retained facts and code do not depend on loaded runtime trip values. | Recovery, representation and admission costs remain separately charged. | The compact static construction is preserved; no end-to-end speedup is inferred. |
| Existing accumulator protection and production outputs are retained. | This does not create a new whole-prefix protection rule or child resource theorem. | The operation adapter reuses the existing architecture rather than replacing its algorithms. |

Neither the finite/core-chain native graph nor its exact-cover theorem is
extended by this adapter. Existing storage-specific protection remains distinct
from the stronger whole-prefix event relation used by this experimental
realization.
