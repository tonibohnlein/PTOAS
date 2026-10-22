# Retained children inside conservative prefix services

Starting point: `5af91f65434fb4dc0ea8f909c7bf50fd2d0ffa30`, clean worktree.
This design precedes dependent source changes. No default selection, allocation
algorithm, native exemption, expert plan or pool size changes.

## Provider contract

Initially admit one proper whole `scf.for` invocation per function, using the
existing immutable `CompletionInlineChild` provider. It may execute repeatedly
in its original enclosing loops/branches; it is never an open iteration. The
provider supplies the retained source, original root, closed notification
matching and within-visit assignment. Its existing append operation preserves
all grouped labels, tuple arities, enclosing visits and guards. A separate
export lists participating pipes and all read/write modes in the parent's
invocation-wide `CompletionStorage` classes, including unknown GM effects.
These are sound may-effects, not exact selectors or last writers. All effects
remain pending at exit. Missing selectors remain unavailable.

The boundary evidence is confinement: every child command is inside the
original whole-loop root, under its original enclosing guard/frame. Thus every
previous visit's issued WAIT on pipe t precedes the next entry SET issued on t,
and every new child SET on pipe s follows the entry WAIT on s. This is evidence
about original pipe command order, not transitive closure of guarded metadata.
Empty entry and closed issued matching remain provider premises; closure is not
payload completion or physical rearming. Opaque/nested completion owners with
unknown notification boundaries are refused by this export.

## Parent construction and proof

Retain the existing finite original-CFG may-access analysis (no KILL or trip
unfolding). For each participating target pipe, union the baseline service
sources at child sites and emit those services at child entry. Also request
`t -> s` for every actual child resource direction `s -> t`. The ordinary
ordered acknowledgment quartets implement these requests. Required same-pipe
prefixes use existing barriers/native discharges. No added ALL is used to
make an import fit.

Internal requirements are covered by the unchanged qualified child. Every
outside-to-child conflict is covered at entry: its source fact reaches a child
site in the raw no-KILL CFG; the union includes its source pipe. Every
child-to-outside conflict remains represented at the original outside consumer,
where the baseline source-pipe prefix includes the actual child work. Sources
from internal/backedge paths in an entry union only strengthen this sufficient
construction; they are not claimed to be work already performed in this visit.

For each reused child ID in direction s->t, the old consumer WAIT on t precedes
entry SET(t->s); its matching entry WAIT on s precedes the next child SET(s->t).
A skipped child executes no child endpoints; a zero-trip child may still execute
a closed entry wrapper. The wrapper stays in the original enclosing branch,
so guard compatibility follows structurally. Its reverse acknowledgments rearm
parent roles under the accepted standalone protocol. Parent roles are reserved
from eligible directional IDs excluding the union of child palettes and target
reserved tails. Shared role/child IDs, missing reverse directions or unavailable
cuts cause refusal. No caller bounds or new coloring is inferred.

The prepared child is decoded by the existing allocator before parent commands
are emitted. Compare its endpoint maps with the original decoded assignment;
a representation conversion may not recolor it. Parent and child commands are
then checked together independently on bounded executions. All work is within
the existing whole-module transaction; any refusal preserves original IR.

Pruning around an imported child is initially unavailable. The raw solver and
allocation algorithms remain unchanged. The explicit `prefix-child-region=N` option names an original region; no
automatic region search is performed. A typed provider callback supports
existing library callers in the staged frame, mutually exclusive with that
region request. Both use the same adapter. An explicit opt-in requests the child
capability; no failed exact export authorizes automatic conservative fallback.

## Validation plan

First check parent-before/after accesses, shared storage, pending child payloads,
zero/one/multiple visits, conditional skips and enclosing counter resets on a
small fixture. Independently interpret original accesses and actual emitted
commands; compare coverage, all command cycles, matching, directional capacity,
causal reuse and both directions of prerequisite inclusion. Negative cases cover
eligibility/collisions, absent directions and boundary/guard evidence, including
whole-module rollback. Then exercise retained Qwen K with its existing symbolic
recipe and compare with the previous prefix and qualified completion plans.
Retain compact-K/replay, zero-inner-body, default three-kernel and hardware
regressions. Pin source snapshot, Linux build, Python PATH, commands, watchdogs
and hashes in the companion receipt. Device behavior remains unqualified.

## Implemented checkpoint

The reusable adapter passes the independent fixture and imports the original
Qwen down-projection whole K loop under six IDs. The explicit option is:

```bash
pto-test-opt --pto-frontier-analysis='conservative-prefix=true physical-output=true prefix-child-region=5 gm-alias=may-not-alias' test/lit/pto/Inputs/pypto_region_survey/qwen-down/input.pto
```

Region 5 is an explicitly supplied original region identity for this capture;
production contains no kernel-name or benchmark-bound recognition. Other
captures must supply their own original whole-loop region. A source body/open
iteration is refused before qualification. A typed `PrefixOptions::childProvider`
can instead return a retained provider in the staged function's original frame.
The option and callback are mutually exclusive. No automatic selection is added.

The initial family is one eventful, closed proper counted-loop invocation per
function, potentially repeated inside its original enclosing loops and guards.
Calls, opaque completion owners, open iterations and additional independently
owned child scopes remain outside this adapter. Event pipes must have modeled
participating phases. Decoder conversions must pass existing total-map
equivalence; a feasible differently colored plan is not substituted. All owned
parent roles and child palettes require empty, exclusive whole-invocation entry,
as in the accepted standalone invocation contract. The wrapper supplies rearming
between child visits; final return ALL supplies payload completion separately.
This is not a certificate for arbitrary child control/readiness contracts or
all event-map representations.

### Provider evidence and implementation

| Information | Existing evidence reused | New consumption |
|---|---|---|
| Internal selected order and commands | `MathematicalResult::completionChild`, sequence/visit owner and fresh preparation callback | Preserve the same family inventory and shared materializer; no demand reanalysis per runtime sample |
| Modes and pending work | Original `SyncInput`, `PhaseIndex`, invocation-wide `CompletionStorage` | `CompletionInlineChild::prefixBoundary` groups every modeled effect by pipe in that same class universe; all remain pending |
| Coordinates and exact guards | Existing `append`, grouped-label translation, enclosing visits and original cuts | No new guard replay, counter state or occurrence remapping |
| Closed issued matching | Qualified provider contract, separately checked symbolic child | Confinement alone is used only for boundary order, never as a matching proof |
| Within-visit physical use | Existing decoded assignment and `resourceFootprint` | `preparePrefixImport` validates every possible label against the same total physical map |
| Cross-visit boundary events | Every child command confined to its original whole-loop invocation | Entry reverse services establish previous-consumption-to-next-publication paths |
| Parent resources | Existing constant F/R roles, eligible pool and reserved-tail query | Filter out every owned child ID in the same actual direction |

`Shared/CompletionPrefixChild.cpp` owns the provider adapter.
`Shared/PrefixChildren.cpp` requests the existing provider, checks scope/alias
compatibility and constructs entry unions. `PrefixPreparation.cpp` reserves
roles and builds commands with the existing `PrefixRecipe`. `PrefixServices.cpp`
lowers the child using `insertLogicalSynchronization` and
`allocatePhysicalEventIds` before emitting disjoint parent commands. The same
module-owned transaction covers every stage and every function.

The translated visit assignment stores its first absolute ID in the tuple base.
The adapter restores the table-local offset when comparing maps. Physical tuple
coordinates and logical record-selector labels remain distinct. It checks all
records, including inactive ones, using the existing map planner/equivalence
service; dynamic grouped selectors are preserved. No first-label substitution,
record deletion, new coloring or general allocation construction is introduced.

The parent still visits the original static child CFG for its raw no-KILL
analysis. It omits parent commands inside the child and hoists their per-target
source union to entry. The per-pipe summary supplies participating pipes and
pending class modes. This is a bound in the supplied static CFG and retained
child representation, not a claim that parent work depends only on a compact
summary's size. No K-dependent loop unfolding is introduced.

The selected conservative protocol is now the child plus its entry wrapper and
outside prefix services. The existing `covers-selected-prefix-order` diagnostic
refers to that newly selected protocol; preservation of the previous stronger
all-primitive prefix order is not promised. Modeled coverage and preservation of
the child's internal realized order are separate properties. Exact selectors
for omitted storage domains remain unavailable. No root exactness or global
minimum-demand guarantee is promoted by a successful import.

### Validity and tested scope

Incoming external conflicts are delivered at entry because the raw no-KILL
analysis retains their source facts. Internal conflicts are served by the
unchanged child. Outgoing effects remain in the raw transfer until an outside
consumer's service completes the relevant source prefix. Native/readiness
admission is unchanged; non-storage paths are not inferred from storage.
The adapter explicitly rejects a refreshed/KILL baseline and child-aware
pruning. Parent and child alias policies and original invocation frames must
agree. Every command endpoint lies inside the child; parent wrapper cuts lie
before the original root, under the same enclosing structural control.

For every reused child ID in direction s->t the uniform boundary argument is:

```
old WAIT on t
  -> entry SET(t->s)
  -> entry WAIT on s
  -> new child SET on s
```

The first and last steps use original same-pipe command order and whole-visit
confinement; the middle step is an actual supported flag handoff. Closed
matching establishes the old intended consumption. Parent and child palettes
are disjoint, and each parent quartet has the existing F/R rearming proof.
These facts hold conditionally whenever the relevant publications execute;
no transitive closure of differently guarded metadata is assumed. Source
contiguity and reference-forward qualified commands keep entry/outside services
from introducing paths through a child interior. Original SCF guards, controls
and visit coordinates remain in place.

The supplied two-bank fixture has parent writes before the invocation and a
parent extraction afterward, with shared storage. Its last child payload can
remain incomplete after every child notification has been consumed. Six bounded
cases cover empty/one/multiple visits, skipped invocations and an enclosing reset
that repeats the inner visit counters. At six child invocations, excess
completion-to-start pairs fall **24 -> 6**, with **18 removed and none added**.
Its two-ID child palette is preserved; parent roles use disjoint IDs.

Five fixture negatives cover child/parent palette exhaustion, wrong tuple arity,
a guard prepared inside the child moved to an enclosing cut, mismatched alias
contracts and unavailable target pipes/directions. Each checks unchanged original
IR on refusal. Four production negatives cover body/open scope, pruning,
insufficient parent roles and a late module failure after successful Qwen import
in the first staged function. No partial commands or success reports escape.
The sufficient resource refusal is not a general infeasibility theorem.

The unmodified automatic small-loop mathematics probe selected sequence analysis
but lacked an allocation certificate. Its refusal is preserved. The independent
fixture uses the already established supplied two-bank certificate; it does not
claim new automatic certificate production for that natural input.

## Qwen outcome and ordering

The unchanged retained K owner has 171 families, 282 prepared endpoint sites,
138 allocation records and 27 grouped endpoint sites (the existing producer
inventory). The new boundary has three participating pipes, five shared storage
classes and four directional child palettes. The inspected root mathematical
arena has 236 nodes and 105 ports. Inventory rows share/inclusively account for
state and are not summed as unique expression or storage totals.

| Actual direction | Retained child IDs | Parent F/R IDs |
|---|---|---|
| M -> MTE1 | 0,1 | 2,3 |
| MTE1 -> M | 0,1 | 2,3 |
| MTE1 -> MTE2 | 0,1,2,3 | 4,5 |
| MTE2 -> MTE1 | 0,1,2,3 | 4,5 |

Every role is checked against directional eligibility and reserved tails. Other
parent directions retain their own eligible IDs. The child returns pending
payload effects; notification closure is not payload completion. This is a
sufficient six-ID construction, not a minimum-budget result.

Nine bounded whole-program executions pass independent declared-layout access
coverage, command acyclicity, matching and causal physical reuse. They include
zero/short K, initialization/accumulation, odd tails, first/last block and repeated
outer visits. Eight additional detailed cases compare the actual imported
commands against one original symbolic child: every executed endpoint retains
its gap, direction, guard activity, record mapping and invocation identity.
**225,792** within-visit comparisons cover all four start/completion combinations
and preserve the child's existing realized order. These checks use original
access operands and an independent command graph, not compiler generators or
cell partitions. A supplementary per-pipe summary check uses the qualified
one-class-per-memory-space enclosures of these fixtures: all 80 observed
small-fixture effects and 1,002 observed Qwen effects (13,378 intervals) retain
their declared space and read/write mode. This check explicitly relies on that
coarse class-domain contract; it is not general fine-geometry recovery. They do not execute numerical matmul or validate native
instruction footprints/cache behavior.

The caller K=17 last-block sample (prepared count `%arg4=17`, rather than a
17-element matrix dimension) has:

| Metric | Previous prefix | Prefix + child | Qualified completion |
|---|---:|---:|---:|
| Executed commands | 2,152 | 535 | 382 |
| Excess completion-to-start prerequisites | 1,123 | 32 | 23 |
| Static SET / WAIT / barrier sites | 124 / 124 / 32 | 177 / 184 / 27 | 121 / 128 / 21 |
| Full-function arithmetic operations | 564 | 2,763 | 2,539 |
| Generated C++ bytes | 35,944 | 68,013 | 62,393 |

The primary same-policy comparison removes **1,091** prerequisites and adds
none in that sample. All nine checked comparisons against the previous prefix
have zero added pairs. This is bounded relation containment, not a uniform
optimality result. Compared with qualified completion, that sample removes four
pairs and adds thirteen, so neither relation contains the other. For K=1 the
child body is empty, and prefix+child executes 163 commands versus the previous
prefix's 136; there is no command-count improvement there. The retained exact
child trades greater static control for fewer executed commands at larger K.
Parent-owned tails and entry wrappers retain their own conservative costs.

Outer counts greater than the pinned caller count are explicitly labeled
structural reset stress tests; their original-kernel applicability is not
inferred. No benchmark-only bound is supplied to the analysis. The wrapper
argument applies under the named provider, control, access and target contracts;
it does not make the old unseparated width-eight stress graph fit six IDs by
relabeling it. The new parent adds explicit separation paths.

## Compactness and construction cost

One symbolic compiled recipe is reused for K=0,1,2,4,8,17,32,64 and repeated
visits. It keeps **171 families, 282 endpoint sites and a 51-node parent CFG**;
no K-specialized easier analysis is substituted. At one visit, executed commands
are 73/121/217/409 for K=2/4/8/16. Two and four visits at K=2 execute
145 and 289 commands. K=32/64 trace-only structural stresses execute 793/1,561
commands, but exceed this prepared A view's fixed 4,352-element width and are
not valid-layout kernel executions. Additional outer visits can likewise exceed
the fixed output width. They establish representation/trace behavior only.
Access-valid in-range growth uses K<=17; the independent two-bank fixture also
exercises repeated visits without changing its allocation/layout. Semantic
checks and caller applicability are separately qualified as above. The existing compact-K/replay scaling
tests also pass. This is not a separate constant-specialization compilation
scaling campaign, nor a partition-independent representation-size claim.

Three serial interleaved pairs with the same current binary and original input
have median subprocess times **164.43 ms** (previous prefix) and **264.46 ms**
(prefix+child). The change buys ordering quality, not compiler speed.
Representative stage medians in milliseconds are:

| Stage | Previous prefix | Prefix + child |
|---|---:|---:|
| Shared access recovery | 1.466 | 1.488 |
| Phase index | 0.118 | 0.116 |
| Existing raw dataflow solve | 0.063 | 0.064 |
| Child mathematical request | — | 19.055 |
| Provider qualification, including original certificate | — | 49.001 |
| Child interface, inclusive | — | 21.543 |
| Child map/assignment verification, subset | — | 21.463 |
| Child insertion/allocation, subset of emission | — | 8.566 |
| Parent role/map preparation | 0.125 | 0.041 |
| Parent protocol check | 0.005 | 0.004 |
| Emission/IR verification, inclusive | 0.525 | 9.848 |

Inclusive stages/subsets are not added. Source generation is a separate
compatibility-backend subprocess, pinned in the receipt. No device timing or
speedup is claimed. Allocation algorithms and the SCC/source-pipe transfer
kernels are unchanged.

## Qualification and remaining limits

A fresh **36/36** focused gate passes. Two supplementary frame/grouped and
scalar-contract tests pass **2/2**. One 121-column test line is wrapped with an
identical Python AST, and probe/emission inputs receive distinct filenames so
both commands retain reproducible inputs. The complete focused gate is rerun
after these harness refinements. The composition case also passes its separate
final recheck.
The three original whole-kernel default paths retain **12/12 byte-identical**
logical/support/physical/C++ artifacts against the saved 5af91f654 tool.
The new small and Qwen outputs generate C++ successfully. Expert protocols,
compact-K/replay, zero-inner-body behavior and hardware exemptions are preserved.

Native serial builds succeed. One covered-switch-default warning is inherited
from unchanged `SessionProducers.cpp`; changed implementations introduce none.
Ruff F checks and `git diff --check` pass. The scoped prefilter's 16 brace matches
are individually reviewed false positives with braced bodies; its raw nonzero
exit and review ledger are retained. No suppression, external analyzer,
sanitizer, full repository campaign or device qualification is claimed.

The environment probe initially exposed that the workspace venv is not an
editable package install by itself. The checked Linux entry point selects
Python on PATH and imports from the configured CMake package root; it does not
source the stale macOS `env.sh`. Timestamp-preserving source copies initially
left a changed test object stale; fresh-timestamp copies, dependency rebuilds,
byte comparison and a no-work build establish the final snapshot. Early harness
failures are superseded, not semantic counterexamples. Stage-specific C++ log
names prevent an early positive prototype profile naming collision; final
claims use the complete gate and separated `pair*-child` profiles.

Remaining scope: one proper eventful counted-loop child, existing sufficient
map equivalence, ready structured control and known operation contracts.
Unqualified whole-loop matching, foreign frames, unsupported event directions,
missing resource evidence or unproved map equality retain refusal. Exact GM
selectors, automatic fallback selection, multiple independently imported child
owners and a general token-to-SET/WAIT construction remain separate work.

## Paper consequences

| Established result | Remaining premise or limitation | Appropriate manuscript claim |
|---|---|---|
| Existing compact child owner and original endpoints survive composition | Qualified whole-loop provider; original modeled effects and frames | Section 5.5 interfaces can retain useful exact child work inside a conservative parent |
| Per-pipe pending effects include unknown GM coverage | Existing sound class enclosure and alias/operation contracts | Missing exact selectors need not prevent conservative synchronization; no fabricated exact GM export |
| Existing raw no-KILL source-pipe analysis covers crossings | Parent still visits supplied original static CFG; no arbitrary refresh pruning | Section 6 conservative reaching facts remain sufficient under this input contract |
| Entry acknowledgments and disjoint palettes prove rearming | Closed matching and within-visit assignment are separate producer premises; target command semantics required | Section 8 has an executable composition for this qualified family |
| Six-ID Qwen composition and bounded causal checks pass | Sufficient assignment; stress applicability and native behavior remain separately qualified | Physical feasibility for the selected composite protocol, without a minimum-ID claim |
| Bounded same-policy order containment improves | Static code grows; completion comparison is mixed | Report measured ordering/code-size tradeoffs, not global maximal concurrency or device speedup |
| Child internal event order is preserved in bounded checks and by confinement argument | Existing child's realized order can already strengthen minimum demands | No promotion of covers-selected-order into global realization equality |

## Pinned reproduction

Working checkout: `codex/frontier-shared-input`, starting at 5af91f654, with the
companion receipt pinning every changed source and the complete dirty diff.
The Linux source is `.local/section8-refactor/source-pinned/candidate-source`,
its build is the neighboring `candidate-build`, and Python is the workspace
`.venv/bin/python` (3.12.13). LLVM 19 uses
`/home/toni/work/llvm19/llvm-project/build-shared`. The C++ compatibility package
and native pass drivers are pinned independently.

```bash
source .local/prefix-child-composition-20261022/linux-env.sh
cmake --build "$PTO_BUILD_DIR" --target pto-test-opt pto-sync-input-test --parallel 1
"$PYTHON_BIN" "$LLVM_BUILD_DIR/bin/llvm-lit" -j1 --filter 'sync_prefix_children\.pto$' "$PTO_BUILD_DIR/test/lit"
"$PYTHON_BIN" .local/prefix-child-composition-20261022/run_final_checks.py
"$PYTHON_BIN" .local/prefix-child-composition-20261022/run_qwen.py
"$PYTHON_BIN" .local/prefix-child-composition-20261022/check_summary_coverage.py
```

The full focused command is `gate-command.json`; `run_outputs.py` reproduces
the original three-kernel comparison. Compiler watchdogs remain 35 seconds.
Builds/tests run serially per job; the only concurrent intensive pair was one
serial gate plus one serial compatibility-output job, within the aggregate
limit of two. Substantial artifacts remain on disk under the task's `.local`
directory. No commit or push is made by this task.

[Machine-readable receipt](frontier-prefix-child-composition-20261022.json).
