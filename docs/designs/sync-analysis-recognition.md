# Analysis-route recognition

## Active implementation

The current executable route is a whole-function numerical template followed
by periodic demand reduction, logical insertion and allocation without repair.
The implementation is in `lib/PTO/Transforms/FrontierSynch/`:

| Stage | Entry point and implementation |
|---|---|
| Shared physical access extraction | `SyncInput::build`, in `InsertSync/`; consumed by both synchronization passes |
| Structure and route checks | `FrontierAnalysis::initialize`, `ProgramRecognition.cpp`, `PhaseIndex.cpp` |
| Numerical effect template | `NumericTemplate.cpp`, `NumericTemplateControl.cpp`, `NumericTemplateEffects.cpp`, `NumericTemplateStorage.cpp` |
| Exact generators and reduction | `NumericTemplateAnalysis.cpp`, `LifetimeScan.cpp`, `PeriodicDemandGraph.cpp`, `PeriodicFrontier.cpp` |
| Endpoint preparation and insertion | `NumericTemplateEndpoints.cpp`, `NumericTemplateInsertion.cpp`, `LogicalInsertion.cpp` |
| Reuse certificate and physical IDs | `PeriodicAllocation.cpp`, `AllocationCertificate.cpp`, `PhysicalAllocation.cpp` |
| Compact endpoint emission | `AllocationCompaction.cpp` |

Arithmetic recognition builds its relation bundle only when a caller requests
`FrontierAnalysis::recognizeArithmetic()`. Same-policy requests reuse the result;
changing the alias policy resets it. MLIR invalidates the analysis after an
IR-changing pass. Direct callers must keep the borrowed IR unchanged for the
analysis lifetime. The full `--recognize` diagnostic requests it; the
numerical insertion path does not. This avoids constructing an unused
arithmetic candidate without changing the periodic result or adding a compiler
mode. Arithmetic and guarded recognizers remain tested library functionality;
they do not yet provide an alternative production insertion backend.

## Recognizer interfaces

The first recognizers inspect original MLIR using `SyncInput`, `PhaseIndex`,
and `SyncStorageEffects`. They neither unfold loops nor construct demands.
The library API is `FrontierSynch/Recognition.h`; its diagnostic client is:

```sh
pto-sync-input-test --recognize input.pto
```

The client reports the function's explicit and finite-guarded candidacy and
inspects both rotating variants of every `scf.for` separately, including loops
inside rejected outer loops. Arithmetic recognition checks the whole function.
Diagnostics identify missing premises. It verifies that the original IR remains
unchanged. This is a developer test tool, not a production synchronization mode.

## Separate recognition from a ready analysis

| Recognition result | Meaning |
|---|---|
| `applicable` | The implemented input checks establish this recognizer's contract, relative to complete supplied effects and valid input execution. |
| `missing-premise` | The shape may qualify, but an effect, geometry, or arithmetic obligation remains unresolved. |
| `not-applicable` | The supplied syntax is outside the implemented class, or contradicts one of its structural premises. |

An applicable input is not Section 8's `Ready` result. The base recognizers
report `backend=unavailable` when no analysis backend is attached. The late
numeric-template path additionally exports minimum demand records,
completion-origin queries and logical endpoint recipes, described below; it
supports the allocation-only physical path for complete whole-function plans,
but does not provide general regional composition or scarcity repair. Failed
recognition neither proves physical infeasibility nor changes the selected
ordering. No compiler fallback is invoked by the diagnostic.

As with the precision layer, completeness of the shared effect producer is an
input contract. Unrepresented side-effecting operations are reported, but this
does not audit every registered operation for missing implicit effects.

## Explicit regions

The explicit recognizer accepts a single-block sequence without nested regions
or ambiguous multi-phase anchors, with assigned pipes and exact supplied cell
effects. It uses the original `PhaseIndex` sequence and never makes a compact
loop explicit. Upper-bound or unknown effects produce `inexact-footprint`.
The result describes one invocation of the block, not repetitions of an
enclosing loop.

## Rotating storage

The first loop recognizer accepts canonical `scf.for` loops with lower bound
zero, step one, no loop-carried SSA arguments, and a fixed body without nested
regions or ambiguous phase order. The upper bound may be a parameter, including
an enclosing invocation's parameter; no enumeration of its values is needed.
Within-loop conditionals require a different guarded route and are not silently
flattened. An enclosing loop containing an inner loop is rejected by this
recognizer even if the inner loop qualifies independently.

An access identifies a fixed physical allocation or a `multi_tile_get` family.
Constant slots and `iv rem slot_count` (signed or unsigned, with nonnegative
canonical induction) are recognized. Simple constant-stride/offset numerators
are normalized to `(stride * iv + offset) mod slot_count`, but currently carry
an `index-arithmetic` obligation: algebraic normalization alone does not prove
equivalence under machine overflow and signed remainder. Runtime moduli,
loop-carried selectors and unsupported expression forms are not accepted.

The recognizer records each fragment's family, normalized stride and offset,
slot count, and candidate refresh distance `slots / gcd(stride, slots)`.
All accesses to one family must have a common stride. Known physical slot
extents must be disjoint, both within and across families. Distinct SSA roots
with overlapping physical addresses are rejected by this initial family
recognizer, not treated as separate storage. Refining such roots into a common
family remains a possible extension.

Within-slot fragments must come from an exact shared access region. The earlier
recognizer-local `tgetval`/`tsetval` recovery shortcut has been removed. Access
regions and their precision now come from the shared access producer; the
recognizer does not maintain a second instruction-effect table. An unresolved
footprint reports `inexact-footprint`, while structural checks still report
slot expressions, families and refresh distances.

## Guarded regions

The finite-guarded recognizer traverses loop-free, single-block regions with
nested `scf.if` arms. It retains a shared predicate DAG: each node records its
parent conjunction, condition value and selected arm. Payload phases and
rotating accesses refer to those nodes. Missing arms are empty; arbitrary loops
and other region control are rejected. Supplied footprints must be exact.

Conditions may be computed within a finite region. Recognition records whether
all conditions are available at entry; a late condition does not invalidate the
finite guarded analysis class, but recognition does not establish that paired
synchronization endpoints can evaluate their guards. Predicate evaluation must
respect the parent path, particularly for values defined inside an arm.

The guarded rotating variant permits nested conditionals in the canonical loop
body only when their conditions are available before the loop. This sufficient
check establishes immutable participation across iterations. It does not try
to prove invariance of expressions defined inside the loop. The same physical
slot and footprint checks as the unguarded route apply. No retention circuits
or guarded endpoint code are generated yet.

## Arithmetic relations

`ArithmeticRecognition.h` checks a supplied bundle of exact primitive relations.
Every role must be present: context, occurrences, order, native order, reads,
writes and extra prerequisites. A present empty union denotes false, whereas a
missing role is an unmet input obligation. Completeness and equivalence to the
program, finite executions and native-order semantics remain producer contracts.
The checker does not infer these facts from arithmetic syntax.

Each primitive declares named coordinates and their roles, including parameters,
storage coordinates and auxiliaries. All coordinates count toward the configured
dimension limit. Every piece is an MLIR `IntegerSet` over quotient coordinates
with explicit residues under one common fixed period. The producer must already
have split modular constraints into these pieces. Raw division, remainder and
nonlinear products are rejected.

The caller supplies fixed limits on pipe count, dimension, period and coefficient
magnitude. These limits must not be inferred from the current kernel. All rows
are collected with checked integer arithmetic and normalized by their coefficient
GCD. Inequalities use floor division, including negative constants; an equality
with a nondivisible constant makes its piece empty. The normalized rows identify
difference constraints, integer octagons, or the broader bounded-coefficient
class. For example, `x+y<=N` has three coordinates and belongs to the broader
class when `N` is a parameter; `u=7v+3w+c` needs coefficient bound at least seven.

An accepted result retains normalized rows and indices into the supplied schema
and residue pieces. Failed recognition publishes no partial normalized bundle.
This is a representation check, not feasibility, projection, reduction or selector
synthesis. Integer coefficients are currently limited to signed 64-bit values;
overflow is reported rather than wrapped.

### Primitive extraction from kernel IR

`ArithmeticProgram.h` now connects the shared input to the arithmetic checker.
It accepts sequences and nested canonical `scf.for` loops (lower bound zero,
step one, no loop-carried SSA arguments). Upper bounds may be constants, function
entry index arguments, or enclosing induction variables. This includes rectangular
and triangular nests and restarted inner-loop buffer rotation. It analyzes the
whole nest without unfolding any trip count or repeatedly composing an inner
summary.

The producer exports a site table and relations for occurrence domains, strict
reference order, strict native reachability, physical byte reads/writes, parameter
context and an empty extra-prerequisite relation. Relation metadata identifies
source/target sites, their coordinate spans, start/completion event kinds and
address space. Order compares common enclosing iterations first and static
program positions afterward. Native relations include same-pipe start/start,
completion/completion and start/completion paths, including each payload's own
start-to-completion edge. They represent native closure, not just adjacent edges.

Fixed exact effects use the shared physical ranges, so overlapping SSA roots
refer to the same bytes. Dynamic slots require `iv rem slot_count`, actual
physical slot intervals, and exact supplied within-slot footprints. Slots need
not be evenly spaced, and distinct allocations need not be disjoint for this
arithmetic route. The slot count must divide the configured residue period.
All quotient substitutions use checked integer arithmetic.

This first producer supports **P in {1,2} and D<=8**, with at most 256 residue
tuples per conjunction. Larger producer configurations are rejected before
splitting. Pipe/dimension limits are checked before constructing site pairs.
The test client fixes `(k,D,P,C)=(8,8,2,8)`; the library takes an explicit
configuration. The broader supplied-bundle checker remains available separately.

Conditionals, noncanonical loops, computed bounds, unresolved footprints,
existing synchronization and additional SSA prerequisites are currently rejected.
In particular, a payload result consumed by another payload, loop control or a
return is not silently treated as free metadata. Rejected inputs export no
partial primitive/site/parameter bundle. Completeness of the shared operation
effect registry remains an input premise.

`pto-sync-input-test --arithmetic input.pto` emits structured test data for these
relations. `pto-arithmetic-recognition-test` exercises the generic checker and
its integer normalization. Neither tool generates demands or synchronization.

## Cost and scope

For a candidate with `N` immediate operations, `A` supplied access records, and
`S` physical slot intervals, recognition takes expected
`O(N + A + S log(S+1))` time and `O(A + S)` additional space. Family lookups
use hash maps; a sorted interval sweep checks disjointness. Expression matching
checks only the documented bounded-depth syntax. These costs exclude building
the shared input and precision partition, and charge the slot intervals
explicitly: this first implementation enumerates known physical slots. It does
not claim the paper's slot-count-independent extraction bound. The output has
at most one normalized fragment per supplied record; deduplication and
generator extraction are subsequent work.

Guarded traversal adds expected linear work in visited operations, phases and
access records, plus the rotating geometry checks when requested. For arithmetic,
let `L` count expression DAG nodes across rows, `D` the configured coordinate
limit, and `M` the input schema/residue metadata. Checking takes expected
`O(M + LD)` fixed-width arithmetic operations, plus coefficient GCD operations.
Expression collection shares nodes within each row. These bounds exclude
constructing the primitive relations and residue splitting.

For the IR producer, let `n` be the number of operations, `h` the nesting depth,
`s` the number of payload sites, and `a` the supplied access/range count. At fixed
supported `D,P`, collection and construction take expected `O(nh+s^2+a)` work,
with `O(s^2+a)` relation output; residue splitting has the explicit factor
`P^D` and polynomial factors in `D`. The quadratic term constructs reference and
native order between static sites, not conflicting dynamic access pairs. Costs
are independent of trip counts. These operation counts exclude shared input
recovery and the later symbolic reduction backend.

Counted-pattern and compositional recognizers remain unimplemented. Practical
coverage next needs producer-supplied coverage or exact subregions through the
shared interface, broader bounds/views and arithmetic conditions. The arithmetic reduction/selector backend and rotating
generator/quotient backend are also still separate work.

## Validation

The lit tests cover missing access-region premises, parameterized
two- and three-slot loops, zero and large trip counts, independent assessment
of nested loops, unknown geometry, shifted-index obligations, mismatched
strides, physical aliasing across SSA roots, runtime moduli, conditional bodies
and unsupported loop steps. Existing shared-input and precision tests remain
applicable.

New tests also cover nested then/else paths, absent arms, immutable versus
iteration-varying loop guards, late finite guards, all three arithmetic classes,
negative-bound normalization, impossible equalities, residue/schema mismatches,
configured bounds and checked coefficient overflow.

The arithmetic program fixture now verifies that missing access contracts and
rejected control/prerequisite cases export no partial bundle. Its former
occurrence-level comparison relied on the removed scalar footprint shortcut;
it is no longer evidence for exact IR extraction. The supplied-relation checker
continues to test the arithmetic classes and immediate rejection of oversized
producer configurations.

## Arithmetic physical identity

Arithmetic access relations use `(memory space, canonical base, byte)` as the
physical identity. For planned local storage the base tag is absent and the
byte is an absolute address. Distinct allocation SSA values therefore still
conflict when their assigned addresses overlap.

For GM accesses the shared view/pointer mapper retains a canonical function-entry
pointer and an exact relative byte map. The arithmetic producer accepts this
base only when it agrees with the shared memory record's root. Views, pointer
casts and `addptr` retain that identity and contribute their byte offsets through
the existing shared maps. A base tag is not an extra integer variable or a
numerical pointer address. Consumers must compare both identity tags before
comparing byte coordinates. The IR owns the borrowed base values, as it owns the
payload anchors.

One canonical GM base is supported under either existing alias policy. Distinct
GM bases are disjoint only under the supplied `MayNotAlias` policy; under
`MayAlias` the producer rejects multiple bases because their relative displacement
is unknown. Opaque selected/carried pointers and mixtures of based and absolute
GM addresses also reject without exporting partial primitives. This changes no
instruction footprint or shared alias policy and introduces no new local
allocation identity. The `--arithmetic` test dump exposes the base's argument
number (`-1` denotes an absolute address); it never prints an address value.

## Fixed local byte-map materialization

The shared range materializer recognizes separable sums of constant multiples
of a coordinate, its quotient and its remainder by one positive constant divisor
per coordinate. Constant extents split exactly into full digit rectangles and a
partial final rectangle. Within each rectangle, sorted strides merge byte
intervals only when they touch or overlap; negative strides are reflected, zero
strides add no bytes, and gaps remain explicit. The result is an exact union of
physical ranges, independent of instruction names and layout names.

Cost is charged to the parsed expression, generated rectangles, dimensions per
rectangle and emitted sparse slices, rather than to every byte or element of a
contiguous tile. Rectangles and aggregate sparse slices are each capped at
4,096; parsing also has explicit depth and work limits. Arithmetic is checked,
and failure preserves the caller's output. Unsupported forms use the previous
exact materializer when possible, otherwise retain their symbolic access map.
This does not guarantee concrete ranges for every layout or resolve symbolic
origins whose simplification requires loop-domain facts.

## Late numeric inner templates

The `numeric-template` candidate keeps one outer counted loop and explicitly
expands certified numeric inner visits into a fixed body. This is the draft's
charged late expansion, not a nested-region composition theorem. It is reported
after the existing candidates and does not select a backend or set
`analysis_ready`. Each expanded payload retains its original phase, before/after
cuts and full tuple of original inner induction values. The outer induction is
`lower + step * ordinal`; bounds remain in the original IR. A statically proven
zero outer trip count produces an empty-invocation certificate without expanding the body or
normalizing unreachable accesses. The certificate is tied to those constant
bounds. Runtime-zero loops retain their body schema and use endpoint guards.

Bounds, carried scalar recurrences and branch decisions must specialize exactly
through the shared scalar analysis. Unknown state, remaining outer-dependent
branches and additional payload prerequisites reject. Shared effects are
substituted for each template occurrence. Local footprints must become exact
constant byte unions; their common atoms use memory space and physical address,
including reuse across distinct allocation SSA values. The resulting local
storage certificate has period one and writer refresh at most one outer visit.

GM effects remain in the records. A complete GM base is discharged only if it
is read-only, or if exactly one **template payload** accesses it and all of that
payload's read/write fragments share a constant translation per outer ordinal.
For the latter case, the translation magnitude must cover the entire union's
span, proving different visits disjoint. A separate reader or another expanded
visit of the same original instruction invalidates this discharge. Distinct GM
bases require the existing `MayNotAlias` policy. Because this first candidate
exports no external GM storage selectors, it is accepted only when the outer
loop is directly in the function and contains all its payloads; no surrounding
payload interactions are discarded.

Preflight counts expanded operation visits and potential payloads before
allocating template occurrences. Fixed representation limits are 65,536 visits,
4,096 payloads, 65,536 footprint/range/atom/reference fragments, and nesting depth
16; callers may only narrow them. Both branch arms contribute to the preflight
bound. Actual work is charged to expanded visits, shared scalar/map processing,
materialized ranges and atom references. Atom endpoints are sorted once, and
range-to-atom lookup uses binary search plus the emitted references. Exceeding a
limit rejects with no partial template export. Recognition supplies the effect
word to the exact numerical analysis below. It does not supply external storage
selectors, emitted endpoint code, event-ID allocation or synchronization commands.


## Exact numerical periodic demand analysis

An accepted numeric template now feeds a generic storage scan and numerical
periodic reducer. The scan accepts explicit payload identities, pipes and
physical atom read/write modes. It consolidates modes before changing state,
tracks the previous writer and the latest read-only reader on each pipe, and
emits RAW, WAR and consecutive-writer WAW generators. Native completion order
protects the earlier readers on that pipe. Deduplicated endpoint pairs retain
their storage witnesses. Supplied forward prerequisites are also supported by
the scan. Its expected cost is `O(n + c + e)` for payloads, atom incidences and
supplied prerequisites; resetting a lifetime discards its reader-map capacity.

For the constant local effect word, every written atom is refreshed in the next
visit. Scanning two visits therefore finds every generator sourced in visit
zero. Keeping those records derives the period-one generator certificate.
The template's existing whole-base GM discharge excludes internal GM conflicts;
all original payloads, including those with only discharged GM effects, remain
in native pipe order. Type numbers continue to identify the original template
payload, its phase, fixed inner coordinates and insertion cuts.

The reducer itself takes arbitrary forward numeric displacement records, not
only distance-zero/one records from this adapter. It deduplicates them, builds
start/completion vertices with native chains and wrap edges (including one-site
pipes), and computes one completion-seeded shortest-path index per pipe. A
best/second incoming-edge scan decides each record's redundancy against the
original graph, preserving ties between distinct edge identities. This avoids
rerunning shortest paths after deleting individual records and never expands
the dependence distance into an iteration window.

Canonical record sorting costs `O(g log g)`. After deduplication, graph and
arithmetic work is `O(k(V + E) log(V + 1) + kE + g)`, with
`O(V + E + km)` space for `m` payload types, `V = 2m` event types and
`E = O(m + g)` edges. The ordered priority set holds at most one entry per
vertex. Integer arithmetic is checked; representational overflow rejects the
complete result without exporting a partial proof. These are arithmetic-operation
bounds, not unit-cost claims for arbitrarily wide integers.

The result contains the exact minimum-demand records and completion-origin
reachability thresholds, including completion-to-completion queries. Unreachable
thresholds are distinct from invalid queries. Finite-prefix helpers check endpoint
presence, support partial final periods, and distinguish reflexive from strict
completion queries. Threshold lookup uses indexed arrays; a profile query adds
an expected constant-time pipe lookup. These results do not provide start-origin
queries, external storage selectors, emitted endpoint code, physical IDs or
repairs; `analysis_ready` remains false until the required interfaces exist.

The numerical test helper accepts synthetic effect words and arbitrary periodic
records. Its independent Python oracle enumerates all conflicting occurrence
pairs, adds native order, and computes finite DAG reachability and covers without
using the lifetime scan. It checks exact demands, per-pipe ranks, finite-prefix
queries and thresholds, alongside very large distances, overflow rejection,
partial periods, initial readers, read-modify-write and tied alternatives.


## Direct logical endpoint recipes

Each retained periodic record now produces a logical direct-insertion recipe.
For a cross-pipe record `a -> b` at displacement `d`, SET follows the original
source payload and executes at source ordinal `t` exactly when
`d < T && t < T - d`. WAIT precedes the original target at ordinal `j` exactly
when `j < T && j >= d`. Both identify the same logical notification by
`(record, source ordinal)`, with source ordinal `j - d` at WAIT. This namespace
belongs to one plan and must be preserved if plans are later composed. These
guards publish no notifications for absent targets, so this route needs no
cleanup for surplus publications. Same-pipe records request a barrier before
the consumer under the same-pipe adjacency premise of the model.

The numeric adapter retains each endpoint's own original phase and complete
inner-coordinate tuple. It also records the actual IR insertion points:
`Before(op) = (block, op)` and `After(op) = (block, op.next)`. Recipes sharing
that static key form a cut group. At each dynamic visit, coordinate predicates
and ordinal guards select the active instances before grouping commands;
different visits of one static cut are never coalesced. Active commands at one
cut are ordered SET, barrier, WAIT, with one barrier per pipe. Metadata or region
boundaries do not make distinct cuts equivalent automatically.

Counted-loop helpers compute trip counts and original-IV ordinals using checked
integer relations, including nonunit steps, zero trips and the full signed
bound span. Invalid steps or off-grid induction values are errors; an endpoint
whose occurrence or partner is absent is inactive. Recipe construction takes
linear work in retained records and anchor-coordinate output, with expected
linear static-cut grouping. Per-recipe ordinal selection is constant arithmetic
work; matching an anchor also reads its coordinate tuple. Ordering `q` active
commands at one cut costs `O(q log q)`. No production helper enumerates the
outer trip count.

The read-only recognition API exports selection recipes and placement
descriptions. The integrated pass below now emits their logical PTO operations;
physical event IDs are assigned by the separate pass described below. No
start-origin query interface, external storage selectors or scarcity repair is added, and the
whole Section 8 result is still not ready. The periodic index remains unchanged.
The command oracle checks fresh logical matching and models the mechanisms
separately: SET observes prior completions without gating later payload starts;
WAIT gates later starts without waiting for prior payload completions. Under
same-pipe adjacency, the resulting payload-event closure equals the selected
required order. For a nonadjacent local demand, the test checks coverage only.


## Periodic directed budgets and cyclic offsets

The numerical periodic result also supplies exact resource budgets for its
canonical direct handoffs, under the draft's command and local-adjacency
premises. The allocation summary groups retained cross-pipe records by direction
and orders each group by source type. It verifies unique source and target
phases and strict target ordering, including the wrap to the next period.
These checks establish that truncating the payload execution retains a prefix
of that direction's communication sequence. Same-pipe records consume no event
IDs and are not included in these groups.

For each phase, completion-to-start thresholds give the first later handoff
whose publication is ordered after its consumption. The maximum phase distance
is the exact uniform budget; absence of every such return path is represented
as infinity. An exact finite-prefix query counts present handoffs and clips
the phase lifetimes, handling zero trips and partial final periods without
unfolding. Infinity is separate from an arithmetic failure. Entire candidate
calculations use widened integers before minimization and subtraction; only an
unrepresentable final finite result fails. No partial budget summary is
published on failure.

The builder makes `O(sum c_direction^2) <= O(m^2)` threshold queries.
Hash grouping is expected linear; per-direction sorting is bounded by the
same quadratic sum. Stored summaries are linear in the retained cross-pipe
records. A finite-prefix budget takes linear work in that direction's phases;
a uniform capacity comparison is constant work. Integer bit costs remain
separate from these arithmetic-operation counts.

For capacity `E > 0`, a handoff of phase `r` in source period `n` has cyclic
local offset `(c*n+r) mod E`, evaluated with widened arithmetic. Capacity
sufficiency is a separate query: evaluating an offset does not establish that
reuse is safe. A zero-capacity direction is sufficient only for an empty finite
handoff sequence; no modulo operation accepts zero capacity. The logical
consumer reconstructs the source period using the existing endpoint identity.

These are per-direction budget and offset recipes, not hardware event IDs.
They assume no device capacity, perform no new hardware qualification, and do
not prove that two directions mapped to overlapping physical resources can
reuse those resources concurrently. A physical resource assignment still needs
its own compatibility and capacity checks. No PTO commands, physical mapping,
scarcity repair or new ordering is emitted here. The summary is relative to the
canonical direct plan on the supplied covers, with no exceptional entry or
interface paths that supply additional reuse.

The independent finite oracle enumerates handoffs from all-conflict DAG covers,
finds their first causally reusable successors, measures interval overlap, and
checks every same-offset pair. Random quotient tests certify these abstract
reuse-order formulas; physical canonical-plan claims additionally retain the
model's local-adjacency premise. Tests also cover infinite uniform budgets with
finite invocation budgets, large prefixes, widened offsets, malformed endpoint
ordering, and representable minima with larger intermediate arithmetic.


## Integrated logical insertion

`pto-frontier-analysis` now computes the numerical periodic demands and inserts
their synchronization in one pass. Its supported input is one accepted
whole-function numerical template, including the tested nested PyPTO and
TileLang GEMMs. The diagnostic `--recognize` continues to inspect inputs without
mutation; `pto-sync-input-test --insert-logical input.pto` runs the production
pass and prints its transformed IR. Other recognized classes still need their
analysis and endpoint backends before they can use this pass.

`prepareNumericTemplateInsertion` is the route adapter: it validates the
numerical result and constructs a `PreparedLogicalPlan`. This common plan
contains legal cuts, source/destination pipes, presence guards, logical record
and source-occurrence identities, and an explicit plan namespace. Guard and
identity arithmetic is built in owned, detached MLIR blocks. Other demand
backends can supply this same interface without numerical-template recipes.

`insertLogicalSynchronization` validates SSA availability, types, cuts and
namespace freshness before changing the function. It then installs the prepared
arithmetic and inserts commands, grouping endpoints by their actual cuts. The
producer must still establish demand correctness, dynamic endpoint matching,
safe evaluation of its arithmetic and legal placement; structural validation
cannot prove those analysis obligations. All endpoints sharing a cut must be
submitted together, and different plans must use distinct namespaces.

The numerical adapter preserves
original payloads, loops, branch decisions and allocation geometry. A command's
guard selects its own inner-coordinate tuple and checks that its partner's
outer iteration exists. Commands at a common original cut are emitted in
SET, barrier, WAIT order. Local barriers on the same pipe at that cut share
the disjunction of their guards. Zero trips execute no handoffs;
a last iteration never publishes a handoff whose consumer is absent. A whole-function
plan adds one terminal `PIPE_ALL` before return to complete outstanding payloads,
including the final store. It is outside all computation loops. A statically
empty invocation needs no drain. This is invocation completion, separate from
the internal minimum-demand relation. The
[Ascend static-Tensor contract](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html)
requires a terminal `PipeBarrier<PIPE_ALL>()`; existing InsertSync emits the same
completion mechanism.

Cross-pipe commands are `pto.logical_set` and `pto.logical_wait`. Both carry
source and destination pipes, a plan/record identity and the source outer
ordinal. The identity is logical and scoped to the function invocation; it
is not a physical hardware event ID. The IR operations survive round trips
and have side effects. Physical lowering must replace them after allocation;
EmitC and VPTO conversion reject unresolved logical commands. Same-pipe
requirements use `pto.barrier` directly. No internal all-pipe barrier is introduced.

Insertion does not assign physical IDs. When the producer has a valid periodic
capacity summary, it preserves the budgets and phase numbering in an owned
`pto.cyclic_allocation` function attribute for the second pass. Insertion
invalidates the borrowed analysis result after mutation. A rejected
input receives a diagnostic instead of a partial synchronization plan.
The independent insertion checker executes the generated integer predicates
and original structured control, observes actual command occurrences, and
compares their payload order with an all-conflict graph.

## Allocation-only physical lowering

Run the second pass immediately after logical insertion:

```text
pto-test-opt input.pto --pto-frontier-analysis \
  --pto-frontier-allocate='eligible-ids=0,1,2,3,4,5'
```

The caller supplies an eligible subset of 0–5. The pass rejects reserved IDs
6–7 even when explicitly supplied. There is no
default pool. `pto-frontier-allocate` consumes the numerical producer's uniform
cyclic certificate and allocates independently within each directed event domain.
For A2/A3, the event identity is `(source pipe, destination pipe, numeric ID)`:
equal numeric IDs in different directions identify different events. The
[2201 synchronization documentation](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)
states this explicitly. The
[TQueSync guidance](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/TQueSync.html)
recommends 0–5 for static-Tensor programming, reserving 6–7 for system/framework
uses. No numeric ID is permanently tied to a pipe pair: every supported pair
can use any eligible number. SET/WAIT matching uses the pair and number together.
The native enum's 0–7 range alone does not establish eligibility.
It lowers the logical endpoints at their existing cuts, preserving their
executed command sequence, payloads and control flow. A one-ID subset emits static flags; larger subsets
emit dynamic flags with offset `(c*n+r) mod E`. Reducing the factors modulo `E`
before multiplication prevents overflow for large source ordinals. Both ends
use the producer's source ordinal and therefore select the same ID.

Success removes every logical endpoint and the certificate. Failure emits a
diagnostic and leaves the function unchanged. Preflight validates the eligible
IDs, certificate shape, endpoint identities, complete static endpoint pairs,
index width and all directed capacities before any rewrite. Existing physical
event commands are rejected because their ID occupancy is not certified here.
The certificate is an internal producer contract, not verification of arbitrary
annotations: payload order, matching, guards and occurrence ordinals must not
change between the two passes. Other demand producers can use the same lowering
when they establish and encode this uniform cyclic contract.

The two current GEMMs have directed budgets 4, 4, 2, 2, 1, 1. Each fits the
six eligible numeric IDs; allocation uses only 0–3 across the six domains.
Summing these budgets to 14 and comparing against six numeric IDs was an
incorrect global-pool restriction. Repeating a number in different directions
does not require a cross-direction reuse witness: those are different events.
Same-direction reuse still requires the certified WAIT-before-SET order.
Finite-invocation budget minimization, shared-resource allocation for a different
target model, and scarcity repair remain separate work. Allocation failure does
not establish infeasibility under those other strategies. Statically empty
examples require no IDs.

With `N` IR operations, allocation preflight and literal emission cost
`O(N + records + endpoints)` for the six eligible numeric IDs. No dynamic loop
is unrolled. Compaction first performs whole-function common-subexpression
elimination; its cost is separate from the per-cut bound below.
After preflight, endpoint compaction combines enumerated coordinates at a common
cut. It retains residual guards and source ordinals, fits a modular ID formula,
and verifies that formula against every original coordinate point. Exact unions
of adjacent boxes describe the guard; missing points are never filled in.
A precedence graph preserves the order of commands whose guards can overlap.
A failed fit, overlapping copies of the same command, an intervening unrepresented
command, or cyclic precedence leaves that cut in literal form. Compaction changes
neither the demands nor handoff grouping. Consecutive physical ID lists use direct
arithmetic instead of a select chain.

For a cut with `r` records, at most `d >= 1` coordinate axes and capacity `E <= 6`, this
optional code-generation step uses `O(E*d*d*r*r + d*r*log(r))` arithmetic/comparison
operations and `O(r*r + d*r)` storage. It does not change demand-analysis complexity.
The finite test oracle compares compact and literal emission command for command,
including actual numeric IDs. Device speed is assessed separately; smaller code
and preserved ordering alone do not establish a latency improvement.
The test oracle executes both plans, checks unchanged cuts and payloads, matches
each physical notification to its original logical identity, and requires every
reuse to have a causal WAIT-before-SET path in the original logical command graph.
It also checks failure atomicity, malformed certificates and insufficient pools.

## Hardware-protected storage conflicts

`HardwareProtectionBuilder` records hardware protection on physical-cell
accesses before the lifetime scan generates demands. The numerical-template
adapter supplies the already resolved accumulator cells and original operations;
the same builder can serve other occurrence producers. Recognition, periodic
reduction and logical insertion contain no matrix-specific exemption.

The first rule implements the A2/A3 [Mmad accumulation contract](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html):
consecutive accumulation into the same accumulator does not require a local
barrier when `(m / 16) * (n / 16) >= 10`. Dimensions are the valid dimensions
used by the native lowering. The implementation requires known compatible
dimensions, an initializing matrix operation, subsequent in-place accumulations,
the same accumulator type and physical cells, and `PIPE_M`. Another initializer,
an incompatible matrix operation, or an intervening access to those accumulator
cells ends the group. Separate template visits get separate group identities.
Unknown dimensions and other targets receive no exemption. This source was
checked against the A2/A3-compatible documentation preview built from
`5c07153f668d` on 2026-09-30; it does not grant all M-pipe conflicts protection.

A group certifies access protection for every ordered pair of its writers on
each grouped cell. It does not assert completion-before-start. The scan retains
every physical read and write and updates the lifetime state normally, but skips
the protected RAW/WAW witnesses. Other cells' witnesses and supplied prerequisites
remain. Filtering after reduction would be wrong: a protected accumulator edge
could already have hidden an indispensable conflict on another cell. Group IDs
are local to a builder; independently constructed inputs must remap IDs before
combination. Repeating a scope must create fresh groups.

The sparse scan remains sufficient for the remaining software requirements.
Along the lifetime-scan witness path for a conflict on one cell, protected links
before a retained software edge can be replaced by native completion order;
protected links after it by native start order. If all links are protected, all
endpoints belong to the same protected
writer group. Protection annotation adds expected linear work in the access
records and does not change the scan or quotient asymptotic bounds.

Tests separately check target and shape eligibility, accumulator resets and
intervening readers. An independent all-pairs conflict oracle removes only
certified protected witnesses, then checks the sparse generators and actual
inserted commands against the resulting required order. Cross-pipe event pairs
are still selected by reduction. UnitFlag, physical ID allocation and command
grouping are separate concerns and are not enabled by this rule.
