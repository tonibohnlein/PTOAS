# Compositional OAHS: baseline, cut prototype, and demand constructor

## Status and migration boundary

This implements the general conservative baseline, **not the completed
replacement/overlap-recovery milestone**. The existing `structured` CLI mode
accepts `structured-precision=false` at the pass level, or
`--insert-sync-structured-precision=false` through the compiler CLI. That path
does not invoke `RegionLayout`, the periodic constructor, or Presburger.

During migration, precision enabled (the existing default) still uses the
previous implementation. This is deliberately temporary: it is not a claim
that the old constructor has already become a compositional precision provider.
Do not switch the default merely because baseline admission or compilation time
improves. The current readiness/release/overlap gate includes the original seven
cases and the frozen historical GEMM. Older seven-case results below retain
their original population and are not retroactively eight-case measurements.

The amended direction is an evolution of InsertSync's physical translation and
structural traversal, not a claim that forward traversal or prefix sharing is
new. Original physical obligations, acquired completion, and event protocols
are separate concerns. The conservative engine composes admitted operations,
sequences, `scf.if`, `scf.for` and `scf.while` without periodic precision. The
new cut-precision candidate uses that same engine and target contracts, but is
currently exposed **only in the native test driver** as `cuts:none`.
It has not replaced the old precision-enabled backend.

The next increment is available as `demands:none` in the native test driver
and explicitly as `--insert-sync-planner=composition` in the compiler.
For this opt-in planner, `--insert-sync-structured-precision=true` enables
demand placement and `false` uses the same engine's conservative construction.
Both settings use the same importer and target contracts, without the old
structured or symbolic constructor as a fallback. The existing planner default
and the migration behavior of `planner=structured` are unchanged. This exposure
permits real pass-pipeline corpus and whole-compilation measurements; it is not
a replacement-quality approval.
It constructs direct transfers from physical completion demands without calling
`discoverCuts()` or constructing per-cell cycles. It is still a **migration
candidate**, not completion of the four-milestone replacement plan. In
particular, cross-domain incoming placement, rotating-generation precision,
and replacement qualification remain open.

The positive operation/effect registry and selected A2/A3 hardware contract are
retained. Moving that registry to a lowering-owned operation interface remains
unfinished; unknown effects are not admitted by their absence from a denylist.

## State and transfers

The pure production core is `StructuredSyncComposition`. Its input is an
immutable postorder tree of operations, sequences, choices, for loops and while
loops. Node children are original control regions, not an occurrence expansion.

Each physical cell stores possible readers and writers **per observer lane**.
The complementary information represents acquired completion: clearing a source
bit for V does not claim that MTE2 or the scalar host has completed that source.
Subsequent source accesses introduce new outstanding bits for every observer.
Writes never kill readers or imply definite initialization.

The native adapter partitions known local intervals, up to 256 cells per
function. Unknown ranges, forwarded local handles, or excessive interval
populations widen to whole-space cells. GM root groups remain separate only
under a qualified distinct-argument contract or explicit `pto.noalias_pairs`.
The latter must be an even-length i64 list of distinct, valid GM-capable argument
indices; only its listed pairs are assumed disjoint. Unknown provenance merges
groups. No all-access-disjoint policy is introduced.

GM write history is separate from completion state and survives barriers and
events. A subsequent possibly aliasing GM read requires visibility. No current
baseline protocol supplies that property, so construction refuses it. This
includes FIX stores; legality of an event direction is not a publication rule.
ACC accesses conservatively retain full completion/resource exclusion; optional
MMAD accumulator-update evidence is not promoted into operand release.

The structural transfers are:

* Sequence forwards the complete state to each child.
* Choice analyzes both arms. A union of pending histories retains every possible
  hazard and only completion justified on both paths. No predicate solving or
  path enumeration occurs.
* A loop first seeds the incoming state with all possible body effects for every
  observer. This forgets completion precision across the entire body, including
  nested siblings. Consumers acquire missing completion inside the original
  control structure; there need not be a blanket fence at the loop header.
* For loops additionally join the zero-trip entry at exit.
* While loops analyze before and after separately. Their exit is the state after
  the before-region, including its effects on the condition-false execution.

The loop seed is an inductive overapproximation: if incoming history is E and
the body may-effects are S, transfer starts from E union S. Synchronization only
removes pending bits, and body operations only add effects already in S.
Therefore the full body/backedge result remains contained in E union S. One
summary pass and one transfer suffice; no fixed-point search or trip expansion
is needed. GM publication history obeys the same union invariant but is never
cleared by a completion primitive.

The native translator's opt-in conservative forwarding mode seeds SCF results
and region arguments before translating payload. Local tile types retain their
actual local address space; the pointer-only address-space helper must not be
used to default those handles to GM. Scalar prerequisite qualification follows
initializer/yield/condition sources and rejects asynchronous producers.

## Reusable protocols and reconstruction

Missing same-lane completion uses a qualified named barrier. Cross-lane
completion uses this fixed-orientation rendezvous, with a fixed pair of keys
throughout the physical lifetime:

```
SET  A -> B
WAIT A -> B
SET  B -> A
WAIT B -> A
```

The four commands are adjacent in original scalar control. The orientation is
canonical, not reversed when the direction of the next memory hazard reverses.
Before another forward publication A must consume the previous reply. Before
another reply B must consume the next forward publication, which itself follows
consumption of the previous reply. Thus both keys have consumption-before-rearm
causality. A lexical boundary does not reset event state.

A rendezvous also transports completion previously acquired by its endpoints.
SET firing observes the source prefix, but **does not gate later source
issue**. B's first WAIT acquires A's prefix; only A's final WAIT also acquires
B's prefix. B's own outstanding work therefore remains pending at B unless
another mechanism establishes its completion. The transfer and independent
oracle both model this asymmetry; a two-cell overwrite regression challenges it.
Routing uses a shortest path in the fixed-size graph of qualified bidirectional
event connections. It does not enable the excluded direct MTE2/MTE3--FIX event
directions. The current fallback uses the first available key in each directed
domain and respects the target pool/reservations.

Reconstruction parses actual packets and barriers, checking orientation, keys,
adjacency and placement. It reimports physical effects, rebuilds the region tree
and verifies all demands against actual mechanisms. It never compares against
the constructor's selected coverage list. Original payload/control/attributes
are preserved transactionally. One unmarked unconditional PIPE_ALL remains at
the physical lifetime exit and receives no ordinary requirement-coverage credit.

Constructor and checker share the abstract state algebra and target contracts.
Their finite test oracle is separately implemented: it expands small concrete
executions into distinct payload issue/completion vertices and checks physical
hazards and event rearm causality by graph search. That oracle is test-only and
does not establish device correctness.

## Reproduction and current evidence

Use the existing configured build; no clean build is needed:

```sh
cmake --build "$BUILD" --parallel 2 --target \
  pto-structured-sync-test pto-test-opt pto-composition-core-test
ctest --test-dir "$BUILD" --output-on-failure \
  -R '^oahs_composition(_core)?$' --parallel 1
```

`check_composition.py` accepts `--historical` and repeated `--corpus-manifest`
arguments. It verifies source hashes, uses the existing corpus preparation rule,
and records prepared hashes, commands, binary hashes, classifications and
first refusals. Timeouts and crashes are distinct from semantic refusals.
Adding `--python-root` with `--historical` also emits synchronized PTO and C++
through the full compiler, retaining output hashes and commands.
These are raw/prepared compatibility measurements, not device or
end-to-end production qualification. `benchmark_buffers.py --arms existing
structured composition` compares unchanged whole-compiler inputs; `composition`
is a benchmark label for structured precision disabled, not another planner mode.

`report_composition.py` compares the resulting corpus summary against supplied
baseline reports, requiring identical original and pre-sync hashes. It also
replays the seven benchmark scenarios through the existing completion-boundary
observer and reports later/earlier source-prefix acquisitions separately from
compilation time. This deliberately does not turn a compile-time pass into
overlap qualification.

The final local baseline campaign (uncommitted work based on `7fb70015`) is
recorded below. Exploratory results predating the SET-semantics correction are
superseded and must not be used to qualify this implementation:

* Both composition CTest gates pass: 12 native positive checks when the optional
  pinned GEMM is included, five expected refusals and ten mutation checks.
* 705,319 core/independent finite graph assertions pass with Clang ASan/UBSan.
  Leak checking is disabled because it is incompatible with the traced runtime.
* On exactly matching pre-sync hashes, the recorded S7.1 PTOAS corpus changes
  from 32/150 to 45/150; PyPTO/pypto-lib changes from 65/213 to 152/213. No
  previously admitted case is lost. These populations must stay separate from
  the older, differently prepared 168-input campaign.
* The pinned historical GEMM compiles through native construction and fresh
  reconstruction with its original A/B/C pairwise contract. No stronger alias
  promise or prefetch rewrite was required. The complete compiler also emits
  synchronized PTO and C++ for it. Generated C++ has not been device-compiled
  or executed.
* Three paired whole-compilation rounds plus one warmup pass the <=2x gate on
  all seven unchanged cases. Baseline median ratios range from 0.989 to 1.030.
  This is **not** an overlap acceptance result: ordinary buffering uses roughly
  twice as many SET/WAIT sites, and late prefix acquisition can wait for more
  source work than the optimized implementation.
* Concrete boundary comparisons observe later producer-prefix acquisitions in
  six of seven cases (not one-buffer). The exact counts are scenario-dependent,
  not a timing or device-speed metric.
* The pre-existing structured core/native CTest gates pass. `oahs_focused`
  fails before its reference checks because this environment lacks `libisl`;
  that dependency was neither installed nor bypassed.

The local detailed artifacts are under `oahs-composition-validation` alongside
the worktree. The corrected evidence is `final-corpus/summary.json`,
`final-native-frontend/summary.json`, `final-benchmark-seven/summary.json` and
`final-comparison.json`. Earlier `qualified-corpus`, `benchmark-seven` and `comparison.json`
outputs predate the SET-semantics correction and are superseded. Visibility
qualification also lowered the earlier exploratory admission count.

## Bounded cut precision and producer-prefix sharing

`constructCuts` derives first/last access cuts for each physical cell by
composing lane-access sequences over the original tree. Adjacent accesses on
one lane coalesce; compatible choice arms retain alternative cuts. A possibly
empty single-lane child can use its common entry/exit cuts. A qualified recurring
cell protocol publishes after its last source access and acquires before the
next lane's first access. Branches may skip a whole protocol visit arbitrarily;
the protocol advances on visits, not on an inferred iteration number.

This provider is bounded to 32 lane groups and eight alternative cuts. Mixed
child recurrence or an exceeded limit declines precision locally. It does not
decline the conservative compilation path, enumerate predicate combinations,
or invoke the symbolic planner. The current provider handles distinct physical
slots; general dynamic selector/generation precision remains unfinished.

All protocols use one function-wide directed-key population. A fitting plan
can use the full available pool. Its keys are reserved against ordinary
conservative packets. If these reservations prevent construction, one
transactional retry leaves a fallback key per direction. No partially emitted
candidate escapes. Children do not allocate independent pools.

For a recurring protocol, entry reconciles incoming cell history before seeding
the release token. The final release is consumed on exit, including zero trips.
An empty forward acknowledgment chain then carries that consumption back to
the seed publisher. Merely consuming the final token on another lane is **not**
enough to make the next invocation's publication reusable. Neither this exit
protocol nor a source-level region boundary resets event state.

Each live certified key also carries an optional producer-prefix receipt across
all physical cells. At SET, it records work not covered by that publication;
every later payload access is added to that remainder. At WAIT, the destination
can acquire earlier source work and the source's prior acquired completion,
but never a newer access sharing the same cell/lane. Branch joins retain only
common receipts and union their possible remaining work. Loop entry forgets
these optional receipts while preserving the separately proved event protocol
and its cell-specific recurrence invariant. GM visibility history is never
cleared by these completion facts.

Fresh reconstruction infers numeric protocols from actual command populations
and original cuts, checks command order, initialization and all original
physical demands, and recognizes ordinary packets only on nonaliased keys.
It does not consume constructor allocation/coverage receipts. Constructor and
checker still share abstract transfer and cut-discovery code; the independent
finite graph oracle is therefore an important, separately scoped check.

### Candidate validation

The latest local artifacts are `cuts-prefix-final/summary.json` and
`historical-gemm.cuts-prefix.pto`, alongside the worktree. On the seven unchanged
inputs, native construction/reconstruction passes. Normalizing this emitted IR
through the compiler does **not** measure whole-compilation performance of the
candidate. Compared with the hash-frozen old structured outputs:

| Case | Later acquired-prefix observations |
| --- | ---: |
| one buffer | 0 |
| two buffers | 0 |
| three buffers | 0 |
| four uses | 0 |
| online softmax | 6 |
| QK matmul | 19 |
| Q projection | 0 |

These are finite scenario observations, not device timings or a count of
executed synchronization commands. Prefix sharing reduced softmax's count from
102 to 6. Its remaining observations are on zero-trip paths; QK still acquires
an incoming panel prefix too late on loop entry. **Quality acceptance is false.**

A separate fresh compilation comparison against existing InsertSync counted
static SET / WAIT / named-barrier sites (each output also has one PIPE_ALL):

| Case | InsertSync | Conservative baseline | Cut-precision candidate |
| --- | ---: | ---: | ---: |
| one buffer | 6 / 6 / 0 | 8 / 8 / 1 | 8 / 8 / 1 |
| two buffers | 12 / 12 / 2 | 16 / 16 / 2 | 16 / 16 / 2 |
| three buffers | 18 / 18 / 3 | 24 / 24 / 3 | 24 / 24 / 3 |
| four uses | 24 / 24 / 2 | 40 / 40 / 2 | 40 / 40 / 2 |
| online softmax | 12 / 12 / 20 | 16 / 16 / 20 | 30 / 30 / 19 |
| QK matmul | 21 / 21 / 2 | 24 / 24 / 3 | 50 / 50 / 3 |
| Q projection | 19 / 19 / 6 | 20 / 20 / 4 | 60 / 60 / 6 |

The candidate does not improve static mechanism count. Maintaining earlier
completion boundaries is not the same as reducing synchronization commands.
These are IR sites, not dynamic counts or runtime measurements. Commands and
input/output/binary hashes are in the local
`mechanism-counts-Ksnb66/summary.json` campaign. Both compositional arms used
the native driver followed by compiler lowering without another insertion pass.

All eight native mutations reject atomically: missing SET/WAIT, duplicate SET,
wrong key, missing retirement, publication before its producer, acquisition
after its consumer, and missing exit acknowledgment. The core passes 1,665,356
assertions with ASan/UBSan, including reentrant protocols, newer accesses after
publication, branch joins, nested empty loops, and randomized cross-cell
traffic. Assertions are enforced in Release builds too. The targeted
`oahs_composition_core`, `oahs_composition`, and `oahs_cuts` gates pass.

Historical GEMM passes native candidate reconstruction under `conservative` and
`may-alias`, retaining its original pairwise alias attribute (130 handoffs,
23 barriers). This is not overlap or device qualification. The candidate has
not received a new frozen corpus sweep or the whole-compilation <=2x campaign;
the baseline results above must not be attributed to it.

Reproduce the native candidate check with:

```sh
python test/experiments/insert_sync/logical_plan/check_cuts.py \
  --driver "$BUILD/tools/pto-test-opt/pto-structured-sync-test" \
  --python-root "$BUILD/python" --output "$RESULTS/cuts" \
  --baseline "$BASELINE/summary.json" --require-quality
```

`BASELINE` is the frozen seven-case benchmark campaign; its round-zero
`structured` outputs must still exist and match their recorded hashes.
Without `--baseline`, the runner compiles existing InsertSync as the reference.
The quality-required command currently fails after recording its results.
The default `oahs_cuts` CTest checks reconstruction and reports quality
separately; a green CTest alone does not authorize the backend switch.

## Remaining work before replacement acceptance

1. Complete incoming/continuation demand cuts across execution domains,
   including first-consumer and empty-path participation. The increment below
   implements same-domain placement and sharing, not this whole milestone.
   Recover softmax and QK boundaries without draining unrelated work or
   inventing completion across a zero-trip loop.
2. Extend finite slot-generation and first/last precision through the same
   transfer/protocol interfaces. The old dense constructor is not yet a bounded
   provider and must not become a hidden fallback.
3. Integrate qualified precision into the general engine's normal path and
   move audited positive effect contracts into the lowering-owned interface.
4. Pass readiness/release quality, final whole-suite checks and device
   qualification separately. Then remove shape-admission/sequence-bridge paths
   and switch the existing structured default.

## Demand-constructor increment

`CompletionDemand` names an original structural scope, publication cut,
acquisition cut, producer, observer, and physical-cell witnesses. It has no
event number. A forward physical-history pass chooses the last relevant
producer access in a sequence; a backward continuation pass computes the last
use of each prospective prefix. A nested child contributes its MAY effects at
its common exit. No branch predicate or numerical trip count is solved.

The completion transfer captures prospective source prefixes at these cuts and
acquires one only when it discharges the complete current source demand. At
most eight prospective cuts per source are retained. Later payload is added to
every remaining-prefix receipt, so an earlier publication cannot complete a
newer access to the same cell. Loss of a prospective snapshot means ordinary
conservative acquisition, not loss of an actual event. Same-observer demands
over several cells naturally share one required producer prefix.

Direct SET/WAIT endpoints currently belong to the same sequence execution
domain. Incoming demand in a child can therefore still publish at that child's
entry, rather than at the ideal outside producer cut. A publication outside a
possibly empty loop is **not** paired with an unconditional per-iteration WAIT.
Supporting that placement requires the remaining incoming/first-consumer
participation work. This limitation must stay visible in QK and GEMM results.

### Protocol sharing and reconstruction

The allocator reserves the canonical rendezvous keys and assigns other keys
across the whole function. Cell witnesses do not receive independent pools.
Several same-direction transfers within one domain share one acknowledgment.
Required return handoffs can discharge that acknowledgment instead: a bounded
pass removes an optional reply only when the actual protocol still proves all
consumption-before-republication obligations. No readiness/release endpoint is
removed by this pass. Scarcity retains the conservative realization.

The checker independently reads actual mechanisms, requires each direct key's
SET/WAIT uses to have one common structural execution domain, and prohibits
sharing that key with an independently executing domain. A vector-clock check
on two copies of the static event word proves every consumption-to-next-SET
edge, including multiple uses of a physical key within one word. The static
edges repeat under translation, giving a periodic protocol certificate—not
the assumption that two arbitrary loop executions establish program safety.
Nested words have disjoint keys; skipped words leave no partial publication.
SET does not advance the source issue gate.

Actual canonical rendezvous packets are expanded into their four commands and
checked by the same two-copy vector-clock routine. Their fixed global
orientation and reserved keys justify sequential reuse across nested/skipped
regions; this does not claim arbitrary independently concurrent domain reuse.
The shared optional reply follows the last forward WAIT but can precede its
consumer payload: it acknowledges event consumption, not consumer completion
or storage release. Construction takes no payload completion credit for it.

Fresh prefix transfer then checks original physical effects, independently of
the constructor's demands and family membership. Native emission retains the
payload snapshot and physical-context retirement checks. A failed emitted-IR
check remains an atomic error; it never triggers successful fallback.

### Bounded completion invariants

For a fixed actual plan, the conservative loop invariant is `Top = E | S`.
One monotone transfer gives the narrower inductive invariant `E | F(Top)`.
The checker proves the physical accesses under it and explicitly checks
backedge closure. The probe does not check hazards or recursively initiate new
probes; the subsequent verifying visit does. Across the entire tree, probes
are limited to **two tree-visit equivalents** in aggregate, including nested
loops, rather than necessarily two literal traversals of the complete tree.

The same rule applies to a local physical cell whose accesses all occur within
one sequence inside a loop. Its initial history is empty, and only visits to
that sequence can create history. `F(Top)` is an invariant over visits, including
arbitrarily skipped visits. Incoming completion can strengthen this invariant;
it is not discarded. Ownership here is an analysis property of the original
physical access population, not a new target ownership mechanism. GM visibility
and actual event state are never cleared by narrowing.

Construction makes at most one further attempt using these established
invariants. The resulting plan must independently establish its own invariants
before emission; otherwise the first plan is retained. Trace counters distinguish
direct handoffs, retained/reused acknowledgments, fallback acquisitions, and
accepted/rejected completion refinement. Visitor counts are transfer statistics,
not wall-clock or byte-allocation budgets.

### Validation and migration status

The finite oracle now accepts per-node, per-invocation trip counts and branch
choices. Tests include independently varying siblings, zero-trip visits,
`while` exits, repeated whole-program calls with the same keys, missing return
paths, moved publications, scarce pools, multi-cell prefix sharing, and bounded
work at 128 levels of nesting.

`Program` explicitly requires cells-sized zero effect vectors on structural
nodes; only physical Operation leaves carry effects. All six constructor/checker
entry points test this contract. Native test-only modes force a one-key fallback
pool and corrupt an optional pre-emission refinement. The latter must retain the
initial verified plan; an actual emitted-IR verification failure still fails
atomically. Native traces assert direct-prefix reuse, fallback packet execution,
owned-cell narrowing, accepted refinement, and rejected-refinement rollback.
The fallback fixture runs nested, empty, skipped, and repeated whole-function
executions with the same observer/key population and has packet-corruption tests.

Transactional staging now preserves the transitive symbol closure and original
module scopes. Native tests retain real helper calls, test nested symbol
shadowing, and reject missing/invalid contracts. These test the kernel
constructor, not a claim that running the whole-function pass independently on
every private helper body is qualified.

`demand_manifest.json` freezes eight inputs, their alias assumptions, and replay
scenarios. The historical input retains its payload and original pairwise alias
attribute; only obsolete canonical-pass RUN/CHECK comments were removed.
`benchmark_buffers.py` now uses this eight-case manifest. The previous
`checkpoint/manifest.json` stays unchanged for historical comparisons.

Run the candidate correctness/observation campaign serially:

```sh
python test/experiments/insert_sync/logical_plan/check_cuts.py \
  --constructor demands --driver "$BUILD/tools/pto-test-opt/pto-structured-sync-test" \
  --python-root "$BUILD/python" --output "$RESULTS/demands"
```

Add `--require-quality` to require both non-worsening observed completion
boundaries and non-increasing executed synchronization in every frozen scenario.
Static mechanisms, initialization/retirement traffic, input/output hashes,
actual binary provenance, and commands are recorded separately. A passing
`oahs_demands` CTest does **not** mean the quality gate passed. Normalization of
native output is not a whole-compiler timing result for the candidate.

The previous corpus admission counts do not qualify this new constructor.
No default is changed; no device or handwritten-GEMM performance conclusion is
claimed. Full corpus parity, first-consumer placement across control domains,
bounded generation precision, and the final compiler/device gates are still
required before the migration is complete.

### Local measurements for this increment

These measurements are from the working-tree increment over `fccb51b40571`,
not a released or device-qualified revision. The targeted native build
succeeded. The six composition/demand/cut/structured
CTests passed; the final standalone C++17 ASan/UBSan run passed **2,139,545
assertions**. LeakSanitizer was disabled because this runtime does not support
its process inspection. Independent varying-trip tests are finite evidence,
not a device or unbounded-execution proof.

All eight frozen candidate inputs passed native construction/reconstruction
and full-compiler PTO normalization. The historical GEMM also emitted C++.
All seven native corruption checks rejected atomically. Static mechanism
counts below exclude the one terminal `PIPE_ALL` present in each output:

| Input | InsertSync SET / WAIT / named barriers | Demand candidate |
| --- | ---: | ---: |
| One buffer | 6 / 6 / 0 | 4 / 4 / 1 |
| Two buffers | 12 / 12 / 2 | 8 / 8 / 2 |
| Three buffers | 18 / 18 / 3 | 16 / 16 / 2 |
| Four use | 24 / 24 / 2 | 28 / 28 / 4 |
| Online softmax | 12 / 12 / 20 | 11 / 11 / 21 |
| QK matmul | 21 / 21 / 2 | 16 / 16 / 4 |
| Q projection | 19 / 19 / 6 | 19 / 19 / 5 |
| Historical GEMM | 44 / 44 / 21 | 63 / 63 / 20 |

**Quality acceptance is false.** Fewer static sites do not establish better
overlap: seven cases still have later-prefix observations, and five have at
least one frozen scenario with more executed synchronization. In particular,
GEMM is compilable but still over-synchronized. These results do not justify a
default switch or a performance-improvement claim.

The separate conservative-baseline corpus rerun retained exactly the same
197 passing `(input ID, prepared-source hash)` pairs among 363 inputs as the
previous campaign: no admissions gained or lost. This is baseline regression
evidence, **not** a corpus sweep of `constructDemands`.

The local build stores detailed hashes, commands and observations in
`test-results/oahs-demands-quality/summary.json` and baseline compatibility in
`test-results/oahs-demand-baseline-corpus/summary.json`. These are local campaign
artifacts, not checked-in qualification certificates. The checked-in manifest
and runner above reproduce the eight-case comparison without those artifacts.
The initial `oahs_focused` run could not locate `libisl`. Review found the GCC
copy outside the normal loader search path; setting
`PTOAS_EVENT_MODEL_ISL_LIBRARY=/usr/lib/gcc/x86_64-redhat-linux/15/libisl.so.23`
enabled all 15 checks preceding allocation. That run then failed the existing
over-capacity fixture: nine readers of one load correctly share one stream,
so they never exercised the intended scarce-domain policy. The repaired fixture
overwrites the shared input between consumers, retaining the original strict
scarce-domain requirement. Its exact expectation is 17 total streams, with
eight dedicated reverse streams and one dedicated forward stream; at least
eight occupied-key trials exercise sharing in the nine-stream forward domain.
The **complete focused gate passed in 149.17 seconds** with this fixture.
The repaired composition/core/demand gates also passed, including explicit
refinement rollback and canonical-packet mutation checks.
Device and candidate paired whole-compilation timing gates were not run.

All 13 InsertSync lit tests passed before the review-only checker additions;
the focused rerun includes all 13 observation-runner unit tests, including
the new constructor-specific report-schema test.
The lit tests used the existing Python-binding build with multithreading
disabled and one test worker; the default local launcher selects an
incompatible Python and cannot import `ptoas._core`. No repository RUN lines
were changed to bypass compiler checks.
The full 1,880-test lit campaign was also attempted with the corrected serial
launcher, but stopped incomplete before the 10-minute campaign cap after its
observed rate made completion within that cap infeasible. It is **not** a
full-suite pass; only the completed 13-test subset is counted above.

## Opt-in pass-pipeline qualification

The subsequent integration exposes the existing demand constructor through
`planner=composition`, without changing any default or the demand algorithm.
Function metadata records `pto.insert_sync.producer = "composition"` and the
actual `pto.insert_sync.precision` setting. Refusal remains strict, including
authored events and unknown hardware-contract strings. The Python launcher
already forwards these options to the native CLI unchanged. The native test
driver's corruption modes are not exposed through production options.

The frozen corpus run with `--corpus-constructor=demands` now exercises the real
pass pipeline, not just one selected function in the test driver. On the same
363 raw/prepared snapshots it admitted exactly the conservative baseline's
197 input-ID/source-hash/prepared-hash tuples: **45/150 PTOAS** and **152/213
PyPTO/pypto-lib**, no gains, losses, crashes or timeouts. Both modes keep the
same semantic refusals; 56 inputs require unsupported visibility and 56 first
refuse `pto.load_scalar`. This is compatibility, not device qualification.
The corpus option applies only to the corpus rows: the runner's separately
categorized native mutation checks continue to exercise the baseline.

Reproduce with the frozen manifests and their original source snapshots:

```sh
python test/experiments/insert_sync/logical_plan/check_composition.py \
  --driver "$BUILD/tools/pto-test-opt/pto-structured-sync-test" \
  --opt "$BUILD/tools/pto-test-opt/pto-test-opt" \
  --corpus-constructor demands --corpus-manifest "$PTOAS_MANIFEST" \
  --corpus-manifest "$PYPTO_MANIFEST" --output "$RESULTS/demand-corpus"
python test/experiments/insert_sync/logical_plan/benchmark_buffers.py \
  --python-root "$BUILD/python" --output "$RESULTS/demand-compiler" \
  --arms existing demands --repeats 3 --warmups 1 --timeout 90 \
  --emit-cpp --require-ratio 2
```

For historical reproducibility, benchmark arm `composition` still means the
previous conservative `structured-precision=false` invocation. The new arm
`demands` selects `planner=composition` with precision enabled. Arm names and
actual producer metadata are checked separately; no older result is relabeled.
All eight manifest inputs, contracts and scenarios are unchanged. The payload
observer accepts only the two recognized GM metadata values; the manifest
records the selected contract and original pairwise alias attributes remain
part of the payload comparison.

The local paired campaign used alternating execution order, one warmup and
three measured rounds per arm, with no concurrent build or test. It includes
synchronization in full PTO emission, followed by untimed C++ emission:

| Input | Median demand / InsertSync whole-compilation ratio |
| --- | ---: |
| One buffer | 1.035 |
| Two buffers | 1.012 |
| Three buffers | 1.016 |
| Four use | 1.017 |
| Online softmax | 1.062 |
| QK matmul | 1.011 |
| Q projection | 1.009 |
| Historical GEMM | 0.956 |

All eight complete paired populations passed the <=2x compilation gate. These
small differences are local compiler measurements, not a device speedup claim.
The unchanged synchronization-quality gate is still false. This qualification
must be repeated after subsequent algorithm changes; it does not certify future
placement or key-sharing implementations.

Local detailed artifacts are `test-results/oahs-demand-native-corpus/summary.json`
and `test-results/oahs-demand-paired-compiler/summary.json` in the existing build.
They record commands and binary hashes. The external corpus snapshots/manifests
are prerequisites, not bundled by these scripts; the eight benchmark inputs
and manifest are checked in. A targeted lit case tests native pass selection,
both precision settings, invalid contracts, and actual full-compiler emission.

## Late event-key assignment

Logical demand placement now precedes physical event numbering. Logical keys
identify publications and acquisitions in one static scope word; they are never
emitted. Canonical fallback keys remain reserved, and independently executing
scopes still have disjoint physical keys. Optional acknowledgment removal checks
only its changed scope word, with at most one trial per directed family there.

A two-copy logical protocol check records each virtual key's first publication
clock, next-copy publication clock, and first consumption tick. For uses of one
direction in publication order, append a use to an existing physical color only
when both conditions hold:

```text
new publication knows the preceding use's consumption
the color's first publication in copy 2 knows the new use's consumption
```

The first condition proves within-word reuse; the second proves the per-color
last-to-first edge. These are actual acquired WAIT-consumption clocks, not
lexical intervals or assumed payload completion. Greedy selection tries only
the fixed physical pool; it does not search colorings or symbolic relations.
Clock construction and key trials depend on static commands and the fixed lane
and key populations, not trip counts. Maps add ordinary logarithmic lookup cost.

If a key cannot be assigned greedily, its publication is removed and its
acquisition receives a canonical packet at the same cut. Other feasible
handoffs remain intact. The replacement acquires a full source prefix at that
later cut and can therefore add ordering; it does not move payload or pretend
that scarcity proves global infeasibility. The complete resulting actual word,
including expanded packets, must pass rearm verification, and the actual plan
must separately pass fresh physical completion verification before emission.
Failed verification is an error, never successful fallback after emission.

Native traces record `protocol_keys`, `shared_protocol_keys`,
`allocation_fallback_keys`, and `allocation_fallback_scopes`. The latter counts
scopes containing at least one fallback key, not keys. Retained reverse ACKs
remain included in `shared_acknowledgments` even if every forward demand in
their family fell back: they still carry checked causal edges, and removing
them would require another protocol proof. Tests include many logical generations
sharing one key per direction, simultaneous early publications that cannot
share, independent scopes competing for keys, repeated/skipped executions, and
the native eight-publication overlap fixture. The compiler pool remains IDs 0–5,
with ID 0 reserved for canonical packets. The fixture must retain five assignable
direct handoffs while realizing the other three conservatively.

This stage does not solve incoming first-consumer participation or previous-visit
release placement. Those remain separate quality obligations; key sharing alone
does not qualify a default switch or a GEMM performance claim.

### Local key-realization measurements

The following compares the preceding `cc86a5928` demand implementation with
this key-realization increment on the unchanged eight-case manifest. Counts
are static SET / WAIT / named barriers; each output additionally has one
terminal `PIPE_ALL`. These are not device timings or executed command counts.

| Input | Before late numbering | After late numbering |
| --- | ---: | ---: |
| One buffer | 4 / 4 / 1 | 4 / 4 / 1 |
| Two buffers | 8 / 8 / 2 | 8 / 8 / 2 |
| Three buffers | 16 / 16 / 2 | 12 / 12 / 3 |
| Four use | 28 / 28 / 4 | 20 / 20 / 8 |
| Online softmax | 11 / 11 / 21 | 12 / 12 / 21 |
| QK matmul | 16 / 16 / 4 | 16 / 16 / 4 |
| Q projection | 19 / 19 / 5 | 18 / 18 / 6 |
| Historical GEMM | 63 / 63 / 20 | 70 / 70 / 24 |

**Quality acceptance remains false.** In this historical key-realization
increment, late scarcity fallback added stronger completion after the demand
transfer finished. Later demands did not exploit that extra completion.
Consequently, that experimental change increased synchronization in softmax
and GEMM. The bounded replay documented below is a subsequent increment,
not a property attributed retroactively to this measurement.

The targeted native build, composition core/demand gates, composition gate,
and `oahs_focused` passed. The C++17 ASan/UBSan run passed **2,158,773
assertions**, with unsupported leak inspection disabled. Native overlap tests
retain five direct handoffs and replace three unassignable acquisitions;
wrong-key, duplicate-SET, and dropped-WAIT mutations reject atomically.

The frozen corpus rerun preserves all 363 statuses and raw/prepared hashes:
45/150 PTOAS and 152/213 PyPTO/pypto-lib admissions. The serial paired full-PTO
compilation run (three measured rounds, one warm-up) passes the median <=2x
gate on all eight inputs; per-case medians range from 0.986 to 1.023 times
InsertSync. Untimed C++ emission also passes. Reproduce with the commands above;
this run used output directories `oahs-key-sharing-corpus` and
`oahs-key-sharing-compiler` below the existing build's `test-results` directory.
Artifacts are local, and no device or complete system-test campaign is claimed.

## Bounded fallback-completion replay

After the initial plan and optional completion-invariant refinement, perform
at most one additional call to the same demand constructor. Freeze failed
forward-demand identities `(scope, publication, acquisition, source, observer)`
and the same invariant hints. At those cuts, the replay performs the existing
canonical `acquire()` on the actual forward state rather than inserting a
packet after construction. It creates no corresponding logical publication or
acquisition receipt. Later demands can reuse the stronger completion.
Prospective source-cut snapshots remain proposals, never emitted-event facts.

This is bounded to three constructor calls total, not an allocation fixed point
or another planner. The replay must pass the complete mixed protocol check and
fresh physical verification. It is rejected if a formerly assignable forward
demand becomes unassignable, or if static command cost increases in any
immediate Sequence (a rendezvous costs four commands). Since those commands
participate on every visit to that sequence, the latter protects independently
varying loop and branch executions, not only whole-function static totals.
Failure retains the exact previously verified plan before native emission.

Canonical packets also participate in the unnumbered protocol's causal clocks.
They use a reserved virtual sentinel, disjoint from logical IDs, and are excluded
from color candidates. Final checking still expands their actual physical IDs.
Ignoring their acknowledgment edges during coloring would unnecessarily reject
useful replays even when the actual physical plan is safe.

Trace counters separate replay attempts, rejected replays, commands removed,
and replayed fallback demands. `allocation_fallback_keys` continues to count
late numbering failures in the selected plan; a fallback already performed
during replay is counted by `replayed_fallback_demands` instead. Native test-only
modes disable replay or corrupt it before checking, requiring byte-identical
rollback. The overlap fixture needs only one packet after replay instead of
three; another finite test includes fallback in the middle of a scope and a
later new write that still requires its own handoff.

This does not establish general incoming/continuation placement or qualify
whole-plan overlap relative to InsertSync. The default remains unchanged.

### Local replay measurements

The final targeted build and native composition/demand gates pass, including
the exact rollback and mixed-packet corruption checks. `oahs_focused` passes
(144.79 seconds); the final core/demand rerun takes 19.08 seconds. The final
C++17 ASan/UBSan run passes **2,166,959 assertions**, with unsupported leak
inspection disabled. Assertions are not added across repeated runs.

The unchanged eight-input campaign records GEMM at **62 SET / 62 WAIT / 21
named barriers**, down from 70 / 70 / 24: 19 static commands removed. All other
static counts equal the preceding key-realization table. Q projection accepts
an equal-cost replay; the implementation does not claim that every accepted
replay improves quality. The overlap fixture removes ten commands: two redundant
packets and one redundant reverse ACK pair.

The final summary still records `quality_qualified=false`. First-consumer and
previous-visit boundaries remain unresolved; a reduction from the preceding
experimental plan is not acceptance against InsertSync or a device speedup.

The final frozen corpus rerun again preserves all 363 statuses and raw/prepared
hashes (45/150 PTOAS, 152/213 PyPTO/pypto-lib). The eight serial paired compiler
populations pass the median <=2x gate, with three measured rounds and one warm-up;
per-case median ratios range from 0.986 to 1.039. Untimed C++ emission passes.
The commands above were used with local output directories `oahs-replay-corpus`
and `oahs-replay-compiler`; no device or full system-test run is claimed.

## Incoming first-consumer episodes

An optional demand summary connects a producer in a loop's parent Sequence to
the first relevant direct consumer in its body. It uses physical ranges and
original structural cuts, not kernel names or a solved predicate. The current
guard adapter qualifies index `scf.for` with constant positive unit step; all
other admitted loops retain ordinary demand construction. Both original bounds
must dominate the publication cut. The source must have no body MAY access on
the consumer's cells, so incoming completion cannot be confused with a current
iteration's producer or storage reuse.

For each selected incoming prefix, emit the forward pair below. A single
exit reply is shared by all these first-consumer waits on the same directed
pair and loop:

```
if original_lower < original_upper: SET P -> Q
for iv = original_lower to original_upper:
    ... independent work ...
    if iv == original_lower: WAIT P -> Q
    first consumer
    ...
if original_lower < original_upper:
    SET Q -> P
    WAIT Q -> P
```

Publication is immediately after the latest required producer, subject to
bound availability; acquisition stays at the first actual consumer. Zero-trip
and untaken enclosing paths execute no episode commands. The exit reply proves
event rearm, but receives no unconditional payload-completion credit.
If earlier first consumers' validated completion receipts already discharge a
later incoming demand, it needs no separate event. A fixed-size per-direction
summary intersects their uncovered remainders and tests the actual later
demand against that summary plus S. This does not infer coverage merely from
lexical positions or scan an unbounded population of selected entries. The
no-body-source-effects qualification makes that completion stable for the
later demand's cells.

The completion receipt stores the **uncovered remainder**, including all later
source work. Unioning the complete loop MAY summary S into that remainder R
makes the transfer `pending := pending & R` idempotent and preserves every
possible body generation. Applying this stable external-prefix credit during
the finite loop proof is not a claim that a First WAIT executes every iteration.
The receipt is discarded at the loop exit; incoming physical history survives
the zero-trip join.

Reconstruction classifies the actual sync-only guards against original SSA
bound/IV identities and independently rederived first-consumer cuts. The exact
generated operation identities are captured before any test mutation. Only
these generated and validated extra operations are excluded from payload
comparison and fresh tree import; an injected balanced packet is rejected too.
It reconstructs
the combined nonempty episode word in original execution order and checks two
copies for consumption-before-rearm. Episode keys are globally reserved and
disjoint from ordinary demand/canonical keys. Conditional episodes provide no
assumed causal edges to ordinary key coloring. Multiple cells can therefore
have independent early incoming prefixes without flattening their participation
into an unconditional sequence.

Failure of an optional pre-emission entry proposal retries the same demand
constructor with entry precision disabled, once and without recursion back to
entry selection. This permits at most four constructor calls including ordinary
refinement/replay, rather than the prior three. Work from the discarded attempt
is retained in the counters. An emitted-IR verification failure still rejects
atomically; it is never converted to fallback success. A replay is accepted
only if its conditional episode population is unchanged, in addition to the
existing per-Sequence command-cost and complete verification checks.

`entry_episodes`, `entry_reply_families` and `rejected_entry_proposals` expose the selected path. The
native fixture uses runtime/nonzero bounds, an enclosing conditional and
unrelated source work inside the loop. Tests check the exact first two acquired
source prefixes, independently varied repeated executions, no empty-path
events, pre-emission fallback, and atomic rejection of wrong first/nonempty
predicates, missing first/reply waits, a moved first wait and an injected packet.
An original event-only conditional remains an explicit-synchronization input
refusal, never a newly generated precision guard. Two native forward episodes
must share exactly one reverse acknowledgment; finite tests also include an
intervening parent-side source acquisition before a shared incoming prefix.

This is incoming-readiness precision, **not** previous-visit release precision
or replacement-quality acceptance. It does not solve rotating generations or
make guarded first-consumer placement a new admission requirement.

### Local incoming-episode measurements

The eight unchanged inputs construct and reconstruct. The incoming fixture
checks original runtime bounds, no empty-path events, exact separate acquired
prefixes and two forward handoffs sharing one reply. Authored conditional
events refuse explicitly. Guard, missing-wait, moved-wait and balanced-injection
mutations reject atomically; pre-emission corruption instead takes the verified
entry-disabled construction. C++17 Clang ASan/UBSan passes **2,176,071
assertions** with unsupported leak inspection disabled.

QK has 18 SET / 18 WAIT / 4 named barriers, versus 16 / 16 / 4 before this
increment. Its measured nonempty executions use 41, 75 and 551 total sync
commands for 1, 2 and 16 iterations, versus InsertSync's 45, 79 and 555. The
first-panel publication no longer includes the independent later panel load.
This does not fix all later-iteration release boundaries.

Softmax has 17 / 17 / 21 static commands versus 12 / 12 / 21 before. Its
2- and 16-iteration executions use 56 and 518 commands versus InsertSync's
45 and 451. The empty/one-panel path uses 17 versus 16. These regressions are
not hidden by the successful incoming-boundary test: entry precision is not
yet an overall quality improvement. Other static counts are unchanged,
including historical GEMM at 62 / 62 / 21. Every output additionally has one
terminal drain. **`quality_qualified` remains false; defaults are unchanged.**

Three serial paired compiler rounds after one warm-up pass the median <=2x
gate on all eight inputs, with median ratios 0.984–1.057 against InsertSync.
Untimed C++ emission passes. The local artifact directory is
`test-results/oahs-entry-compiler` under the existing build. These measurements
are not device results and do not qualify comparison with handwritten GEMM.

The final exact-build core/composition/demand/focused gates all pass (179.35
seconds total; focused 154.81 seconds). The final frozen corpus campaign
`test-results/oahs-entry-final-corpus` preserves all 363 statuses and original /
pre-sync hashes: 45/150 PTOAS and 152/213 PyPTO/pypto-lib admissions. The
software-architecture, algorithms/performance and correctness/design reviewers
accept this opt-in increment subject to those passing checks, not general
replacement or performance qualification. No full system suite or device run
is claimed.

## Closed recurring handoffs within the demand constructor

The next optional provider handles a complete storage-access lane word in one
Sequence. It uses the existing demand constructor, global completion state,
logical event allocation and actual-word reconstruction, not the old cut
constructor's initialization/retirement visitor. For two lanes its word is:

```text
Q: SET Q -> P       // includes Q work from a previous visit, if any
P: WAIT Q -> P
P: access storage
P: SET P -> Q       // earliest cut after this group's last relevant access
Q: WAIT P -> Q      // first relevant consumer, not the region's entry
Q: access storage
```

The first SET publishes Q's existing prefix on the first visit; it does not
wait for a fictitious previous payload. On later visits it follows Q's prior
access and acquisition in Q's queue. Each visit starts with idle/consumed keys
and consumes every publication. There is no external seed, tail publication,
first-iteration exception or exit acknowledgment. A skipped visit executes no
ring commands. The ordinary actual two-copy word checker proves the wraparound
consumption-before-rearm edges and progress, with SET fire distinct from a
source-side issue barrier.

The physical certificate is separate. Original effects must establish that the
word contains the whole cell-access population in its repeated owner. First
and last boundaries must be physical operations that are immediate children
of the same Sequence. No nested structural child may access a certified cell,
the nearest repeated owner must be a counted For, and each directed edge must
occur once per word. A surrounding choice may skip the whole Sequence, but
cannot skip individual endpoints. Partial or repeated-direction words lose
this optional precision. One cell belongs to one such certificate.
Let E be pending history entering that owner, and S its complete MAY effects.
Only after an actual certified WAIT can internal-generation completion reduce
the receiving observer's cell history to `pending & E`. The checker explicitly
requires `pending` to be within `E | S`. Incoming histories, including outer
loop overapproximations, survive this credit. Ordinary prefix receipts still
carry completion across cells, and subsequent payload always adds new pending
history. The zero-trip join retains E. This is a scoped internal-generation
certificate, not a permanent completed bit for a storage slot.

Identical multi-cell lane shapes share one protocol; each cell retains its own
reader/writer effects and incoming-generation mask. This is compatible prefix
coverage, not equality of the cells' effects. Two-group words can also share
when one group's exact first/last boundaries coincide and the other group's
union remains a noninterleaving single-lane interval. Each shared direction
retains a common publication or acquisition within the visit. The wrap pair
can move together to the earliest merged producer: its source publication
still follows the last lane's common previous-visit group and, being earlier,
does not capture additional intervening source work. Initial external history
is still retained in E. Two bounded grouping passes replace pairwise
family search. This covers separate operand preparations feeding one consumer
without recognizing a kernel name or requiring a periodic expression.
A family widened in the first directional grouping pass is excluded from the
second pass, so a final family still has one unchanged group relative to its
original cell words.

The candidate uses disjoint virtual IDs above physical entry keys. Ordinary
global numbering assigns ring keys dedicated colors, excluding them from
unrelated raw reuse, while checking the complete combined word. Reconstruction
rederives shapes and infers exact numeric endpoint populations and same-cut
ordering from actual mechanisms before granting any cell credit. It does not
consume the constructor's cell assignments or selected protocol metadata.

There is at most one optional candidate attempt after the verified demand baseline.
Before constructing it, cheap per-domain key-capacity and per-Sequence minimum
command-count checks reject infeasible or non-saving ring populations.
Allocation, reconstruction or quality rejection restores that baseline exactly.
The quality check preserves conditional entry episodes and requires ordinary
command cost not to increase in *any* independently executing Sequence, with
a strict decrease somewhere. It does not amortize header commands against a
guessed trip count. Counters `ring_candidates`, `rejected_rings`,
`ring_candidate_commands_removed` and `cut_cycles` distinguish attempted, rejected and
actually certified behavior. This command-count check alone is not an overlap
or device-performance qualification; the eight-case boundary gate remains
separate.
The reduction counter compares the entire candidate's ordinary commands with
the baseline. It includes any ordinary allocation/placement changes caused by
reconstruction; it does not attribute every removed command solely to a ring.

Optional discovery is capped at 1,048,576 static node/cell combinations and
retains the existing bounded word/alternative limits. The constructor reuses
one immutable discovery result; fresh verification deliberately rederives it.
Construction and reconstruction use one structural endpoint-pattern generator,
while the independent finite graph oracle implements event semantics separately.
`node_visits` and `cell_visits` accumulate attempted transfer/analysis work,
including rejected candidates; protocol and refinement counters describe the
returned plan. The new ring counters separately expose optional selection.
These work units and the static cap are not a wall-clock or allocation-byte
bound; paired full-compiler timings remain required.

### Local closed-ring qualification

All three reviewers accept this opt-in increment with the following final
local checks: core (4.99 s), composition (3.04 s), demand/native mutation
(16.41 s) and focused (144.31 s) gates pass. The added shared native fixture
constructs and reconstructs two multi-cell handoff families, executes no
partial protocol on skipped visits, and rejects five in-place protocol
mutations. The independent finite graph suite passes **2,198,295 assertions**
under Clang C++17 ASan/UBSan, with unsupported leak inspection disabled. It
includes staggered producers, intervening unrelated last-lane work, an invalid
interleaving merge, exact rollback, nested-segment refusal and cap fallback.
The targeted public-option lit test also passes.

One-/two-/three-buffer outputs now use respectively 4/4, 8/8 and 12/12 SET/WAIT
sites and **zero named barriers**, plus the unchanged terminal drain. These
remove respectively 1, 2 and 3 named barriers from the preceding demand
revision, without adding event sites. InsertSync uses 6/6, 12/12 and 18/18
SET/WAIT sites and 0, 2 and 3 named barriers. The other five regression inputs'
static counts are unchanged, including historical GEMM at 62/62/21. The
remaining later-prefix observations are unchanged too: this is not yet a
solution to cross-slot early release. **`quality_qualified` remains false.**

Three paired compiler rounds after one warm-up pass the <=2x median gate on
all eight unchanged inputs, with median ratios **0.969–1.035** against
InsertSync; untimed C++ emission passes. Artifacts are in the existing build's
`test-results/oahs-rings-compiler` and `test-results/oahs-demands` directories.
The frozen corpus campaign `test-results/oahs-rings-corpus` preserves all
**363** input hashes, prepared-IR hashes and outcomes exactly: **45/150 PTOAS**
and **152/213 PyPTO/pypto-lib** admissions. The shared test fixture is not added
to those frozen denominators. No device execution, full system suite, default
switch or comparison with handwritten GEMM performance is claimed.
