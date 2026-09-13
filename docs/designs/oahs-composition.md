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
Its base constructor derives direct transfers from physical completion demands
without cycle recognition. Optional bounded refinements now share recurring
handoff families and recover qualified incoming and periodic release cuts; they
use `discoverCuts()` as a precision provider, not a semantic admission gate.
It is still a **migration candidate**, not a qualified replacement: the full
benchmark overlap/command-quality gate remains open. Precision disabled skips
periodic scalar qualification and retains the same general composition engine.

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

## Early recurring release: unconditional counted-loop candidate

The closed-ring fallback publishes a storage release at the next visit's
first cut. That publication may include intervening unrelated work on its
source pipeline. An optional deferred-wrap candidate instead publishes at
the immediate cut after the last relevant access, retaining the same physical
cell witnesses and global event-key assignment:

```text
for iv = originalLower to originalUpper step 1:
    if iv != originalLower: WAIT last -> first
    first group
    SET/WAIT between the remaining groups at their original handoff cuts
    last group
    SET last -> first
if originalLower < originalUpper: WAIT last -> first
```

This provider is limited to a complete word that executes unconditionally in
the counted loop's body, with only physical operations and empty scalar nodes.
It rejects an enclosing recurrence and any later receiving-pipeline payload
in the continuation, including continuation outside an enclosing choice: the
final acquisition must not newly stall independent receiving-pipeline work.
It does not add private loop-carried state, rewrite
original loop bounds/yields, or solve arbitrary branch conditions. Words
under choices and loops without the qualified original-IV contract retain
the existing construction. The final acquisition must be at the immediate
original post-loop cut. Empty loops execute no ring events; one-trip loops
skip the first acquisition and consume their only release at exit.

The first cut has no previous internal generation. Removing the conservative
seed for that generation is an abstract initialization fact, not a physical
acquisition: all history E entering the owner is retained. Later visits have
an actual previous-release acquisition. Exit protocol cleanup does not create
unconditional parent completion across a potentially empty loop.

The wrap publication is a producer-prefix receipt, not a receipt restricted
to the buffer that motivated it. Fresh checking reconstructs its source
accesses from the actual publication cut. For each cell, it preserves E,
source accesses after that publication and current-visit accesses before the
receiving wait. Only covered source-lane read/write bits may be removed;
other-lane bits and the separate GM visibility history remain unchanged.
Thus a release after a store can carry the store's completion back to a later
store without asserting a GM-read visibility guarantee. Per-owner access
extents bound this calculation independently of numeric trip counts.

For G groups and T executed iterations, synchronization remains 2*G*T
commands. Each transformed wrap adds one static exit WAIT and two guard
sites. This is a release-placement optimization, not an event-count saving;
the benchmark report records executed comparisons/branches separately.
The native checker reconstructs both generated guards from the original IV
and bound SSA identities, with the existing exact payload snapshot unchanged.
Tests mutate the guards, remove waits and move the acquisitions across their
required boundaries. The source-envelope and combined-protocol checks each
have an aggregate optional-work ceiling of 1,048,576 units, including a
charge-before-copy bound on repeated parent/body protocol words. These are
representation/scan limits, not wall-clock or integer-solver budgets.
Optional rejection retains its stage and reason in the trace while returning
the untouched baseline plan; it never replaces the baseline's success status.

### Native qualification of the unconditional candidate

The four targeted build targets pass, as do `oahs_composition_core`,
`oahs_demands`, `oahs_composition` and `oahs_focused` (the focused gate takes
154.58 seconds). The targeted public-option lit case passes. Clang C++17
ASan/UBSan passes **2,201,494 assertions**, with unsupported leak inspection
disabled. No full system suite or device test was run.

The native fixture exercises two actual deferred families. For T iterations
it executes 4*T SETs and 4*T WAITs, including the guarded final acquisitions;
zero trips execute no events. It evaluates 2*(T+1) comparisons and branches,
including two exit checks on the empty path. These scalar-IR counts are not
machine instruction or device timing measurements. The fixture includes the
GM destination write at the last source operation, so the cross-cell prefix
transfer is necessary for acceptance. Removing a final wait from the optional
candidate retains the verified baseline and reports a protocol-stage rejection.
Eight native emission mutations reject atomically.

All eight unchanged regression inputs, including historical GEMM, pass three
paired compiler rounds after one warm-up, with median ratios **0.988–1.047**
against InsertSync and successful untimed C++ emission. **None exercises this
unconditional provider:** the buffering words are conditional, while the
remaining cases fail its optional shape/cost eligibility. Their synchronization
counts and later-prefix observations are unchanged. The improvement established
here is the isolated native/core release-placement behavior, not an improvement
on the eight-case quality gate. **`quality_qualified` remains false.**

The frozen corpus preserves every original/prepared hash and outcome across
all 363 records: **45/150 PTOAS** and **152/213 PyPTO/pypto-lib**. The new fixture
is excluded from those denominators. Campaign artifacts are in the existing
build's `test-results/oahs-deferred-corpus`, `test-results/oahs-deferred-compiler`
and `test-results/oahs-demands` directories. The default planner is unchanged.
One remaining cost limitation is that continuation eligibility checks payload
lanes, not a later synchronization command that could relay the exit stall to
another lane; that needs qualification before any replacement-quality claim.

## Periodic release within the same demand constructor

The next refinement supports complete leaf words under qualified periodic
choices. A word contains only fixed physical operations and empty scalar cuts;
its payload and source ordering are therefore identical on every active
residue. Arbitrary choices, nested control and unknown expressions still use
the existing demand/closed-ring baseline when this optional fact is unavailable.
No kernel-name or pipeline-name recognition is involved.

`PeriodicScalar` is extracted from the existing structured adapter and reused
by both adapters. There is no duplicated scalar grammar. The new caller uses
only its periodic fragment, with startup splitting disabled, an original index
IV, constant nonnegative lower bound, and unit step. It evaluates at most one
period of at most 32 ordinals. All choices in the owner must be qualified with
a common period; their original nesting determines the active mask without
enumerating independent Boolean assignments. Nested loops are declined by this
provider. The core validates owner ancestry and mask propagation, and native
reconstruction rederives the facts from the original scalar IR.

If the first active ordinal is r and the original lower bound is L, emission
uses checked, representable `firstRaw = L+r`:

```text
original loop and original choice:
    if iv != firstRaw: WAIT previous release
    original first group ... original last group
    SET release at the immediate last-use cut
if firstRaw < originalUpper: WAIT final release
```

The first loop iteration need not execute the word. An empty loop, a loop
ending before its first active ordinal, and skipped residues execute no partial
protocol. The complete ring word provides the consumption-before-republication
cycle; the actual combined owner protocol additionally checks every exit residue
through two periods and repeated whole-owner invocations. This is bounded
qualification of a structural recurring protocol, not general loop unrolling or
a device proof.

Outside-word history is retained across skipped residues. Source receipts keep
incoming history, current-visit source work, the source suffix after publication,
and all owner source effects outside the word. Internal-generation normalization
is restricted to cell witnesses whose complete access population is inside the
word; the code also explicitly retains outside bits and never clears GM
visibility history. A cheap MAY-write-overlap prefilter excludes a release family
when its source writes the same physical cell inside and outside the word.
For ordinary buffering this keeps shared-GM-output families closed while allowing
early input-slot release. It does not infer disjointness from SSA names.

Selection is a deterministic batch of at most eight complete families with
disjoint cell witnesses and directed keys. One combined fresh check accepts the
selected batch or returns the untouched baseline. Skipped, selected, accepted
and rolled-back counts are separate; there are no individual-family acceptance
claims after a failed combined check and no per-family backtracking campaign.
The publication crosses the backedge strictly earlier in dynamic cut order:
`first(v) < tail(v) < first(v+1)`. This does not promise a smaller source prefix
in every input, nor does it turn extra guard execution into a free operation.

The previous continuation limitation is tightened here. Actual synchronization
commands, including nested and other-owner cleanup, contribute their issuing
lanes to the suffix. Only a validated adjacent cleanup prefix belonging to the
same owner at its exact immediate-exit cut is exempted. A publication/acquisition
interleaved into that batch, a different exit cut, or another owner's cleanup
does not receive the exemption. This permits multiple slot waits at one common
retirement boundary without hiding later relay commands.

### Bounded work and evidence scope

There are separate explicit optional-stage ceilings, not a single wall-clock
budget: scalar qualification, eligibility/continuation scans, receipt cells,
actual command population and protocol-word work each have a 1,048,576-unit
ceiling. Repeated demand analysis/closed-ring discovery has a separate
134,217,728-unit representation reservation, checked before either analysis
runs. It accounts for the node/cell scan with bounded lane, group and alternative
widths, cell signature/key matching, and command population sorting. This is a
conservative reservation, not an actual instruction count; its work/refusal
counters are separate from protocol work. Zero allowance retains the byte-for-byte
same closed-ring mechanism plan. `deferred_ring_candidates` counts the families
selected after the max-eight/disjointness filter, not every eligible family;
`deferred_skipped_families` records the difference.
Scalar qualification also caps expression DAGs at 256 values and
structural/expression depth at 64. Residue evaluation is charged for its fresh
per-condition cache. Protocol products are charged before copying words. The
trace separates DAG visits, residue evaluations, eligibility work, receipt cells,
protocol work, selected periodic families and write-overlap exclusions. The
reported work statistic includes native scalar qualification but is not a time
or allocation-byte bound. Reaching any optional cap retains the baseline; it is
never interpreted as an empty execution domain.

The targeted core and native demand gates exercise nonzero first-active bounds,
empty and skipped visits, repeated invocations, corrupted constants/keys/waits,
outside-word read/write corruption, split/interleaved cleanup, and sibling
owners. The unchanged two-/three-buffer inputs exercise actual periodic release
and shared-output exclusion. A separate fixture starts at raw IV 2 and first
executes at raw IV 3. Period overflow, excessive period, unknown predicates and
never-active words retain the native baseline.

Initial eight-case observations decrease from 62 to 37 for two buffers, 76 to 42
for three buffers and 66 to 22 for four-use, counting acquisitions that require
a later producer prefix than InsertSync. These are observer results, not device
timings or complete quality acceptance. Static counts become 8 SET/10 WAIT for
two buffers and 12 SET/15 WAIT for three; executed synchronization remains no
worse than InsertSync in the recorded scenarios. Added guards and cleanup sites
are reported separately. Softmax, QK, Q projection and historical GEMM remain
unqualified on the whole quality gate, and the default planner is unchanged.

### Final periodic-candidate validation

The targeted four-target native rebuild passes. The final build passes all six
selected gates: focused (142.57 s), composition core (5.04 s), composition
(3.06 s), demands (19.94 s), structured core (44.18 s), and structured
(81.63 s). ASan/UBSan passes **2,216,627 assertions**, including exact discovery
reservation boundaries and maximal-integer refusal. The one selected public
option lit test passes; 1,879 other lit tests were excluded. Its Python launcher
requires the build-matching Python 3.12 environment on PATH; the initial system
Python invocation failed to import that extension and was corrected without a
source change.

All eight unchanged inputs pass three paired complete-compiler timing rounds
after one warm-up, with untimed C++ emission. Median candidate/InsertSync ratios
are **0.986–1.054**, passing the <=2x compilation gate. These serial timings ran
without another resource-intensive local worker. They do not measure device
runtime, and the whole synchronization-quality gate remains **false**.

| Input | Candidate SET / WAIT / named barrier | Later-prefix observations | Executed sync no worse in all recorded scenarios |
| --- | ---: | ---: | --- |
| One buffer | 4 / 6 / 0 | 0 | Yes |
| Two buffers | 8 / 10 / 0 | 37 | Yes |
| Three buffers | 12 / 15 / 0 | 42 | Yes |
| Four-use | 20 / 22 / 6 | 22 | No |
| Online softmax | 17 / 17 / 21 | 42 | No |
| QK matmul | 18 / 18 / 4 | 32 | Yes |
| Q projection | 18 / 18 / 6 | 410 | Yes |
| Historical GEMM | 62 / 62 / 21 | 45 | No |

Each output also has one terminal ALL drain. Later-prefix counts compare
acquired producer boundaries with existing InsertSync, not a hardware timing
model. Scalar guard execution is recorded separately by the demand runner.

A rootless replay from the frozen original/prepared content-addressed snapshots
matches **726/726** recorded outcomes (363 inputs in each of conservative and
demand modes), including all refusal reasons and mechanism counts. The frozen
corpus remains **45/150 PTOAS** and **152/213 generated PyPTO/pypto-lib**; the
new focused periodic fixture does not enter either denominator. The corpus has
336 unique original-byte hashes, not 363 distinct production kernels.

The tested native driver SHA-256 is
`e166d5a8ac833e70527a13a47c2949e3797870af08f4494fcdf512208caf8442`.
Detailed artifacts remain outside Git under the existing build's
`test-results/oahs-periodic-frozen-replay`, `test-results/oahs-periodic-compiler`,
and `test-results/oahs-demands`. The separate frozen-corpus milestone supplies
the portable manifest, exact public command templates, and CAS replay runner.
This remains an opt-in implementation checkpoint, not replacement acceptance.

## Shared incoming first-demand summaries

Incoming completion no longer requires a direct operation as the first loop
consumer, or a lexical source operation immediately before that loop. The same
demand constructor can use a bounded first-site summary at an original Choice
cut in a qualified counted loop. No kernel name, opcode recipe, branch-predicate
solver, or periodic occurrence formula participates in this decision.

For each observer lane, an operation contributes its own first site. A Sequence
continues collecting possible first sites only while its preceding children
may issue no operation on that lane. A Choice unions the first sites of its
arms, preserving the may-empty distinction. The optional summary has at most
eight first-site alternatives per node/lane; an unavailable nested-loop summary
or overflow retains ordinary construction. The complete owner MAY summary is
still computed independently and is never replaced by this first-site summary.

The constructor forms the first-effect union once per observer. An incoming
episode must be incoming-only over that complete first-site witness: if the
source accesses any witness cell anywhere inside the owner, the entry proposal
is declined. In particular, it does not spend a key on a stable subset of a
consumer that also needs a fresh in-owner generation from that source. The
ordinary constructor still handles that mixed demand. This all-or-nothing
policy prevents entry-key pressure from degrading otherwise useful local
handoffs.

A lexical producer keeps the earliest qualified publication after its last
relevant source access and after the original trip predicate becomes available.
Without a lexical producer, the constructor may use the actual incoming
source-prefix receipt at that available entry cut. The anchor is not evidence
that payload was produced: absent pending incoming demand produces no episode.
The source prefix may include work from a previous enclosing-loop visit.

The existing globally allocated episode realizes the result:

```text
Nonempty owner: publish the actual incoming source prefix.
First iteration: acquire it before the original common Choice.
Nonempty owner exit: publish and consume the reverse acknowledgment.
```

An empty Choice arm still executes the common first acquisition. It consumes a
real matching token; it does not invent a payload access. This can add four
executed event commands to an otherwise unneeded incoming episode, and that
cost is reported separately from correctness. Zero-trip owners execute no
episode commands. Other conservative mechanisms around the owner may still
execute on a zero-trip path.

The receipt retains the whole-owner MAY effects, later source suffixes and
independent GM visibility history. Reconstruction rederives the first-site
witnesses and source-absence premise from the immutable original program,
classifies actual guarded commands using original bounds/IVs, and verifies
their combined event word. A first acquisition moved inside just one arm or
past the common Choice is rejected atomically.

The new tests also model the recurrence of their initial preloads explicitly:
the complete portable test program is enclosed in a For with two iterations,
and that execution is repeated without resetting the independent oracle or
granting an implicit drain. Native repeated-function tests separately include
the real terminal retirement emitted by the adapter. A later access to panel B
cannot broaden the entry publication selected for the first access to panel A;
an empty arm followed by a later same-cell consumer retains the acquired
incoming receipt. Nine possible first sites decline only optional precision.

### Optional-analysis accounting

A cheap structural prescan avoids allocating the first-site population when
there is no contracted For with a direct-body Choice. Straight-line incoming
operation demands continue to use the same entry constructor without that
population. First-effect unions are computed once per observer and reused
across source candidates.

Scan work and retained storage have separate fixed allowances. Rejected scans
still consume scan work: source-overlap and empty-witness checks are real work,
not free speculation. They do not reserve retained witness storage. Storage is
reserved before materializing viable witnesses and demand/map bookkeeping;
one `(acquisition, observer)` witness is shared across accepted sources.
Alternative overflow or either allowance limit leaves ordinary construction
available. Structural traversal order is deterministic; a finite scan allowance
can still limit which later optional proposals are attempted. No optimality or
order-independence guarantee is claimed.

Trace fields distinguish `entry_summary_slots`, `entry_summary_scans`,
`entry_storage_units`, `entry_candidate_pairs`, `entry_witness_cells`,
`entry_witnesses`, `entry_source_overlap_rejections`, and
`entry_summary_skipped`. These describe the selected demand analysis, not an
aggregate wall-clock or byte-allocation bound across every refinement and fresh
reconstruction attempt.
`entry_summary_skipped` is also set when no eligible Choice entry exists; it is
not exclusively an exhaustion indicator. Both fixed allowances are 1,048,576
representation units, not a time limit or a claim of exact allocator bytes.

### Incoming-choice qualification results

The final targeted native build and all six selected gates pass:
`oahs_focused`, `oahs_composition_core`, `oahs_composition`, `oahs_demands`,
`oahs_structured_core`, and `oahs_structured` (305.89 seconds total). The
Clang C++17 ASan/UBSan core run passes **2,237,669 assertions**, including the
storage-cap refusal case. The single selected
`insert_sync_structured_constructor.pto` lit test passes; the full lit and
system suites were not run.

The rootless two-arm corpus replay again matches **726/726** frozen outcomes,
including mechanism counts and refusal reasons. The cohorts remain **45/150
PTOAS**, **7/35 PyPTO**, and **145/178 PyPTO-lib** snapshots. The new focused
fixture is not added to those denominators. The replay executable SHA-256 is
`832ab7447800772773aadcc730028c48ab39deb5962168f7a676428252da7c2f`;
the separate reconstruction-driver SHA-256 is
`de67cd4c99cf5e70a81591c3a167537fc3a5b4b9c6b7930f8a15f8e1af31de00`.

The unchanged eight-case compiler campaign ran alone, using three paired
samples after one warmup per arm. Every PTO compilation and C++ emission
succeeded, and every individual paired ratio is below 2x (maximum 1.067728).

| Input | Median demand / existing compilation time |
| --- | ---: |
| One buffer | 0.999107 |
| Two buffers | 1.038608 |
| Three buffers | 1.040996 |
| Four-use | 1.000765 |
| Online softmax | 1.045326 |
| QK matmul | 1.025446 |
| Q projection | 1.024067 |
| Historical GEMM | 0.987989 |

The campaign records base `5153e3ca45fb5a1289a9f5dba53e64c1ceb4f42a`, its
task-owned dirty listing, tracked-diff hash, unchanged input hashes and complete
commands. The tested compiler-library SHA-256 is
`46b07f6efa6a7f81f65951c6f77c3f56a7fe9ac8882d63c130853d9f9f70ebfc`.
Artifacts are under the existing build's `test-results/oahs-choice-compiler`,
`test-results/oahs-choice-frozen-replay`, and `test-results/oahs-demands`.
The base-plus-diff provenance is not relabeled as a clean committed run.

For historical GEMM, one shared FIX-to-M incoming episode replaces four
branch-local families. Static SET/WAIT counts fall **62/62 to 56/56**, while
the 21 named barriers and one terminal drain are unchanged. Executed SET/WAIT
counts per kind change from 54 to 52 for two panels, 77 to 73 for three panels,
and 108 to 104 for the distributed-tile scenario. One-panel counts stay at 27
per kind. These reductions are relative to the preceding demand candidate,
not a claim to beat existing InsertSync's 44/44 static sites.

There is a measured placement cost: two extraction operations in the
distributed-tile scenario acquire a later FIX prefix through transitive
completion. The total later-prefix observations relative to existing InsertSync
are 47 rather than the preceding candidate's 45. Other seven-case mechanism
counts and boundary observations remain unchanged, including softmax's 42 and
QK's 32. Empty-arm episodes have their executed commands and scalar guards
recorded separately; the portable controlled case shows four extra event
commands compared with disabling that optional entry episode.

The first-site population is correctly absent on branch-free softmax and QK
(`entry_summary_slots=0`, `entry_summary_skipped=1`). Historical GEMM reports
8,680 first-site slots, 1,385 scan units, 12,133 reserved storage units, three
shared witnesses containing 36 effect cells, and four source-overlap declines.
These are bounded analysis measurements, not hardware costs.

All three reviewers accepted this as an opt-in implementation milestone after
the above conditions were met. **`quality_qualified=false` remains explicit**:
the common-cut placement is not universally as late as the first consumer,
the whole regression quality gate remains open, defaults are unchanged, and
there is no device-correctness or device-performance qualification. A later
bounded path-local acquisition provider must prove participation from actual
control flow before moving waits into mutually exclusive arms.

## Late shared incoming acquisition

The next optional refinement keeps the incoming publication and reverse
acknowledgment unchanged, but places its First acquisition immediately before
each mutually exclusive first observer operation. Independent work preceding
that operation need not inherit the incoming completion. This uses the same
original physical witnesses and completion engine, not a kernel-specific
protocol or a new hardware assumption.

A valid bounded first-site summary with no observer-free path is only a
proposal filter. Such a summary can still name two sites on one execution:
an optional early consumer followed by an unconditional later consumer is a
simple counterexample. Fresh reconstruction therefore checks the actual wait
population against the original sites and propagates unconsumed/consumed state
through Sequence and Choice. Every path must consume exactly once, before its
first observer operation. Missing, duplicate, moved, wrong-key and wrong-owner
waits are rejected.

The state is a set of possibilities encoded by two bits: unconsumed is 1,
consumed is 2, and a mixed path population is 3. Choice uses set union, and
the complete domain must finish with exactly 2, not merely a nonzero bit.
An observer operation also requires exactly that state. An observer-free
nested Choice may legitimately finish unconsumed before a later common first
consumer; it is not required to consume prematurely at every nested join.

The event proof also requires every arm to have the same projected **combined
entry-command word**, preserving the order of different entry families. A
two-copy rearm proof of that word then covers successive executions taking
different arms, not only repetitions of one arm. This is a deliberately bounded
sufficient condition: differing words retain the common placement rather than
triggering path enumeration. Ordinary and deferred protocols retain their own
checked, disjoint key populations.

The refinement is one transaction over an already verified common-cut plan.
Fresh verification applies completion at the actual branch-local waits and
joins the resulting states conservatively. If removing the earlier common
credit exposes a missing handoff elsewhere, the candidate is rejected and the
exact common plan is retained. No successful subset of a failed family is kept.
Empty observer paths, uncertain recurrence, summary overflow and optional work
limits likewise retain the existing construction.

Candidate discovery/copying has an aggregate allowance of 4,194,304 represented
work units. Actual late-entry verification has a separate 1,048,576-unit
allowance, reserved before its subtree scans and conservative word-copy cost.
Common-entry and no-entry plans do not enter that new verification path.
The cardinality population is capped before growing beyond eight actual sites.
These are implementation work bounds, not wall-clock or exact byte estimates;
the conservative reservation may decline a useful large candidate.

Native guards use the original counted loop's exact IV/lower-bound comparison.
Only original Sequence/Choice ancestry is allowed between a late consumer and
that owner. Repeated same-key First guards are bindings to one owner, not an
assertion of exclusivity; actual control-flow reconstruction supplies that proof.

One common WAIT can become up to eight static WAIT/guard sites, while exactly
one of them executes in a nonempty first iteration. Static code size, executed
event commands, scalar comparisons/branches and completion boundaries must be
reported separately. The test-driver `demands:without-late-entry` arm preserves
the common placement; `demands:reject-late-entry` exercises transactional
refusal. Neither changes the public pass options or planner defaults.

### Late-acquisition qualification

All three reviewers accepted this opt-in milestone. The final targeted native
build, six selected OAHS gates (310.84 seconds), and the single selected
`insert_sync_structured_constructor.pto` lit test pass. The final Clang C++17
ASan/UBSan run passes **2,257,565 assertions**. Tests cover exclusive sites,
each missing-arm acquisition, duplicate/oversized populations, wrong owners
and keys, mixed common/branch placement, reversed multi-family arm words,
observer-free inner choices, overlapping possible-first sites, changing loop
visits, repeated whole invocations, and exact budget/rejection fallback.

The final two-arm frozen replay matches **726/726** outcomes, counts and
refusal reasons, with no cohort change: 45/150 PTOAS, 7/35 PyPTO and 145/178
PyPTO-lib snapshots. Its executable SHA-256 is
`845992d32a0b063c3e0a1e6896827c87876023ad1d14b3142dfaea0344b3e362`.
The final native reconstruction-driver SHA-256 is
`d45cba83a901306f8c81848a441538c2d46f11eca4041cc91f2a1075ea5e1e15`.

The isolated compiler campaign uses the unchanged eight inputs, three paired
samples and one warmup per arm. All PTO compilations and C++ emissions pass.

| Input | Median demand / existing compilation time |
| --- | ---: |
| One buffer | 1.007651 |
| Two buffers | 1.030172 |
| Three buffers | 1.008680 |
| Four-use | 1.023954 |
| Online softmax | 1.048058 |
| QK matmul | 1.014139 |
| Q projection | 1.020299 |
| Historical GEMM | 0.995506 |

Every individual paired ratio is below 2x; the maximum is 1.052777. The
compiler-library SHA-256 is
`4e39f3b715efbd08aae74dcbdd67d0ca9411c7bea03087e030a06a989941bee6`.
The campaign records base `4b1b8f521a9459a41e7573dd50d0b0dc46b70131`, its
task-owned dirty listing, tracked-diff hash, commands and unchanged input
hashes. This is base-plus-diff evidence, not a retroactive clean-revision claim.
Artifacts are under the existing build's `test-results/oahs-late-entry-compiler`,
`test-results/oahs-late-entry-final-frozen-replay` and `test-results/oahs-demands`.

GEMM selects one family with four exclusive sites. Relative to the common-cut
candidate, static SET/WAIT counts change **56/56 to 56/59**, with the same 21
named barriers and one terminal drain. Executed event and scalar-operation
counts are identical on all five measured scenarios. Normalized PTO grows from
27,159 to 27,661 bytes. Only two completion observations change: extraction
sites 57 and 58 in the distributed-tile scenario no longer inherit FIX prefix
54. All other observed completion boundaries are unchanged. The later-prefix
count against existing InsertSync consequently returns from 47 to 45.

The dedicated native fixture similarly changes one static first guard into
three, with identical executed synchronization and scalar counts on its eight
scenarios. Forced candidate rejection returns byte-identical normalized PTO
to the disabled-late arm. The other seven benchmark cases keep their mechanism
counts and boundary observations.

The replacement quality gate remains **`quality_qualified=false`**. This fixes
the measured common-wait regression; it does not establish a general runtime
speedup or eliminate the remaining over-ordering. Defaults are unchanged, the
full lit/system suites were not run, and there is no device qualification.

## Ordinary Choice incoming demands

The next demand-placement refinement addresses incoming physical history at
an ordinary conditional, without requiring a periodic or counted-entry
contract. Its motivating structure is a producer of A, a later independent
producer of B on the same lane, and an unknown Choice whose arms first use A
and subsequently use B. Publishing only inside either arm captures B before
the first A consumer needs it.

The proposed common request uses the existing `CompletionDemand`: a source
publication in the parent Sequence and an acquisition before its Choice.
The witness comes only from bounded first observer sites, not from the whole
conditional's MAY footprint. All arms must justify the same incoming frontier;
unsupported or empty paths retain ordinary construction. Publication and
acquisition remain in one execution domain, so no publication outside a loop
is reused as though it ran on each body visit.

The physical receipt is captured at the actual publication cut. Acquiring it
does not remove B's later pending history. Existing child requests continue
to satisfy those residual demands. Concrete numbering, acknowledgments and
fresh verification operate on the complete composed event population; a
conditional does not get an independently allocated private protocol.

This is an optional transaction over the existing verified candidate. Budget
or qualification failure, allocation failure and rejected fresh verification
must preserve the exact pre-refinement command population. The testing arms
`demands:without-choice-demands` and `demands:reject-choice-demands` expose
that baseline and fault-injected rollback without adding public pass options.

Earlier publication can be worth an additional directed acquisition, but
that is a placement/cost tradeoff, not an unconditional performance gain.
Qualification must report static sites, executed commands, scalar guards and
completion-prefix observations separately against both the previous demand
candidate and existing InsertSync. A first-A improvement does not establish
that later B demands or all Q-projection observations have been improved.
Defaults, target contracts and device-qualification requirements are unchanged.

The implementation additionally requires an actual baseline handoff at every
first observer site, published inside the Choice. Together with an intervening
source operation after the new parent cut, this supplies a structural earlier-
prefix certificate; a merely plausible request is not sufficient. Unrelated
regions must retain their mechanisms and placement (physical recoloring is
allowed). Guarded entry/deferred mechanisms must remain exactly unchanged.

The cost comparison folds **signed command deltas**: Sequence adds deltas and
Choice takes the maximum arm delta. It does not subtract two maximum costs,
which could hide a regression on a formerly cheaper arm. Changed nested-loop
interiors are refused instead of treating every loop as one iteration. The
aggregate increase is at most two event-command units per affected owner,
independent of how many requests are promoted there. Existing target and fresh
physical/protocol verification remain mandatory. In particular, the standalone
A-then-B case can retain its old plan when separate acknowledgments exceed this
limit; broadening the first witness to B is not the remedy.

One optional allowance of 1,048,576 represented-work units covers the complete
attempt. The wrapper reserves node/cell/receipt and command-copy work before
rerunning construction. All discovery passes share one `charged/exhausted`
counter through initial construction, refinement, allocation replay, ring and
deferred candidates. No new `DemandAnalysis` resets it. The extra final checker,
baseline-benefit, ancestry and cost scans receive a separate charge within that
same allowance. Exhaustion anywhere returns the exact disabled plan. This is a
conservative accounting model, not a bound on allocator bytes or wall time.

Trace fields distinguish total work, the initial reservation, aggregate
analysis work, the number of analysis passes, and the first pass exhausting
the allowance. Tests check an exactly sufficient budget, one unit less, and
exhaustion after preflight during a later constructor/refinement pass.

### Ordinary-Choice qualification

All three reviewers accepted the opt-in milestone and its final validation
conditions. The targeted incremental native build, six selected OAHS gates
(311.78 seconds), and the single selected
`insert_sync_structured_constructor.pto` lit test pass. The final Clang C++17
ASan/UBSan run passes **2,265,550 assertions**. The native core tests explicitly
reach budget exhaustion in a third-or-later analysis pass and retain the exact
baseline, alongside exact-budget and one-unit-short checks.

The frozen corpus replay completes all **726** baseline/current comparisons.
**681 match exactly; 45 differ only by having two fewer barriers**, with
unchanged SET/WAIT counts, outcomes and first refusals. Those 45 are two PTOAS
snapshots (`hc_head_linear` and `hc_pre_linear`) and 43 PyPTO-lib snapshots.
There are no count increases. Coverage remains 45/150 PTOAS, 7/35 PyPTO and
145/178 PyPTO-lib snapshots. The original frozen manifest is not rewritten.
This is compatibility evidence for the recorded raw/prepared inputs, not
upstream Python regeneration or device qualification.

The final opt-driver SHA-256 is
`da106b30466cc3a1aa20f90cc3fe05a97d87c78b56a9dcca6b91fbf9e20c3c6e`;
the native reconstruction-driver SHA-256 is
`811f98e85ada8279853f92b83d4d47b5be79561667171e43215f1caadd0c7f74`.
The isolated eight-input compiler campaign uses three paired samples and one
warmup per arm. All PTO compilations and all 16 C++ emissions pass.

| Input | Median demand / existing compilation time |
| --- | ---: |
| One buffer | 1.022905 |
| Two buffers | 1.041320 |
| Three buffers | 1.018276 |
| Four-use | 1.013016 |
| Online softmax | 1.035142 |
| QK matmul | 1.018769 |
| Q projection | 1.019808 |
| Historical GEMM | 0.964240 |

Every individual paired ratio is below 2x; the maximum is 1.053255. The
compiler-library SHA-256 is
`e73bcb8671bb5ba682ebdb23c72cba1252ff1b381c056b4609dfc4c165a8919e`.
The campaign records base `dd9a63dc9f4d07c66aa99ed79dfd4fa638bf3e05`, its
task-owned dirty listing and tracked-diff hash, commands and unchanged input
hashes. It is base-plus-diff evidence, not a clean-revision claim. Artifacts
remain in the existing build's `test-results/oahs-choice-final-compiler`,
`test-results/oahs-choice-final-frozen-replay` and `test-results/oahs-demands`.

Q projection selects one common first-demand family. Relative to the previous
demand candidate, exactly 32 measured boundaries change: each first MTE1
extraction acquires an MTE2 prefix three physical positions earlier. No other
boundary changes, and executed payload/event/scalar counts are identical. Its
later-prefix total against existing InsertSync falls **410 to 378**, while
static SET/WAIT counts remain **18/18**, with six named barriers and one drain.
Executed SET/WAIT counts remain **353/353**, versus existing InsertSync's
389/389 in the recorded scenario. The other seven benchmarks retain their
reported mechanism counts and later-prefix totals; GEMM's normalized PTO is
byte-identical to the previous candidate.

The replacement gate remains **`quality_qualified=false`**. This is a bounded
general first-demand improvement, not a solution to all remaining late
acquisitions or acknowledgment costs. A later increment can reuse a return
path that every child already establishes; it must prove that causality from
actual child protocols rather than simply delete the parent's acknowledgment.

## Composed child-return causality

The next refinement reuses an ordinary parent acquisition's acknowledgment
when every child path already returns that consumption to the publishing
lane. It keeps the existing event-key domains and the physical completion
checker. Disjoint storage is not evidence of safe event reuse.

A seven-by-seven Boolean transfer records a **universal causal guarantee**:
`T[out][in]` means that the outgoing lane carries every causal history that
was already present on the incoming lane at this region's entry. It does not
describe outstanding tokens, payload completion, memory visibility or a
possibly executed path. Sequence composes these transfers; Choice intersects
the arm guarantees. For and While export identity for this refinement: no
parent acknowledgment can rely on a possibly skipped body return.

The actual event-word verifier supplies the transfer. SET snapshots the
source's current history without gating subsequent issue. WAIT joins that
immutable snapshot into the observer. Applying a child transfer joins a copy
of the old lane histories simultaneously, not progressively in lane order.
Commands before a child execute once before its body transfer; they are not
counted again as part of that body.

Only the first invocation's symbolic entry-history transfer may be exported.
The word's second copy still checks consumption-before-rearm. Accumulated
history from that second copy is not a guarantee of the first: for example,
`B <-> C; A <-> B` transports incoming A history into C only on a later visit.
Fresh reconstruction derives these summaries from actual commands, never
from the constructor's selected demands or claimed family coverage.

The refinement must leave readiness/release placements and guarded event
protocols unchanged, remove only optional acknowledgment commands, and retain
the exact previous plan on failed qualification or exhausted optional work.
There is no new target capability, predicate solver or default selection.

The constructor currently proposes these removals for ordinary incoming Choice
families with one unambiguous Choice root. Numbering finishes first. An explicit
logical-to-physical mapping identifies the family's adjacent ACK SET/WAIT;
ambiguous or allocation-fallback pairs are not eligible. At most eight pairs
form one combined trial in a numbered candidate. The child summaries must be
derived after all proposed removals, and the complete actual physical/protocol
check must pass before any removal survives. The allocator's reserved key
population (`protocol_keys`) is deliberately unchanged; this increment does
not feed newly freed IDs back into allocation.

One 4,194,304-unit allowance is shared across all child-return construction
passes, including initial/refinement/replay/ring and later verification calls.
It charges summary storage, command expansion, simultaneous matrix application,
copies and the optional physical recheck before performing them. The trial and
fresh checker may each derive summaries, but both charge this same allowance.
The unchanged flat word checker runs first; already valid words do not pay for
structured rescue. Exhaustion is sticky. If it occurs after earlier removals,
one final reconstruction with the feature disabled restores the exact disabled
plan instead of exposing a partially budgeted selection.

`child_return_work` and `child_return_checks` aggregate the entire construction
attempt; `child_return_budget_check` identifies the first exhausting verifier
(zero if exhaustion occurred elsewhere), and `child_return_budget_exhausted`
records whole-attempt rollback. Candidate/rejection counters count attempted
populations across passes. `child_return_acks_removed` counts only removals in
the selected plan. Fresh reconstruction of emitted native IR has its own bounded
checker allowance; it is not part of the constructor's reported allowance.
These are represented-work bounds, not exact allocator-byte or wall-time bounds.

A proposed removal currently must exercise structured rescue even if the flat
checker would already accept it. This conservative qualification can miss an
elision, but does not weaken event reuse. Neither this limitation nor the extra
summary verification is hidden as an admission failure: the disabled plan
remains available for the same semantic input.

### Child-return qualification

All three reviewers accepted the opt-in milestone after the shared-verifier
budget repair. The targeted incremental build passes. The six selected OAHS
gates pass in **320.86 seconds**, including actual native child-return deletion
and wrong-key mutations with atomic rejection. The single selected
`insert_sync_structured_constructor.pto` lit test also passes. Clang C++17
ASan/UBSan passes **2,277,856 assertions**. Tests include 216 independently
checked combinations of child words and parent directions, first-copy and
double-entry counterexamples, empty arms, zero-trip loops, payload after a
return, and exact/one-unit-short budgets. The latter exhausts a later verifier
and restores the exact disabled plan.

The frozen replay completes **726 comparisons: 680 exact, 46 count changes**.
Outcomes, return codes and first refusals are unchanged; admissions remain
45/150 PTOAS, 7/35 PyPTO and 145/178 PyPTO-lib snapshots. Forty-five changes
are the prior two-barrier reductions. The additional change is
`pypto-lib/843b72f276ffb59a5b0b_000`: **12/12/3 becomes 13/13/3**
(SET/WAIT/barriers). It is a generated `gemm_tile` snapshot, not the archived
historical handwritten-GEMM benchmark. Its prepared SHA-256 is
`d64b7d7caeb60fcc0bad14e1a24b2c4f66a27a4122a30f6a404f8452f43d30a5`.
The frozen reference manifest is unchanged.

For that sole count increase, native disabled/current outputs both reconstruct
successfully. Independent boundary replay with arguments `[a,b,c,0,0]` and
`[a,b,c,64,128]` observes **four earlier MTE2-to-MTE1 acquisitions and three
earlier MTE1-to-MTE2 releases**, no later-prefix differences, and identical
payload hashes. The ordinary A-then-B branch fixture similarly preserves B's
later readiness while moving A's acquisition earlier, at **5/5 instead of 4/4
executed SET/WAIT per iteration**, with unchanged scalar work and no additional
drain. These are explicit placement/count tradeoffs, not a runtime speedup.

The eight fixed regression inputs retain their previous static counts and
boundary observations. In particular, Q projection remains 18/18 SET/WAIT
with six named barriers; historical GEMM remains 56/59 with 21 named barriers.
Neither selects a child-return removal in this campaign. Every output retains
one terminal drain. The replacement gate remains **`quality_qualified=false`**.

The isolated whole-compiler campaign uses three paired samples and one warmup.
All compilations and all 16 C++ emissions pass. Median demand/existing ratios:

| Input | Ratio |
| --- | ---: |
| One buffer | 1.033906 |
| Two buffers | 0.997168 |
| Three buffers | 1.035686 |
| Four-use | 1.003166 |
| Online softmax | 1.072448 |
| QK matmul | 1.019610 |
| Q projection | 1.022315 |
| Historical GEMM | 0.974099 |

The maximum individual paired ratio is **1.075521**, below the unchanged 2x
gate. Binary SHA-256 values are:

- opt driver: `71f0f5384f996ef12ba7533e194bfc578ee4f09804503ed1a200415b3d9d6cc9`
- reconstruction driver: `9921e1d75785ef17227949c06848a64bc7f9d58b0226e30cc2e2fd74f69d706d`
- compiler library: `f946ec147273c860e287c5d4033c5a97a659c623dbd57877292e38d66f4aad4d`

The campaign records base `c24475828ce56229b07bae0f4ab037238e5e1dbc`, the
task-owned dirty listing, tracked-diff hash, exact commands and input hashes.
This is base-plus-diff evidence, not a retroactive clean-revision claim.
Artifacts are under the existing build's `test-results/oahs-child-final-compiler`,
`test-results/oahs-child-final-frozen-replay`, `test-results/oahs-demands` and
`test-results/oahs-child-corpus-tradeoff`. The last directory contains both
native outputs, replay scenarios and boundary reports for frozen record 193.
The full lit/system suites and device qualification remain separate and were
not run. Defaults and target contracts are unchanged.

## Shared alternative acquisitions

The next bounded refinement separates a common producer publication from its
alternative branch-local acquisitions. It targets the general situation
`produce A; produce B; unrelated producer work; Choice(consume A; consume B)`.
An incoming A receipt need not force the residual B handoff to publish inside
each branch after the unrelated work has already been issued.

The selected protocol has one parent SET on a globally exclusive forward key.
Every supported branch path has one WAIT on that **same physical key**, then
an adjacent dedicated reverse SET/WAIT acknowledgment. Only the branch SETs
are coalesced; the acquisitions and return commands keep their original sites.
The constructor must preserve all payload and guarded mechanisms and must not
increase executed synchronization or scalar participation work.

Fresh reconstruction qualifies the actual endpoint population independently
of the constructor's demand/family records. It proves exactly one acquisition
on every path, rejects participating recurrence or bypasses, and checks global
key exclusivity including expanded canonical packets. Each bounded ordered
pair of alternative four-command words is checked for consumption-before-rearm.
The direct return gives both causal chains:

```text
WAIT(F, arm i) -> SET(Ri) -> WAIT(Ri) -> next SET(F)
WAIT(Ri) -> later SET(F) -> WAIT(F, arm i) -> later SET(Ri)
```

Skipping other arms does not reset a key. These chains, not lexical encounters
with WAIT, justify the next publication. The virtual copies used by the proof
must not become duplicate physical SETs in emitted IR.

Certified keys are excluded from the ordinary closed-word proof, which still
checks every remaining key. A separately established all-arm causal transfer
may export observer **entry** history to the source at Choice exit, as in the
child-return milestone. It does not export end-of-arm history or payload
completion. The fresh physical-prefix checker retains every actual command;
an A publication cannot cover a newer A generation or later unrelated B work.

This is an optional protocol refinement within the general composition engine,
not a new control-flow admission rule or a GEMM recognizer. Unsupported
participation, shared colors, or exhausted bounded work must retain the exact
pre-refinement plan. Default selection and target contracts are unchanged.

The optional final transaction has a 4,194,304-unit represented-work allowance,
including discovery, command/cell copies, alternative certificates and any
child-return rescue needed by its fresh checker. Earlier construction checks
explicitly disable alternative certification. The already verified baseline's
child-return construction retains its own separate allowance: this is not one
global budget over the entire top-level invocation. Family and alternative
populations stop before a ninth entry is inserted. The actual certificate uses
structural intervals for membership tests; constructor ancestor walks remain
precharged and bounded by the optional allowance.

The post-numbering constructor requires an existing dedicated branch ACK and
does not synthesize new continuation handoffs. This distinction matters on Q
projection: its branch B publication also supplied later C/D readiness. Moving
that SET before the C/D loads would invalidate those later consumers. The fresh
physical check therefore retains the exact baseline on that input. A follow-up
must select the earlier B demand **before** forward construction, leaving C/D
pending so the same constructor discovers their necessary later handoff. It
must not repair that omission with a separate post-numbering insertion planner.

### Alternative-acquisition qualification

All three reviewers accepted this opt-in milestone. The final incremental
targets compile, all six selected OAHS gates pass in **325.60 seconds**, and
the selected `insert_sync_structured_constructor.pto` lit test passes. Native
core checks pass **2,270,051 assertions**; strict C++17 Clang ASan/UBSan passes
**2,284,195**. Counts from the two builds are not added together.

The native three-load fixture publishes B before unrelated C and keeps both
branch acquisitions at their B consumers. Static SET/WAIT sites change from
**5/5 to 4/5**; either arm still executes **3/3**, with unchanged scalar work
and one terminal drain. Independent replay observes B's producer prefix move
from payload 2 to payload 1, with identical payload. The fixture SHA-256 is
`7c5c68aec62cf063e45979e5d05f6181bee3d3bdbd2d6e7a55293ecf7601d29e`.
Adding a later C consumer exercises exact fallback: the transaction is rejected
and the output matches the disabled plan byte-for-byte. Native missing,
duplicate and wrong-key arm waits and deleted returns reject atomically.
Core tests also cover alternating whole invocations, interleaved foreign
families, stale generations, zero-trip bypasses and exact/one-unit-short budgets.

The final frozen replay has **726 comparisons: 680 exact, 46 prior count
differences**. There are no new status, return-code, refusal or mechanism-count
changes relative to the child-return milestone. Admissions remain 45/150
PTOAS, 7/35 PyPTO and 145/178 PyPTO-lib snapshots. The frozen reference and
denominators are unchanged; this remains raw/prepared compiler compatibility.

The eight regression inputs keep their previous command counts and completion
boundaries. Q projection and historical GEMM are not improved by this removal-
only refinement; the whole replacement gate remains `quality_qualified=false`.
The isolated compiler campaign uses one warmup and three paired samples per
case. All samples and all 16 C++ emissions pass. Median demand/existing ratios:

| Input | Ratio |
| --- | ---: |
| One buffer | 1.013140 |
| Two buffers | 0.998218 |
| Three buffers | 1.029744 |
| Four-use | 1.022679 |
| Online softmax | 1.070064 |
| QK matmul | 1.022399 |
| Q projection | 1.038154 |
| Historical GEMM | 1.000242 |

The maximum individual paired ratio is **1.070130**, below 2x. Final SHA-256:

- opt driver: `efd72968b8cefd84118ff8196f3cc6962af5cdd1333e6e0754caab87b072279c`
- reconstruction driver: `15617df7b8122eb040d06a6b682b2a5bad36d8c19d48195bc5b1cdc3a45c1d11`
- compiler library: `cf2b39748d1a563c8a0e7d6b9e8b27143a549267cd7d285a8cbc9fb7154b1b5b`

Provenance records base `ebc201672c5392566470078b5d9c814a5b1a2684`, the
task-owned dirty listing, tracked-diff hash, exact commands and input hashes.
Artifacts reside in the existing build under `test-results/oahs-demands`,
`test-results/oahs-alternative-final-compiler` and
`test-results/oahs-alternative-final-frozen-replay`. This is base-plus-diff
evidence, not a clean-revision or device qualification claim. Full lit/system
suites were not run. No planner default or hardware contract changed.

## Preconstruction alternative demands

This follow-up replaces the preceding milestone's post-numbering publication
edits. Alternative consumers become requests to the existing forward demand
constructor, before event allocation. The fresh alternative-protocol checker
is retained; the constructor no longer tries to repair a numbered plan by
moving SETs and rewriting WAIT keys.

The transaction has three explicit states:

1. Construct and verify the pre-Choice demand plan, retaining the existing
   entry, recurring and other demand refinements.
2. Establish the existing common-Choice plan. This is the exact fallback for
   every failure of the new alternative refinement.
3. Construct one combined common/alternative candidate from immutable demand
   proposals, then globally allocate and freshly verify it.

There is no recursive candidate construction, subset retry, or independent
child allocator. A failed alternative proposal must not discard accepted
common-Choice precision. Common numeric keys may change in the combined plan,
but their semantic cuts, directions, participation and required completion
must remain valid. Existing fixed reservations remain fixed.

An alternative family records the original scope, publication, consumer,
source, observer and complete physical-cell witnesses for every arm. A
structural all-path check requires exactly one matching consumer. Unsupported
recurrence, missing arms, intervening writes to a witnessed generation and
exceeded finite populations decline the proposal. No branch predicate is
solved and no iteration relation is constructed.

At the earlier parent cut, the constructor records a producer-prefix receipt.
That receipt continues to accumulate subsequently issued work as **remaining**
work. Each branch acquisition therefore completes the earlier prefix, not a
frozen snapshot that accidentally clears a newer generation. In particular:

```text
produce B -> publish B prefix -> produce C
    -> Choice(B consumer in each arm) -> C consumer
```

The B acquisitions do not complete C. The same ordinary forward constructor
must discover and realize the C requirement. If doing so exceeds the allowed
cost or available event resources, the entire candidate is discarded.

One shared logical forward key and dedicated arm returns are checked before
global numbering. Their physical colors are exclusive by directed pipeline
domain, including against ordinary, canonical, entry and deferred protocols.
The final checker reconstructs all actual endpoints and rechecks participation,
consumption-before-rearm and physical completion. A causal entry-history
certificate never grants branch-payload completion or resets an event key.

Acceptance compares actual commands against the common-only fallback:
Sequence sums command deltas, Choice takes the worst arm, and changed nested
recurrence is refused rather than assigned a guessed trip count. The maximum
additional cost is one SET/WAIT pair per affected owner execution, not per
physical cell or per proposed family. An earlier producer boundary is required
separately. A candidate that merely adds synchronization is not a precision
improvement.

Common and alternative selection have separate **1,048,576-unit** represented-
work allowances. Alternative discovery, reconstruction and fresh verification
share one monotonically consumed allowance, with reservations deducted before
the final check. Child-return work uses the remaining parent allowance in an
isolated trial: exhaustion is charged but cannot invalidate the accepted
fallback. A missing alternative proposal returns immediately without rerunning
the Common plan. The existing eight-family/eight-alternative limits remain.

Native qualification includes a selected case where a later D production's
ordinary handoff also supplies still-pending C. Both arms preserve B's earlier
publication; executed mechanisms and scalar work are unchanged and one static
SET site is removed. The C-only case explicitly constructs the residual demand
but declines because its extra protocol exceeds the per-owner cost limit.
Tests distinguish these paths with `alternative_choice_continuation_demands`
and `alternative_choice_cost_rejections`, rather than counting fallback success
as evidence that a refinement was selected.

### Local qualification

Validated against `59f1e70ee867d19a5e5c7e628945202267654b33` plus this six-file
milestone, using the existing LLVM/MLIR 19.1.7 build. This is base-plus-diff
evidence, not a device or clean-revision certification.

- The four affected native targets rebuilt successfully. Strict C++17
  ASan/UBSan passed **2,443,127 assertions**.
- All six selected gates passed: `oahs_composition_core`, `oahs_composition`,
  `oahs_demands`, `oahs_focused`, `oahs_structured_core`, and `oahs_structured`.
  After the final charge-on-exhaustion accounting repair, the directly affected
  core and demand gates were rerun and passed (37.51 seconds together).
  The selected `insert_sync_structured_constructor.pto` lit test and Python
  syntax check also passed. Full lit/system suites were not run.
- The final current-arm frozen replay retained **197/363** admissions:
  PTOAS **45/150**, PyPTO **7/35**, and pypto-lib **145/178**. There were no
  admission or first-refusal changes. Its actual results exactly matched the
  preceding two-arm replay in this campaign. The 46 differences from the older
  frozen baseline are already-existing mechanism-count changes, not new
  coverage gains from this milestone. This remains raw/prepared IR compiler
  compatibility, not regeneration from the upstream Python sources.
- All eight paired whole-compiler benchmarks, including historical GEMM,
  passed the **2x** compilation-time gate: one warmup, three paired samples,
  90-second per-run timeout, diagnostics disabled. Median demand/InsertSync
  ratios ranged from **0.984 to 1.059**; all 16 synchronized PTO outputs also
  emitted C++. This measures compiler time, not device execution.
- The eight regression inputs' selected mechanism counts and later-prefix
  observations are unchanged. In particular, historical GEMM still has
  **56 SET / 59 WAIT / 21 named barriers**, versus InsertSync's
  **44 / 44 / 21** (one terminal drain each). `quality_qualified` remains false;
  neither the default nor the hardware contract changed.

The architect, algorithms/performance, and correctness/design reviewers each
accepted this milestone. The positive C/D continuation regression demonstrates
earlier B completion with unchanged executed commands and one fewer static SET;
it is not evidence that the remaining buffering/GEMM quality work is complete.

Artifacts remain outside the source worktree in the existing build:
`test-results/oahs-demands`, `test-results/oahs-universal-final-frozen-replay`,
`test-results/oahs-universal-accounted-frozen-replay`, and
`test-results/oahs-universal-final-compiler`. The frozen manifest is unchanged
(`95beab25427b1dd3a2ff1b59cd881a3f4181e0f83c8a8b44daf35555bf46c3b6`).
Final binary SHA-256 identities are:

```text
pto-test-opt             cbc34056a67cc892d58f7bfe3ccd32c87a311d31d1e468fd6bcd34da93dcf659
pto-structured-sync-test 7568768bb3c144f4a4a2cc812b0eb653191f220db37f43b4f15f4d6fc077e86c
libPTOASCompiler.so      e55cacfdbb27ec814b82d41b5b24ed8847b767c0d51d2ab161325ba76ded5592
```

## Exact scalar-memory coverage and fitting-domain allocation

This follow-up widens semantic coverage through audited operation contracts,
not through another control-flow arrangement recognizer. The general
composition adapter now admits exact one-phase contracts for `pto.tci`,
`pto.tgetval`, `pto.tsetval`, `pto.tconcat`, `pto.tdivs`,
`pto.load_scalar`, and `pto.store_scalar`. `pto.treshape` remains a transparent
SSA/storage alias rather than a physical scalar phase.

The scalar-memory contract is deliberately narrow. `load_scalar` and
`store_scalar` remain physical GM accesses on `PIPE_S`; they are not treated as
pure address arithmetic. Sequential `PIPE_S` work has intrinsic same-pipeline
completion and therefore never receives an illegal scalar barrier. A hazard
between `PIPE_S` and another pipeline still requires completion independently
from any GM cache action. At this milestone that cache action was not yet
realized. No event was reclassified as GM visibility. The CanonicalSync
hardware-model branch was used as read-only semantic evidence for these
boundaries; its symbolic planner was not imported.

The demand allocator now implements the fitting-population part of C1. It may
give every logical protocol a distinct physical key, including the canonical
fallback key, only when the complete population in **both directions** of the
pipeline pair fits. Scope is part of logical protocol identity. If either
direction is over capacity, or an existing packet uses the pair, allocation
retains the previous sharing-first/fallback policy. Native reconstruction now
recognizes a fallback packet from its actual adjacent four-command word and
otherwise accepts any target-available raw event key. Fresh protocol checking
still rejects packet/raw collisions and independent-domain reuse.

### Local qualification

Validated as an uncommitted diff on `85d1130507329f6bf6c6f916bdd8c1b87b526325`
with the existing LLVM/MLIR 19.1.7 build:

- Incremental builds of `pto-composition-core-test`,
  `pto-structured-sync-test`, and `pto-test-opt` passed with at most two build
  workers.
- `oahs_composition_core`, `oahs_composition`, `oahs_demands`,
  `oahs_structured_core`, and `oahs_structured` passed serially. The independent
  composition core reports more than 2.4 million finite assertions. The full
  lit/system suite and device execution were not run.
- The hash-frozen 363-input replay admits **222/363** inputs: PTOAS **53/150**,
  PyPTO **7/35**, and pypto-lib **162/178**. This is 25 admissions above the
  checked-in 197/363 result and 16 above the immediately preceding local
  206/363 coverage slice, with no losses. The manifest remains
  `95beab25427b1dd3a2ff1b59cd881a3f4181e0f83c8a8b44daf35555bf46c3b6`.
  This is prepared-IR compiler compatibility, not source regeneration, device
  correctness, or performance.
- The remaining first refusals are explicit: 81 unqualified visibility cases,
  18 authored cache-maintenance cases, 15 authored-synchronization cases, 11
  communication cases, 8 reserved-buffer cases, four invalid physical-context
  fixtures, two helper-contract cases, and two `tmrgsort` register-effect
  mismatches.
- On the unchanged demand benchmark, exact-fit allocation changes Q projection
  from 18/18 to **17/17** SET/WAIT sites. Historical GEMM remains **56 SET / 59
  WAIT / 21 named barriers**, versus InsertSync's **44 / 44 / 21**. The allocator
  refinement does not address GEMM's over-capacity fallback packets, so quality
  remains unqualified.

Reproducible artifacts are outside the source tree under
`oahs-clean-c1-build/test-results/oahs-coverage-m3-final` and
`oahs-clean-c1-build/test-results/oahs-demands`. The final replay identifies the
candidate `pto-test-opt` binary as
`76e9af55ed00eeabfd8331e765531b46ca2d17dcc36ab29aa5d0596a46f66da4`.

## Qualified scalar GM visibility and TMrgSort register results

The next coverage increment connects the exact scalar payload contract to the
qualified AIV cache-maintenance model. Completion and visibility remain
different facts:

```text
PIPE_S store -> non-scalar GM read:
    whole-GM cache clean, then GM fence

non-scalar GM write -> PIPE_S load:
    GM fence, then whole-GM cache invalidate

PIPE_S store -> non-scalar GM overwrite:
    whole-GM cache clean, then GM fence

non-scalar GM write -> PIPE_S overwrite:
    GM fence only
```

The fence drains the AIV physical pipelines. The cache operation discharges
only the corresponding scalar-cache publication obligation. A SET/WAIT event
still establishes completion without being treated as GM visibility. Ordinary
non-scalar GM crossings and scalar-crossing pure WAR therefore need only their
completion handoff. Scalar-crossing RAW and WAW retain the stronger contract.
The disputed MTE3-to-MTE2 same-address GM **RAW** publication remains
fail-closed and is reported explicitly; this implementation does not infer a
visibility recipe from the existence of an event route.

The composer records GM writer history separately for every observer pipeline.
That prevents an invalidate for `PIPE_S` from publishing the same value to
other observers, and prevents a clean of scalar writes from erasing unrelated
non-scalar publication obligations. A fence-only WAW clears completion but no
cache history; it is credited only at its immediate write-only target cut. A
later scalar read of another GM cell still requires invalidation. Native
emission generates the exact recipes above. Fresh reconstruction accepts only
adjacent, whole-GM CMO/fence pairs in the required order or an exact fence-only
action; deleting either CMO, deleting the WAW fence, or reversing either pair
rejects the clone atomically. Authored CMO and fence operations are still
outside the admitted input contract. Supporting them requires composing their
original synchronization semantics, rather than silently treating them as
generated recipes.

This increment also admits a dead exact register-only result of TMrgSort format
2. The `excuted` `vector<4xi16>` operand is written in place on `PIPE_V`, but is
not translated physical storage. It is excluded only when TMrgSort is its sole
use. Any later observation remains unsupported until register-result
dependencies are represented explicitly. All tile source, temporary, and
destination effects remain required. Both frozen inputs that previously
stopped at this mismatch now proceed to the independently unsupported
MTE3-to-MTE2 GM publication case, so the refinement improves the first-refusal
diagnosis but does not manufacture a corpus admission.

### Local qualification

Validated as an uncommitted diff on `8dd2b608e3703a54bb8a2d0bcca61e5bb3ed5594`
with the existing LLVM/MLIR 19.1.7 build:

- Incremental builds of `pto-composition-core-test`,
  `pto-structured-sync-test`, and `pto-test-opt` passed with two workers.
- `oahs_composition_core`, `oahs_composition`, and `oahs_demands` passed
  serially. The native composition campaign includes scalar RAW/WAW clean,
  invalidate, and fence-only cases plus live/dead TMrgSort results. It deletes
  each CMO independently, deletes the fence-only action, and reverses both
  two-operation visibility recipes.
- The hash-frozen replay admits **247/363** inputs: PTOAS **77/150**, PyPTO
  **7/35**, and pypto-lib **163/178**, with no losses against the preceding
  local result. The increase from the checked-in **222/363** result comes from
  correcting the former blanket GM-publication rule for qualified non-scalar
  pipeline crossings. One admitted input exercises the corrected scalar WAW
  contract: its four SET/WAIT operations are replaced by one CMO and one GM
  fence. The frozen schema counts SET/WAIT/barrier operations only; the native
  campaign separately records CMO and fence counts.
- The remaining 116 refusals are explicit: **58** unqualified MTE3-to-MTE2 GM
  publication cases, **18** authored cache-maintenance cases, **15** authored
  synchronization-summary cases, **11** communication cases, **8** reserved
  buffer cases, **4** invalid physical-context fixtures, and **2** unresolved
  helper contracts. The former TMrgSort effect-mismatch category is gone.

The frozen manifest remains
`95beab25427b1dd3a2ff1b59cd881a3f4181e0f83c8a8b44daf35555bf46c3b6`.
The replay is under
`oahs-clean-c1-build/test-results/oahs-coverage-visibility-final` and records
candidate `pto-test-opt` SHA-256
`136b4e4b978bc03364ccff866e22aa11237c32f1bb07b232222a1c03d5435ba8`.
This is prepared-IR compiler compatibility, not source regeneration, device
correctness, or runtime performance qualification.

## Authored fixed synchronization composition

The general composition engine now treats original `pto.barrier`,
`pto.cmo.cacheinvalid`, and `pto.fence.barrier_all` operations as immutable
program transitions. They are no longer mistaken for generated mechanisms or
silently skipped. Authored event protocols remain fail-closed because the
composer does not yet have an ownership summary for their tokens.

The transition semantics follow the separately maintained canonical hardware
model:

- A qualified per-pipeline barrier completes the named pipeline resource for
  same-pipeline issue and storage-release accounting. It does not transfer
  that prefix to another observing pipeline; a cross-pipeline acquisition is
  still required there.
- `PIPE_ALL` completes all pending resources on the current physical core.
- An AIV GM fence drains `PIPE_S`, `PIPE_V`, `PIPE_MTE2`, and `PIPE_MTE3`; an
  AIC GM fence drains `PIPE_MTE2`, `PIPE_MTE3`, and `PIPE_FIX`.
- A whole-GM cache operation before an AIV fence can publish preceding scalar
  writes. A whole-GM cache operation after the fence can invalidate scalar
  cache state for non-scalar writes published by that fence.
- An addressed cache operation is preserved but receives no abstract-cell
  coverage credit. The composition cell model does not prove cache-line
  geometry, so treating it as whole-cell maintenance would be unsound.
- Neither a hardware event nor a cache operation alone establishes GM
  visibility. The unqualified MTE3-to-MTE2 GM RAW direction remains refused.

These transitions participate in Sequence, Choice, For, and While composition
and in all three constructors. At a Choice join, MAY hazards are unioned while
cache-clean and fence-publication proofs are intersected. A new write
invalidates the corresponding generation-specific proof. Fresh reconstruction
reimports the authored operations from the emitted clone and verifies their
effect on the original physical requirements. Generated visibility packets are
forbidden from borrowing either half of an authored CMO/fence pair. Retirement
counts only the generated terminal drain, so an original `PIPE_ALL` is
preserved without being confused with that drain.

### Local qualification

The incremental two-worker native build of `pto-composition-core-test`,
`pto-structured-sync-test`, and `pto-test-opt` passes. The serial
`oahs_composition_core`, `oahs_composition`, and `oahs_demands` gates pass in
37.3 seconds. The native campaign exercises an authored per-pipeline barrier,
whole-GM clean/fence publication, and the immutability check for every authored
mechanism. A placement mutation moves a generated clean next to an authored
fence, and another moves a generated barrier beside an authored barrier. Both
are rejected by fresh packet reconstruction without changing the original
payload. Portable tests additionally cover reversed clean/fence
order, addressed-CMO refusal of coverage, PIPE_ALL completion, fence-then-
invalidate acquisition, one-arm Choice proofs, AIC fence completion, and AIC
refusal of scalar-cache publication.

The hash-frozen replay admits **249/363** inputs, with no loss from the preceding
247/363 result: PTOAS **79/150**, PyPTO **7/35**, and pypto-lib **163/178**.
The two new admissions are the AIV `lm_head_signal_clear.pto` and
`moe_signal_clear.pto` production kernels, whose original whole-GM cache
operation and fence now supply their scalar-write publication contract. The
remaining 114 refusals are explicit: 61 unqualified MTE3-to-MTE2 GM publication
cases, 39 communication cases, eight reserved-buffer cases, four invalid
physical-context fixtures, and two unresolved helper contracts. Some newly
exposed refusal counts differ from the preceding first-blocker taxonomy because
authored cache maintenance is no longer the first blocker.

The replay is under
`oahs-clean-c1-build/test-results/oahs-coverage-authored-fixed-final`, uses the unchanged
frozen manifest
`95beab25427b1dd3a2ff1b59cd881a3f4181e0f83c8a8b44daf35555bf46c3b6`, and
records candidate `pto-test-opt` SHA-256
`5389439f002274b6151580c4ff79e2e3ee103733ac68b6492b5548dd8e0ee280`.
This is prepared-IR compiler compatibility and mutation evidence, not device or
runtime-performance qualification.

## Bounded structured recurring episodes

The recurring demand-ring provider now admits a bounded structural episode
rather than requiring every endpoint to be a singleton operation and every
directed event edge to occur only once. A recurring word can contain Choice
endpoints, including an empty/active Choice, and can reuse one logical directed
key at several ordered points such as `A -> B -> A -> B`. This is a precision
provider for the existing demand/completion constructor; it is not a symbolic
occurrence analysis and it does not widen the operation or hardware contract.

The certificate is finite and structural. The provider projects the actual
generated SET/WAIT commands through Operation, Sequence, and Choice nodes,
deduplicates at most eight concrete words, and checks every ordered pair of
possible consecutive words. Each pair must start and finish with every owned
event key idle, and every WAIT must consume the unique live publication in its
direction. Several cycles in the same recurring structural domain are checked
as one event population. Nested loops are not assigned an implicit trip count;
an episode command attached to one is rejected.

This follows the hardware-model contract used as a read-only reference:
publication requires an idle key and captures a source prefix, acquisition
consumes that publication, branches must expose complete executable words, and
repetition must prove consumption before rearm. The implementation does not
copy the hardware branch's planner.

The optional proof has one aggregate work allowance covering projection,
Choice products, deduplication, and pair verification. Exhaustion rejects the
whole optional attempt transactionally. The independently selected legacy
closed-ring result remains the baseline, so a larger episode cannot remove a
previously accepted ring when its proof, allocation, or quality gate fails.
New structured episode keys receive dedicated noncanonical physical keys and
are excluded from ordinary scope coloring and flat protocol verification;
legacy singleton/unique-direction rings retain their established allocator and
certificate unchanged. Guarded entry/deferred protocols remain quarantined
until a joint finite automaton owns both populations.

### Local qualification

Validated against the LLVM/MLIR 19.1.7 build with the aggregate two-worker
limit:

- Incremental builds of `pto-composition-core-test`,
  `pto-structured-sync-test`, and `pto-test-opt` passed. The serial
  `oahs_composition_core`, `oahs_composition`, and `oahs_demands` gates passed
  in 38.4 seconds; the core reports 2,435,452 finite
  native-helper/independent-graph assertions.
- Portable tests cover equivalent Choice lane words with arm-specific cuts,
  an empty/active Choice, a final group without a lexical successor, repeated
  directed edges, exact rollback on zero proof allowance, and deletion of a
  middle acquisition.
- Native reconstruction uses an actual PTO fixture with four SETs and four
  WAITs per loop visit. It replays independently varying visit counts across
  repeated whole-program invocations and rejects a wrong key, a removed WAIT,
  a duplicated SET, and a balanced WAIT-before-SET reorder atomically. A
  derived Choice fixture exercises both structural words and repeated
  invocations.
- On the unchanged demand benchmark, the selected episode removes six named
  `PIPE_V` barriers from `four_use`; its event counts remain 20 SET / 22 WAIT.
  The other recorded cases retain their prior counts. Historical GEMM remains
  56 SET / 59 WAIT / 21 named barriers versus InsertSync's 44 / 44 / 21, so
  synchronization quality and device performance remain unqualified.
- The hash-frozen corpus remains **249/363**: PTOAS **79/150**, PyPTO **7/35**,
  and pypto-lib **163/178**. This milestone intentionally improves a recurring
  protocol shape rather than semantic admission. Two already admitted
  pypto-lib records differ in static command counts from the preceding saved
  report; rerunning the exact current binary with structured rings disabled
  produces the same new counts, so those differences are not caused by this
  provider. There are no admission-status changes.

The frozen replay is under
`oahs-clean-c1-build/test-results/oahs-structured-episodes-corpus-reviewed`, uses
the unchanged manifest hash
`95beab25427b1dd3a2ff1b59cd881a3f4181e0f83c8a8b44daf35555bf46c3b6`, and
records candidate `pto-test-opt` SHA-256
`39813d58bdc870f5f0c97f784b7aa617dd5bf81b23c345601814ad706a43ce54`.
No full lit suite, device execution, or runtime-performance qualification is
claimed.

## Authored blocking remote-signal cuts

The compositional importer now preserves original AIV `pto.comm.tnotify` and
`pto.comm.twait` operations as immutable remote-protocol cuts. This is a narrow
semantic-coverage increment, not a collective planner:

- `TWAIT` is an external blocking acquisition. It receives no credit for
  completion of unrelated local pipeline work. It conservatively marks GM
  scalar-cache state stale; a later scalar GM read needs an authored or
  generated target invalidate before it may consume peer-updated payload.
- `TNOTIFY` is accepted only when the current composed state proves that every
  preceding local GM access has been released. Non-scalar GM writes must also
  have crossed an authored GM fence; scalar GM writes require the qualified
  clean-then-fence publication sequence.
- An authored per-pipeline barrier completes only its named local resource. It
  can release that pipeline's preceding GM access for a later remote
  publication, but it does not transfer completion to a different local
  observer and does not establish GM visibility. `PIPE_ALL`, cache maintenance,
  and fences retain their distinct existing transfers.
- The signal word is an authored communication resource rather than ordinary
  intrafunction payload storage. The importer must prove that this resource is
  disjoint from every local GM payload access; otherwise it refuses the input.
  Remote endpoint pairing, payload association, and cross-core protocol
  correctness remain outside this planner. The pass preserves the operations
  but does not invent facts about the peer.
- The qualified contract is AIV-only. `TTEST`, `TPUT`, `TGET`, reserved-buffer
  ownership, queues, and other communication operations remain fail-closed.

Fresh reconstruction reimports both remote operations from the emitted clone.
Payload preservation first rejects deletion of either authored signal
operation atomically; the fresh checker then verifies the unchanged operations'
state transfers independently of plan selection, while intentionally sharing
the same fixed-action hardware semantics. Portable tests separately
challenge unfinished MTE3 publication, completion without GM
visibility, MTE2 read release, the fact that a named MTE2 barrier does not
transfer completion to a V observer, dirty scalar publication, the clean/fence
sequence, and the rule that a remote wait does not satisfy a local physical
hazard. Native negative cases require exact fail-closed diagnostics for an
unreleased publication, signal/payload aliasing, use on AIC, and deletion of
the invalidate required by a scalar GM read after `TWAIT`.

Remote-wait effects are included in the bounded structural loop summary. If
any loop-body path may execute `TWAIT`, loop entry is conservatively seeded
with stale scalar-cache state for every GM cell. This covers a scalar read at
the beginning of iteration `i+1` after a wait at the end of iteration `i`;
the zero-trip join remains a MAY join. The core regression removes the
resulting backedge invalidate and requires fresh verification to reject it.

The source-level composer deliberately requires the local release and GM
publication prefix to be explicit before `TNOTIFY`. The current target manual
also describes a lowering-time drain, while the hardware-reference branch
retains only an attribute-driven legacy lowering hook after removal of its
automatic MemoryConsistency pass. This milestone therefore takes the
fail-closed, explicit-contract interpretation; teaching the constructor to
materialize a qualified release at the notify cut is a separate coverage
improvement.

This milestone also corrects one assertion introduced with structured recurring
episodes. That case is accepted by the forced structured provider, while the
legacy-only selector deliberately keeps the independently verified non-ring
baseline. The asserted legacy cycle count is therefore zero; the generated
plan and fresh verification are unchanged.

### Local qualification

The incremental two-worker build of `pto-composition-core-test`,
`pto-structured-sync-test`, and `pto-test-opt` passes. The serial
`oahs_composition_core`, `oahs_composition`, and `oahs_demands` gates pass; the
portable core reports **2,435,467** assertions. No full build was performed.

The hash-frozen replay admits **253/363** inputs, up from 249/363: PTOAS
**79/150**, PyPTO **7/35**, and pypto-lib **167/178**. Four pypto-lib inputs now
compile. Seven other blocking-signal inputs remain fail-closed because the
signal may alias local payload: four PTOAS inputs use the conservative
`gm-alias=may-alias` policy alongside a different GM access, and three PyPTO
inputs later read the signal storage as scalar payload. The remaining 110 first
refusals are 74 unqualified MTE3-to-MTE2 GM publication cases, nine `TPUT`, six
`TGET`, seven signal/payload alias cases, eight reserved-buffer cases, four
intentionally invalid physical-context fixtures, and two unresolved helper
contracts.

The replay is under
`oahs-clean-c1-build/test-results/oahs-authored-remote-signal-corpus-r4`. It
uses the unchanged PTOAS and PyPTO/pypto-lib manifests with SHA-256
`f36f92f0a2027f0ea575fc951d8256e5a1643fc649bf8926b731b928bdfa79ef`
and `72a6469371b347b62d3068ccd1cc63cdfa0614a819db98a5913bfefc24f41ac6`,
respectively, and records candidate `pto-test-opt` SHA-256
`02ad2515d62f0c39000e3730631a5240cfd99851e5b38d1a1dce21dffcc4e996`.
This is prepared-IR compiler compatibility and mutation evidence, not device,
collective-protocol, source-regeneration, or runtime-performance qualification.
