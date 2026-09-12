# Compositional OAHS: baseline and cut-precision candidate

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
improves. The seven-case readiness/release/overlap gate remains required.

The amended direction is an evolution of InsertSync's physical translation and
structural traversal, not a claim that forward traversal or prefix sharing is
new. Original physical obligations, acquired completion, and event protocols
are separate concerns. The conservative engine composes admitted operations,
sequences, `scf.if`, `scf.for` and `scf.while` without periodic precision. The
new cut-precision candidate uses that same engine and target contracts, but is
currently exposed **only in the native test driver** as `cuts:none`.
It has not replaced the old precision-enabled backend.

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

1. Add backward incoming/continuation demand cuts, including first-consumer and
   empty-path participation. Recover softmax and QK boundaries without draining
   unrelated work or inventing completion across a zero-trip loop.
2. Extend finite slot-generation and first/last precision through the same
   transfer/protocol interfaces. The old dense constructor is not yet a bounded
   provider and must not become a hidden fallback.
3. Move audited positive effect contracts into the lowering-owned interface.
4. Pass readiness/release quality, final whole-suite checks and device
   qualification separately. Then remove shape-admission/sequence-bridge paths
   and switch the existing structured default.
