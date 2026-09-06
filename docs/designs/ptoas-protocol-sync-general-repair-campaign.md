# Parallelism-first general frontier repair campaign

## Governing revision — 2026-09-06

The revised objective supersedes the count-first and serialized-coverage policy
recorded below. Preserve the pipeline overlap permitted by required dependencies
and qualified target mechanisms within the existing operation schedule. Do not
reorder computation or alter buffer allocation. Fewer synchronization actions
are not an improvement when they add blocking between independent operations.

Normal repair publishes each required completion immediately after its source
frontier and acquires it immediately before its target. Initially, only identical
endpoint obligations share a repair. Broader sharing needs a certificate that it
adds no blocking; intersecting legal placement intervals are insufficient.
For `L1, L2, C1, C2` with only `L1 -> C1` and `L2 -> C2`, retain two events:
`set A` after L1, `set B` after L2, `wait A` before C1, `wait B` before C2.
The independent execution oracle must permit L2 and C1 to remain outstanding
together. Delaying the first signal until after L2 is a race-free but unacceptable
normal placement, not a performance optimization.

Every admitted **pattern** identifies a pipeline structure and supplies its known
efficient synchronization implementation, including intended overlap, storage
reuse, participation and event lifecycle. Elementary OneShot handoffs belong to
general repair, not the pattern catalog. Capacity-one control handshakes are
general recurring machinery; proven multi-slot pipeline protocols are useful
specializations. Existing narrow ReadyRelease construction remains transitional
until its overlap contract and general/pattern classification are audited.

Serialization is permitted only as explicit resource recovery after certified
event pressure, not simply because selective synthesis is missing or an analysis
budget expires. First try a non-serializing plan and proven-safe event reuse.
Total-phase loop cycles and V-hub packages remain reference constructors; they
must not win normal selection by having fewer event pairs. Removing them from
normal selection can temporarily lower native admission. Report that regression
honestly rather than counting serialized coverage as campaign completion.

### Revised implementation order

The continuation agreed after P4b requires full general-only corpus coverage:
cube kernels may not depend on adding new pattern recognizers to satisfy this
gate. UB, L1, L0A/L0B and ACC share storage/descriptor/occurrence analysis, with
separately qualified target effects. Pattern optimization follows completion of
the general baseline. The concrete continuation is:

1. Consolidate logical handoffs, typed synthesis refusals, common allocation,
   and memory-plus-token checked whole-channel deletion.
2. Generalize shared provenance, descriptor versions, outstanding effects and
   storage-specific non-draining region transfer interfaces across domains.
3. Qualify remaining cube/scalar/GM/fixed/communication/queue effects alongside
   structured synthesis, rather than deferring their semantics to patterns.
4. Compose selective entry, exit and zero-trip boundaries, then guarded choices,
   nested consumers and actual same-slot reuse with persistent channel scope.
5. Complete non-serializing supply reuse and explicitly attributed resource
   recovery, then close the full frozen corpus in the four target/GM settings.

Real acceptance anchors include `chunked_add`, `lookup_embedding`, `pack_x_hc`,
`lm_head_combine_gather`, `rms_norm`, `markov_logits`, and communication-wait
fragments. Synthetic fixtures retain the boundary, guard, selector and overlap
counterexamples; they are not measurements of frontend coverage or throughput.

The P0--P7 sequence below records the foundation for this continuation:

1. **P0/P1 — readiness policy:** independent direct frontiers, matching policy
   verification and overlap regression; fold normal OneShot selection into direct
   repair; remove count-first and serialized normal-world selection.
2. **P2 — event lifetimes:** prove consuming waits happen before rearming; separate
   proven pressure, conservative unknown interference and exhausted analysis.
   Add explicitly attributed serialization recovery only after that distinction.
3. **P3 — bounded evidence and frontend facts:** finish frozen collection without
   losing failures; descriptor state, access precision, authoritative core/view
   provenance and both GM contracts. Preserve the historical 394-row population.
4. **P4 — canonical structured state:** shared atoms, generations, outstanding
   readers/writers, stable requirement IDs and compositional boundary summaries.
5. **P5 — selective structured synthesis:** ordinary loops, choices, their
   composition and nesting, entry/backedge/exit/bypass and slot distances, with
   semantic verification and arbitrary-trip event invariants in the same slice.
6. **P6 — efficient protocols and remaining effects:** qualify genuine pipeline
   specializations, fixed supply, visibility, queues, communication and ACC/L1
   domains individually. Do not weaken semantic gates for admission.
7. **P7 — closure:** frozen corpus in A2/A3, both GM contracts and patterns on/off;
   native compilation without fallback, C++ emission and fresh verification.
   Report normal, resource-serialized, unsupported, analysis and collection
   outcomes separately. Host overlap witnesses are not device throughput results.

Each accepted slice requires independent algorithm and compiler-integration
reviews and a local commit. No pushes. The historical N-stage ledger below
records prior work, not the current selection policy or acceptance result.

## Contract and frozen source baseline

The campaign starts at PTOAS `61606de36841b7c8d857ed2a5dfc5d8a923bcc1a`.
On 2026-09-06, read-only remote resolution confirmed current `main` heads:

- PyPTO: `9f657f37ed20ce148b46fb7229c267a152a0644e`.
- PyPTO-Lib: `57e9d6a9294d9c38edd042f1edb5bdc50b4622bb`.

These match the clean snapshots under `build/protocol-sync-native-corpus/sources/`.
The input population is not yet collected. The historical externally reported
152 kernels are not a current measurement. Keep the 394-row historical
differential corpus separate from the newly collected frontend population.

Completion means every frozen valid frontend row compiles natively with legacy
fallback disabled, emits C++, and passes fresh concrete verification, separately
for A2/A3 and both GM contracts. General-only measurements disable specialized
pipeline-pattern selection. Completion requires selective, obligation-driven
structured repair; serialized references do not satisfy normal admission.
Host acceptance is not a hardware-correctness measurement.

## Execution and commit policy

Implement, independently review, address findings, and commit each accepted
slice locally. **Do not push**, including backup branches. User authorization
covers these local commits without repeated permission prompts. Preserve
unrelated `third_party/` content and other user worktrees. Use at most two
resource-intensive workers across the entire machine; main owns targeted builds
and tests, review agents inspect read-only. Store durable evidence under `build/`,
not the RAM-backed `/tmp`. Do not rebuild unchanged targets or LLVM.

## Historical ordered stages (superseded by P0–P7 above)

1. **N0 — freeze and baseline:** immutable frontend revisions, unsynchronized
   generated inputs, parameters, collection failures, commands and hashes;
   general-only four-way acceptance and concrete/C++ follow-up; all exposed
   blockers per row. Keep collection failures in the denominator.
2. **N1 — semantic facts:** descriptor-state updates, authoritative core context,
   forwarded/view provenance and Exact/Conservative/Unknown regions; preserve
   both GM contracts and require independent byte-set/descriptor tests.
3. **N2 — canonical requirements:** typed obligations independent of recognition
   and selection, outstanding readers and writes, generations and stable
   provenance; replace old protection only after internal/boundary no-omission
   accounting and dual-run agreement on the existing subset.
4. **N3 — structured interfaces:** memory plus completion/token transfer through
   sequences, joins and ordinary loops including entry/backedge/exit/bypass;
   conditional/nested loops and SSA forwarding; independent path oracle and an
   arbitrary-trip invariant, not bounded unrolling as proof.
5. **N4 — selective backward repair:** query established supply, traverse memory
   links/lanes/region interfaces, share legal handoffs with balanced participation;
   semantic concrete verification and token lifetime/progress witnesses ship
   together. Demonstrate removal of unnecessary adjacent-phase ordering.
6. **N5 — fixed supply and allocation:** import certified existing mechanisms and
   reservations before repair; reuse event generations only after proven death;
   deterministic feasibility retries retain conservative interference otherwise.
7. **N6 — remaining domains:** GM visibility, communication, queue ownership,
   multistage cube/local storage and ACC/proxy effects, prioritized by actual
   corpus blockers. New target claims need documentation/model/qualification
   evidence; no operation-name whitelist or generic body ALL-barrier escape.
8. **N7 — closure:** rerun every frozen row and historical differential population;
   preserve per-row artifacts, metrics and hashes. Missing frontend facts or
   unqualified target effects stay explicit blockers, never denominator removal.

No new pattern optimizations precede the general baseline. Reuse proven
lifecycle facts and complete parameterized handshakes where useful, without
emitting partial cyclic protocols. Each stage requires algorithm/soundness and
compiler-integration review, focused regressions and the ProtocolSync checkpoint
suite before committing. Device validation and target claims remain separately
qualified. Exhausted search budgets are not semantic counterexamples.

## Progress ledger

### P0/P1 — independent readiness policy accepted

This slice over `58bad1ba37570711ccea8c0a3b0006d79e2cd4a3` replaces greedy
interval merging with exact-endpoint repair and matching placement verification.
Mixed normal synthesis no longer selects elementary OneShot candidates or ranks
complete plans by event count. Serialized loop/V-hub recipes remain explicitly
constructed reference tests, not normal selection alternatives.

The new `protocol_sync_independent_readiness.pto` checks separate publications
and late acquisitions in A2/A3 and both GM modes, plus fresh concrete verification
and C++ emission. The direct unit's independent asynchronous oracle establishes
an L2/C1 outstanding-overlap witness and exhaustive bounded safety. Delaying the
first signal or advancing the second wait remains safe but removes the witness.
Identical endpoint obligations still share a repair. Existing reference loop,
choice, boundary, zero-trip and event mutation coverage is retained.

Validation on 2026-09-06:

- Targeted `cmake --build build --parallel 2 --target PTOASCompiler
  pto-protocol-sync-direct-repair-test pto-protocol-sync-mixed-test
  pto-protocol-sync-loop-memory-test` and incremental test-target rebuilds pass.
- With the workspace venv on PATH, `taskset -c 0,1 .venv/bin/python
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit -v -j1
  build/test/lit --filter 'protocol_sync_' -o build/protocol-sync-p1-tests.json`:
  **47/47 pass**, 61.82 seconds. This is the focused host checkpoint, not the
  entire repository suite or a new frontend corpus/device campaign.
  Result JSON SHA-256:
  `613ee4e44ac75dc9a90d0030b6e52f0fde9ac5aa13a0ace8f73fae5e8e055468`.
- Changed-code compliance check against HEAD: 13 files, zero errors/warnings;
  `git diff --check` passes. New C++ remains C++17, target-scoped CMake reuses the
  existing oracle source, and no target-ordering semantics were strengthened.
  Proprietary static analyzers and device qualification were not run.
- Independent read-only algorithm and compiler-integration reviews by
  `campaign_n0_algorithm_review` and `campaign_n0_compiler_review`: accepted.

P2 follows with event-generation death/reuse and honest pressure
classification. No serialized resource-recovery path is enabled by this slice.
Selective structured admission and the frozen corpus closure remain unfinished.

### P2a — once-only acknowledged event reuse

`EventLifetime` proves event consumption using an action-order graph, separate
from physical-memory completion. For each qualified logical handoff, its signal
precedes its consuming wait. Strictly ordered synchronization actions on the
same physical core and pipe supply further edges. The graph crosses directed
event domains: a V-to-MTE2 acknowledgement can prove that an earlier MTE2-to-V
wait consumed its token before MTE2 signals its next generation.

Planned waits are before their target phases and signals after their source
phases; concrete reconstruction uses the actual synchronization operations.
Ambiguous same-point ties add no edges. Every edge advances lexical position or
the before/after offset, so the proof cannot depend on a cyclic assumption about
the assigned event ID. Safe sharing follows by induction over that order: every
previous consumption precedes the next rearm. No signal/wait is moved or added.

Scope is deliberately bounded: unguarded, non-recurring handoffs in a single-block
function, with actual ancestry checked independently of generation metadata.
Loops, choices, missing anchors, recurring protocol kinds, and more than 128
total logical generations yield no new reuse proof. Conservative interference
remains. An absent path is not proof of simultaneous liveness; the existing
`ResourceInfeasible` result describes conservative graph coloring, not certified
physical scarcity. P2b must establish that distinction before any serialization
recovery is enabled.

The regression has seven load/compute generations sharing exact local storage.
Required reverse reclamation already acknowledges each forward acquisition.
All seven forward generations and six reverse generations use one ID per
direction. The native fixture is tested in A2/A3 and both GM contracts, with
fresh concrete verification and C++ emission. The execution oracle checks safe
reused generations and requires an actual unsafe witness when an acknowledgement
is removed. Separate negatives cover absent acknowledgements, forged loop
metadata, guards, missing anchors, and valid-ID inputs exceeding the proof cap.
The P1 independent-readiness regression remains the check against newly added
serialization. This is host/model evidence, not a device performance result.

Validation over P0/P1 commit `ed7037859cc15faf717261fe714acd745afe2f15`
on 2026-09-06:

- Targeted two-worker build of `PTOASCompiler` and the direct, mixed,
  loop-memory, OneShot, ReadyRelease, local-memory and scoreboard unit targets:
  passed; no LLVM rebuild. The OneShot reference regression was updated to
  explicitly check its existing reverse acknowledgement before expecting reuse.
- With the workspace venv on PATH, `taskset -c 0,1 .venv/bin/python
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit -v -j1
  build/test/lit --filter 'protocol_sync_' -o build/protocol-sync-p2a-tests.json`:
  **48/48 passed**, 62.22 seconds. Result SHA-256:
  `84687677044b976222749989a8bd34ff5f43a41ae6da4ec16c9c87b435a8027f`.
  A subsequent named-boolean compliance cleanup changes no test semantics;
  its affected OneShot target is rebuilt and rerun separately.
- Changed-code compliance check: nine files, zero errors/warnings;
  `git diff --check` passed. Independent algorithm and compiler-integration
  reviews accepted this bounded slice. No full system/device or new corpus run.

P2a does not finish P2: unknown interference and proof-budget exhaustion still
need explicit propagation before any pressure-recovery policy is enabled.
Selective loop/choice synthesis and complete frontend admission remain open.

### P2b — allocation failure attribution

The allocation result now distinguishes:

- `event-interference-unresolved`: the conservative interference graph cannot
  fit the available colors. This is not a simultaneous-live hardware witness.
- `no-unreserved-event-ids`: declared reservations leave zero compiler IDs in
  the affected domain. This does not establish that serialization can help.
- `event-allocation-analysis-limit`: feasibility search, domain size, or the
  bounded consumption-order analysis prevents establishing an assignment.

Successful conservative assignments remain accepted when a consumption proof
or color minimization hits its limit. Failure attribution includes the affected
core/directed domain and available-ID count in the raw allocator result; plan
dumps and CLI records currently propagate the typed reason and aggregate counts.
Those domain details are not yet exposed in the CLI. The historical
`max_event_domain_pressure` field means colors used by the computed assignment,
not measured or proven maximum simultaneously live hardware generations.

Direct and mixed plans retain a separate handled allocation-analysis-limit
status. Explicit reference OneShot/ReadyRelease routes retain unsupported status
with allocation attribution. An optional protocol's allocation limitation no
longer aborts mixed selection as an internal error; the direct alternative can
still be tried. Failed allocations keep all event IDs unassigned, and direct/
mixed verification reproduces their status and cause. Budget limits must not
turn into internal-error records merely because verification repeats a bound.

No failure class authorizes serialization. A future pressure certificate needs
a complete supported event/control model and an independently replayable cut
with more live generations than available IDs. P2a's under-approximation of
guaranteed order cannot supply that certificate by an absent path or clique.

Validation correction: P2b inspection found that earlier P0/P1 and P2a checkpoint
runs used a stale statically linked `pto-test-opt` for several native lit tests.
The compiler library and affected unit binaries were current, but the reported
47/47 and 48/48 totals were not entirely current-head validation. P2b rebuilds
that driver explicitly with every affected unit target and reruns the checkpoint.
In particular, the old 16-world expectation and the reference OneShot distinct-ID
expectation must not be interpreted as current normal-planner behavior.

Validation over `95b083310f1caa88ad2c7c981f60ddefa784e7b8`, 2026-09-06:

- Targeted `taskset -c 0,1 cmake --build build --parallel 2 --target
  PTOASCompiler pto-test-opt pto-protocol-sync-direct-repair-test
  pto-protocol-sync-mixed-test pto-protocol-sync-loop-memory-test
  pto-protocol-sync-one-shot-test pto-protocol-sync-ready-release-test
  pto-protocol-sync-local-memory-test pto-protocol-sync-scoreboard-test` passed.
  Shared statistics/plan interfaces required dependent PTOAS recompilation;
  review corrections used incremental builds, with no LLVM rebuild.
- Fresh-linked `llvm-lit -v -j1 build/test/lit --filter 'protocol_sync_'
  -o build/protocol-sync-p2b-tests.json`, using the same venv/affinity command
  as P2a: **49/49 passed**, 66.71 seconds. SHA-256:
  `9447defe08b9b491348c1e33d5e551e53a855b146c318eeebede538e4cc6cce4`.
  This supersedes the stale-driver totals above. Corrected the old same-pipe
  fixture to retain its second late acquisition before the required WAW barrier.
- `.venv/bin/python -m unittest discover -s test/experiments/protocol_sync
  -p test_records.py`: **28 passed**. The corpus blocker matrix retains all
  three terminal allocation reasons, without treating failed optional
  alternatives inside admitted functions as terminal blockers.
- Changed-code prefilter: 24 code files in the worktree, zero errors/warnings;
  `git diff --check` passed. Independent algorithm and compiler-integration
  reviews accepted. No new device or frontend acceptance campaign in this slice.

The new native 129-generation fixture checks handled analysis limits in A2/A3
and both GM contracts. Unit regressions cover zero reserved IDs even above the
domain cap, successful bounded minimization, forged attribution and cleared
assignments. Certified physical-pressure recovery remains deferred; proceed
with P3 corpus collection and semantic facts without enabling serialization.

### P3a — preserve source-backed frontend imports

The static collector now places an example driver's directory on its import
path, matching its absolute sibling imports, while still checking that the
loaded entry is exactly the inventoried source file. Collector arguments remain
hidden from module-local parsers. This is collection infrastructure only: it
does not change compiler admission or invoke a kernel/device runtime. Trusted
module top-level Python still executes. Failed seeds and partial PTO outputs
retain their separate records.

Both independent algorithm and compiler reviews accepted the bounded change.
On 2026-09-06, `taskset -c 0 .venv/bin/python -m unittest discover
-s test/experiments/protocol_sync -p 'test_*.py'` passed all **38 tests**.
The changed-code prefilter and `git diff --check` passed. No compiler rebuild
was needed. The previous interrupted `static-n0` collection remains untouched;
the next collection uses a fresh disk-backed result directory and the same
frozen frontend source commits. Static seeds are not the complete parameterized
kernel acceptance population.

### P3b — source-driver samples and completed static collection

The completed, source/tool-stable `build/protocol-sync-native-corpus/static-p3a/`
collection used the frozen frontend commits above and the collector snapshot at
PTOAS `13c19d818`. Its **349 static seeds** comprise 213 collected, 102 failed,
17 declared drafts and 17 requiring construction adapters. It retained **9,754
raw PTO files**, with 9,563 distinct file hashes (390,470,335 raw input bytes).
No failed seed in this run produced partial PTO. These are not the historical
152 kernels and are not native compiler admission results. Parameterized tests
and unresolved factories still require separate adapters.

The failed seeds comprise 81 TypeError, 16 ValueError, three ModuleNotFoundError
and two RuntimeError records. Examples include bare tensor signatures, frontend
UB capacity failures, a `contract` import collision and absent CANN devkit headers.
These are collection/frontend blockers, not ProtocolSync rejection categories.
No frontend memory-planning or runtime contract was changed to hide them.

Artifact SHA-256 values:

- `manifest.tsv`: `8d7fa280b832f49723e68d970a35994cedca5322f7bbc29281ef8c909245b44a`.
- `collection.json`: `2fcdf4e425e162b2f587a103417d091187f424f1563813827fd092ca958d2745`.
- `summary.json`: `756908937f22881c52ab373886332e4ae2d6ade3a1cbc8d4e66a62f6a6b9577e`.
- `run.json`: `88fb338b1074f21d8f79fc839c49fd01e00a752ce75e6ae52701a52f957dac94`.
- `hashes.json`: `df7b4ae192cdc733c4843faa558e10bce2c4b6328124f308521bbaadf8516fd3`.

The separate `--adapter driver` now recovers explicit source-backed samples for
bare signatures without executing drivers. It uses bounded AST interpretation,
explicit dtypes, source-order bindings, per-call provenance and worker-side source
revalidation. Unknown mutations, imports/rebindings, definition-time effects and
control retain unresolved records. Meta tensors preserve repeated-argument
identity but imply no disjointness; literal scalars remain specialized. The
campaign's fixed compiler settings explicitly override recorded RunConfig input.

Independent algorithm and compiler reviews accepted after correcting stale
bindings and failed/nested call accounting. On 2026-09-06, `taskset -c 0
.venv/bin/python -m unittest discover -s test/experiments/protocol_sync
-p 'test_*.py'` passed **55 tests**; the changed-code prefilter and diff check
passed. A frozen-frontend hello-world meta specialization emitted raw PTO under
`driver-p3b-smoke/`, using one CPU and all codegen/BLAS worker counts set to one.
No compiler rebuild or device execution was needed. Full driver collection and
native acceptance remain subsequent work, not results implied by this smoke.

### P3b follow-up — driver population and native baseline

The completed source-driver collection at `0294705ec790` retains **35 call-site
seeds: 15 collected, 20 needing a driver adapter**, with no source-inventory
failure placeholders. The collected seeds emit **18 raw PTO kernels**. This is
a separate, deliberately bounded population, not a replacement for the 349
static seeds, 9,754 static raw outputs, historical 394 rows, or parameterized
frontend tests. Source/tool stability checks passed. No driver or device kernel
was executed.

Artifacts below are relative to `build/protocol-sync-native-corpus/`:

- `driver-p3b/manifest.tsv`: `9e1906b6cb091d4487fc5a380c0a17025769ab049b95706b75baba179471b37a`.
- `driver-p3b/collection.json`: `021efd1174bf7614e991ee0626c46f399f3c6bf89540a2ea7a4965606f93c143`.
- `driver-p3b/summary.json`: `e82d0dd3b08ca4bca5d937d3deb2d7841fe5112a098ffa989499a0d49ea83979`.
- `driver-p3b/run.json`: `43ef3fe0287b974050fce9961d02a7067254f998524c1a9068d26bca5a0dbdcb`.
- `driver-p3b/hashes.json`: `8da1331cbcf469963fcaa4dd1058d050ae116c01ee39f47781731f6900a44cef`.

The acceptance runner used `--mode acceptance --workers 1 --expected-rows 18`,
the driver manifest/input root, `--patterns off`, and all four combinations of
`--arch a2|a3` and `--gm-alias may-alias|assume-disjoint-arguments`. Each run used
`taskset -c 0,1 .venv/bin/python test/experiments/protocol_sync/campaign.py`;
exact commands, compiler fingerprints and per-row hashes are in each run's
metadata. All four configurations admitted **16/18** natively with fallback
disabled; all 18 diagnostic probes completed. Runner `failed_rows=0` denotes
probe health, not 18 admitted kernels.

The same two rows reject in each configuration:

- `719dac71d4fe5c6bce89_000`, `chunked_add` from
  `examples/beginner/02_elementwise.py`: ordinary loop/reuse and recurring
  endpoint obligations remain unsupported by selective repair.
- `0053ad4c1a05b5b1c9dc_000`, `matmul_acc_64` from
  `examples/intermediate/04_matmul_acc.py`: ACC, conflicting-range and
  unknown-alias obligations remain unresolved.

The four `acceptance-p3b-*/summary.json` SHA-256 values are:

| Configuration | SHA-256 |
| --- | --- |
| A2, may alias | `f14614070e6de94253fd68ad68b20731421ba420dc46a016d3d7762938e16a7d` |
| A2, disjoint arguments | `60f74150c6b9a45263b0d21bb74d9e5fe9bdb8e7585471db774c8544647f36f8` |
| A3, may alias | `58d4cdd4330e37ccafd0e3a8a7ff3282adfcdced2b9dbbfe8af9300d378b92f7` |
| A3, disjoint arguments | `6f0d91695d59e030b13179cebd23dd7efea2c21f3c27f978d59868750b20c3b7` |

`native_followup.py --workers 1` over these four campaigns passed **128/128**
checks: 16 fresh concrete-verifier runs and 16 C++ emissions per configuration.
Compiler stability passed; `followup-p3b/summary.json` hashes to
`d749a020662de2ab2180952772b1f50972b335b141627d78855f31c5cf1bbcbe`.
These measurements precede the P3c compiler rebuild. They establish neither
device correctness/performance nor acceptance of the larger static population.

### P3c — conservative full-width UB row views

Recover bounded full-width row slices of direct addressed UB allocations when
the inherited and result-type physical strides agree. Bounds, positive static
dimensions, sizes/type agreement, ordinary row-major/unboxed/non-padded layout,
and checked address arithmetic are required. Preserve **Conservative** access
precision: an allocation or view extent is not an exact instruction footprint.

This deliberately excludes column cuts, nested views, slot selection, dynamic
offsets/addresses, unsupported layouts and partial/dynamic view valid extents.
`PTOResolveBufferSelect.cpp` can retain the parent's physical type when strides
differ; logical contiguity alone is not a sufficient footprint proof. No
legacy approximate subview span is imported as exact evidence.

The independent byte-set oracle covers all 36 contained row intervals of an
8-by-16 f16 tile, reconstructs their atom union, and checks precision. Negative
cases cover each unsupported boundary above. A fresh concrete reference accepts
two disjoint views, then rejects changing one offset to create overlap while
leaving synchronization unchanged. This is an address mutation of a manual
reference, not a claim that every legal descriptor change must race.

The native regression covers disjoint and partially overlapping views, including
different allocation SSA roots at the same physical address. Independent
readiness signals remain separate; overlapping reuse needs its reverse handoff.
It runs A2/A3 and both GM modes with patterns/fallback disabled, fresh concrete
verification for both targets, and A3 C++ emission. Its GM accesses are reads;
this fixture does not qualify GM alias or publication hazards.

Independent algorithm and compiler reviews accepted the bounded recovery proof.
On 2026-09-06, the compiler DSO, `pto-test-opt` and seven ProtocolSync unit
targets were rebuilt incrementally with `taskset -c 0,1 cmake --build build
--parallel 2 --target ...`; no LLVM rebuild or competing intensive job ran.
The local-memory unit target alone was relinked for subsequent test-only fixes.
The final configured LLVM lit invocation, through `.venv/bin/python` with the
venv on PATH and `taskset -c 0,1`, used `-v -j1 build/test/lit --filter
protocol_sync_ -o build/protocol-sync-p3c-tests.json`: **50/50 passed**, 70.22 s.
Earlier failures were a missing output expectation and malformed raw-IR type
syntax in the new test; both were corrected before this clean checkpoint.
Changed-code prefilter and `git diff --check` passed. No system/device suite ran.

### P3d — constant bounded descriptor-state flow

`set_validshape` and `get_validshape` now have descriptor-effect summaries,
separate from physical payload effects and fixed synchronization supply.
Observed direct allocations supply initialization actions. The immutable
schedule retains each descriptor version's exact handle, defining action,
constant row/column provenance and dimensions; accesses and metadata reads bind
to the version at their own lexical point. Same-address handles do not share
descriptor state.

The first native subset requires direct addressed ordinary UB allocations,
literal non-negative dimensions within the physical shape, and all descriptor
and payload users in one unconditional non-recurring block. Successful handle
validation is cached. Aliases, forwarding, escapes, helper/macro users,
conditional/recurring updates and unresolved scalar dimensions remain explicit
`unsupported-descriptor-state` failures. A later constant update cannot conceal
an unsupported initialization. Getter-to-setter feedback is also outside this
literal-only slice, even if a future constant-state analysis could resolve it.

Payload footprints remain full conservative allocation bounds: valid extents
do not establish exact byte effects, definite overwrites, or absence of padding
writes. Zero valid dimensions therefore do not erase memory obligations. A
metadata mutation supplies no completion edge, publication or payload WAW.

The binding verifier rejects non-descriptor actions carrying state IDs,
wrong-handle and missing/stale access versions, reads of future versions, and
corrupted defining actions/constants/bounds. Fresh concrete verification rebuilds
the descriptor state from emitted IR. The residual interpreter exempts only
bound descriptor effects; scalar tracing terminates at a certified constant
metadata read. Physically produced dimensions need prerequisites ending before
the metadata action and stay unsupported rather than silently losing that edge.

The test oracle independently interprets raw allocation/set/get operations and
compares per-handle dimensions at each metadata read and physical access over
25 initial/update combinations, including zero extents. It also moves a read
across an update and tests forged state bindings. Moving a legal read can change
the observed version without causing a race; the test requires reconstruction
of that change, not indiscriminate mutation rejection.

The native test preserves independent load readiness in A2/A3 and both GM modes,
with patterns and fallback disabled. It includes a getter/cast/physical-scalar
consumer chain, fresh concrete verification, and A3 C++ emission. Negative
cases retain dynamic, physically produced and conditional descriptor failures.
These are targeted semantic tests, not an updated corpus-admission count or
hardware/performance campaign. The source basis is `PTOOps.td`'s descriptor
contract and `PTOToEmitC.cpp`'s tile SetValidShape/GetValidRow/GetValidCol lowering;
no new target ordering or visibility rule is inferred from those definitions.

Independent algorithm and compiler reviews accepted this slice after adding
binding replay, cached handle validation and the physical scalar-consumer test.
On 2026-09-06, the targeted compiler/test-driver/seven-unit build completed
with the aggregate two-worker limit. Changing shared semantic records required
refreshing their ProtocolSync consumers; LLVM was not rebuilt. The first build
found an invalid ordered-map key in the new test oracle; it was corrected to a
Value-keyed map and the interrupted build resumed. A subsequent two-TU rebuild
and relink covered the final diagnostic and provider-binding changes.

The final configured LLVM lit invocation used the venv on PATH, `.venv/bin/python`,
`taskset -c 0,1`, and `-v -j1 build/test/lit --filter protocol_sync_ -o
build/protocol-sync-p3d-final.json`: **52/52 passed**, 84.13 s. An earlier
52-test pass preceded the final diagnostic regression; the final result includes
it. Initial focused lit failures were an invalid pass-level option spelling,
not failed native emission. Changed-code prefilter: 13 code/build files,
zero errors/warnings; `git diff --check` passed. No full system/device suite ran.

### P4a — selective recurring event-lifetime proof foundation

The next native loop target is the frozen driver's `chunked_add`: two independent
MTE2 loads, a V read/read/write join, and an MTE3 store. Its actual local hazards
provide forward readiness plus store-to-next-load and vector-to-next-load
reclamation. They do not require a load-to-load completion barrier. Native
selection/materialization and concrete memory verification are still pending;
this foundation does not enable the old serialized loop alternative.

`RecurringEventLifetime` checks a closed unconditional event-action schedule,
with one set/wait per logical channel per iteration and distance zero or one.
Same-lane action order carries distance zero, lane wrap carries distance one,
and set-to-wait causality carries the channel's declared distance. These are
event-action edges, never implicit completion of physical accesses.

The proof rejects zero-distance cycles. For a distance-d channel it requires a
path from its consuming wait to the next set of exactly distance 1-d. Zero
closure plus at most one carried edge answers that query without bounded
unrolling. Nonnegative causal distances and the acyclic zero-distance graph
establish progress; consumption-before-rearm preserves the one-token invariant.
Distance-one channels require a source-lane prime before the body and a
destination-lane drain afterward, including the zero-trip path. Distance-zero
channels start and end empty. The caller must independently reconstruct these
boundaries, complete participation and action ordering, unique concrete keys,
and target legality. All lanes need equal trip counts and no unmodeled blocking
operations. Eventual physical completion remains a target assumption.

This API alone certifies neither memory coverage nor visibility, and does not
allow different channels to share IDs. It currently has no production caller.
An immutable 64-channel budget returns `AnalysisLimit`, never a safety proof or
resource-scarcity certificate.

The independent test interpreter executes binary tokens and enumerates lane
interleavings rather than using the production closure. Tests cover the
five-channel selective fork/join at zero, one, two, three, four and seven trips;
384 two-channel order encodings (64 distinct direction/distance/lane-order
configurations); and explicit deadlock, missing acknowledgement, initial-credit
rearm, wrong-distance, missing boundary and ambiguous-position failures.
Bounded tests support the regression claim, not the arbitrary-trip proof.

Algorithm and compiler reviewers accepted this foundation. The targeted
`cmake --build build --parallel 2 --target pto-protocol-sync-direct-repair-test`
build passed; no existing production caller was changed. On 2026-09-06 the
configured lit invocation (`-v -j1`, filter `protocol_sync_direct_repair_unit`)
passed, 0.07 s, including the existing direct-repair and overlap checks plus
the new token oracle. Valid 64-channel and invalid 65-channel inputs, same-lane
channels and unsupported distances are tested explicitly. The changed-code
prefilter checked six code/build files with zero errors or warnings, and
`git diff --check` passed. No full suite, native corpus or device run is claimed
for this proof-only slice. Results: `build/protocol-sync-p4a-lit.json`.

The first native acceptance expectation is the declared disjoint-GM contract.
In safe GM mode, possible output/input aliasing across iterations can require
publication that these completion events do not qualify. That remains an
explicit blocker, not permission to waive the visibility obligation.

### P4b — native selective isolated-loop repair

The P4a token proof now has a production caller. An isolated unconditional
`scf.for` with fixed bounded UB footprints and ordinary Vector-core MTE2/V/MTE3
phases can receive an atomic selective repair without a recognized storage
protocol. Forward and distance-one carried obligations select handoffs at their
actual source/target phases. There is no total-phase cycle, V hub, implicit
ordered-loop grant, or count-first merging of independent readiness frontiers.

Cross-lane handoffs are established first. A same-pipe cut is added only if
their iteration-aware completion closure does not already satisfy its hazard.
This matters for repeated loads: a redundant MTE2 backedge barrier before the
second load could also drain the first load of the current iteration. Avoiding
that barrier preserves the intended independent handoffs. Distinct logical
channels retain distinct IDs in each directed domain; resource failure does
not silently enable serialization.

The mixed planner, final emission gate and residual interpreter consume explicit
completion relations. Canonical local obligations remain present; the scope
option accounts for analyzed loop accesses without declaring them ordered.
Independent concrete reconstruction reads actual sets, waits, primes, drains,
barriers and action order, then checks event consumption before rearm. A separate
all-access-pair checker verifies local and GM occurrence coverage without the
sparse atom/requirement builder. Fixed-footprint writer self-recurrence composes
distance-one coverage to arbitrary positive distances; this argument does not
extend to symbolic slots or conditional execution. Loop-carried SSA arguments
remain outside this native slice.

Scope is deliberately bounded: one isolated top-level loop, positive constant
step, no iter_args, choices, physical prefix/suffix, nested loops, macros, queue
effects, descriptor mutation or hidden/fixed synchronization. Planner bounds on
operations, phases, accesses, channels, barriers and expanded completions match
the concrete verifier's supported budget. Unknown scope or exceeded budget
remains unsupported before materialization. GM may-alias write/read publication
remains unqualified; the disjoint-argument contract is explicit, not inferred.

Tests include two independent readiness handoffs, a fork/join/store loop,
one-phase self-recurrence, zero/one/odd/even trips, missing prime and backedge
mutations, shifted physical overlap, budget rejection and mixed-plan metadata
corruption. The asynchronous oracle admits simultaneous second-load and
first-consumer work. It does not claim that two loads separated by a blocking
source signal can execute concurrently. Its reduced fixtures have 512-byte
footprints; it is not an execution proof for arbitrary corpus dimensions.

The frozen driver `719dac71d4fe5c6bce89_000` (`chunked_add`) compiled natively
under A3/disjoint-GM with patterns disabled and fallback disabled: six event
pairs, zero targeted body barriers, one mandatory exit drain, maximum directed
domain pressure two. This is one targeted admission, not a rebaselined corpus
percentage or a device/performance measurement. Artifacts:
`build/protocol-sync-p4b-chunked-add.pto` and
`build/protocol-sync-p4b-native.json` (two native lit tests passed before the
final budget/self-recurrence additions).

Algorithm and compiler reviews accepted the slice after fixing redundant
same-pipe cuts, deduplicating concrete supply and enforcing emitted-size budgets.
Final validation on 2026-09-06:

- Targeted two-worker build of `PTOASCompiler`, `pto-test-opt` and the seven
  ProtocolSync unit executables passed; all static callers were freshly linked.
- Configured LLVM lit through the workspace venv, `taskset -c 0,1`,
  `-v -j1 build/test/lit --filter protocol_sync -o
  build/protocol-sync-p4b-final.json`: **54/54 passed**, 114.97 seconds.
- The preceding 55-test run had 54 passes and one incorrect new scalar-SSA
  expectation: scalar order was intrinsic. That unrelated test and speculative
  SSA change were removed; the final rebuild/rerun above includes the retained
  selective-loop tests unchanged.
- Changed-code prefilter: 17 code/build files, zero errors/warnings.
  `git diff --check` passed. No full system or device suite was run.

The next implementation scope is selective entry/exit/bypass composition, then
choices; the campaign and full-corpus objective remain incomplete.

### P4b follow-up — clean native rebaseline and broader seed sample

All runs below used clean implementation commit
`10cc08cf2325f6fba39a9ab9e869f9414894767f`, unchanged compiler fingerprints,
patterns disabled, native fallback disabled, and at most two aggregate workers.
No source or compiler changed while a campaign or concrete/C++ follow-up ran.
Each acceptance campaign directory records `run.json`, `rows.jsonl`, aggregate `summary.json`,
per-probe compressed diagnostics, emitted IR and `hashes.json`. Commands and
toolchain revisions are recorded in `run.json`. Follow-up directories instead
record input campaign provenance in `provenance.json`, with per-probe records,
summaries and hashes. An independent read-only compiler review checked the
counts, sampling rule, extraction split and all hash anchors below and accepted
this evidence record; no extra build was needed for the documentation update.

The frozen 18-row driver population now gives:

| Target | GM contract | Native | Rejected |
|---|---|---:|---:|
| A2 | may-alias | 16 | 2 |
| A2 | disjoint arguments | 17 | 1 |
| A3 | may-alias | 16 | 2 |
| A3 | disjoint arguments | 17 | 1 |

The newly admitted disjoint-mode row is `chunked_add`; the ACC example remains
rejected in every mode, and `chunked_add` remains rejected in may-alias mode.
All **132/132** follow-up probes passed: fresh concrete verification and C++
emission for each admitted row in each configuration. These are compiler/host
checks, not device execution or performance results.

Run commands used `.venv/bin/python test/experiments/protocol_sync/campaign.py`
under `taskset -c 0,1`, with `--mode acceptance --workers 2 --expected-rows 18`,
`--input-root build/protocol-sync-native-corpus/driver-p3b`, its `manifest.tsv`,
`--patterns off`, each explicit architecture/GM contract, and result directories
`build/protocol-sync-native-corpus/acceptance-p4b-{arch}-{contract}`. All four
campaign commands exited zero. `native_followup.py --workers 2` with all four
`--campaign` paths and `--results .../followup-p4b` exited zero.

To avoid treating that small sample as broad coverage, a second bounded sample
takes the **first manifest row of each successful static parent seed**, in
the frozen manifest's order. This produces 213 rows with 186 distinct input
hashes. It is intentionally not topology-stratified, randomly sampled, or a
deduplicated/full-corpus denominator. The remaining outputs from those seeds
and the 136 unsuccessful/draft/adapter-required seeds stay outside this run,
not outside the campaign's unresolved collection population.

With A3/disjoint arguments, that sample gives **14 admitted, 194 rejected and
5 pre-ProtocolSync failures**. Of the rejected rows, 104 have reported semantic
extraction failures and 90 have no reported extraction failure. All five
pre-pass failures report `pto.tci` requiring explicit temporary storage when
PlanMemory is skipped. They are preserved as
`incomplete-or-invocation-error`, not counted as safe planner rejection or
silently removed. The campaign exits one because of those five rows (ten failed
acceptance probes: five empty-world and five strict), despite completing all
213 records. Source
stability is true; no crash or timeout exit code occurred. All **28/28** concrete
and C++ follow-up probes for the 14 admitted rows passed.

Overlapping observed blockers include unsupported descriptor state on
`pto.alloc_tile` in 61 rows, communication `twait` in 35 and `tnotify` in 33,
unsupported recurring direct repair in 64, and unsupported direct control in
17. These are exposed diagnostic counts, not a disjoint partition or a promise
of admissions after each fix. Timeline rejection and empty-world obligation
counts can also occur in admitted rows; they are not actual uncovered-plan
hazard counts.

The broader command used the same runner with `--expected-rows 213`,
`--input-root .../static-p3a`, `--manifest .../static-p3a/first-per-seed-p4b.tsv`,
`--arch a3 --gm-alias assume-disjoint-arguments --patterns off`, and
`--results .../acceptance-p4b-first-per-seed-a3-disjoint`. The follow-up directory
is `.../followup-p4b-first-per-seed`. Here `...` denotes
`build/protocol-sync-native-corpus` in all paths.

SHA-256 anchors (per-file diagnostic/emission hashes remain in each archive):

| Artifact | SHA-256 |
|---|---|
| Driver manifest | `9e1906b6cb091d4487fc5a380c0a17025769ab049b95706b75baba179471b37a` |
| First-per-seed manifest | `a08791b3255bf7944ac8cf6b0bf2f0168a4718d35666711b930792309613adf2` |
| A2 may-alias driver rows | `68e06d96230928ba0e6ce408e93dbcfab022b4b17ee3547d637fd4b5aedbe091` |
| A2 disjoint driver rows | `a3fcdc492fb00ef374c306bd44465be86bd44fb499f7db23adf54a124a4f2189` |
| A3 may-alias driver rows | `f3c8a975252cafbb276e04d0a1db27717e856196f6081863e4d7265ca978bc68` |
| A3 disjoint driver rows | `406e6bb5a50429c66529e847d8e6de1c6d62685dc23a72708fb0ef9b005aa3c3` |
| First-per-seed rows | `cb82ff35ac4e9659837ac7e42e35c676720ed90fd41e7880e49d9f1fc2623b5a` |
| Driver follow-up summary | `67503132deecd3f6eb593b1e8f30d9ed5611ac5cf45f7b3a9f77f5b2c23a7664` |

### R1 — logical selective channels and checked deletion

The selective builder now returns unallocated logical handoffs. Event identity
is determined independently of ID assignment, so an unassigned cross-lane
handoff remains present in token proofs and cost/budget accounting. Selected
channels enter the same allocator as direct and protocol resources. Recurring
channels still conservatively interfere within each directed domain; failure
to color that graph is not permission to add serialization.

Builder attempts distinguish unsupported scope/effects, analysis limits and
unproved token contracts. Mixed-plan verification reconstructs both successful
plans and refused attempts, including replay of resource failure. An internal
certificate contradiction remains a failure, not ordinary unsupported synthesis.

A bounded reverse sweep attempts deletion of whole event channels, including
their prime/body/drain actions. Each accepted deletion must pass the independent
occurrence-pair memory check and a newly constructed recurring-token proof.
The original accesses are unchanged. This is not an optimality or even general
deletion-minimality claim. Same-pipe cuts and independent readiness positions
are not broadened or moved.

The fork/join regression and real frozen `chunked_add` both reduce from six
event pairs to five. The latter retains zero targeted body barriers, one
mandatory exit drain and event pressure two. Its removed compute-to-next-load-A
relation remains supplied through compute-to-store-to-next-load-A. The updated
execution oracle checks zero, one, two, three, four, seven and eleven trips;
the existing separate acknowledged-token tests retain negative rearm witnesses.

Additional tests exercise a seven-channel directed domain reaching allocation,
verified mixed resource/analysis refusals, partial and colliding assignments,
and selected deletion/action statistics. The full focused invocation passed
54/55 tests; the new resource test expected the enum name instead of the public
`event-interference-unresolved` diagnostic. After correcting that assertion,
all three selective CLI tests passed. No production change followed that run.
The changed-code prefilter reports zero errors/warnings; no full static-analysis,
device or performance campaign was run.

Four patch-frozen 18-row driver runs used patterns off and fallback fail. Both
A2 and A3 retain 16/18 native rows in may-alias mode and 17/18 in disjoint mode.
All 132 fresh concrete/C++ follow-ups pass. This slice improves representation,
verification and emitted synchronization; it does not broaden admission.

Evidence is under `build/protocol-sync-native-corpus/acceptance-r1-{arch}-{contract}`
and `followup-r1`. Each acceptance `run.json` records source `c26c5ece3`, the
tracked patch hash `ca791fdbdd3a22ffb70c05c05371cca1c8e099d2178686ad76dcc7b4882e0374`,
the untracked regression hash, toolchain/compiler hashes, exact commands and
`source_stable=true`. The follow-up reports `compiler_stable=true`. The ledger
addition itself follows those frozen runs. Commands use the checkout venv in
PATH, `campaign.py --mode acceptance --workers 2 --expected-rows 18 --patterns off
--allow-dirty`, the driver manifest and each architecture/GM contract; the
follow-up consumes all four campaign directories with `--workers 2`.

Algorithm and compiler-integration reviewers accepted the bounded slice after
failure routing, optional-plan handling and statistics findings were resolved.
Selective scope still rejects fixed/hidden synchronization and reservations;
composition must import and verify those before relaxing that gate. No new
control scope, cube qualification or pattern family is claimed here.

### Next boundary proof gate

The algorithm review recommends first/last relevant occurrence handoffs, not
unconditional outside-loop waits. Even when an entry consumer is first on its
lane, moving its wait before a zero-trip loop can block unrelated suffix work:
prefix B is ready, independent prefix A is slow, the body would consume A,
but the suffix only consumes B. Covering required bypass hazards does not prove
that this added A-to-suffix order is harmless.

The next slice should preserve the selective recurring body while giving entry,
exit and bypass separate occurrence relations and initially distinct once-only
event keys. First/last-iteration guarded actions need independent participation
and zero-trip verification. Existing canonical shared-atom boundary requirements
can be reused; the old serialized boundary-edge chain cannot. Post-loop drains
must not become an indiscriminate completion hub that delays independent suffix
work. This gate is a design review outcome, **not implemented native boundary
support**. The full 9,754-output campaign, general choices/nesting, broader
descriptor provenance and communication/ACC target qualification remain open.

### Historical N0 progress

- N0 in progress: source heads resolved; added general-only mixed-mode selection
  control and evidence-policy propagation. No current-head corpus results yet.
- Collection environment: the PTOAS venv has no PyPTO, pytest or torch. Recover
  an isolated compatible frontend environment without altering user worktrees.
- Next: validate/review the selection-control slice, then collect the frozen
  frontend inputs and record all collection and compiler blockers.

### N0a — general-only selection control accepted

`--protocol-sync-mixed --protocol-sync-patterns=off` now retains general direct,
loop and structured alternatives while disabling OneShot/ReadyRelease synthesis.
This is selection-policy plumbing, not broader semantic admission. The
acceptance runner and C++ follow-up preserve the policy and require explicit
zero selected-protocol counters. Historical follow-ups omit the new switch
when their metadata predates it, preserving old compiler compatibility.

Independent read-only reviews `campaign_n0_algorithm_review` and
`campaign_n0_compiler_review` accepted this slice. The latter found the old
compiler compatibility issue; it was corrected and re-reviewed. Added tests
cover policy mismatches/missing counts and native plus concrete straight-line,
choice, loop-choice and prefix/loop/suffix paths with patterns disabled.

Validation on 2026-09-06:

- `taskset -c 0,1 cmake --build build --parallel 2 --target PTOASCompiler pto-test-opt`:
  passed. Adding the generated pass option required rebuilding its dependent
  translation units; no LLVM rebuild or parallel competing job was started.
- `.venv/bin/python -m unittest discover -s test/experiments/protocol_sync -p test_records.py`:
  27 passed.
- `.venv/bin/python .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py --repo . --base HEAD`:
  9 changed code files, zero errors/warnings. This prefilter does not substitute
  for unavailable external analyzers or hardware qualification.
- With `.venv/bin` prepended to `PATH`, run the configured LLVM `llvm-lit`
  through `.venv/bin/python` under `taskset -c 0,1`, `-v -j1 build/test/lit
  --filter protocol_sync -o build/protocol-sync-general-n0-suite.json`:
  46/46 passed in 58.61 seconds. The focused new test also passed separately.
- `git diff --check`: passed. No full system or device suite was run.

Next is N0 collection/environment recovery and the four-way native baseline;
N0 as a whole and the full campaign remain incomplete.

### N0b — static frontend collection adapter accepted

Built the frozen PyPTO snapshot's `pypto_core` incrementally in `build-corpus`
with an isolated Python 3.14.6 environment, system-installed CPU torch 2.12.0,
and nanobind. Initialized only the pinned libbacktrace, msgpack-c and Simpler
source submodules. The first build exposed a missing Simpler header; after
recovering that pinned source, the same targeted build resumed successfully.
No frontend tracked source, unrelated checkout, or global environment changed.
The build used two workers; nested libbacktrace make was explicitly limited to
one worker via `CMAKE_BUILD_PARALLEL_LEVEL=1` at configuration time.

The static adapter inventories 349 example/model entry seeds at these source
revisions. This is not a generated-kernel count or a complete acceptance
population: nested constructors, missing signature facts and parameterized tests
still require collection work. Raw PTO comes from public pre-pass specialization
and `ir.compile(skip_ptoas=True, memory_planner=PYPTO)`, not cached C++ or legacy
synchronization. Partial outputs retain their failed parent record. Module
imports execute trusted frontend top-level Python; the adapter never invokes
the compiled kernel or a device runner.

Both independent N0 reviewers accepted after fixing an implicit codegen thread
pool: `PYPTO_CODEGEN_MAX_WORKERS=1` now accompanies the BLAS/OpenMP limits in
both parent and standalone worker paths. Added source-file import verification,
runtime/package/module provenance, syntax-error accounting and explicit scope
documentation. The hello-world smoke emitted raw physically planned PTO.

Validation on 2026-09-06: 36 stdlib experiment tests passed; focused evidence lit
test passed; the 46-test ProtocolSync suite passed in 58.99 seconds without a
new native build. Changed-code prefilter: four files, zero errors/warnings.
Artifacts: `build/protocol-sync-collector-focused.json`,
`build/protocol-sync-collector-suite.json`, and
`build/protocol-sync-native-corpus/smoke-hello-world/`.
No full system/device campaign was run. Next: execute collection and extend
construction/parameterized adapters, preserving their separate denominators.
