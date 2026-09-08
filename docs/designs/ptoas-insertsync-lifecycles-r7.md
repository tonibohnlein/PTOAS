# InsertSync R7 — guarded consumer regions and selective retry

## Status and base

This cumulative patch builds on the exact R6 package and the integrated R4/R5
repository base `1509506a43a619fc0f888a49ead90aecce92b0fc` in
`tonibohnlein/PTOAS:codex/insertsync-revision-r1`. The remote head read during
implementation is still that R5 revision; R6 is not assumed remotely integrated.
The unchanged R6 archive is retained in `r6-base/` with a SHA-256 pin.

R7 implements guarded lifecycle actions, a qualified existing-expression
classifier, and bounded optional-candidate retry. Host tests compile the actual
C++ protocol/guard core. **The MLIR adapter, native tests, original GEMM, corpus,
and device runs were not executed here: the native PTOAS/LLVM tools are absent.**
`evidence/native-preflight.json` records that limitation. No new GEMM inventory,
selected channel count, launch correctness, or speedup is claimed.

The intended milestone is an effective native lifecycle on the **unchanged**
GEMM, not merely a new recognizer. The implementation is supplied for that native
gate; the milestone must not be declared achieved from the host tests alone.

## 1. Concrete R6 restrictions addressed

R6 tied a role to a static phase. A repeated read can be first, intermediate, and
last in different iterations, so that requirement was too restrictive. R7 retains
R6's exact-slot/bundle semantics but derives roles per guarded analysis node and
factors their predicates into executable existing-IR conditions.

The current NativeGraph understands first/last classes and direct modulo guards.
Its existing first predicate required identical SSA operands. The frozen GEMM
contains duplicate zero constants and a continuation test of the form
`iv + step < upper`. R7 recognizes equal constant values for first tests and
qualifies continuation predicates when existing scalar types/expressions prove
that the addition cannot wrap. For example, an upper bound obtained as signed
32-bit K divided by 512 has a small enough maximum for a unit increment.

The Boolean domain also needs a concrete correction: signed APInt extraction of
`i1 true` yields -1 while the guard product represents Booleans as 0/1. R7
normalizes only one-bit constants before recording equality constraints. This
prevents a feasible true-equality path from being discarded. Ordinary signed
integer constants are unchanged. A focused native fixture and core checks are
included; no corresponding device failure was reproduced.

This is a source-derived qualification, not evidence that this was the actual
native fallback reason. Native per-candidate diagnostics still decide that.

R6 also abandoned all optional channels on a resource/combined-proof failure.
R7 can omit one implicated candidate and retry from the original input. It does
not keep the old residual graph after removing its supplying protocol.

## 2. Guarded occurrence roles

`LifecycleBoundaryProtocol.h` adds:

- `LifecycleBoundaryFacts`: proved empty-consumer bypasses and the scope boundary
  at which any readerless generation must be closed;
- `recognizeBoundaryLifecycle`: a finite-state fixed point over the actual guard
  product, with per-node roles;
- `synthesizeRoleGuard`: factor action participation into a disjunction of
  conjunctions over available three-valued facts;
- `chooseLifecycleRetry`: deterministic candidate choice in a failed domain.

The shared loop construction remains `empty | one | first middle* last`, with a
backedge rather than a chosen unroll horizon. The role state includes the member
write mask, in-use state, and whether a proved empty consumer region was crossed.
The completed episode renames its slot generation, as in R6.

Each physical reader may therefore have:

```
first occurrence:       acquire Ready; read
middle occurrence:      read
last occurrence:        read; publish Free
single occurrence:      acquire Ready; read; publish Free
```

Every consumer must still read the full selected bundle on one consumer lane.
Partial/unknown overlap, extra independent reader lanes, partial production,
history-dependent roles without an available predicate, and unmodeled effects
remain unsupported by this optional constructor. They do not reject the input.

The R6 uniform-role recognizer remains available for its original callers/tests;
R7 does not relabel it as a general conditional protocol solver.

### Guard factoring

At a static insertion point, all represented graph occurrences are considered,
including the occurrences where the action must **not** execute. Available facts
include actual first/last loop tests, emptiness, and dominating scalar equality
predicates. Unknown facts cannot rule out a failing alternative. A conjunction
is accepted only when it excludes every represented false case. Deterministic
literal/subsumption deletion reduces control expressions without changing that
criterion. Predicate count, work, and term budgets are explicit.

The adapter checks operand availability and physical scope. It emits actual
arithmetic/SCF expressions, not trusted role attributes. Boolean AND/OR expressions
are decomposed into constraints when the concrete output is reconstructed. This
avoids treating a generated conjunction as an unrelated Boolean coin flip.

## 3. Empty consumer regions

A generation that was produced but never read is not automatically free.

```
produce Panel
publish Ready
empty consumer region
independent suffix U
acquire Ready       // completes the actual producer
publish Free
next acquire Free / overwrite, or final drain
```

R7 permits this bridge only after the analysis has crossed a qualified empty,
read-only consumer loop. It does not turn an arbitrary consecutive write into a
valid lifecycle. The bridge is deferred until the next overwrite or until the
containing definition scope ends, so an independent suffix is not unnecessarily
made to acquire the panel's readiness. The existing guard must still be available.

This is a conservative target realization, not a global optimality claim. A
reverse signal after suffix work can include that source-pipe prefix; alternative
readerless recipes with other ordering tradeoffs are not synthesized here.

Physical-scope cleanup and drains remain before the original terminal ALL. R7
saves that boundary before adding SCF guards, so a new guard cannot hide the ALL
from a subsequent backward placement scan. No PIPE_ALL is removed or inserted
as resource recovery.

## 4. Native integration

`LifecycleBoundarySynthesis.cpp` adapts the core to existing PTO/MLIR:

1. Identify supported direct-body reader loops with no selected-slot writes in
   the consumer region. Loop lower bounds are known nonnegative constants; steps
   are positive constants; signed index semantics are used.
2. Import the R5 graph's actual guard environments, loop first/last classes, loop
   exits and block exits. These are internal snapshots, not new input annotations.
3. Recognize the complete lifecycle, including qualified empty alternatives.
4. Build before/after/block-end actions from per-occurrence roles.
5. Use dominance and original scalar expressions to build action predicates.

The constructor still runs before ordinary InsertSync repair. Its supply callback
removes only exact access pairs on a fully selected member. It does not set a
pipe-wide `alreadySync` fact. MMAD intrinsic ACC ordering remains separate from
operand completion. Residuals retain the established analyzer and target rules.

The native materializer builds guarded acquisition/publication at actual phases.
There is no kernel-name matcher, synthetic global slot counter, new caller
noalias promise, or no-wrap annotation. Original arithmetic and physical accesses
are not rewritten. New scalar operations are synchronization-control predicates.
Their runtime cost is not measured in this package.

### What R7 does not yet generalize

- Arbitrary dynamically selected multibuffer/event arrays and carried-handle swaps.
- Conditional last-reader synthesis when available scalar facts cannot distinguish
  the histories, or multiple independent consumer lanes.
- Full alternating lookahead, partially produced unused prefetches, accumulator
  lifecycle synthesis, or composite GEMM pattern coverage.
- A global generation-specific replacement for the general completion model.
- GM occurrence disjointness beyond the inherited supported query.
- General helper/queue/descriptor or cross-core semantics.

In particular, the historical GEMM's explicit L0 use sites may already have
uniform roles. The new repeated-region capability must not be presented as a
measured explanation of every missing optimization in that kernel.

## 5. Resource and composition retry

A failed dedicated-key assignment names its directed domain and candidate.
An expected residual-resource or conservative combined-event-proof failure can
request one deterministic candidate exclusion. The driver allows at most eight
attempts. Each attempt:

```
clones the original unsynchronized function
retranslates its accesses
recognizes candidates except the excluded identities
reruns all residual dependency insertion, motion, cleanup, and allocation
rebuilds the complete emitted plan and its proofs
```

Dropping a bundle can allow an independently valid smaller candidate on a later
attempt. The bound prevents unlimited subset exploration. This is not optimal
selection or a proof that serialization is unavoidable.

Malformed graphs, invalid allocation input, payload mutation, failed MLIR
verification, and contradictions in concrete protocol reconstruction remain
internal errors. Analysis limits are not reported as event scarcity. A failure of
the conservative combined residual event query is an explicitly unproved
composition, not a demonstrated runtime deadlock.

No failed attempt's SyncIR, omitted dependencies, event IDs, or mutated body is
reused. If no attempt succeeds, the untouched original takes the ordinary path.
Counters for selected channels/actions describe committed changes only.

## 6. Reconstruction and evidence

The reverse event key now reconstructs prime/release and acquire/drain through
the actual protocol state, rather than a lexical heuristic about surrounding
regions. Its state machine still requires complete production, matching readiness,
full-bundle reads, release, and exit consumption. A readerless bridge explicitly
acquires readiness before returning Free.

New keys have their own causal consumption-before-rearm check, followed by the
combined residual event check. Original operation identities, attributes, result
types and operands are checked. Residual memory correctness still depends on the
established InsertSync effect and dependency contracts. No whole-corpus/device
certificate is implied.

The new host fixtures use the same C++ loop-control, guard partition, role
recognizer, and predicate factoring as the adapter. Their independent asynchronous
interpreter reconstructs access requirements from explicit read/write regions,
separates issue/completion/signal submission/signal firing/wait consumption, and
checks both inherited signal-order models.

Positive tests retain:

- an independent suffix U outstanding with the panel load when the reader loop is
  empty;
- U from the preceding generation outstanding with the next panel load when a
  nonempty region releases its panel at the last reader.

Moving that release after U remains safe in the model but removes the latter
witness. Missing readiness and prematurely publishing release are distinct unsafe
mutations. None of these tests are a device timing experiment.

## 7. Native acceptance on the unchanged GEMM

Run R6 natively in a separate source/binary snapshot where available; R7's own
comparison runs same-compiler lifecycle-off/on arms. All cases retain their
original input hashes and caller contract. R6's source archive and commands are
preserved for an exact preceding-revision comparison; no R6 native result was
available here.

`compare_benchmarks.py` now parses per-candidate/attempt stages, requires R7
attempt diagnostics to detect a stale compiler, and exposes
`--require-selected-gemm`. The latter treats fallback as an effectiveness-gate
failure, separately from compilation correctness. Committing a protocol still
is not by itself a performance win.

Required report:

- original input, native library and wrapper hashes;
- matched physical slots/bundles and actual fallback stage;
- committed channels, guarded actions, retries, and supplied dependency counts;
- sets/waits separately; balanced static pair inventories only when defined;
- named barriers by pipe and PIPE_ALL separately;
- executed counts and preserved payload replay for each existing scenario;
- native/default corpus compatibility;
- numerical/progress results and device timing, or NOT_RUN.

A static first/last/empty split can have unbalanced static set/wait site totals
while preserving balanced dynamic execution. A combined count/score is never
used. No input is removed from the denominator merely because the optimization
falls back. The 213-row corpus has not been rerun here.

## Source and policy basis

- R6 package assessment and exact source, retained read-only in `r6-base/`.
- Integrated R4/R5 NativeGraph and InsertSync at the pinned base.
- Supplied comparative assessment: retain requirements independently of their
  realization; distinguish outstanding work from completion; port complete
  first/last/bypass recipes and preserve the production compiler path.
- Supplied amendment: remove an unallocatable optional candidate and rebuild;
  do not conceal internal defects through fallback.
- Existing SCF/Arith semantics and the already-used event contracts. This patch
  introduces no new hardware, GM publication, or cross-core guarantee.
