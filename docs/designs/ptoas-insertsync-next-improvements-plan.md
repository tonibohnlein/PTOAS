# InsertSync: concrete improvements after the analysis-utilization audit

Planning baseline: 2026-09-08, R8 `4c19cc1c` plus the uncommitted per-buffer
generation implementation. The requirements below are retained as the design
and acceptance plan. Implementation status is now tracked explicitly here and
in the [generation design](ptoas-insertsync-buffer-generations.md).

The objective is to remove unnecessary serialization on unchanged inputs while
preserving numerical behavior, asynchronous progress, and return-time completion.
Generation, required access ordering, full completion, and event consumption
remain distinct facts. Extend the existing analysis and import rather than
creating another parallel analysis stack.

## Current implementation status

| Step | Implemented | Remaining acceptance or extension |
| --- | --- | --- |
| Shared open flow | Shared reaching definitions, R5 frontiers, guarded order and immediate lane predecessors; constructor and ordinary repair consume it | More precise general partial-write/ownership transfers |
| Exit completion | Independent omitted-barrier proof and concrete token-consumption check; compiler-owned tails only | Device correctness/progress qualification |
| M/FIX and composition | Qualified MMAD predecessor queries with emitted recheck; full-completion barrier cleanup; residual-only R4/R5 barrier deletion | R5 event movement requires protocol identity rebinding; guarded refinement with selected protocols is deferred |
| Native occurrences | Populated affine/modulo and physical slot relations; full-period wraparound protection; fresh output recheck | Richer nested selectors and dynamic-slot protocol construction |
| Region/handle transfers | Parameterized reaching-definition substitution; supported guarded regions and two-/three-slot handle rotations used natively | Fully parameterized reader/ownership transfer and arbitrary nested permutations |
| Handoffs/allocation | Complete bundle compatibility; proved residual-key sharing using the combined logical event plan; exact assignment-failure domain/key mask | Native scarcity benefit, joint sharing of selected lifecycle keys and more precise non-allocation retry witnesses |

There are no kernel-name, frozen-address or prescribed nesting tests in the new
analysis. Native controls use different slot depths, loop-carried handle rotations,
open accumulator flow, incompatible MMAD cases and an undrained final store.
The implementation remains opt-in with conservative fallback. The broad remaining
extensions above are not claimed complete or device-validated.

## Baseline and concrete targets

| GEMM plan | Set/wait pairs | MTE2 | MTE1 | M | FIX | PIPE_ALL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hand-tuned | 53 | 0 | 0 | 0 | 0 | 0 |
| R8 staged + MMAD | 44 | 3 | 2 | 4 | 1 | 1 |
| Current generation implementation | 56 | 0 | 0 | 4 | 1 | 1 |

The current generation candidate commits four L1 slots and two L0 operand
bundles on its first attempt, without allocation retries. The remaining M
barriers are immediately before the initializing/accumulating matrix operations
in the two branch variants. The FIX barrier is immediately before the output
store. The exit PIPE_ALL follows explicit FIX→M, M→MTE1, and MTE1→MTE2 drain
waits. These positions identify proof questions; they do not prove redundancy.

The first optimization target is the exit PIPE_ALL. The next is the five named
barriers. Fewer pairs are a later event-placement/resource objective. Do not
force 53 pairs or infer performance from a combined synchronization count.

## 1. Make the existing generation result usable without a selected protocol

**Problem:** generation discovery, initialization admission, and closed-token
construction currently share one entry point. Failed candidates disappear
before ordinary repair or R5 placement can use their storage facts.

**Change:** split `analyzeBufferGenerations()` into storage-flow discovery and
protocol construction. Evolve the current result rather than adding a third
engine. The flow result must not contain a protocol acceptance certificate.
Represent live-in content as an alternative, retaining other proved facts;
protocol construction decides whether that alternative is supported. Move
single-producer/single-consumer lane and four-member recipe limits out of the
general flow analysis where they are not semantic limitations.

For the initially supported exact physical regions, retain access identities,
reaching-write alternatives, first/final readers per lane, next overwrites,
guards, and exit obligations. Expose qualified queries for required ordering;
keep established ordering/full completion in the selected-plan supply service.
Unknown information preserves the existing conservative requirement.

`InsertSyncAnalysis` must be able to receive this read-only result even when no
complete lifecycle is selected. R5's placement-anchor discovery must consume
the same result for this supported domain. Its older conservative path may
remain an explicit fallback for other domains; do not claim that the two
representations are equivalent outside the migrated domain.

**Files:** `BufferGenerationAnalysis.h`, `LifecycleBoundarySynthesis.cpp`,
`LifecycleSynthesis.{h,cpp}`, `InsertSyncAnalysis.{h,cpp}`, and
`StorageFrontierAnalysis.{h,cpp}` / `StorageFrontierQueries.h`.

**Acceptance:** a buffer with readers on two consumer lanes is rejected by the
single-consumer protocol recipe but still supplies reader/overwrite obligations
to ordinary repair and placement. A resource-rejected candidate retains the
same storage facts after rollback. Native consumers use those witnesses to
select repair frontiers; populated fields or counters alone do not satisfy the
test. Preserve the current 11-fixture inventories in this foundation change.

## 2. Prove whether the final PIPE_ALL is redundant

**Problem:** a complete explicit drain may coexist with the unconditional
automatic tail barrier. Successful lifecycle construction bypasses ordinary
post-emission cleanup, and current frontier refinement excludes PIPE_ALL.

**Change:** add a narrowly scoped exit-completion query to the existing
completion machinery and invoke it on the emitted lifecycle-plus-residual
plan. Evaluate the program with the candidate tail barrier excluded from the
proof. For every reachable return, establish completion of every relevant
issued operation and external effect in each physical context. Also preserve
the concrete event-consumption/rearm proof. Slot ownership or balanced token
counts alone are insufficient.

Remove only the compiler-owned tail barrier when the query succeeds. Keep it
for any unproved lane/effect/path, and record the unresolved source/exit
obligation. Do not replace it with a new body-wide cut. Retain the existing
independent reimport/reconstruction and verify the final trial before commit.

**Files:** `LifecycleSynthesis.cpp`, existing completion domain/query headers,
and `PTOInsertSync.cpp` or its common verified-finalization helper. Keep tail
ownership explicit; do not infer it from absence of user metadata.

**Acceptance:** the unchanged GEMM is the positive target for PIPE_ALL 1→0,
conditional on proof. Negative tests retain it for an undrained final store,
an unrelated outstanding operation, and a skipped-drain branch. Include idle
core, zero-work, one-iteration, repeated-iteration, and scope-exit paths. If the
current completion domain cannot prove GEMM, retain the barrier and identify
the missing occurrence/exit fact rather than weakening the query.

## 3. Remove unnecessary M/FIX repairs using shared requirements and supply

**Problem:** the four M and one FIX sites remain after the L1/L0 improvement.
The accumulator recipe rejects a possible live-in path, even though useful
requirements and qualified MMAD relationships may remain available.

**Change:** retain the exact source/target access obligations for these five
sites before placement. At each repair decision distinguish: no remaining
hazard under the occurrence relation; target-qualified MMAD ordering; matching
generation readiness/reclamation; full physical completion; and unknown.
Check the existing R8 mixed-completion query before requesting a same-pipe cut.
Use the retained witnesses again against the actual emitted plan.

For accumulator flow, distinguish initialization, producer updates, publication
to FIX, store consumption, and overwrite for the next output. Preserve
zero-inner-trip behavior: a possible live-in value is not an excuse to discard
all nonempty-path facts, and a benchmark's positive K is not an IR precondition.
Where needed, express qualified synchronization participation using conditions
derived from existing loop bounds/guards, retaining conservative repairs on the
complement. No unmatched waits, new input promises, or payload duplication.

Introduce post-refinement conservatively: first admit deletion of proved
redundant residual barriers with lifecycle actions fixed. Then admit residual
event movement only with explicit ownership, unchanged participation, and fresh
combined proof. Do not simply remove the pass's early return and expose all
protocol endpoints to R5 movement.

**Files:** `InsertSyncAnalysis.cpp`, `MmadChainAnalysis.{h,cpp}` only if a
specific qualified query is missing, `LifecycleSynthesis.cpp`,
`StorageFrontierAnalysis.cpp`, and pass finalization/ownership plumbing.

**Acceptance:** each removed or guarded GEMM site has a retained obligation and
an independently rechecked discharge reason. Include initialize-versus-update,
zero/one/many inner trips, multiple output tiles, and an intervening incompatible
matrix operation. A negative test must show that MMAD ordering does not release
L0 operands or imply full M completion. Exercise lifecycle on/off crossed with
refinement on/off and verify the documented composition behavior.

## 4. Populate native occurrence relations from existing selectors

**Problem:** exact-slot discovery rejects multiple possible addresses, while
the richer local overlap API receives constant regions and `OrderedUnknown`.

**Change:** reuse `SlotAffineAnalysis` and `SyncSlotMapping` to normalize actual
`multi_tile_get` selectors and physically compatible views. Start with supported
affine/modulo selectors over one loop and known carried distances. Populate
`AccessSlice`/`SlotMap` and `OccurrenceRelation` at native queries. Preserve
loop scope, arithmetic range/no-wrap qualifications, aliasing, and byte extent.
Discovery and emitted-IR checking use the same normalization semantics, with
fresh extraction from their respective IR.

**Acceptance:** show the full native chain from original selector SSA to a
populated relation, changed dependency/placement decision, and checked output.
Test depths 2/3/5, same-iteration separation, wraparound reuse, and supported
distance-one dependencies. Add negatives for incompatible physical views,
unknown arithmetic bounds, and guards whose truth changes across iterations.
Preserve the existing three GEMM parity-spelling variants.

## 5. Add explicit loop transfers where occurrence import needs them

**Problem:** whole-graph fixed points and region boundary snapshots do not
provide substitution over arbitrary incoming generations or loop-carried
handle permutations.

**Change:** extend the shared flow result with a transfer for one physical
region: incoming content/ownership obligations, guarded outgoing alternatives,
exported reader/reuse obligations, and recurrence correspondence. Represent
zero-trip identity separately from nonempty execution and compose inner-loop
summaries without conflating dynamic invocations. Initially support structured
`scf.for` and bounded handle permutations recoverable from operands/yields.
Keep other cases conservative; do not invent a flattened iteration identity.

**Acceptance:** nested preload/consume loops, skipped inner work, decimated final
use, and a two-/three-slot loop-carried permutation use the transfer natively.
Compare summarized and unsummarized flow on small cases. A first reader from
one invocation must never acquire a later invocation's generation. At least one
previously rejected native case must produce a checked improved plan.

## 6. Combine compatible handoffs and improve allocation only where needed

**Problem:** dedicated whole-scope keys are conservative, and late failure can
trigger removal of an unrelated candidate. This is not the current GEMM
admission blocker: its six channels already fit.

**Change:** first combine handoffs whose producer completion, complete consumer
set, guards, recurrence, and release boundaries are compatible. Sharing must
not delay an independent consumer or release a member early. Then extend the
existing allocator with qualified logical lifetimes/interference, including
residual streams and hidden reservations. Preserve consume-before-rearm.

Return a concrete conflicting stream/key/obligation for retry; do not use a
broad predecessor cone as an exact dependency witness or call dedicated-key
failure unavoidable hardware scarcity. Keep independently reconstructed
verification after assignment.

**Acceptance:** a scarcity microkernel fits by proved sharing without new
PIPE_ALL or shortened overlap, while an overlapping-lifetime negative cannot
share. Retry removes a participant in the actual conflict. GEMM pair inventory
and placement are compared with manual, but 53 is not a mandated answer.

## Corpus coverage and device validation

Do not bundle general macro/import expansion into these first changes. Pick
one qualified operation summary at a time from the existing TopK/helper/TAXPY/
FFTS blockers, using the translator and target semantic contracts. Require an
unchanged affected fixture to reach a real optimization decision. Unknown
resource effects cannot become pure operations, and physical contexts remain
distinct.

For each implementation increment, run the focused native tests and the existing
11-fixture regression script with the matching runtime. Use incremental targets
with at most two aggregate local resource-intensive workers. A source-only
refactor has an explicit unchanged-output gate; an optimization claim requires
a committed output improvement or a precise, retained proof limitation. Record
sets, waits, named barriers by pipe, body PIPE_ALL, and exit PIPE_ALL separately.

Measure the current generation implementation on device before attributing any
performance effect to it. Once an optimization changes emitted code, compare
that candidate against the pinned current baseline and correct hand-tuned arm;
deduplicate byte-identical binaries. Reuse the device harness and correctness
cases. Distinguish synchronized batch=1 launch latency from batched per-launch
throughput, with device time alongside where available. The R8 GDN/KDA two-arm
rotation must be corrected before reusing its runner; a balanced targeted repeat
is sufficient. These device checks are validation gates, not evidence already
obtained by this plan.

## Supporting records

- [Current generation implementation](ptoas-insertsync-buffer-generations.md)
- [Analysis-utilization follow-up](../../test/experiments/insert_sync/performance/ANALYSIS_UTILIZATION_FOLLOWUP.md)
- [Local generation regression](../../test/experiments/insert_sync/performance/BUFFER_GENERATION_RESULTS.md)
- [R8 device results and timing-method findings](../../test/experiments/insert_sync/performance/R8_DEVICE_RESULTS.md)
