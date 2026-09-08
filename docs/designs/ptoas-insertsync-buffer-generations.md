# InsertSync per-buffer generation analysis

The [shared-requirements follow-up](ptoas-insertsync-shared-requirements.md) records
new integration of direct repairs, protocol ownership and guarded GM refinement.

The experimental `--insert-sync-buffer-generations` option constructs local
ready/release protocols from physical buffer generations before ordinary
InsertSync repairs the remaining dependencies. It defaults to off and implies
the lifecycle construction path; `--insert-sync-lifecycle-synthesis` continues
to select the earlier constructor when used alone. The native pass option is
`buffer-generations=true`.

```sh
ptoas --enable-insert-sync --insert-sync-buffer-generations \
  --insert-sync-defer-same-pipe --insert-sync-mmad-chains \
  --insert-sync-gm-alias=assume-disjoint-arguments input.pto -o output.cpp
```

The GM alias option requires the existing caller contract. The generation
analysis introduces no input annotations, changed buffer addresses, or payload
reordering.

## Analysis and construction

`BufferGenerationAnalysis.h` consumes the existing guarded branch/loop graph
and translated physical accesses. `analyzeBufferGenerationFlow()` exports open
read associations, ordering and reader frontiers, guarded lane predecessors and
continuation summaries, independently of protocol acceptance. It reuses the R5
ordering/frontier analysis; both early construction and the R5 snapshot consume
this shared result. `analyzeBufferGenerations()` separately qualifies a complete
protocol, which the native adapter materializes.

1. **Forward reaching writes.** Whole-slot definitions kill previous reaching
   definitions for that slot; may-writes retain previous alternatives. The general
   native import uses may-writes unless a whole-slot recipe proves replacement.
   Branch joins retain alternatives. Backedges reach
   a finite fixed point without selecting a numerical trip count. A defining
   site denotes a family of dynamic generations, with each overwrite starting
   the next generation. Reads retain the latest reaching writer alternatives;
   producer read/modify/write operations consume the preceding definition and
   produce the next definition within the same ownership episode.
2. **Backward continuation summaries.** For each guarded occurrence, determine
   whether the next relevant access is a reader, overwrite, producer update,
   or scope exit. Other buffers do not invalidate these facts. A read can be
   final on one path and followed by further readers on another.
3. **Generation transitions through control flow.** Track free, filling,
   published, and acquired states. Retain a published preload through nested
   loops. Publish release after a proved final reader, or at a branch boundary
   when that is the earliest representable final-use boundary. A generation
   whose consumers are skipped returns its unused readiness credit at a
   boundary proving there are no remaining readers before overwrite or exit.
   No empty-reader pattern name or prescribed loop nesting is required.
4. **Events and composition.** Publish readiness after the final producer;
   ambiguous producer frontiers may defer publication to the first consumer.
   Acquire readiness before the first reader, and acquire returned permission
   before overwrite. Slots can share a handoff when all consumer sites require
   all members in the same pipeline domain and their combined generation
   transitions prove compatible. Residual event keys can be compacted using the
   complete logical plan before allocating dedicated lifecycle keys. Sharing
   preserves every endpoint, requires consume-before-rearm and verifies that
   each wait still consumes its original logical stream.

The current bundle cap is four members. Pipeline domains are MTE2/V and V/MTE3
on Vector, and MTE2/MTE1, MTE1/M, and M/FIX on Cube. Initial/final free credits
span the physical function or section scope. A one-shot slot stays with ordinary
insertion. Accumulator updates preserve internal producer dependencies; the
protocol does not grant a blanket M-completion or MMAD-ordering exemption.

## Checks and fallback

The existing native guard materializer proves each emitted condition against
all represented occurrences and checks SSA dominance. Source expressions such
as signed remainder, unsigned remainder, equality, and inequality use the
existing semantic guard domains rather than a GEMM-specific parity recognizer.

New protocol actions remain outside legacy motion and cleanup. After residual
allocation and emission, the compiler reconstructs the actual new events and
buffer actions, checks event consumption before reuse, checks combined residual
and protocol completion, verifies payload preservation and MLIR, and commits
the cloned function only after success. Reconstruction separately checks that
producer updates occur after initialization and before publication.

Unknown/partial physical overlaps, unsupported effect adapters, unresolved
history-dependent boundaries, resource exhaustion, or failed combined proofs
retain ordinary InsertSync. Reads that may reach the live-in/uninitialized
alternative currently prevent specialization of that channel; this is a
restriction of the optional constructor, not a production correctness verdict.

For the frozen GEMM, all four L1 slots and both L0 operand bundles now commit.
The A0 preload's readerless path is returned at the enclosing boundary, removing
the previous residual MTE2/MTE1 proof obstruction. The accumulator channel still
has a live-in alternative when the K loop executes zero times; the input does
not communicate a positive-K precondition to this analysis. Its dependencies
therefore remain with ordinary insertion. The small initialized-accumulator lit
fixture separately exercises successful M/FIX generation construction.

## Open facts, completion and native occurrence queries

Ordinary repair receives the shared flow even when no complete lifecycle fits.
It can discard an impossible ordered occurrence, or ask the existing qualified
MMAD rule about all immediate M-lane predecessors recovered through guarded
control. Neither answer establishes full M completion. New MMAD discharges are
retained and rechecked from freshly translated emitted IR. This removes the
remaining GEMM M repairs without inventing a positive-K precondition.

A separate completion query evaluates the emitted program with each proposed
barrier omitted. A named barrier can disappear only when its source prefix is
complete before the next operation/publication observing that lane. An automatic
exit PIPE_ALL can disappear only when every relevant issued phase is complete
at every reachable return and concrete event consumption remains proved. An
undrained store, skipped drain, stale token after reissue, or unsupported effect
retains the barrier. Input barriers retain their original ownership.

`SlotAffineAnalysis` normalizes supported existing selectors over one `scf.for`.
`SyncSlotMapping` populates byte extents, physical slot maps and same-instance or
known-distance relations. It checks actual physical intervals, arithmetic bounds
and loop correspondence. Legacy backedge records protect the entire finite
selector period: distance-one disjointness never discards wraparound reuse.
Selected dependency removals retain access/SSA witnesses and repeat the query
on fresh translated output. Dynamic event IDs do not prevent this storage-only
recheck; their event protocol remains with the existing insertion machinery.

Two-/three-slot handle permutations are recovered from `scf.for` initial values
and yielded block arguments. The translator records the physical addresses in
recurrence order. Unsupported yields retain unknown-range behavior, and loop
results include the initial handle for zero trips. The shared flow also uses
parameterized reaching-definition transfers for supported single-entry/exit
regions. Substitution preserves the incoming alternative across skipped writes;
unsummarizable regions retain the whole-graph fixed point. This is not a complete
parameterized ownership/reader transfer for arbitrary nested handle programs.

## Composition and remaining limits

Successful generation synthesis now runs completion cleanup and can invoke R4/R5
residual barrier refinement. Selected protocols and original operation identities
remain fixed. R5's general event-placement implementation commits a cloned body;
that would invalidate retained protocol witnesses. Therefore selected-protocol
composition permits proved residual barrier deletion and invariant overflow guards
around residual barriers, preserving payload and protocol operation identities.
Residual event movement still requires explicit identity rebinding before it can
compose. Ordinary insertion without selected protocols retains its existing R4/R5
placement behavior and receives the shared GM refinement when generations are enabled.

Key compaction uses concrete logical lifetimes and checks token ownership as well
as event causality. It currently shares residual keys; selected lifecycle keys
remain dedicated. Assignment failure records the candidate, directed domain and
occupied key mask, and retries that candidate. This is not a proof of unavoidable
hardware scarcity. Core tests demonstrate that sharing can make room for a
lifecycle; a native scarcity win is not yet established by the frozen benchmark.

The current slot constructor still requires exact selected slots/bundles. Native
modulo/permutation facts improve residual repair; they do not yet create every
dynamic-slot protocol. General queue/helper imports, arbitrary parameterized
region ownership and joint sharing of selected protocol keys remain outside the
supported domain. The independent event checker is unchanged in its acceptance
contract. No output counter or planner certificate is trusted as input evidence.

## Validation and benchmark entry points

Native tests include `insert_sync_buffer_generations.pto`,
`insert_sync_buffer_generation_accumulator.pto`, `insert_sync_generation_open_flow.pto`,
`insert_sync_generation_occurrences.pto` and `insert_sync_generation_completion.pto`. The first uses the unchanged
frozen GEMM, checks that the selected protocols commit, and compares three
semantically equivalent parity expressions. It requires zero named barriers
and zero PIPE_ALL for all three forms, then checks C++ emission.

The independent C++ core tests are in
`test/experiments/insert_sync/buffer_generations_test.cpp`. They cover loop
backedges, branch joins, uninitialized reads and updates, alternative final
readers, overwritten unread generations, complete/partial bundles, depths
one/two/three/five, reconstructed update actions, and exhausted budgets.

The existing performance runner accepts `--buffer-generations` and records it
in `results.json`; `--lifecycle-synthesis` selects the preceding constructor.
Both switches apply only to automatic arms. Input hashes, PTO/C++ compilation,
scalar payload replay, and separate mechanism accounting remain in that runner.

```sh
python test/experiments/insert_sync/performance/run.py \
  --python-root /path/to/matching/build/python --arch a3 \
  --mmad-chains --buffer-generations --output /disk/results/generations
```

Compilation and scalar replay do not establish device numerical correctness,
asynchronous progress, or wall-time improvement.

Latest validation and the complete 11-fixture campaign are recorded in the
[recorded mechanism table](../../test/experiments/insert_sync/performance/BUFFER_GENERATION_RESULTS.md).
