# Experimental handoff integration

The first integration stage provides the diagnostic flag
`--insert-sync-handoff-facts-dir=DIR` (empty/off by default). It captures
qualified pass-entry physical/control facts on a clone before synchronization
selection. Export does not change the generated synchronization. Unsupported
projection or bypass produces an explicit record; an output I/O failure is a
compilation error. The equivalent function-pass option is `handoff-facts-dir`.

The reference uses a conservative local-memory projection. Global-memory and
ACC resource obligations remain native; a passing reference comparison is not
a native transformation or device correctness proof. No new input annotations,
dialect operations, alias promises, or Python IR builders are introduced. The
existing Python CLI forwards the same native flag.

See [native bridge contracts, results and reproduction](../../test/experiments/insert_sync/event_model/NATIVE_BRIDGE.md).

## Native planning experiment

`--insert-sync-handoff-planning` opts into the native engine after the current
InsertSync construction supplies a feasible seed. Both ordinary repair and the
selected lifecycle path reach this engine. Authored flags still bypass the
production pass. The old straight-line experiment now calls the same engine.

The native planner retains the initial independently extracted requirements,
including conservative GM and cross-lane ACC resource ordering. It reads
backward first-demand deadlines and chooses advancing source prefixes for
handoffs within one original block invocation. It can advance a publication,
delay an acquisition, or split a broad handoff with a fresh finite event key.
The complete structured trial is checked across branches, loop recurrence,
zero trips and exits. A block-local proposal never substitutes an invented
iteration distance for that proof.

Complete concrete key families can then be removed if the remaining combined
plan still supplies every requirement and valid event participation. All
participants must be owned. A consecutive set/wait episode in one uninterrupted
block can also be removed while retaining the key's other episodes. Both
endpoints execute together; the remaining key word and its ownership survive.
Logical actions retain operation identities;
closing over concrete key participants is a realization restriction, not the
identity of a logical handoff. Barriers supplied by the caller can be considered
by the engine; production currently keeps them fixed because ownership may
include authored barriers.

This first native subset preserves existing keys or reserves a fresh directed
key over the whole scope. It does not implement arbitrary symbolic AST event
lowering or optimal joint coloring. Failed allocation/proof/budget leaves the
last proved plan in place. It introduces no scarcity drain or acknowledgement.
The reference remains separate; the compiler does not spawn Python or depend
on a system libisl runtime.

The planner has one eight-million-work-unit budget and at most 64 trial
rewrites. These bound analysis work rather than promise a wall-clock limit.
Internal analysis or IR-verification errors fail compilation; unsupported
semantics, unproved ordering and exhausted resources retain a proved plan.
Native issue identities are compared by guarded occurrences, independently of
the order in which the graph builder discovers its partition nodes.

The old private straight-line `LinearHandoffNeeds` engine is removed. The test
adapter and production option now use the same requirements, construction and
reconstruction path. The current implementation proposes block-local cuts
inside supported structured executions; it does not synthesize arbitrary
cross-region cuts or perform joint allocation of all logical streams.
Functions with dynamic event IDs are retained unchanged: their selector and
resource recurrence must be imported before a new static key can be assigned.

## Stage 2 native acceptance evidence

The incremental `PTOASCompiler`/`pto-test-opt` build and the two focused
`insert_sync_handoff` lit tests pass. The latter includes splitting, required
additional readers, full finite-key pools, invalid seeds/contracts, dynamic and
high-level resource refusal, zero/nonzero/skipped loops, and physical sections
with no terminator. Both the ordinary and lifecycle production paths invoke
the option. Default-off output matches commit 1 byte-for-byte on five retained
fixtures (one buffer, historical GEMM, QK, Q projection and online softmax).

The unchanged eight Qwen inputs and historical GEMM compile in both arms.
All nine retain identical payload, ABI, view/allocation and scalar-control
projections. The table counts static mechanisms independently:

| Fixture | Seed sets/waits | Experiment sets/waits | Named barriers (both) | PIPE_ALL (both) |
| --- | ---: | ---: | --- | ---: |
| rmsnorm | 40/40 | 40/40 | V=29, MTE3=3 | 1 |
| softmax | 15/15 | 15/15 | V=14 | 0 |
| online_softmax | 20/20 | 12/12 | V=20 | 1 |
| silu | 3/3 | 3/3 | V=6 | 1 |
| rope_kv_cache | 15/15 | 15/15 | V=12, MTE3=2 | 0 |
| qk_matmul | 22/21 | 22/21 | M=2 | 0 |
| sv_matmul | 25/25 | 25/25 | M=2 | 0 |
| q_proj | 39/39 | 29/29 | M=6 | 1 |
| historical_gemm | 56/56 | 56/56 | 0 | 0 |

Online softmax executes 188 → 96 set/wait pairs at bound 16; at bounds
0/1/2 the comparisons are 8→6, 8→6, and 20→12. Every replay retains
identical payload hashes, scalar counts and barrier counts. This is native
command reduction on a looping kernel, distinct from the structured split
fixture's earlier readiness demonstration. Historical GEMM retains its 56-pair
barrier-free seed because the stronger experimental seed proof is unavailable
(phase 26→56); its preservation is fallback, not newly proved optimization.

Serial CLI+inspection durations on the final nine-row sample range from
0.41–2.35 seconds without the option and 0.43–1.82 seconds with it.
Online softmax is 0.530→0.599 seconds and Q projection 0.512→0.575 seconds.
These single observations include process/translation/inspection costs, not
statistically established compiler speed ratios. No device run was performed.

Final native library SHA-256: `10a5278e92f3f3f4d9408a6bc1de28fa6c50787c9fade25d3416bf858f35b419`.
Raw commands/IR/metrics and the focused verification records are under
`insertsync-builds/campaign/handoff-native-0387/commit2/native-examples-final`
and `commit2/final-focused-accepted`. Earlier failed development attempts
remain separate; their physical-section restoration bug is fixed and covered.
