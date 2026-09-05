# ProtocolSync single-loop occurrence foundation

## Status and scope

This is step 2A of the structured-repair continuation, based on `e433b74fa`.
It adds opt-in canonical local-memory requirements for one unconditional
`scf.for` with a straight-line body and straight-line prefix/suffix. It does
not yet enable ordinary-loop synchronization emission or increase native kernel
admission. The subsequent [acknowledged loop-frontier repair](ptoas-protocol-sync-loop-frontier-repair.md)
adds a low-level recurring recipe and independent concrete cycle verification;
boundary composition and integration with F remain required for native emission.

The supported analysis subset is one function block, one positive-constant-step
loop without iter_args, ordinary vector-core phases, and independently recovered
bounded UB allocation footprints. Prefix, body and suffix share one physical
atom partition. Choices, nesting, physical sections, dynamic/subview/modulo
addresses, and unmodeled effects remain unsupported. Allocation footprints are
conservative effect bounds, not exact instruction byte sets or definite kills.
GM aliasing and visibility, communication, queues, and ACC/proxy effects are
outside this local-memory analysis; it does not discharge their obligations.

`SyncLocalFlowOptions::analyzeSingleLoop` requests the analysis. It returns
`Complete`, `Unsupported`, or `LimitExceeded` separately from internal failure.
Default production analysis remains unchanged. All loop diagnostics retain an
explicit boundary and leave `coveredAccesses` empty. No timeline rejection is
removed, no planner candidate is selected, and no new event reuse is authorized.

## Dynamic relations in the canonical requirement record

All records use `SyncResidualObligation`: stable local requirement identity,
source/target access and phase IDs, shared atom memberships, required completion
or reclamation, conservative precision, and a typed iteration relation.

| Relation | Meaning |
| --- | --- |
| SameIteration | Two body phases in iteration `i`, or ordinary outer phases |
| LoopCarried | Source in `i` precedes target in `i+1`, with explicit carrier |
| LoopEntry | Prefix source precedes the target in every executing iteration |
| LoopExit | Source in every executing iteration precedes the suffix target |
| LoopBypass | Prefix source precedes suffix target on the zero-trip path |

The boundary quantifiers are deliberately conservative. In particular,
`LoopExit` must not be interpreted as only the final iteration: read-only atoms
can retain outstanding readers from every iteration. `LoopEntry` and `LoopExit`
instantiate no body accesses when the trip count is zero. The empty lexical
guard (`MustExecute`) does not assert that the loop executes at least once.

A static phase may be both endpoints of a valid carried requirement. For
example, `load(i)` and `load(i+1)` writing the same physical UB bytes require
WAW completion despite sharing an operation and access ID. The old direct
verifier still rejects such a repair; it is not bypassed by this analysis.

## Sparse transfer and its invariants

Per atom, retain an outstanding writer and outstanding readers, each annotated
with prefix/body/suffix position and dynamic iteration instance. Coalesce reads
and writes from one physical phase before updating that frontier.

1. A read requires completion of the outstanding writer, then becomes an
   outstanding reader.
2. A write requires completion of the preceding writer and reclamation from
   every outstanding reader, then replaces the discovery frontier.
3. Replacement does not prove a semantic contents kill or physical completion:
   the earlier effects remain ordered through mandatory requirement links.
4. Transfer the prefix once. Analyze two body instances to expose same-iteration
   and adjacent-iteration frontier links. Analyze the suffix from the outgoing
   frontier and separately from the incoming frontier for zero trips.
5. Deduplicate by requirement kind, access endpoints and occurrence relation;
   merge atom memberships exactly once. Signal/wait placement never participates
   in these identities.

For an atom written in the body, its outgoing writer/readers stabilize after
one body traversal up to iteration renaming. Older effects remain connected
through the next body's first relevant write. For a read-only body atom,
readers do not stabilize as a finite set of dynamic instances; boundary
requirements quantify over all iterations instead. This explains why merely
copying the final unrolled state would lose obligations.

These are invariants of requirement discovery, not an event-consumption proof.
Two transfer traversals do not certify an arbitrary emitted loop, its deadlock
freedom, or hardware visibility. The optional requirement-membership budget is
transactional: exhaustion exposes neither partial requirements nor coverage.
It is not a bound on all storage-partition or compiler work.

## Independent bounded oracle

The host test constructs small programs with prefix effects, a three-operation
body, and suffix effects. It expands concrete operation occurrences in lexical
order for selected trip counts, independently computes physical overlap from
the fixtures' byte sets, and enumerates every RAW/WAR/WAW pair across all
expanded occurrences. It then instantiates canonical requirement relations and
checks transitive coverage. It adds no intrinsic same-pipe ordering edges.

The matrix spans all 81 four-position R/W/RW combinations, partial overlap
between `[0,512)` and `[256,768)`, and trips 0, 1, 2, 3, 4, 7, 8 and 11:
648 traces. Additional checks remove entry, exit, backedge and zero-trip
relations; exercise same-static-phase WAW recurrence; force budget exhaustion;
and reject choices, nested loops and dynamic addressed storage. The production
path must remain non-admitting for these loop inputs.

The oracle shares semantic extraction but not production atom construction or
the sparse frontier algorithm. It establishes bounded coverage within the
declared local footprint universe, not the completeness of operation semantics,
general iteration analysis, or target hardware correctness.

Coverage is checked at phase-occurrence granularity. The oracle does not
validate each requirement's access IDs, atom memberships, or obligation kind;
corrupting those while preserving phase endpoints and iteration relations can
leave this check passing. Separate record-consistency checks and provenance
mutations are required before these requirements authorize selection or emission.

## Next: native recurring repair and concrete verification

Before setting loop accesses covered in production:

- Establish a legal recurring completion/reclamation recipe from canonical
  requirements, including prefix and suffix frontiers and zero-trip bypass.
- Prove consumption before rearming for arbitrary trips. A capacity-one control
  handshake is not a capacity-one storage assertion.
- Independently reconstruct actual sets, waits, guards and carrier relations
  from staged IR; verify local hazards and event lifetimes, not planner tags.
- Make F consume the new boundary relations explicitly. Its current graph and
  direct verifier intentionally do not accept them as selected completion supply.
- Preserve both GM contracts and fail closed on unqualified visibility or other
  effect domains. Keep the existing mandatory exit drain separate from body
  synchronization; do not add a generic body `PIPE_ALL` escape hatch.

## Validation

Validated against baseline `e433b74fabcdc0b4d9c08274156bbab26c3d0fcf` plus this
working-tree milestone, using the existing LLVM/MLIR 19.1.7 build:

```bash
cmake --build build --parallel 2 --target \
  pto-protocol-sync-loop-memory-test PTOASCompiler pto-test-opt \
  pto-protocol-sync-local-memory-test pto-protocol-sync-scoreboard-test \
  pto-protocol-sync-one-shot-test pto-protocol-sync-ready-release-test \
  pto-protocol-sync-direct-repair-test pto-protocol-sync-mixed-test
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python \
  /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit \
  -v -j 2 build/test/lit --filter protocol_sync \
  -o build/protocol-sync-loop-occurrences-lit-results.json
```

The targeted build succeeded. The three focused loop-memory, local-memory and
scoreboard tests passed; the full ProtocolSync selection passed **42/42** tests
in 56.57 seconds. The new loop oracle checks 81 programs across 648 traces,
alongside the boundary-removal, self-phase, unsupported-input and budget tests.
The changed-code compliance check reported 10 code files, zero errors and zero
warnings; `git diff --check` passed. These are host-analysis regression results.

No device tests, performance measurements, or corpus admission results are
claimed. The analysis remains opt-in and does not enable native loop emission.

Independent algorithm and compiler-integration reviews accepted this
diagnostic-only milestone with no blocking findings. Both identified the
phase-level oracle limitation documented above. Their acceptance does not
qualify recurring emission, arbitrary-trip event lifetimes, or hardware behavior.
