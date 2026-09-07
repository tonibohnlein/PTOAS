# InsertSync: structured accumulator chains and targeted MMAD ordering

Base: `c456adc12a6e04b5bef7caa5ffab4987de81557e` on
`tonibohnlein/PTOAS:codex/insertsync-revision-r1`.
ProtocolSync source examined: `5cd3cb24f562539e24a88d32cacf98c79842d29c`.
This package makes no remote repository changes.

## Decision

Reuse structure-aware facts, not the ProtocolSync pass or its admission policy.
Keep the current InsertSync translator, dependency scan, motion, allocation and
code generation. No new operation-completeness gate is introduced.

The immediate optimization is the large ordinary MMAD accumulation dependency.
This needs BOTH a qualified target rule and information about the preceding
matrix instruction on every feasible control-flow path. Neither a GEMM function
name nor the existence of a large ACC allocation supplies that proof.

The patch adapts ProtocolSync's separation of immutable facts, storage-specific
state, and completion, and its conservative structured transfer discipline.
It is not a literal cherry-pick of its dependency-bound types. ProtocolSync's
CompletionSupply remains straight-line; its bounded whole-path specialization
is not imported. No existing ReadyRelease matcher is claimed to implement this
new matrix-intrinsic rule.

## Useful reuse and port boundaries

| Existing component | Useful part | Treatment |
| --- | --- | --- |
| ProtocolSync LocalMemoryAnalysis | Required ordering is separate from generation identity and completed work; per-domain region transfers | Adapted in the small read-only predecessor analysis here |
| ProtocolSync StructuredSyncIR / region transfer | Choices merge alternatives; loop entry/bypass remains explicit; exact phase/physical context | Adapted as a compact CFG with symbolic backedges, not body unrolling |
| ProtocolSync CompletionSupply | A qualified completion relation has a source, target and occurrence relation | Preserve as next-stage interface; no claim this patch ports its full implementation |
| ProtocolSync first/last/nonempty loop recipes | Complete participation and bypass constructions for moved handoffs | Next targeted motion patch, not copied piecemeal here |
| ProtocolSync RecurringEventLifetime and complete-channel deletion | Check event rearm/progress as well as memory coverage | Keep for later allocation/deletion work; this patch generates no new recurring protocol |
| Existing native ReadyRelease | Proven slots, separate ready/free streams, complete prime/body/drain | Adapt after shared slot and region interfaces; do not emit flags then call unchanged InsertSync |
| Covering-performance ownership matchers | Bundled L0 operands, hierarchical L1 readers, alternating prefetch | Later semantic recognizers; do not import broad region endpoints or covering objectives |

A missing match means no optimization at that point, not compiler rejection.
All restored compatibility and prior WAW/overflow/slot/provenance fixes stay.

## Implemented algorithm

A read-only graph follows the original function's structured execution:

* Sequence connects frontiers in source order.
* A choice connects both arms and a join. Correlated predicates are deliberately
  overapproximated, never assumed to choose the same arm on different iterations.
* A `scf.for` has entry-to-header, header-to-body, body-to-header and header-to-exit
  edges. Zero iterations remain possible. Nested loops are represented directly.
* Physical Cube sections receive distinct context keys. Unsupported regions,
  macros, unqualified M instructions, and other ACC effects break the known chain.
* Known pure scalar/descriptor-construction instructions and ordinary work on
  other pipes do not turn issue order into completion.

Each program point has a three-kind fact:

```
unreachable
unknown or differing predecessors
one qualified (physical context, exact ACC allocation, valid M/N/K) identity
```

At a compatible MMAD, the outgoing fact becomes its identity. At a merge,
that identity survives only if every reachable predecessor supplies it. A
bounded worklist solves the finite lattice to a fixed point. Loops are not
sampled. Budget failure discards all optimization facts, not the function.

Only an in-place `tmatmul.acc` whose incoming fact equals its own identity is
eligible. Initializing `tmatmul` is NEVER itself elided by this rule. An
initialization can establish the predecessor identity for a later accumulation.

InsertSync still derives RAW/WAR/WAW requirements normally. For a same-pipe M
candidate, it may omit explicit repair only if every dependency pair describes
the same exact ACC operand and both endpoints satisfy the target/identity rule.
It does not omit a partially matching mixed dependency group.

Most important implementation constraint:

> An intrinsic accumulator relation never sets `alreadySync[PIPE_M]` and never
> supplies completion of the M lane or of its L0 operand reads.

Thus MTE1-to-M readiness, M-to-MTE1 operand reclamation, M-to-FIX publication,
FIX-to-M accumulator reuse, GM semantics and function-exit completion remain
ordinary requirements. The staged/combined choice remains independent.

## Why older same-ACC dependencies may be omitted at an eligible target

The universal predecessor fact provides a compatible immediately preceding
matrix operation P for each execution of target T. The target contract orders
P's in-place ACC update relative to T. Earlier conflicting ACC writers are
ordered relative to P by its own required dependencies: either unchanged
InsertSync repair or the same intrinsic rule. Induct over dynamic matrix
instructions. A reset/unknown predecessor cannot start an intrinsic-only chain;
its first subsequent matrix keeps normal dependency repair. The unknown entry
fact and zero-trip edges prevent a self-supporting circular proof in a loop.

This is a data-dependency argument, NOT an assertion that P fully completed
before T was issued. The implementation consequently must not reuse it to
publish L0 read completion or to remove a cross-pipe ACC protection.

The argument relies on the existing compiler satisfying the other retained
requirements. It is not a correctness certificate for all of legacy InsertSync.

## Qualified target subset

Opt-in: `--insert-sync-mmad-chains` (pass option `mmad-chains=true`). Default off.

The first implementation accepts only:

* explicit A2/A3 target and a known Cube physical context;
* ordinary bias-free `TMatmul`/in-place `TMatmulAcc`, unspecified AccPhase;
* direct, constant-address, suitably aligned planned LEFT/RIGHT/ACC allocations;
* f16 x f16 -> f32, standard LEFT/RIGHT/ACC layouts;
* full static valid dimensions; matching exact ACC identity and M/N/K;
* M,N,K aligned to 16 and in [16,4095]; `(M/16)*(N/16) >= 10`;
* no escaping/forwarded/mutably shaped descriptor in the participating handles.

BF16, GEMV, MX/sparse/bias modes, UnitFlag protocols, partial/symbolic valid
shapes, aliasing-but-different ACC handles and unsupported effects keep their
existing synchronization. This is an optimization domain, not an admission
policy. Generalizing each condition needs a target/IR argument and regression.

This contract is sourced from the official MMAD guide and corroborating CATLASS
implementation discussion; the online guide is a development preview. No A2 or
A3 silicon qualification was performed while creating this package.

## Expected relevance to the frozen GEMM

The recorded GEMM uses 128x256x64 ordinary f16 matrix steps on a common f32 ACC.
It therefore has relevant large accumulation chains. The 18 static PIPE_M sites
must still be traced individually: initializations and first accumulations with
an unproved predecessor do not qualify. The analysis deliberately does not use
`K-panel != 0` to infer a loop-entry generation, so some first-use cases remain.

No native new counts or recovered percentage are claimed here. In particular,
`mmad_dependency_groups_elided` is a diagnostic count, NOT a count of removed
barriers. Existing later motion/allocation can change the eventual output.

The target priority is separate PIPE_M improvement while preserving the true
MTE2 WAW correction. MTE1, MTE2, FIX and PIPE_ALL are not blanket-elided.

## Next implementation after native validation

1. Run flag-off/on against the same compiler and the unchanged historical GEMM.
   Compare pairs, each named pipe and PIPE_ALL independently. Confirm all
   non-sync payload/allocation replay hashes and count eligible/elided targets.
2. Run native negative tests, full check-pto, then the unchanged corpus. Record
   real admission and diagnostics, not a smaller specially admitted population.
3. Run numerical/progress stress on both supported device targets before enabling
   by default. Separately measure device-only timing; do not promise recovery of
   the reported 16–18% from a source rule.
4. Trace the three MTE2 and two MTE1 sites to their actual hazard witnesses. Port
   storage-specific final-use and next-overwrite summaries to show which are
   already ordered by mandatory release handoffs. Do not restore the tload/tload
   exemption and do not replace it with a rule based on a GEMM name.
5. Port complete first/last-use boundary lowering on one real panel pipeline.
   Release L1 after its final MTE1 read, not after unrelated M or output work;
   acquire at the first relevant user with explicit bypass and cleanup behavior.
6. Only then adapt bundled operand/persistent slot protocol construction. Keep
   event realization and scarcity serialization separately attributed.

## Evidence classes

A target-rule proof, a host dataflow test, a valid MLIR build, corpus admission,
correct numerical device execution and device timing are different results.
This package currently supplies the first two plus native regression inputs and
runners. Native adapter build/corpus/device results are explicitly NOT_RUN.

## Sources

* https://github.com/tonibohnlein/PTOAS/tree/c456adc12a6e04b5bef7caa5ffab4987de81557e
* https://github.com/tonibohnlein/PTOAS/blob/5cd3cb24f562539e24a88d32cacf98c79842d29c/include/PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h
* https://github.com/tonibohnlein/PTOAS/blob/5cd3cb24f562539e24a88d32cacf98c79842d29c/include/PTO/Transforms/ProtocolSync/CompletionSupply.h
* https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html
  Consulted 2026-09-07; page declares master `375727ac30e2` plus uncommitted
  modifications, a preview rather than a frozen commercial specification.
* https://catlass.readthedocs.io/en/latest/1_Practice/06_tile_development/
* https://github.com/hw-native-sys/pto-isa/blob/1216c55831fcc4ed2f096e4ca582ff75a633fbb6/include/pto/npu/a2a3/TMatmul.hpp

The original ProtocolSync documents' prohibition on evolving InsertSync is
superseded for this package by the subsequent explicit InsertSync revision
strategy. Their separation of storage, ordering, completion and finite events
remains useful; their old global covering and broad frontier policies do not.
