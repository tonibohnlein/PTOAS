# Qualification of `final-block-pair-v1` (revision 0.9)

## Status and boundaries

This is **reference-contract qualification**, accompanied by an implemented phase
adapter, an endpoint-complete order monitor, and finite inductive certificates.
It is not device qualification of Ascend UnitFlag and not a claim that the SDK
lowering of an arbitrary MMAD/FIXPIPE instruction implements every premise below.
The concrete profile is supplied; the checker does not enable UnitFlag, select
modes, alter layouts, insert code inside operations, or synthesize phase protocols.

The old exact-cell core remains unchanged. This module realizes one narrow
instance of the conditional phase-interface contract in revision 0.8.

## Sources and what they actually establish

* Huawei's Ascend C development UnitFlag page, accessed 2026-09-15, describes a
  flag per 512-byte L0C block, final mode 3, producer writes with writable
  permission, and consumer reads with readable permission. It describes final
  transitions in both directions, coupled enablement, matching coverage, and
  traversal concerns. The page identifies master `ac856984c266` **with uncommitted
  modifications**. Its revision marker is not an immutable release qualification.
  URL: https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute_key_features/UnitFlag.html
* The versioned 8.3.RC1 high-level best-practice page is retained as historical
  block-overlap evidence from v0.8. In this revision's direct retrieval, the body
  was not available as substantive parsed text. It is not newly audited evidence.
  URL: https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_10003.html
* Versioned 8.2.RC1 and 8.5.0 Fixpipe API search excerpts show target/form and mode
  restrictions. Direct full-page retrieval failed or returned navigation. They
  are **not** used to claim a release-pinned full operational contract.

No numerical throughput or latency assumption is used.

## Admission contract

1. A finite, exact set of pairwise-disjoint physical cells is supplied. Protected
   cells have ACC identity, are exactly 512 bytes and 512-byte aligned. Unknown
   aliases, partial cells and duplicate maps are rejected, not rounded away.
2. Each resource group has one fixed producer engine and one fixed consumer
   engine. Every admitted producer/consumer visits the complete identical ordered
   block list. The ordinary-effect vocabulary cannot access protected cells.
3. The producer is a supplied **final non-accumulating** writer with complete LEFT
   and RIGHT read effects. Those input reads are conservatively live over the
   whole producer operation. There is no KEEP chain or accumulator-input read.
4. The consumer is a supplied final identity copy with one exact 512-byte GM
   output cell per input block. No quantization, partial copy, NZ2ND/ChannelMerge,
   arbitrary shape inference, reset, unknown private events or mode transition.
   `exact-block-map-v1` asserts a checked fixture-level map, not an SDK shape proof.
5. Final-producer / final-consumer requests alternate in the common reference
   traversal. Dynamic resource service is in each participant's per-block request
   order. Waiting phases have eventual progress when their predecessors complete.
   The **ordered-service premise is additional**: two flag values alone do not
   identify the intended producer generation. Native qualification remains open.
6. Initial writable permission and its causal entry anchor are supplied. Every
   terminating path consumes every final produced block and every software event.
   Writable/empty exit bits are not a reset of causal history. Loops and joins
   retain the complete interface.
7. Command submission and all operations obey the supplied bounded forward
   fragment contract. No finite command-queue-capacity guarantee is claimed.

Items 1--4 and the syntactic part of 5 are checked on the model input. The graph
checker checks the all-path resource balance, hazards and software-event reuse.
The hardware realization of items 3--5 and native progress is a deployment proof
obligation; the API documentation alone is not advertised as discharging it.

## Operation fragments

A producer has issue I, finish C, and each result block's write begin/end b_j,e_j:

```
I -> b_j -> e_j -> C
LEFT/RIGHT reads span I ... C
```

A consumer has I,C, result-block read b_j,e_j and GM-write begin/end o_jb,o_je:

```
I -> b_j -> e_j -> C
I -> o_jb -> o_je -> C
e_j -> o_je
```

This last end-to-end dataflow edge is a supplied identity-copy contract: a block's
output cannot be fully written before that block has been captured. It does NOT
require the entire read to finish before the output write begins. Different
blocks are not serialized by the adapter. These are analytical endpoints, not
legal locations for emitted SET/WAIT commands.

Resource edges are only:

```
previous read end of u -> next write begin of u
current write end of u -> current read begin of u
```

Their source is the **current permission anchor**, replaced at each generation.
They do not make the issuing engine's launch gate equal to its completion prefix.

## Independent operational meaning

For each block, use writable -> writing -> readable -> reading -> writable.
Begin/end events acquire/change these states; phase requests of the same role are
served in order. This is a semantic sequencer, not a claim that all four states
are hardware-visible bits. For the admitted request word, its possible schedules
are exactly the linear extensions of the resource-edge graph (subject to the
same other native edges). The draft gives the induction.

`operational_checks.py` compares reachable prefixes for three finite instances.
It also exhibits the necessity of service order: if P1 overtakes P0 while the bit
is writable, C0 can read generation 1 even though the flag sequence is legal.
That example is **not** an alleged hardware bug.

## Accepted combinations and limits

Ordinary SET/WAIT and fences use the original nonblocking-publication semantics
and share the same causal interface with the permission ports. A native block
path can legitimately carry event-consumption knowledge, but only through an
actual path. A ready result block does not imply that LEFT/RIGHT reads have ended,
that the matrix command finished, or that GM output is complete or visible.

The model checks completion ordering for declared coherent cells. Cache policy,
remote publication, atomics and externally visible GM visibility are not modeled.

The automatic v0.7/v0.8 SMT constructor is NOT upgraded in this package. The new
checker validates supplied phase-aware schemas and fixed points. Exact checking
is relative to the declared phase contract, not a hardware throughput optimum.
