# Prefill fixed-protocol capacity witness

The existing prefill logical protocol needs at least **20 directed MTE2→V event
IDs** for one admitted four-token invocation. This is a checked lower bound for
the fixed logical commands, matching and legal cuts. It is not a hardware lower
bound, a bound for another synchronization realization, or a successful allocation
export. The typed coordinate-allocation investigation remains open.

## Invocation and preserved execution

The original prepared source is
`test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto`,
SHA256 `4a2da46ad42fbded8d15dd4b64cdabc93ae0df9570ea6d5d165712361ce16831`.
The invocation fixes block index zero, sequence boundaries `[0,4]`, cache index
zero and initial-state flag one. Valid initialized original storage and the
existing nonalias premise are assumed. Four tokens keep the final reads at
`end-1`, `end-2` and `end-3` within the declared input. Both original integer flag
tiles have sole `texpands(1)` writers, so their three observed reads are one.
This is not an independent proof of all original payload-memory feasibility.

The binder retains each observed load, block-index query and flag read, renaming
its result and supplying the chosen scalar value to its uses. It distinguishes
the two sequence-boundary loads by their original SSA index definitions: block
index and block index plus one. Every original PTO operation stays in order.
No allocation or compiler optimization runs on the specialized replay artifact;
its inherited allocation metadata, if any, must not be used for lowering.

Separate structured traces of the original kernel and concrete logical output
have exactly the same **97 ordered payload events**, including original phase
types, coordinates and pipes. The logical trace adds **124 commands**. The saved
fixture contains both complete reports, the source hash, scalar bindings, full
expected capacity certificate and provenance hashes.

## Command-graph proof

The independent oracle in `check_fixed_protocol_capacity.py` constructs payload
start/completion and command vertices, native pipe order, matching SET→WAIT edges,
and existing barrier effects. Textual SET-to-WAIT order supplies no automatic
causal edge. Two handoffs can share an ID only when consumption of the earlier
handoff causally precedes publication of the later handoff.

For direction `[4,1]` (`PIPE_MTE2`→`PIPE_V`), all 20 matching handoffs have **zero
legal reuse edges**. They form a 20-element antichain, exceeding the eligible pool
of six. The oracle checks unique matching identities and uses bipartite matching
and an antichain certificate for each directed pair. Independent correctness
review also reconstructed the graph with a separate DFS implementation, without
importing the capacity helper, and confirmed the 20-element witness.

## Reproduction and status

`test/lit/pto/Inputs/prefill_fixed_protocol.json` pins the original and logical
reports. `test/lit/pto/Inputs/check_prefill_protocol_capacity.py` checks their exact
payload equality and complete certificate, including damaged-payload, unmatched
and duplicate-endpoint controls. The existing
`test/lit/pto/sync_fixed_protocol_capacity.pto` invokes this cheap certificate
check alongside decode and Delta.

For a fresh concrete logical output, pass `--logical PATH --tracer PATH` to the
checker. It replays both original and logical bodies serially, then requires the
same complete protocol and capacity result. Only the four known scalar-work
counters may differ; all ordered events and other trace fields remain exact.
Temporary replay files stay beside the logical artifact on disk.

The original source-pinned replay passed under the parent's bounded runner;
its raw reports and receipt are in
`.local/milestone-loop/prefill-fixed-protocol/run-n4-initial1/`. The permanent test
passed its focused certificate check and a fresh original/logical replay using
the permanent helper. All five protocol/expectation gate tests passed; receipts
are in `.local/milestone-loop/protocol-expectation-gate/`. No
kernel compilation or device result follows from this witness. It closes this
scoped fixed-protocol capacity question while leaving allocation implementation
and other realizations separate.

The current opt-in allocation-construction probe reached lifetime-extrema work
before its 300-second watchdog: the sampled stack passes through
`Proof::extremum`, subtraction and `isSubsetOf`. The optimized sample does not
identify the current family or completed family count. The older four successful
two-ID family proofs and `11→56` composition stop remain historical evidence;
they do not describe this new sampled stopping stage. Neither timeout establishes
impossibility or resolves the typed export.
