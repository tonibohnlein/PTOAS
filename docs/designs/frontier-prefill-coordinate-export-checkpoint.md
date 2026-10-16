# Completed opt-in prefill coordinate export

The prepared causal-convolution prefill now completes its experimental typed
coordinate export. The frozen 221-input optimizer finished in 142.016 seconds
under one CPU, 4 GiB address space and a 300-second watchdog. The preceding
300-second probe was unfinished in lifetime extrema. This is completion evidence,
not a paired speedup or a physical allocation. Unary extrema implementation is
committed in `8b319e9f1`; the tested snapshot also contains the separately pending
Delta construction candidate. Its complete frozen manifest and tool hashes are
retained with the receipts.

The completed output reports `allocation_support = "available"`. Its finite
regional-palettes v2 certificate has 55 groups: 39 with budget one and 16 with
budget two, represented by 71 lane rows. All 16 arithmetic families passed their
coordinate reuse proofs; all 32 unary extrema qualified and were selected.
Projection, difference, extrema, containment and publication work remain separate
profile scopes. Difference construction still dominates this allocation probe;
completion does not establish that its cost is satisfactory for default use.
The default single-ID route remains unchanged and the wider candidate stays
behind `experimental-coordinate-allocation=true`.

## Export premises and independent checks

The source audit checks original-source-IV transport at publication and selected
partner-source coordinates at acquisition. Coordinate families have no alternative
publication or cleanup choices; the 25 cleanup-choice families are disjoint from
them. Parent sharing derives from qualified last-target to first-source reachability;
terminal cleanup cannot establish outgoing reuse. Exact original loop/guard domains,
a representable counted ordinal and the positive unit step retain finite invocation
bounds. Existing independent arithmetic and parent-composition regression controls
remain required; no new direct-handoff theorem is imposed on this protocol.

`test/lit/pto/Inputs/check_prefill_coordinate_artifact.py` executes 24 serial trace
calls across lengths 0, 3, 4 and 12, with both initial-state values. It compares
all original payloads and the entire default and coordinate logical protocols,
checks matching identities, original source-IV transport and same-slot causal
reuse, and rejects four damaged evidence cases. All eight cases passed. Length
zero is only a structural zero-trip control: original final state-update accesses
would need a padding premise. Positive cases assume valid original storage and
the recorded scalar valuations.

All 16 coordinate records were observed. Same-slot reuse was exercised only for
records 48, 61, 74 and 87; the other 12 records retain symbolic proof coverage.
**Parent-sharing, lane-sharing and lane-successor trace check counts are all zero.**
The receipt lists 135 unobserved parent pairs and 155 unobserved lane pairs/edges.
Those obligations remain covered by the reviewed symbolic producer proofs, not
by these traces. Observed source-IV checks use the latest original payload on the
same source pipe; all-input source-cut identity also remains a producer obligation.

The permanent fixed-protocol checker independently replayed the completed artifact:
97 original payloads and 124 commands are unchanged for length four/initial one.
Its 20 mutually non-reusable MTE2-to-vector handoffs retain the
[scoped capacity result](frontier-prefill-fixed-protocol-capacity.md).
The six-ID split allocator still reports an uncertified regional assignment; that
sufficient-placement refusal alone is not the lower-bound proof. An additional
64-ID probe was rejected as an invalid pool and establishes no placement result.
No successful physical assignment, device compilation or execution is asserted.

## Pinned receipts and reproduction

All paths below are workspace campaign evidence; hashes identify exact contents.
The frozen source manifest is
`.local/milestone-loop/retry-lifecycle-audit-gate/source-manifest.json` and its
retained optimizer/tracer hashes are in `tool-manifest.json` in that directory.
The eight-case command is stored as an exact argument vector in
`.local/milestone-loop/prefill-coordinate-artifact-check/command.json`.
Reproduction first generates the original prepared input with
`--pto-frontier-analysis=profile-stages=true gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5 experimental-coordinate-allocation=true`,
then invokes the checker with original, default logical, coordinate logical and
retained tracer paths. This audit requires no native rebuild.

| Receipt | SHA256 |
|---|---|
| `.local/milestone-loop/prefill-unary-export/result.json` | `ab0f9640b5598558a92d9c66c4dfcf167775e037a14ab528fbce5d28fa0d2ae3` |
| `.local/milestone-loop/prefill-unary-export/coordinate.out` | `f6e1cc81b14cb98d7d126ffc6200e221b230a7fcad38ddbe45df4ab8b1de91c1` |
| `.local/milestone-loop/prefill-unary-export/evidence-manifest.json` | `197d9b0a45893c0bd8922c04526af35f1f44362e1d758e70a79afc27bb17d4a1` |
| `.local/milestone-loop/prefill-coordinate-artifact-check/execution-receipt.json` | `cd5b3d896ff1c121ac9bb87705004886b2a5b9868a3885af7adc8a440e66cd4c` |
| `.local/milestone-loop/prefill-coordinate-artifact-check/run/receipt.json` | `bac03deb2ae34bf7836a1171aa81b88ef503d4d5b7b35c9d1b6111e2b0d4b5d7` |
| `.local/milestone-loop/prefill-coordinate-artifact-check/source-manifest.json` | `215e14c3e194738fb6031035b1d980abc26fce225e74e4bc231f6af4ffc95b9c` |
