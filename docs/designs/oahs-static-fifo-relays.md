# Static FIFO slots and separate relay deadlines

Implemented on the `d1bf07ee5` working tree, 2026-09-21. This supersedes the
supplied-plan prototype in [the attention diagnosis](oahs-attention-bank-prefix-diagnosis.md).
It is part of ordinary `handoff` construction, with no new pass option.

## Physical-use view

The lowering-owned `SyncProtocolModel` now exposes the existing two-slot GM
envelope for unsplit ACC sends and MAT receives, alongside the VEC path.
The contract remains A3, pinned to PTO-ISA `0c112d61`, unit flags disabled,
bidirectional, positive bounded tile dimensions, and a tile fitting its slot.
Both directions address the same GM allocation. Each cursor starts at zero and
advances on its own participating operation.

`refineStaticSlots` propagates each cursor's two-bit possible-slot set over the
original control graph. It admits only a singleton identity for every static
send and receive, including backedges. Ambiguous participation declines. The
native wrapper additionally requires one entry-initialized handle, a private
imported root with one storage origin, and no other aliases or unrepresented
root accesses. The existing alternating-slot qualifier remains the fallback.

Admitted accesses use whole-slot conservative byte footprints. They gain no
definite-write assertion, local completion, peer completion or event credit.
Non-GM effects are unchanged. The original graph, payload operations, guards
and command observations are unchanged; no counter or period product is added.

The `Program::StaticFifoSlots` record is a shared physical-use view: cells and
participating send/receive operations. It is validated against the effects.
Storage analysis and the ordinary causal frontier consume the refined cells;
there is no second completion state.

## Relay construction

Direct transfers and ordinary routing outside this interface are unchanged.
For a non-common-cut two-hop transfer of a qualified slot's writer completion:

1. Keep the selected publication and final consumer deadline distinct.
2. Find the last intervening send of this cohort. A relay cannot be advanced
   across the issue of another bank's send. This is a placement lower bound,
   not a demand for that send's completion and not a later publication.
3. For each eligible intermediate engine, find its first command or payload
   after that bound. Forward before that word's unrelated activity; if there
   is no earlier activity, forward at the final consumer.
4. Compare actual prefix coverage beyond the source's fresh completion,
   excluding completion already acquired or required at the receiver. Also
   compare new prerequisites on crossed middle work and selected endpoints.
   Prefer a subset in both views; for equal sets prefer the later relay gap.
   Keep incomparable choices deterministic. Protect an already selected receipt
   to the same receiver from a later receipt prepended into its word; use the
   final deadline when that early gap would broaden the earlier receipt.
   See the [bounded correction and tests](oahs-relay-selection.md).
5. Bind both keys using actual empty/consumed state and neighboring selected
   uses at the exact source gaps. An early relay uses the incoming word state,
   not the broader post-word state.
6. Materialize the four endpoints privately, check the complete event protocol
   on the original graph, then commit those identical ordered endpoints.
   Failure leaves the ledger unchanged and ordinary binding remains available.
7. Replay actual transfers, refresh the residual, and retain final cold and
   native reconstruction checks.

The middle WAIT and following SET are inserted together at word start when
forwarding early. The source SET stays at its selected publication; the final
WAIT stays at the real consumer. This is neither global SET-before-WAIT
sorting nor movement of a completed plan. The staging check allows explicitly
pending payload obligations but rejects protocol/phase failures. It grants no
prospective rearming credit. The selector currently requires existing key
credit before probing; it does not search all helper-augmented bindings.

The two-view comparison is a placement heuristic, **not a general no-added-order
certificate or a least-order theorem**. It accounts for crossed middle gates
and existing outward publications, but does not certify arbitrary future edits
or complete open event interfaces. Complete payload-order comparisons remain
an explicit acceptance check for the tested native cases.

## Result on single-block attention

The first body receive reads slot 0, requiring QK2's FIX push. QK3's push writes
slot 1. Ordinary construction now selects:

```
after QK2 push:      publish FIX -> MTE1
before PV0 receive: acquire FIX -> MTE1; publish/acquire MTE1 -> MTE2

after QK3 push:      publish FIX -> M
before PV0 compute: acquire FIX -> M; publish M -> MTE2
before PV1 receive: acquire M -> MTE2
```

The second line no longer imports QK3. The later forwarding occurs before the
word's MTE1 readiness waits, preserving M's earlier prefix. Slot 1 still gets
its required FIX completion, and actual transfers carry consumption knowledge
for subsequent key publications. No private acknowledgment pairs are needed
for these two selected relays.

Slot precision also removes a pooled GM WAW requirement between disjoint
slot writes, eliminating its FIX fence. This explains why the implemented
plan removes more ordering than the earlier supplied-plan prototype. Other
payloads, queue operations and genuine requirements remain checked.

## Validation

The [linked regression](../../test/benchmarks/attention_relay/README.md) imports
both pinned prepared modules, runs the ordinary constructor, and compares its
words against the captured pre-change default. The reference is guarded by
input hashes and an exact original-graph signature. Candidate words are not
supplied by the test. Both modules also pass native emission/reconstruction.

Each input passes these independent issue/finish graph comparisons:

| Outer entries / inner lengths | Event pairs before → after | Removed relations | Added |
| --- | ---: | ---: | ---: |
| Empty | 2 → 2 | 0 | 0 |
| One / 3 | 103 → 109 | 80 | 0 |
| Two / 3,3 | 204 → 216 | 160 | 0 |
| One / 1 | 53 → 55 | 28 | 0 |
| One / 2 | 78 → 82 | 54 | 0 |
| Two / 1,2 | 129 → 135 | 82 | 0 |

The last three rows are checker-CFG stress cases; the actual native inner trip
count is three. The full six-entry explicit-prefix diagnostic removes 480
relations and adds none. All tests preserve payload identity and required
same-slot completion, and explicitly forbid QK3 completion before PV0's
receive and operand preparations.

Removing either early leg fails memory checking. Removing the second leg or
later receipt also breaks rearming in the independent graph, while events
remain balanced and acyclic. Postponing the middle forwarding to the later
receive stays safe but adds 82 relations versus baseline in the one-entry
trace. Safety and ordering quality are tested separately.

Portable slot tests cover independently counted cursors across phases and
varying entry lengths, skipped sends, unrepresented users, duplicate roles,
overflow, and repeated refinement. Five native cases cover the positive
ACC/MAT path and odd participation, undersized slots, split and non-unsplit
negative cases. The two native diagnostic executables and all 23 portable
suites pass. Targeted corpus controls retain identical plans for down, gate/up,
LM head, post-RMSNorm and partial attention; only the two single-block AIC
plans change. Their AIV companions remain unchanged. GEMM is byte-identical
to its saved control and passes the 200/394/782-pair trace checks with zero
named barriers.

## Work accounting and remaining boundary

Each admitted single-block AIC construction records two relay proposals,
1,318 candidate-analysis site evaluations and 1,434 relay-preparation site
visits. The graph stays at 239 sites. These counters are separate from selected
replay, recurring omission and final helper trials. A small slot domain is not
a claim of constant total construction cost: each staged protocol analysis
traverses the original graph, and placement scans are charged separately.

Evidence is under `../attention-native-work/`, including the probe, corpus
records, commands, graph comparisons and suite logs. There is **no new device
measurement**. Coupled AIC/AIV numerical, queue-progress and GM-visibility
qualification is still required before treating this as an attention speedup.
