# Constructor compatibility follow-up — 2026-09-22

## Diagnosis

The shared instruction translator successfully imported all 19 exact sweep
inputs. Six then failed in construction. These failures were not an opcode
admission problem and did not demonstrate hardware event-ID exhaustion.

| Inputs | Constructor problem | Correction |
| --- | --- | --- |
| `topk_select`, Qwen `attn_swpipe_aiv` (two inputs) | The hypothesis traversal accepted a payload before considering its complete refined-loop continuation; later replay found an outstanding access. | Use the existing contextual pending-effects evaluation from the start when original payloads or command words have multiple reachable analytical occurrences. |
| `mtp_linear` | The all-occurrence key query encountered unvisited copies of a shared word and rejected even virgin IDs. | Populate those states through the same contextual evaluation; retain the strict all-occurrence key test. |
| `kernel_softmax_prepare` | Shared first/repeated words were bound using one representative occurrence, without complete participation and next-publication support. | Check complete word participation and apply the existing explicit return mechanism to ordinary shared receipts, independently of the first-write experiment. |
| `hc_head_reduce` | The consumed key had different last-WAIT identities on alternative paths. F7 required a single WAIT and could not return their joined consumption knowledge. | Check and stage a return at the existing publication position, followed by the forward pair, using actual source-time state and the complete event protocol. |

A related hypothesis defect was isolated: a hypothetical preceding synchronous
access inherited invocation-root relationships through its scalar gate. Forget
that gate's outgoing consequences, as already done for its source prefix. A
synchronous access is completed at its own gate; a hypothetical one does not
thereby precede every other engine's launch. This changes construction-only
hypotheses, not the checker, instruction effects, or the target contract.

## Why the refined control needs full states

A first, repeated, or final copy is an analytical occurrence of an original
instruction. Copies may share a physical command word, or have distinct words
under original predicates. A key query and a completion query must see the
incoming state at all affected copies.

A reduced linked case is an outer loop with a vector write, an inner counted
read/write loop, and a vector read after the child. The old hypothesis traversal
loses the child's final outstanding write at the continuation. The ordinary
constructor succeeds when it uses its existing full original-control evaluator.
The regression uses abstract accesses; no native opcode or kernel name appears.

Full contextual evaluation issues unfinished payloads with their effects still
pending. It does not insert desired dependency edges. Finalized requirements are
rechecked after stabilization, and final cold checking and emitted-command
reconstruction remain mandatory. Unrefined single-occurrence programs retain the
existing cheaper traversal. The refined path can cost more; this change is a
compatibility repair, not a compile-time optimization.

## Returning consumption after alternative paths

Suppose either branch consumes event `P→Q, 0`, followed by a new P publication.
The token is empty, and Q's joined gate contains the actual consumption on each
path. P may still lack that knowledge. Select:

```
at the existing early publication position:
    SET  Q→P, r
    WAIT Q→P, r
    SET  P→Q, 0

at the original consumer deadline:
    WAIT P→Q, 0
```

The initial admission requires distinct publication and consumer cuts.
Common-cut repairs retain the existing closed-exchange path.
Admission requires must-empty forward occupancy, alternative actual consumption
identities, a supported forward interval, and independent reverse-key credit at
every source occurrence. Apply the reverse primitives privately to check that
they establish the forward key's consumption. Then check the exact complete
exchange, including later uses and repeated entries, before mutating the ledger.
The new forward receipt can itself rearm the reverse leg on the next visit, so
checking only the reverse half would reject a valid protocol.

This is a nonrecursive resource repair. It neither moves the forward publication
nor invents knowledge from emptiness. It can import the reverse engine's prefix,
so no general no-added-order or speedup claim is made. There is no single WAIT
anchor for this helper; it stays outside the single-anchor online discharge
mechanism. Existing final helper trials still require their normal cold check.
Separate `joinedAcknowledgments`, `acknowledgmentChecks`, and
`acknowledgmentCheckSites` counters account for this work.

## Allocation architecture

A physical key is a source pipeline, destination pipeline and event-ID tuple.
Storage analysis identifies obligations before key selection, but construction
currently integrates endpoint placement and physical binding. Reusing a key can
require a real return; that return can change completion carried by subsequent
publications. Allocation therefore affects causal ordering, not just numbering.

An unlimited-logical-event plan followed by allocation is a possible design, but
its allocator would need to preserve endpoint gaps or explicitly repair and
recheck them. These failures do not justify that redesign: they include rejection
of virgin keys through missing occurrence states, and consumed keys without a
supported return certificate. Fixing those queries is necessary in either design.

## Corpus changes and cost

All 88 modules construct and reconstruct. 84 output plans are byte-identical to
the saved default baseline. The four changed modules are
`pypto_lib__prefill_fwd__44` through `__47`; their AIC functions are unchanged,
and their AIV functions receive different receipt selection and explicit return
support. Each module has the same following static counts:

| Metric | Saved default | Compatibility repair |
| --- | ---: | ---: |
| SET instructions | 119 | 134 |
| WAIT instructions | 119 | 134 |
| Barriers | 89 | 89 |
| AIV replay site evaluations | 610,965 | 616,865 |
| AIC replay site evaluations | 102,810 | 102,810 |

These are static command counts, not executed event-pair counts. The additional
returns are conservative support selected during construction; their presence
does not establish that the old complete plans were unsafe. Complete payload-order
comparison and coupled device evaluation of these changed plans remain open.
There is no no-added-order or performance claim for this compatibility repair.
The recorded comparison is `constructor-corpus-changes.json` in the artifact
directory below.

## Validation and artifacts

Artifacts are under `../sweep-followup-work/` relative to the repository.

| Check | Result | Log |
| --- | --- | --- |
| Portable suites | 25/25 pass | `constructor-final-core-tests.log` |
| Native import/atomicity suite | Pass | `constructor-native-test.log` |
| Native selected-plan suite | Pass | `constructor-selected-test.log` |
| Exact sweep prepared inputs | 19/19 construct and reconstruct | `constructor-final-cases.log` |
| Existing native corpus | 88/88 construct and reconstruct; 84 identical | `constructor-final-corpus.log` |

The initial integrated candidate's exact-input result is also retained in
`joined-ack-cases.log`. Native suite diagnostics include expected rejection
mutations; suite exit status and its final success message determine acceptance.

Focused regressions cover the refined child continuation, an ordinary repeated
receipt without the experimental first-write option, synchronous loop
hypotheses, and branch-consumption rearming. The branch case checks both paths
with the independent finite graph oracle, preserves the early source before an
unrelated load, and rejects missing return support, an unavailable reverse key,
and an unconsumed branch.

The independent synthetic probe (`refined-loop-probe.cpp` / `.log`) ran the
production constructor with the old traversal forced, and with automatic
contextual evaluation. It records the old completion failures and successful new
construction. It is diagnosis scaffolding, not a second production planner.

These checks establish host construction and checking of synchronization plans.
They do not constitute complete PyPTO model builds or new device validation.
