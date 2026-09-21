# Attention bank prefix diagnosis on the current plan

Local inspection on `d1bf07ee5`, 2026-09-21, alongside the opt-in reused-key gap
extension. This page records the initial diagnosis and supplied-plan prototype.
The subsequent [native implementation](oahs-static-fifo-relays.md) now imports
slot identity and selects the separate receipt deadlines automatically. No new
queue/peer contract is claimed.
The manual-attention device transcription remains separate, unfinished evidence.

## What is already precise

The prepared single-block AIC inputs `prefill_fwd__48` and `__49` have this
matrix schedule per outer entry:

```
QK0 QK1
QK2 QK3 PV0 PV1
QK4 QK5 PV2 PV3
QK6 QK7 PV4 PV5
PV6 PV7
```

Their physical use sequence is not the semantic probability-message sequence:

| Preparation | Previous LEFT/RIGHT reader | LEFT interval | RIGHT interval |
| --- | --- | --- | --- |
| PV0 | QK2 | [0,4096) | [0,32768) |
| PV1 | QK3 | [4096,8192) | [32768,65536) |
| QK4 | PV0 | [0,4096) | [0,32768) |
| PV6 | PV4 | [0,4096) | [0,32768) |
| PV7 | PV5 | [4096,8192) | [32768,65536) |

The current bank-zero M -> MTE1 release is **already published after QK2 and
before QK3**. Its consumption precedes the LEFT overwrite for PV0 and also
protects the later RIGHT overwrite. The return itself carries QK0..QK2, not
QK3. Repeating the earlier proposal to move this release would miss the current
source of extra ordering.

## Another path already carries QK3

Before PV0's probability receive, the selected plan has a FIX -> M -> MTE2
relay. Its M publication observes the M prefix containing QK3. The later
MTE2 -> MTE1 input-readiness transfer carries that completion to the operand
preparation. One explicit local causal path is:

```
QK3 completion
  -> SET M -> MTE2
  -> WAIT M -> MTE2
  -> SET MTE2 -> MTE1
  -> WAIT MTE2 -> MTE1
  -> PV0 LEFT preparation
```

Thus two different statements hold: the bank release is narrow, while the
preparation's total acquired prefix still includes QK3. The native frontier
export reports an unknown-occurrence FIX -> MTE2 RAW requirement on imported
cell 4. The admitted target routes that transfer through M. This does **not**
prove that the requirement is unnecessary, that a direct direction is admitted,
or that its publication can safely move. The exact queue/alias occurrence is
the next attribution target, before any placement change.

## Evidence and limits

Both variants were freshly constructed and reconstructed with default policy
and `--source-gaps`; all four outputs are byte-identical to the saved current
plans. A local explicit-prefix diagnostic follows 96 matrix occurrences for
`arg16=0` (six outer entries), preserving emitted event matching and rearming.
It records physical LEFT/RIGHT predecessor names and the selected causal paths.
It uses zero scalar values for address-only inspection, does not validate GM
addresses or numerical data, and does not prove queue/peer progress. It is not
a device trace or a complete attention correctness checker. Hard collectives
provide no extra credit in this diagnostic, matching current native policy.

Artifacts in `../source-reuse-quality-work/`:
`attention_trace.py`, `attention48-trace.json`, `attention49-trace.json`,
`attention48-frontiers.tsv`, and fresh native plans/logs under `corpus/`.
The trace records each plan's SHA-256. Original source anchors for `__48` are
QK2 at line 101, QK3 at 118, receive at 121, LEFT overwrite at 123 and PV0 at 128.

## Initial next steps (addressed by the local follow-up below)

1. Explain the relay's imported cell and producer/consumer occurrences using
   shared protocol, storage and requirement views. Distinguish a required
   queue ordering from an incidental prefix introduced by a legal route.
2. Check the manual-attention transcription's actual overload and FIFO contract
   before transferring a local proposal to it. Keep message, bank and semantic
   row identities separate; do not infer local-slot release from tile TFree.
3. Only after finding a concrete unnecessary dependency, qualify the narrower
   source or route and test prologue/body/epilogue plus missing-support cases.
   A changed queue premise requires the full coupled device harness.

This refines item 3; it does not claim general cross-phase bank qualification or
an attention performance improvement. Item 1's last-reader device work remains
parked until the other agent returns its results.

## Follow-up: the obligation is the shared GM FIFO slot

The relay's cell 4 is the GM allocation behind the bidirectional pipe, not a
LEFT/RIGHT operand bank. The shared protocol model imports `TPush` as a write
and `TPop` as a read of `globalStorage`. The current native footprint pools
both FIFO slots into that one cell. Source:
`lib/PTO/IR/PTOPipeline/PTOSyncProtocolModel.cpp`.

The pinned A3 lowering (`pto-isa` commit
`0c112d61f41342bd0867ce1080c29f1590d72484`,
`include/pto/npu/a2a3/TPush.hpp`) has separate producer and consumer cursors,
both initially zero. Each uses:

```
GM base + (its cursor % slot_num) * slot_size + entry_offset
```

Both directions use the same GM base in these inputs. It would be incorrect
to give each direction a separate allocation. Here `slot_num=2`,
`slot_size=8192`, entry offset zero and unsplit transfers apply. QK sends occupy
8192 bytes; probability receives read 4096 bytes. Conservatively representing
each access by its whole selected slot suffices for this experiment.

QK0 and QK2 send to slot 0. QK1 and QK3 send to slot 1. The first body receive
(PV0 probability) reads slot 0: its preceding local FIX writer is **QK2's
push**, whose prefix also covers QK0's push. QK3's push is relevant to the next
receive, not this one. This identifies a local prior-writer obligation;
the probability data itself comes from the remote AIV producer. Peer matching,
backpressure and GM visibility are not proved by this local model.

The diagnostic propagates each cursor's two-state parity separately over the
unchanged CFG. Each static send/receive has one slot identity, including
prologue, repeated body, epilogue and successive outer entries. There is no
product graph, emitted counter or new guard. The effect view rejects ambiguous
slot identities, other accesses to the pooled root, and other imported cells
sharing that origin. This fixture-scoped implementation is not a production
alias or general queue qualifier.

## Two sources of widening, and a supported narrower arrangement

Splitting the GM footprint moves the source from post-QK3 push (cut 117) to
post-QK2 push (cut 100). That alone is insufficient: the default FIX -> M ->
MTE2 relay still publishes from M after QK3. The target already admits another
two-hop route, FIX -> MTE1 -> MTE2; no new hardware direction is assumed.

The successful controlled variant starts from the original selected words:

1. Add FIX -> MTE1 publication after QK2's push, at cut 100.
2. Acquire it and forward MTE1 -> MTE2 before the first receive, at cut 118.
3. Keep the original broad FIX -> M publication after QK3's push.
4. Move its remaining acquisition/relay endpoints from cut 118 to the **start
   of cut 125**, before the existing waits for PV0 computation.

This leaves both PV0 operand preparations before the broad receipt while
preserving that receipt before the second receive and later key reuse.
Existing actual transfers rearm the two new directional keys (ID 5); no
private acknowledgment pairs are added. All other selected words, fences,
payloads and queue operations remain unchanged. The price is two additional
event pairs per inner-body visit.

Simply rerunning construction with the alternate route is not an acceptable
substitute. An exploratory six-entry explicit-prefix comparison found new
payload ordering after both slot refinement alone (492 added relations) and
slot refinement with the alternate route (784 added). The controlled variant
instead removed 432 relations and added none in that diagnostic. Those counts
are local finite evidence, not device measurements or a general route theorem.

## Reproducible checks and limits

The [fixture-scoped linked probe](../../test/benchmarks/attention_relay/README.md)
runs against both pinned prepared inputs. The production cold causal checker
accepts the narrower words on the slot-refined model. Independently,
`GraphOracle.h` checks required memory relations, event balance, rearming,
acyclicity and the complete payload issue/finish relation:

| Outer entries / inner lengths | Payload operations | Removed relations | Added |
| --- | ---: | ---: | ---: |
| Empty | 0 | 0 | 0 |
| One / 3 | 83 | 72 | 0 |
| Two / 3,3 | 166 | 144 | 0 |
| One / 1 | 43 | 24 | 0 |
| One / 2 | 63 | 48 | 0 |
| Two / 1,2 | 106 | 72 | 0 |

Results agree for both inputs. The last three rows stress the imported checker
CFG with varying lengths; they are **not executions of the native constant-trip
loop**. Actual inner length is three. Explicit assertions remove the second
QK's completion before the first receive and both preparations, while retaining
each receive's same-slot FIX completion.

Missing either new leg fails the cold memory check. Removing the retained broad
path fails both memory and rearming in the independent graph while retaining
balanced, acyclic events. Moving that receipt past PV0 computation is safe but
adds 82 relations versus baseline in the one-entry trace: it is an ordering
negative, not a missing-completion negative.

Raw outputs and build recipe: `../attention-relay-work/repro/`. Manually assembled
variants export commands and a fresh cold certificate, not stale constructor
ledger/decision records. Only baseline PTO uses official native emission and
reconstruction. Candidate words have not been integrated into the constructor,
emitted/reconstructed through its native wrapper, or tested with a coupled
AIC/AIV device harness. No attention speedup is claimed.

## Subsequent implementation milestone (now completed locally)

Expose the qualified per-cursor slot view through the existing protocol/effect
interface, then select relay prefixes and acquisitions according to their
separate physical-storage and key-republication deadlines. The regression above
is the acceptance target. Do not hard-code these fixture cuts, globally prefer
MTE1 relays, separate the shared FIFO by direction, or delete the broader
receipt. A production candidate needs native reconstruction and full coupled
queue qualification before device timing.

The follow-up is now implemented and tested in ordinary native construction;
see [the current result and remaining device boundary](oahs-static-fifo-relays.md).
The supplied-plan counts above remain the historical prototype result.
