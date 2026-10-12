# GroupNorm: manual event reuse and deferred scarcity repair

This note records the GroupNorm investigation at revision
`4217e8abdab1807fb2eedc50c62d7a7521e073dd`. It is evidence for future event
allocation work; it introduces no recognizer, allocation algorithm or scarcity
repair.

The manual plan uses barriers and a bank-return protocol to constrain execution.
Its small event footprint does not prove that Frontier's selected minimum-demand
graph can use the same IDs without additional ordering. Conversely, the current
allocation failure does not establish a hardware minimum.

## Input and comparison

The prepared serial kernel has FP32 data, N=4, G=4, 16 channels per group,
S=256 and block S=64. Both loops execute four iterations. There are 16 blocks;
only subblock zero computes. The channel-pipeline variant is outside this port.

The [manual plan](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/expert.pto)
and [unsynchronized input](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto)
have identical payload IR: deleting the manual plan's `pto.set_flag`,
`pto.wait_flag` and `pto.barrier` lines reproduces the input byte for byte.
The [port generator](../../test/npu_validation/benchmarks/auto_sync/port_groupnorm.py)
and [metadata](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/port.json)
identify the serial upstream example at commit
`cbc67f2697e8ee18357adc447272691432b3948d`.

Source pins:

- Upstream snapshot SHA256:
  `bc18b91ed9041890e980e70e16cde602b3cfad2bb7569bb4a36c248c27b09b39`.
- Manual plan SHA256:
  `0990e280b3fb6a4bc3ede9733e30524c7bccf434dbab8725c7ffe05876f7bc4f`.

The port is not device-qualified. This investigation establishes neither device
correctness nor device performance.

## Current Frontier allocation failure

The pinned optimized replay establishes exact demands for all 23 original
regions under both GM alias policies. Production logical preparation succeeds;
physical allocation fails with:

```text
regional lane assignment not certified within directed capacity;
no minimum-capacity claim; scarcity repair not implemented yet
```

The retained `regional-palettes` certificate contains 17 groups and 25 lanes,
including ten MTE2-to-V lanes. Its MTE2-to-V conflict graph contains an
eight-lane clique:

| Origin | Lanes in the clique |
| --- | ---: |
| First-loop cyclic handoff, record 0, group 0 | 2 |
| First-loop tail, record 18, group 2 | 1 |
| Gamma load, record 31, group 5 | 1 |
| Beta load, record 34, group 6 | 1 |
| Second-loop cyclic handoff, record 37, group 7 | 2 |
| Second-loop initial load, record 61, group 16 | 1 |

The global lane indices are `[24, 10, 9, 8, 7, 4, 1, 0]`. Merely reordering
greedy coloring cannot fit this fixed conflict certificate into six IDs.

These conflicts mean that the existing sufficient reuse proof did not establish
either lifetime order. They need not imply an unavoidable conflict in the actual
command graph. [Regional lane certification](../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalLaneCertificate.cpp)
uses guarded first/last envelopes. Its
[handoff reuse test](../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalEvents.cpp)
requires the prior consumer's completion before the next producer's start when
they use different pipes. This is stronger than merely showing that the prior
WAIT consumed its token before the next SET. Improving this proof is distinct
from adding ordering to repair scarcity.

## How the manual plan reuses events

The explicit flags use numeric IDs 0 and 1 in three directed namespaces:

| Direction | IDs | Meaning |
| --- | --- | --- |
| MTE2 to V | 0, 1 | Input bank is ready |
| V to MTE3 | 0, 1 | Output bank is ready |
| MTE3 to MTE2 | 0, 1 | Store is complete; bank can be reused |

These are six `(direction, ID)` combinations, using two distinct numeric labels.
Frontier also allocates per directed namespace; its failure is not caused by
treating all directions as one pool. This table counts explicit flags only and
does not account for any target implementation of an ALL barrier.

The first phase has an ALL barrier after the initial load and at the end of each
iteration. Further ALL barriers separate the gamma/beta loads, broadcasts and
second phase. These drain work that Frontier's selected graph need not drain.

The second phase seeds two MTE3-to-MTE2 tokens, then follows this cycle for each
bank:

```text
WAIT(store complete) -> load -> SET(input ready)
  -> WAIT(input ready) -> vector compute -> SET(output ready)
  -> WAIT(output ready) -> store -> SET(store complete)
```

Each next load waits for the previous store on that bank. This feedback protects
bank reuse and constrains pipeline progress beyond the minimum ordering needed
just to protect an input read. Final waits drain both return tokens, followed by
the port's terminal ALL barrier. The barriers and feedback allow cyclic ID reuse
by accepting additional serialization.

## Validation and deferred work

Local diagnosis artifacts are retained under
`.local/three-paths/group-norm-allocation-diagnosis/`; the source-pinned replay is
under `.local/three-paths/tilelang-rerun-4217e8abd-c9mc9fl7/`. These ignored
artifacts are supporting measurements, not checked-in regression tests.

A scalar/control interpreter traced the current manual IR for active blocks 0,
3 and 15 and inactive subblock 1. Each active trace contained 78 payloads and
eight executed ALL barriers. All SET/WAIT tokens matched and drained. An
independent command-graph check established prior-WAIT-to-next-SET reachability
for all eight repeated SET instances in each active trace. This checks event
matching and causal reuse for those traces; it does not prove byte-access safety,
numerical correctness or device behavior.

Future work should distinguish two changes:

1. Improve occurrence-, guard- and lane-sensitive reuse proofs for the unchanged
   selected graph. A weaker sufficient certificate may recover allocation
   without sacrificing concurrency.
2. Implement explicit scarcity repair that adds ordering, drains or an equivalent
   bank-return protocol when the selected graph cannot be allocated. Validate
   the resulting graph and event lifetimes, and measure the concurrency cost
   separately from logical-demand minimality.

The manual plan is a useful comparison for that second task. It is not a reason
to add a GroupNorm-specific recognizer or silently change the current
allocation-only policy.

The [decode investigation and Section 7 comparison](frontier-decode-event-scarcity.md)
relates this protocol, prefill and decode to the current draft's merges,
return handshakes and allocation certificates. These are future resource
mechanisms, separate from changes to Section 5.4 demand algorithms.
