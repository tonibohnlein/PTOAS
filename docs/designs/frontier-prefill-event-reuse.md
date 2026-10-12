# Causal-convolution prefill: manual event reuse and pipeline throttling

This note records source inspection at revision
`e77539ec40f3a642c40fca7d97206c3ca4249960` for future event-allocation work.
It introduces no allocation changes or scarcity repair. The manual protocol
limits pipeline progress to reuse a small set of event IDs; the amount of
parallelism lost relative to Frontier's exact result has not been measured or
established by a complete event-graph comparison.

## Input and evidence

The [prepared manual plan](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/expert.pto)
and [unsynchronized input](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto)
contain identical payload IR: removing all `pto.set_flag`, `pto.wait_flag` and
`pto.barrier` lines from the manual plan reproduces the input byte for byte.
The [port generator](../../test/npu_validation/benchmarks/auto_sync/port_conv.py)
and [metadata](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/port.json)
identify the upstream source at commit
`cbc67f2697e8ee18357adc447272691432b3948d`.

Source pins:

- Upstream `causal_conv1d_pto.py` snapshot SHA256:
  `72ababa5fbddd034ee5986a8b3e7d255682b02c144126204d4c2f1003ba0011f`.
- Prepared manual plan SHA256:
  `4a1dfc5f2be1b47a11c816cb1eb3cfaf68830233df73951f6370064bb2905b0d`.

The port uses FP16 rows of 128 elements, four-token batches and two banks.
Its stated input requirements include positive sequence lengths divisible by
four and distinct valid cache indices. These frontend requirements are not
automatically assumptions of Frontier's supplied scalar IR. The port is not
device-qualified; this inspection establishes neither device correctness nor
device performance.

## Event protocol

Each notification covers a four-row batch, rather than one individual operation.
Explicit flags use two IDs in each of three directed namespaces:

| Direction | IDs | Meaning |
| --- | --- | --- |
| MTE2 to V | 0, 1 | Input bank is ready |
| V to MTE3 | 0, 1 | Output bank is ready |
| MTE3 to MTE2 | 0, 1 | Stores are complete; bank can be reused |

These are six `(direction, ID)` combinations, using two distinct numeric labels.
Frontier's six-ID capacity is per directed namespace. This table counts explicit
flags, without accounting for any target implementation of ALL barriers.

The plan seeds two bank-return tokens and consumes bank zero's token for the
initial prefetch. Iteration `i` uses bank `i % 2` and prefetches iteration `i+1`
into the other bank when that iteration exists. The prefetch waits for that
bank's return token. Computation consumes the input-ready token; output stores
consume the output-ready token and then publish the bank-return token.
The loop exit consumes both remaining return tokens before the state-update tail.

## Where execution is constrained

The bank-return protocol enforces:

```text
load batch i -> compute batch i -> store batch i
                                      |
                                      v
                              load batch i+2
```

In particular, MTE2 cannot load batch `i+2` until MTE3 finishes batch `i`'s
stores. Input banks occupy bytes `[0, 2048)` and output banks `[2048, 4096)`,
with 1024 bytes per bank. The feedback couples reuse of these separate input
and output areas into one bank lifecycle. It bounds how far loads can advance
and supports cyclic reuse of the same event IDs. Prefetching the next batch
still permits overlap with current computation; the main loop has no ALL barrier.

The prepared plan also contains four static ALL-barrier sites: after weight
loads, after history initialization, between conditional final-state loads and
stores, and at function exit. The terminal barrier is a port addition. Inside
the main loop, expanded arithmetic helpers contribute 24 static V-barrier sites
(six per output row); these constrain vector work without draining every pipe.

Exact minimum demands describe required ordering for the supplied storage model
and prerequisites. The manual bank lifecycle is a stronger scheduling policy
than independently reusing input and output resources. However, separate storage
alone does not prove that each feedback edge is absent from the exact graph's
transitive closure: recurrence and other dependencies can already imply ordering.
A complete graph comparison is still needed to identify strictly additional
ordering and quantify lost parallelism. No claim that the 24 V barriers are
unnecessary follows from this inspection either.

## Deferred allocation work

At the recorded revision, public logical preparation succeeds under both GM
alias policies. Physical allocation reports a missing finite or uniform
allocation export. It does not establish that the logical plan exceeds six IDs.
See [shared guarded-publication validation](frontier-guarded-publication.md).

Future allocation work should first distinguish improving reuse proofs for the
unchanged logical graph from deliberately adding ordering to fit event capacity.
The manual feedback is a reference for the latter: accept reduced concurrency
to bound event lifetimes and permit reuse. Compare complete event graphs before
attributing an ordering loss, and measure it separately from allocation success.
The [GroupNorm investigation](frontier-group-norm-event-scarcity.md) records the
same distinction for a kernel whose current allocation certificate fails.

The [decode investigation and Section 7 comparison](frontier-decode-event-scarcity.md)
relates all three manual protocols to the draft's allocation and repair
mechanisms, including the possibility of command-level reuse without adding
payload ordering.
