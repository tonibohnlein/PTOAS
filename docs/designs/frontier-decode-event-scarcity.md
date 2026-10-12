# Causal-convolution decode: grouped handoffs and event capacity

This note records the decode inspection at PTOAS revision
`e77539ec40f3a642c40fca7d97206c3ca4249960`, and relates the manual protocols
for decode, prefill and GroupNorm to Section 7 of the current synchronization
draft. It records evidence for later allocation work, without implementing
allocation changes or scarcity repair.

## Current compilation result

An optimized source-pinned replay (`-O2 -UNDEBUG`, C++17) gives:

| GM alias policy | Logical preparation | Logical seconds | Six-ID allocation | Allocation seconds |
| --- | --- | ---: | --- | ---: |
| MayNotAlias | Ready | 0.202 | Certificate does not fit | 0.207 |
| MayAlias | Ready | 0.404 | Certificate does not fit | 0.206 |

Both allocation attempts report:

```text
regional lane assignment not certified within directed capacity;
no minimum-capacity claim; scarcity repair not implemented yet
```

Each retained certificate has nine pairwise conflicting MTE2-to-V lanes:
the four weights, bias, three conditional history loads and current input.
Their group indices are 0 through 8, with record IDs
`[0, 1, 2, 3, 4, 5, 7, 9, 11]`. Six colors cannot fit this certificate.
This is not a hardware lower bound, nor a proof that every realization of
the selected payload order needs nine IDs. The current certificate uses
sufficient reuse tests; changing communication placement or grouping is a
separate question from coloring these unchanged conflicts.

The replay verified 1,671 compiler/header/build files against the build mirror,
and used compiler SHA256
`05556c11ecac504a7d3545c980ed6e1b22bc03cadd66a4570103817e05d6a65d`.
It ran serially with a 120-second external watchdog and 4-GiB address-space
limit; neither limit was reached. Measurements and certificate summaries are
retained in `.local/three-paths/decode-allocation-e77539ec4-9uwxcs09/`.
These are diagnostic replays, not device or controlled performance measurements.

## Manual synchronization

Removing the synchronization lines from the
[prepared manual plan](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/expert.pto)
reproduces the [input](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/input.pto)
byte for byte. The
[generator](../../test/npu_validation/benchmarks/auto_sync/port_conv.py)
and [metadata](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/port.json)
identify upstream commit `cbc67f2697e8ee18357adc447272691432b3948d`.
The prepared manual plan SHA256 is
`6b76e30d9bfd258dd07d4673cb7a85d5e60e10fde7fb629483463ec76e016528`;
the upstream decode snapshot SHA256 is
`9c77625c815179121e0a18d8de1df3277b8fc9e9d69806ad5203f73bffd7b420`.

| Direction | IDs | Notification |
| --- | --- | --- |
| MTE2 to V | 1 | All four weight loads and bias together |
| MTE2 to V | 2, 3, 4 | One for each optional history load |
| MTE2 to V | 5 | Current input |
| V to MTE3 | 0 | Output conversion complete |
| V to MTE3 | 6 | All three state-save conversions complete |

There are five labels in the first direction and two in the second. Numeric
ID 6 is outside the selected `0..5` pool, but renaming it to an unused eligible
label in V-to-MTE3 would preserve explicit matching without adding ordering.
The port retains the authored label and marks its target/API qualification
as outstanding. No renaming or device qualification was performed here.

The grouped weight notification waits for all five loads before any of their
conversions can start. The grouped save notification waits for all three
conversions before any state store can start. These are coarser readiness
boundaries than notifications for individual operands. The prepared decode
port has no repeating bank-return protocol. It contains six V barriers from
helper expansion and a port-added terminal ALL barrier.

The manual port states MayNotAlias and valid-cache-index contracts and is not
device-qualified. Its coverage must not be assumed for arbitrary overlapping
GM arguments. A complete comparison with the exact event graph is still
needed to identify strictly additional payload order; grouping observed in
the source alone does not quantify lost parallelism.

## Correspondence with Section 7

The inspected reference is `sections/event_id_reuse.tex`, headed
"Finite event IDs: allocation and repair", in paper revision
`1581f0ca5afee17fe5f791ee4cb49796a218b340`. Its working-tree SHA256 is
`ce76b9ff39a486bc37bfffff2ebebecdc1b05511bba76a80cae1b14d4288336a`.
The paper working tree was clean at inspection.

Section 7 separates allocation of unchanged handoffs from repairs that change
communications or command order. Its mechanisms point directly toward the
manual protocols, subject to the following qualifications:

| Case | Relevant mechanism | What still needs proof or implementation |
| --- | --- | --- |
| Decode | Section 7.3 merges consecutive handoffs into one SET after the last source and one WAIT before the first consumer, matching the grouped weight/bias and save notifications. | Construct legal blocks from retained demands, preserve guards and matching, and check coverage and all directional capacities. |
| Prefill | Return handshakes, cyclic lanes and periodic reuse certificates match the purpose of its two-bank feedback. | Its three-pipe protocol and separate input/output areas are not literally the two-pipe rotating-buffer theorem. Entry tokens, finite exit, guards and every directional capacity need their own certificate. Its current compiler failure is a missing allocation export, not established scarcity. |
| GroupNorm | Grouped handoffs, return handshakes and phase separation resemble its manual bank protocol and ALL barriers. | Distinguish better reuse proofs from added ordering. An order-preserving ALL-barrier replacement requires the draft's complete-separation and notification-closure conditions. |

The draft's precise acknowledgment construction signals after the prior WAIT
and waits just before the next SET. This protects consumption before
republication. A manual store-completion return is a stronger placement;
it need not be the least costly repair. The draft explicitly demonstrates
that an acknowledgment can sometimes preserve the original payload order
when its WAIT is placed late enough. Thus a failure of direct allocation
does not establish that serialization is necessary.

For fixed communications, adding V-only barriers does not create missing
cross-pipe reuse paths (Section 7.3, `lem:local-barrier-invariance`). Those
helper barriers cannot by themselves explain reduced directional ID use.
The relevant mechanisms here are communication grouping, feedback, command
placement and, where qualified, all-pipe separation.

The draft requires each repaired plan to certify acyclicity, demand coverage,
matching, consumption before republication, legal cuts and eligible IDs,
including flags used by return handshakes. Its generic repair family takes
an explicit handoff list; compact periodic entry/exit and structured guards
are separate obligations. The rotating-buffer optimality claims apply to
their stated families, not arbitrary whole-kernel repairs.

Section 7.4 also supplies a serial fallback under qualified directions, cuts
and control availability. This is a conservative floor rather than the manual
kernels' overlapping pipeline. Section 7.5 explicitly leaves general minimal
repair synthesis open; implementing these mechanisms does not promise a
globally optimal or equally fast manual plan.

See the [prefill note](frontier-prefill-event-reuse.md) and
[GroupNorm note](frontier-group-norm-event-scarcity.md) for their source evidence.
The next allocation investigation should try reuse without added payload order
first, then assess merges and return placements by their actual added order.
No Section 5.4 algorithm change is required to pursue these Section 7 mechanisms.
