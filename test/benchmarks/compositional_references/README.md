# Three distinct reference benchmark tasks

Prepared 2026-09-20. These tasks supersede prioritizing another ordinary
double-buffered GEMM campaign. They adopt new source programs; candidate PTO
arms and device results do not yet exist for them.

| Task | Mechanism | Primary reference |
| --- | --- | --- |
| [Retained A](retained_a.md) | One generation reused across output tiles while B turns over | CATLASS example 25 |
| [Cross-tile preload](preload.md) | Next output block's loads overlap current work | CATLASS example 06 |
| [Manual attention](attention.md) | Delayed QK/PV; separate ingress/output lifetimes; real queue protocol | PTO-ISA manual flash attention |

[COMMON.md](COMMON.md) defines matched-payload arms, source normalization checks,
fixed target contracts, numerical gates, performance/profile attribution and
deliverables. Each archive includes that protocol, one `DEVICE_TASK.md`, all
three pinned source trees, harness examples and a checksum manifest. It can be
sent to an agent independently without workstation access.

Compiler pin: `495fb9cbda7649f15a7fc09d6b1d91ffb7737d54`. The tasks measure this
compiler; general constructor fixes follow the resulting evidence separately.
CATLASS unit flags and half output must not be silently replaced by the earlier
FP32-output ordinary GEMM contract.

## Dispatch

Attach the corresponding archive and send its `DISPATCH.txt` contents. Agents
should use all eight allocated devices for independent matched comparisons.
Serialize timed/profiled work only within a device, with all compared arms on
that same device. See the [scheduling addendum](../../../docs/designs/oahs-eight-device-scheduling.md).
There is no reason to wait for the earlier ordinary GEMM campaign before
beginning independent ready tasks.

The task agent owns source adoption and harness adaptation as well as device
measurement. If the full kernel cannot be imported, it must preserve the native
baseline, give an exact unsupported-contract reproducer and label reduced
experiments honestly. An unavailable comparison must not be reported as parity.
