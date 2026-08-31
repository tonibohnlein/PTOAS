# CanonicalSync target synchronization evidence

This document records the target facts admitted by the CanonicalSync target
provider. It is an implementation design record, not a portable PTO language
contract. Unknown target behavior remains disabled.

## Evidence baseline

The initial provider uses the following official Ascend sources:

- intra-core synchronization overview and the 2201 AIC/AIV tables, asc-devkit
  revision `7008190b73b1`;
- SetFlag/WaitFlag semantics and static-Tensor event-ID constraints, the same
  asc-devkit revision;
- synchronization key features, including the 3510 same-`PIPE_V` guarantee,
  the same asc-devkit revision;
- the stable CANN 8.5 PipeBarrier contract.

The asc-devkit pages are development documentation. Their revision is part of
the provider identity so a later specification change cannot silently alter
an existing compiler contract.

## Core identity

The 2201 event table is selected only by an authoritative
`pto.kernel_kind` on the function or an enclosing module, or by explicit
`pto.section.cube`/`pto.section.vector` ownership inside the function:

- `cube` selects AIC;
- `vector` selects AIV;
- a cube or vector section selects its corresponding domain;
- missing identity leaves the core unresolved;
- conflicting attributes or section identities produce a conflict.

Observed pipeline kinds do not authorize a core-domain choice. An unresolved
or conflicting core has no compiler-usable directed event pair.

## 2201 directed events

The AIV hardware and compiler-exposed sets are identical: every directed pair
between distinct members of `{S, V, MTE2, MTE3}` is available.

The AIC compiler-exposed set is:

| Source | Compiler-exposed targets |
| --- | --- |
| M | MTE1, MTE2, FIX |
| MTE1 | M, MTE2, MTE3, FIX |
| MTE2 | M, MTE1, MTE3 |
| MTE3 | MTE1, MTE2 |
| FIX | M, MTE1 |

The hardware table additionally records `MTE2 -> FIX`, `MTE3 -> FIX`,
`FIX -> MTE2`, and `FIX -> MTE3` as existing without a current programming
scenario. The provider retains these four pairs for diagnostics but does not
allow CanonicalSync to emit them. Cells marked as not involved are absent from
both sets.

The same 2201 contract is used for A2, A3, and their intersection profile.
No 2201 pair is reused as evidence for A5/3510.

## Same-pipeline completion and barriers

Ordinary issue order does not establish completion. A2/A3 same-pipeline
dependencies use a legal PipeBarrier on `V`, `M`, `MTE1`, `MTE2`, `MTE3`, or
`FIX`; `PIPE_S` is never admitted. The A5 partial contract retains the
documented same-`PIPE_V` hardware-completion rule, so CanonicalSync does not
insert a `PIPE_V` barrier there.

`PIPE_ALL` is handled separately as the mandatory function-tail fence. It is
not a direct cover candidate in the strict correctness mode.

## Event IDs

Although the documented physical range is 0 through 7 on A2, A3, and 950,
static-Tensor code is advised to leave IDs 6 and 7 to system/internal uses.
CanonicalSync therefore owns only IDs 0 through 5. Hidden/manual reservations
are subtracted from that set. An allocation outside the provider-owned set is
rejected again during materialization.

## Materialization boundary

Before staging a generated action, materialization independently checks:

1. the exact directed event is in the compiler-exposed table;
2. every allocated ID is compiler usable;
3. every targeted barrier is legal for the target;
4. an unresolved or conflicting core cannot emit an event.

These checks do not replace final event-state or demand verification. They
only ensure that an abstract recipe cannot cross the target-capability
boundary with an unsupported primitive.
