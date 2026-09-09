# Explicit retirement and endpoint normalization

This records the step-2 implementation after `82311818f`, using the existing
Release `-O1` assertion build and unchanged work allowance. These are local
construction/replay results; device correctness and timing have not been run.

| Fixture/scenario | Executed pairs before → after | Scalar/control before → after | Current PTO emission seconds |
| --- | ---: | ---: | ---: |
| one_buffer/trips_16 | 63 → 62 | 344 → 324 | 0.89 |
| online_softmax/16 | 94 → 93 | 680 → 625 | 13.24 |
| q_proj/core0 | 509 → 507 | 3327 → 2050 | 33.27 |
| qk_matmul/16 | 255 → 253 | 1049 → 841 | 4.36 |

The old columns use the frozen accepted logical checkpoint. Current outputs
have one explicit terminal `PIPE_ALL`; the old checkpoint used ordinary-lane
handoffs for exit. This qualification change is intentional and the historical
artifacts remain untouched. The terminal drain is never used to justify body
ordering or event consumption. Payload, allocation, views and ABI match exactly;
all recorded cross-lane boundary checks and QK independent-preload checks pass.

Complementary branch producers now feed one unconditional wait at their common
consumer. A variant with overlapping optional writes retains MTE2 write/write
completion on every Boolean combination; deleting its wait is rejected. The
retirement suite covers final MTE3/FIX, readerless loads, repeated invocation,
legacy tail hints and malformed drains/unconsumed publications.

Optional endpoint normalization indexes exact boundaries, preserves intervening
commands, and merges only disjoint domains of identical directed key/actions.
Balanced unions avoid growing-prefix copies. Intersection product and constraint
size are checked before materialization; optional work uses at most 100,000 or
1/32 of remaining work, whichever is smaller, charged to the original allowance.
Unknown or exhausted optional queries retain the certified endpoints. Same-block
predicate reuse requires the defining operation to precede the actual insertion
cursor; it cannot move a remainder outside its nonnegative guard.

Compilation remains too expensive. These invocation times are not isolated pass
times or a paired performance comparison with older reports. The performance
reviewer continues to block scope expansion until eager all-pair ordering and
repeated requirement/relation scans are replaced and measured. No budget increase
or device speedup is claimed.

Raw artifacts: `insertsync-builds/campaign/logical-plan/endpoint-01/four-cases`,
`endpoint-01/check1`, and `retirement/acceptance1` in the parent workspace.
The named focused gate additionally reruns retirement and endpoint challenges.

The final `check-oahs-focused` target passed in 66.63 seconds. Compiler,
architecture and performance reviews accepted this step; scalability and
two-/three-buffer milestone acceptance remain outstanding.
