# Empty whole-invocation allocation

A retained whole-function analysis may prepare an empty logical plan when its
single original invocation loop has constant signed 64-bit bounds with
`lower >= upper` and a positive constant step. The original whole-invocation
check must cover every payload operation; surrounding payloads, other regions,
unsupported effects, manual synchronization and preserved synchronization
protocols cannot use this shortcut. This is an original execution-domain proof,
not an inference from an empty demand set or from one empty child.

Preparation and allocation use the same qualification. The fresh plan contains
no endpoint, scalar preparation, family recipe or terminal drain. Allocation
exports the existing empty finite certificate. Split allocation accepts an
actually empty eligible-ID pool and emits no commands. Combined physical output
retains a private empty output owner after validating the exact finite encoding,
plan identity and ordinary allocator preflight. The owner seals the certificate
and rechecks the empty executable plan at insertion; it introduces no new public
certificate schema. Physical output removes allocation metadata as usual.

The public combined pass still supplies its ordinary default pool when no pool
is specified. For that route the claim is zero IDs used, while the split
regression also checks an empty supplied pool. Dynamic bounds and nonempty
invocations retain their existing allocation obligations. Original manual
protocols retain the public pass's existing skip behavior.

The focused regression covers equal and reversed signed bounds, signed extremes,
original IR preservation, zero payload/command traces, dynamic and nonempty
empty-pool refusals, surrounding payloads and a preserved barrier, and unchanged
manual SET/WAIT commands. Both source reviews and bounded execution reviews
accepted this batch. The native incremental build passed, followed by all 11
early focused tests, including the empty-invocation and existing physical
allocation tests. The broader 63-test gate is separate.

The exact early receipt is `.local/milestone-loop/qualified-transport-empty-trace-gate/first-lit.json`,
SHA256 `1a6cad7d008aacb8e9a90ae8ffe8eb676d8da4621d38ee7688bd590f5759d69a`.
This verifies host tools and these protocols; kernel compilation remains separate.
