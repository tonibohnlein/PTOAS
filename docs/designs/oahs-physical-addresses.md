# InsertSync physical-address contract

Physical address interpretation is shared by ordinary InsertSync, the logical
constructor and emitted-IR reconstruction. Repeating an incorrect translation
is not independent evidence of correct physical effects. Address admission
therefore precedes automatic planner selection; authored-event preservation
continues to run first.

`SyncAddressEvaluator` interprets integer constants and integer/index casts with
APInt bit semantics. Truncation discards high bits; signed and unsigned extension
remain distinct. The qualified PTO EmitC index width is 64 bits. A conflicting
MLIR data-layout width disables index evaluation. Unsupported arithmetic is
unknown, not stripped or treated as an identity operation. Cached successes and
unknown results belong to the original SSA expressions; depth, integer width
and unique-node limits bound evaluation. Subview offsets use the same evaluator.

Admission checks this address contract, not completeness of operation effects or
logical-planner support. It has three results:

- Safe: no rejected or unresolved explicit allocation address was encountered.
  This does not qualify every possible access interval in the function.
- Conservative: an explicit runtime address is unresolved. Translated accesses
  carry `aliasesUnknownRange`, which preserves possible conflicts with other
  roots in the same address space. Logical construction may decline precision;
  legacy fallback still receives conservative aliasing.
- Rejected: a known address, footprint, multi-tile alignment or slot interval is
  incompatible with the physical address contract. Neither automatic planner
  may run on that geometry. Translation failure is propagated as a compiler
  error rather than an internal abort or a partially translated success.

Multi-tile storage uses the same checked slot offset/address utility in
translation and buffer-select lowering. Physical footprint and aligned stride
are different quantities. A 32-byte LEFT tile has a 512-byte slot stride, so
slot one aliases a separate tile at base plus 512. Planned address tables keep
their original slot order. Unknown dynamic selectors continue to use the
lowerer's slot-zero default; this change does not introduce modulo semantics.

Unknown placeholder ranges and equal may-address unions are not equality
proofs. Identity-TMOV removal must retain copies whose address records are
unknown or contain multiple possible intervals. Ordinary exact singleton
identity cases remain eligible.

The focused physical-address tests invoke the pass directly with retained cast
SSA, challenge both alias queries, and check actual ordering in existing,
logical and hybrid modes. Separate lowering tests evaluate emitted address
selection rather than trusting the translator's table. Full default-pipeline
regression and device qualification remain distinct from this source-level
contract.
