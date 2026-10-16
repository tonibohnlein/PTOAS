# Definedness of three archived guard comparisons

The arithmetic-definedness gap is closed for the 24 parsed guard comparisons in
archived pairs 168, 249 and 350. This result assumes the existing 64-bit index
interpretation and corresponding defined original scalar values. It combines
an independent totality argument with the archived equality proofs; it does not
turn those earlier jointly-defined checker receipts into whole-kernel proofs.

The six exact artifacts, three original sources, parser, equality/control sources
and archived receipts are pinned in
`.local/milestone-loop/guard-definedness-audit/inventory.json`. Its SHA256 is
`f244e4a61566853ea3eb617bdf5ac5f9520447a40ae2ee44360514ffc445bba1`.
The source audit and independent correctness review verified the inventory
and accepted the bounded argument. No compiler, checker or kernel was rerun for this source audit.

## Totality argument

Every scalar operation feeding the generated guards is a constant, index cast,
integer comparison, Boolean operation, select, unflagged integer add/subtract/
multiply, signed min/max, or one of three original division/remainder operations.
There are no additional unknown scalar operations, shifts, undefined-value
producers or arithmetic overflow/exact flags.

Casts are i32 to index, index to i32, and index to i128. At the pinned index width,
these are total sign extensions or truncations. Unflagged arithmetic has modular
integer semantics. Comparisons, Boolean operations, min/max and select are total
on defined operands. This argument covers all eagerly constructed operands and
both select arms, including operands that a lazy checker might skip.

The only partial integer operations divide a sign-extended argument by constant
2, or divide/take the remainder by constant 128 in the original loop body.
Their positive constant divisors exclude zero and signed MIN/-1 overflow.
The only loop runs from 0 to 128 with step 1; visited values and the final
increment are representable. Induction over the original SSA definitions thus
establishes total scalar guard construction on each defined original execution.

No positive or nonnegative length premise is needed. A loaded i32 length is
sign-extended, incremented and bounded by signed min with 128; these operations
remain defined for every defined loaded value. The audit introduces no compiler
admission constraint or runtime check.

Together with the archived eight-guard equality proof for each pair, this removes
the arithmetic-definedness caveat for those comparisons.

## Scope limits

This does not prove pointer bounds, initialization, payload validity, correspondence
of original load values between executions, parser cut-field completeness,
notification/member identities, physical reuse or device semantics. It does not
cover a different index width or current production artifacts. Those obligations
retain their own protocol, lowering and validation evidence.
