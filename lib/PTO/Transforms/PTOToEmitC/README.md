# Scalar integer emission

The scalar type converter preserves integer widths 1, 8, 16, 32, 64 and 128.
The 128-bit domain uses the existing `__int128` and `unsigned __int128` C++
extensions, including the intermediate values used by synchronization predicates.
Other widths fail conversion explicitly.

Wide constants are built from two unsigned 64-bit literal limbs. Signed negative
constants use a representable positive complement followed by negation and
subtraction, including the minimum signed 128-bit value. No wide constant is
converted through a host `int64_t` or emitted as an oversized decimal literal.

Ordinary scalar operations retain their existing signed or unsigned lowering.
Extended addition and multiplication currently support operands through 64 bits;
their 128-bit operand variants require an unavailable wider result domain and
fail conversion. Target compilation still requires a compiler that supports the
C++ integer extensions already used by the arithmetic lowering.

`test/lit/pto/arith_i128_emitc.pto` checks source emission and independently
executes the extracted scalar functions on a host compiler, with undefined
behavior checking. This test does not establish device execution support.
