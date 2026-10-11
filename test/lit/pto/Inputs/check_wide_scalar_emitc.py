# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Execute emitted scalar functions against independent Python bit-vector values."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile

WIDTH = 128
MASK = (1 << WIDTH) - 1
LIMB_MASK = (1 << 64) - 1


def signed(value, width=WIDTH):
    """Interpret the unsigned bit pattern at its specified width."""
    return value - (1 << width) if value >> (width - 1) else value


def operand(value):
    """Build a host operand using two representable unsigned 64-bit literals."""
    return f"bits({value >> 64}ULL, {value & LIMB_MASK}ULL)"


def scalar_functions(source):
    """Copy only complete fixture scalar functions, excluding hardware support."""
    functions = []
    pattern = r"AICORE [^\n{]*\bwide_\w+\([^(){};\n]*\)\s*\{"
    for match in re.finditer(pattern, source):
        end = match.end()
        depth = 1
        while depth and end < len(source):
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        if depth:
            raise RuntimeError("Unterminated emitted scalar function")
        functions.append(source[match.start():end].replace("AICORE ", "", 1))
    if len(functions) != 15:
        raise RuntimeError(f"Expected 15 emitted scalar functions, got {len(functions)}")
    return "\n".join(functions)


def oracle_cases():
    """Cover constants, wrapped arithmetic, order, shifts and signed extensions."""
    rows = [("wide_min()", 1 << 127), ("wide_max()", (1 << 127) - 1),
            ("wide_high()", 1 << 63), ("wide_negative()", (-1 << 100) & MASK)]
    values = [0, 1, (1 << 63) - 1, 1 << 63, LIMB_MASK, 1 << 64,
              (1 << 100) - 1, 1 << 100, (1 << 127) - 1, 1 << 127, MASK]
    for x in values:
        sx = signed(x)
        rows.extend([(f"wide_unsigned_identity({operand(x)})", x),
                     (f"wide_floor({operand(x)})", (sx // 7) & MASK),
                     (f"wide_truncate({operand(x)})", signed(x & LIMB_MASK, 64) & MASK)])
        for shift in [0, 1, 63, 64, 127]:
            rows.extend([(f"wide_left({operand(x)}, {shift})", (x << shift) & MASK),
                         (f"wide_logical_right({operand(x)}, {shift})", x >> shift),
                         (f"wide_arithmetic_right({operand(x)}, {shift})", (sx >> shift) & MASK)])
        for y in values:
            rows.extend([(f"wide_arithmetic({operand(x)}, {operand(y)})", ((x + y) * (x - y)) & MASK),
                         (f"wide_signed_less({operand(x)}, {operand(y)})", int(sx < signed(y))),
                         (f"wide_unsigned_less({operand(x)}, {operand(y)})", int(x < y))])
    for value in [0, 1, (1 << 63) - 1, 1 << 63, LIMB_MASK]:
        arg = f"static_cast<int64_t>({value}ULL)"
        rows.extend([(f"wide_signed_extend({arg})", signed(value, 64) & MASK),
                     (f"wide_unsigned_extend({arg})", value)])
    return rows



def rejected_inputs(emitter, directory):
    """Unsupported widths and unavailable double-width operations must fail cleanly."""
    bodies = [
        ("i256", "return %x : i256"),
        ("i128", '%sum, %overflow = "arith.addui_extended"(%x, %y) : '
         '(i128, i128) -> (i128, i1)\n'
         '%value = arith.select %overflow, %x, %sum : i128\nreturn %value : i128'),
        ("i128", '%low, %high = "arith.mulsi_extended"(%x, %y) : '
         '(i128, i128) -> (i128, i128)\n'
         '%value = arith.xori %low, %high : i128\nreturn %value : i128'),
        ("i128", '%low, %high = "arith.mului_extended"(%x, %y) : '
         '(i128, i128) -> (i128, i128)\n'
         '%value = arith.xori %low, %high : i128\nreturn %value : i128')]
    for index, (kind, body) in enumerate(bodies):
        source = directory / f"unsupported-{index}.pto"
        source.write_text('module attributes {pto.target_arch = "a5"} {\n' +
                          f'func.func @unsupported(%x: {kind}, %y: {kind}) -> {kind} {{\n' +
                          body + '\n}\n}\n')
        result = subprocess.run([emitter, "--pto-level=level3", "--enable-insert-sync=false",
                                 "--enable-plan-memory=false", str(source), "-o", os.devnull],
                                capture_output=True, text=True, timeout=30)
        if result.returncode <= 0 or "failed to legalize" not in result.stderr:
            raise RuntimeError(f"Unsupported wide case {index} did not fail conversion cleanly: {result.stderr}")


def main():
    source = Path(sys.argv[1]).read_text()
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        raise RuntimeError("Host C++ compiler unavailable for wide scalar oracle")
    rows = oracle_cases()
    prologue = """#include <cstdint>
#include <cstdio>
using U = unsigned __int128;
static __int128 bits(uint64_t high, uint64_t low) {
    return static_cast<__int128>((static_cast<U>(high) << 64) | low);
}
static void output(U value) {
    std::printf("%016llx%016llx\\n", static_cast<unsigned long long>(value >> 64),
                static_cast<unsigned long long>(value));
}
"""
    calls = "\n".join(f"output(static_cast<U>({expression}));" for expression, _ in rows)
    with tempfile.TemporaryDirectory(prefix="wide-scalar-oracle-") as directory:
        path = Path(directory)
        cpp = path / "oracle.cpp"
        executable = path / "oracle"
        cpp.write_text(prologue + scalar_functions(source) + "\nint main() {\n" + calls + "\n}\n")
        # Keep the complete generated functions: their existing device preamble
        # and expression folding can leave unused aliases or temporaries.
        subprocess.run([compiler, "-std=c++17", "-O2", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-undefined-trap-on-error", str(cpp),
                        "-o", str(executable)], check=True, timeout=60)
        result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
        emitter = shutil.which(sys.argv[2])
        if not emitter:
            raise RuntimeError("PTOAS emitter unavailable for unsupported-width checks")
        rejected_inputs(emitter, path)
    actual = [int(line, 16) for line in result.stdout.splitlines()]
    expected = [value for _, value in rows]
    if actual != expected:
        raise RuntimeError("Emitted scalar result differs from independent bit-vector oracle")
    print(f"wide scalar emission: {len(rows)} host bit-vector checks passed")


if __name__ == "__main__":
    main()
