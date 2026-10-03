# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile actual emitted wide constants and compare with their PTO IR values."""
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    ir, emitted, stem = map(Path, sys.argv[1:])
    expected = {int(value) for value in re.findall(r"arith.constant (-?\d+) : i128", ir.read_text())
                if abs(int(value)) >= 1 << 62}
    expressions = []
    for expression in re.findall(r"__int128 v\d+ = (.+);", emitted.read_text()):
        if "static_cast<__int128>" in expression and "<< 64" in expression and not re.search(r"\bv\d+\b", expression):
            expressions.append(expression)
    assert expected and expressions, "fixture must exercise wide guard constants"
    declarations = "\n".join(f"  print({expression});" for expression in expressions)
    program = r"""#include <algorithm>
#include <iostream>
#include <string>
void print(__int128 value) {
  bool negative = value < 0;
  unsigned __int128 bits = static_cast<unsigned __int128>(value);
  unsigned __int128 magnitude = negative ? 0 - bits : bits;
  std::string text;
  do { text += static_cast<char>('0' + magnitude % 10); magnitude /= 10; } while (magnitude);
  if (negative) { text += '-'; }
  std::reverse(text.begin(), text.end());
  std::cout << text << '\n';
}
int main() {
""" + declarations + "\n}\n"
    source = Path(str(stem) + ".cpp")
    binary = Path(str(stem) + ".exe")
    source.write_text(program, encoding="utf-8")
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(binary)], check=True)
    actual = {int(value) for value in subprocess.check_output([str(binary)], text=True).splitlines()}
    assert actual == expected, f"wide target constants changed: {actual} != {expected}"
    print("actual C++ wide endpoint constants preserve exact IR values")


if __name__ == "__main__":
    main()
