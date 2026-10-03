# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile actual emitted periodic guards and check wide Index boundary values."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def extracted_guard(text, name):
    function = text.split("AICORE void " + name + "(", 1)[1].split("AICORE void ", 1)[0]
    upper = re.search(r"int64_t (\w+)\) \{", function).group(1)
    loop = re.search(r"for \(int64_t (\w+) = .*?\) \{", function)
    iv = loop.group(1)
    before, body = function[:loop.start()], function[loop.end():]
    constants = re.findall(r"^\s*((?:const )?(?:__int128|int64_t|int32_t) (\w+) = .*?;)$", before, re.M)
    aliases = re.findall(r"^\s*((?:__int128|int64_t|int32_t) \w+ = .*?;)$", body.split("TLOAD(", 1)[0], re.M)
    guards = re.findall(r"if \((.*)\) \{", body)
    if len(guards) != 2:
        raise ValueError("expected paired offset-one guards in emitted C++")
    used = set(re.findall(r"\bv\d+\b", " ".join([*guards, *aliases])))
    constants = [declaration for declaration, variable in constants if variable in used]
    return "\n".join([
        f"bool check_{name}(int64_t n, int64_t iteration, bool incoming, bool outgoing) {{",
        f"int64_t {upper} = n; int64_t {iv} = iteration;", *constants, *aliases,
        f"return (({guards[0]}) == incoming) && (({guards[1]}) == outgoing);", "}"])


def main():
    output, artifact = map(Path, sys.argv[1:])
    text = output.read_text(encoding="utf-8")
    sources = ["#include <cstdint>"]
    checks = []
    for name, lower, step in (("periodic_pipeline", 0, 1), ("periodic_shifted", -5, 2)):
        sources.append(extracted_guard(text, name))
        maximum = (1 << 63) - 1
        samples = [(lower + step, lower), (lower + 2 * step, lower),
                   (lower + 2 * step, lower + step),
                   ((1 << 40) + 1, (1 << 40) - 1),
                   ((1 << 40) + 3, (1 << 40) - 1),
                   (maximum, maximum - (2 if step == 2 else 1)),
                   (maximum, maximum - (4 if step == 2 else 2))]
        for bound, iv in samples:
            incoming = "true" if iv - step >= lower else "false"
            outgoing = "true" if iv + step < bound else "false"
            checks.append(f"if (!check_{name}({bound}LL, {iv}LL, {incoming}, {outgoing})) return 1;")
    sources += ["int main() {", *checks, "return 0; }"]
    source = Path(str(artifact) + ".cpp").resolve()
    binary = Path(str(artifact) + ".exe").resolve()
    source.write_text("\n".join(sources) + "\n", encoding="utf-8")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run([*compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    print("emitted C++ periodic guards preserve wide Index boundary values")


if __name__ == "__main__":
    main()
