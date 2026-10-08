# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check whole-visit choice products, shared node reuse and the expansion limit."""
from pathlib import Path
import subprocess
import sys
import tempfile


def invoke(tool, path):
    result = subprocess.run([tool, "--finite-visit-input-checks", str(path)],
                            capture_output=True, text=True, check=False, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return result.stdout


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    result = invoke(tool, fixture)
    assert "4 types, 16 cached boundaries, 7 analyzed nodes, demand-only exports" in result, result
    # Nine independent Boolean arguments exceed the explicit 256-type adapter.
    # Recognition must report this before constructing child/pair analyses.
    begin = source.index("      scf.if %choose_first")
    end = source.index('      pto.textract', begin)
    choice = source[begin:end]
    suffix = source.index('      pto.textract', source.index('"second_else"'))
    independent = "".join(choice.replace("%choose_first", f"%independent{i}") for i in range(9))
    expanded = source[:begin] + independent + source[suffix:]
    arguments = ", ".join(f"%independent{i}: i1" for i in range(9))
    expanded = expanded.replace("%cut: index)", f"%cut: index, {arguments})")
    expanded = expanded.replace("test.whole_visit", "test.type_expansion_limit")
    with tempfile.TemporaryDirectory(prefix="finite-visit-whole-") as scratch:
        path = Path(scratch) / "case.pto"
        path.write_text(expanded)
        result = invoke(tool, path)
        assert "explicit producer limit retained" in result, result
        # Reusing one SSA condition nine times has only two reachable types.
        # Each reaches both pipes and refreshes both written cells. Impossible
        # independent choices could omit either refresh and reject the contract.
        extract = "pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%left : !left)"
        multiply = "pto.tmatmul ins(%left, %right : !left, !right) outs(%acc : !acc)"
        decisions = []
        for i in range(9):
            then_op, else_op = (extract, multiply) if i % 2 == 0 else (multiply, extract)
            decisions.append(f"      scf.if %choose_first {{\n        {then_op}\n"
                             f"      }} else {{\n        {else_op}\n      }}\n")
        body_begin = source.index("      pto.textract")
        body_end = source.index("    }\n    return", suffix)
        repeated = source[:body_begin] + "".join(decisions) + source[body_end:]
        repeated = repeated.replace("test.whole_visit", "test.repeated_choices = 9 : i64")
        path.write_text(repeated)
        result = invoke(tool, path)
        assert "2 types, 4 cached boundaries, 18 analyzed nodes, demand-only exports" in result, result
        # The same whole-visit adapter automatically projects an original GM
        # family owned by its outer visit, retaining the complete internal body.
        owned = source.replace("%cut: index)", "%cut: index, %p: !pto.ptr<f32, gm>)")
        owned = owned.replace("test.whole_visit}", "test.whole_visit, test.owned_prefix}")
        owned = owned.replace("    %base =", "    %value = arith.constant 1.0 : f32\n    %base =")
        insertion = owned.index("      pto.textract")
        owned = (owned[:insertion] +
                 '      pto.store %value, %p[%i] {test.label = "owned"} : !pto.ptr<f32, gm>, f32\n' +
                 owned[insertion:])
        path.write_text(owned)
        result = invoke(tool, path)
        assert "4 types, 16 cached boundaries, 7 analyzed nodes, demand-only exports" in result, result
    print("finite whole visits: common prefix/middle/suffix, sequential choices, node reuse and type limit passed")


if __name__ == "__main__":
    main()
