# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check actual shared-input footprints against independent byte selections."""
import ast
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def evaluate(text, coordinates):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return coordinates[int(node.id[1:])]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return visit(node.left) + visit(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return visit(node.left) * visit(node.right)
        raise AssertionError(ast.dump(node))
    return visit(ast.parse(text, mode="eval"))


def case(arch, kind):
    tile = "!pto.tile_buf<vec, 1x256xf32, valid=?x?>"
    setters = f"pto.set_validshape %src, %one, %cols : {tile}\npto.set_validshape %dst, %one, %cols : {tile}"
    action = f"pto.texp ins(%src : {tile}) outs(%dst : {tile})"
    if kind == "equal":
        body = f"scf.if %guard {{ {setters} }} else {{ {setters} }}\n{action}"
    elif kind == "unequal":
        body = f"scf.if %guard {{ {setters} }} else {{ {setters.replace('%cols', '%other')} }}\n{action}"
    elif kind == "strong":
        body = f"scf.if %guard {{ {setters} }}\n{setters}\n{action}"
    elif kind == "late":
        body = f"scf.for %i = %zero to %count step %one {{ {action}\n{setters} }}"
    elif kind == "early":
        body = f"scf.for %i = %zero to %count step %one {{ {setters}\n{action} }}"
    elif kind == "zero":
        body = f"scf.for %i = %zero to %zero step %one {{ {setters} }}\n{action}"
    else:
        raise AssertionError(kind)
    return f"""module attributes {{pto.target_arch = "{arch}"}} {{
func.func @access(%guard: i1, %count: index) {{
%zero = arith.constant 0 : index
%one = arith.constant 1 : index
%cols = arith.constant 32 : index
%other = arith.constant 64 : index
%initial = arith.constant 128 : index
%a = arith.constant 0 : i64
%b = arith.constant 4096 : i64
%src = pto.alloc_tile addr = %a valid_row = %one valid_col = %initial : {tile}
%dst = pto.alloc_tile addr = %b valid_row = %one valid_col = %initial : {tile}
{body}
return
}}
}}"""


def main():
    tool = sys.argv[1]
    checked = 0
    with tempfile.TemporaryDirectory(prefix="pto-descriptor-access-") as scratch:
        path = Path(scratch) / "case.pto"
        for arch, kind in itertools.product(["a3", "a5"], ["equal", "unequal", "strong", "late", "early", "zero"]):
            path.write_text(case(arch, kind))
            result = subprocess.run([tool, "--step1-json", str(path)], text=True, capture_output=True,
                                    timeout=30, check=False)
            assert result.returncode == 0, (arch, kind, result.stderr)
            records = [e for e in json.loads(result.stdout)["access_records"] if e["operation"] == "pto.texp"]
            assert len(records) == 2, (arch, kind, records)
            for record in records:
                write = record["mode"] == "write"
                precise = kind not in ["unequal", "late"] and (arch == "a3" or write or kind == "zero")
                columns = 128 if kind == "zero" else 32
                size = columns * 4 if precise else 256 * 4
                base = 4096 if write else 0
                actual = set()
                for region in record["regions"]:
                    extents = [int(x) for x in region["extents"]]
                    for coordinate in itertools.product(*(range(x) for x in extents)):
                        start = evaluate(region["offset"], coordinate)
                        actual.update(range(start, start + region["element_bytes"]))
                assert actual == set(range(base, base + size)), (arch, kind, record)
                description = "declared-region" if precise else "allocation-enclosure"
                assert record["access_description"] == description, record
                assert (record["fallback_reason"] == "none") == precise, record
                checked += 1
    print(f"descriptor access facts: {checked} independently checked target/control footprints")


if __name__ == "__main__":
    main()
