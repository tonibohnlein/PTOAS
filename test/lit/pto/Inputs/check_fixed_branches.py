# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check fixed branch pruning against machine-width values and concrete execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_finite_expansion import invoke
from check_periodic_demands import closure, native


def check_graph(form, expanded):
    for sample in expanded["samples"]:
        present = [i for i, active in enumerate(sample["presence"]) if active]
        position = {site: i for i, site in enumerate(present)}
        # Every selected visit extracts the same LEFT bank and reads it in M.
        # The accumulator orders M payloads; the next extract overwrites LEFT.
        reads = [{"mat"} if form["sites"][site]["pipe"] == 3 else {"left", "right"}
                 for site in present]
        writes = [{"left"} if form["sites"][site]["pipe"] == 3 else {"acc"} for site in present]
        expected = native([form["sites"][site]["pipe"] for site in present])
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                if writes[a] & (reads[b] | writes[b]) or reads[a] & writes[b]:
                    expected.add((2 * a + 1, 2 * b))
        target, _ = closure(2 * len(present), expected)
        for name in ["generators", "retained"]:
            edges = native([form["sites"][site]["pipe"] for site in present])
            for a, b, active in sample[name] + sample["native"]:
                if active:
                    edges.add((2 * position[a] + 1, 2 * position[b]))
            assert closure(2 * len(present), edges)[0] == target, (name, sample)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text().replace("attributes {test.finite_expansion_session}", "")
    source = source.replace("%slot = arith.remui %i, %two : index", "%slot = arith.constant 0 : index")
    source = source.replace("} else {\n        pto.tmatmul ins(%left, %right : !left, !right)",
                            "} else {\n        pto.tmatmul.acc ins(%acc, %left, %right : !acc, !left, !right)")
    marker = "%first = arith.cmpi eq, %outer, %i : index"
    cases = [
        ("varying", "%first = arith.cmpi eq, %i, %one : index", [True, False]),
        ("wrap", "%a = arith.constant 127 : i8\n      %b = arith.constant 1 : i8\n"
         "      %sum = arith.addi %a, %b : i8\n      %first = arith.cmpi slt, %sum, %b : i8", [True, True]),
        ("unsigned", "%a = arith.constant -1 : i8\n      %b = arith.constant 1 : i8\n"
         "      %first = arith.cmpi ult, %a, %b : i8", [False, False]),
        ("nan", "%a = arith.constant 0x7FC00000 : f32\n      %b = arith.constant 0.0 : f32\n"
         "      %first = arith.cmpf ueq, %a, %b : f32", [True, True]),
        ("runtime", marker, None),
        ("unknown-and-false", "%unknown = arith.cmpi eq, %outer, %i : index\n"
         "      %false = arith.constant false\n      %first = arith.andi %unknown, %false : i1", None),
        ("invalid-fptosi", "%a = arith.constant 0x7FC00000 : f32\n"
         "      %b = arith.constant 0 : i8\n      %sum = arith.fptosi %a : f32 to i8\n"
         "      %first = arith.cmpi eq, %sum, %b : i8", None),
        ("overflow", "%a = arith.constant 127 : i8\n      %b = arith.constant 1 : i8\n"
         "      %sum = arith.addi %a, %b overflow<nsw> : i8\n"
         "      %first = arith.cmpi slt, %sum, %b : i8", None),
        ("shift", "%a = arith.constant 1 : i8\n      %b = arith.constant 8 : i8\n"
         "      %sum = arith.shli %a, %b : i8\n      %first = arith.cmpi eq, %sum, %a : i8", None),
        ("divide", "%a = arith.constant 1 : i8\n      %b = arith.constant 0 : i8\n"
         "      %sum = arith.divsi %a, %b : i8\n      %first = arith.cmpi eq, %sum, %a : i8", None),
    ]
    with tempfile.TemporaryDirectory(prefix="fixed-branches-") as directory:
        path = Path(directory) / "case.pto"
        for name, expression, choices in cases:
            rendered = source.replace(marker, expression)
            path.write_text(rendered)
            for policy in ["may-not-alias", "may-alias"]:
                result = subprocess.run([tool, "--gm-alias=" + policy, "--finite-expansion", str(path)],
                                        capture_output=True, text=True, timeout=60, check=True)
                documents = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines()
                             if line.startswith("expanded-json ")]
                expanded = documents[0]
                assert "finite-expansion: source-unchanged" in result.stdout
                if choices is None:
                    # Unknown or invalid expressions may retain both arms or
                    # fail an existing predicate adapter, but cannot prune.
                    if not expanded["error"]:
                        assert expanded["pruned_arms"] == 0, (name, expanded)
                        assert expanded["sites"] == 6, (name, expanded)
                    assert path.read_text() == rendered
                    continue
                form, expanded = invoke(tool, path, policy)
                assert expanded["pruned_arms"] == 2, (name, expanded)
                assert [site["fixed"] for site in form["sites"]] == [[1], [1], [3], [3]]
                expected = [op for choice in choices for op in
                            ["pto.textract", "pto.tmatmul" if choice else "pto.tmatmul.acc"]]
                assert [site["op"] for site in form["sites"]] == expected, (name, form)
                assert all(sample["presence"] == [1] * 4 for sample in expanded["samples"])
                check_graph(form, expanded)
                assert path.read_text() == rendered
        # A proved inactive arm can contain an otherwise unsupported dynamic
        # loop; it contributes no occurrences. Unknown conditions cannot do so.
        dead = source.replace(marker, "%first = arith.constant false")
        dead = dead.replace("        pto.tmatmul ins(%left, %right : !left, !right) outs(%acc : !acc)",
            "        scf.for %k = %zero to %outer step %one {\n"
            "          pto.tmatmul ins(%left, %right : !left, !right) outs(%acc : !acc)\n        }")
        path.write_text(dead)
        for policy in ["may-not-alias", "may-alias"]:
            form, expanded = invoke(tool, path, policy)
            assert expanded["pruned_arms"] == 2 and expanded["sites"] == 4, expanded
            assert [site["op"] for site in form["sites"]] == ["pto.textract", "pto.tmatmul.acc"] * 2
            check_graph(form, expanded)
        # Index arithmetic must use the original scoped target layout. The
        # 32-bit adapter may reject it, but a 64-bit fold cannot select an arm.
        narrow = source.replace(marker, "%sum = arith.addi %i, %one : index\n"
            "      %first = arith.cmpi eq, %sum, %two : index")
        narrow = narrow.replace('pto.target_arch = "a3"', 'pto.target_arch = "a3", '
            'dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>')
        path.write_text(narrow)
        result = subprocess.run([tool, "--finite-expansion", str(path)], capture_output=True,
                                text=True, timeout=60, check=True)
        expanded = next(json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines()
                        if line.startswith("expanded-json "))
        assert expanded["error"] or expanded["pruned_arms"] == 0, expanded
    print("fixed-branches: machine-width choices, unknown guards, target layout and exact graphs checked")


if __name__ == "__main__":
    main()
