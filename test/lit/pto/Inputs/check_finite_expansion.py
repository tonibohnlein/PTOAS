# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare original-coordinate finite expansion with small concrete executions."""
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from check_periodic_demands import closure, native


def invoke(tool, path, policy):
    result = subprocess.run([tool, "--gm-alias=" + policy, "--finite-expansion", str(path)],
                            capture_output=True, text=True, timeout=60, check=True)
    documents = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines()
                 if line.startswith("arithmetic-json ") or line.startswith("expanded-json ")]
    expanded = next(item for item in documents if "samples" in item or "error" in item)
    assert not expanded["error"], expanded
    form = next(item for item in documents if item.get("region") == "anonymous")
    assert expanded["exports_blocked"]
    return form, expanded


def check(form, expanded, coordinates, slots, parameter_reader=False, physical=True, protected_acc=False):
    expected_coordinates = [list(iv) if isinstance(iv, tuple) else [iv] for iv in coordinates for _ in range(3)]
    assert [site["fixed"] for site in form["sites"]] == expected_coordinates
    assert [site["pipe"] for site in form["sites"]] == [p for _ in coordinates for p in (3, 2, 2)]
    assert all(site["depth"] == 0 for site in form["sites"])
    for sample in expanded["samples"]:
        value = sample["parameters"][0] if sample["parameters"] else 0
        present = [i for i, active in enumerate(sample["presence"]) if active]
        expected_presence = []
        for visit, coordinate in enumerate(coordinates):
            induction = coordinate[-1] if isinstance(coordinate, tuple) else coordinate
            expected_presence.extend([3 * visit, 3 * visit + (1 if value == induction else 2)])
        assert present == expected_presence, (sample, expected_presence)
        if physical:
            # Physical spaces are the IR contract: MAT=2, LEFT=3, RIGHT=4, ACC=5.
            # 16x16 f16 banks occupy 512 bytes; the f32 accumulator occupies 1024.
            access = {}
            for relation in form["relations"]:
                if relation["kind"] not in (4, 5):
                    continue
                key = (relation["source"], relation["kind"], relation["space"])
                actual_bytes = access.setdefault(key, set())
                for piece in relation["pieces"]:
                    if piece["empty"]:
                        continue
                    linear = []
                    for row in piece["rows"]:
                        assert len(row["coefficients"]) == 1 + len(sample["parameters"])
                        constant = row["constant"] + sum(a * b for a, b in
                            zip(row["coefficients"][1:], sample["parameters"]))
                        linear.append((row["coefficients"][0], constant, row["equality"]))
                    lower, upper = -1, 2048
                    for coefficient, constant, equality in linear:
                        if coefficient == 0:
                            if (constant != 0 if equality else constant < 0):
                                lower, upper = 1, 0
                                break
                        elif equality:
                            if (-constant) % coefficient:
                                lower, upper = 1, 0
                                break
                            point = (-constant) // coefficient
                            lower, upper = max(lower, point), min(upper, point)
                        elif coefficient > 0:
                            lower = max(lower, -(constant // coefficient))
                        else:
                            upper = min(upper, constant // (-coefficient))
                    actual_bytes.update(range(lower, upper + 1))
            expected_access = {}
            for i in present:
                visit, kind = divmod(i, 3)
                bank = slots[visit]
                reader = value if parameter_reader else bank
                if kind == 0:
                    expected_access[i, 4, 2] = set(range(512))
                    expected_access[i, 5, 3] = set(range(512 * bank, 512 * (bank + 1)))
                else:
                    expected_access[i, 4, 3] = set(range(512 * reader, 512 * (reader + 1)))
                    expected_access[i, 4, 4] = set(range(512))
                    expected_access[i, 5, 5] = set(range(1024))
            assert {key: data for key, data in access.items() if data} == expected_access
        pipes = [form["sites"][i]["pipe"] for i in present]
        effects = []
        for i in present:
            visit, kind = divmod(i, 3)
            bank = "left" + str(slots[visit])
            reader = "left" + str(value) if parameter_reader else bank
            effects.append(({"mat"}, {bank}) if kind == 0 else ({reader, "right"}, {"acc"}))
        expected = native(pipes)
        for a, (reads, writes) in enumerate(effects):
            for b in range(a + 1, len(effects)):
                other_reads, other_writes = effects[b]
                a_matrix = form["sites"][present[a]]["pipe"] == 2
                b_matrix = form["sites"][present[b]]["pipe"] == 2
                target_acc = form["sites"][present[b]]["op"] == "pto.tmatmul.acc"
                protected = {"acc"} if protected_acc and a_matrix and b_matrix and target_acc else set()
                if (writes - protected) & (other_reads | other_writes) or (reads - protected) & other_writes:
                    expected.add((2 * a + 1, 2 * b))
        target, _ = closure(2 * len(present), expected)
        index = {original: i for i, original in enumerate(present)}
        for name in ["generators", "retained"]:
            actual = native(pipes)
            for source, destination, active in sample[name] + sample["native"]:
                assert active in (0, 1), (name, active)
                if active:
                    assert source in index and destination in index
                    actual.add((2 * index[source] + 1, 2 * index[destination]))
            graph, _ = closure(2 * len(present), actual)
            assert graph == target, (name, coordinates, slots, value, graph, target)
        # Reduction must produce exactly the nonnative covers of the selected graph.
        covers = set()
        base, _ = closure(2 * len(present), native(pipes))
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                source, destination = 2 * a + 1, 2 * b
                if not (target[source] >> destination) & 1 or (base[source] >> destination) & 1:
                    continue
                if not any(k not in (source, destination) and (target[source] >> k) & 1
                           and (target[k] >> destination) & 1 for k in range(2 * len(present))):
                    covers.add((a, b))
        retained = {(index[a], index[b]) for a, b, active in sample["retained"] if active}
        assert retained == covers, (retained, covers)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    begin, end = source.index("    // BODY"), source.index("    // END_BODY")
    template = source[begin:end]
    with tempfile.TemporaryDirectory(prefix="finite-expansion-") as directory:
        path = Path(directory) / "case.pto"
        for policy in ["may-not-alias", "may-alias"]:
            for lower, upper, step in [(0, 0, 1), (0, 1, 1), (0, 4, 1), (1, 5, 2)]:
                names = {0: "%zero", 1: "%one", 2: "%two", 3: "%three", 5: "%five"}
                bound = names.get(upper, "%four")
                body = template.replace("%one to %five step %two", f"{names[lower]} to {bound} step {names[step]}")
                rendered = source[:begin] + body + source[end:]
                rendered = rendered.replace("%five = arith.constant 5 : index",
                                            "%five = arith.constant 5 : index\n    %four = arith.constant 4 : index")
                path.write_text(rendered)
                form, expanded = invoke(tool, path, policy)
                coordinates = list(range(lower, upper, step))
                check(form, expanded, coordinates, [iv % 2 for iv in coordinates])
                carried = re.sub(r"scf.for (%i = .*?) \{",
                                 r"%result = scf.for \1 iter_args(%bank = %one) -> (index) {", rendered)
                carried = carried.replace("%slot = arith.remui %i, %two : index",
                                          "%next = arith.addi %bank, %one : index\n"
                                          "      %slot = arith.remui %next, %two : index")
                carried = carried.replace("    } {frontier.test_expanded_region}",
                                          "      scf.yield %slot : index\n"
                                          "    } {frontier.test_expanded_region}")
                path.write_text(carried)
                form, expanded = invoke(tool, path, policy)
                check(form, expanded, coordinates, [i % 2 for i in range(len(coordinates))])
                dynamic = rendered.replace(
                    "      %first = arith.cmpi",
                    "      %read = pto.multi_tile_get %banks[%outer] : !banks -> !left\n"
                    "      %first = arith.cmpi")
                dynamic = dynamic.replace("ins(%left, %right", "ins(%read, %right")
                path.write_text(dynamic)
                form, expanded = invoke(tool, path, policy)
                check(form, expanded, coordinates, [iv % 2 for iv in coordinates], parameter_reader=True)
            nested = template.replace(" {frontier.test_expanded_region}", "")
            body = ("    scf.for %j = %zero to %two step %one {\n" + nested +
                    "    } {frontier.test_expanded_region}\n")
            path.write_text(source[:begin] + body + source[end:])
            form, expanded = invoke(tool, path, policy)
            coordinates = [(j, i) for j in range(2) for i in (1, 3)]
            check(form, expanded, coordinates, [1] * 4)
        exceeded = source.replace("%five = arith.constant 5", "%five = arith.constant 70000")
        path.write_text(exceeded)
        result = subprocess.run([tool, "--finite-expansion", str(path)], capture_output=True,
                                text=True, timeout=60, check=True)
        limit = next(json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines()
                     if line.startswith("expanded-json "))
        assert limit["error"] and "samples" not in limit
        protected = source.replace("16x16", "64x64")
        protected = protected.replace("%one to %five step %two", "%zero to %five step %one")
        for accumulate in [False, True]:
            candidate = protected
            if accumulate:
                marker = "      } else {\n        pto.tmatmul ins(%left, %right : !left, !right)"
                candidate = candidate.replace(marker,
                    "      } else {\n        pto.tmatmul.acc ins(%acc, %left, %right : !acc, !left, !right)")
            path.write_text(candidate)
            for policy in ["may-not-alias", "may-alias"]:
                form, expanded = invoke(tool, path, policy)
                check(form, expanded, list(range(5)), [i % 2 for i in range(5)],
                      physical=False, protected_acc=accumulate)
        prerequisite = """!tile = !pto.tile_buf<vec, 16x16xf32>
module {
  func.func @prerequisites() {
    %base = arith.constant 0 : i64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    %tile = pto.alloc_tile addr = %base : !tile
    scf.for %i = %zero to %three step %one {
      %value = pto.tgetval ins(%tile, %zero : !tile, index) outs : f32
      pto.tsetval ins(%zero, %value : index, f32) outs(%tile : !tile)
    } {frontier.test_expanded_region}
    return
  }
}
"""
        path.write_text(prerequisite)
        for policy in ["may-not-alias", "may-alias"]:
            form, expanded = invoke(tool, path, policy)
            assert [site["fixed"] for site in form["sites"]] == [[i] for i in range(3) for _ in range(2)]
            for sample in expanded["samples"]:
                assert {(a, b) for a, b, active in sample["native"] if active} == {(0, 1), (2, 3), (4, 5)}
                assert not any(active for _, _, active in sample["retained"])
        catalog = json.loads((Path(fixture).resolve().parents[3] /
                              "docs/designs/frontier-tractable-catalog.json").read_text())
        root = Path(fixture).resolve().parents[3]
        for name in ["tilelang_gemm", "pypto_gemm", "persistent_gemm"]:
            case = next(item for item in catalog["inputs"] if item["id"] == name)
            loop = next(item for item in case["loops"] if item["constant_trip_count"] == 4)
            lines = (root / case["input"]).read_text().splitlines()
            start = loop["source_line"] - 1
            assert "scf.for " in lines[start]
            balance = 0
            for finish in range(start, len(lines)):
                balance += lines[finish].count("{") - lines[finish].count("}")
                if balance == 0:
                    break
            lines[finish] += " {frontier.test_expanded_region}"
            path.write_text("\n".join(lines) + "\n")
            for policy in ["may-not-alias", "may-alias"]:
                print("finite expansion GEMM:", name, policy, flush=True)
                form, expanded = invoke(tool, path, policy)
                assert [site["fixed"] for site in form["sites"]] == [[i] for i in range(4) for _ in range(4)]
                assert [site["pipe"] for site in form["sites"]] == [3, 3, 2, 2] * 4
                assert expanded["samples"], (name, "missing sample bindings")
                for sample in expanded["samples"]:
                    present = [i for i, active in enumerate(sample["presence"]) if active]
                    assert len(present) == 12
                    assert all(present[3 * i:3 * i + 2] == [4 * i, 4 * i + 1] for i in range(4))
                    assert all(present[3 * i + 2] in [4 * i + 2, 4 * i + 3] for i in range(4))
                    # Two independent L0 banks: both extracts precede MAD, and
                    # a reused bank cannot be overwritten until the old MAD completes.
                    pipes = [3, 3, 2] * 4
                    expected = native(pipes)
                    for i in range(4):
                        expected.update({(6 * i + 1, 6 * i + 4), (6 * i + 3, 6 * i + 4)})
                        if i + 2 < 4:
                            expected.update({(6 * i + 5, 6 * (i + 2)), (6 * i + 5, 6 * (i + 2) + 2)})
                    target, _ = closure(24, expected)
                    index = {site: i for i, site in enumerate(present)}
                    for kind in ["generators", "retained"]:
                        actual = native(pipes)
                        for a, b, active in sample[kind] + sample["native"]:
                            if active:
                                assert active == 1 and a in index and b in index
                                actual.add((2 * index[a] + 1, 2 * index[b]))
                        graph, _ = closure(24, actual)
                        assert graph == target, (name, policy, kind, graph, target)
    print("finite expansion: exact original visits, conditional conflict closure and covers; both alias policies")


if __name__ == "__main__":
    main()
