#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE for details. Provided AS IS, WITHOUT WARRANTIES OF ANY KIND.
"""Challenge pre-plan obligation discovery with complete native effect exports.

Range enumeration is independent of native requirement selection. Original
fixture structure supplies the reference execution order. Shared translation
is explicitly trusted here; the f32/f16 extent assertions separately challenge
that boundary. No definite-write or exact last-value claim is made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from check_relations import ISL, isl_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    isl = ISL()
    results = []
    original = (HERE / "inputs/emission_contract.pto").read_text()
    reader = '        pto.tcvt ins(%in0 {rmode = #pto<round_mode ROUND>} : !pto.tile_buf<vec, 16x16xf32>) outs(%out0 : !pto.tile_buf<vec, 16x16xf16>)'
    samples = {
        "skipped_reader": original,
        "overlapping_roots": original.replace("%ia1 = arith.constant 1024", "%ia1 = arith.constant 512"),
        "additional_reader": original.replace(reader, reader + "\n" + reader),
        "equivalent_predicate": original.replace("scf.if %take {", "%true = arith.constant true\n        %not = arith.xori %take, %true : i1\n        %again = arith.xori %not, %true : i1\n        scf.if %again {"),
        "online_softmax": (ROOT / "test/samples/Qwen3DecodeA3/kernels/aiv/online_softmax.pto").read_text(),
    }
    inventories = {}
    for name, source in samples.items():
        path = args.output / (name + ".pto")
        path.write_text(source)
        process = subprocess.run([str(args.driver.resolve()), str(path), "facts"], text=True,
                                 capture_output=True, timeout=90)
        (args.output / (name + ".stdout")).write_text(process.stdout)
        (args.output / (name + ".stderr")).write_text(process.stderr)
        assert process.returncode == 0, (name, process.returncode, process.stderr[-2000:])
        data = json.loads(process.stdout)
        assert data["applied"] and data["export_complete"], (name, data)
        accesses = {a["id"]: a for a in data["accesses"]}
        phases = data["phases"]
        assert [p["id"] for p in phases] == list(range(len(phases)))
        assert len(data["orders"]) == len(data["points"]) ** 2
        assert len(data["points"]) == len(phases) + 1  # explicit virtual exit
        if name == "online_softmax":
            roles = ["body" if line.startswith("    ") else "root"
                     for line in source.splitlines() if line.lstrip().startswith("pto.t")]
            assert len(roles) == len(phases)
            first, last = roles.index("body"), len(roles) - 1 - roles[::-1].index("body")
            params = "p0"
            bounds = f"{-2**63}<=p0<={2**63-1}"
            domain = lambda p, iv: f"1<={iv}<p0" if first <= p <= last else f"{iv}=0"
        else:
            params = "p0,p1"
            bounds = f"{-2**63}<=p0<={2**63-1} and 0<=p1<=1"
            first, last = 0, len(phases) - 1
            def domain(p, iv):
                if p == len(phases):
                    return f"{iv}=0"
                guard = " and p1=1" if phases[p]["op"] in ("pto.tcvt", "pto.tstore") else ""
                return f"0<={iv}<p0" + guard
        orders = {}
        for edge in data["orders"]:
            p, q = edge["source"], edge["target"]
            pi, qi = first <= p <= last, first <= q <= last
            before = ("i<j" + (" or i=j" if p < q else "")) if pi and qi else ("true" if p < q else "false")
            expected = isl.map(f"[{params}] -> {{[{p},i] -> [{q},j]: {bounds} and {domain(p, 'i')} and {domain(q, 'j')} and ({before})}}")
            assert isl.map(isl_text(edge["relation"])).equal(expected), (name, p, q)
            orders[p, q] = expected

        def qualified(a):
            return (a["scope"] != "gm" and a["physical"] and not a["unknown_range"]
                    and a["base_addresses"] and int(a["allocation_bytes"]) > 0)

        def overlap(a, b):
            return a["scope"] == b["scope"] and any(
                max(int(x), int(y)) < min(int(x) + int(a["allocation_bytes"]), int(y) + int(b["allocation_bytes"]))
                for x in a["base_addresses"] for y in b["base_addresses"])

        expected = set()
        excluded_pairs = 0
        for p in phases:
            for q in phases:
                for kind, left, right in (("RAW", "writes", "reads"), ("WAR", "reads", "writes"), ("WAW", "writes", "writes")):
                    for a in p[left]:
                        for b in q[right]:
                            if not (qualified(accesses[a]) and qualified(accesses[b])):
                                excluded_pairs += 1
                                continue
                            if overlap(accesses[a], accesses[b]) and not orders[p["id"], q["id"]].empty():
                                expected.add((kind, p["id"], q["id"], a, b))
        actual = set()
        for requirement in data["requirements"]:
            p, q = requirement["source"], requirement["target"]
            assert isl.map(isl_text(requirement["occurrences"])).equal(orders[p, q])
            a, b = requirement["source_access"], requirement["target_access"]
            if requirement["kind"] in ("RAW", "WAR", "WAW") and qualified(accesses[a]) and qualified(accesses[b]):
                actual.add((requirement["kind"], p, q, a, b))
        assert actual == expected, (name, "missing", expected - actual, "extra", actual - expected)
        assert actual, (name, "vacuous local coverage")
        inventories[name] = actual
        if name != "online_softmax":
            convert = next(p for p in phases if p["op"] == "pto.tcvt")
            assert [int(accesses[a]["allocation_bytes"]) for a in convert["reads"]] == [1024]
            assert [int(accesses[a]["allocation_bytes"]) for a in convert["writes"]] == [512]
        results.append({"case": name, "local_requirements_checked": len(actual),
                        "order_relations_checked": len(orders), "excluded_nonlocal_or_unknown_access_pairs": excluded_pairs,
                        "work": data["work"], "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
        print(name, len(actual), "local requirements;", len(orders), "occurrence relations", flush=True)
    assert inventories["skipped_reader"] == inventories["equivalent_predicate"]
    assert len(inventories["overlapping_roots"]) > len(inventories["skipped_reader"])
    assert len(inventories["additional_reader"]) > len(inventories["skipped_reader"])
    (args.output / "summary.json").write_text(json.dumps({"status": "passed", "cases": results,
        "isl": isl.version, "driver_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
