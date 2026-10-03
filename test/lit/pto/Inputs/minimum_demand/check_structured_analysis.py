# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Execute bounded source control independently, then compare imported relations."""

import copy
import itertools
import json
import pathlib
import subprocess
import sys
import time


DECLS = """!vec = !pto.tile_buf<vec, 16x16xf32>
module attributes {pto.target_arch = "a3"} {
func.func private @v(!vec) attributes {pto.tileop.helper,
  pto.tileop.kind = "vector", pto.tileop.effects = ["readwrite"]}
func.func private @m(!vec) attributes {pto.tileop.helper, pto.tileop.kind = "cube", pto.tileop.effects = ["readwrite"]}
"""
CONSTANTS = """%a = pto.alloc_tile : !vec
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%cm2 = arith.constant -2 : index
"""


def call(pipe="v"):
    return f"func.call @{pipe}(%a) : (!vec) -> ()"


def source(body, args="%n: index", constants=CONSTANTS):
    return DECLS + f"func.func @test({args}) {{\n" + constants + body + "\nreturn\n}\n}"


def fixture(name, body, traces, depths, reads=(), writes=(), **options):
    contexts = []
    candidates = sorted(set(point for _, trace in traces for point in trace))
    # Probe every small coordinate tuple, including holes and opposite branches.
    coordinates = [value for _, trace in traces for point in trace for value in point[1:]]
    envelope = range(min(coordinates, default=0) - 1, max(coordinates, default=0) + 2)
    for site, depth in enumerate(depths):
        for coordinates in itertools.product(envelope, repeat=depth):
            point = (site,) + coordinates
            if point not in candidates:
                candidates.append(point)
        point = (site,) + (99,) * depth
        if point not in candidates:
            candidates.append(point)
    for values, trace in traces:
        contexts.append({"values": values, "points": candidates, "trace": trace})
    bounds = [[min(values[i] for values, _ in traces), max(values[i] for values, _ in traces)]
              for i in range(len(traces[0][0]))] if traces else []
    options.setdefault("bounds", bounds)
    return {"name": name, "source": source(body), "depths": depths,
            "reads": reads, "writes": writes, "contexts": contexts, **options}


def corpus():
    cases = []
    body = call() + "\nscf.for %i = %cm2 to %n step %c2 {\n" + call("m") + "\n}\n" + call()
    cases.append(fixture("signed-nonunit-outside", body,
        [([n], [(0,)] + [(1, i) for i in range(-2, n, 2)] + [(2,)]) for n in [-3, 0, 3]],
        [0, 1, 0], reads=[[1, 0, 1]], writes=[[0, 0], [2, 0]]))
    body = ("scf.for %i = %c0 to %n step %c1 {\n" + call() +
            "\nscf.for %j = %c0 to %i step %c1 {\n" + call("m") + "\n}\n" + call() + "\n}")
    traces = []
    for n in [0, 2, 4]:
        trace = []
        for i in range(n):
            trace += [(0, i)] + [(1, i, j) for j in range(i)] + [(2, i)]
        traces.append(([n], trace))
    cases.append(fixture("nested-triangle-resets", body, traces, [1, 2, 1],
                         writes=[[0, 0, 1]], reads=[[1, 0, 0, 1], [2, 0, 1]]))
    body = ("scf.for %i = %c0 to %n step %c1 {\n" + call() +
            "\n}\nscf.for %j = %cm2 to %n step %c2 {\n" + call("m") + "\n}")
    cases.append(fixture("sibling-loop-identities", body,
        [([n], [(0, i) for i in range(n)] + [(1, j) for j in range(-2, n, 2)]) for n in [0, 3]],
        [1, 1], writes=[[0, 0, 1]], reads=[[1, 0, 1]]))
    body = """affine.for %i = max affine_map<()[s0] -> (-2, s0 - 3)>()[%n]
        to min affine_map<()[s0] -> (s0, 4)>()[%n] step 2 {
        affine.if affine_set<(d0) : (d0 mod 2 == 0)>(%i) {
    """ + call() + "\n} else {\n" + call("m") + "\n}\n}"
    cases.append(fixture("affine-max-min-origin-and-parity", body,
        [([n], [(0 if i % 2 == 0 else 1, i) for i in range(max(-2, n - 3), min(n, 4), 2)]) for n in [-2, 0, 3, 4, 6]],
        [1, 1], writes=[[0, 0, 1]], reads=[[1, 0, 1]]))
    body = """scf.for %i = %cm2 to %n step %c1 {
        %q = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%i)
        %r = affine.apply affine_map<(d0) -> (d0 mod 2)>(%i)
        %t = arith.cmpi slt, %q, %c0 : index
        %u = arith.cmpi eq, %r, %c1 : index
        %both = arith.xori %t, %u : i1
        scf.if %both {
    """ + call() + "\n} else {\n" + call("m") + "\n}\n}"
    cases.append(fixture("negative-floor-Boolean-negation", body,
        [([n], [(0 if ((i // 2 < 0) != (i % 2 == 1)) else 1, i) for i in range(-2, n)]) for n in [0, 4]],
        [1, 1]))
    body = """affine.for %i = -2 to %n {
      affine.if affine_set<(d0) : (d0 ceildiv 2 - 1 >= 0, d0 mod 3 == 1)>(%i) {
    """ + call() + "\n} else {\n" + call("m") + "\n}\n}"
    cases.append(fixture("affine-else-total-divisions", body,
        [([4], [(0 if (-((-i) // 2) >= 1 and i % 3 == 1) else 1, i) for i in range(-2, 4)])], [1, 1]))
    body = "scf.if %flag {\n" + call() + "\n} else {\n" + call("m") + "\n}"
    immutable = fixture("immutable-i1", body, [([flag], [(0 if flag else 1,)]) for flag in [0, 1]], [0, 0])
    immutable["source"] = source(body, "%flag: i1")
    cases.append(immutable)
    opaque = copy.deepcopy(immutable)
    opaque.update(name="qualified-opaque-i1", opaque=True)
    opaque["source"] = source("%flag = arith.cmpi ult, %n, %c2 {sync.test.parameter} : index\n" + body)
    opaque["bounds"] = [[0, 3], [0, 1]]
    opaque["contexts"] = [
        {"values": [n, int(n < 2)], "points": [(0,), (1,)], "trace": [(0 if n < 2 else 1,)]}
        for n in [0, 3]]
    opaque["contexts"].append({"values": [0, 0], "points": [(0,), (1,)], "trace": []})
    cases.append(opaque)
    body = call() + "\n" + call("m")
    cases.append(fixture("sole-demand-downstream", body, [([], [(0,), (1,)])], [0, 0],
        writes=[[0, 0]], reads=[[1, 0]], downstream=True, materialize=True))
    cases[-1]["source"] = source(body, "")
    body = "scf.for %i = %c0 to %n step %c1 {\n" + call() + "\n}"
    cases.append(fixture("loop-rmw-downstream", body, [([2], [(0, 0), (0, 1)])], [1],
        writes=[[0, 0, 0]], reads=[[0, 0, 0]], downstream=True))
    empty = fixture("empty-and-inadmissible", "scf.for %i = %c0 to %n step %c1 {" + call() + "}",
                    [([0], []), ([-1], [])], [1], bounds=[[0, 4]])
    cases.append(empty)
    cases.append(fixture("no-effect-phase", call(), [([], [(0,)])], [0]))
    cases[-1]["source"] = source(call(), "")
    macro_path = pathlib.Path(__file__).resolve().parents[2] / "tgather_compare_macro_sync_model.pto"
    macro = fixture("multiphase-anchor", "", [([], [(i,) for i in range(4)])], [0] * 4)
    macro["source"] = macro_path.read_text().replace("@tgather_compare_macro_model", "@test")
    cases.append(macro)
    huge = fixture("wide-source-integers", "", [([0], [(0,)]), ([1], [(0,)])], [0])
    huge["source"] = source("%wide = arith.constant 1267650600228229401496703205376 : i128\n"
                            "%sum = arith.addi %wide, %wide : i128\n"
                            "%flag = arith.cmpi slt, %n, %sum : i128\n"
                            "scf.if %flag {" + call() + "}", "%n: i128")
    cases.append(huge)
    casts = fixture("qualified-index-cast", "", [([2], [(0, 0), (0, 1)])], [1])
    casts["source"] = source("%end = arith.index_cast %n : i64 to index\n"
                             "scf.for %i = %c0 to %end step %c1 {" + call() + "}", "%n: i64")
    cases.append(casts)
    negatives = []
    def reject(base, name, status, **changes):
        value = copy.deepcopy(base)
        value.update(name=name, expected_status=status, **changes)
        negatives.append(value)
    reject(cases[0], "missing-index-qualification", 6, qualify=False)
    reject(cases[0], "bad-provider-role-before-qualification", 1, wrong_role=True)
    reject(cases[0], "foreign-schema-before-qualification", 1, foreign_schema=True)
    reject(cases[0], "provider-declines", 5, decline=True)
    reject(macro, "missing-multiphase-order", 6, phase_order=False)
    reject(cases[0], "variable-step", 2, source=cases[0]["source"].replace("step %c2", "step %n"))
    reject(cases[0], "unsigned-condition", 4, reads=[], writes=[],
           source=source("%flag = arith.cmpi ult, %n, %c2 : index\n" +
                         body.replace("scf.for %i = %c0 to %n step %c1", "scf.if %flag")))
    reject(immutable, "signed-i1-condition", 4, reads=[], writes=[],
           source=source("%false = arith.constant false\n%flag = arith.cmpi slt, %arg, %false : i1\n" +
                         "scf.if %flag {" + call() + "}", "%arg: i1"))
    reject(cases[0], "phase-free-loop-needs-qualification", 6, qualify=False,
           source=source("scf.for %i = %c0 to %n step %c1 {}\n" + call()), reads=[], writes=[])
    reject(cases[0], "unknown-varying-guard", 3,
           source=source("scf.for %i = %c0 to %n step %c1 { %r = arith.remsi %i, %c2 : index\n"
                         "%flag = arith.cmpi eq, %r, %c0 : index\nscf.if %flag {" + call() + "}}"),
           reads=[], writes=[])
    reject(macro, "invalid-multiphase-permutation", 1, bad_phase_order=True)
    reject(cases[0], "unstructured-self-loop", 2,
           source=source(call(), ""), bad_terminator=True, reads=[], writes=[], bounds=[])
    reject(cases[0], "valid-multiblock-control", 1,
           source=source("cf.br ^next\n^next:\n" + call(), ""), reads=[], writes=[], bounds=[])
    reject(cases[0], "unsupported-while", 2,
           source=source("scf.while : () -> () { %false = arith.constant false\n"
                         "scf.condition(%false) } do { scf.yield }\n" + call(), ""),
           reads=[], writes=[], bounds=[])
    reject(huge, "uncertified-arithmetic", 6, qualify=False)
    return cases + negatives


def closure(edges):
    reach = [set(row) | {i} for i, row in enumerate(edges)]
    for middle in range(len(edges)):
        for row in reach:
            if middle in row:
                row.update(reach[middle])
    return reach


def graph(test, trace, pipes):
    edges = [set() for _ in range(2 * len(trace))]
    last = {}
    for i, point in enumerate(trace):
        edges[2 * i].add(2 * i + 1)
        pipe = pipes[point[0]]
        if pipe in last:
            previous = last[pipe]
            edges[2 * previous].add(2 * i)
            edges[2 * previous + 1].add(2 * i + 1)
        last[pipe] = i
    native = closure(edges)
    def cells(point, mode):
        return {row[1] + sum(a * b for a, b in zip(row[2:], point[1:])) for row in test[mode] if row[0] == point[0]}
    generators = set()
    for i, a in enumerate(trace):
        for j in range(i + 1, len(trace)):
            b = trace[j]
            if cells(a, "writes") & (cells(b, "reads") | cells(b, "writes")) or cells(a, "reads") & cells(b, "writes"):
                edges[2 * i + 1].add(2 * j)
                generators.add((2 * i + 1, 2 * j))
    reach = closure(edges)
    minimum = {(a, b) for a, b in generators if b not in native[a] and
               not any(z != a and z != b and z in reach[a] and b in reach[z] for z in range(len(edges)))}
    return reach, minimum


def verify(test, result):
    assert result["unchanged"], test["name"]
    assert result["status"] == test.get("expected_status", 0), (test["name"], result)
    if test.get("expected_status"):
        if test.get("wrong_role") or test.get("foreign_schema"):
            assert result["qualifications"] == 0
        return
    assert result["depths"] == test["depths"]
    assert len(result["contexts"]) == len(test["contexts"])
    for expected, actual in zip(test["contexts"], result["contexts"]):
        admitted = all(low <= val <= high for val, (low, high) in zip(expected["values"], test.get("bounds", [])))
        if test.get("opaque"):
            admitted = admitted and expected["values"][1] == int(expected["values"][0] < 2)
        assert actual["admitted"] == admitted
        if not admitted:
            continue
        trace = [tuple(point) for point in expected["trace"]]
        positions = {point: i for i, point in enumerate(trace)}
        points = [tuple(point) for point in expected["points"]]
        assert actual["present"] == [point in positions for point in points], test["name"]
        ref = [[a in positions and b in positions and positions[a] < positions[b] for b in points] for a in points]
        assert actual["reference"] == ref, test["name"]
        if test.get("downstream"):
            reach, minimum = graph(test, trace, result["pipes"])
            all_events = [(point, kind) for point in points for kind in range(2)]
            def relation(a, b, cover):
                if a[0] not in positions or b[0] not in positions:
                    return None
                x, y = 2 * positions[a[0]] + a[1], 2 * positions[b[0]] + b[1]
                return (x, y) in minimum if cover else y in reach[x]
            assert actual["reach"] == [[relation(a, b, False) for b in all_events] for a in all_events]
            assert actual["minimum"] == [[relation(a, b, True) for b in all_events] for a in all_events]
            if test.get("materialize"):
                assert actual["materialized"]


def main():
    driver, path = sys.argv[1:]
    cases = corpus()
    results = []
    for test in cases:
        pathlib.Path(path).write_text(json.dumps([test]))
        start = time.monotonic()
        try:
            completed = subprocess.run([driver, "--structured-json", path], check=True, text=True,
                                       capture_output=True, timeout=60)
        except subprocess.CalledProcessError as error:
            print(test["name"], error.stdout, error.stderr, file=sys.stderr)
            raise
        value = json.loads(completed.stdout)
        assert len(value) == 1
        verify(test, value[0])
        results.append(value[0])
        print(f"{test['name']}: {time.monotonic() - start:.3f}s", file=sys.stderr)
    mutations = 0
    for index, edit in [(0, "presence"), (1, "order"), (2, "identity"), (3, "context"), (8, "cover"), (8, "reach")]:
        bad = copy.deepcopy(results[index])
        context = bad["contexts"][0]
        if edit == "presence": context["present"][0] = not context["present"][0]
        elif edit == "order": context["reference"][0][1] = not context["reference"][0][1]
        elif edit == "identity": context["reference"][0][0] = True
        elif edit == "context": context["admitted"] = False
        elif edit == "cover": context["minimum"][1][2] = False
        else: context["reach"][0][0] = False
        try:
            verify(cases[index], bad)
        except AssertionError:
            mutations += 1
        else:
            raise AssertionError(f"undetected mutation {edit}")
    print(f"verified {len(cases)} actual structured MLIR cases and {mutations} oracle mutations")


if __name__ == "__main__":
    main()
