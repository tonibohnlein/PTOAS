# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check actual endpoint commands and arbitrary varying-region query exports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def run(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, check=True, timeout=90)
    return json.loads(result.stdout)


def graph(payloads, bounded=False):
    effects = []
    for p in payloads:
        kind, coordinates = p["type"], p["coordinates"]
        slot = coordinates[-1] % 2 if coordinates else 0
        compute = kind == (1 if bounded else 2)
        effects.append(({("left", slot), ("right", 0)}, {("acc", 0)}) if compute else
                       ({("mat", 0)}, {("left", slot)}))
    edges = native([p["pipe"] for p in payloads])
    for a, (r, w) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            rb, wb = effects[b]
            if w & (rb | wb) or r & wb:
                edges.add((2*a+1, 2*b))
    return closure(2*len(effects), edges)[0]


def check_commands(document, bounded=False):
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    payloads = [x for x in trace["events"] if x["kind"] == "payload"]
    expected = graph(payloads, bounded)
    commands = []
    for event in trace["events"]:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(payloads)
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands([p["pipe"] for p in payloads], commands)
    required = [row & ~(1 << i) for i, row in enumerate(expected)]
    if not bounded:
        assert actual == required, (document, expected, actual)
    else:
        # A nonadjacent same-pipe demand places a barrier before its consumer.
        # Check safety and exact cross-pipe cover endpoints separately from the
        # extra local ordering deliberately introduced by this insertion rule.
        assert all((want & got) == want for want, got in zip(required, actual))
        covers = {(a, b) for a in range(len(payloads)) for b in range(a+1, len(payloads))
                  if expected[2*a+1] & (1 << (2*b)) and
                  not any(z not in (2*a+1, 2*b) and expected[2*a+1] & (1 << z) and
                          expected[z] & (1 << (2*b)) for z in range(2*len(payloads)))}
        publications = {}
        pairs, local_targets = set(), set()
        for command in commands:
            pipe, gap = command["pipe"], command["gap"]
            if command["kind"] == "set":
                publications[command["identity"]] = max(i for i in range(gap) if payloads[i]["pipe"] == pipe)
            else:
                consumer = next(i for i in range(gap, len(payloads)) if payloads[i]["pipe"] == pipe)
                if command["kind"] == "wait":
                    pairs.add((publications.pop(command["identity"]), consumer))
                else:
                    local_targets.add(consumer)
        assert not publications
        assert pairs == {(a, b) for a, b in covers if payloads[a]["pipe"] != payloads[b]["pipe"]}
        assert local_targets == {b for a, b in covers if payloads[a]["pipe"] == payloads[b]["pipe"]}
    return payloads, expected


def main():
    tool, fixture = sys.argv[1:]
    text = Path(fixture).read_text()
    tests = 0
    with tempfile.TemporaryDirectory(prefix="compact-endpoints-") as directory:
        path = Path(directory) / "case.pto"
        # Include zero trips, startup, the transition and several suffix periods.
        for count in (0, 1, 2, 4, 6, 8):
            case = text.replace("array<i64: 2, 3>", f"array<i64: {count}, 3>")
            path.write_text(case)
            report = run(tool, "--sequence-analysis", path)
            assert report["prepared"] and not report["error"] and report["numeric_visits"] == 0, report
            check_commands(run(tool, "--structured-trace", path))
            tests += 1
            if count not in (2, 8):
                continue
            # Query only the nested child: the surrounding work is irrelevant
            # to paths whose endpoints lie inside this contiguous region.
            requested = [(kind, v, i) for v in range(count) for i in (0, v+1) for kind in range(2)]
            encoded = ", ".join(str(x) for row in requested for x in row)
            query_attribute = f"attributes {{test.varying_queries = array<i64: {encoded}>, test.trace_arguments"
            path.write_text(case.replace("attributes {test.trace_arguments", query_attribute))
            report = run(tool, "--structured-trace", path)
            assert not report["error"], report
            all_payloads = [{"type": kind+1, "coordinates": [v, i], "pipe": pipe}
                            for v in range(count) for i in range(v+2) for kind, pipe in ((0, 3), (1, 2))]
            expected = graph(all_payloads)
            ids = {(p["type"]-1, *p["coordinates"]): i for i, p in enumerate(all_payloads)}
            events = [2*ids[row]+kind for row in requested for kind in range(2)]
            assert report["queries"] == [bool(expected[a] & (1 << b)) for a in events for b in events]
        for intercept in (0, 1):
            for count in (0, 1, 2, 5, 8):
                case = text.replace("%length = arith.addi %visit, %two", "%length = arith.addi %visit, " +
                                    ("%zero" if intercept == 0 else "%one"))
                path.write_text(case.replace("array<i64: 2, 3>", f"array<i64: {count}, 3>"))
                check_commands(run(tool, "--structured-trace", path))
                tests += 1
        doubled = text.replace("%length = arith.addi %visit, %two : index",
                               "%doubled = arith.muli %visit, %two : index\n"
                               "      %length = arith.addi %doubled, %two : index")
        for count, step in ((0, "%one"), (2, "%one"), (5, "%one"), (3, "%two")):
            case = doubled.replace("%i = %zero to %length step %one", f"%i = %zero to %length step {step}")
            path.write_text(case.replace("array<i64: 2, 3>", f"array<i64: {count}, 3>"))
            check_commands(run(tool, "--structured-trace", path))
            tests += 1
        path.write_text(text)
        small = run(tool, "--sequence-analysis", path)
        path.write_text(text.replace("arith.constant 16 : index", "arith.constant 1000000000 : index"))
        large = run(tool, "--sequence-analysis", path)
        assert large["prepared"] and small["emitted"] == large["emitted"], (small, large)
        window = text.replace("    scf.for %visit = %zero to %visits step %one {\n"
                              "      %length = arith.addi %visit, %two : index\n"
                              "      scf.for %i = %zero to %length step %one {",
                              "    scf.for %i = %zero to %n step %one {")
        window = window.replace("      }\n    }\n", "    }\n")
        outer_access = "    pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%first : !left)\n"
        window = window.replace(outer_access, "")
        window = window.replace("attributes {test.trace_arguments",
                                "attributes {test.bounded_lifetime_insertion, test.trace_arguments")
        compute = "      pto.tmatmul ins(%left, %right : !left, !right) outs(%acc : !acc)"
        optional = ("\n      %take = arith.cmpi eq, %slot, %zero : index\n"
                    "      scf.if %take {\n"
                    "        pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%left : !left)\n"
                    "      }")
        window = window.replace(compute, compute + optional)
        for count in (0, 1, 2, 3, 5, 9):
            path.write_text(window.replace("array<i64: 2, 3>", f"array<i64: {count}, 3>"))
            check_commands(run(tool, "--structured-trace", path), True)
            tests += 1
        for bound in (1, 4, 100):
            threshold = window.replace("%take = arith.cmpi eq, %slot, %zero : index",
                                       "%take = arith.cmpi ult, %i, %m : index")
            path.write_text(threshold.replace("array<i64: 2, 3>", f"array<i64: 7, {bound}>"))
            check_commands(run(tool, "--structured-trace", path), True)
            tests += 1
        narrow = window.replace("%take = arith.cmpi eq, %slot, %zero : index",
                                "%small = arith.index_cast %i : index to i8\n"
                                "      %bias = arith.constant 127 : i8\n"
                                "      %smallzero = arith.constant 0 : i8\n"
                                "      %wrapped = arith.addi %small, %bias : i8\n"
                                "      %take = arith.cmpi slt, %wrapped, %smallzero : i8")
        path.write_text(narrow.replace("array<i64: 2, 3>", "array<i64: 6, 3>"))
        check_commands(run(tool, "--structured-trace", path), True)
        tests += 1
        unsafe = window.replace("%take = arith.cmpi eq, %slot, %zero : index",
                                "%quot = arith.divsi %i, %m : index\n"
                                "      %take = arith.cmpi eq, %quot, %zero : index")
        path.write_text(unsafe)
        report = run(tool, "--structured-trace", path)
        assert not report["accepted"] and report["unchanged_on_failure"], report
    print(f"{tests} compact command closures and varying-region all-event queries passed")


if __name__ == "__main__":
    main()
