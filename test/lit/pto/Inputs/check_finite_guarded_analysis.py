# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate finite guarded circuits and actual commands against concrete DAGs."""

import itertools
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def invoke(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, check=False, timeout=90)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout) if mode != "--insert-logical" else result.stdout


def load(label="L", destination="left"):
    return (
        f"pto.textract ins(%mat, %zero, %zero : !mat, index, index) "
        f'outs(%{destination} : !left) {{test.label = "{label}"}}'
    )


def compute(label="C", rmw=False):
    inputs = "%acc, %left, %right : !acc, !left, !right" if rmw else "%left, %right : !left, !right"
    return f'pto.tmatmul{".acc" if rmw else ""} ins({inputs}) outs(%acc : !acc) {{test.label = "{label}"}}'


def required(payloads, effects):
    edges = native([item["pipe"] for item in payloads])
    for a, item in enumerate(payloads):
        reads, writes = effects[item["label"]]
        for b in range(a + 1, len(payloads)):
            later_reads, later_writes = effects[payloads[b]["label"]]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    return closure(2 * len(payloads), edges)[0]


def validate(document, labels, effects):
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    payloads = [item for item in trace["events"] if item["kind"] == "payload"]
    assert [item["label"] for item in payloads] == labels, payloads
    graph = required(payloads, effects)
    commands = []
    gap = 0
    for item in trace["events"]:
        if item["kind"] == "payload":
            gap += 1
            continue
        assert item["gap"] == gap
        if item["kind"] == "barrier" and item["pipe"] == 6:
            assert gap == len(payloads), "completion drain inserted inside a child"
            continue
        command = dict(item)
        if item["kind"] != "barrier":
            command["identity"] = (
                item["plan"],
                item["record"],
                item["source_ordinal"],
                tuple(item.get("members", [])),
            )
        commands.append(command)
    actual = closure_with_commands([item["pipe"] for item in payloads], commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(graph)], (labels, actual, graph)


def render(source, body, g, h, n=3):
    begin = source.index("    // BODY")
    end = source.index("    // END_BODY", begin)
    result = source[:begin] + "\n".join("    " + line for line in body.splitlines()) + "\n" + source[end:]
    return result.replace("array<i64: 1, 1, 3>", f"array<i64: {g}, {h}, {n}>")


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    effects = {label: ({"mat"}, {"left"}) for label in ("L", "B", "E")}
    effects.update({label: ({"left", "right"}, {"acc"}) for label in ("C", "D")})
    effects["R"] = ({"left", "right", "acc"}, {"acc"})
    effects["X"] = ({"mat"}, {"second"})
    cases = [
        (
            f"{load()}\nscf.if %g {{\n{compute()}\n}}\n{load('E', 'alias')}",
            lambda g, h: ["L"] + (["C"] if g else []) + ["E"],
        ),
        (
            f"scf.if %g {{\n{load()}\n}} else {{\n{load('B', 'alias')}\n}}\n{compute()}\n{load('E')}",
            lambda g, h: ["L" if g else "B", "C", "E"],
        ),
        (
            f"scf.if %g {{\n{load()}\nscf.if %h {{\n{compute()}\n}}\n}} else {{\n"
            f"{load('B')}\n{compute('D')}\n}}\n{load('E')}",
            lambda g, h: (["L"] + (["C"] if h else [])) + ["E"] if g else ["B", "D", "E"],
        ),
        (
            f"scf.if %g {{\n}} else {{\n{load()}\n}}\n{compute()}\n{load('E')}",
            lambda g, h: ([] if g else ["L"]) + ["C", "E"],
        ),
        (
            f"scf.if %g {{\n{compute()}\n}}\nscf.if %h {{\n{compute('D')}\n}}\n{load('E')}",
            lambda g, h: (["C"] if g else []) + (["D"] if h else []) + ["E"],
        ),
        (
            f"{load()}\nscf.if %g {{\n{compute('R', True)}\n}}\n{load('E')}",
            lambda g, h: ["L"] + (["R"] if g else []) + ["E"],
        ),
        (
            f"{load()}\n%late = arith.xori %g, %h : i1\nscf.if %late {{\n{compute()}\n}}\n{load('E')}",
            lambda g, h: ["L"] + (["C"] if g != h else []) + ["E"],
        ),
        (
            f"{load()}\nscf.if %g {{\n%local = arith.xori %g, %h : i1\n"
            f"scf.if %local {{\n{compute()}\n}}\n}}\n{load('E')}",
            lambda g, h: ["L"] + (["C"] if g and g != h else []) + ["E"],
        ),
    ]
    checked = 0
    with tempfile.TemporaryDirectory(prefix="finite-guarded-") as scratch:
        path = Path(scratch) / "case.pto"
        for case, (body, select) in enumerate(cases):
            for g, h in itertools.product((0, 1), repeat=2):
                program = render(source, body, g, h)
                path.write_text(program)
                validate(invoke(tool, "--structured-trace", path), select(g, h), effects)
                checked += 1
                if case < 6:
                    path.write_text(
                        program.replace(
                            "attributes {test.trace_arguments",
                            "attributes {test.queries, test.trace_arguments",
                        )
                    )
                    report = invoke(tool, "--finite-guarded-analysis", path)
                    assert not report["error"] and report["prepared"] and report["unchanged"], report
                    active = [event for event in report["trace"]["events"] if event["kind"] == "payload"]
                    graph = required(active, effects)
                    position = {event["label"]: i for i, event in enumerate(active)}
                    expected = []
                    for a in range(2 * len(report["labels"])):
                        for b in range(2 * len(report["labels"])):
                            ia = position.get(report["labels"][a // 2])
                            ib = position.get(report["labels"][b // 2])
                            expected.append(
                                ia is not None
                                and ib is not None
                                and bool(graph[2 * ia + a % 2] & (1 << (2 * ib + b % 2)))
                            )
                    assert report["queries"] == expected, (report, expected)
        # A guarded child's selectors must compose across two compact loop boundaries.
        loop = f"scf.for %i = %zero to %n step %one {{\n{load()}\n{compute()}\n}}"
        middle = f"scf.if %g {{\n{load('B')}\nscf.if %h {{\n{compute('D')}\n}}\n}}"
        body = f"{loop}\n{middle}\n{loop}\n{load('E')}"
        for n in (0, 1, 4):
            for g, h in itertools.product((0, 1), repeat=2):
                path.write_text(render(source, body, g, h, n))
                labels = ["L", "C"] * n + (["B"] + (["D"] if h else []) if g else [])
                labels += ["L", "C"] * n + ["E"]
                validate(invoke(tool, "--structured-trace", path), labels, effects)
                checked += 1
        # Many independent branch decisions stay a finite shared DAG; no valuation expansion.
        for count, shared in ((8, False), (24, False), (4, True), (8, True)):
            declarations = []
            body = []
            for i in range(count):
                declarations += [
                    f"%address{i} = arith.constant {0 if shared else 512 * i} : i64",
                    f"%buffer{i} = pto.alloc_tile addr = %address{i} : !left",
                ]
                body.append(f"scf.if %g{i} {{\n{load(f'X{i}', f'buffer{i}')}\n}}")
            expanded = render(source, "\n".join(declarations + body), 0, 0)
            args = ", ".join(f"%g{i}: i1" for i in range(count))
            expanded = expanded.replace("%g: i1, %h: i1, %n: index", args)
            path.write_text(expanded)
            report = invoke(tool, "--finite-guarded-analysis", path)
            assert not report["error"] and report["prepared"] and report["unchanged"], report
            assert report["ports"] == count and report["analysis_expressions"] < 20 * count**3, report
            print(
                "guarded size:",
                {
                    "sites": count,
                    "shared": shared,
                    **{key: report[key] for key in ("analysis_expressions", "expressions", "emitted")},
                },
            )
            if shared:
                masks = range(1 << count) if count == 4 else (0, 85, 170, 255)
                for mask in masks:
                    bits = [(mask >> i) & 1 for i in range(count)]
                    arguments = ", ".join(str(bit) for bit in bits)
                    path.write_text(expanded.replace("array<i64: 0, 0, 3>", f"array<i64: {arguments}>"))
                    labels = [f"X{i}" for i, bit in enumerate(bits) if bit]
                    shared_effects = {f"X{i}": ({"mat"}, {"left"}) for i in range(count)}
                    validate(invoke(tool, "--structured-trace", path), labels, shared_effects)
                    checked += 1
        # A non-adjacent local cover must not be replaced by a stronger pipe barrier.
        body = f"{load()}\nscf.if %g {{\n{load('X', 'second')}\n}}\n{load('E')}"
        path.write_text(render(source, body, 1, 0))
        result = invoke(tool, "--structured-trace", path)
        assert not result["accepted"] and result["unchanged_on_failure"], result
        report = invoke(tool, "--sequence-analysis", path)
        assert not report["error"] and not report["prepared"] and report["queries_available"], report
        # Inactive arm-local poison must be masked before it reaches SET/WAIT predicates.
        body = (
            f"{load()}\nscf.if %g {{\n%one64 = arith.constant 1 : i64\n"
            "%sum = arith.addi %x, %one64 overflow<nsw> : i64\n"
            "%nested = arith.cmpi slt, %sum, %one64 : i64\n"
            f"scf.if %nested {{\n{compute()}\n}}\n}}\n{load('E')}"
        )
        poison = render(source, body, 0, 0).replace("%n: index)", "%n: index, %x: i64)")
        poison = poison.replace("array<i64: 0, 0, 3>", "array<i64: 0, 0, 3, 9223372036854775807>")
        path.write_text(poison)
        emitted = invoke(tool, "--insert-logical", path)
        additions = set(re.findall(r"(%[\w]+) = arith.addi [^\n]+overflow<nsw>", emitted))
        comparisons = {
            result
            for result, operand in re.findall(r"(%[\w]+) = arith.cmpi slt, (%[\w]+),", emitted)
            if operand in additions
        }
        assert any(
            re.search(r"arith.select %arg0, " + re.escape(predicate) + r", %false", emitted)
            for predicate in comparisons
        ), emitted
        # Replay must be deterministic: Pure/speculatable LLVM freeze is not duplicable.
        body = (
            f"{load()}\n%poison = llvm.mlir.poison : i1\n%frozen = llvm.freeze %poison : i1\n"
            f"scf.if %frozen {{\n{compute()}\n}}\n{load('E')}"
        )
        path.write_text(render(source, body, 1, 0))
        report = invoke(tool, "--finite-guarded-analysis", path)
        assert not report["error"] and not report["prepared"] and report["queries_available"], report
        assert report["unchanged"], report
        # A future region-valued condition is analyzable but is not speculated at an early SET.
        body = (
            f"{load()}\n%choice = scf.if %g -> i1 {{\nscf.yield %h : i1\n}} else {{\n"
            f"scf.yield %g : i1\n}}\nscf.if %choice {{\n{compute()}\n}}\n{load('E')}"
        )
        path.write_text(render(source, body, 1, 1))
        result = invoke(tool, "--structured-trace", path)
        assert not result["accepted"] and result["unchanged_on_failure"], result
        report = invoke(tool, "--sequence-analysis", path)
        assert not report["error"] and not report["prepared"] and report["queries_available"], report
    print(f"finite guarded: {checked} concrete valuations and three transactional rejections passed")


if __name__ == "__main__":
    main()
