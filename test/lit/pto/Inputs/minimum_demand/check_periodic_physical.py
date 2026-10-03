# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Execute actual periodic physical IR against an independent storage/command DAG."""
import sys
import json
from pathlib import Path
from ptoas.mlir.ir import ArrayAttr, Context, DictAttr, IntegerAttr, Module, StringAttr
from ptoas.mlir.dialects import pto
from check_banked_xor import close, command_graph, execute, expected


def selected_closure(metadata, payload_count):
    """Unfold the declared selected graph; do not call its production reducer."""
    vertices = IntegerAttr(metadata["vertices"]).value
    body_size = vertices // 2
    if payload_count % body_size:
        raise ValueError("fixture execution is not a complete repeating prefix")
    trips = payload_count // body_size
    graph = [set() for _ in range(2 * payload_count)]
    for record in ArrayAttr(metadata["closure_graph"]):
        source, target, delay = list(ArrayAttr(record))
        source, target = IntegerAttr(source).value, IntegerAttr(target).value
        delay = int(StringAttr(delay).value)
        for period in range(trips):
            if period + delay < trips:
                graph[vertices * period + source].add(vertices * (period + delay) + target)
    return close(graph)


def rearm(commands):
    details = {}
    command_graph(commands, details)
    previous = {}
    sets, waits = {}, {}
    for kind, direction, finish in details["events"]:
        if kind == "set":
            if direction in previous and finish not in details["reach"][previous[direction]]:
                raise ValueError("physical ID republished before causal consumption")
            sets[direction] = sets.get(direction, 0) + 1
        elif kind == "wait":
            previous[direction] = finish
            waits[direction] = waits.get(direction, 0) + 1
    if sets != waits:
        raise ValueError("publications and acquisitions do not balance")


def rejected(commands, expected):
    try:
        actual = command_graph(commands)
        rearm(commands)
    except ValueError:
        return True
    return actual != expected


def check_costs(path, functions):
    reports = [json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith("{")]
    reports = [report for report in reports if report.get("report") == "frontier-costs-v1"]
    assert {report["function"] for report in reports} == functions, "cost report lost a function"
    for report in reports:
        metrics = report["metrics"]
        assert report["failure_category"] is None, "certified plan reported a failure"
        assert isinstance(metrics["pool_capacities"], list), "missing physical pool report"
        for pool in metrics["pool_capacities"]:
            assert isinstance(pool["available"], int), "pool capacity was not reported"
            assert isinstance(pool["source"], int) and isinstance(pool["target"], int)
            assert pool["allocator"], "missing allocator identity"
        if "physical_ids_scope" in metrics:
            assert metrics["physical_ids"] is None, "compact IDs were spuriously enumerated"


def main():
    source, output = sys.argv[1:3]
    with Context() as context:
        pto.register_dialect(context)
        original_module = Module.parse(Path(source).read_text(encoding="utf-8"))
        output_module = Module.parse(Path(output).read_text(encoding="utf-8"))
        for original_view, output_view in zip(original_module.body.operations, output_module.body.operations):
            original, compiled = original_view.operation, output_view.operation
            status = StringAttr(compiled.attributes["pto.frontier.physical_status"]).value
            constant_case = StringAttr(original.attributes["sym_name"]).value in ("periodic_zero", "periodic_one")
            if status != "certified-periodic-pools" and not (constant_case and status == "certified-finite"):
                raise ValueError("normal compiler did not emit a periodic physical plan")
            metadata = DictAttr(compiled.attributes["pto.frontier.physical"])
            if not len(ArrayAttr(metadata["pools"])) and not constant_case:
                raise ValueError("fixture did not exercise a directed pool")
            for n in (-8, -5, -1, 0, 1, 2, 3, 6, 9):
                required = expected(original, n, True)
                commands = execute(compiled, n, correlated=True)
                analysis = DictAttr(compiled.attributes["pto.frontier.analysis"])
                selected = (selected_closure(analysis, len(required) // 2)
                            if StringAttr(analysis["representation"]).value == "weighted-periodic-quotient"
                            else required)
                if not all(row <= selected[i] for i, row in enumerate(required)):
                    raise ValueError("selected closure omits an original storage requirement")
                if command_graph(commands) != selected:
                    raise ValueError(f"physical plan changes selected payload closure at n={n}")
                rearm(commands)
                acquisition = next((i for i, command in enumerate(commands) if command[0] == "wait"), None)
                if acquisition is not None:
                    missing = commands[:acquisition] + commands[acquisition + 1:]
                    if not rejected(missing, selected):
                        raise ValueError("missing acquisition escaped the independent oracle")
                publication = next((i for i, command in enumerate(commands) if command[0] == "set"), None)
                if publication is not None:
                    early = list(commands)
                    moved = early.pop(publication)
                    first_payload = next(i for i, command in enumerate(early) if command[0] == "payload")
                    early.insert(first_payload, moved)
                    if not rejected(early, selected):
                        raise ValueError("publication before source escaped the independent oracle")
        if len(sys.argv) > 3:
            check_costs(sys.argv[3], {StringAttr(view.operation.attributes["sym_name"]).value
                                    for view in original_module.body.operations})
        print("periodic physical closure, matching, uniform rearm and mutations verified")


if __name__ == "__main__":
    main()
