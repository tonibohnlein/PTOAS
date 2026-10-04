# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check uniform endpoint matching and command semantics against payload order."""
from collections import Counter
import shutil
import sys
from check_periodic_demands import closure, requests, retained_instances, run, unfolded


def identity(command):
    return command["record"], command["source_ordinal"]


def command_order(case, result):
    types = len(case["pipes"])
    count = types * case["logical_trips"]
    cuts = {entry["cut"]: entry["commands"] for entry in result["logical"]["cuts"]}
    events, commands, by_pipe, payload = [], {}, {}, {}
    publications, acquisitions = {}, {}
    for position in range(count + 1):
        for command in cuts.get(position, []):
            vertex = len(events)
            events.append(("command", command))
            commands[vertex] = command
            by_pipe.setdefault(command["pipe"], []).append(("command", vertex))
            if command["kind"] == 0:
                publications[identity(command)] = vertex
            elif command["kind"] == 2:
                acquisitions[identity(command)] = vertex
        if position < count:
            start = len(events)
            events.extend([("start", position), ("complete", position)])
            payload[position] = start
            by_pipe.setdefault(case["pipes"][position % types], []).append(("payload", position))
    edges = {(start, start + 1) for start in payload.values()}
    for sequence in by_pipe.values():
        prior_payloads, prior_commands, gates = [], [], []
        for category, value in sequence:
            if category == "payload":
                start = payload[value]
                if prior_payloads:
                    edges.add((prior_payloads[-1], start))
                    edges.add((prior_payloads[-1] + 1, start + 1))
                edges.update((gate, start) for gate in gates)
                prior_payloads.append(start)
                continue
            command = commands[value]
            edges.update((start, value) for start in prior_payloads)
            edges.update((gate, value) for gate in gates)
            if command["kind"] in (0, 1):  # SET/barrier observe prior completions and commands.
                edges.update((start + 1, value) for start in prior_payloads)
                edges.update((earlier, value) for earlier in prior_commands)
            if command["kind"] in (1, 2):  # Only barrier/WAIT gate later starts and commands.
                gates.append(value)
            prior_commands.append(value)
    assert publications.keys() == acquisitions.keys()
    edges.update((publications[key], acquisitions[key]) for key in publications)
    actual, _ = closure(len(events), edges)
    expected, covers = unfolded(case, count)
    adjacency = True
    for source, target in covers:
        a, b = source // 2, target // 2
        if case["pipes"][a % types] == case["pipes"][b % types]:
            earlier = [i for i in range(b) if case["pipes"][i % types] == case["pipes"][b % types]]
            adjacency &= bool(earlier) and earlier[-1] == a
    for source in range(2 * count):
        for target in range(2 * count):
            observed = bool((actual[payload[source // 2] + source % 2] >>
                             (payload[target // 2] + target % 2)) & 1)
            required = bool((expected[source] >> target) & 1)
            assert observed or not required, (case, source, target, "missing requirement")
            if adjacency:
                assert observed == required, (case, source, target, "extra order")
    return adjacency


def check(case, result):
    logical = result["logical"]
    assert not logical["error"] and logical["selection_recipes_ready"]
    assert not logical["physical_ids_ready"] and not logical["ir_emitted"]
    recipes = logical["recipes"]
    count = len(case["pipes"]) * case["logical_trips"]
    expected = retained_instances(result, count)
    sets = Counter(identity(c) for c in logical["active"] if c["kind"] == 0)
    waits = Counter(identity(c) for c in logical["active"] if c["kind"] == 2)
    assert sets == waits and all(value == 1 for value in sets.values())
    realized = set()
    for command in logical["active"]:
        edge = result["generators"][command["record"]]
        source = command["source_ordinal"] * len(case["pipes"]) + edge["source"]
        target = (command["source_ordinal"] + edge["displacement"]) * len(case["pipes"]) + edge["target"]
        assert source < target < count
        if command["kind"] in (1, 2):
            realized.add((2 * source + 1, 2 * target))
        endpoint = source if command["kind"] == 0 else target
        assert command["ordinal"] * len(case["pipes"]) + command["type"] == endpoint
    assert realized == expected
    assert len(recipes) <= 2 * len(result["retained"])
    for entry in logical["cuts"]:
        values = entry["commands"]
        assert [value["kind"] for value in values] == sorted(value["kind"] for value in values)
        barriers = [value["pipe"] for value in values if value["kind"] == 1]
        assert len(barriers) == len(set(barriers))
        keys = [(c["kind"], c["pipe"], identity(c), c["ordinal"], c["type"]) for c in values]
        assert len(keys) == len(set(keys))
    return command_order(case, result)


def main():
    tool = shutil.which(sys.argv[1])
    assert tool
    cases = [case for case in requests() if "scan" not in case][:108]
    for index, case in enumerate(cases):
        case["logical_trips"] = index % 5
        case["prefix"] = len(case["pipes"]) * case["logical_trips"]
    cases.extend([
        {"pipes": [0, 1, 0], "records": [[0, 1, 0]], "logical_trips": 2},
        {"pipes": [0, 1, 1], "records": [[0, 2, 0]], "logical_trips": 2},
        {"pipes": [0, 1], "records": [[0, 1, 5]], "logical_trips": 2},
    ])
    outputs = run(tool, cases)
    exact = sum(check(case, result) for case, result in zip(cases, outputs))
    assert exact >= 20
    maximum = 2**64 - 1
    corner = {"pipes": [0, 1], "records": [[0, 1, 1]], "logical_trips": 0,
              "endpoint_queries": [[0, maximum, maximum - 2], [0, maximum, maximum - 1],
                                   [1, maximum, maximum - 1], [1, maximum, 0], [9, 3, 0]],
              "counted_loops": [[-2**63, 2**63 - 1, 1, 2**63 - 2], [3, 10, 2, 9],
                                [3, 10, 2, 8], [0, 0, 1, 0], [3, -7, 2, 3], [0, 9, 0, 0]]}
    result = run(tool, [corner])[0]["logical"]
    answers = result["endpoint_answers"]
    assert [value["active"] for value in answers] == [True, False, True, False, False]
    assert answers[0]["instance"]["source_ordinal"] == answers[2]["instance"]["source_ordinal"] == maximum - 2
    assert answers[-1]["error"] == 1
    loops = result["counted_answers"]
    assert loops[0] == {"trip_error": 0, "trips": maximum, "ordinal_error": 0, "ordinal": maximum - 1}
    assert loops[1] == {"trip_error": 0, "trips": 4, "ordinal_error": 0, "ordinal": 3}
    assert loops[2]["ordinal_error"] == 1
    assert loops[3]["trips"] == loops[4]["trips"] == 0
    assert loops[5]["trip_error"] == 1
    print(f"logical endpoints: {len(cases)} matching oracles, {exact} exact command-order oracles passed")


if __name__ == "__main__":
    main()
