# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent finite all-conflict DAG oracle for the periodic demand backend."""
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile


def modes(items):
    result = {}
    for item in items:
        old = result.setdefault(item["atom"], [False, False])
        old[0] |= item["read"]
        old[1] |= item["write"]
    return result


def conflicts(left, right):
    return any((left[atom][1] and any(right[atom])) or (left[atom][0] and right[atom][1])
               for atom in left.keys() & right.keys())


def native(pipes):
    edges = {(2 * i, 2 * i + 1) for i in range(len(pipes))}
    previous = {}
    for i, pipe in enumerate(pipes):
        if pipe in previous:
            edges.add((2 * previous[pipe], 2 * i))
            edges.add((2 * previous[pipe] + 1, 2 * i + 1))
        previous[pipe] = i
    return edges


def closure(size, edges):
    outgoing = [set() for _ in range(size)]
    for source, target in edges:
        assert source < target, (source, target)
        outgoing[source].add(target)
    reach = [1 << v for v in range(size)]
    for source in reversed(range(size)):
        for target in outgoing[source]:
            reach[source] |= reach[target]
    covers = {(source, target) for source, target in edges if source % 2 and not target % 2
              and not any(other != target and (reach[other] >> target) & 1
                          for other in outgoing[source])}
    return reach, covers


def unfolded(case, count):
    types = len(case["pipes"])
    pipes = [case["pipes"][i % types] for i in range(count)] if types else []
    edges = native(pipes)
    if "word" in case:
        effects = [modes(case["word"][i % types]) for i in range(count)]
        # This enumerates EVERY conflicting pair. It does not implement the scan.
        for source in range(count):
            for target in range(source + 1, count):
                if conflicts(effects[source], effects[target]):
                    edges.add((2 * source + 1, 2 * target))
    for source, target, distance in case.get("records", []):
        for occurrence in range(source, count, types):
            destination = occurrence // types * types + distance * types + target
            if destination < count:
                edges.add((2 * occurrence + 1, 2 * destination))
    return closure(2 * count, edges)


def retained_instances(result, count):
    types = len(result["payloads"])
    pairs = set()
    for identity in result["retained"]:
        record = result["generators"][identity]
        for source in range(record["source"], count, types):
            target = source // types * types + record["displacement"] * types + record["target"]
            if target < count:
                pairs.add((2 * source + 1, 2 * target))
    return pairs


def check_graph(case, result):
    assert not result["error"], result["error"]
    assert result["minimum_demands_ready"] and result["completion_queries_ready"]
    assert not result["interfaces_ready"]
    assert all(result[field] == 1 for field in ["invalid_threshold", "invalid_endpoint", "invalid_kind"])
    count = case.get("prefix", 0)
    reach, covers = unfolded(case, count)
    assert retained_instances(result, count) == covers, (case, result["retained"], covers)
    types = len(case["pipes"])
    for source in range(count):
        for target in range(2 * count):
            expected = bool((reach[2 * source + 1] >> target) & 1)
            assert result["reachable"][source][target] == expected, (case, source, target)
            assert result["strict"][source][target] == (expected and target != 2 * source + 1)
    for target in range(2 * count):
        for row, frontier in enumerate(result["frontiers"]):
            occurrences = [i for i in range(count) if case["pipes"][i % types] == frontier["pipe"]]
            expected = max((rank for rank, i in enumerate(occurrences, 1)
                            if (reach[2 * i + 1] >> target) & 1), default=0)
            assert result["ranks"][target][row] == expected, (case, target, frontier)
    # A simple quotient path has <=2*m-1 edges and our random records have
    # weight <=3. This UNFOLDED all-conflict oracle contains every shortest lift.
    long_count = types * (6 * types + 1)
    long_reach, _ = unfolded(case, long_count)
    for source in range(types):
        for target in range(2 * types):
            expected = next((d for d in range(6 * types + 1)
                             if (long_reach[2 * source + 1] >> (2 * types * d + target)) & 1), None)
            assert result["thresholds"][source][target] == expected, (case, source, target, expected)
    if "word" in case:
        assert not result["scan"]["error"]
        assert len(result["scan"]["generators"]) <= len(result["scan"]["witnesses"])


def check_scan(case, result):
    assert not result["error"]
    values = case["scan"]
    ids = {value["id"]: i for i, value in enumerate(values)}
    edges = native([value["pipe"] for value in values])
    original = set(edges)
    effects = [modes(value["accesses"]) for value in values]
    for source in range(len(values)):
        for target in range(source + 1, len(values)):
            if conflicts(effects[source], effects[target]):
                original.add((2 * source + 1, 2 * target))
    for source, target, _ in case.get("prerequisites", []):
        original.add((2 * ids[source] + 1, 2 * ids[target]))
    for source, target in result["generators"]:
        edges.add((2 * ids[source] + 1, 2 * ids[target]))
    assert closure(2 * len(values), original)[0] == closure(2 * len(values), edges)[0]
    assert len({tuple(pair) for pair in result["generators"]}) == len(result["generators"])
    for edge, _, _ in result["witnesses"]:
        assert edge < len(result["generators"])


def access(atom, read=False, write=False):
    return {"atom": atom, "read": read, "write": write}


def requests():
    cases = [
        {"pipes": [], "word": [], "prefix": 0},
        {"pipes": [4], "word": [[access(0, read=True)]], "prefix": 3},
        {"pipes": [4], "word": [[access(0, read=True, write=True)]], "prefix": 4},
        {"pipes": [0, 1, 0], "word": [[access(0, read=True)], [access(0, write=True)],
                                       [access(0, read=True)]], "prefix": 8},
        {"pipes": [0, 1, 0], "word": [[access(0, write=True), access(1, write=True)],
                                       [access(0, read=True), access(1, write=True)],
                                       [access(0, write=True), access(1, read=True)]], "prefix": 9},
        {"pipes": [0, 1, 2], "records": [[0, 1, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0]], "prefix": 8},
        {"pipes": [0, 1], "records": [], "prefix": 7},
        {"pipes": [0, 0], "records": [], "prefix": 5,
         "rank_queries": [[0, 0, 1, (2**64 - 1) // 2, 2**64 - 1],
                          [0, 1, 1, (2**64 - 1) // 2, 2**64 - 1],
                          [55, 0, 1, 0, 2]]},
    ]
    rng = random.Random(512037)
    for _ in range(100):
        size = rng.randrange(1, 9)
        word = []
        for _ in range(size):
            effects = []
            for atom in range(5):
                mode = rng.randrange(4)
                if mode:
                    effects.append(access(atom, bool(mode & 1), bool(mode & 2)))
                    if rng.randrange(4) == 0:
                        effects.append(dict(effects[-1]))
            word.append(effects)
        cases.append({"pipes": [rng.randrange(4) for _ in range(size)], "word": word,
                      "prefix": rng.randrange(4 * size + 1)})
    for _ in range(100):
        size = rng.randrange(1, 9)
        records = []
        for _ in range(3 * size):
            source, target, distance = rng.randrange(size), rng.randrange(size), rng.randrange(4)
            if distance or source < target:
                records.append([source, target, distance])
        cases.append({"pipes": [rng.randrange(4) for _ in range(size)], "records": records,
                      "prefix": rng.randrange(4 * size + 1)})
    for _ in range(20):
        size = rng.randrange(1, 20)
        ids = rng.sample(range(100, 999), size)
        values = [{"id": ids[i], "pipe": rng.randrange(4),
                   "accesses": [access(atom, bool(mode & 1), bool(mode & 2))
                                for atom in range(5) if (mode := rng.randrange(4))]}
                  for i in range(size)]
        cases.append({"scan": values, "prerequisites": [[ids[0], ids[-1], 0]] if size > 1 else []})
    return cases


def run(tool, cases):
    with tempfile.TemporaryDirectory(prefix="pto-periodic-check-") as directory:
        request = Path(directory) / "requests.json"
        request.write_text(json.dumps(cases), encoding="utf-8")
        completed = subprocess.run([tool, "--periodic-checks", str(request)], check=True,
                                   capture_output=True, text=True, timeout=120)
    return json.loads(completed.stdout)


def main():
    tool = shutil.which(sys.argv[1])
    assert tool, "test tool must exist"
    cases = requests()
    results = run(tool, cases)
    assert len(results) == len(cases)
    for case, result in zip(cases, results):
        if "scan" in case:
            check_scan(case, result)
        else:
            check_graph(case, result)
    assert results[7]["rank_answers"] == [{"error": 0, "value": 2**64 - 1},
                                          {"error": 1, "value": 0}, {"error": 0, "value": 0}]
    special = [
        {"pipes": [0, 1], "records": [[0, 1, 10**9]], "prefix": 4},
        {"pipes": [0, 1], "records": [[1, 0, 0]]},
        {"pipes": [0, 1], "records": [[0, 2, 1]]},
        {"pipes": [0, 0, 1], "records": [[0, 2, 2**64 - 1]]},
        {"scan": [{"id": 1, "pipe": 0, "accesses": []}, {"id": 1, "pipe": 0, "accesses": []}]},
        {"scan": [{"id": 7, "pipe": 0, "accesses": []}, {"id": 3, "pipe": 0, "accesses": []}],
         "prerequisites": [[3, 7, 0]]},
        {"scan": [{"id": 1, "pipe": 0, "accesses": [access(0)]}]},
    ]
    outputs = run(tool, special)
    assert not outputs[0]["error"] and outputs[0]["graph_edges"] == 7
    assert outputs[0]["thresholds"][0][2] == 10**9 and outputs[0]["retained"] == [0]
    for result in outputs[1:]:
        assert result["error"] and not result["generators"]
        assert not result.get("frontiers") and not result.get("witnesses")
    # Same-pipe reader compression keeps only the last reader, although all
    # earlier readers remain protected through their native completion chain.
    compressed = {"scan": [{"id": i, "pipe": 2, "accesses": [access(0, read=True)]} for i in range(10)] +
                          [{"id": 10, "pipe": 3, "accesses": [access(0, write=True)]}]}
    result = run(tool, [compressed])[0]
    check_scan(compressed, result)
    assert result["generators"] == [[9, 10]]
    print(f"periodic demands: {len(cases)} finite oracles, large-distance and rejection checks passed")


if __name__ == "__main__":
    main()
