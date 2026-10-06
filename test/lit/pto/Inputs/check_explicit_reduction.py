# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent all-conflict graph oracle for the explicit completion-rank reducer."""
import itertools
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def occurrence(identity, pipe, accesses=()):
    return {"id": identity, "pipe": pipe, "accesses": list(accesses)}


def access(atom, read=False, write=False, group=0):
    return {"atom": atom, "read": read, "write": write, "protection_group": group}


def normalized(occ):
    result = {}
    for item in occ["accesses"]:
        modes = result.setdefault(item["atom"], [False, False, item["protection_group"]])
        modes[0] |= item["read"]
        modes[1] |= item["write"]
    return result


def conflicts(occurrences):
    result = set()
    modes = [normalized(occ) for occ in occurrences]
    for a, first in enumerate(occurrences):
        for b in range(a + 1, len(occurrences)):
            second = occurrences[b]
            for atom in modes[a].keys() & modes[b].keys():
                read_a, write_a, group_a = modes[a][atom]
                read_b, write_b, group_b = modes[b][atom]
                protected = (write_a and write_b and group_a != 0 and group_a == group_b
                             and first["pipe"] == second["pipe"])
                if not protected and (write_a and (read_b or write_b) or read_a and write_b):
                    result.add((a, b))
    return result


def closure(occurrences, edges):
    count = 2 * len(occurrences)
    graph = [set() for _ in range(count)]
    previous = {}
    for position, occ in enumerate(occurrences):
        graph[2 * position].add(2 * position + 1)
        prior = previous.get(occ["pipe"])
        if prior is not None:
            graph[2 * prior].add(2 * position)
            graph[2 * prior + 1].add(2 * position + 1)
        previous[occ["pipe"]] = position
    for source, target in edges:
        graph[2 * source + 1].add(2 * target)
    reachable = [set() for _ in range(count)]
    for source in reversed(range(count)):
        for target in graph[source]:
            reachable[source].add(target)
            reachable[source].update(reachable[target])
    return reachable


def edge_positions(records, occurrences):
    positions = {occ["id"]: position for position, occ in enumerate(occurrences)}
    return {(positions[record[0]], positions[record[1]]) for record in records}


def expected_edges(case):
    occurrences = case["scan"]
    if "reduction_records" in case:
        return edge_positions(case["reduction_records"], occurrences)
    return conflicts(occurrences) | edge_positions(case.get("prerequisites", []), occurrences)


def check(case, output):
    result = output["reduction"]
    occurrences = case["scan"]
    require(not result["error"], result["error"])
    edges = expected_edges(case)
    native = edge_positions(case.get("native_prerequisites", []), occurrences)
    reach = closure(occurrences, edges | native)
    retained = edge_positions(result["retained"], occurrences)
    covers = {(a, b) for a, b in edges - native
              if not any(2 * b in reach[z] for z in reach[2 * a + 1])}
    require(len(result["retained"]) == len(retained), "duplicate retained edges")
    require(retained == covers, f"cover mismatch: {retained} != {covers}; {case}")
    require(closure(occurrences, retained | native) == reach, "reduced closure changed")
    check_rows(occurrences, reach, result)


def check_rows(occurrences, reach, result):
    labels = list(dict.fromkeys(occ["pipe"] for occ in occurrences))
    ranks, totals = [], dict.fromkeys(labels, 0)
    for occ in occurrences:
        totals[occ["pipe"]] += 1
        ranks.append(totals[occ["pipe"]])
    require(result["payloads"] == [occ["id"] for occ in occurrences], "payload identities changed")
    require(result["pipes"] == labels, "pipe columns incorrect")
    require(result["columns"] == [labels.index(occ["pipe"]) for occ in occurrences], "column map incorrect")
    require(result["ranks"] == ranks, "local ranks incorrect")
    for b in range(len(occurrences)):
        for kind, field in enumerate(("starts", "completions")):
            row = [0] * len(labels)
            for a, occ in enumerate(occurrences):
                if 2 * b + kind in reach[2 * a + 1] or (a == b and kind == 1):
                    column = labels.index(occ["pipe"])
                    row[column] = max(row[column], ranks[a])
            require(result[field][b] == row, f"{field} row {b}: {result[field][b]} != {row}")


def requests():
    cases = [{"scan": [], "reduce": True},
             {"scan": [occurrence(2**32 - 1, 2**32 - 1), occurrence(0, 2**32 - 1)], "reduce": True}]
    for pipes in itertools.product((0, 17, 2**32 - 1), repeat=3):
        occurrences = [occurrence(identity, pipe) for identity, pipe in zip((90, 12, 44), pipes)]
        candidates = [(90, 12, 0), (90, 44, 0), (12, 44, 0)]
        for mask in range(8):
            edges = [edge for bit, edge in enumerate(candidates) if mask & (1 << bit)]
            cases.append({"scan": occurrences, "reduce": True, "reduction_records": edges + edges})
    rng = random.Random(82472)
    for _ in range(180):
        size = rng.randrange(1, 18)
        ids = rng.sample(range(100, 999), size)
        occurrences = []
        for identity in ids:
            modes = []
            for atom in range(5):
                mode = rng.randrange(4)
                if mode:
                    modes.append(access(atom, bool(mode & 1), bool(mode & 2)))
                    if rng.randrange(5) == 0:
                        modes.append(modes[-1].copy())
            occurrences.append(occurrence(identity, rng.choice((0, 17, 2**32 - 1)), modes))
        extra = [[ids[a], ids[b], 0] for a in range(size) for b in range(a + 1, size) if rng.randrange(20) == 0]
        native = [[ids[a], ids[b], 0] for a in range(size) for b in range(a + 1, size) if rng.randrange(20) == 0]
        cases.append({"scan": occurrences, "prerequisites": extra, "native_prerequisites": native, "reduce": True})
    # An unrelated-buffer path covers x's direct conflict, with reader and RMW chains.
    cases.append({"reduce": True, "scan": [
        occurrence(31, 0, [access(0, write=True), access(1, write=True)]),
        occurrence(8, 1, [access(1, read=True), access(2, write=True)]),
        occurrence(64, 2, [access(2, read=True), access(0, read=True, write=True)]),
        occurrence(1, 2, [access(0, read=True)]),
        occurrence(17, 0, [access(0, write=True)])]})
    cases.append({"reduce": True, "scan": [
        occurrence(4, 2, [access(0, read=True, write=True, group=9)]),
        occurrence(9, 2, [access(0, read=True, write=True, group=9)]),
        occurrence(1, 8, [access(0, read=True)]),
        occurrence(2, 2, [access(0, read=True, write=True, group=9)])]})
    for pipe_count in (1, 8):
        occurrences = [occurrence(1000 - i, i % pipe_count) for i in range(128)]
        records = [[1000 - i, 999 - i, 0] for i in range(127)] if pipe_count > 1 else []
        cases.append({"scan": occurrences, "reduce": True, "reduction_records": records})
    return cases


def invalid_requests():
    normal = [occurrence(19, 2), occurrence(3, 4)]
    return [
        {"scan": [occurrence(1, 0), occurrence(1, 2)], "reduction_records": [], "reduce": True},
        *({"scan": normal, "reduction_records": [edge], "reduce": True}
          for edge in ([3, 19, 0], [19, 19, 0], [19, 99, 0], [99, 3, 0])),
    ]


def run(tool, cases):
    # Bounded small JSON requests only; no build artifacts in the temporary directory.
    with tempfile.TemporaryDirectory(prefix="pto-explicit-reduction-") as directory:
        request = Path(directory) / "requests.json"
        request.write_text(json.dumps(cases), encoding="utf-8")
        completed = subprocess.run([tool, "--periodic-checks", str(request)], check=True,
                                   capture_output=True, text=True, timeout=120)
    results = json.loads(completed.stdout)
    require(len(results) == len(cases), "incorrect result count")
    return results


def main():
    tool = shutil.which(sys.argv[1])
    require(tool is not None, "test tool must exist")
    cases = requests()
    for case, result in zip(cases, run(tool, cases)):
        check(case, result)
    for output in run(tool, invalid_requests()):
        result = output["reduction"]
        require(bool(result["error"]), "invalid graph accepted")
        require(not any(value for key, value in result.items() if key != "error"), "partial failure result leaked")
    print(f"explicit reduction: {len(cases)} oracle cases and 5 invalid cases passed")


if __name__ == "__main__":
    main()
