# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check rotating extraction against independent physical access unfolding."""
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from check_periodic_demands import closure, native, retained_instances


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fragment(payload, slots, stride=1, offset=0, read=False, write=False, family=0, atom=0, group=0):
    return {"payload": payload, "slots": slots, "stride": stride, "offset": offset,
            "family": family, "atom": atom, "read": read, "write": write, "protection_group": group}


def effects(case):
    types = len(case["pipes"])
    result = []
    for position in range(case["prefix"]):
        iteration, site = divmod(position, types)
        modes = {}
        for item in case["rotating"]:
            if item["payload"] != site:
                continue
            slot = (item["stride"] * iteration + item["offset"]) % item["slots"]
            cell = (item["family"], slot, item["atom"])
            state = modes.setdefault(cell, [False, False, item["protection_group"]])
            state[0] |= item["read"]
            state[1] |= item["write"]
        result.append(modes)
    return result


def conflicting(first, second, same_scope):
    for cell in first.keys() & second.keys():
        read_a, write_a, group_a = first[cell]
        read_b, write_b, group_b = second[cell]
        protected = same_scope and write_a and write_b and group_a != 0 and group_a == group_b
        if not protected and (write_a and (read_b or write_b) or read_a and write_b):
            return True
    return False


def reference(case):
    types, count = len(case["pipes"]), case["prefix"]
    pipes = [case["pipes"][position % types] for position in range(count)]
    edges = native(pipes)
    accesses = effects(case)
    for source in range(count):
        for target in range(source + 1, count):
            same_scope = source // types == target // types and pipes[source] == pipes[target]
            if conflicting(accesses[source], accesses[target], same_scope):
                edges.add((2 * source + 1, 2 * target))
    return pipes, closure(2 * count, edges)


def instances(records, types, count):
    edges = set()
    for source_type, target_type, advance in records:
        for source in range(source_type, count, types):
            target = (source // types + advance) * types + target_type
            if target < count:
                edges.add((2 * source + 1, 2 * target))
    return edges


def check(case, result):
    require(not result["extraction_error"], result["extraction_error"])
    require(not result["error"], result["error"])
    count, types = case["prefix"], len(case["pipes"])
    pipes, (reach, covers) = reference(case)
    extracted = result["extracted"]
    require(len(extracted) <= 2 * len(case["rotating"]), "nonsparse generator family")
    require(len(extracted) == len(set(map(tuple, extracted))), "duplicate extraction records")
    expected_refresh = max((item["slots"] // math.gcd(item["slots"], item["stride"])
                            for item in case["rotating"]), default=0)
    require(result["refresh"] == expected_refresh, "incorrect refresh distance")
    generated = native(pipes) | instances(extracted, types, count)
    require(closure(2 * count, generated)[0] == reach, f"generator closure mismatch: {case}; {extracted}")
    require(retained_instances(result, count) == covers, f"cover mismatch: {case}; {result['generators']}")
    for source in range(count):
        for target in range(2 * count):
            expected = bool((reach[2 * source + 1] >> target) & 1)
            require(result["reachable"][source][target] == expected, "quotient query mismatch")


def basic_patterns():
    cases = [
        {"pipes": [], "rotating": []},
        {"pipes": [0, 1], "rotating": [fragment(0, 2, write=True), fragment(1, 2, read=True)]},
        {"pipes": [0, 1], "rotating": [fragment(0, 6, stride=2, offset=1, write=True),
                                        fragment(1, 6, stride=2, offset=5, read=True)]},
        {"pipes": [0, 1], "rotating": [fragment(0, 5, stride=0, offset=3, write=True),
                                        fragment(1, 5, stride=0, offset=3, read=True)]},
        {"pipes": [0, 1], "rotating": [fragment(0, 4, stride=2, offset=0, write=True),
                                        fragment(1, 4, stride=2, offset=1, read=True)]},
        {"pipes": [0, 1], "rotating": [fragment(0, 3, read=True), fragment(1, 3, read=True)]},
        {"pipes": [0, 0, 1], "rotating": [fragment(0, 2, write=True, read=True, group=7),
                                           fragment(1, 2, write=True, read=True, group=7),
                                           fragment(2, 2, read=True)]},
    ]
    mixed = []
    for family, slots in enumerate((2, 3, 5)):
        mixed.extend((fragment(0, slots, write=True, family=family),
                      fragment(1, slots, read=True, family=family)))
    cases.append({"pipes": [17, 99], "rotating": mixed})
    # Overlapping fixed subregions have already been split into disjoint atoms.
    cases.append({"pipes": [0, 1, 2], "rotating": [fragment(0, 3, write=True, atom=0),
        fragment(0, 3, write=True, atom=1), fragment(1, 3, read=True, atom=1),
        fragment(2, 3, write=True, read=True, offset=2, atom=0)]})
    # Multiple selectors at one payload, and duplicate R/W declarations.
    cases.append({"pipes": [0, 1], "rotating": [fragment(0, 3, write=True), fragment(0, 3, read=True),
        fragment(0, 3, offset=1, write=True), fragment(1, 3, offset=1, read=True)]})
    for slots, stride in ((10**9, 1), (2**63 - 1, 2**63 - 3)):
        cases.append({"pipes": [0, 1], "rotating": [fragment(0, slots, stride=stride, write=True),
            fragment(1, slots, stride=stride, offset=1, read=True)]})
    return cases


def requests():
    patterns = basic_patterns()
    rng = random.Random(107391)
    for _ in range(70):
        sites = rng.randrange(1, 7)
        values = []
        for family in range(rng.randrange(1, 4)):
            slots = rng.randrange(1, 8)
            stride = rng.randrange(2 * slots + 1)
            for site in range(sites):
                for atom in range(2):
                    mode = rng.randrange(4)
                    if mode:
                        item = fragment(site, slots, stride, rng.randrange(2 * slots + 1),
                                        bool(mode & 1), bool(mode & 2), family, atom)
                        values.append(item)
                        if rng.randrange(8) == 0:
                            values.append(item.copy())
        patterns.append({"pipes": [rng.randrange(3) for _ in range(sites)], "rotating": values})
    cases = []
    for pattern in patterns:
        sites = len(pattern["pipes"])
        for count in sorted({0, sites, 2 * sites, 7 * sites, max(0, 3 * sites - 1)}):
            cases.append(dict(pattern, prefix=count))
    return cases


def invalid_requests():
    first = fragment(0, 2, write=True)
    return [
        {"pipes": [0], "rotating": [dict(first, payload=1)]},
        {"pipes": [0], "rotating": [dict(first, slots=0)]},
        {"pipes": [0], "rotating": [dict(first, read=False, write=False)]},
        {"pipes": [0], "rotating": [first, dict(first, slots=3)]},
        {"pipes": [0], "rotating": [first, dict(first, stride=0)]},
        {"pipes": [0], "rotating": [dict(first, read=True, write=False, protection_group=1)]},
        {"pipes": [0, 1], "rotating": [dict(first, protection_group=1),
                                        dict(first, payload=1, protection_group=1)]},
        {"pipes": [0], "rotating": [first, dict(first, protection_group=1)]},
    ]


def run(tool, cases):
    with tempfile.TemporaryDirectory(prefix="pto-rotating-extraction-") as directory:
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
    outputs = run(tool, cases)
    for case, result in zip(cases, outputs):
        check(case, result)
    for case, result in zip(cases, outputs):
        if any(item["slots"] >= 10**9 for item in case["rotating"]):
            require(len(result["payloads"]) == 2 and result["graph_edges"] <= 10,
                    "large distance changed quotient size")
    for result in run(tool, invalid_requests()):
        require(bool(result["extraction_error"]), "invalid rotating input accepted")
        require(result["extracted"] == [] and result["refresh"] == 0, "partial error result leaked")
    print(f"rotating extraction: {len(cases)} finite executions and 8 invalid cases passed")


if __name__ == "__main__":
    main()
