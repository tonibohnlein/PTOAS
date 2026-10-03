# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare produced integer relations with independently unfolded fixture traces."""
import itertools
import json
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def traces(name, arguments):
    """Return (site, iteration tuple, read/write kind, first byte) in program order."""
    if name == "nested_reset":
        n, k = arguments
        return [event for t in range(n) for event in (
            [(s, (t, i), 5 if s == 0 else 4, 1024 * (i % 2)) for i in range(k) for s in (0, 1)]
            + [(2, (t,), 4, 0)])]
    if name == "triangular":
        return [event for t in range(arguments[0]) for event in (
            [(0, (t,), 5, 0)] + [(1, (t, i), 4, 0) for i in range(t)])]
    if name == "siblings":
        n, k = arguments
        return ([(0, (), 5, 0)] + [(1, (i,), 4, 0) for i in range(n)]
                + [(2, (i,), 5, 0) for i in range(k)] + [(3, (), 4, 0)])
    if name == "empty_minimum":
        return []
    raise RuntimeError("unknown fixture " + name)


def relation_index(document):
    result = {}
    for relation in document["relations"]:
        key = (relation["kind"], relation["source"], relation["target"],
               relation["source_event"], relation["target_event"], relation["space"])
        # Direct residue indexing also verifies each queried point's unique split.
        for piece in relation["pieces"]:
            if not piece["empty"]:
                result.setdefault(key, {}).setdefault(tuple(piece["residues"]), []).append(piece["rows"])
    return result


def contains(index, period, key, coordinates):
    residues = tuple(x % period for x in coordinates)
    quotient = tuple(x // period for x in coordinates)
    for rows in index.get(key, {}).get(residues, []):
        valid = True
        for row in rows:
            require(len(row["coefficients"]) == len(quotient), "coordinate mismatch")
            value = sum(a * b for a, b in zip(row["coefficients"], quotient)) + row["constant"]
            valid &= value == 0 if row["equality"] else value >= 0
        if valid:
            return True
    return False


def verify(document):
    name = document["function"]
    if name in ("ssa_prerequisite", "existing_sync", "opaque_bound"):
        require(not document["relations"] and not document["sites"], "partial exports on rejection")
        return 0
    schemas = {
        "nested_reset": ([2, 2, 1], [0, 1]),
        "triangular": ([1, 2], [0]),
        "siblings": ([0, 1, 1, 0], [0, 1]),
        "empty_minimum": ([1], []),
    }
    depths, parameters = schemas[name]
    require([site["depth"] for site in document["sites"]] == depths, name + ": site schema")
    require(document["parameters"] == parameters, name + ": parameter schema")
    index = relation_index(document)
    period = document["period"]
    count = 0
    parameter_count = len(document["parameters"])
    for arguments in itertools.product(range(-1, 4), repeat=parameter_count):
        trace = traces(name, arguments)
        positions = {(site, coords): pos for pos, (site, coords, _, _) in enumerate(trace)}
        candidates = [(site, coords) for site, record in enumerate(document["sites"])
                      for coords in itertools.product(range(-1, 4), repeat=record["depth"])]
        for source, left in candidates:
            expected = (source, left) in positions
            require(contains(index, period, (1, source, -1, 0, 0, -1), left + arguments) == expected,
                    f"{name}: occurrence {source} {left} {arguments}")
            count += 1
            for target, right in candidates:
                before = (expected and (target, right) in positions
                          and positions[source, left] < positions[target, right])
                coords = left + right + arguments
                require(contains(index, period, (2, source, target, 0, 0, -1), coords) == before,
                        f"{name}: order {source,left} {target,right} {arguments}")
                same_pipe = document["sites"][source]["pipe"] == document["sites"][target]["pipe"]
                for start, end in ((1, 1), (2, 2), (1, 2), (2, 1)):
                    native = same_pipe and ((before and (start, end) != (2, 1)) or
                                           (expected and (source, left) == (target, right) and (start, end) == (1, 2)))
                    require(contains(index, period, (3, source, target, start, end, -1), coords) == native,
                            f"{name}: native {source,left} {target,right} {start,end} {arguments}")
                count += 5
            # Check byte boundaries, holes between slots, and an incorrect space.
            access = next((event for event in trace if event[:2] == (source, left)), None)
            for kind in (4, 5):
                for byte in (-1, 0, 1, 3, 4, 1023, 1024, 1027, 1028):
                    for space in (6, 0):
                        wanted = bool(access and access[2] == kind and access[3] <= byte < access[3] + 4 and space == 6)
                        observed = contains(index, period, (kind, source, -1, 0, 0, space), left + (byte,) + arguments)
                        require(observed == wanted, f"{name}: access {source,left} {kind,space,byte} {arguments}")
                        count += 1
    return count


def main():
    result = subprocess.run([sys.argv[1], "--arithmetic", sys.argv[2]], check=True,
                            capture_output=True, text=True, timeout=60)
    documents = [json.loads(line[len("arithmetic-json "):]) for line in result.stdout.splitlines()
                 if line.startswith("arithmetic-json ")]
    require(len(documents) == 7, "missing fixture functions")
    comparisons = sum(verify(document) for document in documents)
    print(f"arithmetic program checks passed: {comparisons} comparisons")


if __name__ == "__main__":
    main()
