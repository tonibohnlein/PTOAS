# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check supplied rotating effects against all physical conflicts in finite runs.

The oracle expands physical accesses and compares generic transitive closures.
It does not compute phases, circular writer neighbors, or modular inverses.
Huge values have analytic expectations and sampled physical-witness checks.
"""

import copy
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys


def expect(actual, expected, label):
    """Fail in release Python too, with a reproducible mismatch."""
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def case(sites, families, fragments, **options):
    """Encode arbitrary integers as decimal strings for lossless JSON transport."""
    return dict(sites=sites,
                families=[[str(slots), str(stride), atoms] for slots, stride, atoms in families],
                fragments=[[site, family, atom, str(offset), mode]
                           for site, family, atom, offset, mode in fragments], **options)


def normalized(instance):
    """Obtain the canonical physical descriptors independently with a dictionary."""
    modes = {}
    for site, family, atom, offset, mode in instance["fragments"]:
        key = (family, atom, int(offset) % int(instance["families"][family][0]), site)
        modes[key] = modes.get(key, 0) | mode
    return [[site, family, atom, str(offset), modes[family, atom, offset, site]]
            for family, atom, offset, site in sorted(modes)]


def physical(instance, fragment, iteration):
    """Evaluate a declared physical cell, using Python's Euclidean modulo."""
    _, family, atom, offset, _ = fragment
    slots, stride, _ = instance["families"][family]
    return family, (int(stride) * iteration + int(offset)) % int(slots), atom


def expanded_conflicts(instance, trips):
    """Keep all reads/writes, and enumerate every ordered conflicting pair."""
    count = instance["sites"]
    effects = [[set(), set()] for _ in range(count * trips)]
    for iteration in range(trips):
        for fragment in instance["fragments"]:
            site, _, _, _, mode = fragment
            cell = physical(instance, fragment, iteration)
            for bit in (0, 1):
                if mode & (1 << bit):
                    effects[iteration * count + site][bit].add(cell)
    edges = set()
    for consumer, (reads, writes) in enumerate(effects):
        for source in range(consumer):
            old_reads, old_writes = effects[source]
            if old_writes & (reads | writes) or old_reads & writes:
                edges.add((source, consumer))
    return edges


def closure(count, edges):
    """Generic DAG reachability; deliberately independent of lifetime frontiers."""
    successors = [[] for _ in range(count)]
    for source, consumer in edges:
        if not 0 <= source < consumer < count:
            raise ValueError("nonforward or out-of-range occurrence edge")
        successors[source].append(consumer)
    reachable = [0] * count
    for source in reversed(range(count)):
        for consumer in successors[source]:
            reachable[source] |= (1 << consumer) | reachable[consumer]
    return reachable


def instantiated(instance, result, trips):
    """Restrict periodic records to endpoints present in one invocation."""
    count = instance["sites"]
    return {(iteration * count + edge["source"],
             (iteration + int(edge["distance"])) * count + edge["consumer"])
            for edge in result["edges"]
            for iteration in range(max(0, trips - int(edge["distance"])))}


def endpoints(result):
    """Return records without their independent witnesses."""
    return {(edge["source"], edge["consumer"], int(edge["distance"])) for edge in result["edges"]}


def check_witnesses(instance, result):
    """Every reported hazard must identify actual matching physical effects."""
    fragments = result["fragments"]
    for edge in result["edges"]:
        source, consumer, distance = edge["source"], edge["consumer"], int(edge["distance"])
        expect(0 <= source < instance["sites"] and 0 <= consumer < instance["sites"], True, "site bounds")
        expect(0 <= distance <= int(result["bound"]), True, "distance bounds")
        expect(distance > 0 or source < consumer, True, "strict occurrence order")
        expect(bool(edge["witnesses"]), True, "nonempty witnesses")
        for hazard, first, second in edge["witnesses"]:
            expect(0 <= first < len(fragments) and 0 <= second < len(fragments), True, "fragment bounds")
            left, right = fragments[first], fragments[second]
            expect((left[0], right[0]), (source, consumer), "witness sites")
            expect(hazard in (0, 1, 2), True, "hazard kind")
            expect(bool(left[4] & (1 if hazard == 1 else 2)), True, "source mode")
            expect(bool(right[4] & (1 if hazard == 0 else 2)), True, "consumer mode")
            for iteration in (-17, 0, 1, 100000000000000000003):
                expect(physical(instance, left, iteration), physical(instance, right, iteration + distance),
                       "physical overlap")


def check(instance, result):
    """Verify certificates, sparse encoding and all finite physical-conflict paths."""
    if instance.get("invalid"):
        expect(result, {"valid": False, "empty": True}, "atomic failure")
        return
    expect(result["valid"], True, "valid input")
    expect(result["fragments"], normalized(instance), "mode/offset normalization")
    certificates = []
    for family in sorted({row[1] for row in instance["fragments"]}):
        slots, stride, _ = instance["families"][family]
        divisor = math.gcd(int(slots), int(stride))
        certificates.append([family, str(divisor), str(int(slots) // divisor)])
    expect(result["certificates"], certificates, "family certificates")
    bound = max((int(row[2]) for row in certificates), default=0)
    expect(int(result["bound"]), bound, "distance certificate")
    expect(len(endpoints(result)), len(result["edges"]), "endpoint deduplication")
    expect(len(result["edges"]) <= 2 * len(result["fragments"]), True, "sparse record bound")
    check_witnesses(instance, result)
    if "expected" in instance:
        expect(endpoints(result), set(map(tuple, instance["expected"])), "analytic generators")
    if "witness_families" in instance:
        for edge in result["edges"]:
            witnessed = {result["fragments"][first][1] for _, first, _ in edge["witnesses"]}
            expect(witnessed, set(instance["witness_families"]), "merged physical witnesses")
    trips = (0, 1, 2, 3) if bound > 20 else sorted({0, 1, 2, max(0, bound - 1), bound, bound + 1, 2 * bound + 1})
    for count in trips:
        all_edges = expanded_conflicts(instance, count)
        supplied = instantiated(instance, result, count)
        expect(supplied <= all_edges, True, "every generated instance is a conflict")
        expect(closure(instance["sites"] * count, supplied), closure(instance["sites"] * count, all_edges),
               f"all-conflict closure at {count} trips")


def named_cases():
    """Stationary, disjoint atoms/residues, RMW, duplicate modes and initial readers."""
    yield case(0, [], [])
    yield case(3, [(5, 1, 2)], [])
    yield case(1, [(1, 0, 1)], [(0, 0, 0, 0, 3)], expected=[(0, 0, 1)])
    yield case(3, [(1, 0, 1)], [(0, 0, 0, 0, 2), (1, 0, 0, 0, 1), (2, 0, 0, 0, 2)],
               expected=[(0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 0, 1)])
    yield case(3, [(3, -1, 2)], [(0, 0, 0, -4, 1), (1, 0, 0, 2, 2), (2, 0, 1, 0, 1)])
    yield case(2, [(6, 2, 1)], [(0, 0, 0, 0, 2), (1, 0, 0, 1, 1)], expected=[(0, 0, 3)])
    yield case(2, [(3, 0, 2)], [(0, 0, 0, 0, 1), (0, 0, 0, 3, 2), (1, 0, 1, 0, 2)])
    yield case(2, [(2, 1, 1), (3, -1, 1), (5, 2, 1)],
               [(site, family, 0, site, mode) for family in range(3) for site, mode in ((0, 2), (1, 1))])
    yield case(2, [(2, 1, 1), (2, 1, 1)],
               [(site, family, 0, 0, mode) for family in range(2) for site, mode in ((0, 2), (1, 1))],
               witness_families=[0, 1])
    yield case(2, [(5, -3, 2)], [(0, 0, 0, 0, 1), (1, 0, 1, 0, 1)], expected=[])


def grouped_cases():
    """Transform original iterations into positions borrowing repeated anchors."""
    for group in (2, 3):
        for slots, stride in ((1, 0), (6, 2), (5, -2)):
            original = case(3, [(slots, stride, 2)],
                            [(0, 0, 0, 0, 1), (1, 0, 0, 1, 3), (2, 0, 1, -1, 2),
                             (2, 0, 0, 0, 1)])
            grouped = case(3 * group, [(slots, group * stride, 2)],
                           [(3 * phase + site, family, atom, stride * phase + int(offset), mode)
                            for phase in range(group)
                            for site, family, atom, offset, mode in original["fragments"]],
                           site_refs=list(range(3)) * group, original=original, group=group)
            yield grouped


def check_grouped(instance, result):
    """Compare partial grouped prefixes with original physical conflicts."""
    original, group = instance["original"], instance["group"]
    for trips in range(3 * group + 1):
        count = original["sites"] * trips
        edges = {(source, consumer)
                 for source, consumer in instantiated(instance, result, (trips + group - 1) // group)
                 if source < count and consumer < count}
        physical_edges = expanded_conflicts(original, trips)
        expect(edges <= physical_edges, True, "grouped physical witnesses")
        expect(closure(count, edges), closure(count, physical_edges), "partial grouped prefix")


def generated_cases():
    """Exhaustive small pairs and reproducible multi-family/multi-atom cases."""
    for slots in (1, 2, 3, 5):
        for stride in (-2, 0, 1, 2):
            for left, right in itertools.product((1, 2, 3), repeat=2):
                for offset in range(slots):
                    yield case(2, [(slots, stride, 1)], [(0, 0, 0, 0, left), (1, 0, 0, offset, right)])
    rng = random.Random(90210)
    for _ in range(420):
        sites = rng.randrange(1, 7)
        families = [(rng.choice((1, 2, 3, 5, 6)), rng.randrange(-9, 10), rng.randrange(1, 4))
                    for _ in range(rng.randrange(1, 4))]
        fragments = []
        for _ in range(rng.randrange(1, 18)):
            family = rng.randrange(len(families))
            fragments.append((rng.randrange(sites), family, rng.randrange(families[family][2]),
                              rng.randrange(-19, 20), rng.randrange(1, 4)))
        yield case(sites, families, fragments)


def huge_cases():
    """Analytic records exercise large products, negative strides and inverses."""
    for slots in (2**257 + 1, 10**500 + 7):
        distance = slots // 3
        yield case(2, [(slots, 2, 1)], [(0, 0, 0, 0, 2), (1, 0, 0, -2 * distance, 1)],
                   expected=[(0, 0, slots), (0, 1, distance), (1, 0, slots - distance)])
    slots, stride = 6 * 2**256, -4 * 2**128
    refresh = slots // math.gcd(slots, stride)
    yield case(1, [(slots, stride, 1)], [(0, 0, 0, -(2**600), 3)], expected=[(0, 0, refresh)])
    yield case(2, [(2**300, -(2**900), 1)], [(0, 0, 0, 2**999, 1), (1, 0, 0, 0, 2)],
               expected=[(1, 0, 1), (0, 1, 0), (1, 1, 1)])


def invalid_cases():
    """Malformed semantic representations are passed through to production."""
    base = case(2, [(2, 1, 1)], [(0, 0, 0, 0, 2)], invalid=True)
    for key, position, value in (("families", 0, "0"), ("families", 0, "-3"), ("families", 2, 0),
                                 ("fragments", 0, 2), ("fragments", 1, 1), ("fragments", 2, 1),
                                 ("fragments", 4, 0), ("fragments", 4, 4)):
        item = copy.deepcopy(base)
        item[key][0][position] = value
        yield item
    yield dict(base, null_site=True)
    yield case(0, [(2, 1, 1)], [(0, 0, 0, 0, 2)], invalid=True)


def mutations(instance, result):
    """Prove that the verifier rejects six independent corruptions."""
    candidates = []
    changed = copy.deepcopy(result)
    changed["edges"] = []
    candidates.append(changed)
    changed = copy.deepcopy(result)
    changed["edges"][0]["distance"] = "0"
    candidates.append(changed)
    changed = copy.deepcopy(result)
    changed["edges"][0]["witnesses"] = []
    candidates.append(changed)
    changed = copy.deepcopy(result)
    changed["bound"] = "0"
    candidates.append(changed)
    changed = copy.deepcopy(result)
    changed["fragments"][0][4] = 1
    candidates.append(changed)
    changed = copy.deepcopy(result)
    changed["edges"].append(copy.deepcopy(changed["edges"][0]))
    candidates.append(changed)
    for candidate in candidates:
        try:
            check(instance, candidate)
        except ValueError:
            continue
        raise ValueError("oracle accepted a deliberately corrupted result")
    return len(candidates)


def main():
    """Run a bounded driver batch and check every result before reporting."""
    if len(sys.argv) != 3:
        raise ValueError("usage: check_rotating_analysis.py driver scratch.json")
    driver = shutil.which(sys.argv[1])
    if driver is None:
        raise ValueError("rotating driver not found")
    path = Path(sys.argv[2]).resolve()
    cases = (list(named_cases()) + list(generated_cases()) + list(huge_cases()) + list(invalid_cases())
             + list(grouped_cases()))
    try:
        path.write_text(json.dumps(cases), encoding="utf-8")
        completed = subprocess.run([str(Path(driver).resolve()), str(path)], check=True, capture_output=True,
                                   text=True, timeout=60)
        results = json.loads(completed.stdout)
    finally:
        path.unlink(missing_ok=True)
    expect(len(results), len(cases), "batch size")
    for position, (instance, result) in enumerate(zip(cases, results)):
        try:
            check(instance, result)
            if "original" in instance:
                check_grouped(instance, result)
        except ValueError as error:
            raise ValueError(f"case {position}: {instance}") from error
    count = mutations(cases[2], results[2])
    print(f"verified {len(cases)} rotating cases: physical conflict closure, witnesses, certificates, atomic failure")
    print(f"verified {count} rotating oracle mutations")


if __name__ == "__main__":
    main()
