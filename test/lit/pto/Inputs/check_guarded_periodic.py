# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Evaluate shared symbolic periodic circuits against independent finite DAGs."""
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from check_periodic_demands import closure, native


def dense_thresholds(case, sample):
    """Independent old-style scalar Floyd oracle on actual present sites."""
    m = len(case["pipes"])
    distance = [[None] * (2 * m) for _ in range(2 * m)]

    def edge(a, b, weight):
        old = distance[a][b]
        distance[a][b] = weight if old is None else min(old, weight)

    for a in range(m):
        if not sample[a]:
            continue
        edge(2 * a, 2 * a, 0)
        edge(2 * a + 1, 2 * a + 1, 0)
        edge(2 * a, 2 * a + 1, 0)
        for b in range(m):
            if sample[b] and case["pipes"][a] == case["pipes"][b]:
                edge(2 * a, 2 * b, int(a >= b))
                edge(2 * a + 1, 2 * b + 1, int(a >= b))
    weighted = [(r, sample[m + 2 * i], sample[m + 2 * i + 1])
                for i, r in enumerate(case["records"])]
    weighted += [(r, 1, r["distance"]) for r in case.get("native_prerequisites", [])]
    for record, enabled, weight in weighted:
        a, b = record["source"], record["target"]
        if enabled and sample[a] and sample[b]:
            edge(2 * a + 1, 2 * b, weight)
    for k in range(2 * m):
        for a in range(2 * m):
            for b in range(2 * m):
                if distance[a][k] is not None and distance[k][b] is not None:
                    edge(a, b, distance[a][k] + distance[k][b])
    return distance


def check(case, result, trips=7):
    assert not result["error"], result
    m = len(case["pipes"])
    records = case["records"]
    assert len(result["values"]) == len(case["samples"])
    for sample, values in zip(case["samples"], result["values"]):
        present = sample[:m]
        active = sample[m::2]
        distances = sample[m + 1::2]
        occurrences = [(iteration, site) for iteration in range(trips)
                       for site in range(m) if present[site]]
        position = {occurrence: i for i, occurrence in enumerate(occurrences)}
        edges = native([case["pipes"][site] for _, site in occurrences])
        fixed = set()
        for record in case.get("native_prerequisites", []):
            a, b, distance = record["source"], record["target"], record["distance"]
            if present[a] and present[b]:
                fixed.update((2 * position[i, a] + 1, 2 * position[i + distance, b])
                             for i in range(trips - distance))
        edges |= fixed
        identities = {}
        for ident, (record, enabled, distance) in enumerate(zip(records, active, distances)):
            assert distance <= record["bound"]
            source, target = record["source"], record["target"]
            if not enabled or not present[source] or not present[target]:
                continue
            assert distance or source < target
            for iteration in range(trips - distance):
                edge = (2 * position[iteration, source] + 1, 2 * position[iteration + distance, target])
                edges.add(edge)
                identities.setdefault(edge, ident)
        reach, covers = closure(2 * len(occurrences), edges)
        actual = {}
        for ident, retained in enumerate(values[:len(records)]):
            assert retained in (0, 1)
            if not retained:
                continue
            record, distance = records[ident], distances[ident]
            assert active[ident] and present[record["source"]] and present[record["target"]]
            for iteration in range(trips - distance):
                edge = (2 * position[iteration, record["source"]] + 1,
                        2 * position[iteration + distance, record["target"]])
                assert edge not in actual, "duplicate retained endpoint pair"
                actual[edge] = ident
        assert actual == {edge: identities[edge] for edge in covers - fixed}, (case, sample, actual, covers)
        thresholds = values[len(records):]
        assert len(thresholds) == 2 * (2 * m)**2
        dense = dense_thresholds(case, sample)
        for a in range(2 * m):
            for b in range(2 * m):
                index = 2 * (a * (2 * m) + b)
                reachable, distance = thresholds[index:index + 2]
                assert reachable in (0, 1)
                assert (distance if reachable else None) == dense[a][b], (case, sample, a, b, distance, dense[a][b])
                if not present[a // 2] or not present[b // 2]:
                    assert not reachable
                    continue
                for source_iteration in range(trips):
                    source = 2 * position[source_iteration, a // 2] + a % 2
                    for target_iteration in range(trips):
                        target = 2 * position[target_iteration, b // 2] + b % 2
                        predicted = bool(reachable and target_iteration >= source_iteration
                                         and target_iteration - source_iteration >= distance)
                        assert predicted == bool(reach[source] & (1 << target)), (sample, a, b, distance)


def main():
    tool = sys.argv[1]
    cases = []
    randomizer = random.Random(606)
    # Shared-pipe singleton valuations exercise selfwrap; disappearing middle
    # sites require native skip edges. Repeated endpoint records exercise ties.
    for pipes in ([0], [0, 0, 0], [0, 1, 0], [0, 1, 2, 0]):
        m = len(pipes)
        records = [{"source": a, "target": b, "bound": 3}
                   for a in range(m) for b in range(m)]
        records += [dict(record) for record in records[:2]]
        samples = []
        for presence in itertools.product((0, 1), repeat=m):
            for variant in range(3):
                sample = list(presence)
                for record in records:
                    distance = randomizer.randrange(0 if record["source"] < record["target"] else 1, 4)
                    sample += [1 if variant == 0 else randomizer.randrange(2), distance]
                # Force equal enabled duplicate records for deterministic first wins.
                if variant == 0:
                    for i in range(min(2, m * m)):
                        sample[m + 2 * (m * m + i):m + 2 * (m * m + i) + 2] = sample[m + 2 * i:m + 2 * i + 2]
                samples.append(sample)
        cases.append({"pipes": pipes, "records": records, "samples": samples})
    # Symbolic distance bounds change only encoded metadata. Never unfold the
    # billion-sized bound; compare the same circuit and small actual valuation.
    for bound in (7, 1000000007):
        cases.append({"pipes": [0, 1, 0], "records": [
            {"source": 0, "target": 1, "bound": bound},
            {"source": 1, "target": 0, "bound": bound}],
            "samples": [[1, 1, 1, 1, 1, 1, 2]]})
    # Vary potential site count with independent presence and record guards.
    for m in (4, 8, 12):
        cases.append({"pipes": [i % 3 for i in range(m)], "records": [
            {"source": a, "target": b, "bound": 3} for a in range(m) for b in range(m)], "samples": []})
    # Native prerequisites participate in every exclusion, including ties with
    # a removable record. All-presence valuations check latent skip wires.
    cases.append({"pipes": [0, 1, 0], "records": [
        {"source": 0, "target": 1, "bound": 1}, {"source": 1, "target": 2, "bound": 2}],
        "native_prerequisites": [{"source": 0, "target": 1, "distance": 0}],
        "samples": [list(p) + [a, 0, b, 1] for p in itertools.product((0, 1), repeat=3)
                    for a, b in itertools.product((0, 1), repeat=2)]})
    cases.append({"pipes": [], "records": [], "samples": [[]]})
    # Largest accepted scaled-score bound: h=2, V=4, seed at most1.
    # Evaluate the large distance itself, beyond every unfolded test prefix.
    maximum_safe = (2**64 - 2) // 8
    cases.append({"pipes": [0, 0], "records": [
        {"source": 0, "target": 1, "bound": maximum_safe}],
        "samples": [[1, 1, 1, maximum_safe], [0, 1, 1, maximum_safe]]})
    for case in cases:
        case["export_thresholds"] = True
    sparse_begin = len(cases)
    # Construction-only requests explicitly omit quadratic threshold export.
    # Fixed three pipes and linear edges give quadratic relaxation work.
    for m in (6, 12, 24):
        cases.append({"pipes": [i % 3 for i in range(m)], "records": [
            {"source": i, "target": (i + 1) % m, "bound": 3} for i in range(m)],
            "samples": [], "export_thresholds": False})
    # Overflow is rejected statically rather than wrapping min-plus distances.
    cases.append({"pipes": [0, 0], "records": [
        {"source": 0, "target": 1, "bound": maximum_safe + 1}], "samples": []})
    with tempfile.TemporaryDirectory(prefix="guarded-quotient-") as scratch:
        path = Path(scratch) / "request.json"
        path.write_text(json.dumps(cases))
        run = subprocess.run([tool, "--guarded-periodic-checks", str(path)], capture_output=True,
                             text=True, check=False, timeout=120)
        assert run.returncode == 0, run.stderr + run.stdout
        results = json.loads(run.stdout)
    assert len(results) == len(cases)
    for case, result in zip(cases[:6], results[:6]):
        check(case, result)
    for field in ("expressions", "emitted", "edges"):
        assert results[4][field] == results[5][field], (field, results[4], results[5])
    for case, result in zip(cases[6:9], results[6:9]):
        assert not result["error"], result
        m = len(case["pipes"])
        assert result["expressions"] < 200 * (2 * m)**3, result
        metrics = {key: result[key] for key in ("expressions", "emitted", "edges")}
        print("guarded quotient size:", {"sites": m, **metrics})
    for case, result in zip(cases[9:sparse_begin], results[9:sparse_begin]):
        check(case, result)
    for case, result in zip(cases[:-1], results[:-1]):
        m, k = len(case["pipes"]), len(set(case["pipes"]))
        edges = result["edges"]
        assert result["frontier_entries"] == 4 * m * k
        assert result["relaxations"] == 2 * k * max(0, 2 * m - 1) * edges
        assert result["exclusions"] == k * (edges - 2 * m)
        assert result["threshold_queries"] == ((2 * m)**2 if case["export_thresholds"] else 0)
        assert result["scaling_additions"] <= 2 * k * edges * max(1, m.bit_length())
    for case, result in zip(cases[sparse_begin:-1], results[sparse_begin:-1]):
        m = len(case["pipes"])
        assert result["edges"] == 4 * m
        assert result["expressions_after_queries"] == result["expressions"]
        assert result["expressions"] < 200 * (len(case["records"]) + 3 * (2 * m) * result["edges"])
        print("sparse guarded frontier:", {"sites": m, **{key: result[key] for key in
              ("edges", "frontier_entries", "relaxations", "exclusions", "expressions")}})
    assert results[-1]["error"], results[-1]
    valuations = sum(len(case["samples"]) for case in cases)
    print("guarded quotient:", valuations, "valuations and all-event finite DAGs passed")


if __name__ == "__main__":
    main()
