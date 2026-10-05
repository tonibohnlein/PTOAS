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
        assert actual == {edge: identities[edge] for edge in covers}, (case, sample, actual, covers)
        thresholds = values[len(records):]
        assert len(thresholds) == 2 * (2 * m)**2
        for a in range(2 * m):
            for b in range(2 * m):
                index = 2 * (a * (2 * m) + b)
                reachable, distance = thresholds[index:index + 2]
                assert reachable in (0, 1)
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
    # Overflow is rejected statically rather than wrapping min-plus distances.
    cases.append({"pipes": [0, 1], "records": [
        {"source": 0, "target": 1, "bound": 2**64 - 1}], "samples": []})
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
    assert results[-1]["error"], results[-1]
    valuations = sum(len(case["samples"]) for case in cases)
    print("guarded quotient:", valuations, "valuations and all-event finite DAGs passed")


if __name__ == "__main__":
    main()
