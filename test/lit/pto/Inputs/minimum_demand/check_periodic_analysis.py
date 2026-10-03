# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Numeric quotient oracle using independently unfolded start/completion DAGs.

Generic finite reachability and covers never use weighted paths or incoming-edge
minima. Rotating cases derive every conflict from physical accesses. Small
certified windows and explicit rank profiles provide additional comparisons.
"""

import copy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from check_rotating_analysis import case, closure, expect, expanded_conflicts, named_cases


def pipe_sequence(instance):
    """A grouped period may borrow one shared phase at several positions."""
    references = instance.get("site_refs", range(instance["sites"]))
    return [instance["pipes"][reference] for reference in references]


def unfold(records, width, count):
    """Instantiate only record endpoints present in a finite prefix."""
    return {(source + period * width, consumer + (period + int(distance)) * width)
            for source, consumer, distance in records for period in range((count + width - 1) // width)
            if source + period * width < count and consumer + (period + int(distance)) * width < count}


def input_edges(instance, count):
    """Raw records or all physical conflicts form the independent input graph."""
    width = instance["sites"]
    if not width:
        return set()
    edges = unfold(instance["generators"], width, count)
    if instance.get("from_storage"):
        edges |= {(source, target) for source, target in expanded_conflicts(instance, (count + width - 1) // width)
                  if target < count}
    return edges


def event_graph(instance, count, supplied=None):
    """I->C and separate I/I, C/C chains never imply native C->next-I."""
    pipes = pipe_sequence(instance)
    demands = input_edges(instance, count) if supplied is None else supplied
    edges = {(2 * source + 1, 2 * target) for source, target in demands}
    previous = {}
    for occurrence in range(count):
        edges.add((2 * occurrence, 2 * occurrence + 1))
        pipe = pipes[occurrence % len(pipes)]
        if pipe in previous:
            prior = previous[pipe]
            edges.add((2 * prior, 2 * occurrence))
            edges.add((2 * prior + 1, 2 * occurrence + 1))
        previous[pipe] = occurrence
    reached = closure(2 * count, edges)
    predecessors = [0] * (2 * count)
    for source, targets in enumerate(reached):
        while targets:
            bit = targets & -targets
            predecessors[bit.bit_length() - 1] |= 1 << source
            targets ^= bit
    covers = {(source, target) for source, target in demands
              if not reached[2 * source + 1] & predecessors[2 * target]}
    return reached, covers


def finite_thresholds(instance, bound, source_kind=1):
    """A simple quotient path has at most 2m-1 edges; expand this small test bound."""
    width = instance["sites"]
    periods = (2 * width - 1) * max(1, bound) + 1 if width else 0
    reached, _ = event_graph(instance, width * periods)
    matrices = []
    for kind in (0, 1):
        matrix = []
        for source in range(width):
            row = []
            for target in range(width):
                candidates = [period for period in range(periods)
                              if reached[2 * source + source_kind] & (1 << (2 * (period * width + target) + kind))]
                if kind == source_kind and source == target:
                    candidates.append(0)
                row.append(str(min(candidates)) if candidates else None)
            matrix.append(row)
        matrices.append(matrix)
    return matrices


def frontier_from_graph(instance, count, consumer, reached, kind):
    """Maximum required completion ranks, including identity only for C queries."""
    pipes = pipe_sequence(instance)
    columns = list(dict.fromkeys(pipes))
    ranks = dict.fromkeys(columns, 0)
    result = dict.fromkeys(columns, 0)
    for source in range(count):
        pipe = pipes[source % len(pipes)]
        ranks[pipe] += 1
        if reached[2 * source + 1] & (1 << (2 * consumer + kind)) or (kind == 1 and source == consumer):
            result[pipe] = ranks[pipe]
    return [result[pipe] for pipe in columns]


def check_injection(instance, covers):
    """Nonnative covers are partial injections for each ordered pipe pair."""
    pipes = pipe_sequence(instance)
    sources, targets = set(), set()
    for source, target in covers:
        pair = pipes[source % len(pipes)], pipes[target % len(pipes)]
        if (*pair, source) in sources or (*pair, target) in targets:
            raise ValueError("covers violate pipe-pair partial injection")
        sources.add((*pair, source))
        targets.add((*pair, target))


def check_prefixes(instance, result, retained):
    """Compare finite covers, every S/T coordinate, and the production rank route."""
    width = instance["sites"]
    expect([summary["prefix"] for summary in result["prefixes"]], instance["prefixes"], "prefix requests")
    for summary in result["prefixes"]:
        count = summary["prefix"]
        reached, covers = event_graph(instance, count)
        chosen = unfold(retained, width, count) if width else set()
        expect(chosen, covers, "finite canonical F*")
        expect(event_graph(instance, count, chosen)[0], reached, "retained event closure")
        check_injection(instance, covers)
        expect(len(summary["rows"]), count, "profile count")
        for consumer, row in enumerate(summary["rows"]):
            for kind, label in enumerate(("S", "T")):
                expected = frontier_from_graph(instance, count, consumer, reached, kind)
                expect(list(map(int, row[label])), expected, f"{label} event profile")
                explicit = dict(zip(summary["pipes"], row[f"explicit_{label}"]))
                expect([explicit.get(pipe, 0) for pipe in result["pipes"]], expected, f"{label} explicit profile")


def check_queries(instance, result):
    """Invalid queries remain distinct from valid infinity/unreachable answers."""
    width = instance["sites"]
    expected = []
    for source, target, kind, distance in instance["queries"]:
        kind, distance = int(kind), int(distance)
        if source >= width or target >= width or kind not in (0, 1) or distance < 0:
            expected.append(None)
        else:
            threshold = result["start" if kind == 0 else "completion"][source][target]
            expected.append(threshold is not None and int(threshold) <= distance)
    expect(result["queries"], expected, "reachability queries")
    for query, actual in zip(instance["frontier_queries"], result["frontier_queries"]):
        target, kind, period, prefix = query
        period, prefix = int(period), int(prefix)
        if target >= width or kind not in (0, 1) or min(period, prefix) < 0 or period * width + target >= prefix:
            expect(actual, None, "invalid frontier query")
        elif prefix <= 128:
            reached, _ = event_graph(instance, prefix)
            expected = frontier_from_graph(instance, prefix, period * width + target, reached, kind)
            expect(list(map(int, actual)), expected, "finite frontier query")
    if "huge_frontiers" in instance:
        expect(result["frontier_queries"], instance["huge_frontiers"], "analytic huge frontiers")


def check(instance, result):
    """Validate canonical origins, all query thresholds and representative windows."""
    if instance.get("invalid"):
        expect(result, {"valid": False, "empty": True}, "atomic invalid input")
        return
    expect(result["valid"], True, "valid quotient")
    supplied = result.get("storage", []) + instance["generators"]
    canonical = sorted({(a, b, int(d)) for a, b, d in supplied})
    expected = [[a, b, str(d), [index for index, edge in enumerate(supplied)
                              if (edge[0], edge[1], int(edge[2])) == (a, b, d)]] for a, b, d in canonical]
    expect(result["records"], expected, "canonical records and origins")
    width = instance["sites"]
    expect(result["vertices"], 2 * width, "quotient vertices")
    expect(result["edges"], 3 * width + len(canonical), "quotient edges")
    expect(result["pipes"], list(dict.fromkeys(pipe_sequence(instance))), "pipe columns")
    expect(result["runs"], len(result["pipes"]), "one shortest-path run per pipe")
    expect(result["index_entries"], 2 * width * len(result["pipes"]), "compact per-pipe frontier index")
    expect(result["retained"], sorted(result["retained"]), "stable canonical record identity order")
    expect(len(set(result["retained"])), len(result["retained"]), "unique retained IDs")
    expect(all(0 <= index < len(canonical) for index in result["retained"]), True, "retained ID bounds")
    retained = [canonical[index] for index in result["retained"]]
    bound = max((distance for _, _, distance in canonical), default=0)
    if "huge_thresholds" in instance:
        matrices = instance["huge_thresholds"]
        expect(set(retained), set(map(tuple, instance["huge_retained"])), "analytic huge covers")
    else:
        matrices = finite_thresholds(instance, bound)
        _, window_covers = event_graph(instance, (bound + 1) * width)
        representatives = {(a, b % width, b // width) for a, b in window_covers if a < width}
        expect(set(retained), representatives, "certified representative window")
    expect([result["start"], result["completion"]], matrices, "all required-order thresholds")
    if "huge_thresholds" in instance:
        # These supplied huge examples have distinct pipes; native starts add
        # only reflexive I identity. Finite examples use the independent DAG.
        start_matrices = copy.deepcopy(matrices)
        for source in range(width):
            start_matrices[0][source][source] = "0"
    else:
        start_matrices = finite_thresholds(instance, bound, source_kind=0)
    expect([result["from_start_to_start"], result["from_start_to_completion"]],
           start_matrices, "all start-origin thresholds")
    check_prefixes(instance, result, retained)
    check_queries(instance, result)


def periodic(pipes, records, **options):
    """Build a supplied-record fixture with finite-prefix and invalid-query probes."""
    width = len(pipes)
    bound = max((int(edge[2]) for edge in records), default=0)
    prefixes = sorted({0, 1, max(0, width - 1), width, width + 1, 2 * width + 1,
                       (min(bound, 4) + 1) * width}) if width else [0]
    return dict(case(width, [], []), periodic=True, pipes=pipes,
                generators=[[source, target, str(distance)] for source, target, distance in records], prefixes=prefixes,
                queries=[[0, 0, "0", "0"], [0, 0, "1", "0"], [0, width, "0", "0"],
                         [width, 0, "0", "0"], [0, 0, "2", "0"], [0, 0, "0", "-1"]],
                frontier_queries=[[0, 0, "0", "0"], [0, 0, "-1", "2"], [0, 0, "0", "-1"],
                                  [width, 0, "0", "2"], [0, 2, "0", "2"]], **options)


def fixtures():
    """Ties, parallel weights, native-only chains, grouped anchors and empty bodies."""
    yield periodic([], [])
    yield periodic([1], [])
    yield periodic([1, 1, 4], [])
    yield periodic([1, 2], [(0, 1, 0), (0, 1, 0), (0, 1, 1), (1, 0, 2), (0, 0, 2)])
    yield periodic([1, 2, 3], [(0, 1, 0), (1, 2, 0), (0, 2, 0)])
    yield periodic([1, 2, 2], [(0, 1, 1), (0, 2, 1), (0, 2, 2)])
    yield periodic([1, 1, 2, 2], [(0, 2, 0), (1, 3, 0), (0, 3, 0), (3, 0, 2)])
    for periods in (2, 3):
        yield periodic([1, 2] * periods, [(0, 1, 0), (1, 2, 0), (2 * periods - 1, 0, 1)],
                       site_refs=[0, 1] * periods)
    rng = random.Random(783145)
    for _ in range(300):
        width = rng.randrange(1, 6)
        pipes = [rng.choice((1, 2, 5)) for _ in range(width)]
        records = []
        for _ in range(rng.randrange(0, 16)):
            source, target = rng.randrange(width), rng.randrange(width)
            distance = rng.randrange(0 if source < target else 1, 5)
            records.append((source, target, distance))
        yield periodic(pipes, records)
    for rotating in named_cases():
        item = periodic([1 + site % 3 for site in range(rotating["sites"])], [])
        item.update(families=rotating["families"], fragments=rotating["fragments"], from_storage=True)
        if rotating["sites"]:
            item["prefixes"] = sorted({0, 1, rotating["sites"] - 1, 7 * rotating["sites"] - 1})
        yield item
    item = periodic([1, 2], [(0, 1, 0), (1, 0, 1)])
    item.update(families=[["2", "1", 1]], fragments=[[0, 0, 0, "0", 2], [1, 0, 0, "0", 1]],
                from_storage=True)
    yield item


def huge_fixtures():
    """Huge distances need the same four/six vertices as their small analogues."""
    for distance in (2**63, 2**300 + 123, 10**500 + 1):
        item = periodic([1, 2], [(0, 1, distance), (0, 1, distance + 1)])
        item["huge_thresholds"] = [[[None, str(distance)], [None, None]], [["0", str(distance)], [None, "0"]]]
        item["huge_retained"] = [(0, 1, distance)]
        period = distance + 7
        item["frontier_queries"] = [[1, 0, str(period), str(2 * period + 2)],
                                     [1, 1, str(period), str(2 * period + 2)]]
        item["huge_frontiers"] = [["8", "0"], ["8", str(period + 1)]]
        item["queries"] += [[0, 1, "0", str(distance - 1)], [0, 1, "0", str(distance)],
                            [0, 1, "1", str(distance + 1)]]
        yield item
    distance = 2**200
    item = periodic([1, 2, 3], [(0, 1, distance), (1, 2, distance), (0, 2, 2 * distance + 1)])
    item["huge_thresholds"] = [[[None, str(distance), str(2 * distance)], [None, None, str(distance)],
                                [None, None, None]],
                               [["0", str(distance), str(2 * distance)], [None, "0", str(distance)], [None, None, "0"]]]
    item["huge_retained"] = [(0, 1, distance), (1, 2, distance)]
    yield item
    item = periodic([1, 2], [(0, 1, distance) for distance in range(1, 258)] + [(0, 1, 1)])
    item["huge_thresholds"] = [[[None, "1"], [None, None]], [["0", "1"], [None, "0"]]]
    item["huge_retained"] = [(0, 1, 1)]
    yield item


def invalid_fixtures():
    """Exercise production input validation after a nonempty successful result."""
    for record in ((2, 1, 1), (0, 2, 1), (0, 1, -1), (1, 0, 0), (0, 0, 0)):
        yield periodic([1, 2], [record], invalid=True)
    yield periodic([1, 2], [], invalid=True, null_site=True)
    yield periodic([], [(0, 0, 1)], invalid=True)


def mutations(instance, result):
    """Check loss/addition of covers, thresholds, native separation and provenance."""
    changes = []
    changed = copy.deepcopy(result)
    changed["retained"] = []
    changes.append(changed)
    changed = copy.deepcopy(result)
    changed["retained"] = list(range(len(result["records"])))
    changes.append(changed)
    changed = copy.deepcopy(result)
    changed["start"][0][0] = "0"
    changes.append(changed)
    changed = copy.deepcopy(result)
    changed["completion"][0][0] = "1"
    changes.append(changed)
    changed = copy.deepcopy(result)
    changed["records"][0][3] = []
    changes.append(changed)
    changed = copy.deepcopy(result)
    changed["queries"][0] = not result["queries"][0]
    changes.append(changed)
    for candidate in changes:
        try:
            check(instance, candidate)
        except ValueError:
            continue
        raise ValueError("oracle accepted deliberately corrupted quotient output")
    return len(changes)


def main():
    """Run the bounded driver and verify every finite event relation independently."""
    if len(sys.argv) != 3:
        raise ValueError("usage: check_periodic_analysis.py driver scratch.json")
    driver = shutil.which(sys.argv[1])
    if driver is None:
        raise ValueError("periodic driver not found")
    path = Path(sys.argv[2]).resolve()
    cases = list(fixtures()) + list(huge_fixtures()) + list(invalid_fixtures())
    try:
        path.write_text(json.dumps(cases), encoding="utf-8")
        output = subprocess.run([str(Path(driver).resolve()), str(path)], check=True, capture_output=True,
                                text=True, timeout=60)
        results = json.loads(output.stdout)
    finally:
        path.unlink(missing_ok=True)
    expect(len(results), len(cases), "batch count")
    for position, (instance, result) in enumerate(zip(cases, results)):
        try:
            check(instance, result)
        except ValueError as error:
            raise ValueError(f"case {position}: {instance}") from error
    count = mutations(cases[3], results[3])
    print(f"verified {len(cases)} periodic cases: event covers, thresholds, frontiers, windows, atomic failure")
    print(f"verified {count} periodic oracle mutations")


if __name__ == "__main__":
    main()
