# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare production lifetime/rank analysis with unrestricted event reachability.

The reference enumerates every modeled conflict over pairwise aliases. It never
uses writer/reader frontiers or the production S/T recurrence. Synthetic batches
exercise the analysis boundary; compiler fixtures separately check translation.
"""

import copy
import itertools
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from event_graph import Graph, Payload, add_native, finish, minimum_generators, start


def case(pipes, reads, writes, footprints=1, aliases=(), **options):
    """Build one explicit test instance with supplied incidence identities."""
    return dict(pipes=pipes, reads=reads, writes=writes, footprints=footprints,
                aliases=list(aliases), **options)


def related(instance, first, second):
    """Aliases are reflexive pairs, never a transitive equivalence relation."""
    return first == second or sorted((first, second)) in instance["aliases"]


def conflicts(instance):
    """Enumerate RAW, WAR, WAW edges directly, without lifetime replacement."""
    edges = set()
    for consumer in range(len(instance["pipes"])):
        for source in range(consumer):
            earlier_reads = instance["reads"][source]
            earlier_writes = instance["writes"][source]
            later_reads = instance["reads"][consumer]
            later_writes = instance["writes"][consumer]
            pairs = itertools.chain(itertools.product(earlier_writes, later_reads + later_writes),
                                    itertools.product(earlier_reads, later_writes))
            if any(related(instance, first, second) for first, second in pairs):
                edges.add((source, consumer))
    return edges


def graph(payloads, edges):
    """Construct a start/completion graph using only the declared contracts."""
    result = Graph()
    add_native(result, payloads)
    for source, consumer in edges:
        result.add(finish(source), start(consumer))
    return result


def expect(actual, expected, label):
    """Report a precise mismatch without depending on Python assertion mode."""
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def check_witnesses(instance, result, closure):
    """Witnesses must identify actual effects and valid reader-mediated paths."""
    expect(len(result["witnesses"]), len(result["generators"]), "witness list length")
    for (source, consumer), witnesses in zip(result["generators"], result["witnesses"]):
        if not witnesses:
            raise ValueError("storage generator lost its witnesses")
        for witness in witnesses:
            first, second, kind = witness["source"], witness["consumer"], witness["kind"]
            source_effect = instance["reads" if kind == 1 else "writes"][source]
            target_effect = instance["reads" if kind == 0 else "writes"][consumer]
            expect(first in source_effect and second in target_effect and related(instance, first, second),
                   True, "storage witness")
            if "writer" in witness:
                writer = witness["writer"]
                expect(writer <= source and first in instance["writes"][writer], True, "previous writer")
    for writer, reader, consumer, first, second in result["bypasses"]:
        expect(writer < reader < consumer, True, "bypass order")
        expect(first in instance["writes"][writer] and first in instance["reads"][reader]
               and second in instance["writes"][consumer] and related(instance, first, second), True, "bypass effects")
        expect(start(reader) in closure[finish(writer)] and start(consumer) in closure[finish(reader)],
               True, "bypass reachability")


def check_case(instance, result):
    """Check closure, canonical covers, every rank coordinate and adjacency."""
    if instance.get("invalid", False):
        expect(result, {"valid": False, "empty": True}, "failure atomicity")
        return
    expect(result["valid"], True, "valid instance")
    payloads = [Payload(str(pipe)) for pipe in instance["pipes"]]
    modeled = conflicts(instance)
    storage_closure = graph(payloads, modeled).closure()
    generated = set(map(tuple, result["generators"]))
    expect(graph(payloads, generated).closure(), storage_closure, "lifetime closure")
    check_witnesses(instance, result, storage_closure)
    used = set(map(tuple, result["used"]))
    required = modeled | set(map(tuple, instance.get("extra", ())))
    closure = graph(payloads, required).closure()
    expect(graph(payloads, used).closure(), closure, "supplied generator closure")
    retained = set(map(tuple, result["retained"]))
    expect(len(result["retained"]), len(retained), "unique retained endpoints")
    incoming = [(instance["pipes"][source], consumer) for source, consumer in retained]
    expect(len(incoming), len(set(incoming)), "one retained source per incoming pipe")
    expect(retained, minimum_generators(payloads, required), "canonical F*")
    expect(graph(payloads, retained).closure(), closure, "reduced closure")
    pipes = list(dict.fromkeys(instance["pipes"]))
    expect(result["pipes"], pipes, "pipe columns")
    counts = dict.fromkeys(pipes, 0)
    ranks = []
    for pipe in instance["pipes"]:
        counts[pipe] += 1
        ranks.append(counts[pipe])
    expect(result["ranks"], ranks, "pipe ranks")
    for consumer, summary in enumerate(result["summaries"]):
        expected_s, expected_t = [], []
        for pipe in pipes:
            sources = [source for source, value in enumerate(instance["pipes"]) if value == pipe]
            expected_s.append(max((ranks[source] for source in sources
                                   if start(consumer) in closure[finish(source)]), default=0))
            expected_t.append(max((ranks[source] for source in sources
                                   if source == consumer or finish(consumer) in closure[finish(source)]), default=0))
        expect(summary, {"S": expected_s, "T": expected_t}, f"summary at {consumer}")
    nonadjacent = {(source, consumer) for source, consumer in retained
                   if instance["pipes"][source] == instance["pipes"][consumer]
                   and ranks[source] != ranks[consumer] - 1}
    expect(set(map(tuple, result["nonadjacent"])), nonadjacent, "post-reduction adjacency")


def named_cases():
    """Boundary cases isolate the draft's important lifetime/native distinctions."""
    return [
        case([], [], [], footprints=0),
        case([1, 1, 1], [[], [], []], [[], [], []], footprints=0),
        case([1, 2], [[], [0]], [[0], []]),
        case([1, 2], [[0], []], [[], [0]]),
        case([1, 2], [[], []], [[0], [0]]),
        case([1, 2, 3], [[], [0], []], [[0], [], [0]]),
        case([1, 2, 3], [[], [0], [0]], [[0], [], [0]]),
        case([1, 2, 2, 3], [[], [0], [0], []], [[0], [], [], [0]]),
        case([1, 2, 3], [[], [0], [0, 1]], [[0], [1], []], footprints=2),
        case([1, 2], [[], [0, 1]], [[0, 1], []], footprints=2),
        case([1, 2, 3, 1], [[], [], [2], []], [[0], [1], [], [0]],
             footprints=3, aliases=[[0, 2], [1, 2]]),
        case([1, 1, 1], [[], [], [0]], [[0], [1], []], footprints=2),
        case([1, 1, 2, 1], [[], [], [0], [0]], [[0], [1], [], []], footprints=2, extra=[[2, 3]]),
        case([1, 1], [[], [0]], [[0], []]),
        case([1, 2, 1, 2, 1, 2, 1, 2], [[], [0], [], [1], [], [0], [], [1]],
             [[0], [], [1], [], [0], [], [1], []], footprints=2),
        case([1, 2, 3], [[], [1], [0]], [[0], [], [1]], footprints=2, aliases=[[0, 1]]),
        case([1, 2], [[], [0]], [[0], []], generators=[[0, 0]], invalid=True),
        case([1, 2], [[], [0]], [[0], []], generators=[[1, 0]], invalid=True),
        case([1, 2], [[], [0]], [[0], []], aliases=[[0, 0]], invalid=True),
        case([1, 2], [[], [0]], [[0], []], footprints=2, aliases=[[1, 0]], invalid=True),
        case([1, 2], [[], [0]], [[0], []], generators=[[0, 2]], invalid=True),
        case([1, 2], [[], [0]], [[0], []], aliases=[[0, 1]], invalid=True),
        case([1, 2, 1], [[], [], [0]], [[0], [1], []], footprints=2, aliases=[[0, 1]]),
    ]


def batches():
    """Exhaust short exact traces and sample pairwise conservative alias graphs."""
    result = named_cases()
    for choices in itertools.product(range(12), repeat=3):
        pipes = [choice // 4 for choice in choices]
        reads = [[0] if choice % 4 & 1 else [] for choice in choices]
        writes = [[0] if choice % 4 & 2 else [] for choice in choices]
        result.append(case(pipes, reads, writes))
    rng = random.Random(20260929)
    for _ in range(800):
        count, footprints = rng.randrange(1, 13), rng.randrange(1, 6)
        pipes = [rng.randrange(4) for _ in range(count)]
        reads = [[id_ for id_ in range(footprints) if rng.randrange(5) == 0] for _ in pipes]
        writes = [[id_ for id_ in range(footprints) if rng.randrange(5) == 0] for _ in pipes]
        aliases = [[first, second] for first in range(footprints) for second in range(first + 1, footprints)
                   if rng.randrange(3) == 0]
        extra = [[source, consumer] for consumer in range(count) for source in range(consumer)
                 if rng.randrange(16) == 0]
        instance = case(pipes, reads, writes, footprints, aliases, extra=extra)
        result.append(instance)
        equivalent = dict(instance)
        supplied = [list(edge) for edge in conflicts(instance)]
        rng.shuffle(supplied)
        supplied += supplied[:3]  # Duplicate endpoints and arbitrary input order.
        equivalent["generators"] = supplied
        result.append(equivalent)
    return result


def check_mutations(instances, results):
    """The oracle must reject missing requirements, false covers and wrong ranks."""
    false_s = copy.deepcopy(results[1]["summaries"])
    false_s[1]["S"] = [1]
    mutations = [
        (1, "summaries", false_s),
        (2, "generators", []),
        (2, "retained", []),
        (11, "nonadjacent", []),
        (10, "generators", results[10]["generators"] + [[0, 1]]),
        (9, "witnesses", [[]]),
    ]
    for position, field, value in mutations:
        changed = copy.deepcopy(results[position])
        changed[field] = value
        try:
            check_case(instances[position], changed)
        except ValueError:
            continue
        raise ValueError(f"oracle accepted mutation of {field} in case {position}")
    print(f"verified {len(mutations)} oracle mutations")


def main():
    """Run one bounded production batch and check its output independently."""
    if len(sys.argv) != 3:
        raise ValueError("usage: check_demand_analysis.py pto-frontier-demand-test cases.json")
    executable = shutil.which(sys.argv[1])
    if executable is None:
        raise ValueError("demand analysis test executable not found")
    path = Path(sys.argv[2]).resolve()
    instances = batches()
    path.write_text(json.dumps(instances), encoding="utf-8")
    completed = subprocess.run([executable, str(path)], check=True, capture_output=True, text=True, timeout=60)
    results = json.loads(completed.stdout)
    expect(len(results), len(instances), "batch length")
    for position, (instance, result) in enumerate(zip(instances, results)):
        try:
            check_case(instance, result)
        except ValueError as error:
            raise ValueError(f"case {position}: {instance!r}: {error}") from error
    # Explicit checks prevent named examples quietly losing their intended role.
    expect(results[1]["summaries"], [{"S": [0], "T": [1]}, {"S": [0], "T": [2]},
                                      {"S": [0], "T": [3]}], "native C->C is not C->I")
    expect(results[5]["generators"], [[0, 1], [1, 2]], "reader-mediated WAW")
    expect(len(results[9]["witnesses"][0]), 2, "merged cross-footprint witnesses")
    expect(results[11]["nonadjacent"], [[0, 2]], "genuine nonadjacent local cover")
    expect(results[12]["nonadjacent"], [], "redundant local demand removed before adjacency")
    expect(results[22]["generators"], [[0, 1], [0, 2], [1, 2]], "partial alias preserves prior writer demand")
    print(f"verified {len(instances)} explicit cases: lifetime closure, witnesses, S/T, canonical F*, adjacency")
    check_mutations(instances, results)


if __name__ == "__main__":
    main()
