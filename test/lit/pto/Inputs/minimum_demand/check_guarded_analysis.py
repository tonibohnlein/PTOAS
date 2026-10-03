# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independently unfold guarded potential sites and check generic graph covers.

Predicate evaluation here interprets the emitted shared circuit. Required graphs
come from all effect conflicts on selected sites, not production guarded closure.
"""
import copy
import itertools
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from check_demand_analysis import case, check_case as check_explicit, conflicts, expect, graph
from event_graph import Payload, finish, minimum_generators, start


def evaluate(result, values):
    """Evaluate a topologically shared Boolean circuit without rewriting it."""
    expect(len(values), result["atoms"], "atom valuation length")
    evaluated = []
    seen = set()
    for index, node in enumerate(result["nodes"]):
        kind, first, second = node["kind"], node["first"], node["second"]
        if kind >= 3 and not (0 <= first < index and 0 <= second < index):
            raise ValueError("predicate operand is not an earlier arena node")
        identity = (kind, node.get("atom"), first, second)
        if identity in seen:
            raise ValueError("predicate arena lost interning")
        seen.add(identity)
        if kind == 0:
            value = False
        elif kind == 1:
            value = True
        elif kind == 2:
            value = values[node["atom"]]
        elif kind == 3:
            value = not evaluated[first]
        elif kind == 4:
            value = evaluated[first] and evaluated[second]
        elif kind == 5:
            value = evaluated[first] or evaluated[second]
        else:
            raise ValueError("unknown predicate operation")
        evaluated.append(value)
    expect(evaluated[:2], [False, True], "constant predicates")
    return evaluated


def literal_guard(literals, values):
    """Interpret original branch literals, independently of the emitted DAG."""
    return all(literal if isinstance(literal, bool) else values[abs(literal) - 1] == (literal > 0)
               for literal in literals)


def selected_edges(edges, predicates):
    """Select only the demands whose emitted predicates hold."""
    pairs = [(edge["source"], edge["consumer"]) for edge in edges if predicates[edge["guard"]]]
    expect(len(pairs), len(set(pairs)), "merged guarded endpoints")
    return set(pairs)


def check_witnesses(instance, edges, predicates, selected, modeled, extra):
    """Every active storage witness names real effects in this execution."""
    for edge in edges:
        source, consumer = edge["source"], edge["consumer"]
        pair = (source, consumer)
        if not predicates[edge["guard"]]:
            if any(predicates[witness["guard"]] for witness in edge["witnesses"]):
                raise ValueError("witness escaped its demand guard")
            continue
        if source not in selected or consumer not in selected:
            raise ValueError("demand escaped endpoint occurrence")
        active = [witness for witness in edge["witnesses"] if predicates[witness["guard"]]]
        if pair in modeled and not active:
            raise ValueError("storage demand lost guarded witnesses")
        if not active and pair not in extra:
            raise ValueError("demand has no storage or supplied-prerequisite witness")
        identities = [(witness["kind"], witness["source"], witness["consumer"]) for witness in active]
        expect(len(identities), len(set(identities)), "distinct normalized storage witnesses")
        for witness in active:
            first, second, kind = witness["source"], witness["consumer"], witness["kind"]
            source_effect = instance["reads" if kind == 1 else "writes"][source]
            target_effect = instance["reads" if kind == 0 else "writes"][consumer]
            related = first == second or sorted((first, second)) in instance["aliases"]
            expect(first in source_effect and second in target_effect and related, True, "guarded witness")


def unfold(instance, values):
    """Construct one concrete explicit input by selecting original occurrences."""
    selected = [id_ for id_, guard in enumerate(instance["guards"]) if literal_guard(guard, values)]
    dense = {id_: index for index, id_ in enumerate(selected)}
    extra = [[dense[source], dense[consumer]] for source, consumer, guard in instance.get("guarded_extra", ())
             if source in dense and consumer in dense and literal_guard(guard, values)]
    projected = case([instance["pipes"][id_] for id_ in selected],
                     [instance["reads"][id_] for id_ in selected],
                     [instance["writes"][id_] for id_ in selected],
                     instance["footprints"], instance["aliases"], extra=extra)
    return selected, projected


def check_valuation(instance, result, values):
    """Compare every retained edge, readiness query and adjacency predicate."""
    predicates = evaluate(result, values)
    presence = [literal_guard(guard, values) for guard in instance["guards"]]
    expect([predicates[id_] for id_ in result["occurrences"]], presence, "occurrence guards")
    selected, projected = unfold(instance, values)
    dense = {id_: index for index, id_ in enumerate(selected)}
    modeled = conflicts(projected)
    extra = set(map(tuple, projected["extra"]))
    extra_original = {(selected[source], selected[consumer]) for source, consumer in extra}
    required = modeled | extra
    payloads = [Payload(str(pipe)) for pipe in projected["pipes"]]
    closure = graph(payloads, required).closure()
    required_original = {(selected[source], selected[consumer]) for source, consumer in required}
    modeled_original = {(selected[source], selected[consumer]) for source, consumer in modeled}
    expect(selected_edges(result["generators"], predicates), required_original, "guarded conflict graph")
    expected = {(selected[source], selected[consumer]) for source, consumer in minimum_generators(payloads, required)}
    retained = selected_edges(result["retained"], predicates)
    expect(retained, expected, "guarded canonical F*")
    reduced = {(dense[source], dense[consumer]) for source, consumer in retained}
    expect(graph(payloads, reduced).closure(), closure, "guarded reduced closure")
    for edges in (result["generators"], result["retained"]):
        check_witnesses(instance, edges, predicates, selected, modeled_original, extra_original)
    count = len(instance["pipes"])
    expect(len(result["readiness"]), count, "readiness row count")
    for source, row in enumerate(result["readiness"]):
        expect(len(row), count, "readiness column count")
        for consumer, guard in enumerate(row):
            expected = (source in dense and consumer in dense
                        and start(dense[consumer]) in closure[finish(dense[source])])
            expect(predicates[guard], expected, "completion-before-start query")
    actual_local = set()
    for local in result["local"]:
        edge = result["retained"][local["demand"]]
        if predicates[local["guard"]]:
            actual_local.add((edge["source"], edge["consumer"]))
    expected_local = {(source, consumer) for source, consumer in retained
                      if instance["pipes"][source] == instance["pipes"][consumer]
                      and any(presence[id_] and instance["pipes"][id_] == instance["pipes"][source]
                              for id_ in range(source + 1, consumer))}
    expect(actual_local, expected_local, "executed local adjacency")


def guarded(pipes, reads, writes, guards, atoms=1, footprints=1, aliases=(), **options):
    """Attach branch-context literals to an explicit potential occurrence set."""
    return case(pipes, reads, writes, footprints, aliases, guards=guards, atoms=atoms, **options)


def named_cases():
    """Small examples exercise bypass chains and branch-dependent covering paths."""
    return [
        guarded([], [], [], [], atoms=0, footprints=0),
        guarded([1, 1, 1], [[], [], []], [[], [], []], [[], [1], []], footprints=0),
        guarded([1, 2], [[], [0]], [[0], []], [[1], []]),
        guarded([1, 2], [[], [0]], [[0], []], [[], [1]]),
        guarded([1, 1, 2], [[], [], [0]], [[0], [0], []], [[1], [-1], []]),
        guarded([1, 2, 3], [[], [1], [0, 2]], [[0, 1], [2], []], [[], [1], []], footprints=3),
        guarded([1, 2, 2, 3], [[], [0], [1], [0, 2]], [[0], [1], [2], []],
                [[], [1], [-1], []], footprints=3),
        guarded([1, 1, 1], [[], [], [0]], [[0], [1], []], [[], [1], []], footprints=2),
        guarded([1, 2, 1], [[], [0], [0]], [[0], [], [0]], [[], [1], []]),
        guarded([1, 2, 3, 1], [[], [], [2], []], [[0], [1], [], [0]],
                [[], [1], [-1], []], footprints=3, aliases=[[0, 2], [1, 2]]),
        guarded([1, 2], [[], [0]], [[0], []], [[1, -1], []]),
        guarded([1, 2], [[], []], [[], []], [[], []], footprints=0,
                guarded_extra=[[0, 1, [1]], [0, 1, [-1]]]),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], aliases=[[0, 0]], invalid=True),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], guarded_extra=[[0, 2, []]], invalid=True),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], guarded_extra=[[1, 0, []]], invalid=True),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], invalid_predicate=True, invalid=True),
        guarded([0, 0, 0, 1], [[], [], [], [0, 1]], [[0], [], [1], []],
                [[], [1], [], []], footprints=2),
        guarded([1, 0, 0, 0], [[], [0], [], [0]], [[0], [], [], []], [[], [], [1], []]),
        guarded([1, 2, 0], [[0], [0], []], [[], [], [0]], [[1], [-1], []]),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], footprints=2, aliases=[[1, 0]], invalid=True),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], aliases=[[0, 1]], invalid=True),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], guarded_extra=[[0, 0, []]], invalid=True),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], invalid_count=True, invalid=True),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], duplicate_phase=True, invalid=True),
        guarded([1, 2], [[], [0]], [[0], []], [[], []], null_phase=True, invalid=True),
    ]


def batches():
    """Enumerate short effects and sample nested/shared guard contexts deterministically."""
    result = named_cases()
    for choices in itertools.product(range(8), repeat=3):
        pipes = [choice // 4 for choice in choices]
        reads = [[0] if choice % 4 & 1 else [] for choice in choices]
        writes = [[0] if choice % 4 & 2 else [] for choice in choices]
        result.append(guarded(pipes, reads, writes, [[], [1], []]))
    rng = random.Random(20260929)
    for _ in range(160):
        count, atoms, footprints = rng.randrange(1, 11), rng.randrange(1, 5), rng.randrange(1, 5)
        pipes = [rng.randrange(4) for _ in range(count)]
        reads = [[id_ for id_ in range(footprints) if rng.randrange(5) == 0] for _ in pipes]
        writes = [[id_ for id_ in range(footprints) if rng.randrange(5) == 0] for _ in pipes]
        guards = [[(atom + 1) * rng.choice((-1, 1)) for atom in range(atoms) if rng.randrange(3) == 0]
                  for _ in pipes]
        aliases = [[first, second] for first in range(footprints) for second in range(first + 1, footprints)
                   if rng.randrange(3) == 0]
        extra = [[source, consumer, [rng.choice((-1, 1))]] for consumer in range(count)
                 for source in range(consumer) if rng.randrange(20) == 0]
        result.append(guarded(pipes, reads, writes, guards, atoms, footprints, aliases, guarded_extra=extra))
    result.append(guarded([id_ % 3 for id_ in range(22)], [[0]] * 22, [[0]] * 22,
                          [[], *[[id_] for id_ in range(1, 21)], []], atoms=20))
    return result


def valuations(instance):
    """Exhaust small contexts; sample only the separate representation-size case."""
    count = instance["atoms"]
    if count <= 4:
        return itertools.product((False, True), repeat=count)
    return ([False] * count, [True] * count, [id_ % 2 == 0 for id_ in range(count)])


def check_case(instance, result):
    """Check failure atomicity or all selected concrete executions."""
    if instance.get("invalid", False):
        expect(result, {"valid": False, "empty": True}, "guarded failure atomicity")
        return 0
    expect(result["valid"], True, "valid guarded result")
    expect(result["pipes"], instance["pipes"], "borrowed pipe assignments")
    expect(len(result["occurrences"]), len(instance["pipes"]), "occurrence count")
    count = 0
    for values in valuations(instance):
        check_valuation(instance, result, values)
        count += 1
    return count


def mutations(instances, results):
    """Reject lost guards, incompatible paths, native serialization and lost witnesses."""
    changed = []
    relay = copy.deepcopy(results[5])
    relay["retained"] = [edge for edge in relay["retained"] if (edge["source"], edge["consumer"]) != (0, 2)]
    changed.append((5, relay))
    incompatible = copy.deepcopy(results[6])
    incompatible["retained"] = [edge for edge in incompatible["retained"]
                                if (edge["source"], edge["consumer"]) != (0, 3)]
    changed.append((6, incompatible))
    for case_id, field, value in [(2, "occurrences", [1, 1]), (7, "local", [])]:
        result = copy.deepcopy(results[case_id])
        result[field] = value
        changed.append((case_id, result))
    result = copy.deepcopy(results[1])
    result["readiness"][0][2] = 1
    changed.append((1, result))
    result = copy.deepcopy(results[2])
    result["retained"][0]["witnesses"] = []
    changed.append((2, result))
    for case_id, result in changed:
        try:
            check_case(instances[case_id], result)
        except ValueError:
            continue
        raise ValueError("guarded oracle accepted a mutation")
    print("verified {} guarded oracle mutations".format(len(changed)))


def compare_explicit(executable, path, instances, guarded_results):
    """Cross-check concrete selections against the existing production S/T route."""
    projected, origins = [], []
    for case_id, instance in enumerate(instances):
        if instance.get("invalid", False):
            continue
        for values in valuations(instance):
            selected, explicit = unfold(instance, values)
            projected.append(explicit)
            origins.append((case_id, selected, values))
    for begin in range(0, len(projected), 512):
        batch = projected[begin:begin + 512]
        path.write_text(json.dumps(batch), encoding="utf-8")
        completed = subprocess.run([executable, str(path)], check=True, capture_output=True, text=True, timeout=60)
        results = json.loads(completed.stdout)
        expect(len(results), len(batch), "explicit comparison batch length")
        for offset, (instance, result) in enumerate(zip(batch, results)):
            check_explicit(instance, result)
            case_id, selected, values = origins[begin + offset]
            actual = {(selected[source], selected[consumer]) for source, consumer in result["retained"]}
            expected = selected_edges(guarded_results[case_id]["retained"], evaluate(guarded_results[case_id], values))
            expect(actual, expected, "guarded versus explicit production reduction")
    print("verified {} independent concrete S/T comparisons".format(len(projected)))


def main():
    """Run bounded synthetic analysis; compiler fixtures test extraction separately."""
    if len(sys.argv) != 3:
        raise ValueError("usage: check_guarded_analysis.py pto-frontier-demand-test cases.json")
    executable = shutil.which(sys.argv[1])
    if executable is None:
        raise ValueError("guarded analysis executable not found")
    instances = batches()
    path = Path(sys.argv[2]).resolve()
    path.write_text(json.dumps(instances), encoding="utf-8")
    completed = subprocess.run([executable, str(path)], check=True, capture_output=True, text=True, timeout=60)
    results = json.loads(completed.stdout)
    expect(len(results), len(instances), "guarded batch length")
    visits = sum(check_case(instance, result) for instance, result in zip(instances, results))
    events = 2 * len(instances[-1]["pipes"])
    if len(results[-1]["nodes"]) > 6 * events ** 3 + 8 * events ** 2 + 128:
        raise ValueError("guarded reduction exceeded its polynomial gate bound")
    print("verified {} guarded cases and {} concrete valuations".format(len(instances), visits))
    print("verified 20-atom shared representation without branch-path enumeration")
    mutations(instances, results)
    compare_explicit(executable, path.with_suffix(".explicit.json"), instances, results)


if __name__ == "__main__":
    main()
