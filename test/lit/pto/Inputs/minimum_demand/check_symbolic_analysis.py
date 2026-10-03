# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exact symbolic relations checked against independently enumerated physical event DAGs."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from check_rotating_analysis import case, closure, expect


def row(count, terms=(), constant=0):
    """An integer constraint; JSON decimal strings preserve arbitrary precision."""
    result = [0] * (count + 1)
    for column, coefficient in terms:
        result[column] += coefficient
    result[-1] = constant
    return list(map(str, result))


def piece(equalities=(), inequalities=(), locals_count=0):
    """Raw integer polyhedron; locals are independently quantified per disjunct."""
    return dict(eq=list(equalities), ge=list(inequalities), locals=locals_count)


def event(site, coordinates=(), kind=0):
    return dict(site=site, coords=list(map(str, coordinates)), kind=kind)


def reference(depth, parameters):
    """Uniform lexicographic reference order: coordinates, then static site."""
    width = depth + 1
    axes = list(range(1, width)) + [0]
    equalities, result = [], []
    for axis in axes:
        result.append(piece(equalities, [row(2 * width + parameters, [(width + axis, 1), (axis, -1)], -1)]))
        equalities = equalities + [row(2 * width + parameters, [(width + axis, 1), (axis, -1)])]
    return result


def fixture(name, pipes, depth, modes, parameters, *, triangular=False, parity=False, signed=False,
            shift=0, stride=1, offset=0, extra=False, smaller=False):
    """Relations supplied to C++; physical effects below use separate direct Python formulas."""
    width, symbols = depth + 1, len(parameters[0])
    present, reads, writes = [], [], []
    for site, mode in enumerate(modes):
        count = width + symbols + int(parity)
        equalities = [row(count, [(0, 1)], -site)]
        inequalities = []
        if depth:
            inequalities += [row(count, [(1, 1)], -shift),
                             row(count, [(width, 1), (1, -1)], shift - 1)]
            if signed:
                inequalities[0] = row(count, [(1, 1)], 1)
            if triangular:
                inequalities += [row(count, [(2, 1)]), row(count, [(1, 1), (2, -1)], -shift)]
            if parity:
                equalities += [row(count, [(1, 1), (count - 1, -2)], -3 * site - shift)]
        present.append(piece(equalities, inequalities, int(parity)))
        effect_count = width + 1 + symbols
        effect = [row(effect_count, [(0, 1)], -site),
                  row(effect_count, [(width, 1)] + ([(1, -stride)] if depth else []), -offset)]
        if mode & 1:
            reads.append(piece(effect))
        if mode & 2:
            writes.append(piece(effect))
    extras = []
    if extra:
        count = 2 * width + symbols
        # Context flag 0 has 0->1, context flag 1 has 1->2, never both.
        for source, target, flag in [(0, 1, 0), (1, 2, 1)]:
            extras.append(piece([row(count, [(0, 1)], -source), row(count, [(width, 1)], -target),
                                 row(count, [(2 * width + symbols - 1, 1)], -flag)]))
    context = [piece([], [row(symbols, [(0, 1)])])] if symbols else [piece()]
    instance = case(len(pipes), [], [], pipes=pipes, symbolic=True, name=name, depths=[depth] * len(pipes),
                    parameter_types=["i4096"] * symbols, cell_axes=1, context=context, present=present,
                    reference=reference(depth, symbols), reads=reads, writes=writes, extras=extras, contexts=[])
    if smaller:
        instance["generators"] = []
    expected = []
    for binding in parameters:
        admitted = not binding or binding[0] >= 0
        count = binding[0] if depth else 1
        occurrences = []
        if admitted:
            for first in range(-1 if signed else 0, count):
                for second in range(first + 1) if triangular else [0]:
                    for site, pipe in enumerate(pipes):
                        coordinates = (first + shift, second) if triangular else (first + shift,) if depth else ()
                        if parity and first % 2 != site:
                            continue
                        cell = stride * (first + shift if depth else 0) + offset
                        effects = {} if not modes[site] else {cell: modes[site]}
                        occurrences.append((site, coordinates, pipe, effects))
        # Static depth-zero body occurs once, irrespective of context values.
        if not depth and admitted:
            occurrences = [(site, (), pipe, {} if not modes[site] else {offset: modes[site]})
                           for site, pipe in enumerate(pipes)]
        points = [event(site, coordinates, kind) for site, coordinates, _, _ in occurrences for kind in (0, 1)]
        points += [event(len(pipes)), event(0, (shift + 100,) if depth else (), 2)]
        if depth:
            points.append(event(0, (shift + 100, 0) if triangular else (shift + 100,)))
        instance["contexts"].append(dict(bindings=list(map(str, binding)), events=points))
        expected.append(physical(occurrences, points, admitted, binding[-1] if extra else None))
    return instance, expected


def physical(occurrences, points, admitted, flag):
    """All physical conflicts plus distinct native I/I,C/C chains; no relation operations."""
    count = len(occurrences)
    edges, native, previous = set(), set(), {}
    for target, (_, _, pipe, effects) in enumerate(occurrences):
        native.add((2 * target, 2 * target + 1))
        if pipe in previous:
            source = previous[pipe]
            native.update(((2 * source, 2 * target), (2 * source + 1, 2 * target + 1)))
        previous[pipe] = target
        for source, (_, _, _, before) in enumerate(occurrences[:target]):
            if any((mode | before[cell]) & 2 for cell, mode in effects.items() if cell in before):
                edges.add((2 * source + 1, 2 * target))
    if flag is not None and count:
        edges.add((2 * flag + 1, 2 * (flag + 1)))
    reach = [{b for b in range(2 * count) if bits & (1 << b)} for bits in closure(2 * count, native | edges)]
    native_reach = [{b for b in range(2 * count) if bits & (1 << b)} for bits in closure(2 * count, native)]
    covers = {(a, b) for a, b in edges if b not in native_reach[a] and
              not any(z != a and z != b and z in reach[a] and b in reach[z] for z in range(2 * count))}
    presence = ([True] * (2 * count) + [None, None] + [False] * (len(points) - 2 * count - 2)
                if admitted else [None] * len(points))
    matrix, minimum = [], []
    for a in range(len(points)):
        matrix.append([((a == b or b in reach[a]) if a < 2 * count and b < 2 * count else None)
                       for b in range(len(points))])
        minimum.append([((a, b) in covers if a < 2 * count and b < 2 * count else None)
                        for b in range(len(points))])
    return dict(admitted=admitted, presence=presence, reach=matrix, minimum=minimum,
                generators=[[(a, b) in edges for b in range(len(points))] for a in range(len(points))])


def corpus():
    """Small quantified relations, many contexts, no production expression interpreter."""
    result = [fixture("sole-demand-strictness", [0, 1], 0, [2, 1], [()]),
              fixture("readonly", [0, 1], 1, [1, 1], [(0,), (1,), (2,)]),
              fixture("rmw", [0, 1], 1, [3, 3], [(0,), (1,), (2,)]),
              fixture("raw-war-waw", [0, 1], 1, [1, 2], [(1,), (2,), (3,)]),
              fixture("triangular", [0, 1], 2, [2, 1], [(0,), (1,)], triangular=True),
              fixture("parity-fresh-existentials", [0, 1], 1, [2, 3], [(2,)], parity=True, stride=0),
              fixture("signed-coordinates", [0, 1], 1, [2, 1], [(0,), (1,), (2,), (-1,)], signed=True),
              fixture("shared-context", [0, 1, 2], 0, [0, 0, 0], [(1, 0), (1, 1)], extra=True),
              fixture("smaller-generator-extras", [0, 1, 2], 0, [0, 0, 0], [(1, 0), (1, 1)], extra=True, smaller=True),
              fixture("huge-signed-affine", [0, 1], 1, [3, 1], [(1,), (2,)], shift=10**40,
                      stride=-(10**35 + 1), offset=10**80),
              fixture("no-effects-native", [0, 0], 1, [0, 0], [(0,), (1,), (3,)])]
    # i1 is Boolean 0/1, never signed -1; unsigned large parameters also retain meaning.
    boolean = fixture("boolean-context", [0, 1, 2], 0, [0, 0, 0], [(1, 0), (1, 1)], extra=True)
    boolean[0]["parameter_types"][1] = "i1"
    boolean[0]["uniform"] = True
    result.append(boolean)
    unsigned = fixture("unsigned-context", [0], 0, [0], [(2**127 + 1,)])
    unsigned[0]["parameter_types"] = ["u128"]
    result.append(unsigned)
    result.append(fixture("triangular-one-pipe", [0], 2, [3], [(2,)], triangular=True))
    complex_case = copy.deepcopy(result[4][0])
    complex_case["name"] = "complex-construction-only"
    complex_case["contexts"] = []
    result.append((complex_case, []))
    zero_cell = fixture("zero-axis-cell-role", [0], 0, [0], [()])
    zero_cell[0]["cell_axes"] = 0
    result.append(zero_cell)
    result[0][0]["uniform"] = True
    result[7][0]["uniform"] = True
    return result


def verify(instance, expected, actual):
    expect(actual["valid"], True, "build")
    expect(actual["probes"], True, "ownership/types/lazy construction")
    expect(actual["nodes"] - actual["prepared"], len(set(instance["pipes"])) + 6, "shared DAG growth")
    expect(len(actual["contexts"]), len(expected), "contexts")
    for index, (want, got) in enumerate(zip(expected, actual["contexts"])):
        expect(got["probes"], True, "binding/artifact ownership")
        if instance.get("uniform"):
            expect(got["uniform_reach"], want["reach"], "uniform same-context reachability")
            expect(got["uniform_minimum"], want["minimum"], "uniform same-context minimum")
        for key, value in want.items():
            if key == "generators":
                continue
            expect(got[key], value, f'{instance["name"]}/{index}/{key}')


def mutations(instance, expected, actual):
    """Catch lost strictness, lost identity, false native serial order, and ignored failure."""
    bad = []
    for key, a, b, value in [("minimum", 1, 2, False), ("reach", 0, 0, False),
                              ("reach", 1, 0, True), ("reach", 4, 0, False)]:
        changed = copy.deepcopy(actual)
        changed["contexts"][0][key][a][b] = value
        bad.append(changed)
    changed = copy.deepcopy(actual)
    changed["nodes"] += 1
    bad.append(changed)
    changed = copy.deepcopy(actual)
    changed["probes"] = False
    bad.append(changed)
    for changed in bad:
        try:
            verify(instance, expected, changed)
        except ValueError:
            continue
        raise ValueError("symbolic oracle accepted a mutation")
    return len(bad)


def main():
    executable = shutil.which(sys.argv[1]) or sys.argv[1]
    path = Path(sys.argv[2])
    cases = corpus()
    path.write_text(json.dumps([instance for instance, _ in cases]), encoding="utf-8")
    actual = []
    for instance, _ in cases:
        path.write_text(json.dumps([instance]), encoding="utf-8")
        try:
            process = subprocess.run([executable, str(path)], check=True, text=True, capture_output=True, timeout=20)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print(f"symbolic schema failed: {instance['name']}", file=sys.stderr)
            for label, captured in (("stdout", error.stdout), ("stderr", error.stderr)):
                if isinstance(captured, bytes):
                    captured = captured.decode("utf-8", errors="replace")
                print(f"captured {label}:\n{captured or '<empty>'}", file=sys.stderr)
            raise
        actual.extend(json.loads(process.stdout))
    path.write_text(json.dumps([instance for instance, _ in cases]), encoding="utf-8")
    expect(len(actual), len(cases), "case count")
    for (instance, expected), got in zip(cases, actual):
        verify(instance, expected, got)
    count = mutations(*cases[0], actual[0])
    contexts = sum(len(want) for _, want in cases)
    print(f"verified {len(cases)} symbolic schemas / {contexts} contexts: "
          "exact physical event DAGs and typed lazy relations")
    print(f"verified {count} symbolic oracle mutations")


if __name__ == "__main__":
    main()
