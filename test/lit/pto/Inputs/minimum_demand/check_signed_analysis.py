# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Uniform signed arithmetic: physical DAGs plus bounded direct integer evaluation."""

import copy
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from check_rotating_analysis import case, expect
from check_symbolic_analysis import event, fixture, physical, piece, row



def select_points(instance, expected, indices):
    """Select queries without removing any physical event from the independent DAG."""
    original = copy.deepcopy(instance["contexts"])
    for context in instance["contexts"]:
        context["events"] = [context["events"][index] for index in indices]
    return instance, expected, original

def prepare(item, indices=None, pipes=None):
    instance, expected = item
    instance["affine"] = True
    instance["selector_pipes"] = sorted(set(instance["pipes"])) if pipes is None else pipes
    original = copy.deepcopy(instance["contexts"])
    if indices is not None:
        return select_points(instance, expected, indices)
    return instance, expected, original

def tail(name, bound, *, reflected=False, free=False, unbounded=False, huge=False):
    """Variable writer site followed by one static reader; fractional visible optimum."""
    sign, scale = (-1 if reflected else 1), (1 if huge else 2)
    present = [piece([row(3, [(0, 1)])], [row(3, [(1, sign)])] +
                     ([] if unbounded else [row(3, [(2, 1), (1, -scale * sign)])])),
               piece([row(3, [(0, 1)], -1), row(3, [(1, 1)])])]
    # Reference order compares site FIRST, then signed traversal coordinates.
    order = [piece([], [row(5, [(2, 1), (0, -1)], -1)]),
             piece([row(5, [(2, 1), (0, -1)])], [row(5, [(3, sign), (1, -sign)], -1)])]
    source = [piece([row(4, [(0, 1)]), row(4, [(2, 1)])])]
    target = [piece([row(4, [(0, 1)], -1), row(4, [(2, 1)])])]
    instance = case(2, [], [], pipes=[0, 0] if huge else [0, 1], symbolic=True, affine=True, name=name,
                    depths=[1, 0], parameter_types=["i4096"], cell_axes=1, context=[piece()], present=present,
                    reference=order, reads=target, writes=source, extras=[], selector_pipes=[0],
                    orders=[[[0, sign]], None], contexts=[])
    if free:
        for key in ("context", "present", "reference", "reads", "writes"):
            for part in instance[key]:
                part["locals"] += 1
                for constraint in part["eq"] + part["ge"]:
                    constraint.insert(-1, "0")
    if huge or unbounded:
        instance["all_pairs"] = False
        instance["contexts"] = [dict(bindings=[str(bound)], events=[event(1)])]
        return instance, [], []
    occurrences = [(0, (sign * i,), 0, {0: 2}) for i in range(bound // scale + 1)]
    occurrences += [(1, (), instance["pipes"][1], {0: 1})]
    points = [event(site, coords, kind) for site, coords, _, _ in occurrences for kind in (0, 1)]
    points += [event(2), event(0, (), 2), event(0, (100,))]
    instance["contexts"] = [dict(bindings=[str(bound)], events=points)]
    return instance, [physical(occurrences, points, True, None)], copy.deepcopy(instance["contexts"])

def valid(instance, point):
    return (point["site"] < instance["sites"] and point["kind"] in (0, 1) and
            len(point["coords"]) == instance["depths"][point["site"]])

def outcome(instance, points, presence, a, b=None, value=None):
    if not valid(instance, points[a]) or (b is not None and not valid(instance, points[b])):
        return ["invalid-event", None]
    if b is not None and (not presence[a] or not presence[b]):
        return ["absent-endpoint", None]
    return ["success", value]

def verify_graph(instance, expected, originals, actual):
    expect(actual["valid"], True, "analysis")
    expect(len(actual["contexts"]), len(instance["contexts"]), "context count")
    for context_index, (query, got) in enumerate(zip(instance["contexts"], actual["contexts"])):
        name = instance["name"]
        if name == "invalid-certificate":
            expect(got["status"], "invalid-input", name)
            continue
        if name == "unbounded-visible":
            expect(got["incoming"][0][0][:2], ["unbounded-endpoint", None], name)
            continue
        if name == "huge-bound":
            partner = event(0, (10**40,), 1)
            result = got["incoming"][0][0]
            expect(result[:2], ["success", partner], name)
            if result[2] > 200000 or result[3] > 10000:
                raise ValueError(f"huge bound caused excessive integer work: {result[2:]}")
            continue
        want = expected[context_index]
        expect(got["status"], "success" if want["admitted"] else "inadmissible-context", name)
        if not want["admitted"]:
            continue
        expect(got["probes"], True, "immutable bindings / negatives")
        original = originals[context_index]["events"]
        ids = [original.index(point) for point in query["events"]]
        for local, index in enumerate(ids):
            point = original[index]
            expect(got["presence"][local], outcome(instance, original, want["presence"], index,
                                                   value=want["presence"][index]), f"{name}/presence")
            if instance.get("all_pairs", True):
                for other, target in enumerate(ids):
                    for key in ("reach", "minimum"):
                        correct = outcome(instance, original, want["presence"], index, target,
                                          want[key][index][target])
                        expect(got[key][local][other], correct, f"{name}/{key}/{index},{target}")
            for direction in ("incoming", "outgoing"):
                for pipe_index, pipe in enumerate(instance["selector_pipes"]):
                    if not valid(instance, point) or point["kind"] != (0 if direction == "incoming" else 1):
                        correct = ["invalid-event", None]
                    elif not want["presence"][index]:
                        correct = ["absent-endpoint", None]
                    else:
                        matches, candidates = [], []
                        for candidate, candidate_point in enumerate(original):
                            if (not want["presence"][candidate] or not valid(instance, candidate_point) or
                                    instance["pipes"][candidate_point["site"]] != pipe or
                                    candidate_point["kind"] != (1 if direction == "incoming" else 0)):
                                continue
                            a, b = (candidate, index) if direction == "incoming" else (index, candidate)
                            if want["minimum"][a][b]:
                                matches.append(candidate_point)
                            if want["generators"][a][b]:
                                candidates.append(candidate_point)
                        if len(matches) > 1:
                            raise ValueError("independent cover violates pipe injection")
                        unsupported = any(instance["depths"][candidate["site"]] and
                                          instance.get("orders", [True] * instance["sites"])[candidate["site"]] is None
                                          for candidate in candidates)
                        correct = (["unsupported-order", None] if unsupported else
                                   ["success", matches[0] if matches else None])
                    expect(got[direction][local][pipe_index][:2], correct, f"{name}/{direction}/{index}/{pipe}")

def atom(terms, bound):
    return dict(terms=terms, bound=str(bound))


def signed_piece(domain=(), target=(), residues=(), atoms=(), locals_count=0):
    return dict(domain=list(domain), range=list(target), residues=list(map(str, residues)),
                atoms=list(atoms), locals=locals_count)


def split(instance, entries, domain, target, period):
    """Test transport lowering of explicit affine fixtures, never used by expected DAGs."""
    depth = max(instance["depths"], default=0)
    parameters = len(instance["parameter_types"])
    roles = dict(unit=0, occurrence=depth + 1, cell=instance["cell_axes"])
    tags = lambda role: range(instance["sites"]) if role == "occurrence" else [None]
    output = []
    for a, b in itertools.product(tags(domain), tags(target)):
        da = instance["depths"][a] if a is not None else roles[domain]
        db = instance["depths"][b] if b is not None else roles[target]
        for residues in itertools.product(range(period), repeat=da + db + parameters):
            for entry in entries:
                mapping, offset = [], 0
                for role, tag, active, prefix in ((domain, a, da, "d"), (target, b, db, "r")):
                    if tag is not None:
                        mapping.append((None, 0, tag))
                    for index in range(active):
                        mapping.append(((prefix, index), period, residues[offset + index]))
                    if tag is not None:
                        mapping.extend([(None, 0, 0)] * (depth - active))
                    offset += active
                mapping += [(("p", i), period, residues[da + db + i]) for i in range(parameters)]
                mapping += [(("l", i), 1, 0) for i in range(entry["locals"])]
                atoms, possible = [], True
                for equality, rows in ((True, entry["eq"]), (False, entry["ge"])):
                    for constraint in rows:
                        for sign in (1, -1) if equality else (-1,):
                            coefficients, constant = {}, -sign * int(constraint[-1])
                            for value, (axis, scale, residue) in zip(constraint[:-1], mapping):
                                value = sign * int(value)
                                constant -= value * residue
                                if axis is not None and value * scale:
                                    coefficients[axis] = coefficients.get(axis, 0) + value * scale
                            if not coefficients:
                                possible &= constant >= 0
                                continue
                            divisor = math.gcd(*coefficients.values())
                            terms = []
                            for (role, index), coefficient in coefficients.items():
                                coefficient //= divisor
                                terms += [[role, index, 1 if coefficient > 0 else -1]] * abs(coefficient)
                            if len(terms) > 2:
                                raise ValueError("fixture outside supplied signed class")
                            atoms.append(atom(terms, constant // divisor))
                if possible:
                    output.append(signed_piece(() if a is None else (a,), () if b is None else (b,),
                                               residues, atoms, entry["locals"]))
    return output


def convert(item, period=1):
    instance, expected, originals = prepare(item)
    instance.pop("affine")
    instance.update(signed=True, period=str(period))
    for name, domain, target in (("context", "unit", "unit"), ("present", "unit", "occurrence"),
                                 ("reference", "occurrence", "occurrence"), ("reads", "occurrence", "cell"),
                                 ("writes", "occurrence", "cell"), ("extras", "occurrence", "occurrence")):
        instance[name] = split(instance, instance[name], domain, target, period)
    return instance, expected, originals


def reflected():
    instance, _ = fixture("reflected", [0, 1], 1, [2, 1], [(0,), (2,), (3,)])
    # Writer cell=2-i and reader cell=i; constant2 preserves octagonal syntax.
    instance["writes"] = [piece([row(4, [(0, 1)]), row(4, [(2, 1), (1, 1)], -2)])]
    expected = []
    for context in instance["contexts"]:
        n = int(context["bindings"][0])
        occurrences = [(site, (i,), site, {2-i if site == 0 else i: 2 if site == 0 else 1})
                       for i in range(n) for site in (0, 1)]
        expected.append(physical(occurrences, context["events"], True, None))
    return instance, expected


def varying_depth():
    instance, _, _ = tail("varying-depth", 0, huge=True)
    instance["pipes"] = [0, 1]
    instance["all_pairs"] = True
    instance["contexts"] = []
    instance["context"] = [piece([], [row(1, [(0, 1)])])]
    expected = []
    for n in (0, 2, 4):
        occurrences = [(0, (i,), 0, {0: 2}) for i in range(n+1)] + [(1, (), 1, {0: 1})]
        points = [event(site, coords, kind) for site, coords, _, _ in occurrences for kind in (0, 1)]
        points += [event(2), event(0, (), 2), event(0, (100,))]
        instance["contexts"].append(dict(bindings=[str(n)], events=points))
        expected.append(physical(occurrences, points, True, None))
    return instance, expected


def graph_cases():
    result = [convert(fixture("sole", [0, 1], 0, [2, 1], [()])),
              convert(fixture("triangle", [0, 1], 2, [2, 1], [(0,), (1,), (2,), (3,)], triangular=True)),
              convert(reflected()),
              convert(fixture("residue-negative", [0, 1], 1, [3, 1], [(0,), (2,), (3,)], signed=True, stride=0), 2),
              convert(fixture("conditional-parity", [0, 1], 1, [2, 3], [(0,), (2,), (3,), (-1,)],
                              parity=True, stride=0), 2),
              convert(fixture("interleaved", [0, 1, 0], 1, [2, 1, 2], [(1,), (2,)], stride=0)),
              convert(fixture("readonly", [0, 1], 1, [1, 1], [(0,), (2,)]))]
    result.append(convert(fixture("extra-contexts", [0, 1, 2], 0, [0, 0, 0],
                                  [(0,), (1,)], extra=True, smaller=True)))
    result.append(convert(varying_depth()))
    result[0][0]["interchange"] = True
    free = convert(fixture("free-auxiliary", [0, 1], 0, [2, 1], [()]))
    for field in ("context", "present", "reference", "reads", "writes", "extras"):
        for part in free[0][field]:
            part["locals"] += 1
    result.append(free)
    return result


def base(name, period=1, **kwargs):
    return case(2, [], [], pipes=[0, 1], symbolic=True, signed=True, arithmetic=True,
                depths=[1, 0], parameter_types=["i4096"], cell_axes=1, contexts=[], name=name,
                period=str(period), **kwargs)


def bounded(residues=(0, 0, 0), extra=(), locals_count=0):
    axes = [("d", 0), ("r", 0), ("p", 0)] + [("l", i) for i in range(locals_count)]
    atoms = [atom([[role, i, sign]], 2) for role, i in axes for sign in (-1, 1)] + list(extra)
    return signed_piece(residues=residues, atoms=atoms, locals_count=locals_count)


def member(parts, source, target, parameters, period):
    """Direct integer evaluation with bounded existential witnesses; no elimination algorithm."""
    values = source["coords"] + target["coords"] + list(parameters)
    residues, quotient = [int(x) % period for x in values], [int(x) // period for x in values]
    for part in parts:
        if (part["domain"] != source["tag"] or part["range"] != target["tag"] or
                residues != list(map(int, part["residues"]))):
            continue
        for locals_value in itertools.product(range(-2, 3), repeat=part["locals"]):
            axes = {"d": quotient[:len(source["coords"])],
                    "r": quotient[len(source["coords"]):len(source["coords"]) + len(target["coords"])],
                    "p": quotient[len(source["coords"]) + len(target["coords"]):], "l": locals_value}
            if all(sum(sign * axes[role][index] for role, index, sign in constraint["terms"]) <=
                   int(constraint["bound"])
                   for constraint in part["atoms"]):
                return True
    return False


def point(coordinates=(), tag=()):
    return dict(coords=list(map(str, coordinates)), tag=list(tag))


def arithmetic_cases():
    rng = random.Random(1926)
    cases = []
    axes = [("d", 0), ("r", 0), ("p", 0), ("l", 0)]
    for number in range(20):
        period = 1 + number % 2
        residues = tuple(rng.randrange(period) for _ in range(3))
        extra = [atom([[*rng.choice(axes), rng.choice((-1, 1))] for _ in range(2)], rng.randrange(-3, 4))
                 for _ in range(3)]
        left = [bounded(residues, extra, 1)]
        right = [bounded(residues, [atom([["d", 0, 1], ["r", 0, -1]], number % 3 - 1)])]
        if number % 4 == 0:
            right.append(bounded(tuple(1-r if period == 2 else r for r in residues),
                                 [atom([["p", 0, -1]], 0)]))
        operation = ("import", "subtract", "intersection", "union", "compose")[number % 5]
        instance = base(f"arithmetic-{number}", period, operation=operation, left=left, right=right)
        queries, expected = [], []
        for a, b, p in itertools.product(range(-3, 4), repeat=3):
            source, target = point((a,)), point((b,))
            x = member(left, source, target, [p], period)
            y = member(right, source, target, [p], period)
            answer = {"import": x, "subtract": x and not y, "intersection": x and y, "union": x or y}.get(operation)
            if operation == "compose":
                answer = any(member(left, source, point((middle,)), [p], period) and
                             member(right, point((middle,)), target, [p], period)
                             for middle in range(-2*period, 3*period))
            queries.append(dict(source=source, target=target, parameters=[str(p)]))
            expected.append(["success", answer])
        instance["queries"] = queries
        cases.append((instance, expected))
    # Explicit rationally feasible but integer-empty: x=y and x+y=1.
    eq = [atom([["d", 0, 1], ["r", 0, -1]], 0), atom([["d", 0, -1], ["r", 0, 1]], 0),
          atom([["d", 0, 1], ["r", 0, 1]], 1), atom([["d", 0, -1], ["r", 0, -1]], -1)]
    cases.append((base("integer-empty", left=[bounded(extra=eq)], queries=[]), []))
    # Universe/absent-signature subtraction; all parameters still have residues.
    for name, right in (("universe", [signed_piece(residues=(0, 0, 0))]),
                        ("absent-signature", [signed_piece(residues=(1, 0, 0))])):
        instance = base(name, 2, operation="subtract", left=[bounded()], right=right,
                        queries=[dict(source=point((0,)), target=point((0,)), parameters=["0"])])
        cases.append((instance, [["success", name == "absent-signature"]]))
    for name, changes in (("bad-residue", dict(left=[bounded((2, 0, 0))])),
                          ("bad-axis", dict(left=[bounded(extra=[atom([["d", 2, 1]], 0)])])),
                          ("bad-period", dict(period="0")),
                          ("empty-foreign", dict(operation="union", foreign_space=True, right=[])),
                          ("empty-role", dict(operation="union", right=[], right_domain="unit"))):
        instance = base(name, left=[], queries=[])
        instance.update(changes)
        cases.append((instance, "invalid-input"))
    # Nonfunctional within one piece and between complete output identities.
    common = [atom([["d", 0, 1]], 0), atom([["d", 0, -1]], 0)]
    variable = signed_piece((0, 1), (0, 0), (0, 0, 0), common +
                            [atom([["r", 0, 1]], 1), atom([["r", 0, -1]], 0)])
    for name, parts in (("nonfunctional-interval", [variable]),
                        ("nonfunctional-tag", [signed_piece((0, 1), (0, 0), (0, 0, 0), common +
                                                            [atom([["r", 0, 1]], 0), atom([["r", 0, -1]], 0)]),
                                               signed_piece((0, 1), (1, 0), (0, 0), common)]),
                        ("nonfunctional-residue", [signed_piece((0, 1), (0, 0), (0, residue, 0), common +
                                                               [atom([["r", 0, 1]], 0), atom([["r", 0, -1]], 0)])
                                                    for residue in (0, 1)])):
        cases.append((base(name, 2, domain="event", range="event", operation="selector", left=parts), "non-functional"))
    # Explicit projection, inverse and shared-parameter context operations.
    projection_piece = bounded(extra=[atom([["d", 0, 1], ["r", 0, -1]], -1),
                                      atom([["r", 0, 1], ["p", 0, -1]], 0)])
    for operation in ("domain", "range", "inverse", "context"):
        queries, expected = [], []
        context_parts = [signed_piece(residues=(0,), atoms=[atom([["p", 0, -1]], 0)])]
        for a, b, p in itertools.product(range(-3, 4), repeat=3):
            source, target = point((a,)), point((b,))
            if operation in ("domain", "range"):
                if a != 0:
                    continue
                source = point()
                answer = any(member([projection_piece], point((b if operation == "domain" else middle,)),
                                    point((middle if operation == "domain" else b,)), [p], 1)
                             for middle in range(-2, 3))
            elif operation == "inverse":
                answer = member([projection_piece], target, source, [p], 1)
            else:
                answer = member([projection_piece], source, target, [p], 1) and p >= 0
            queries.append(dict(source=source, target=target, parameters=[str(p)]))
            expected.append(["success", answer])
        cases.append((base(f"explicit-{operation}", left=[projection_piece], operation=operation,
                           right=context_parts, right_domain="unit", right_range="unit", queries=queries,
                           interchange=operation == "domain"), expected))
    # Arbitrary-precision constants and a free unbounded auxiliary; no unfolding.
    huge, period = 10**100 + 1, 3
    values = (huge, -huge, huge + 1)
    atoms = [atom([[role, 0, sign]], sign*(value//period))
             for role, value in zip(("d", "r", "p"), values) for sign in (-1, 1)]
    huge_piece = signed_piece(residues=[x % period for x in values], atoms=atoms, locals_count=1)
    queries = [dict(source=point((huge + delta,)), target=point((-huge,)), parameters=[str(huge+1)])
               for delta in (0, 1, -1)]
    cases.append((base("huge-constants", period, left=[huge_piece], queries=queries, interchange=True),
                  [["success", True], ["success", False], ["success", False]]))
    overlapping = []
    for low, high in ((0, 2), (1, 3)):
        atoms = [atom([["d", 0, -1]], -low), atom([["d", 0, 1]], high),
                 atom([["r", 0, 1], ["d", 0, -1]], 0), atom([["r", 0, -1], ["d", 0, 1]], 0)]
        overlapping.append(signed_piece((0, 1), (0, 0), (0, 0, 0), atoms))
    queries = [dict(source=point((x,), (0, 1)), parameters=["0"]) for x in range(-1, 5)]
    cases.append((base("overlapping-functional", domain="event", range="event", operation="selector",
                       left=overlapping, queries=queries),
                  dict(selector=[["success", event(0, (x,)) if 0 <= x <= 3 else None] for x in range(-1, 5)])))
    zero_roles = base("empty-zero-arity-roles", left=[], right=[], operation="union", right_domain="unit")
    zero_roles["cell_axes"] = 0
    cases.append((zero_roles, "invalid-input"))
    cases.append((base("negative-residue", left=[bounded((-1, 0, 0))]), "invalid-input"))
    cases.append((base("bad-tag", left=[signed_piece((0,), (), (0, 0, 0))]), "invalid-input"))
    doubled = bounded(extra=[atom([["d", 0, 1], ["d", 0, 1]], -1),
                              atom([["d", 0, -1], ["d", 0, -1]], 3)])
    queries = [dict(source=point((x,)), target=point((0,)), parameters=["0"]) for x in (-2, -1, 0)]
    cases.append((base("negative-doubled-rounding", left=[doubled], queries=queries),
                  [["success", False], ["success", True], ["success", False]]))
    false_constant = bounded(extra=[atom([["r", 0, 1], ["r", 0, -1]], -1)])
    cases.append((base("false-cancellation", left=[false_constant], queries=[]), []))
    return cases


def run(executable, path, instance):
    path.write_text(json.dumps([instance]), encoding="utf-8")
    begin = time.monotonic()
    try:
        completed = subprocess.run([executable, str(path)], text=True, capture_output=True, check=True, timeout=90)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(instance["name"], error.stdout, error.stderr, file=sys.stderr)
        raise
    actual = json.loads(completed.stdout)[0]
    sizes = actual.get("sizes", actual.get("pieces"))
    print(f"checked {instance['name']} in {time.monotonic()-begin:.2f}s sizes={sizes}",
          file=sys.stderr, flush=True)
    return actual


def verify_arithmetic(instance, expected, actual):
    if isinstance(expected, str):
        expect(actual["status"], expected, instance["name"])
    elif isinstance(expected, dict):
        expect(actual["status"], "success", instance["name"])
        expect(actual["answers"], expected["selector"], instance["name"])
    else:
        expect(actual["status"], "success", instance["name"])
        expect(actual["answers"], expected, instance["name"])
        expect(actual["probes"], True, "interchange")
        if instance["name"] in ("integer-empty", "false-cancellation"):
            expect(actual["empty"], True, "integer infeasibility")


def main():
    executable, path = shutil.which(sys.argv[1]) or sys.argv[1], Path(sys.argv[2])
    graphs, arithmetic = graph_cases(), arithmetic_cases()
    arithmetic_outputs = []
    for instance, expected in arithmetic:
        actual = run(executable, path, instance)
        verify_arithmetic(instance, expected, actual)
        arithmetic_outputs.append(actual)
    outputs = []
    for instance, expected, original in graphs:
        actual = run(executable, path, instance)
        verify_graph(instance, expected, original, actual)
        for supplied, context in zip(instance["contexts"], actual["contexts"]):
            if context["status"] != "success":
                continue
            for a, source in enumerate(supplied["events"]):
                for b, target in enumerate(supplied["events"]):
                    reach = context["reach"][a][b]
                    strict = ["success", reach[1] and source != target] if reach[0] == "success" else reach
                    expect(context["strict"][a][b], strict, "H=R minus full identity")
        outputs.append(actual)
    huge = convert(varying_depth())[0]
    huge["name"] = "huge-uniform-endpoint"
    n = 10**80 + 7
    huge["contexts"] = [dict(bindings=[str(n)], events=[event(1)])]
    actual = run(executable, path, huge)
    expect(actual["valid"], True, "huge uniform build")
    expect(len(actual["contexts"]), 1, "huge context")
    bound = actual["contexts"][0]
    expect(bound["status"], "success", "huge binding")
    expect(bound["incoming"], [[["success", event(0, (n,), 1)], ["success", None]]], "huge selector")
    expect(bound["strict"], [[["success", False]]], "huge strict identity")
    path.write_text(json.dumps([a for a, _ in arithmetic] + [a for a, _, _ in graphs] + [huge]), encoding="utf-8")
    mutations = []
    for field, a, b in (("minimum", 1, 2), ("reach", 0, 0)):
        changed = copy.deepcopy(outputs[0])
        changed["contexts"][0][field][a][b] = ["success", False]
        mutations.append(changed)
    changed = copy.deepcopy(outputs[0])
    changed["contexts"] = []
    mutations.append(changed)
    changed = copy.deepcopy(outputs[0])
    changed["contexts"][0]["incoming"][2][0][1] = None
    mutations.append(changed)
    for changed in mutations:
        try:
            verify_graph(*graphs[0], changed)
        except ValueError:
            pass
        else:
            raise ValueError("signed physical oracle accepted mutation")
    for name in ("integer-empty", "negative-doubled-rounding"):
        index = next(i for i, (instance, _) in enumerate(arithmetic) if instance["name"] == name)
        changed = copy.deepcopy(arithmetic_outputs[index])
        if name == "integer-empty":
            changed["empty"] = False
        else:
            changed["answers"][1] = ["success", False]
        try:
            verify_arithmetic(*arithmetic[index], changed)
        except ValueError:
            pass
        else:
            raise ValueError("signed arithmetic oracle accepted mutation")
    print(f"verified {len(graphs)+1} signed uniform schemas and {len(arithmetic)} exact arithmetic cases")
    print(f"verified {len(mutations)+2} signed oracle mutations")


if __name__ == "__main__":
    main()
