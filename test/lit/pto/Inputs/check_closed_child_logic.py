# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check a prepared symbolic K child with independent matching and poset width."""

import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import resource
import re
import shutil

from assess_whole_kernel_access import annotate, digest, invoke
from compact_vector_oracle import command_order
from check_finite_allocation import physical_check
from pypto_region_oracle import interpret, mathematical
from survey_pypto_regions import specialize


def identity(event):
    return (event["plan"], event["record"], event["source_ordinal"], tuple(event.get("members", [])))


def chain_cover(order):
    """Maximum bipartite matching plus a checked equal-size antichain witness."""
    size = len(order)
    right = [-1] * size
    def augment(left, seen):
        for target in range(size):
            if not order[left] >> target & 1 or target in seen:
                continue
            seen.add(target)
            if right[target] < 0 or augment(right[target], seen):
                right[target] = left
                return True
        return False
    for left in range(size):
        augment(left, set())
    successor = {left: target for target, left in enumerate(right) if left >= 0}
    reached_left = set(range(size)) - set(successor)
    reached_right = set()
    pending = list(reached_left)
    while pending:
        left = pending.pop()
        for target in range(size):
            if not order[left] >> target & 1 or successor.get(left) == target or target in reached_right:
                continue
            reached_right.add(target)
            if right[target] >= 0 and right[target] not in reached_left:
                reached_left.add(right[target])
                pending.append(right[target])
    antichain = sorted(reached_left - reached_right)
    colors = {}
    for head in range(size):
        if right[head] >= 0:
            continue
        lane = head
        color = len(set(colors.values()))
        while True:
            if lane in colors:
                raise AssertionError("Cyclic chain partition")
            colors[lane] = color
            if lane not in successor:
                break
            lane = successor[lane]
    width = size - len(successor)
    if len(antichain) != width or len(colors) != size:
        raise AssertionError("Primal/dual width mismatch")
    if any(order[a] >> b & 1 for a in antichain for b in antichain):
        raise AssertionError("Invalid antichain")
    return width, colors, antichain


def child_phases(original):
    start = original.index("scf.for %arg11 = %c0 to %8")
    opening = original.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (original[end] == "{") - (original[end] == "}")
        end += 1
    # Independent static numbering agrees with the bounded interpreter's
    # source-operation inventory. The later tail loop also has two coordinates.
    sites = list(re.finditer(r"pto\.t[\w.]+ ins", original))
    return {i for i, site in enumerate(sites) if opening < site.start() < end}


def analyze_trace(trace, original, outer_visits, k_count):
    items = interpret(original)
    required, _ = mathematical(items)
    payload = [e for e in trace["events"] if e["kind"] == "payload"]
    if [(e["type"], e["coordinates"], e["pipe"]) for e in payload] != [
        (e["phase"], e["coordinates"], e["pipe"]) for e in items
    ]:
        raise AssertionError("Original payload mapping changed")
    phases = child_phases(original)
    children = defaultdict(list)
    for index, event in enumerate(payload):
        if event["type"] in phases:
            children[event["coordinates"][0]].append(index)
    commands = [e for e in trace["events"] if e["kind"] != "payload"]
    owners = {}
    pending = {}
    pairs = {}
    occupied = defaultdict(set)
    directions = defaultdict(list)
    for index, event in enumerate(commands):
        if event["kind"] == "barrier":
            continue
        owner = [v for v, sites in children.items() if sites[0] <= event["gap"] <= sites[-1] + 1]
        if len(owner) != 1:
            raise AssertionError(("Endpoint outside closed child", event))
        key = identity(event)
        if event["kind"] == "set":
            if key in owners or key in pending:
                raise AssertionError("Publication identity repeated across resets")
            owners[key] = owner[0]
            pending[key] = index
            occupied[owner[0]].add(key)
        else:
            if key not in pending or owners[key] != owner[0]:
                raise AssertionError("Unpublished or cross-invocation acquisition")
            publication = pending.pop(key)
            occupied[owner[0]].remove(key)
            pairs[key] = (publication, index)
            directions[(event["source_pipe"], event["target_pipe"])].append(key)
    if pending or any(occupied.values()):
        raise AssertionError("Child exit has unmatched publications")
    # Independent command causality includes asymmetric SET/WAIT semantics and
    # rejects cycles; matched identities are not inferred from compiler demands.
    reach, _ = command_order(trace, include_commands=True)
    if any(row >> i & 1 for i, row in enumerate(reach)):
        raise AssertionError("Command cycle")
    child_sites = {i: owner for owner, sites in children.items() for i in sites}
    missing = [(a, b) for a in child_sites for b in child_sites
               if child_sites[a] == child_sites[b] and required[2*a+1] >> (2*b) & 1
               and not reach[2*a+1] >> (2*b) & 1]
    if missing:
        raise AssertionError(("Child demand coverage", missing[:3]))
    result = {"outer_visits": outer_visits, "K": k_count, "payloads": len(items),
              "commands": len(commands), "pairs": len(pairs), "matching": "passed",
              "invocation_identity": "distinct", "logical_entry_exit": "no unmatched publications",
              "payload_completion_at_exit": "not implied", "acyclic": True,
              "required_child_ci_missing": 0, "directional_widths": []}
    assignment = {}
    offset = 2 * len(payload)
    for direction, keys in sorted(directions.items()):
        order = [sum(1 << j for j, later in enumerate(keys)
                     if reach[offset+pairs[key][1]] >> (offset+pairs[later][0]) & 1)
                 for key in keys]
        if any(order[i] >> i & 1 for i in range(len(order))):
            raise AssertionError("Reflexive lifetime relation")
        if any(order[j] & ~order[i] for i in range(len(order)) for j in range(len(order)) if order[i] >> j & 1):
            raise AssertionError("Lifetime relation is not transitive")
        width, colors, antichain = chain_cover(order)
        result["directional_widths"].append({"direction": direction, "handoffs": len(keys),
            "minimum": width, "antichain": [keys[i] for i in antichain]})
        assignment.update({keys[i]: color for i, color in colors.items()})
    fits = all(d["minimum"] <= 6 for d in result["directional_widths"])
    result["fits_six_for_this_execution"] = fits
    if fits:
        physical = copy.deepcopy(trace)
        for event in physical["events"]:
            if event["kind"] not in ("payload", "barrier"):
                event["physical_id"] = assignment[identity(event)]
        physical_check({"accepted": True, "allocated": True, "trace": trace, "physical": physical}, set(range(6)))
        result["concrete_assignment"] = [{"identity": key, "id": value} for key, value in assignment.items()]
        result["causal_reuse"] = "passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace-tool", "input", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    tool = Path(shutil.which(str(args.trace_tool)) or args.trace_tool).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    original = args.input.read_text()
    source = args.output / "symbolic-input.pto"
    attributes = ('test.contract_math_only, test.contract_region = 5 : i64, '
                  'test.contract_gm_alias = "may-not-alias", test.contract_completion_emit')
    source.write_text(annotate(original, attributes))
    emitted = args.output / "symbolic-child.pto"
    run = invoke([tool, "--compact-region-contract", source], emitted, 35)
    if run["status"] != "success":
        raise AssertionError(run)
    results = {"input_sha256": digest(args.input), "tool_sha256": digest(tool), "emission": run,
               "scope": "One symbolic retained recipe, evaluated afterwards; no physical compiler certificate implied",
               "cases": []}
    for visits, k in ((0,2), (1,0), (1,1), (1,2), (1,3), (1,4), (1,6), (2,2), (2,3), (3,4)):
        values = {"%arg3":20*visits,"%arg4":k,"%arg5":0,"%arg6":0,"%arg7":4352,"%arg8":0,"%arg9":20}
        path = args.output / f"visits{visits}-K{k}.pto"
        path.write_text(annotate(specialize(emitted.read_text(), values), "test.trace_existing"))
        run = invoke([tool, "--structured-trace", path], path.with_suffix(".json"), 35)
        if run["status"] != "success":
            raise AssertionError(run)
        trace = json.loads(Path(run["stdout"]).read_text())
        if not trace.get("accepted") or trace["trace"].get("error"):
            raise AssertionError(trace)
        row = analyze_trace(trace["trace"], specialize(original, values), visits, k)
        row["run"] = run
        results["cases"].append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2)+"\n")
        print(visits, k, [(d["direction"], d["minimum"]) for d in row["directional_widths"]], flush=True)


if __name__ == "__main__":
    main()
