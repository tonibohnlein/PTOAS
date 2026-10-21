# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare native projected states with the independent origin fixed point."""

import argparse
import copy
import json
from pathlib import Path
from assess_whole_kernel_access import annotate, digest, invoke
from check_reaching_access_research import site
from reaching_access_research import Graph, generated, require, solve


def reference_plan(tree, sites, possible, physical):
    result = copy.deepcopy(tree)
    for item in result:
        if item["kind"] == "payload":
            index = item["site"]
            target = sites[index]["pipe"]
            sources = sorted(possible[index] - {target})
            item["kind"] = "service"
            commands = []
            for source in sources:
                key = f"F:{source}:{target}" if physical else source
                commands.append(dict(op="S", pipe=source, source=key))
                if physical:
                    reverse = f"R:{target}:{source}"
                    commands += [
                        dict(op="W", pipe=target, source=key),
                        dict(op="S", pipe=target, source=reverse),
                        dict(op="W", pipe=source, source=reverse),
                    ]
            if target in possible[index]:
                commands.append(dict(op="B", pipe=target))
            if not physical:
                commands += [
                    dict(op="W", pipe=target, source=source) for source in sources
                ]
            item["commands"] = commands
        else:
            keys = ("body",) if item["kind"] == "loop" else ("then", "else")
            for key in keys:
                item[key] = reference_plan(item[key], sites, possible, physical)
    return result


def compare(data, result):
    require(not result["error"], result["error"])
    sites = data["sites"]
    graph = Graph(data["tree"])
    refresh = {i: item.get("refresh", []) for i, item in enumerate(sites)}
    old = solve(graph, sites, refresh)

    def project(facts):
        return {(cell, mode, sites[origin]["pipe"]) for cell, mode, origin in facts}

    for node, origin in enumerate(graph.nodes):
        facts = old["incoming"][node]
        require(
            project(facts) == set(map(tuple, result["incoming"][node])), "IN differs"
        )
        if origin is not None:
            facts = {fact for fact in facts if fact[0] not in refresh[origin]}
            facts |= generated(origin, sites[origin]["facts"])
        require(
            project(facts) == set(map(tuple, result["outgoing"][node])), "OUT differs"
        )
    possible = {i: set() for i in range(len(sites))}
    for origin, target in old["obligations"]:
        possible[target].add(sites[origin]["pipe"])
    require(
        [sorted(possible[i]) for i in possible] == result["possible"], "services differ"
    )
    for physical, key in ((False, "logical_plan"), (True, "physical_plan")):
        require(
            reference_plan(data["tree"], sites, possible, physical) == result[key],
            "command plan differs",
        )
    roles = {}
    for target, sources in possible.items():
        q = sites[target]["pipe"]
        for p in sources - {q}:
            roles.setdefault((p, q), set()).add("F")
            roles.setdefault((q, p), set()).add("R")
    expected = {
        (role, p, q): i
        for (p, q), names in roles.items()
        for i, role in enumerate(sorted(names))
    }
    actual = {
        (row["role"], row["source"], row["target"]): row["id"]
        for row in result["role_ids"]
    }
    require(expected == actual, "role mapping differs")
    work = result["metrics"]
    bound = 2 * data["classes"] * len({s["pipe"] for s in sites})
    transmissions = sum(
        len(result["outgoing"][i]) * len(graph.successors[i])
        for i in range(len(graph.nodes))
    )
    require(work["edge_fact_transmissions"] == transmissions, "duplicate edge delivery")
    require(transmissions <= bound * work["cfg_edges"], "transmission bound")
    require(
        all(len(facts) <= bound for facts in result["incoming"] + result["outgoing"]),
        "state bound",
    )
    return dict(
        old_origin_pairs=len(old["obligations"]),
        old_out_facts=old["facts_retained"],
        **work,
    )


def native(args, data, name):
    path = args.output / f"{name}.json"
    path.write_text(json.dumps(data))
    run = invoke(
        [args.trace_tool, "--source-pipe-json", path],
        path.with_suffix(".result.json"),
        35,
    )
    require(run["status"] == "success", str(run))
    result = json.loads(Path(run["stdout"]).read_text())
    return result


def fixtures():
    payload = lambda i: dict(kind="payload", site=i)
    sites = [
        site(0, writes=(0,)),
        site(0, writes=(0,)),
        site(1, reads=(0,), writes=(1,)),
        site(1, reads=(1,)),
    ]
    for refresh in (False, True):
        for empty in (False, True):
            inventory = copy.deepcopy(sites)
            if refresh:
                inventory[1]["refresh"] = [0]
            branch = dict(kind="if", line=3, then=[payload(1)], **{"else": []})
            body = [] if empty else [branch, payload(2), payload(3)]
            tree = [
                payload(0),
                dict(kind="loop", line=1, body=[dict(kind="loop", line=2, body=body)]),
            ]
            yield (
                f"reset-{refresh}-{empty}",
                dict(classes=2, sites=inventory, tree=tree),
            )
    for n in (0, 1, 16, 32, 64, 128, 256):
        inventory = [site(i % 2, reads=(0, 1), writes=(0, 1)) for i in range(n)]
        yield (
            f"scaling-{n}",
            dict(
                classes=2,
                sites=inventory,
                tree=[dict(kind="loop", line=1, body=[payload(i) for i in range(n)])],
            ),
        )
    # KILL must retain IN for selection and must not kill this node's own GEN.
    inventory = [
        site(0, writes=(0,)),
        site(1, reads=(0,), writes=(0,)),
        site(1, reads=(0,)),
    ]
    inventory[1]["refresh"] = [0]
    yield (
        "refresh-regeneration",
        dict(classes=2, sites=inventory, tree=[payload(i) for i in range(3)]),
    )
    yield "empty", dict(classes=0, sites=[], tree=[])
    yield (
        "empty-loop",
        dict(classes=0, sites=[], tree=[dict(kind="loop", line=1, body=[])]),
    )
    yield "no-access", dict(classes=2, sites=[site(0)], tree=[payload(0)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "delta.pto"
    source.write_text(annotate(args.input.read_text(), "test.reaching_access"))
    run = invoke(
        [args.trace_tool, "--step1-json", source], args.output / "delta-input.json", 35
    )
    require(run["status"] == "success", str(run))
    data = json.loads(Path(run["stdout"]).read_text())["reaching_access"]
    require(not data["error"], data["error"])
    rows = {"delta": compare(data, data["projected"])}
    for name, fixture in fixtures():
        result = native(args, fixture, name)
        rows[name] = compare(fixture, result)
        if name == "refresh-regeneration":
            require(
                result["possible"][1] == [0] and result["possible"][2] == [1],
                "refresh selection changed",
            )
    negatives = [
        dict(classes=1, sites=[dict(pipe=0, facts=[4])], tree=[]),
        dict(classes=1, sites=[dict(pipe=0, facts=[1], refresh=[1])], tree=[]),
        dict(classes=1, sites=[], tree=[dict(kind="payload", site=0)]),
        dict(classes=2, sites=[dict(pipe=0, facts=[1])], tree=[]),
    ]
    for i, fixture in enumerate(negatives):
        require(
            native(args, fixture, f"negative-{i}")["error"], "invalid input accepted"
        )
    receipt = dict(
        input_sha256=digest(args.input),
        tool_sha256=digest(args.trace_tool),
        run=run,
        equality="all IN/OUT, services, structured plans, role IDs",
        results=rows,
        scope="declared access model; no native Delta realization claim",
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("source-pipe projection, plans and role IDs agree with origin oracle")


if __name__ == "__main__":
    main()
