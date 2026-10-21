# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bounded original Delta control/reservation checker; not numerical execution."""

import itertools
import re
from check_periodic_demands import closure, native
from check_reaching_access_research import line_of
from compact_vector_oracle import command_order
from reaching_access_research import conflict, require, unfold

PIPES = dict(S=0, V=1, M=2, MTE1=3, MTE2=4, MTE3=5, ALL=6, FIX=10)


def parse_program(text, sites):
    names = {s["operation"] for s in sites}
    groups, pending, stack, tree = {}, [], [], []
    current = tree
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("scf.for ") or line.startswith("scf.if "):
            require(not pending, "commands crossed a control boundary")
            node = (
                dict(kind="loop", body=[])
                if line.startswith("scf.for ")
                else dict(kind="if", then=[], **{"else": []})
            )
            current.append(node)
            stack.append((current, node))
            current = node.get("body", node.get("then"))
            continue
        if line == "} else {":
            require(not pending, "commands crossed branch join")
            current = stack[-1][1]["else"]
            continue
        if line == "}" and stack:
            require(not pending, "commands escaped body")
            current, _ = stack.pop()
            continue
        flag = re.search(
            r"pto\.(set_flag|wait_flag)\[<PIPE_(\w+)>, <PIPE_(\w+)>, <EVENT_ID(\d+)>\]",
            line,
        )
        if flag:
            kind, p, q, ident = flag.groups()
            p, q = PIPES[p], PIPES[q]
            pending.append(
                dict(
                    kind="set" if kind == "set_flag" else "wait",
                    pipe=p if kind == "set_flag" else q,
                    source_pipe=p,
                    target_pipe=q,
                    physical_id=int(ident),
                )
            )
            continue
        barrier = re.search(r"pto\.barrier <PIPE_(\w+)>", line)
        if barrier:
            pending.append(dict(kind="barrier", pipe=PIPES[barrier[1]]))
            continue
        payload = re.match(r"(?:%[\w]+ = )?(pto\.[\w]+)", line)
        if payload and payload[1] in names:
            site = len(groups)
            require(
                site < len(sites) and sites[site]["operation"] == payload[1],
                "original payload changed",
            )
            groups[site] = pending
            pending = []
            current.append(dict(kind="payload", site=site))
    require(len(groups) == len(sites) and not stack, "missing payload/control")
    require(pending == [dict(kind="barrier", pipe=6)], "return completion changed")
    return tree, groups


def shape(tree):
    return [
        (
            n["kind"],
            n["site"]
            if n["kind"] == "payload"
            else shape(n["body"])
            if n["kind"] == "loop"
            else (shape(n["then"]), shape(n["else"])),
        )
        for n in tree
    ]


def service_units(group, target, roles):
    """Decode intended pairs independently from the actual four-command blocks."""
    result, index = [], 0
    while index < len(group):
        a = group[index]
        if a["kind"] == "barrier":
            require(
                index == len(group) - 1 and a["pipe"] == target, "local barrier moved"
            )
            result.append([a])
            break
        require(len(group) - index >= 4, "partial acknowledgment service")
        a, b, c, d = group[index : index + 4]
        require(
            [e["kind"] for e in (a, b, c, d)] == ["set", "wait", "set", "wait"],
            "service order changed",
        )
        source = a["source_pipe"]
        require(
            source != target and a["target_pipe"] == target, "service target changed"
        )
        require(
            a["pipe"] == source
            and b["pipe"] == target
            and c["pipe"] == target
            and d["pipe"] == source,
            "service endpoint pipe changed",
        )
        require(
            all(
                (e["source_pipe"], e["target_pipe"]) == (source, target) for e in (a, b)
            ),
            "forward matching changed",
        )
        require(
            all(
                (e["source_pipe"], e["target_pipe"]) == (target, source) for e in (c, d)
            ),
            "reverse matching changed",
        )
        require(
            a["physical_id"] == b["physical_id"]
            and c["physical_id"] == d["physical_id"],
            "intended consumer changed",
        )
        for role, event in (("F", a), ("R", c)):
            direction = event["source_pipe"], event["target_pipe"]
            key = role, direction
            require(
                key not in roles or roles[key] == event["physical_id"],
                "role was recolored",
            )
            roles[key] = event["physical_id"]
            other = ("R" if role == "F" else "F"), direction
            require(
                other not in roles or roles[other] != event["physical_id"],
                "directional roles collide",
            )
            require(event["physical_id"] in range(6), "capacity violation")
        result.append([a, b, c, d])
        index += 4
    return result


def check_service_subsequence(parsed, sites):
    roles = [{}, {}]
    for site in range(len(sites)):
        before, after = [
            service_units(groups[site], sites[site]["pipe"], roles[i])
            for i, (_, groups) in enumerate(parsed)
        ]
        cursor = 0
        for unit in before:
            if cursor < len(after) and unit == after[cursor]:
                cursor += 1
                continue
            require(
                len(unit) == 4
                and unit[0]["source_pipe"] != 0
                and unit[0]["target_pipe"] != 0,
                "protected scalar service/local barrier removed",
            )
        require(cursor == len(after), "service added or moved")
    require(
        all(roles[0].get(key) == value for key, value in roles[1].items()),
        "assignment changed",
    )


def scalar_dependencies(text, sites):
    names = {s["operation"] for s in sites}
    deps, controls, static, frames = {}, [], {}, {}
    variable = re.compile(r"%[A-Za-z_][A-Za-z_0-9]*")
    phase = 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line.startswith("scf.for "):
            values = variable.findall(line)
            iv = values[0]
            local = set().union(*(deps.get(v, set()) for v in values[1:]))
            deps[iv] = local
            controls.append((number, local, "loop"))
            continue
        if line.startswith("scf.if "):
            controls.append((number, deps.get(variable.findall(line)[0], set()), "if"))
            continue
        if line == "} else {":
            continue
        if line == "}" and controls:
            controls.pop()
            continue
        definition = re.match(r"(%[A-Za-z_][A-Za-z_0-9]*) = (.*)", line)
        lhs, rhs = definition.groups() if definition else (None, line)
        values = variable.findall(rhs)
        payload = re.match(r"(pto\.[\w]+)", rhs)
        if payload and payload[1] in names:
            require(
                sites[phase]["operation"] == payload[1], "SSA payload inventory changed"
            )
            static[phase] = set().union(
                *(deps.get(v, set()) for v in values), *(c[1] for c in controls)
            )
            frames[phase] = [c[0] for c in controls if c[2] == "loop"]
            if lhs:
                deps[lhs] = {phase}
            phase += 1
        elif lhs:
            deps[lhs] = set().union(*(deps.get(v, set()) for v in values))
    require(phase == len(sites), "incomplete original SSA inventory")
    return static, frames


def graph_for(word, sites, groups, dependencies, frames):
    events, last, scalar, controls = [], {}, [], []
    command_count = 0
    for gap, (site, coords) in enumerate(word):
        for producer in dependencies[site]:
            require(producer in last, "scalar definition did not execute")
            earlier = last[producer]
            old, now = dict(word[earlier][1]), dict(coords)
            require(
                all(old[line] == now[line] for line in frames[producer]),
                "SSA history crossed reset",
            )
            scalar.append((earlier, gap))
            controls.extend(
                (earlier, command_count + i) for i in range(len(groups[site]))
            )
        last[site] = gap
        events.extend(dict(e, gap=gap) for e in groups[site])
        command_count += len(groups[site])
        events.append(
            dict(
                kind="payload",
                type=site,
                coordinates=[v for _, v in coords],
                pipe=sites[site]["pipe"],
            )
        )
    events.append(dict(kind="barrier", pipe=6, gap=len(word)))
    rows, reuse = command_order(
        dict(events=events, error=""),
        scalar=scalar,
        include_commands=True,
        native_serial_pipes=(0,),
        command_prerequisites=controls,
    )
    require(
        all(not (row >> i & 1) for i, row in enumerate(rows)), "command/SSA graph cycle"
    )
    if any(e["kind"] == "set" for e in events):
        require(reuse["status"] == "passed", str(reuse))
    return rows, dict(
        commands=len(events) - len(word),
        reuse=reuse,
        scalar_edges=len(scalar),
        command_control_edges=len(controls),
    )


def normalized_payload(program):
    lines = [
        line
        for line in program.splitlines()
        if not re.search(r"pto\.(?:set_flag|wait_flag|barrier)", line)
    ]
    token = re.compile(r"%[A-Za-z_0-9]+")
    uses = {}
    for line in lines:
        definition = re.match(r"\s*(%[A-Za-z_0-9]+) = (.*)", line)
        rhs = definition[2] if definition else line
        for value in token.findall(rhs):
            uses[value] = uses.get(value, 0) + 1
    retained = []
    for line in lines:
        definition = re.match(r"\s*(%[A-Za-z_0-9]+) = arith.constant ", line)
        if definition and not uses.get(definition[1]):
            continue
        retained.append(line)
    names = {}

    def rename(match):
        key = match[0]
        if key not in names:
            names[key] = f"%v{len(names)}"
        return names[key]

    return token.sub(rename, "\n".join(retained))


def check_delta(text, data, programs):
    sites = data["sites"]
    parsed = [parse_program(program, sites) for program in programs]
    require(
        all(shape(tree) == shape(data["tree"]) for tree, _ in parsed),
        "original control shape changed",
    )
    check_service_subsequence(parsed, sites)

    # The map emitter also leaves unused scalar constants. Remove only
    # individually unreferenced constants, then alpha-rename SSA tokens. Keep
    # all other calculations, literals, control and payloads in exact order.
    require(
        normalized_payload(programs[0]) == normalized_payload(programs[1]),
        "payload/control text changed",
    )
    dependencies, frames = scalar_dependencies(text, sites)
    outer = line_of(text, r"scf.for %iv39")
    token = line_of(text, r"scf.for %iv73")
    active = line_of(text, r"scf.if %v40")
    initial = line_of(text, r"scf.if %v50")
    soft = line_of(text, r"scf.if %v81")
    prefetch = line_of(text, r"scf.if %v101")
    cases = []
    controls = [
        (2, count, state, mask)
        for count in range(4)
        for state in (False, True)
        for mask in itertools.product((False, True), repeat=count)
    ]
    controls += [(0, 0, False, ()), (1, 1, False, (False,))]
    for visits, count, state, mask in controls:

        def loops(line, coords):
            return visits if line == outer else count if line == token else 0

        def branches(line, coords):
            frame = dict(coords)
            if line == active:
                return frame[outer] < 1
            if line == initial:
                return state
            if line == soft:
                return mask[frame[token]]
            if line == prefetch:
                return frame[token] + 1 < count
            raise ValueError(f"unknown Delta control {line}")

        word = unfold(data["tree"], loops, branches)
        results = [
            graph_for(word, sites, groups, dependencies, frames) for _, groups in parsed
        ]
        before, after = results[0][0], results[1][0]
        count_events = 2 * len(word)
        bits = (1 << count_events) - 1
        require(
            all(not (after[i] & bits & ~before[i]) for i in range(count_events)),
            "candidate added payload order",
        )
        edges = native([sites[site]["pipe"] for site, _ in word], (0,))
        checks = 0
        for b, (target, _) in enumerate(word):
            for a, (source, _) in enumerate(word[:b]):
                if conflict(sites[source], sites[target]):
                    edges.add((2 * a + 1, 2 * b))
                    checks += 1
        last = {}
        for b, (site, _) in enumerate(word):
            edges.update((2 * last[p] + 1, 2 * b) for p in dependencies[site])
            last[site] = b
        wanted, _ = closure(count_events, edges)
        require(
            all(not ((row & ~(1 << i)) & ~after[i]) for i, row in enumerate(wanted)),
            "modeled requirement uncovered",
        )
        excess = [
            sum(
                bool(rows[2 * a + 1] >> (2 * b) & 1)
                and not bool(wanted[2 * a + 1] >> (2 * b) & 1)
                for a in range(len(word))
                for b in range(len(word))
            )
            for rows, _ in results
        ]
        cases.append(
            dict(
                visits=visits,
                tokens=count,
                initial=state,
                mask=list(mask),
                payloads=len(word),
                scope="configured outer count; supplied outcomes"
                if visits == 2
                else "structural derivative",
                checks=checks,
                baseline=results[0][1],
                candidate=results[1][1],
                excess=excess,
                added_pairs=0,
                removed_pairs=sum(
                    (before[i] & bits & ~after[i]).bit_count()
                    for i in range(count_events)
                ),
            )
        )
    return cases
