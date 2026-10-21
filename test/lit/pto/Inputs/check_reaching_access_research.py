# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bounded storage-dataflow qualification, explicitly without command realization."""

import argparse
import itertools
import json
from pathlib import Path
import re
from assess_whole_kernel_access import annotate, digest, invoke
from check_periodic_demands import closure, native
from reaching_access_research import (
    Graph,
    check_execution,
    conflict,
    require,
    solve,
    unfold,
)


def line_of(text, pattern):
    lines = [
        i for i, line in enumerate(text.splitlines(), 1) if re.search(pattern, line)
    ]
    require(len(lines) == 1, f"ambiguous original control {pattern}")
    return lines[0]


def delta_cases(data, text):
    require(not data["error"], data["error"])
    outer = line_of(text, r"scf.for %iv39")
    token = line_of(text, r"scf.for %iv73")
    active = line_of(text, r"scf.if %v40")
    initialized = line_of(text, r"scf.if %v50")
    softplus = line_of(text, r"scf.if %v81")
    prefetch = line_of(text, r"scf.if %v101")
    graph = Graph(data["tree"])
    solution = solve(graph, data["sites"])
    rows = []
    for visits, count, initial in itertools.product((0, 1, 2), range(4), (False, True)):
        for mask in itertools.product((False, True), repeat=count):

            def loops(line, coordinates):
                return visits if line == outer else count if line == token else 0

            def branches(line, coordinates):
                frame = dict(coordinates)
                if line == active:
                    return frame[outer] < 1  # Original guard, not a new caller bound.
                if line == initialized:
                    return initial
                if line == softplus:
                    return mask[frame[token]]
                if line == prefetch:
                    return frame[token] + 1 < count
                raise RuntimeError(f"unmodeled branch {line}")

            word = unfold(data["tree"], loops, branches)
            checked = check_execution(data["sites"], word, solution)
            rows.append(
                dict(
                    outer_visits=visits,
                    token_count=count,
                    initial_state=initial,
                    softplus_outcomes=list(mask),
                    **checked,
                )
            )
    counts = {
        key: value
        for key, value in solution.items()
        if key not in ("incoming", "obligations")
    }
    counts.update(
        static_obligation_pairs=len(solution["obligations"]),
        classes=data["classes"],
        static_sites=len(data["sites"]),
    )
    return {
        "counts": counts,
        "cases": rows,
        "refresh_facts": [],
        "scope": "original effect reservations and control abstraction; not numeric/device execution",
        "realization": "unavailable: dynamic origin selection/history, publication cuts and flag reuse",
    }


def site(pipe, reads=(), writes=()):
    modes = [
        (1 if cell in reads else 0) | (2 if cell in writes else 0) for cell in range(2)
    ]
    effects = [
        {
            "space": 6,
            "bases": [32 * cell],
            "bytes": 4,
            "known": True,
            "write": mode == 2,
        }
        for mode, cells in ((1, reads), (2, writes))
        for cell in cells
    ]
    return {"pipe": pipe, "facts": modes, "effects": effects}


def certify_refresh(sites, proposed):
    for origin, cells in proposed.items():
        for cell in cells:
            require(
                any(
                    effect
                    == {
                        "space": 6,
                        "bases": [32 * cell],
                        "bytes": 4,
                        "known": True,
                        "write": True,
                    }
                    for effect in sites[origin]["effects"]
                ),
                "no proved full-class overwrite",
            )
    return proposed


def synthetic_cases():
    sites = [
        site(0, writes=(0,)),
        site(0, writes=(0,)),
        site(1, reads=(0,)),
        site(1, reads=(1,)),
    ]
    payload = lambda i: {"kind": "payload", "site": i}
    body = [
        {"kind": "if", "line": 3, "then": [payload(1)], "else": []},
        payload(2),
        payload(3),
    ]
    tree = [
        payload(0),
        {
            "kind": "loop",
            "line": 1,
            "body": [{"kind": "loop", "line": 2, "body": body}],
        },
    ]
    graph = Graph(tree)
    refresh = certify_refresh(sites, {1: (0,)})
    baseline, refined = solve(graph, sites), solve(graph, sites, refresh)
    rows = []
    for visits in range(4):
        for counts in itertools.product(range(3), repeat=visits):
            for mask in itertools.product((False, True), repeat=sum(counts)):
                offsets = list(itertools.accumulate((0,) + counts))

                def loops(line, coordinates):
                    return visits if line == 1 else counts[dict(coordinates)[1]]

                def branches(line, coordinates):
                    frame = dict(coordinates)
                    return mask[offsets[frame[1]] + frame[2]]

                word = unfold(tree, loops, branches)
                before = check_execution(sites, word, baseline)
                after = check_execution(sites, word, refined)
                rows.append(
                    {
                        "visits": visits,
                        "counts": counts,
                        "mask": mask,
                        "before": before,
                        "after_refresh": after,
                    }
                )
    # Deliberately absent overwrite proof must never become a definite kill.
    rejected = False
    try:
        certify_refresh(sites, {2: (0,)})
    except RuntimeError:
        rejected = True
    require(rejected, "read-only site accepted as refresh")
    for solution in (baseline, refined):
        require((0, 2) in solution["obligations"], "skipped writer erased entry writer")
    # A dominating full overwrite can remove old origins without assuming a
    # refresh period. This is a separate fully known synthetic class.
    straight = Graph([payload(0), payload(1), payload(2)])
    plain = solve(straight, sites)
    killed = solve(straight, sites, refresh)
    require(
        (0, 2) in plain["obligations"] and (0, 2) not in killed["obligations"],
        "proved refresh did not replace older reaching fact",
    )
    sample = [(i, ()) for i in (0, 1, 2)]
    return {
        "cases": rows,
        "reset_and_skipped_writer_pass": True,
        "read_only_refresh_rejected": rejected,
        "dominating_refresh": {
            "before_facts": plain["facts_retained"],
            "after_facts": killed["facts_retained"],
            "before": check_execution(sites, sample, plain),
            "after": check_execution(sites, sample, killed),
        },
    }


def publication_cut():
    # Init_x; Q0(read x); U0(read y); P1(write x); Q1; U1.
    sites = [
        site(0, writes=(0,)),
        site(1, reads=(0,)),
        site(1, reads=(1,)),
        site(0, writes=(0,)),
        site(1, reads=(0,)),
        site(1, reads=(1,)),
    ]
    edges = native([x["pipe"] for x in sites])
    for b, target in enumerate(sites):
        for a in range(b):
            if conflict(sites[a], target):
                edges.add((2 * a + 1, 2 * b))
    required, _ = closure(12, edges)
    # A release published after U0 at the next writer's static cut adds U0->P1.
    late, _ = closure(12, edges | {(5, 6)})
    added = [
        (a, b)
        for a in range(6)
        for b in range(6)
        if late[2 * a + 1] >> (2 * b) & 1 and not required[2 * a + 1] >> (2 * b) & 1
    ]
    require(added == [(2, 3), (2, 4), (2, 5)], f"cut witness changed: {added}")
    return {
        "extra_pairs": [["U0", "P1"], ["U0", "Q1"], ["U0", "U1"]],
        "earlier_publication_cut": "not qualified: next branch guard availability is unproved",
        "uniform_set_wait_realization": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    text = annotate(args.input.read_text(), "test.reaching_access")
    source = args.output / "delta-input.pto"
    source.write_text(text)
    run = invoke(
        [args.trace_tool, "--step1-json", source], args.output / "input.json", 35
    )
    require(run["status"] == "success", f"access input export failed: {run}")
    data = json.loads(Path(run["stdout"]).read_text())["reaching_access"]
    report = {
        "input_sha256": digest(args.input),
        "tool_sha256": digest(args.trace_tool),
        "run": run,
        "delta": delta_cases(data, text),
        "synthetic": synthetic_cases(),
        "two_class_publication_cut": publication_cut(),
        "production_changed": False,
    }
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print("reaching-access prototype coverage passed; realization remains unqualified")


if __name__ == "__main__":
    main()
