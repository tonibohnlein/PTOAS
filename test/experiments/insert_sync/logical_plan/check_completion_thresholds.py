#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0.
"""Exercise native completion-update thresholds and nested budget scopes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def relation(pairs: set[tuple[int, int]]) -> dict:
    pieces = []
    for source, target in sorted(pairs):
        pieces.append({"locals": 0,
                       "eq": [[1, 0, -source], [0, 1, -target]],
                       "ge": []})
    return {"d": 1, "r": 1, "s": 0, "pieces": pieces}


def fixed_pairs(value: dict) -> set[tuple[int, int]]:
    result = set()
    for piece in value["pieces"]:
        source = target = None
        for row in piece["eq"]:
            coefficients = [int(item) for item in row]
            if len(coefficients) != 3:
                continue
            if coefficients[1] == 0 and coefficients[0] in (-1, 1):
                source = -coefficients[2] // coefficients[0]
            if coefficients[0] == 0 and coefficients[1] in (-1, 1):
                target = -coefficients[2] // coefficients[1]
        assert source is not None and target is not None, piece
        result.add((source, target))
    return result


def run(driver: Path, output: Path, name: str, request: dict, expected_exit: int = 0) -> dict:
    (output / (name + ".request.json")).write_text(json.dumps(request, indent=2) + "\n")
    completed = subprocess.run([str(driver)], input=json.dumps(request), text=True,
                               capture_output=True, check=False, timeout=40)
    (output / (name + ".stdout")).write_text(completed.stdout)
    (output / (name + ".stderr")).write_text(completed.stderr)
    assert completed.returncode == expected_exit, (completed.returncode, completed.stderr[-4000:])
    if expected_exit:
        return {"returncode": completed.returncode, "stderr": completed.stderr}
    return json.loads(completed.stdout)


def threshold_case(driver: Path, output: Path, name: str, base_count: int, added_count: int) -> dict:
    base = {(0, target) for target in range(1, base_count + 1)}
    added = {(0, target) for target in range(base_count + 1, base_count + added_count + 1)}
    identity = {(target, target) for target in range(base_count + added_count + 1)}
    expected = base | added
    request = {"op": "completion", "a": relation(base), "issue_order": relation(identity),
               "record_steps": True, "budget": 8000000,
               "needs": [
                   {"relation": relation({(0, base_count)}), "rounds": 0},
                   {"relation": relation({(0, base_count + added_count)}),
                    "add_handoffs": relation(added), "rounds": 0},
                   # supply() is query-scoped, not the complete cached state.
                   # Request all targets explicitly to challenge retention.
                   {"relation": relation(expected), "rounds": 0},
               ]}
    response = run(driver, output, name, request)
    assert [answer["status"] for answer in response["answers"]] == ["proved"] * 3
    assert len(response["steps"]) == 3, response
    assert fixed_pairs(response["steps"][0]["supply"]) == {(0, base_count)}
    step = response["steps"][1]
    assert fixed_pairs(step["supply"]) == {(0, base_count + added_count)}
    assert fixed_pairs(response["steps"][2]["supply"]) == expected
    assert fixed_pairs(response["relation"]) == expected, (name, response["relation"])
    for counter in ("frontier_incremental_updates", "frontier_replay_updates"):
        assert response["steps"][2][counter] == step[counter], (name, counter, response)
    return {"name": name, "base_pieces": base_count, "added_pieces": added_count,
            "incremental_updates": step["frontier_incremental_updates"],
            "replay_updates": step["frontier_replay_updates"],
            "fresh_pairs": len(expected)}


def main() -> None:
    if not __debug__:
        raise RuntimeError("Completion threshold checks require assertions enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    budget = run(args.driver, args.output, "nested-budget", {
        "op": "budget_scope", "budget": 100, "a": relation(set())})
    assert budget["outer_first"] and budget["inner"]
    assert budget["outer_second"] and not budget["outer_over"], budget
    assert budget["work"] == 10, budget
    # accepted, local-exhausted, local remainder, parent remainder, charge.
    expected_budgets = {
        "local-overrun": (False, True, 0, 90, 10),
        "parent-overrun": (False, True, 0, 90, 10),
        "maximum-request": (False, True, 0, 90, 10),
        "zero-scope": (False, True, 0, 100, 0),
        "exact-scope": (True, False, 0, 90, 10),
        "partial-scope": (True, False, 7, 97, 3),
        "zero-request": (True, False, 0, 100, 0),
        "whole-parent": (False, True, 0, 0, 100),
        "clipped-scope": (False, True, 0, 0, 10),
        "empty-parent": (False, True, 0, 0, 0),
    }
    actual_budgets = {case["name"]: tuple(case[key] for key in (
        "accepted", "exhausted", "local_remaining", "parent_remaining", "work"))
        for case in budget["scope_cases"]}
    assert len(budget["scope_cases"]) == len(expected_budgets)
    assert actual_budgets == expected_budgets, actual_budgets
    assert budget["nested_overrun"] == {
        "first": True, "rejected": True, "child_exhausted": True, "after_child": 3,
        "resumed": True, "over": False, "parent_exhausted": True,
        "parent_remaining": 90, "work": 10,
    }, budget["nested_overrun"]
    assert budget["unscoped_overrun"] == {"accepted": False, "remaining": 0, "work": 10}

    # The input protocol uses JSON numbers; the arbitrary-precision OUTPUT
    # strings must not be fed to the signed-int64 input reader unchecked.
    malformed = relation({(0, 1)})
    malformed["pieces"][0]["eq"][0][0] = "1"
    invalid = run(args.driver, args.output, "reject-string-coefficient", {
        "op": "normalize", "a": malformed}, expected_exit=2)
    assert "relation coefficient must be a signed integer JSON number" in invalid["stderr"]

    cases = [
        threshold_case(args.driver, args.output, "pieces-16", 16, 1),
        threshold_case(args.driver, args.output, "pieces-17", 17, 1),
        threshold_case(args.driver, args.output, "product-256", 16, 16),
        threshold_case(args.driver, args.output, "product-272", 16, 17),
    ]
    assert cases[0]["incremental_updates"] == 1 and cases[0]["replay_updates"] == 0, cases[0]
    assert cases[1]["incremental_updates"] == 0 and cases[1]["replay_updates"] == 1, cases[1]
    assert cases[2]["incremental_updates"] == 1 and cases[2]["replay_updates"] == 0, cases[2]
    assert cases[3]["incremental_updates"] == 0 and cases[3]["replay_updates"] == 1, cases[3]
    summary = {"status": "passed", "nested_budget": budget, "thresholds": cases,
               "fresh_reference": "identity issue order; expected relation is base union addition"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
