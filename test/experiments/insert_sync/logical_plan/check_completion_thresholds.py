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
                       "eq": [["1", "0", str(-source)], ["0", "1", str(-target)]],
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


def run(driver: Path, output: Path, name: str, request: dict) -> dict:
    completed = subprocess.run([str(driver)], input=json.dumps(request), text=True,
                               capture_output=True, check=False, timeout=40)
    (output / (name + ".stdout")).write_text(completed.stdout)
    (output / (name + ".stderr")).write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout)


def threshold_case(driver: Path, output: Path, name: str, base_count: int, added_count: int) -> dict:
    base = {(0, target) for target in range(1, base_count + 1)}
    added = {(0, target) for target in range(base_count + 1, base_count + added_count + 1)}
    identity = {(target, target) for target in range(base_count + added_count + 1)}
    request = {"op": "completion", "a": relation(base), "issue_order": relation(identity),
               "record_steps": True, "budget": 8000000,
               "needs": [
                   {"relation": relation({(0, base_count)}), "rounds": 0},
                   {"relation": relation({(0, base_count + added_count)}),
                    "add_handoffs": relation(added), "rounds": 0},
               ]}
    response = run(driver, output, name, request)
    assert [answer["status"] for answer in response["answers"]] == ["proved", "proved"]
    expected = base | added
    assert fixed_pairs(response["relation"]) == expected, (name, response["relation"])
    step = response["steps"][1]
    return {"name": name, "base_pieces": base_count, "added_pieces": added_count,
            "incremental_updates": step["frontier_incremental_updates"],
            "replay_updates": step["frontier_replay_updates"],
            "fresh_pairs": len(expected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    budget = run(args.driver, args.output, "nested-budget", {
        "op": "budget_scope", "budget": 100, "a": relation(set())})
    assert budget["outer_first"] and budget["inner"]
    assert budget["outer_second"] and not budget["outer_over"], budget

    cases = [
        threshold_case(args.driver, args.output, "pieces-16", 16, 1),
        threshold_case(args.driver, args.output, "pieces-17", 17, 1),
        threshold_case(args.driver, args.output, "product-256", 16, 16),
        threshold_case(args.driver, args.output, "product-257", 16, 17),
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
