#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate S6 diagnostic accounting. No solver, native importer, or hardware oracle.

Use on stderr produced with PTOAS_STRUCTURED_PLAN_JSON=1. Diagnostic mode runs
additional per-handoff proof queries; do not mix those timings into benchmarks.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "oahs.s7.plan.v2"
# v2 only adds sequence_bridges, so a v1 record still validates. Both are
# accepted: the emitter produces v2 while existing fixtures carry v1.
ACCEPTED_SCHEMAS = ("oahs.s6.plan.v1", SCHEMA)
PREFIX = "OAHS_PLAN "


def parse_reports(text):
    reports = []
    for line in text.splitlines():
        if line.startswith(PREFIX):
            value = json.loads(line[len(PREFIX):])
            if not isinstance(value, dict) or value.get("schema") not in ACCEPTED_SCHEMAS:
                raise ValueError("unsupported structured diagnostic schema")
            reports.append(value)
    if not reports:
        raise ValueError("no S6 diagnostic: a missing report is not an empty plan")
    return reports


def lane(value):
    if value.get("core") not in ("AIC", "AIV") or value.get("pipe") not in (
            "S", "V", "M", "MTE1", "MTE2", "MTE3", "FIX"):
        raise ValueError("invalid lane in diagnostic")
    return value["core"] + ":" + value["pipe"]


def plain(action):
    return (action["participation"] == "every" and int(action["ordinal_distance"]) == 0
            and int(action["existence_residue"]) == 0 and action["invocation"] == 0
            and not action["invocation_body_guard"] and int(action["invocation_residue"]) == 0)


def signature(action):
    return (action["kind"], action["after"], lane(action["source_lane"]),
            lane(action["target_lane"]), action["key"])


def validate_report(report):
    if report.get("schema") not in ACCEPTED_SCHEMAS or report.get("producer") != "structured":
        raise ValueError("unexpected diagnostic producer/schema")
    if not isinstance(report.get("new_hardware_elision"), bool) or report.get("retirement_sites") != 1:
        raise ValueError("unexpected hardware premise or retirement policy")
    totals = Counter(retirement=1)
    directions = Counter()
    units = []
    unit_ids = set()
    intrinsic_count = 0
    for unit in report["units"]:
        if unit["unit"] in unit_ids:
            raise ValueError("duplicate unit")
        unit_ids.add(unit["unit"])
        atoms = {a["atom"]: a for a in unit["atoms"]}
        actions = {a["action"]: a for a in unit["actions"]}
        if len(atoms) != len(unit["atoms"]) or len(actions) != len(unit["actions"]):
            raise ValueError("duplicate atom/action")
        seen = Counter()
        prior, emitted = defaultdict(list), defaultdict(list)
        for a in sorted(actions.values(), key=lambda x: int(x["planner_order"])):
            if a["anchor"] not in atoms:
                raise ValueError("missing original atom")
            prior[a["anchor"], a["after"]].append(a["action"])
        merges = 0
        for number, site in enumerate(unit["sites"]):
            if site["site"] != number or len(site["actions"]) not in (1, 2):
                raise ValueError("invalid site identity/population")
            try:
                members = [actions[i] for i in site["actions"]]
            except KeyError as error:
                raise ValueError("site refers to absent action") from error
            if len(members) == 2:
                a, b = members
                aa, bb = atoms[a["anchor"]], atoms[b["anchor"]]
                if (int(unit["period"]) != 1 or unit["wrapper_count"] != 0
                        or aa["original_phase"] != bb["original_phase"]
                        or aa["role"] != "initial" or bb["role"] != "steady"
                        or signature(a) != signature(b) or not plain(a) or not plain(b)):
                    raise ValueError("non-equivalent startup site members")
                merges += 1
            kind = members[0]["kind"]
            if kind not in ("set", "wait", "barrier"):
                raise ValueError("unknown command kind")
            totals[kind] += 1
            directions[lane(members[0]["source_lane"]) + "->" + lane(members[0]["target_lane"])] += 1
            for a in members:
                seen[a["action"]] += 1
                emitted[a["anchor"], a["after"]].append(a["action"])
        if seen != Counter({i: 1 for i in actions}) or prior != emitted:
            raise ValueError("site partition lost/duplicated an action or reordered a phase FIFO")
        contract = unit.get("hardware_contract", "conservative")
        if contract not in ("conservative", "a2a3-mmad-acc-v1"):
            raise ValueError("unknown hardware contract")
        for requirement in unit["requirements"]:
            intrinsic = requirement.get("intrinsic_accumulator_order", False)
            if not isinstance(intrinsic, bool):
                raise ValueError("intrinsic verdict must be Boolean")
            if intrinsic:
                if (contract != "a2a3-mmad-acc-v1" or
                        requirement.get("property") != "accumulator-update"):
                    raise ValueError("intrinsic result has the wrong property/contract")
                if any(requirement.get(key) not in atoms for key in ("source", "target")):
                    raise ValueError("intrinsic result refers to a missing atom")
                for key in ("source", "target"):
                    if lane(atoms[requirement[key]]["lane"]) != "AIC:M":
                        raise ValueError("intrinsic result is not same-M accumulator ordering")
                intrinsic_count += 1
        requirement_ids = {r["requirement"] for r in unit["requirements"]}
        for audit in unit["deletion_audit"]:
            if not set(audit["completion_unproved_without"]) <= requirement_ids:
                raise ValueError("audit references a missing original requirement")
        units.append(dict(unit=unit["unit"], virtual_actions=len(actions),
                          static_sites=len(unit["sites"]), coalesced_sites=merges,
                          original_requirements=len(requirement_ids),
                          reentrant=bool(unit["wrapper_count"])))
    if (int(report.get("intrinsic_accumulator_requirements", 0)) != intrinsic_count or
            report["new_hardware_elision"] != bool(intrinsic_count)):
        raise ValueError("hardware accounting disagrees with original requirements")
    total = sum(totals.values())
    if total != int(report["generated_static_sites_including_retirement"]):
        raise ValueError("static site accounting disagrees with emitted groups")
    return dict(status="accounting-checked", static_sites=total, counts=dict(totals),
                intrinsic_accumulator_requirements=intrinsic_count,
                command_sites_by_direction=dict(sorted(directions.items())), units=units,
                semantic_verification="performed by native checker; NOT by this accounting utility")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stderr", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [validate_report(r) for r in parse_reports(args.stderr.read_text())]
    text = json.dumps(summaries, indent=2) + "\n"
    if args.output:
        # Do not overwrite an earlier qualification result.
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
