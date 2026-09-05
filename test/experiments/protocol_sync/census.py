# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Aggregate C7.1 evidence with explicit denominators and structural-only labels."""

import hashlib
import json
import math
from collections import Counter, defaultdict

from acceptance import summarize_acceptance
from records import ids


NON_ADDITIVE = {"max_event_domain_pressure", "maximum_event_id", "interpreter_peak_states",
                "storage_projection_maximum_atoms_per_access", "storage_projection_maximum_atoms_per_component"}


def distribution(values):
    """Use nearest-rank quantiles and report the population explicitly."""
    ordered = sorted(values)
    result = {f"p{percent}": ordered[max(0, math.ceil(len(ordered) * percent / 100) - 1)] if ordered else 0
              for percent in (50, 95)}
    return dict(result, maximum=ordered[-1] if ordered else 0, count=len(ordered), total=sum(ordered))


def topology_hash(function):
    """Hash structured control and phase resources, excluding names and addresses."""
    topology = {
        "regions": [{key: region.get(key) for key in ("id", "kind", "parent", "cardinality", "arm", "elements")}
                    for region in function["regions"]],
        "phases": [{key: phase.get(key) for key in ("id", "region", "core", "pipe", "guard", "loops")}
                   for phase in function["phases"]],
    }
    return hashlib.sha256(json.dumps(topology, sort_keys=True).encode()).hexdigest()


def first_control_gate(function):
    """Refine E's reported control rejection using its actual region scan order."""
    seen_loop = False
    for region in function["regions"]:
        kind = region["kind"]
        if kind in {"choice", "alternative", "physical-section"}:
            return f"control.{kind}"
        if kind != "loop":
            continue
        if seen_loop:
            return "control.multiple-loops"
        if region["cardinality"] != "zero-or-more":
            return "control.cardinality"
        if region.get("op") != "scf.for":
            return "control.loop-operation"
        seen_loop = True
    return "control.unresolved" if seen_loop else "control.no-loop"


def first_proof(function, transition):
    reason = transition["checkpoint-e-reason"]
    if reason == "unsupported-control-flow":
        return first_control_gate(function)
    if reason == "schedule-failure" and function["failures"]:
        failure = function["failures"][0]
        return f"schedule.{failure['reason']}:{failure['op']}"
    return reason


def population(records, key):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    return {name: {"records": len(group), "function_instances": len({(r["case_id"], r["function"]) for r in group}),
                   "function_names": len({r["function"] for r in group}),
                   "topologies": len({r["topology_hash"] for r in group})}
            for name, group in sorted(grouped.items())}


def recurring_records(rows):
    result = []
    for row in rows:
        for function in row["functions"]:
            for transition in function["transitions"]:
                if transition.get("checkpoint-e") != "rejected":
                    continue
                result.append({"case_id": row["case_id"], "function": function["name"],
                               "transition": transition["id"], "topology_hash": topology_hash(function),
                               "first_proof": first_proof(function, transition),
                               "e_reason": transition["checkpoint-e-reason"]})
    return result


def classify_track(function, track):
    """Classify observable use shapes, not asynchronous lifetime correctness."""
    family_ids = ids(track["families"])
    families = {family["id"]: family for family in function["families"]}
    roots = {families.get(family, {}).get("root") for family in family_ids}
    scopes = {families.get(family, {}).get("root-lexical-scope") for family in family_ids}
    if len(roots) == 1 and None not in roots:
        return "equivalent-root-views"
    if None in roots or None in scopes or "#4294967295" in scopes:
        return "unresolved-root-or-scope"
    if len(scopes) > 1:
        return "cross-scope-numeric-coincidence-candidate"
    occurrences = [item for item in function["occurrences"] if item["track"] == track["id"]]
    if any(item.get("guard") != "[]" or item.get("loops") != "[]" for item in occurrences):
        return "unresolved-control-or-iteration"
    bounds = defaultdict(list)
    for item in occurrences:
        bounds[item["family"]].append(ids(item["phase"])[0])
    intervals = sorted((min(points), max(points)) for points in bounds.values())
    if len(intervals) != len(family_ids):
        return "unresolved-occurrences"
    if all(left[1] < right[0] for left, right in zip(intervals, intervals[1:])):
        return "sequential-use-candidate"
    return "overlapping-use-candidate"


def projection_details(rows):
    rejected = []
    tracks = []
    expansion = []
    for row in rows:
        for function in row["functions"]:
            accesses = {access["id"]: access for access in function["accesses"]}
            phases = {phase["id"]: phase for phase in function["phases"]}
            for item in function["unprojected"]:
                access = accesses.get(item["access"], {})
                phase = phases.get(access.get("phase"), {})
                rejected.append(dict(item, case_id=row["case_id"], function=function["name"],
                                     corpus=row.get("corpus", "unspecified"), source=row["source"],
                                     op=phase.get("op", "unknown"), space=access.get("space", "unknown"),
                                     root=access.get("root", "unknown")))
            membership = Counter(item["access"] for item in function["occurrences"])
            expansion.extend(membership.values())
            for track in function["tracks"]:
                if len(ids(track["families"])) > 1:
                    tracks.append({"case_id": row["case_id"], "function": function["name"],
                                   "track": track["id"], "classification": classify_track(function, track)})
    return {"unprojected_records": rejected, "unprojected_by_op": dict(Counter(r["op"] for r in rejected)),
            "unprojected_by_space": dict(Counter(r["space"] for r in rejected)),
            "unprojected_by_reason": dict(Counter(r["reason"] for r in rejected)),
            "unprojected_by_corpus": dict(Counter(r["corpus"] for r in rejected)),
            "multi_family_records": tracks,
            "multi_family_shapes": dict(Counter(t["classification"] for t in tracks)),
            "atoms_per_projected_access": distribution(expansion)}


def summarize(rows, mode):
    functions = [function for row in rows for function in row["functions"]]
    statistics = [function["statistics"] for function in functions if "statistics" in function]
    sums = Counter()
    maxima = {}
    histograms = defaultdict(Counter)
    timings = defaultdict(list)
    for record in statistics:
        for key, value in record.get("time_us", {}).items():
            timings[key].append(value)
        for key, value in record["counts"].items():
            if key in NON_ADDITIVE:
                maxima[key] = max(maxima.get(key, value), value)
            else:
                sums[key] += value
        for key, value in record.items():
            if isinstance(value, dict) and key not in {"counts", "time_us"}:
                histograms[key].update(value)
    transitions = [t for function in functions for t in function["transitions"]]
    candidates = [c for function in functions for c in function["candidates"]]
    recurring = recurring_records(rows)
    verdicts = [function["verdict"] for function in functions if "verdict" in function]
    totals = {"rows": len(rows), "failed_rows": sum(row["return_code"] != 0 for row in rows),
              "functions": len(statistics), "missing_statistics_rows": sum(
                  not row["functions"] or any("statistics" not in f for f in row["functions"]) for row in rows),
              "transitions": len(transitions), "candidates": len(candidates), "recurring_rejected": len(recurring)}
    result = {"mode": mode, "totals": totals, "counts_sum": dict(sums), "counts_max": maxima,
              "time_us_per_function": {key: distribution(values) for key, values in timings.items()},
              "histograms": dict(histograms), "recurring_by_first_proof": population(recurring, "first_proof"),
              "recurring_by_e_reason": population(recurring, "e_reason"), "recurring_records": recurring,
              "target_queries": dict(Counter(t["target-query"] for t in transitions)),
              "concrete_verdicts": dict(Counter(v["status"] for v in verdicts)),
              "concrete_first_failed_stage": dict(Counter(v["first-failed-stage"] for v in verdicts)),
              "programs_all_concrete_accepted": sum(bool(row["functions"]) and all(
                  f.get("verdict", {}).get("status") == "accepted" for f in row["functions"]) for row in rows)}
    if mode in {"storage", "acceptance"}:
        result.update(projection_details(rows))
    if mode == "acceptance":
        result["acceptance"] = summarize_acceptance(rows)
        totals["failed_acceptance_probes"] = sum(
            row["empty_world"]["return_code"] != 0 for row in rows)
        totals["failed_acceptance_probes"] += result["acceptance"]["strict_status"].get(
            "incomplete-or-invocation-error", 0)
    return result
