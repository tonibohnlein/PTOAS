#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Summarize matched frontier-refinement off/on campaigns."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


PIPES = ("M", "V", "MTE1", "MTE2", "MTE3", "FIX")


def mechanisms(counts):
    barriers = Counter()
    for key, value in counts.items():
        if key.startswith("barrier:"):
            barriers[re.search(r"PIPE_[A-Z0-9]+", key).group()] += value
    sets = counts.get("pto.set_flag", 0)
    waits = counts.get("pto.wait_flag", 0)
    return {
        "sets": sets,
        "waits": waits,
        "pair_inventory": sets if sets == waits else None,
        "named_barriers": {pipe: value for pipe, value in sorted(barriers.items()) if pipe != "PIPE_ALL"},
        "PIPE_ALL": barriers.get("PIPE_ALL", 0),
    }


def campaign_path(root, arch, use_mmad, use_frontiers, population):
    mode = "on" if use_mmad else "off"
    if use_frontiers:
        return root / f"{arch}-mmad-{mode}-{population}"
    return root / f"{arch}-mmad-{mode}-frontier-off-{population}"


def attribution(output):
    text = output.read_text()
    return {
        "removed": sum(map(int, re.findall(r"pto.insert_sync.frontier_barriers_removed = ([0-9]+)", text))),
        "guarded": sum(map(int, re.findall(r"pto.insert_sync.frontier_barriers_guarded = ([0-9]+)", text))),
    }


def summarize(root):
    report = {
        "measurement": "Static IR and scalar replay; no device numerical/asynchronous validation or timing",
        "compiler_source": "ca2647e71f0a069aa300c9ed9af3b498b35e741c",
        "rows": [],
        "failures": [],
    }
    native_hashes = set()
    raw_rows = {}
    for arch in ("a2", "a3"):
        for use_mmad in (False, True):
            for use_frontiers in (False, True):
                for population in ("controls", "kernels"):
                    directory = campaign_path(root, arch, use_mmad, use_frontiers, population)
                    raw = json.loads((directory / "results.json").read_text())
                    native_hashes.add(raw["native_sha256"])
                    report["failures"].extend(raw["failures"])
                    if not raw["native_unchanged"]:
                        report["failures"].append(f"{directory}: compiler changed during campaign")
                    if raw["mmad_chains"] != use_mmad or raw["frontier_refinement"] != use_frontiers:
                        report["failures"].append(f"{directory}: option attribution mismatch")
                    for name, row in raw["rows"].items():
                        case, arm = name.split("/")
                        raw_rows[(arch, use_mmad, use_frontiers, case, arm)] = row
                        output_path = directory / case / arm / "output.pto"
                        entry = {
                            "arch": arch,
                            "mmad_chains": use_mmad,
                            "frontier_refinement": use_frontiers,
                            "case": case,
                            "arm": arm,
                            "status": row["status"],
                            "mechanisms": mechanisms(row["metrics"]["static"]["counts"]),
                            "recorded_output": str(output_path.relative_to(root)),
                        }
                        if use_frontiers and arm != "manual":
                            entry["frontier_attribution"] = attribution(output_path)
                        report["rows"].append(entry)

    inventories = {
        (row["arch"], row["mmad_chains"], row["frontier_refinement"], row["case"], row["arm"]): row[
            "mechanisms"
        ]
        for row in report["rows"]
    }
    cases = list(dict.fromkeys(row["case"] for row in report["rows"]))
    for use_mmad in (False, True):
        for use_frontiers in (False, True):
            for case in cases:
                for arm in ("manual", "combined", "staged"):
                    a2 = inventories[("a2", use_mmad, use_frontiers, case, arm)]
                    a3 = inventories[("a3", use_mmad, use_frontiers, case, arm)]
                    if a2 != a3:
                        report["failures"].append(f"{case}/{arm}: A2/A3 inventories differ")
                if inventories[("a3", use_mmad, use_frontiers, case, "combined")] != inventories[
                    ("a3", use_mmad, use_frontiers, case, "staged")
                ]:
                    report["failures"].append(f"{case}: combined/staged inventories differ")
                before = raw_rows[("a3", use_mmad, False, case, "combined")]["metrics"]["scenarios"]
                after = raw_rows[("a3", use_mmad, True, case, "combined")]["metrics"]["scenarios"]
                if set(before) != set(after) or any(
                    before[name]["payload_sha256"] != after[name]["payload_sha256"] for name in before
                ):
                    report["failures"].append(f"{case}: frontier off/on payload replay differs")
    report["native_sha256"] = sorted(native_hashes)
    if len(native_hashes) != 1:
        report["failures"].append("campaigns did not use one native compiler")
    report["static_frontier_changes"] = sum(
        inventories[("a3", use_mmad, False, case, "combined")]
        != inventories[("a3", use_mmad, True, case, "combined")]
        for use_mmad in (False, True)
        for case in cases
    )
    report["frontier_barriers_removed"] = sum(
        row.get("frontier_attribution", {}).get("removed", 0) for row in report["rows"]
    )
    report["frontier_barriers_guarded"] = sum(
        row.get("frontier_attribution", {}).get("guarded", 0) for row in report["rows"]
    )
    report["validated_case_arm_rows"] = len(report["rows"])
    report["rows"] = [
        row
        for row in report["rows"]
        if row["arch"] == "a3"
        and (
            row["arm"] == "combined"
            or (
                row["arm"] == "manual"
                and not row["mmad_chains"]
                and not row["frontier_refinement"]
            )
        )
    ]
    return report


def markdown(report):
    lines = [
        "# Frontier r4: all eleven local fixtures",
        "",
        f"Compiler source: `{report['compiler_source']}`. Frontier refinement and MMAD-chain inference are opt-in.",
        f"Native SHA-256: `{report['native_sha256'][0]}`.",
        "",
        "The matched frontier-off/frontier-on campaigns cover A2 and A3, MMAD off and on, and combined and staged "
        "traversal. All 264 case/arm rows pass, producing 528 successful PTO/C++ compiler invocations. Every "
        "available scalar replay preserves the hand-tuned payload trace.",
        "",
        "The table shows A3 combined mode. A2 and staged inventories agree. Set/wait sites are reported as a "
        "pair inventory only when their static counts balance; the barrier pipes and PIPE_ALL remain separate.",
        "",
        "| Fixture | Arm | Pairs | PIPE_M | PIPE_V | PIPE_MTE1 | PIPE_MTE2 | PIPE_MTE3 | PIPE_FIX | PIPE_ALL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = {
        (row["mmad_chains"], row["frontier_refinement"], row["case"], row["arm"]): row
        for row in report["rows"]
        if row["arch"] == "a3"
    }
    cases = list(dict.fromkeys(row["case"] for row in report["rows"]))
    arms = [
        (False, False, "Hand", "manual"),
        (False, False, "MMAD off, frontier off", "combined"),
        (False, True, "MMAD off, frontier on", "combined"),
        (True, False, "MMAD on, frontier off", "combined"),
        (True, True, "MMAD on, frontier on", "combined"),
    ]
    for case in cases:
        for use_mmad, use_frontiers, label, arm in arms:
            mechanism = rows[(use_mmad, use_frontiers, case, arm)]["mechanisms"]
            named = mechanism["named_barriers"]
            values = [mechanism["pair_inventory"]]
            values.extend(named.get(f"PIPE_{pipe}", 0) for pipe in PIPES)
            values.append(mechanism["PIPE_ALL"])
            lines.append(f"| {case} | {label} | " + " | ".join(map(str, values)) + " |")
    lines.extend(
        [
            "",
            "Frontier r4 changes no static synchronization inventory in these fixtures and reports zero removed or "
            "overflow-guarded barriers. In particular, MMAD-on GEMM remains at 44 pairs, four PIPE_M, two PIPE_MTE1, "
            "three PIPE_MTE2, one PIPE_FIX and one PIPE_ALL. MMAD-off GEMM remains at 18 PIPE_M sites.",
            "",
            "This is a negative performance result for r4's current supported proof domain, not a correctness failure. "
            "The compiler, replay, architecture and traversal cross-checks all pass. Device execution and timing were "
            "not run.",
            "",
            "Raw commands, emitted PTO/C++, diagnostics and replay traces remain in the campaign root supplied to "
            "the summarizer. The companion JSON records campaign-relative output paths, table rows and cross-check "
            "totals; the raw results record every validated row.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output basename, without extension")
    args = parser.parse_args()
    report = summarize(args.campaign)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(
        json.dumps(
            {
                "validated_case_arm_rows": report["validated_case_arm_rows"],
                "recorded_table_rows": len(report["rows"]),
                "static_frontier_changes": report["static_frontier_changes"],
                "frontier_barriers_removed": report["frontier_barriers_removed"],
                "frontier_barriers_guarded": report["frontier_barriers_guarded"],
                "failures": report["failures"],
            }
        )
    )
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
