#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare two completed native corpus arms, serially, without recompilation.

Reuse the benchmark's MLIR verification and exact payload/SSA projection. Keep
static mechanisms separate; placement differences are not performance verdicts.
Optional diagnostics rerun only the selected pass-entry functions, not builds.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

from compare_native import digest, run
sys.path.insert(0, str(Path(__file__).resolve().parent / "performance"))
from run_qwen_additions import analyze, hashed, pass_entry


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--pto-test-opt", type=Path, help="Also retain native pass diagnostics for every row")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_root.resolve()))
    manifests = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    inventory = {row["case_id"]: row for row in json.loads(args.inventory.read_text())}
    reports = [json.loads((arm / "results.json").read_text()) for arm in (args.original, args.current)]
    parser_native = args.python_root / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so"
    if digest(parser_native) != reports[1]["native_sha256"]:
        raise ValueError("measurement parser does not match the recorded current compiler")
    for report in reports:
        if (report["manifest_sha256"] != digest(args.manifest) or not report.get("native_unchanged")
                or [r["case_id"] for r in report["rows"]] != [r["case_id"] for r in manifests]):
            raise ValueError("incomplete, changed, or mismatched compiler population")
    args.output.mkdir(parents=True, exist_ok=False)
    out = {"manifest_sha256": digest(args.manifest), "inventory_sha256": digest(args.inventory),
           "row_count": len(manifests), "unique_inputs": len({r["sha256"] for r in manifests}),
           "parent_seeds": len({r["parent_seed"] for r in manifests}),
           "arms": [{k: v for k, v in r.items() if k != "rows"} for r in reports],
           "device_execution": "not-run", "rows": [], "projection_failures": []}
    totals, statuses, barriers, counters = Counter(), Counter(), Counter(), Counter()
    cache = {}

    def inspect(path):
        key = digest(path)
        if key not in cache:
            data = analyze(path)
            cache[key] = {"mechanisms": data["mechanisms"],
                          "status_attributes": data["status_attributes"],
                          "sync_control": data["sync_control"],
                          "hashes": {k: hashed(data[k]) for k in
                                     ("payload", "placements", "allocations", "views", "abi")}}
        return cache[key]

    for i, (manifest, left, right) in enumerate(zip(manifests, reports[0]["rows"], reports[1]["rows"])):
        case = manifest["case_id"]
        if any(r["input_sha256"] != manifest["sha256"] for r in (left, right)):
            raise ValueError(f"input drift: {case}")
        seed = inventory[manifest["parent_seed"]]
        row = {**manifest, "entry_source": seed["source"], "entry": seed["entry"], "family": seed["family"]}
        for arm, source, record in (("original", args.original, left), ("current", args.current, right)):
            passed = all(record["runs"][kind]["status"] == "pass" for kind in ("pto", "cpp"))
            row[arm] = {"compiled": passed,
                        "pto_status": record["runs"]["pto"]["status"],
                        "cpp_status": record["runs"]["cpp"]["status"],
                        "process_seconds": sum(r["seconds"] for r in record["runs"].values())}
            totals[f"{arm}_compiled"] += passed
            for kind in ("pto", "cpp"):
                totals[f"{arm}_{kind}_passed"] += record["runs"][kind]["status"] == "pass"
            if not passed:
                row[arm]["errors"] = sorted({line.strip() for kind in ("pto", "cpp")
                    for line in (source / case / f"{kind}.stderr").read_text().splitlines()
                    if "error:" in line or "Error:" in line or "Assertion" in line})
                continue
            try:
                row[arm].update(inspect(source / case / "output.pto"))
            except Exception as error:
                row[arm]["projection_error"] = str(error)
                out["projection_failures"].append(f"{case}/{arm}: {error}")
        if row["original"]["compiled"] and not row["current"]["compiled"]:
            totals["new_compile_failure"] += 1
        if all("hashes" in row[arm] for arm in ("original", "current")):
            a, b = row["original"], row["current"]
            row["equal"] = {key: a["hashes"][key] == b["hashes"][key] for key in a["hashes"]}
            totals["paired_verified"] += 1
            for key, equal in row["equal"].items():
                totals[key + "_identical"] += equal
            am, bm = a["mechanisms"], b["mechanisms"]
            totals["same_inventory"] += am == bm
            for key in ("sets", "waits", "PIPE_ALL"):
                direction = "fewer" if bm[key] < am[key] else "more" if bm[key] > am[key] else "same"
                totals[f"{key}_{direction}"] += 1
            for pipe in am["named"].keys() | bm["named"].keys():
                delta = bm["named"].get(pipe, 0) - am["named"].get(pipe, 0)
                barriers[f"{pipe}_{'more' if delta > 0 else 'fewer' if delta < 0 else 'same'}"] += 1
            for attributes in b["status_attributes"]:
                statuses[attributes.get("pto.insert_sync.status", "absent").strip('"')] += 1
                for key, value in attributes.items():
                    match = re.fullmatch(r"(\d+) : i\d+", value)
                    if match and int(match[1]):
                        counters[key + ":functions"] += 1
                        counters[key + ":sum"] += int(match[1])
            if not all(row["equal"][k] for k in ("payload", "allocations", "views", "abi")):
                save(args.output / f"{case}.projection-difference.json", {
                    arm: analyze(source / case / "output.pto") for arm, source in
                    (("original", args.original), ("current", args.current))})
        if args.pto_test_opt and row["current"]["compiled"]:
            directory = args.output / case
            directory.mkdir()
            try:
                before = directory / "before.pto"
                before.write_text(pass_entry((args.current / case / "pto.stderr").read_text()))
                command = [str(args.pto_test_opt.resolve()), str(before), "--mlir-disable-threading",
                           "--mlir-print-op-on-diagnostic=false",
                           "--pto-insert-sync=buffer-generations=true defer-same-pipe=true "
                           "mmad-chains=true gm-alias=assume-disjoint-arguments effect-coverage=report",
                           "-o", str(directory / "native.pto")]
                code, stdout, stderr, seconds = run(command, 120)
                (directory / "native.stderr").write_text(stderr)
                row["diagnostic"] = {"returncode": code, "seconds": seconds, "command": command,
                                     "remarks": [line.strip() for line in stderr.splitlines() if "remark:" in line]}
                if code == 0:
                    row["diagnostic"]["mechanisms_match_cli"] = (
                        inspect(directory / "native.pto")["mechanisms"] == row["current"]["mechanisms"])
            except Exception as error:
                row["diagnostic"] = {"unavailable": str(error)}
        out["rows"].append(row)
        if (i + 1) % 100 == 0 or i + 1 == len(manifests):
            print(f"{i + 1}/{len(manifests)} verified={totals['paired_verified']} "
                  f"payload_equal={totals['payload_identical']}", flush=True)
    out.update(totals=dict(totals), status_counts=dict(statuses), named_barrier_changes=dict(barriers),
               counters=dict(counters))
    save(args.output / "results.json", out)
    with (args.output / "table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case_id", "entry_source", "entry", "generated_path", "original_compiled", "current_compiled",
                         "original_sets", "original_waits", "original_named", "original_PIPE_ALL",
                         "current_sets", "current_waits", "current_named", "current_PIPE_ALL",
                         "payload_identical", "allocation_identical", "views_identical", "abi_identical", "placements_identical"])
        for row in out["rows"]:
            values = [row[k] for k in ("case_id", "entry_source", "entry", "generated_path")]
            values += [row[a]["compiled"] for a in ("original", "current")]
            for arm in ("original", "current"):
                m = row[arm].get("mechanisms", {})
                values += [m.get("sets", ""), m.get("waits", ""), json.dumps(m.get("named", {}), sort_keys=True), m.get("PIPE_ALL", "")]
            values += [row.get("equal", {}).get(k, "") for k in ("payload", "allocations", "views", "abi", "placements")]
            writer.writerow(values)


if __name__ == "__main__":
    main()
