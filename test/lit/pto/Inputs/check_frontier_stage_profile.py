# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare opt-in diagnostics with identical public compilation requests."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(optimizer, source, checkpoint, profile):
    options = ["analysis-only=true"] if checkpoint else []
    if profile:
        options.append("profile-stages=true")
    flag = "--pto-frontier-analysis" + ("=" + " ".join(options) if options else "")
    return subprocess.run([optimizer, "--mlir-disable-threading", flag, "-"],
                          input=source, capture_output=True, text=True, check=False, timeout=60)


def split_report(stderr):
    profiles, control, diagnostics = [], [], []
    for line in stderr.splitlines(keepends=True):
        if line.startswith("frontier-profile "):
            profiles.append(json.loads(line.removeprefix("frontier-profile ")))
        elif line.startswith("frontier-control-profile "):
            control.append(json.loads(line.removeprefix("frontier-control-profile ")))
        else:
            diagnostics.append(line)
    return profiles, control, "".join(diagnostics)


def check_control(reports):
    require([report["stage"] for report in reports] == ["prepared", "normalized", "inserted"],
            "missing control construction boundary")
    prepared, normalized, inserted = reports
    require(prepared["record_members"] == normalized["record_members"], "compaction lost provenance")
    require(normalized["endpoint_sites"] <= prepared["endpoint_sites"] + prepared["recipe_sites"],
            "compaction expanded sites")
    for report in reports:
        work = report["preparation"]
        require(0 <= work["endpoint_reachable_arithmetic"] <= work["required_arithmetic"] <= work["arithmetic"],
                "invalid preparation reachability inventory")
    require(inserted["preparation"]["arithmetic"] == 0, "insertion retained detached arithmetic")
    commands = inserted["function_control"]
    require(commands["sets"] == commands["waits"] and commands["sets"] > 0,
            "fixture command inventory missing")


def check_stages(report):
    stages = report["stages"]
    require(bool(stages), "empty profile")
    children = [0] * len(stages)
    for index, stage in enumerate(stages):
        require(stage["id"] == index + 1 and 0 <= stage["parent"] < stage["id"], "invalid stage identity")
        require(stage["request"] > 0, "missing request identity")
        if stage["parent"]:
            parent = stages[stage["parent"] - 1]
            require(stage["parent_request"] == parent["request"], "inconsistent parent request")
            children[stage["parent"] - 1] += stage["elapsed_ns"]
    aggregate_counts = {}
    for stage in stages:
        if "aggregate_calls" not in stage:
            continue
        require(stage["parent"] > 0 and stage["aggregate_calls"] > 0, "unowned aggregate")
        require(stage["first_dispatches"] == 0 and stage["cache_hits"] == 0,
                "allocation operations counted as backend dispatches")
        aggregate_counts[stage["parent"]] = aggregate_counts.get(stage["parent"], 0) + 1
    require(all(count <= 9 for count in aggregate_counts.values()), "per-query ledger growth")
    for stage, elapsed in zip(stages, children):
        require(stage["exclusive_ns"] + elapsed == stage["elapsed_ns"], "nested time counted twice")
    require(report["contract_audit"]["materializations"] <= report["contract_audit"]["refresh_requests"],
            "audit materialization without a refresh")


def main():
    require(len(sys.argv) in (3, 4), "expected optimizer, fixture, and optional regional marker")
    optimizer = shutil.which(sys.argv[1])
    require(optimizer is not None, "optimizer unavailable")
    source = Path(sys.argv[2]).read_text(encoding="utf-8")
    for checkpoint in (False, True):
        plain = run(optimizer, source, checkpoint, False)
        measured = run(optimizer, source, checkpoint, True)
        require(plain.returncode == measured.returncode == int(checkpoint), plain.stderr + measured.stderr)
        plain_reports, plain_control, plain_diagnostics = split_report(plain.stderr)
        reports, control, measured_diagnostics = split_report(measured.stderr)
        require(not plain_reports and len(reports) == 1, "profiling opt-in violated")
        require(not plain_control, "control profiling was not opt-in")
        if checkpoint:
            require(not control, "analysis-only mode prepared control")
        else:
            check_control(control)
        require(plain.stdout == measured.stdout and plain_diagnostics == measured_diagnostics,
                "profiling changed emitted commands or failure diagnostics")
        require(reports[0]["workload"] == ("exhaustive-certification" if checkpoint else "production-root"),
                "workload mislabeled")
        check_stages(reports[0])
        if not checkpoint and len(sys.argv) == 4:
            require(control[0]["recipe_sites"] > 0,
                    "sequence materialized every numerical child before allocation")
            names = {stage["stage"] for stage in reports[0]["stages"]}
            expected = {"allocation-groups-aggregate", "allocation-lanes-aggregate", "allocation-summary-aggregate"}
            require(expected <= names,
                    "regional allocation was not profiled")
    print("stage-profile public equivalence: emitted IR and checkpoint diagnostics unchanged")


if __name__ == "__main__":
    main()
