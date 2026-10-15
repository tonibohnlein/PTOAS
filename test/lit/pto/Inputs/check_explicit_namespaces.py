# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check direct explicit namespace ownership through all physical output routes."""
import json
from pathlib import Path
import sys
from check_deferred_family_output import arithmetic, executable, invoke
from check_finite_allocation import physical_check


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    optimizer, tracer, fixture = sys.argv[1:]
    optimizer, tracer = map(executable, (optimizer, tracer))
    source = Path(fixture).read_text(encoding="utf-8")
    command = [optimizer, "--mlir-disable-threading"]
    options = "profile-stages=true gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5"
    products = {}
    for name, mode in (("logical", ""), ("combined", " physical-output=true"),
                       ("saved", " deferred-output=true")):
        result = invoke([*command, "--pto-frontier-analysis=" + options + mode, "-"], source)
        products[name] = result.stdout
        profiles = [json.loads(line.split(" ", 1)[1]) for line in result.stderr.splitlines()
                    if line.startswith("frontier-control-profile ")]
        prepared = [row for row in profiles if row["stage"] == "prepared"]
        require(len(prepared) == 1 and prepared[0]["producer"] == "explicit", "explicit producer not selected")
        require(prepared[0]["endpoint_sites"] == 0 and prepared[0]["recipe_sites"] > 0,
                "explicit preparation materialized endpoints")
        require(prepared[0]["preparation"]["blocks"] == prepared[0]["preparation"]["arithmetic"] == 0,
                "explicit preparation emitted scalar code")
    require("pto.logical_family" in products["saved"] and
            arithmetic(products["saved"]) == arithmetic(source), "saved recipe constructed scalar endpoints")
    allocate = "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"
    products["split"] = invoke([*command, allocate, "-"], products["logical"]).stdout
    products["restored"] = invoke([*command, allocate, "-"], products["saved"]).stdout
    traces = {}
    for name in ("logical", "combined", "split", "restored"):
        require(products[name].count("test.exact_form_explicit") == 1,
                "missing or duplicate original function binding: " + name)
        text = products[name].replace("test.exact_form_explicit", "test.exact_form_explicit, test.trace_existing")
        report = json.loads(invoke([tracer, "--structured-trace", "-"], text).stdout)
        require(report["accepted"] and not report["trace"]["error"], "trace failed: " + name)
        traces[name] = report["trace"]
    members = {}
    for event in traces["logical"]["events"]:
        if event["kind"] not in ("set", "wait"):
            continue
        require(event["source_ordinal"] == 0 and len(event.get("members", [])) == 1,
                "explicit original-record identity changed")
        direction = (event["source_pipe"], event["target_pipe"])
        members.setdefault(direction, set()).add(event["members"][0])
    require(any(len(records) > 1 for records in members.values()), "repeated-direction regression was not exercised")
    for event in traces["logical"]["events"]:
        if event["kind"] in ("set", "wait"):
            direction = (event["source_pipe"], event["target_pipe"])
            require(event["record"] == min(members[direction]), "namespace is not the first original record")
    for name in ("combined", "split", "restored"):
        physical_check({"accepted": True, "allocated": True,
                        "trace": traces["logical"], "physical": traces[name]}, set(range(6)))
    require(traces["combined"]["events"] == traces["restored"]["events"],
            "saved and combined output communications differ")
    require(arithmetic(products["combined"]) == arithmetic(products["restored"]),
            "saved and combined construction counts differ")
    print("explicit namespaces: original members, exact cuts, split/combined/saved matching and causal reuse passed")


if __name__ == "__main__":
    main()
