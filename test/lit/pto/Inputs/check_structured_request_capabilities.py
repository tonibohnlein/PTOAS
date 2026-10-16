# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Keep logical trace requests separate from optional physical allocation."""
import json
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import tempfile

from check_fixed_protocol_capacity import same_fixed_protocol
from check_logical_insertion import invoke, recognized, validate
from check_physical_allocation import check_physical


def numeric_trace(trace):
    # Convert original IVs to the existing oracle's visit numbering using this
    # fixture's original lower=1, step=2, upper=8, never generated control.
    result = deepcopy(trace)
    for event in result["events"]:
        if event["kind"] == "payload":
            assert len(event["coordinates"]) == 1
            coordinate = event["coordinates"][0]
            assert coordinate in (1, 3, 5, 7)
            event["ordinal"] = (coordinate - 1) // 2
    return result


def main():
    tool = shutil.which(sys.argv[1])
    assert tool
    source = Path(sys.argv[2]).read_text()
    assert "scf.for %t = %one to %n step %two" in source and "array<i64: 8>" in source
    assert "%one = arith.constant 1 : index" in source and "%two = arith.constant 2 : index" in source
    attribute = ", test.eligible_ids = array<i64: 1, 3>"
    assert source.count(attribute) == 1
    variants = {"logical": source.replace(attribute, ""), "physical": source,
                "empty": source.replace(attribute, ", test.eligible_ids = array<i64>")}
    reports = {}
    # Only three small, short-lived text fixtures are kept in this directory.
    with tempfile.TemporaryDirectory(prefix="structured-request-") as directory:
        path = Path(directory) / "case.pto"
        path.write_text(source)
        template = recognized(tool, path)
        for name, text in variants.items():
            path.write_text(text)
            report = json.loads(invoke(tool, "--structured-trace", path))
            assert report["accepted"], report
            evidence = report["preparation_request"]
            assert evidence["mode"] == "fallback" and evidence["synchronization"]
            assert evidence["logical_preparations"] > 0 and not evidence["completion"]
            assert evidence["allocation_requested"] == (name != "logical")
            assert evidence["allocation_exports"] == (0 if name == "logical" else 1)
            assert evidence["allocation_certificate"] == (name != "logical")
            validate(template, numeric_trace(report["trace"]))
            reports[name] = report
        logical, physical, empty = (reports[name] for name in ("logical", "physical", "empty"))
        assert "allocated" not in logical and "physical" not in logical
        assert {report["selected_logical_backend"] for report in reports.values()} == {
            logical["selected_logical_backend"]}
        assert same_fixed_protocol(logical["trace"], physical["trace"])
        assert same_fixed_protocol(logical["trace"], empty["trace"])
        check_physical(template, {"allocated": physical["allocated"],
                                  "logical": numeric_trace(physical["trace"]),
                                  "physical": numeric_trace(physical["physical"])}, {1, 3})
        assert not empty["allocated"] and empty["allocation_unchanged_on_failure"]
    print("structured requests: logical-only, physical and empty-pool capabilities passed")


if __name__ == "__main__":
    main()
