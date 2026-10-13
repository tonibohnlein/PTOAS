# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check production families stay unmaterialized through allocation support."""
import json
from pathlib import Path
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(optimizer, source, options):
    result = subprocess.run([optimizer, "--mlir-disable-threading", *options, "-"],
                            input=source, capture_output=True, text=True, timeout=60, check=False)
    require(result.returncode == 0, result.stderr)
    return result


def main():
    optimizer, fixture, producer = sys.argv[1:]
    source = Path(fixture).read_text(encoding="utf-8")
    for allocation in (False, True):
        result = run(optimizer, source, ["--pto-frontier-analysis=profile-stages=true prepare-allocation=" +
                                        str(allocation).lower()])
        reports = [json.loads(line.split(" ", 1)[1]) for line in result.stderr.splitlines()
                   if line.startswith("frontier-control-profile ")]
        prepared = [report for report in reports if report["stage"] == "prepared"]
        require(any(report["producer"] == producer for report in prepared), "expected producer not exercised")
        require(bool(prepared), "no production preparation report")
        for report in prepared:
            require(report["recipe_sites"] > 0 and report["endpoint_sites"] == 0,
                    "numerical producer materialized endpoints during preparation")
            require(report["preparation"]["blocks"] == report["preparation"]["arithmetic"] == 0,
                    "family construction generated scalar MLIR")
        for report in reports:
            if report["stage"] == "normalized":
                require(report["recipe_sites"] == 0 and report["endpoint_sites"] > 0,
                        "recipe was not consumed at the materialization boundary")
        require(allocation == ("pto.cyclic_allocation" in result.stdout), "allocation capability mismatch")
        if allocation:
            run(optimizer, result.stdout, ["--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"])
    if producer == "numerical-periodic":
        # Physical-section projection destroys its temporary function after
        # remapping. No deferred recipe may keep coordinates from that function.
        section = source.replace("    scf.for %t", "    pto.section.cube {\n    scf.for %t", 1)
        section = section.replace("    return", "    }\n    return", 1)
        projected = run(optimizer, section, ["--pto-frontier-analysis"])
        require("pto.sync_context_plans" in projected.stdout, "projection path not exercised")
        run(optimizer, projected.stdout, ["--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"])
    print("family recipe pipeline: allocation precedes scalar materialization; six-ID allocation succeeds")


if __name__ == "__main__":
    main()
