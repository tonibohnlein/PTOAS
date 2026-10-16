# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Measure retained guarded preparation and check filter/failure isolation."""
import json
from pathlib import Path
import sys
from check_deferred_family_output import executable, invoke
from check_finite_family_pipeline import render
from check_finite_guarded_analysis import compute, load


def main():
    tool, fixture = sys.argv[1:]
    tool = executable(tool)
    template = Path(fixture).read_text(encoding="utf-8")
    reports = []
    for size in (8, 16, 32, 64):
        body = "\n".join(load(f"L{i}") + "scf.if %g { " + compute(f"C{i}") + " }" for i in range(size))
        source = render(template, body).replace("test.trace_arguments =",
                        'test.finite_qualification = "guarded-preparation", test.trace_arguments =')
        report = json.loads(invoke([tool, "--gm-alias=may-not-alias", "--finite-guarded-analysis", "-"], source).stdout)
        samples = report["samples"]
        if len(samples) != 7 or not samples[5]["refused"]:
            raise RuntimeError("missing failed-filter preparation probe")
        for sample in samples:
            if sample["requests"] != 1 or sample["inventories"] != 1:
                raise RuntimeError("preparation inventory measurement disagrees")
        if samples[4]["publication_requests"] or samples[4]["recipe_builds"] or samples[4]["families"]:
            raise RuntimeError("false filter constructed active publications")
        for request in (1, 2, 6):
            if samples[request]["families"] != samples[0]["families"]:
                raise RuntimeError("filter or failure contaminated unfiltered preparation")
        report["size"] = size
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
