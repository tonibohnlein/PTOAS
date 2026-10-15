# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check saved recipes, matching identities and causal reuse across processes."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from check_finite_allocation import physical_check


def invoke(command, source, succeeds=True):
    result = subprocess.run(command, input=source, capture_output=True,
                            text=True, timeout=60, check=False)
    if (result.returncode == 0) != succeeds:
        raise RuntimeError(result.stderr or "unexpected pipeline result")
    return result


def executable(name):
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"missing executable: {name}")
    return str(Path(resolved).resolve(strict=True))


def arithmetic(source):
    return len(re.findall(r"^\s*%[^\n=]+ = arith\.", source, re.MULTILINE))


def main():
    optimizer, trace_tool, compiler, fixture = sys.argv[1:]
    optimizer, trace_tool, compiler = map(executable, (optimizer, trace_tool, compiler))
    source = Path(fixture).read_text(encoding="utf-8")
    command = [optimizer, "--mlir-disable-threading"]
    options = "allocation-ids=0,1,2,3,4,5"
    saved = invoke([*command, "--pto-frontier-analysis=" + options + " deferred-output=true", "-"], source).stdout
    if "pto.logical_family" not in saved or "pto.logical_set" in saved or arithmetic(saved) != arithmetic(source):
        raise RuntimeError("deferred production constructed scalar endpoints")
    # A different process sees only serialized IR, with no producer callbacks.
    split = invoke([*command, "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5", "-"], saved).stdout
    combined = invoke([*command, "--pto-frontier-analysis=" + options + " physical-output=true", "-"], source).stdout
    logical = invoke([*command, "--pto-frontier-analysis=" + options, "-"], source).stdout
    if "pto.logical_" in split or "pto.deferred_family" in split or arithmetic(split) != arithmetic(combined):
        raise RuntimeError("saved and combined routes disagree on live construction")
    for bound in (0, 1, 2, 5, 8, 11):
        reports = []
        for text in (logical, split, combined):
            if text.count("test.trace_arguments = array<i64: 8>") != 1:
                raise RuntimeError("expected exactly one original trace invocation binding")
            text = text.replace("test.trace_arguments = array<i64: 8>",
                                f"test.trace_existing, test.trace_arguments = array<i64: {bound}>")
            report = json.loads(invoke([trace_tool, "--structured-trace", "-"], text).stdout)
            if not report["accepted"] or report["trace"]["error"]:
                raise RuntimeError("unfolded physical trace failed")
            reports.append(report["trace"])
        for physical in reports[1:]:
            physical_check({"accepted": True, "allocated": True,
                            "trace": reports[0], "physical": physical}, set(range(6)))
        if reports[1]["events"] != reports[2]["events"]:
            raise RuntimeError("saved and combined unfolded communications differ")
    # The saved fixtures already use the final tile-buffer form. Match their
    # level-three compilation route so unrelated level-two checks do not mask
    # the required unallocated-artifact diagnostic.
    refused = invoke([compiler, "--pto-arch=a3", "--pto-level=level3", "-"], saved, succeeds=False)
    if "logical_family" not in refused.stderr and "deferred logical families" not in refused.stderr:
        raise RuntimeError("device compilation did not identify the unallocated logical artifact: "
                           + refused.stderr)
    invoke([*command, "--pto-frontier-analysis=deferred-output=true physical-output=true", "-"], source,
           succeeds=False)
    print("deferred families: separate-process roundtrip, six causal-reuse traces, "
          "live construction and device refusal checked")


if __name__ == "__main__":
    main()
