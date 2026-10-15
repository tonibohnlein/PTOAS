# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Prepare fixtures and compare saved traces; execution stays in the shell test."""
import json
from pathlib import Path
import re
import sys


def main():
    action, *arguments = sys.argv[1:]
    if action == "prepare":
        source, output, inner = arguments
        text = Path(source).read_text(encoding="utf-8")
        text = text.replace("    %two =", f"    %fixed = arith.constant {int(inner)} : index\n    %two =")
        text = text.replace("%i = %zero to %m", "%i = %zero to %fixed")
        Path(output).write_text(text, encoding="utf-8")
    elif action == "trace":
        source, output, outer, inner = arguments
        text = Path(source).read_text(encoding="utf-8")
        if "pto.logical_" in text or "pto.endpoint_families" in text:
            raise RuntimeError("physical output retained a logical interface")
        text = text.replace("array<i64: 2, 3>", f"array<i64: {int(outer)}, {int(inner)}>")
        text, count = re.subn(r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
                             r"\1attributes {test.trace_existing, ", text, count=1)
        if count != 1:
            raise RuntimeError("missing original trace invocation")
        Path(output).write_text(text, encoding="utf-8")
    elif action == "profile":
        diagnostic, source = arguments
        prefix = "frontier-control-profile "
        reports = [json.loads(line[len(prefix):]) for line in
                   Path(diagnostic).read_text(encoding="utf-8").splitlines() if line.startswith(prefix)]
        inserted = [report for report in reports if report["stage"] == "inserted"]
        if len(inserted) != 1:
            raise RuntimeError("missing combined output inventory")
        text = Path(source).read_text(encoding="utf-8")
        for kind, command in (("sets", "set"), ("waits", "wait")):
            sites = len(re.findall(rf"pto\.{command}_flag(?:_dyn)?\b", text))
            if not sites or inserted[0]["function_control"][kind] != sites:
                raise RuntimeError("physical command inventory does not match emitted sites")
    elif action == "compare":
        reports = [json.loads(Path(name).read_text(encoding="utf-8")) for name in arguments]
        if len(reports) != 2 or any(not report["accepted"] or report["trace"]["error"] for report in reports):
            raise RuntimeError("physical trace failed")
        if reports[0]["trace"]["events"] != reports[1]["trace"]["events"]:
            raise RuntimeError("combined/split physical trace mismatch")
    else:
        raise RuntimeError(f"unknown fixture action: {action}")


if __name__ == "__main__":
    main()
