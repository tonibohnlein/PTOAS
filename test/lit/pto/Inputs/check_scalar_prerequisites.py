# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check scalar SSA ordering without dropping asynchronous storage hazards."""
import json
import shutil
import subprocess
import sys


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("synchronization test tool is unavailable")
    result = subprocess.run([tool, "--explicit-analysis", sys.argv[2]], check=True,
                            capture_output=True, text=True, timeout=60)
    report = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("{")))
    if report["error"] or report["span_error"]:
        raise RuntimeError(report)
    # Vector writes must finish before the scalar read. The scalar value then
    # orders its scalar consumer natively, including through pure arithmetic.
    if report["retained"] != [[0, 1]] or report["span_retained"] != [[0, 1]]:
        raise RuntimeError(("incorrect scalar demands", report))
    graph = report["event_reachable"]
    if not graph[3][4] or not graph[1][2] or graph[4][3]:
        raise RuntimeError(("incorrect native scalar reachability", graph))
    print("scalar prerequisites: native value edge and asynchronous storage hazard passed")


if __name__ == "__main__":
    main()
