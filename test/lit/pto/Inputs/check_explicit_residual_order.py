# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check exact residual eligibility and reference-ordered reset protection."""
import json
import sys


def main():
    reports = [json.loads(line) for line in sys.stdin if line.startswith("{")]
    expected = {"reset_then_writer": [], "writer_then_reset": [[0, 1]]}
    if len(reports) != len(expected) or {report["function"] for report in reports} != set(expected):
        raise RuntimeError("missing residual-order production reports")
    for report in reports:
        name = report["function"]
        if name not in expected or report["error"]:
            raise RuntimeError("unexpected or unsuccessful explicit analysis")
        if report["residual_eligible"] != [False, True]:
            raise RuntimeError("fixture did not exercise one-sided residual eligibility")
        if report["retained"] != expected[name]:
            raise RuntimeError("accumulator protection did not preserve reference order")
    print("explicit residuals: one-sided eligibility preserves reset ordering")


if __name__ == "__main__":
    main()
