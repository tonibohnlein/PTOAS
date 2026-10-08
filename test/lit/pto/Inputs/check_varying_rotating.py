# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check automatic certificates on a nested rotating source, without unrolling."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def report(tool, path):
    result = subprocess.run([tool, "--numeric-analysis", str(path)], capture_output=True, text=True,
                            check=True, timeout=90)
    return json.loads(result.stdout.splitlines()[0])


def attempts(document, route):
    return [a for n in document["nodes"] for a in n["attempts"] if a["route"] == route]


def main():
    tool, source = sys.argv[1:]
    text = Path(source).read_text()
    document = report(tool, source)
    accepted = [a for a in attempts(document, "varying-rotating") if a["state"] == "applicable"]
    assert len(accepted) == 1, document
    result = accepted[0]
    assert result["slope"] == 1 and result["intercept"] == 2, result
    # Recognition is independent of demand/export construction. The companion
    # check_compact_endpoints test validates those stages against unfolded graphs.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "varying.pto"
        # Wrapped machine arithmetic must not be treated as an affine trip count.
        path.write_text(text.replace("%zero to %visits", "%zero to %n"))
        bad = attempts(report(tool, path), "varying-rotating")
        assert not any(a["state"] == "applicable" for a in bad), bad
        # The same modeled accesses with independently changing participation
        # need a refresh certificate, not an invariant-execution certificate.
        window = text.replace("    scf.for %visit = %zero to %visits step %one {\n"
                              "      %length = arith.addi %visit, %two : index\n"
                              "      scf.for %i = %zero to %length step %one {",
                              "    scf.for %i = %zero to %n step %one {")
        window = window.replace("      }\n    }\n", "    }\n")
        compute = "      pto.tmatmul ins(%left, %right : !left, !right) outs(%acc : !acc)"
        optional = ("\n      %take = arith.cmpi eq, %slot, %zero : index\n"
                    "      scf.if %take {\n"
                    "        pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%left : !left)\n"
                    "      }")
        path.write_text(window.replace(compute, compute + optional))
        bounded = attempts(report(tool, path), "bounded-lifetime")
        good = [a for a in bounded if a["state"] == "applicable"]
        assert len(good) == 1 and good[0]["refresh_span"] == 2, bounded
        assert not good[0]["entry_guards_available"], good
    print("varying rotation, bounded refresh, and machine-overflow exclusion passed")


if __name__ == "__main__":
    main()
