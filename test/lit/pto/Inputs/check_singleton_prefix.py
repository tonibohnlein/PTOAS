# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check original singleton cuts through the actual prefix capability adapter."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from check_repeated_region import check, run


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text(encoding="utf-8")
    cases = 0
    with tempfile.TemporaryDirectory(prefix="singleton-prefix-") as scratch:
        path = Path(scratch) / "case.pto"
        for active in (0, 1, 2, 9):
            for enabled in (0, 1):
                program = source.replace("%active = arith.constant 1", f"%active = arith.constant {active}")
                if active == 9:
                    program = program.replace("%three = arith.constant 3", "%three = arith.constant 11")
                    program = program.replace("    scf.for %visit", "    %lower = arith.constant 7 : index\n"
                                              "    %step = arith.constant 2 : index\n    scf.for %visit")
                    program = program.replace("%visit = %zero to %three step %one",
                                              "%visit = %lower to %three step %step")
                program = program.replace("array<i64: 1>", f"array<i64: {enabled}>")
                path.write_text(program, encoding="utf-8")
                document = json.loads(run(tool, "--structured-trace", path))
                payloads = [event for event in document["trace"]["events"] if event["kind"] == "payload"]
                expected = [(0, [active]), (1, [active])] if enabled else []
                assert [(event["type"], event["coordinates"]) for event in payloads] == expected
                normalized = copy.deepcopy(document)
                for event in normalized["trace"]["events"]:
                    if event["kind"] == "payload":
                        event["coordinates"] = [0]
                check(normalized, [enabled], external_work=False)
                cases += 1
    print(f"singleton prefix: {cases} original-cut payload and command closures passed")


if __name__ == "__main__":
    main()
