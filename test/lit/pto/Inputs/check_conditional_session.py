# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check cached conditional child preparation against event and allocation oracles."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_compact_endpoints import check_commands
from check_finite_allocation import physical_check


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text().replace(
        "attributes {test.trace_arguments",
        "attributes {test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>, test.trace_arguments")
    tested = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "conditional.pto"
        for alternate in (False, True):
            selected = source.replace("scf.if %guard {", "scf.if %guard {\n    } else {") if alternate else source
            for guard in (0, 1):
                for trips in (0, 1, 3, 5):
                    path.write_text(selected.replace("array<i64: 5, 1>", f"array<i64: {trips}, {guard}>"))
                    result = subprocess.run([tool, "--structured-trace", str(path)],
                                            capture_output=True, text=True, check=True, timeout=90)
                    document = json.loads(result.stdout)
                    payloads, _ = check_commands(document)
                    expected = trips if bool(guard) != alternate else 0
                    if len(payloads) != 2 + 2 * expected:
                        raise RuntimeError("conditional child changed original occurrence presence")
                    physical_check(document, set(range(6)))
                    tested += 1
    print(f"{tested} cached conditional child command closures and allocations passed")


if __name__ == "__main__":
    main()
