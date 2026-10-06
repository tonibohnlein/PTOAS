# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check disjoint strided GM visits against independent byte enumeration."""
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    checked = 0
    with tempfile.TemporaryDirectory(prefix="strided-streaming-") as directory:
        path = Path(directory) / "case.pto"
        for rowstride in (16, 32, 64):
            for direction in (1, -1, 0):
                for trips in (1, 2, 3, 4, 5):
                    text = source.replace("%n = arith.constant 4", f"%n = arith.constant {trips}")
                    text = text.replace("%rowstride = arith.constant 64", f"%rowstride = arith.constant {rowstride}")
                    if direction < 0:
                        text = text.replace("%column = arith.muli %i, %sixteen : index",
                            "%origin = arith.constant 80 : index\n"
                            "      %delta = arith.muli %i, %sixteen : index\n"
                            "      %column = arith.subi %origin, %delta : index")
                    if direction == 0:
                        text = text.replace("%column = arith.muli %i, %sixteen : index",
                                            "%column = arith.constant 0 : index")
                    path.write_text(text)
                    result = subprocess.run([tool, "--recognize", str(path)],
                                            text=True, capture_output=True, check=True)
                    accepted = "recognize rotating: applicable " in result.stdout
                    visits = [{row * rowstride + direction * 16 * i + element
                               for row in range(2) for element in range(16)}
                              for i in range(trips)]
                    disjoint = all(not (visits[a] & visits[b])
                                   for a in range(trips) for b in range(a + 1, trips))
                    # A constant address also has a valid non-discharged rotating
                    # description; only translated GM writes need this proof.
                    if direction:
                        assert accepted == disjoint, (rowstride, direction, trips, result.stdout)
                    else:
                        assert not disjoint or trips == 1
                    checked += 1
        path.write_text(source.replace("%n = arith.constant 4", "%n = arith.constant 0"))
        result = subprocess.run([tool, "--sequence-analysis", str(path)],
                                text=True, capture_output=True, check=True)
        assert '"error":""' in result.stdout, result.stdout
    print(f"strided streaming: {checked} footprint cases plus zero-trip check passed")


if __name__ == "__main__":
    main()
