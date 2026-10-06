# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Byte-membership oracle for bounded uint32 scalar-address truncation."""
import json
import shutil
import subprocess
import sys

from arithmetic_membership import membership_index

executable = shutil.which(sys.argv[1])
if executable is None:
    raise RuntimeError("arithmetic test executable unavailable")
result = subprocess.run([executable, "--arithmetic", sys.argv[2]], check=True,
                        capture_output=True, text=True, timeout=30)
documents = {doc["function"]: doc for line in result.stdout.splitlines()
             if line.startswith("arithmetic-json ")
             for doc in [json.loads(line.removeprefix("arithmetic-json "))]}
assert set(documents) == {"bounded", "negative_quotient", "crosses_wrap", "empty", "quotient"}
# An interval spanning a uint32 wrap must not be silently linearized.
assert not documents["crosses_wrap"]["relations"]
for name, shift, upper in (("bounded", 0, 8), ("negative_quotient", -8, 8), ("empty", 0, 0)):
    doc = documents[name]
    assert len(doc["sites"]) == 1 and not doc["parameters"]
    contains = membership_index(doc)
    for i in range(-1, 10):
        begin = ((i + shift) % (1 << 32)) * 4
        for byte in (begin - 1, begin, begin + 1, begin + 3, begin + 4, 0, 31):
            expected = 0 <= i < upper and begin <= byte < begin + 4
            assert contains((5, 0, -1, 0, 0), (i, byte)) == expected, (name, i, byte)
print("bounded scalar address maps preserve uint32 wrap and empty domains")

# Evaluate appended existential quotient symbols independently. Unlike period
# expansion, a fixed divisor seven adds one local and keeps original period two.
doc = documents["quotient"]
assert len(doc["sites"]) == 1 and not doc["parameters"]
accesses = [relation for relation in doc["relations"] if relation["kind"] == 5]
assert accesses
for i in range(-1, 23):
    for byte in range(-1, 14):
        actual = False
        values = (i, byte)
        for relation in accesses:
            for piece in relation["pieces"]:
                if piece["empty"] or tuple(piece["residues"]) != tuple(x % doc["period"] for x in values):
                    continue
                assert all(len(row["coefficients"]) == 3 for row in piece["rows"])
                for local in range(-2, 8):
                    coordinates = tuple(x // doc["period"] for x in values) + (local,)
                    actual |= all((value == 0 if row["equality"] else value >= 0)
                                  for row in piece["rows"]
                                  for value in [row["constant"] + sum(a * b for a, b in
                                      zip(row["coefficients"], coordinates))])
        expected = 0 <= i < 21 and (i // 7) * 4 <= byte < (i // 7) * 4 + 4
        assert actual == expected, ("quotient", i, byte)
print("constant division uses exact quotient locals without period expansion")
