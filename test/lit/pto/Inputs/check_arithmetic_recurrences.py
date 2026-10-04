# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check metadata and exact carried-bank recurrences without using the recognizer."""
import itertools
import json
import shutil
import subprocess
import sys

from arithmetic_membership import membership_index




def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing recognition executable")
    run = subprocess.run([tool, "--arithmetic", sys.argv[2]], check=True,
                         capture_output=True, text=True, timeout=45)
    documents = [json.loads(line.removeprefix("arithmetic-json ")) for line in run.stdout.splitlines()
                 if line.startswith("arithmetic-json ")]
    assert len(documents) == 6
    checks = 0
    for document in documents:
        name = document["function"]
        if name in {"unknown_carried", "unknown_carried_address", "loop_result_address"}:
            assert not document["sites"] and not document["relations"], name
            continue
        expected = {"metadata_view": (0, []), "carried_banks": (1, [0]),
                    "carried_nested_reset": (2, [0, 1])}[name]
        assert len(document["sites"]) == 1 and document["sites"][0]["depth"] == expected[0], name
        assert document["parameters"] == expected[1], name
        contains = membership_index(document)
        candidates = list(itertools.product(range(-1, 4), repeat=expected[0]))
        for parameters in itertools.product(range(-1, 4), repeat=len(expected[1])):
            if name == "metadata_view":
                execution = [()]
            elif name == "carried_banks":
                execution = [(i,) for i in range(parameters[0])]
            else:
                execution = [(t, i) for t in range(parameters[0]) for i in range(parameters[1])]
            ranks = {coords: i for i, coords in enumerate(execution)}
            for a in candidates:
                assert contains((1, 0, -1, 0, 0), a + parameters) == (a in ranks), (name, a, parameters)
                origin = 0 if not a else 1024 * ((1 + a[-1]) % 2)
                for byte in (-1, 0, 3, 4, 1023, 1024, 1027, 1028):
                    expected_read = a in ranks and origin <= byte < origin + 4
                    assert contains((4, 0, -1, 0, 0), a + (byte,) + parameters) == expected_read, (name, a, byte)
                    assert not contains((5, 0, -1, 0, 0), a + (byte,) + parameters), name
                    checks += 2
                for b in candidates:
                    before = a in ranks and b in ranks and ranks[a] < ranks[b]
                    values = a + b + parameters
                    assert contains((2, 0, 0, 0, 0), values) == before, (name, a, b)
                    for source, target in itertools.product((1, 2), repeat=2):
                        expected_native = (before and (source, target) != (2, 1)) or (
                            a == b and a in ranks and (source, target) == (1, 2))
                        assert contains((3, 0, 0, source, target), values) == expected_native, (name, a, b)
                        checks += 1
    print("arithmetic recurrences: 3 accepted, 3 rejected; semantic comparisons:", checks)


if __name__ == "__main__":
    main()
