# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Global GM discharge preserves aliasing and repeated-writer obligations."""
import json
import shutil
import subprocess
import sys

from arithmetic_membership import membership_index

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


tool = shutil.which(sys.argv[1])
if tool is None:
    raise RuntimeError("arithmetic test executable unavailable")
for policy in ("may-alias", "may-not-alias"):
    result = subprocess.run([tool, "--gm-alias=" + policy, "--arithmetic", sys.argv[2]],
                            check=True, capture_output=True, text=True)
    regions = [doc for line in result.stdout.splitlines() if line.startswith("arithmetic-json ")
               for doc in [json.loads(line.removeprefix("arithmetic-json "))] if "region" in doc]
    require(len(regions) == 6, "missing GM discharge fixtures")
    for doc in regions:
        name = doc["region"]
        require(len(doc["sites"]) == 1 and not doc["issues"], (policy, name, doc))
        discharged = name in {"stream", "read_only"} or (name == "different_base_peer" and policy == "may-not-alias")
        require(doc["discharged_effects"] == int(discharged), (policy, name, doc))
        contains = membership_index(doc)
        if discharged:
            require(not doc["parameters"], (policy, name, "discharged-only address parameter retained"))
            for i in range(2):
                for byte in range(12):
                    for kind in (4, 5):
                        require(not contains((kind, 0, -1, 0, 0), (i, byte)), (policy, name, kind, i, byte))
        else:
            # Both a proved preserving cast and an opaque cast-result binding
            # are exported as the original index SSA value, never raw i64.
            require(doc["parameter_kinds"] == ["entry-value"], (policy, name, doc))
            for base in (-1, 0, 1, 3):
                for i in range(-1, 3):
                    offset = base if name in {"same_address", "cast_binding"} else base + i
                    for byte in range(-8, 25):
                        expected = 0 <= i < 2 and 4 * offset <= byte < 4 * offset + 4
                        require(contains((5, 0, -1, 0, 0), (i, byte, base)) == expected,
                                (policy, name, i, byte, base))
print("GM discharge: distinct visits, global aliases, repeated writes and index bindings checked")
