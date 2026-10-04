# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Rejected access contracts must not leak partial arithmetic exports."""
import json
import shutil
import subprocess
import sys

executable = shutil.which(sys.argv[1])
if executable is None:
    raise RuntimeError("arithmetic test executable unavailable")
result = subprocess.run([executable, "--arithmetic", sys.argv[2]],
                        check=True, capture_output=True, text=True, timeout=30)
documents = [json.loads(line.removeprefix("arithmetic-json ")) for line in result.stdout.splitlines()
             if line.startswith("arithmetic-json ")]
if len(documents) != 7:
    raise RuntimeError("expected all seven arithmetic fixtures")
for document in documents:
    if document["relations"] or document["sites"] or document["parameters"]:
        raise RuntimeError(document["function"] + ": partial exports on rejection")
print("arithmetic rejection exports: 7 passed")
