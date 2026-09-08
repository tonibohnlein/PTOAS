#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Require the real native loop capture to expose broken interchange contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from check_native_facts import check

record = json.loads(Path(sys.argv[1]).read_text())
for bound in (0, 1, 2, 16):
    for take in (0, -1):
        result = check(record, [0, bound, take])
        assert result["status"] == "PASS", result

def rejected(name, mutate):
    candidate = deepcopy(record)
    mutate(candidate)
    try:
        check(candidate, [0, 2, -1])
    except (ValueError, KeyError, AssertionError):
        return
    raise AssertionError("checker accepted " + name)

rejected("lost native requirements", lambda r: r["native"].update(obligations=[]))
rejected("wrong hazard kind", lambda r: r["native"]["obligations"][0].update(hazards=0))
rejected("lost loop", lambda r: r["model"]["statements"][1].update(domain="false"))
rejected("wrong footprint", lambda r: r["model"]["statements"][0]["writes"][0].update(address="32768"))
rejected("wrong lane", lambda r: r["model"]["statements"][0].update(lane="V"))
rejected("wrong target context", lambda r: r.update(context_json="{}"))
print("8 native loop/skip replays and 6 broken interchange mutations detected")
