# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check occupancy in one straight-line emitted word (not a causal proof).

The fixture's single-trip loops must already be folded. Cross-core queue
protocols are deliberately outside this local directional-event check.
"""
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
start = re.search(r"AICORE void " + re.escape(sys.argv[2]) + r"\(", text)
assert start, "missing kernel"
body_start = text.index("{", start.end())
depth = 1
end = body_start + 1
while depth:
    depth += (text[end] == "{") - (text[end] == "}")
    end += 1
body = text[body_start + 1:end - 1]
assert not re.search(r"\b(?:for|while|if|switch)\s*\(", body), "requires a straight-line word"
commands = re.findall(
    r"(set|wait)_flag\((PIPE_\w+), (PIPE_\w+), EVENT_ID(\d+)\)", body
)
assert commands, "no local events exercised"
if "--remove-first-wait" in sys.argv[3:]:
    del commands[next(i for i, command in enumerate(commands) if command[0] == "wait")]
live = set()
published = set()
reused = set()
for kind, source, target, number in commands:
    key = (source, target, number)
    if kind == "set":
        assert key not in live, f"publication overwrites an unconsumed event: {key}"
        if key in published:
            reused.add(key)
        published.add(key)
        live.add(key)
    else:
        assert key in live, f"acquisition has no publication: {key}"
        live.remove(key)
assert not live, f"events remain live at exit: {live}"
assert any(key[:2] == ("PIPE_V", "PIPE_MTE2") for key in reused), "reuse not exercised"
