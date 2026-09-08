# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Check native handoff decisions, negative controls and repeat-run stability."""
from pathlib import Path
import re
import subprocess
import sys

binary, source, existing, prefix = sys.argv[1:]
original = Path(source).read_text()


def run(name, text, optimize=True):
    path = Path(prefix + "." + name + ".pto")
    path.write_text(text)
    command = [binary, "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false", str(path)]
    if optimize:
        command.append("--pto-experiment-handoff-planning")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout, result.stderr


positive, log = run("split", original)
assert "1 split; 1 removed cross-lane completion facts" in log, log
assert positive.count("pto.set_flag") == positive.count("pto.wait_flag") == 2
second, log = run("idempotent", positive)
assert second == positive and "0 trials" in log, log

# A common consumer needs both productions: combining them is appropriate.
bundled = original.replace("pto.tabs ins(%a : !V) outs(%c : !V)",
                           "pto.tadd ins(%a, %b : !V, !V) outs(%c : !V)")
# A whole-function event pool is occupied. Do not widen boundaries or alias a
# live logical stream to force the otherwise useful split.
reservations = "".join(
    f"    pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID{i}>]\n"
    f"    pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID{i}>]\n"
    for i in range(1, 8))
full_pool = original.replace("    pto.tload", reservations + "    pto.tload", 1)
bad_token = original.replace("    pto.wait_flag", "    pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]\n    pto.wait_flag", 1)
# Input synchronization is already insufficient here. Seed proof failure must
# leave it unchanged rather than manufacturing a successful optimization.
bad_read = original.replace("outs(%b : !V)", "outs(%a : !V)", 1)
variants = {"bundle": bundled, "finite_pool": full_pool, "bad_token": bad_token,
            "bad_dependency": bad_read}
for name, text in variants.items():
    expected, _ = run(name + ".baseline", text, False)
    actual, log = run(name, text)
    assert actual == expected, (name, log, actual)

# The general release/readiness examples are independent of this new fixture.
placed, log = run("existing_boundaries", Path(existing).read_text())
assert log.count("1 advanced") == 2 and log.count("1 delayed") == 1, log
all_readers = placed.split("func.func @all_readers_needed", 1)[1].split("func.func @", 1)[0]
assert re.search(r"pto.tabs[^\n]*\n\s*pto.tabs[^\n]*\n\s*pto.set_flag", all_readers), all_readers

for bound in ("%z", "%s"):
    loop = original.replace("    pto.tload", f"    scf.for %i = %z to {bound} step %one {{\n    pto.tload", 1)
    loop = loop.replace("    return", "    }\n    return")
    expected, _ = run("loop.baseline", loop, False)
    actual, log = run("loop", loop)
    assert actual == expected and "structured recurrence/choice" in log, log
print("Handoff experiment: split, repeat stability, bundled demand, finite pool, invalid seeds, "
      "release/readiness placement and zero/nonzero loop refusal passed")
