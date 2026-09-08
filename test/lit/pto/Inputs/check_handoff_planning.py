# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check native handoff decisions, negative controls and repeat-run stability."""
from pathlib import Path
import re
import subprocess
import sys

binary, source, existing, prefix = sys.argv[1:]
original = Path(source).read_text()


def run(name, text, optimize=True, success=True):
    path = Path(prefix + "." + name + ".pto")
    path.write_text(text)
    command = [binary, "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false", str(path)]
    if optimize:
        command.append("--pto-experiment-handoff-planning")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == success, result.stderr
    return result.stdout, result.stderr


positive, log = run("split", original)
assert "1 split;" in log, log
assert positive.count("pto.set_flag") == positive.count("pto.wait_flag") == 2
second, log = run("idempotent", positive)
assert second == positive and "0 split;" in log, log

# A common consumer needs both productions: combining them is appropriate.
bundled = original.replace("pto.tabs ins(%a : !V) outs(%c : !V)",
                           "pto.tadd ins(%a, %b : !V, !V) outs(%c : !V)")
# A whole-function event pool is occupied. Do not widen boundaries or alias a
# live logical stream to force the otherwise useful split.
reservations = "".join(
    f"    %ai{i} = arith.constant {2048 + i * 1024} : i64\n"
    f"    %bi{i} = arith.constant {2560 + i * 1024} : i64\n"
    f"    %in{i} = pto.alloc_tile addr = %ai{i} : !V\n"
    f"    %out{i} = pto.alloc_tile addr = %bi{i} : !V\n"
    f"    pto.tload ins(%input : !P) outs(%in{i} : !V)\n"
    f"    pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID{i}>]\n"
    f"    pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID{i}>]\n"
    f"    pto.tabs ins(%in{i} : !V) outs(%out{i} : !V)\n"
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

# A runtime-selected key may collide with the fresh key needed for a split.
# Until its selector/recurrence semantics are imported, retain the whole seed.
dynamic = original.replace("    pto.tload",
    "    pto.set_flag_dyn [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_V>, %one]\n"
    "    pto.wait_flag_dyn [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_V>, %one]\n    pto.tload", 1)
expected, _ = run("dynamic.baseline", dynamic, False)
actual, log = run("dynamic", dynamic)
assert actual == expected and "dynamic event selectors/reservations not imported" in log, log

high_level = original.replace("    pto.tload",
    "    pto.record_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID1>]\n"
    "    pto.wait_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID1>]\n    pto.tload", 1)
expected, _ = run("high_level.baseline", high_level, False)
actual, log = run("high_level", high_level)
assert actual == expected and "incomplete physical effect coverage" in log, log

# Diagnosed malformed contracts are hard failures, not unsupported seeds.
_, log = run("invalid_contract", original.replace("assume-disjoint-arguments", "invalid"), success=False)
assert "invalid GM contract" in log, log

# Explicit test-only ownership exercises barrier trials without authorizing
# production to erase authored barriers. A needed drain must survive rejection.
marked = original.replace("    pto.wait_flag", "    pto.barrier <PIPE_MTE2> {pto.handoff_test_owned}\n    pto.wait_flag", 1)
actual, log = run("redundant_barrier", marked)
assert "pto.handoff_test_owned" not in actual, (actual, log)
marked_exit = original.replace("pto.barrier <PIPE_ALL>",
    "pto.barrier <PIPE_ALL> {pto.handoff_test_owned, pto.auto_sync_tail_barrier}")
actual, log = run("required_exit", marked_exit)
assert "pto.barrier <PIPE_ALL>" in actual, (actual, log)

# Physical sections have no required terminator. The last synchronization
# action has no next-node anchor; restoration must retain its original block.
section = original.replace("    %z =", "    pto.section.vector {\n    %z =", 1).replace("    return", "    }\n    return")
actual, log = run("section_tail", section)
assert "1 split;" in log and "pto.barrier <PIPE_ALL>" in actual, log
section = section.replace("    }\n    return", "    pto.barrier <PIPE_V> {pto.handoff_test_owned}\n    }\n    return")
actual, log = run("section_redundant_tail", section)
assert "pto.handoff_test_owned" not in actual, log

# The general release/readiness examples are independent of this new fixture.
placed, log = run("existing_boundaries", Path(existing).read_text())
assert log.count("1 advanced") == 1 and log.count("1 delayed") == 1 and log.count("1 split;") == 2, log
all_readers = placed.split("func.func @all_readers_needed", 1)[1].split("func.func @", 1)[0]
assert re.search(r"pto.tabs[^\n]*\n\s*pto.tabs[^\n]*\n\s*pto.set_flag", all_readers), all_readers

# Complete per-invocation reverse acknowledgement supports key recurrence.
# Empty/skipped invocations execute neither endpoint of either direction.
for bound in ("%z", "%s"):
    loop = original.replace("    pto.tload", f"    scf.for %i = %z to {bound} step %one {{\n    pto.tload", 1)
    loop = loop.replace("    pto.barrier <PIPE_ALL>",
        "    pto.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]\n"
        "    pto.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]\n    pto.barrier <PIPE_ALL>")
    loop = loop.replace("    return", "    }\n    return")
    actual, log = run("loop", loop)
    if bound == "%s":
        assert "1 split;" in log, log
    skipped = loop.replace("%src: !pto.ptr<f16>", "%src: !pto.ptr<f16>, %take: i1").replace(
        "    scf.for", "    scf.if %take {\n    scf.for").replace("    return", "    }\n    return")
    actual, log = run("skipped", skipped)
    if bound == "%s":
        assert "1 split;" in log, log
print("Native handoff planning: split, repeat stability, bundled demand, invalid seeds, "
      "release/readiness placement and complete zero/nonzero/skipped loops passed")
