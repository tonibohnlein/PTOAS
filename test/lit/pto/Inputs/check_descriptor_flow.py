# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.


"""Descriptor versions remain scalar state; partial production is not a kill."""
import json
from pathlib import Path
import subprocess
import sys

binary, source, helpers, prefix = sys.argv[1:]
sys.path.insert(0, helpers)
from run_qwen_additions import analyze
text = Path(source).read_text()
updates = """      pto.set_validshape %a, %size, %size : !V
      pto.set_validshape %b, %size, %size : !V"""
partial = updates.replace("%size, %size", "%partial, %size")
choice = "scf.if %choose {\n" + partial + "\n      }"
variants = {
    "full": (text, True),
    "partial": (text.replace(updates, partial), False),
    "join_partial": (text.replace("// BEFORE_PAYLOAD", choice), False),
    "restore_full": (text.replace("// DESCRIPTOR_UPDATE", choice), True),
    "carried_partial": (text.replace(updates, choice), False),
    "unknown": (text.replace(updates, updates.replace("%size, %size", "%trips, %size")), False),
}
reports = {}
for name, (body, selected) in variants.items():
    input_path, output = Path(prefix + "." + name + ".input.pto"), Path(prefix + "." + name + ".pto")
    input_path.write_text(body)
    command = [binary, str(input_path), "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false",
               "--pto-insert-sync=buffer-generations=true defer-same-pipe=true gm-alias=assume-disjoint-arguments effect-coverage=report",
               "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (name, result.stderr)
    Path(prefix + "." + name + ".stderr").write_text(result.stderr)
    before, after = analyze(input_path), analyze(output)
    for key in ("payload", "allocations", "views", "abi"):
        assert before[key] == after[key], (name, key)
    committed = 'pto.insert_sync.status = "lifecycle-plus-residuals"' in output.read_text()
    assert committed == selected, (name, result.stderr)
    assert "unmodeled non-payload" not in result.stderr
    if not selected:
        assert "no complete exact-slot lifecycle" in result.stderr
    reports[name] = {"mechanisms": after["mechanisms"], "selected": committed}
Path(prefix + ".results.json").write_text(json.dumps(reports, indent=2) + "\n")
print("Descriptor full/partial, branch restore/join, carried and unknown versions preserve all payload contracts")
