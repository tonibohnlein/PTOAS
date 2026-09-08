# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Native Q-projection readiness sharing with independent operand hazard replay."""
import json
from pathlib import Path
import re
import subprocess
import sys

binary, source, helpers, prefix = sys.argv[1:]
sys.path.insert(0, helpers)
from compare_boundaries import Boundaries
from measure import children, fingerprint, replay
from run_qwen_additions import analyze
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto


def check(path):
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        module = ir.Module.parse(path.read_text())
        function = next(op for op in children(module.operation) if op.name == "func.func")
        trace, handles, writers, readers = Boundaries(), {}, {}, {}
        def observe(op, point, signature):
            trace.observe(op, point, signature)
            if op.name == "pto.alloc_tile":
                kind = re.search(r"tile_buf<(\w+),", str(op.results[0].type))[1]
                handles[fingerprint(signature)] = (kind, signature[1][0])
            elif op.name == "pto.textract":
                slot = handles[signature[1][-1]]
                if slot[0] not in ("left", "right"):
                    return
                before = trace.before[-1]["completed"]
                assert before.get("PIPE_M", -1) >= readers.get(slot, -1), ("operand overwrite", slot)
                writers[slot] = len(trace.payload) - 1
            elif op.name in ("pto.tmatmul", "pto.tmatmul.acc"):
                for operand in signature[1]:
                    slot = handles.get(operand)
                    if slot and slot[0] in ("left", "right"):
                        assert slot in writers, ("missing operand", slot)
                        assert trace.before[-1]["completed"].get("PIPE_MTE1", -1) >= writers[slot], ("operand readiness", slot)
                        readers[slot] = len(trace.payload) - 1
        metric = replay(function, ["INPUT", "WEIGHT", "OUTPUT", 0, 24], observer=observe)
        assert not trace.tokens
        assert metric["counts"].get("pto.set_flag", 0) == metric["counts"].get("pto.wait_flag", 0)
        return metric

original = Path(source).read_text()
loop = re.search(r"scf.for (%[\w]+) = (%[\w]+) to (%[\w]+) step (%[\w]+)", original)
assert loop
upper = loop[3]
variants = {"unchanged": original, "renamed": original.replace("@q_proj", "@reused_operands")}
for count in (0, 2, 4):
    variants["bound_" + str(count)] = re.sub(re.escape(upper) + r" = arith.constant 64 : index",
                                             upper + " = arith.constant " + str(count) + " : index", original)
reports = {}
for name, text in variants.items():
    input_path, output = Path(prefix + "." + name + ".input.pto"), Path(prefix + "." + name + ".pto")
    input_path.write_text(text)
    command = [binary, str(input_path), "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false",
               "--pto-insert-sync=buffer-generations=true defer-same-pipe=true "
               "mmad-chains=true gm-alias=assume-disjoint-arguments", "-o", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, (name, completed.stderr)
    Path(prefix + "." + name + ".stderr").write_text(completed.stderr)
    before, after = analyze(input_path), analyze(output)
    for key in ("payload", "allocations", "views", "abi"):
        assert before[key] == after[key], (name, key)
    if name in ("unchanged", "renamed"):
        assert "generation_ready_streams_supplied = 4" in output.read_text()
        assert "generation_ready_sets_removed = 6" in output.read_text()
        assert "generation_ready_waits_removed = 6" in output.read_text()
    metric = check(output)
    reports[name] = {"static": after["mechanisms"], "execution": metric}
    if name == "unchanged":
        # Balanced counts alone cannot detect missing operand readiness. Remove
        # every MTE1->M ready stream but retain all M->MTE1 reuse acknowledgements.
        broken = re.sub(r"(?m)^.*pto.(?:set|wait)_flag\[<PIPE_MTE1>, <PIPE_M>,[^\n]+\n", "", output.read_text())
        assert broken != output.read_text()
        path = Path(prefix + ".broken.pto")
        path.write_text(broken)
        try:
            check(path)
        except AssertionError as error:
            assert "operand readiness" in str(error), str(error)
        else:
            raise AssertionError("balanced missing readiness was not detected")
Path(prefix + ".report.json").write_text(json.dumps(reports, indent=2) + "\n")
print("Shared readiness: unchanged and renamed Q projection lose six static pairs; "
      "operand production/reclamation, empty/one/repeated executions and a broken balanced plan checked")
