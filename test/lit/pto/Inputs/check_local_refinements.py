# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Unchanged native loops: graph admission, token participation and command cost.

Frozen baselines retain the previously improved boundaries. The independent
scalar replay checks actual tokens and compares every physical cross-lane cut;
it does not model device latency or certify unmodeled target memory semantics.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

binary, inputs, helpers, prefix = sys.argv[1:]
inputs = Path(inputs)
sys.path.insert(0, helpers)
from compare_boundaries import compare, run
from run_qwen_additions import analyze

for row in json.loads((inputs / "manifest.json").read_text())["files"]:
    assert hashlib.sha256((inputs / row["path"]).read_bytes()).hexdigest() == row["sha256"]

reports = {}
for name, arguments, removed in (("gemm_tile", [0, 1048576, 2097152, 0, 0], 1),
                                 ("proj", [0, 67108864, 134217728, 0], 2),
                                 ("exp_gate_mm", [0, 67108864, 134217728, 0, 0, 0, 0], 0)):
    original = (inputs / (name + ".input.pto")).read_text()
    loop = re.search(r"scf.for (%[\w]+) = (%[\w]+) to (%[\w]+) step (%[\w]+)", original)
    assert loop
    variants = {"unchanged": original, "renamed": original.replace("@" + name, "@independent_name")}
    for count in (0, 1, 2):
        # Replace only this loop's bound, not the constant's other payload uses.
        variants["bound_" + str(count)] = original[:loop.start()] + (
            "%test_bound = arith.constant " + str(count) + " : index\n  " +
            original[loop.start():].replace("to " + loop[3] + " step", "to %test_bound step", 1))
    for variant, text in variants.items():
        key = name + "." + variant
        source, target = Path(prefix + "." + key + ".input.pto"), Path(prefix + "." + key + ".pto")
        source.write_text(text)
        result = subprocess.run([binary, str(source), "--mlir-disable-threading",
            "--mlir-print-op-on-diagnostic=false",
            "--pto-insert-sync=buffer-generations=true defer-same-pipe=true "
            "mmad-chains=true gm-alias=assume-disjoint-arguments", "-o", str(target)],
            capture_output=True, text=True, timeout=60)
        Path(prefix + "." + key + ".stderr").write_text(result.stderr)
        assert result.returncode == 0, (key, result.stderr)
        before, after = analyze(source), analyze(target)
        for field in ("payload", "allocations", "views", "abi"):
            assert before[field] == after[field], (key, field)
        trace, metric = run(target, {"arguments": arguments})
        assert metric["counts"].get("pto.set_flag", 0) == metric["counts"].get("pto.wait_flag", 0)
        if variant in ("unchanged", "renamed"):
            assert 'pto.insert_sync.status = "lifecycle-plus-residuals"' in target.read_text()
            if name == "exp_gate_mm":
                assert after["mechanisms"]["PIPE_ALL"] == 0
                assert "structural node limit" not in result.stderr
                assert "proposed=11" in result.stderr
            else:
                baseline_path = inputs / (name + ".baseline.pto")
                baseline, old_metric = run(baseline_path, {"arguments": arguments})
                assert not compare(baseline, trace), (key, "completion boundary changed")
                assert metric["scalar_counts"] == old_metric["scalar_counts"], (key, "new scalar control")
                for mechanism in ("pto.set_flag", "pto.wait_flag"):
                    assert metric["counts"][mechanism] == old_metric["counts"][mechanism] - removed
                old_static = analyze(baseline_path)["mechanisms"]
                for mechanism in ("sets", "waits"):
                    assert after["mechanisms"][mechanism] == old_static[mechanism] - removed
                for mechanism in ("named", "PIPE_ALL"):
                    assert after["mechanisms"][mechanism] == old_static[mechanism]
        reports[key] = {"mechanisms": after["mechanisms"], "execution": metric,
                        "physical_operations": len(trace.payload)}
Path(prefix + ".report.json").write_text(json.dumps(reports, indent=2) + "\n")
print("15 native loop cases: unchanged payloads, actual token balance, empty/one/two-trip "
      "paths; smaller GEMMs retain every baseline cross-lane cut with fewer commands")
