#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Serial unchanged-input captures with export on/off and finite parity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PERFORMANCE = Path(__file__).resolve().parents[1] / "performance"
sys.path.insert(0, str(PERFORMANCE))
from run_qwen_additions import SERIAL_DRIVER, REVISED_FLAGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(options.python_root.resolve()))
    from check_native_facts import check, walk
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    cases = []
    for row in json.loads((PERFORMANCE / "manifest.json").read_text())["cases"]:
        item = row["sources"]["auto"]
        cases.append((row["case_id"], PERFORMANCE / item["path"], row["level"], item["sha256"]))
    for row in json.loads((PERFORMANCE / "qwen-additions-manifest.json").read_text())["cases"]:
        if row["case_id"] in ("online_softmax", "qk_matmul", "q_proj"):
            cases.append((row["case_id"], ROOT / row["source"], row["level"], row["sha256"]))
    report = {"scope": "native export/output parity and qualified finite local interchange",
              "native_library_sha256": hashlib.sha256((options.python_root / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so").read_bytes()).hexdigest(),
              "rows": [], "failures": [], "device": "not-run"}
    for name, source, level, expected in cases:
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen source changed: " + str(source))
        directory = options.output / name
        directory.mkdir()
        row = {"case": name, "input_sha256": expected, "runs": [], "parity": []}
        for export in (False, True):
            label = "on" if export else "off"
            command = [sys.executable, "-c", SERIAL_DRIVER, str(options.python_root.resolve()),
                       "--pto-arch=a3", "--pto-level=" + level, "--enable-insert-sync", "--emit-pto-ir",
                       *REVISED_FLAGS, str(source), "-o", str(directory / (label + ".pto"))]
            if export:
                command.append("--insert-sync-handoff-facts-dir=" + str(directory / "facts"))
            start = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            (directory / (label + ".stderr")).write_text(result.stderr)
            row["runs"].append({"command": command, "returncode": result.returncode,
                                "seconds": time.monotonic() - start})
            if result.returncode:
                report["failures"].append(name + ": compilation " + label)
                break
        if len(row["runs"]) == 2 and all(item["returncode"] == 0 for item in row["runs"]):
            row["identical_output"] = (directory / "off.pto").read_bytes() == (directory / "on.pto").read_bytes()
            if not row["identical_output"]:
                report["failures"].append(name + ": output changed")
            records = list((directory / "facts").glob("*.json"))
            if not records:
                report["failures"].append(name + ": missing capture")
            for path in records:
                record = json.loads(path.read_text())
                if record["status"] != "supported-local-projection":
                    row["parity"].append({"status": "UNSUPPORTED", "reason": record["reason"]})
                    continue
                with ir.Context() as context:
                    context.enable_multithreading(False)
                    pto.register_dialect(context, load=True)
                    module = ir.Module.parse("module attributes " + record["parent_module_attributes"] + " {\n" + record["input_ir"] + "\n}")
                    function = next(op for op in walk(module.operation) if op.name == "func.func")
                    count = len(function.regions[0].blocks[0].arguments)
                for bound in (0, 1, 2, 16):
                    arguments = [0] * count
                    for binding in record["native"]["parameter_bindings"]:
                        arguments[binding["argument"]] = bound if binding["signed_width"] > 1 else 0
                    try:
                        row["parity"].append(check(record, arguments))
                    except Exception as error:
                        row["parity"].append({"status": "FAIL", "bound": bound, "error": str(error)})
                        report["failures"].append(name + ": parity " + str(bound))
        report["rows"].append(row)
        print(name, "output-identical=" + str(row.get("identical_output")),
              [(item["status"], item.get("reason", item.get("error", ""))) for item in row["parity"]], flush=True)
        (options.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
