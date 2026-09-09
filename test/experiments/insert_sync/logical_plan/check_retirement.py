#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Explicit retirement, token-independence and repeated-invocation challenges.

Native mutation tests challenge fresh reconstruction. Scalar replay checks
actual emitted participation and terminal-drain execution while retaining one
observer across invocations; it does not simulate hardware or check numerics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from compare_boundaries import Boundaries
from measure import attrs, children, replay
from observations import SERIAL_DRIVER, analyze

HERE = Path(__file__).resolve().parent


def main():
    if not __debug__:
        raise RuntimeError("Retirement acceptance requires assertions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--python-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows = []
    commands = []

    def run(name, command):
        start = time.monotonic()
        process = subprocess.run([str(c) for c in command], text=True, capture_output=True,
                                 env=env, timeout=90)
        (args.output / (name + ".stdout")).write_text(process.stdout)
        (args.output / (name + ".stderr")).write_text(process.stderr)
        commands.append({"name": name, "command": [str(c) for c in command],
                         "seconds": time.monotonic() - start, "exit": process.returncode})
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        assert process.returncode == 0, (name, process.returncode, process.stdout[-2000:], process.stderr[-2000:])
        return process

    cases = {
        "retirement_final_mte3": [["src", "dst", n, take]
                                   for n in (0, 1, 3, 0, 2) for take in (False, True)],
        "retirement_final_fix": [["src", "dst"]] * 4,
        "retirement_readerless": [["src", take] for take in (False, True, False, True)],
    }
    # The final drain must not be silently converted by an existing tail hint.
    for case, arguments in list(cases.items()):
        cases[case + "_hint"] = arguments
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    for case, arguments in cases.items():
        basename = case.removesuffix("_hint")
        source = HERE / "inputs" / (basename + ".pto")
        if case.endswith("_hint"):
            text = source.read_text().replace(
                "attributes {pto.kernel_kind =", "attributes {pto.auto_sync_tail_hint = \"setwait_mte3_to_s_event0\", pto.kernel_kind =")
            source = args.output / (case + ".input.pto")
            source.write_text(text)
        answer = json.loads(run(case + ".construct", [args.driver, source, "retirement"]).stdout)
        assert answer["applied"] and answer["invoked"] and answer["export_complete"], answer
        obligations = [r for r in answer["requirements"] if r["kind"] == "retirement"]
        assert {r["source"] for r in obligations} == {p["id"] for p in answer["phases"]}, answer
        assert all(r["target"] == len(answer["phases"]) and
                   r["property"] == "completed-before-kernel-retirement" for r in obligations), answer
        output = args.output / (case + ".logical.pto")
        output.write_text('module attributes {pto.target_arch = "a3"} {\n' + answer["emitted_ir"] + "\n}\n")
        original_metrics, emitted_metrics = analyze(source), analyze(output)
        for key in ("payload", "allocations", "views", "abi"):
            assert original_metrics[key] == emitted_metrics[key], (case, key)
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            original_module = ir.Module.parse(source.read_text())
            emitted_module = ir.Module.parse(output.read_text())
            original = next(op for op in children(original_module.operation) if op.name == "func.func")
            emitted = next(op for op in children(emitted_module.operation) if op.name == "func.func")
            body = list(emitted.regions[0].blocks[0].operations)
            assert body[-1].operation.name == "func.return"
            assert body[-2].operation.name == "pto.barrier"
            assert attrs(body[-2].operation) == {"pipe": "#pto.pipe<PIPE_ALL>"}
            # State is intentionally not reset between invocations. Every run
            # must consume its publications and retire its issued payload.
            observer = Boundaries()
            replays = []
            for values in arguments:
                expected = replay(original, values)
                actual = replay(emitted, values, observer=observer.observe)
                assert actual["payload_sha256"] == expected["payload_sha256"], (case, values)
                assert not observer.tokens, (case, values, "live flags at return")
                assert all(observer.drained.get(pipe, -1) >= point
                           for pipe, point in observer.issued.items()), (case, values, "outstanding payload")
                assert actual["counts"].get("barrier:#pto.pipe<PIPE_ALL>:outside-loop", 0) == 1, (case, values)
                if basename == "retirement_readerless":
                    assert actual["counts"].get("pto.set_flag", 0) == 0
                    assert actual["counts"].get("pto.wait_flag", 0) == 0
                replays.append({"arguments": values, **actual})
        cpp = args.output / (case + ".cpp")
        run(case + ".cpp", [sys.executable, "-c", SERIAL_DRIVER, args.python_root.resolve(),
                            "--pto-arch=a3", "--pto-level=level3", output.resolve(), "-o", cpp.resolve()])
        generated = cpp.read_text()
        # Helper definitions may contain the alternative implementation; only
        # the kernel body after its own definition is relevant to this check.
        kernel_body = generated[generated.index(basename + "("):]
        assert "pipe_barrier(PIPE_ALL)" in kernel_body, (case, "lost terminal ALL in C++")
        assert "ptoas_auto_sync_tail(" not in kernel_body, (case, "retirement became an unchecked tail policy")
        mutations = []
        if not case.endswith("_hint"):
            for mutation in ("erase-retirement", "wrong-retirement-pipe", "early-retirement", "hint-retirement"):
                result = json.loads(run(case + "." + mutation, [args.driver, source, mutation]).stdout)
                assert result["invoked"] and result["changed"] and not result["applied"]
                assert result["original_preserved"] and "retirement drain" in result["reason"], result
                mutations.append({"mutation": mutation, "reason": result["reason"]})
            if basename == "retirement_readerless":
                result = json.loads(run(case + ".unconsumed", [args.driver, source, "unconsumed-before-retirement"]).stdout)
                assert result["invoked"] and result["changed"] and not result["applied"] and result["original_preserved"], result
                mutations.append({"mutation": "unconsumed-before-retirement", "reason": result["reason"]})
        rows.append({"case": case, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "work": answer["work"], "retirement_obligations": len(obligations),
                     "mechanisms": emitted_metrics["mechanisms"], "replays": replays, "mutations": mutations})
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(case, "passed", flush=True)
    print("Retirement reconstruction and repeated-invocation replay passed; device qualification is separate")


if __name__ == "__main__":
    main()
