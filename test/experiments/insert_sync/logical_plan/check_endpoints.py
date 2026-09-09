#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Challenge actual domain coalescing and retained overlapping writes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from compare_boundaries import Boundaries
from measure import children, replay

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Endpoint acceptance requires assertions")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.python_root.resolve()))
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    original = (HERE / "inputs/alternative_producers.pto").read_text()
    start = original.index("    scf.if %p {")
    end = original.index("    pto.tabs", start)
    load = next(line for line in original.splitlines() if "pto.tload" in line).strip()
    overlap = original[:start] + "    " + load + "\n    scf.if %p {\n      " + load + "\n    }\n    scf.if %q {\n      " + load + "\n    }\n" + original[end:]
    overlap = overlap.replace("%p: i1)", "%p: i1, %q: i1)")
    summary = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for name, source, arguments in (
            ("alternatives", original, [["src", p] for p in (0, 1)]),
            ("straight", original[:start] + "    " + load + "\n" + original[end:], [["src", 0]]),
            ("overlap", overlap, [["src", p, q] for p in (0, 1) for q in (0, 1)])):
        path = args.output / (name + ".pto")
        path.write_text(source)
        result = subprocess.run([str(args.driver), str(path), "retirement"],
                                text=True, capture_output=True, env=env, timeout=60)
        (args.output / (name + ".stdout")).write_text(result.stdout)
        (args.output / (name + ".stderr")).write_text(result.stderr)
        assert result.returncode == 0, (name, result.stderr)
        answer = json.loads(result.stdout)
        assert answer["applied"] and answer["invoked"], answer
        emitted = "module attributes {pto.target_arch = \"a3\"} {\n" + answer["emitted_ir"] + "\n}"
        (args.output / (name + ".logical.pto")).write_text(emitted)
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            modules = [ir.Module.parse(text) for text in (source, emitted)]
            functions = [next(op for op in children(m.operation) if op.name == "func.func") for m in modules]
            body = list(children(functions[1]))
            waits = [op for op in body if op.name == "pto.wait_flag"]
            # Distinct guarded production streams share one unconditional
            # acquisition, proving coalescing actually happened, not merely CSE.
            assert len(waits) == 1, (name, emitted)
            results = []
            for values in arguments:
                expected = replay(functions[0], values)
                observer = Boundaries()
                actual = replay(functions[1], values, observer=observer.observe)
                assert expected["payload_sha256"] == actual["payload_sha256"]
                assert not observer.tokens
                assert actual["counts"].get("pto.wait_flag", 0) == 1
                loads = [i for i, point in enumerate(observer.before) if point["lane"] == "PIPE_MTE2"]
                if name == "overlap":
                    for previous, current in zip(loads, loads[1:]):
                        assert observer.before[current]["completed"].get("PIPE_MTE2", -1) >= previous
                consumer = next(point for point in observer.before if point["lane"] == "PIPE_V")
                assert consumer["completed"].get("PIPE_MTE2", -1) == loads[-1]
                results.append({"arguments": values, "commands": actual["counts"]})
            summary.append({"case": name, "observations": results})
        negative = subprocess.run([str(args.driver), str(path), "erase-wait"],
                                  text=True, capture_output=True, env=env, timeout=60)
        (args.output / (name + ".negative.json")).write_text(negative.stdout)
        assert negative.returncode == 0, negative.stderr
        rejected = json.loads(negative.stdout)
        assert rejected["changed"] and not rejected["applied"] and rejected["original_preserved"], rejected
        if name == "straight":
            for mutation in ("duplicate-episode", "orphan-head-episode", "orphan-tail-episode"):
                negative = subprocess.run([str(args.driver), str(path), mutation], text=True,
                                          capture_output=True, env=env, timeout=60)
                (args.output / (name + "." + mutation + ".json")).write_text(negative.stdout)
                assert negative.returncode == 0, (mutation, negative.stdout, negative.stderr)
                rejected = json.loads(negative.stdout)
                assert rejected["changed"] and not rejected["applied"] and rejected["original_preserved"], rejected
                assert "payload cut" in rejected["reason"], rejected
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Endpoint coalescing: alternatives, real overlaps and missing-wait negatives passed")


if __name__ == "__main__":
    main()
