#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Challenge reconstructed cuts by moving events without moving payload.

The compiler mutation callback changes actual emitted endpoints on its clone.
Positive construction and concrete participation are mandatory first; failures
must preserve the original unsynchronized input and the static event inventory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from check_scalability import source_for
from compare_boundaries import Boundaries
from measure import children, replay
from observations import analyze

HERE = Path(__file__).resolve().parent


def main():
    if not __debug__:
        raise RuntimeError("Cut reconstruction acceptance requires assertions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.python_root.resolve()))
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    fanout = source_for(3, False)
    independent = source_for(2, False)
    load = next(line for line in independent.splitlines() if "pto.tload" in line)
    independent = independent.replace(load,
        '    %other_address = arith.constant 1536 : i64\n'
        '    %other_input = pto.alloc_tile addr = %other_address : !pto.tile_buf<vec, 16x16xf16>\n'
        + load + "\n" + load.replace("outs(%input", "outs(%other_input"))
    independent = independent.replace(
        'pto.tabs ins(%input : !pto.tile_buf<vec, 16x16xf16>) outs(%output1',
        'pto.tabs ins(%other_input : !pto.tile_buf<vec, 16x16xf16>) outs(%output1')
    looping = fanout.replace('@reader_fanout(%src: !pto.ptr<f16>)',
                             '@reader_fanout(%src: !pto.ptr<f16>, %trip: index)')
    start = looping.index("    pto.tload")
    end = looping.index("    return", start)
    looping = (looping[:start] + "    scf.for %iv = %c0 to %trip step %c1 {\n" +
               "\n".join("  " + line for line in looping[start:end].splitlines()) +
               "\n    }\n" + looping[end:])
    import re
    writechain = re.sub(r'outs\(%output[0-9]+', 'outs(%input', fanout)
    cases = (
        ("writechain", writechain, [["src"]] * 3),
        ("fanout", fanout, [["src"]] * 3),
        ("independent_preload", independent, [["src"]] * 3),
        ("looping_fanout", looping, [["src", trip] for trip in (0, 1, 3, 0, 2)]),
    )
    rows = []

    def run(name, path, mutation):
        process = subprocess.run([str(args.driver.resolve()), str(path.resolve()), mutation],
                                 env=env, text=True, capture_output=True, timeout=90)
        (args.output / (name + ".stdout")).write_text(process.stdout)
        (args.output / (name + ".stderr")).write_text(process.stderr)
        assert process.returncode == 0, (name, process.returncode, process.stdout[-3000:], process.stderr[-3000:])
        return json.loads(process.stdout)

    for name, source, arguments in cases:
        path = args.output / (name + ".input.pto")
        path.write_text(source)
        positive = run(name + ".positive", path, "retirement")
        assert positive["applied"] and positive["invoked"], positive
        output = args.output / (name + ".logical.pto")
        output.write_text('module attributes {pto.target_arch = "a3"} {\n' + positive["emitted_ir"] + "\n}\n")
        original_projection, actual_projection = analyze(path), analyze(output)
        for key in ("payload", "allocations", "views", "abi"):
            assert original_projection[key] == actual_projection[key], (name, key)
        observations = []
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            modules = [ir.Module.parse(p.read_text()) for p in (path, output)]
            original, actual = [next(op for op in children(m.operation) if op.name == "func.func") for m in modules]
            observer = Boundaries()
            for values in arguments:
                begin = len(observer.before)
                expected = replay(original, values)
                result = replay(actual, values, observer=observer.observe)
                assert result["payload_sha256"] == expected["payload_sha256"]
                assert not observer.tokens
                assert all(observer.drained.get(pipe, -1) >= point for pipe, point in observer.issued.items())
                if name == "independent_preload":
                    loads = [p for p in range(begin, len(observer.before))
                             if observer.before[p]["lane"] == "PIPE_MTE2"]
                    first_reader = next(p for p in range(begin, len(observer.before))
                                        if observer.before[p]["lane"] == "PIPE_V")
                    acquired = observer.before[first_reader]["completed"].get("PIPE_MTE2", -1)
                    assert len(loads) == 2 and acquired == loads[0] and acquired < loads[1]
                observations.append({"arguments": values, "counts": result["counts"]})
        mutations = ["publication-before-source", "acquisition-after-consumer", "unrepresented-event-lane"]
        if name == "writechain":
            mutations += ["barrier-after-consumer", "unrepresented-barrier-lane"]
        if name == "independent_preload":
            mutations.append("publication-after-independent-load")
        rejected = []
        for mutation in mutations:
            result = run(name + "." + mutation, path, mutation)
            assert result["invoked"] and result["changed"] and result["counts_preserved"], result
            assert not result["applied"] and result["original_preserved"], result
            assert "original payload" not in result["reason"], (name, "mutation moved payload instead of endpoint", result)
            expected_reason = ("barrier" if mutation in ("barrier-after-consumer", "unrepresented-barrier-lane") else
                               "event domain" if mutation == "unrepresented-event-lane" else "handoff boundary")
            assert expected_reason in result["reason"], result
            rejected.append({"mutation": mutation, "reason": result["reason"]})
        rows.append({"case": name, "observations": observations, "rejected": rejected,
                     "mechanisms": actual_projection["mechanisms"], "work": positive["work"]})
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(name, "actual endpoint movement rejected; positive reconstruction passed", flush=True)
    print("Fresh cut reconstruction: straight-line and looping positives, endpoint movement and absent-lane negatives passed")


if __name__ == "__main__":
    main()
