# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise module delegation using a prepared corpus cube/vector FIFO kernel."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    opt, probe, fixture, *modes = sys.argv[1:]
    library = modes == ["--library"]
    if modes and not library:
        raise ValueError("expected optional --library")
    source = Path(fixture).read_text()
    parts = source.split("  func.func ")
    prefix = parts[0]
    functions = ["  func.func " + part for part in parts[1:]]
    functions[-1] = functions[-1][:functions[-1].rfind("\n}")]
    stem = "matmul_tpush_tpop_print"
    with tempfile.TemporaryDirectory(prefix="closed-callees-") as directory:
        path = Path(directory) / "case.pto"

        def run(text, flags, expected=True):
            path.write_text(text)
            if library:
                policy = "may-alias" if any("gm-alias=may-alias" in flag for flag in flags) else "may-not-alias"
                result = subprocess.run([probe, "--gm-alias=" + policy, "--insert-module-library", str(path)],
                                        capture_output=True, text=True, check=False, timeout=60)
                if "failed module preparation mutated original IR" in result.stderr:
                    raise RuntimeError(result.stderr)
                allocate = any("pto-frontier-allocate" in flag for flag in flags)
                if result.returncode == 0 and allocate:
                    certificates = ("pto.cyclic_allocation", "pto.finite_allocation")
                    if not any(name in result.stdout for name in certificates):
                        raise RuntimeError("module preparation did not attach allocation certificates")
                    logical = Path(directory) / "logical.pto"
                    logical.write_text(result.stdout)
                    allocation_flags = [flag.replace("pto-frontier-analysis,", "") for flag in flags
                                        if flag.startswith("--pass-pipeline=")]
                    if not allocation_flags:
                        allocation_flags = ["--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"]
                    result = subprocess.run([opt, *allocation_flags, str(logical)],
                                            capture_output=True, text=True, check=False, timeout=60)
                    if result.returncode == 0 and any(name in result.stdout for name in certificates):
                        raise RuntimeError("allocation did not consume nested leaf certificates")
            else:
                result = subprocess.run([opt, *flags, str(path)], capture_output=True,
                                        text=True, check=False, timeout=60)
            assert (result.returncode == 0) == expected, result.stdout + result.stderr
            return result

        flags = ["--pto-frontier-analysis", "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"]
        for program in (source, prefix + "".join(reversed(functions)) + "\n}\n",
                        source.replace(stem, "unrelated_name")):
            result = run(program, flags)
            assert "pto.logical_set" not in result.stdout and "pto.logical_wait" not in result.stdout
            assert result.stdout.count("pto.barrier <PIPE_ALL>") == 2, result.stdout
            for operation in ("pto.tpush", "pto.tpop", "pto.tfree", "call @"):
                assert result.stdout.count(operation) == program.count(operation), operation
        nested = "module {\n" + source[source.index("module attributes"):] + "\n}\n"
        nested_result = run(nested, ["--pass-pipeline=builtin.module(pto-frontier-analysis,"
                                    "builtin.module(func.func(pto-frontier-allocate{eligible-ids=0,1,2,3,4,5})))"])
        assert "pto.logical_set" not in nested_result.stdout
        # Actuals can alias despite different formal names. Called bodies use
        # the same may-alias model under either root policy.
        aliased = source.replace("_cube(%arg0, %arg1, %arg2)", "_cube(%arg0, %arg0, %arg0)")
        loose = run(aliased, ["--pto-frontier-analysis=gm-alias=may-not-alias"])
        strict = run(aliased, ["--pto-frontier-analysis=gm-alias=may-alias"])
        assert loose.stdout == strict.stdout
        path.write_text(source)
        recognized = subprocess.run([probe, "--recognize", str(path)], check=True,
                                   capture_output=True, text=True, timeout=60)
        reports = [json.loads(line.removeprefix("closed-callee-json "))
                   for line in recognized.stdout.splitlines() if line.startswith("closed-callee-json ")]
        assert len(reports) == 1 and reports[0]["state"] == "applicable", reports
        assert not reports[0]["closure_established"]
        assert reports[0]["callees"] == [stem + "_cube", stem + "_vector"]
        call = next(line for line in source.splitlines() if "call @" + stem + "_cube" in line)
        manual_event = "pto.set_flag [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_V>, #pto.event<EVENT_ID0>]"
        variants = [source.replace(call, call + "\n" + call),
                    source.replace(call, call.replace(stem + "_cube", stem)),
                    source.replace("pto.tprint ins(%1", manual_event + "\n    pto.tprint ins(%1")]
        for program in variants:
            rejected = run(program, ["--pto-frontier-analysis"], False)
            assert "closed-callee composition" in rejected.stderr
        manual_caller = source.replace(call, "    " + manual_event + "\n" + call)
        rejected = run(manual_caller, ["--pto-frontier-analysis"], False)
        assert "event-lifetime contract" in rejected.stderr
        # Multiple scalar helper calls do not fit the delegation route but
        # remain supported by ordinary shared scalar analysis (corpus table helper).
        scalar = """module attributes {pto.target_arch = "a3"} {
          func.func @caller(%p: !pto.ptr<i64, gm>, %v: i64) {
            call @store(%p, %v) : (!pto.ptr<i64, gm>, i64) -> ()
            call @store(%p, %v) : (!pto.ptr<i64, gm>, i64) -> ()
            return
          }
          func.func private @store(%p: !pto.ptr<i64, gm>, %v: i64) {
            %zero = arith.constant 0 : index
            pto.store %v, %p[%zero] : !pto.ptr<i64, gm>, i64
            return
          }
        }"""
        scalar_result = run(scalar, flags)
        assert "pto.set_flag" not in scalar_result.stdout
        # Recognition of the wrapper is not proof that its callees compile.
        declaration = "  func.func private @unavailable()\n"
        broken = source.replace("    pto.tprint ins(%1", "    call @unavailable() : () -> ()\n    pto.tprint ins(%1")
        broken = broken[:broken.rfind("\n}")] + "\n" + declaration + "}\n"
        run(broken, ["--pto-frontier-analysis"], False)
    print("closed callees: corpus delegation, symbol order, names, scopes, aliases and rejection checks passed")


if __name__ == "__main__":
    main()
