# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare module-owned allocation under serial and threaded pass managers."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from check_frontier_public_pipeline import unchanged_failure


def run(tool, source, *flags):
    result = subprocess.run([tool, *flags, "-"], input=source, text=True,
                            capture_output=True, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout


def multiply(source, count):
    start = source.index("  func.func ")
    end = source.rfind("}")
    body = source[start:end]
    name = re.search(r"func.func @([a-zA-Z0-9_]+)", body).group(1)
    return source[:start] + "\n".join(body.replace("@" + name, "@" + name + str(i))
                                       for i in range(count)) + source[end:]


def check(tool, saved, marker):
    if saved.count(marker) < 4:
        raise RuntimeError("fixture did not exercise multiple allocation plans: " + marker)
    unchanged_failure(tool, saved, ["--pto-frontier-allocate"])
    allocate = "pto-frontier-allocate{eligible-ids=0,1,2,3,4,5}"
    serial = run(tool, saved, "--mlir-disable-threading", "--pass-pipeline=builtin.module(" + allocate + ")")
    for _ in range(3):
        threaded = run(tool, saved, "--pass-pipeline=builtin.module(" + allocate + ")")
        if threaded != serial or "pto.logical_" in threaded:
            raise RuntimeError("threaded and serial allocation differ")
    # Parallel adaptor tasks own distinct modules; each contains several functions.
    start = saved.index("module ")
    module = saved[start:]
    nested = saved[:start] + "module {\n" + "\n".join(
        module.replace("module ", "module @partition" + str(i) + " ", 1) for i in range(2)) + "\n}"
    pipeline = "--pass-pipeline=builtin.module(builtin.module(" + allocate + "))"
    serial = run(tool, nested, "--mlir-disable-threading", pipeline)
    if run(tool, nested, pipeline) != serial:
        raise RuntimeError("parallel module allocations differ")


def main():
    tool = shutil.which(sys.argv[1])
    if not tool:
        raise RuntimeError("missing optimizer")
    tool = str(Path(tool).resolve(strict=True))
    # LLVM's default pool observes Linux CPU affinity. Bound total test workers.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    fixtures = Path(sys.argv[2])
    for name, options, marker in (
            ("sync_mixed_deferred_output.pto", "deferred-output=true", "pto.deferred_family ="),
            ("frontier_sections.pto", "", "pto.sync_context_plans =")):
        source = multiply((fixtures / name).read_text(encoding="utf-8"), 4)
        saved = run(tool, source, "--mlir-disable-threading", "--pto-frontier-analysis=" + options)
        check(tool, saved, marker)
    counters = '''module attributes {pto.target_arch = "a3"} {
  func.func @counter(%n: index) attributes {
    pto.cyclic_allocation = {version = 2 : i64, plan = 0 : i64, scope = "function",
      strategy = "executed-family-counters", total_budget = 2 : i64,
      families = [{record = 0 : i64, source = 4 : i64, target = 3 : i64, width = 2 : i64}]}
  } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
      pto.logical_set [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 0 ordinal %zero members(%i)
      pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 0 ordinal %zero members(%i)
    }
    return
  }
}'''
    saved_counters = multiply(counters, 4)
    check(tool, saved_counters, "executed-family-counters")
    prefix, suffix = saved_counters.rsplit("width = 2 : i64", 1)
    unchanged_failure(tool, prefix + "width = 7 : i64" + suffix,
                      ["--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"])
    print("module allocation: deferred, context and counter multi-function serial/threaded agreement")


if __name__ == "__main__":
    main()
