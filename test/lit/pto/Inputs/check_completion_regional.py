# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Qualify scoped completion construction and threaded production insertion."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

from check_allocation_module_ownership import multiply, run


PREFIX = """!tile = !pto.tile_buf<vec, 1x8xf32>
module attributes {pto.target_arch = "a3", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @scopes() attributes {test.many_completion} {
    %addr = arith.constant 0 : i64
    %one = arith.constant 1.0 : f32
    %tile = pto.alloc_tile addr = %addr : !tile
"""
SUFFIX = "  return\n  }\n}\n"


def storage_scaling(optimizer):
    for count in (8, 32, 128, 512):
        header = PREFIX[:PREFIX.index("  func.func")] + "  func.func @classes(%unknown: i64) {\n"
        body = "    %one = arith.constant 1.0 : f32\n"
        for index in range(count):
            body += f"    %addr{index} = arith.constant {index * 64} : i64\n"
            body += f"    %tile{index} = pto.alloc_tile addr = %addr{index} : !tile\n"
            body += f"    pto.texpands ins(%one : f32) outs(%tile{index} : !tile)\n"
        flags = ("--pto-frontier-analysis=completion-only=true",)
        known = run(optimizer, header + body + SUFFIX, *flags)
        if f"storage_classes = {count}" not in known or known.count("pto.barrier <PIPE_ALL>") != 1:
            raise RuntimeError("disjoint reservation scaling lost storage classes or added a conflict")
        body += "    %any = pto.alloc_tile addr = %unknown : !tile\n"
        body += "    pto.texpands ins(%one : f32) outs(%any : !tile)\n"
        wildcard = run(optimizer, header + body + SUFFIX, *flags)
        if f"storage_classes = {count}" not in wildcard or wildcard.count("pto.barrier <PIPE_ALL>") != 2:
            raise RuntimeError("unknown reservation failed to conflict with existing classes")
    print("completion classes: 8/32/128/512 disjoint and wildcard public pipelines passed")


def main():
    tools = [shutil.which(name) for name in sys.argv[1:3]]
    if len(tools) != 2 or not all(tools):
        raise RuntimeError("missing compiler validation tools")
    checker, optimizer = [str(Path(tool).resolve(strict=True)) for tool in tools]
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    for count in (8, 32, 128, 512):
        source = PREFIX + ('    "pto.texpands"(%one, %tile) {test.region_root} : (f32, !tile) -> ()\n' * count) + SUFFIX
        result = subprocess.run([checker, "--completion-region-checks", "-"], input=source,
                                capture_output=True, text=True, timeout=60, check=False)
        expected = f"regional scaling: children={count} scoped-visits={count} phase-visits={count} vertices={2 * count}"
        if result.returncode or expected not in result.stdout:
            raise RuntimeError(result.stderr + result.stdout)
    storage_scaling(optimizer)
    source = PREFIX.replace("test.many_completion", "test.production_parent") + """
    pto.texpands ins(%one : f32) outs(%tile : !tile)
    pto.tadds ins(%tile, %one : !tile, f32) outs(%tile : !tile)
    %false = arith.constant false
    scf.while : () -> () {
      scf.condition(%false)
    } do {
      scf.yield
    }
""" + SUFFIX
    source = multiply(source, 4)
    flags = ("--pto-frontier-analysis", "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5")
    serial = run(optimizer, source, "--mlir-disable-threading", *flags)
    for _ in range(3):
        if run(optimizer, source, *flags) != serial:
            raise RuntimeError("threaded regional fallback changed output")
    if serial.count("pto.barrier <PIPE_V>") != 4 or serial.count("pto.completion_protocol =") != 4:
        raise RuntimeError("production fallback did not preserve the exact children's local barriers")
    print("regional completion: four geometric scopes and serial/threaded multi-function pipelines passed")


if __name__ == "__main__":
    main()
