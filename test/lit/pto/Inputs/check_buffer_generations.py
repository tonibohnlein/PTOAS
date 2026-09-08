# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Native generation acceptance and equivalent-expression regression."""
import collections
import pathlib
import re
import subprocess
import sys

binary, source, prefix = sys.argv[1:]
original = pathlib.Path(source).read_text()
constants = {name: int(value) for name, value in re.findall(
    r"(%[\w]+) = arith.constant (-?\d+) : index", original)}
selected = []
for result, iv, divisor in re.findall(
        r"(%[\w]+) = arith.remsi (%[\w]+), (%[\w]+) : index", original):
    if constants.get(divisor) != 2:
        continue
    loop = re.search(r"scf.for " + re.escape(iv) + r" = (%[\w]+) to .*? step (%[\w]+)", original)
    if not loop or constants.get(loop[1]) != 0 or constants.get(loop[2]) != 1:
        continue
    compare = re.search(r"arith.cmpi eq, " + re.escape(result) + r", (%[\w]+) : index", original)
    if compare and constants.get(compare[1]) == 0:
        selected.append((result, iv, divisor, loop[2], compare.group()))
assert len(selected) == 1, selected
result, iv, divisor, one, compare = selected[0]
variants = {
    "signed_eq": original,
    "signed_ne": original.replace(compare, f"arith.cmpi ne, {result}, {one} : index"),
    "unsigned_eq": original.replace(f"{result} = arith.remsi {iv}, {divisor}",
                                    f"{result} = arith.remui {iv}, {divisor}"),
}
counts = []
for name, text in variants.items():
    path = pathlib.Path(prefix + "." + name + ".input.pto")
    output = pathlib.Path(prefix + "." + name + ".output.pto")
    path.write_text(text)
    command = [binary, str(path), "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false",
               "--pto-insert-sync=buffer-generations=true defer-same-pipe=true "
               "mmad-chains=true gm-alias=assume-disjoint-arguments", "-o", str(output)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    emitted = output.read_text()
    assert "pto.insert_sync.buffer_generations" in emitted, run.stderr
    assert "lifecycle_channels = 6" in emitted, emitted.splitlines()[1]
    pipes = collections.Counter(re.findall(r"pto.barrier <(PIPE_\w+)>", emitted))
    assert pipes["PIPE_MTE2"] == pipes["PIPE_MTE1"] == 0, pipes
    assert not pipes, pipes
    sets, waits = emitted.count("pto.set_flag"), emitted.count("pto.wait_flag")
    assert sets == waits, (sets, waits)
    counts.append((sets, waits, pipes))
assert counts[0] == counts[1] == counts[2], counts
print("GEMM generation construction and parity equivalence passed:", counts[0])
