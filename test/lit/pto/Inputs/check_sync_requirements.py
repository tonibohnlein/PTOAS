# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Native requirement sharing: selected protocols, ordinary repair and guarded fallback."""
import pathlib
import re
import subprocess
import sys

binary, source, prefix = sys.argv[1:]
original = pathlib.Path(source).read_text()
loop = re.search(r"scf.for (%[\w]+) = (%[\w]+) to (%[\w]+) step (%[\w]+)", original)
assert loop, original
upper = loop[3]
constant = next(name for name, value in re.findall(r"(%[\w]+) = arith.constant (\d+) : index", original)
                if value == "16")
variants = {
    "symbolic": original,
    "bounded": original.replace(f"to {upper} step", f"to {constant} step"),
    # Both slots write the very same output region: there is no partition proof.
    "overlap": re.sub(r"(pto.partition_view %\w+, offsets = )\[%\w+, (%\w+)\]", r"\1[\2, \2]", original),
}
for name, source in variants.items():
    path = pathlib.Path(prefix + "." + name + ".pto")
    output = pathlib.Path(prefix + "." + name + ".out.pto")
    path.write_text(source)
    command = [binary, str(path), "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false",
               "--pto-insert-sync=buffer-generations=true defer-same-pipe=true gm-alias=assume-disjoint-arguments",
               "-o", str(output)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    emitted = output.read_text()
    assert "lifecycle_channels = 4" in emitted, run.stderr
    assert "pto.barrier <PIPE_ALL>" not in emitted
    barriers = emitted.count("pto.barrier <PIPE_MTE3>")
    if name == "symbolic":
        assert barriers == 2, emitted
        assert "generation_storage_barriers_guarded = 2" in emitted
        assert emitted.count("pto.insert_sync.frontier_overflow_guard") == 2
        assert "arith.cmpi sgt" in emitted
    elif name == "bounded":
        assert barriers == 0, emitted
        witnesses = re.search(r"generation_global_witnesses = (\d+)", emitted)
        assert witnesses and int(witnesses[1]) > 0, emitted
    else:
        assert barriers >= 2, emitted
        assert "pto.insert_sync.frontier_overflow_guard" not in emitted
print("Shared requirements: symbolic guards, bounded direct repair, overlapping-store fallback passed")
