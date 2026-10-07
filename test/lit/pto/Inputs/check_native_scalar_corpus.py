# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native scalar protection needs no address or loop dependence solver."""
from pathlib import Path
import re
import subprocess
import sys


def run(tool, source, option):
    return subprocess.run([tool, "--mlir-disable-threading", option], input=source,
                          text=True, capture_output=True, timeout=30)


def main():
    tool = sys.argv[1]
    samples = Path(sys.argv[2]).read_text().split("// -----")
    for source in samples:
        logical = run(tool, source, "--pto-frontier-analysis")
        assert logical.returncode == 0, logical.stderr
        assert "pto.logical_set" not in logical.stdout
        assert "pto.logical_wait" not in logical.stdout
        assert "<PIPE_S>" not in logical.stdout
        assert logical.stdout.count("pto.barrier <PIPE_ALL>") == 1
        for op in ("pto.load", "pto.store", "scf.for", "scf.if"):
            assert source.count(op) == logical.stdout.count(op), op
        physical = run(tool, logical.stdout,
                       "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5")
        assert physical.returncode == 0, physical.stderr
        assert not re.search(r"pto.(?:set_flag|wait_flag)", physical.stdout)
    # The same function with an explicit protocol cannot use the empty-plan proof.
    barrier = samples[0].replace("    return", "    pto.barrier #pto.pipe<PIPE_S>\n    return")
    rejected = run(tool, barrier, "--pto-frontier-analysis")
    assert rejected.returncode != 0
    print("native scalar corpus: three structured programs allocate with zero event IDs")


if __name__ == "__main__":
    main()
