# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check selector contracts, guard recipes and exact gap-preserving byte unions."""
import re
import shutil
import subprocess
import sys


def route(block, name):
    match = re.search(r"recognize " + name + r": ([^\n]+)\n(.*?)(?=^recognize |^program-json |\Z)",
                      block, re.MULTILINE | re.DOTALL)
    assert match, name
    return match.group(1).split()[0], match.group(2)


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing recognition executable")
    run = subprocess.run([tool, "--recognize", sys.argv[2]], check=True, capture_output=True,
                         text=True, timeout=45)
    blocks = {block.splitlines()[0]: block for block in
              re.split(r"^recognition ", run.stdout, flags=re.MULTILINE)[1:]}
    assert len(blocks) == 8
    state, text = route(blocks["parameter_rotation"], "guarded-rotating")
    assert state == "applicable", text
    assert "parameters=1" in text and "stride=1" in text and "atom=[0,4)" in text, text
    state, text = route(blocks["recomputed_invariant_guard"], "guarded-rotating")
    assert state == "applicable", text
    assert "entry-guards=late entry-expressions=1" in text, text
    state, text = route(blocks["varying_guard"], "guarded-rotating")
    assert state != "applicable" and "issue guard-invariance" in text, text
    for name in ("signed_unknown_selector", "nonpower_wrapping_selector"):
        state, text = route(blocks[name], "guarded-rotating")
        assert state != "applicable", (name, text)
        assert "issue slot-expression" in text or "issue index-arithmetic" in text, (name, text)
    state, text = route(blocks["yielded_payload_result"], "finite-guarded")
    assert state == "missing-premise" and "issue additional-prerequisite" in text, text
    for name, expected in {
        "partial_overlap_ranges": [(0, 64, 1, 1, 0), (64, 128, 2, 1, 1), (128, 192, 1, 0, 1)],
        "gap_ranges": [(0, 32, 1, 1, 0), (64, 96, 2, 1, 1), (128, 160, 1, 0, 1)],
    }.items():
        state, text = route(blocks[name], "rotating")
        assert state == "applicable", (name, text)
        atoms = re.findall(r"atom=\[(\d+),(\d+)\).*?effects=(\d+) reads=(\d+) writes=(\d+)", text)
        actual = sorted(tuple(map(int, atom)) for atom in atoms)
        assert actual == expected, (name, actual, text)
    print("extended rotation: parameter offset, guard recipe, three rejections, exact gaps and yielded prerequisites")


if __name__ == "__main__":
    main()
