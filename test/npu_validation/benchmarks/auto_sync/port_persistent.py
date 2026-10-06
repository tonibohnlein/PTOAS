#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Persistent GEMM port using the already audited intrinsic GEMM body.

The source default domain is [64, 4]. Persistent's grouping factor is four,
so task = wave*20 + cid maps to (task//4, task%4), with a final partial wave.
The bank pipeline is identical to the existing intrinsic port's body.
"""

from pathlib import Path
import re


def persistent_gemm():
    source = Path(__file__).parent / "prepared/tilelang/expert.pto"
    text = source.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.startswith("// Source:"))
    text = text.replace("@tilelang_gemm", "@persistent_gemm")
    text = text.replace("    %c4096 = arith.constant 4096 : index", """    %c8192 = arith.constant 8192 : index
    %c1024 = arith.constant 1024 : index
    %c20 = arith.constant 20 : index
    %c13 = arith.constant 13 : index
    %c32 = arith.constant 32 : index
    %cid_i64 = pto.get_block_idx
    %cid = arith.index_cast %cid_i64 : i64 to index""")
    text = text.replace("shape = [%c256, %c4096], strides = [%c4096, %c1]",
                        "shape = [%c8192, %c8192], strides = [%c8192, %c1]")
    text = text.replace("shape = [%c4096, %c256], strides = [%c256, %c1]",
                        "shape = [%c8192, %c1024], strides = [%c1024, %c1]")
    text = text.replace("shape = [%c256, %c256], strides = [%c256, %c1]",
                        "shape = [%c8192, %c1024], strides = [%c1024, %c1]")
    text = text.replace("    scf.for %tile = %c0 to %c2 step %c1 {", """    scf.for %wave = %c0 to %c13 step %c1 {
      %wave_start = arith.muli %wave, %c20 : index
      %task = arith.addi %wave_start, %cid : index
      %valid_task = arith.cmpi slt, %task, %c256 : index
      scf.if %valid_task {
      %tile = arith.divui %task, %c4 : index
      %tile_col = arith.remui %task, %c4 : index
      %col = arith.muli %tile_col, %c256 : index""")
    text = text.replace("%out = pto.partition_view %cv, offsets = [%row, %c0]",
                        "%out = pto.partition_view %cv, offsets = [%row, %col]")
    text = text.replace("%bp0 = pto.partition_view %bv, offsets = [%c0, %c0]",
                        "%bp0 = pto.partition_view %bv, offsets = [%c0, %col]")
    text = text.replace("%bp = pto.partition_view %bv, offsets = [%column, %c0]",
                        "%bp = pto.partition_view %bv, offsets = [%column, %col]")
    text = text.replace("to %c16 step", "to %c32 step").replace("%next, %c16 :", "%next, %c32 :")
    text = text.replace("[#pto.pipe<PIPE_FIX>, #pto.pipe<PIPE_M>, %c1]",
                        "[#pto.pipe<PIPE_FIX>, #pto.pipe<PIPE_M>, %c0]")
    end = "      pto.set_flag_dyn [#pto.pipe<PIPE_FIX>, #pto.pipe<PIPE_M>, %c0]\n    }"
    text = text.replace(end, end + "\n    }")
    # Match the source's priming and draining order as well as its IDs.
    for action in ("set", "wait"):
        flag = f"    pto.{action}_flag_dyn [#pto.pipe<PIPE_FIX>, #pto.pipe<PIPE_M>, %c0]"
        lines = text.splitlines()
        indices = [i for i, line in enumerate(lines) if line == flag]
        if len(indices) != 1:
            raise ValueError("Existing GEMM fixture protocol changed")
        item = lines.pop(indices[0])
        last = f"    pto.{action}_flag_dyn [#pto.pipe<PIPE_M>, #pto.pipe<PIPE_MTE1>, %c1]"
        lines.insert(lines.index(last) + 1, item)
        text = "\n".join(lines) + "\n"
    bare = "\n".join(line for line in text.splitlines()
                       if not re.search(r"\bpto\.(set_flag_dyn|wait_flag_dyn|barrier)\b", line)) + "\n"
    return {"expert": text, "input": bare}, {
        "shape": {"M": 8192, "N": 1024, "K": 8192}, "blocks": [128, 256, 64],
        "K_L1": 256, "stages": [2, 2], "launch_blocks": 20, "waves": 13,
        "scope": "Full source default shape and persistent task mapping; NZ L1 layout; terminal completion added."}
