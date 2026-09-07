#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate synthetic ready/free controls; these are not verbatim vendor examples."""

from pathlib import Path


def generate(depth, uses=1):
    """Keep computation and storage identical between manual and automatic arms."""
    lines = [
        "module {",
        "  func.func @pipeline(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>, %trip: index)",
        "      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {",
        "    %c0 = arith.constant 0 : index",
        "    %c1 = arith.constant 1 : index",
        "    %c16 = arith.constant 16 : index",
        f"    %depth = arith.constant {depth} : index",
        f"    %uses = arith.constant {uses} : index",
        "    %rows = arith.muli %trip, %c16 : index",
        "    %outrows = arith.muli %rows, %uses : index",
        "    %source = pto.make_tensor_view %src, shape = [%rows, %c16], "
        "strides = [%c16, %c1] : !pto.tensor_view<?x?xf16>",
        "    %target = pto.make_tensor_view %dst, shape = [%outrows, %c16], "
        "strides = [%c16, %c1] : !pto.tensor_view<?x?xf16>",
    ]
    tile = "!pto.tile_buf<vec, 16x16xf16>"
    view = "!pto.partition_tensor_view<16x16xf16>"

    def flag(action, source, target, slot, indent="    "):
        lines.append(f"{indent}pto.{action}_flag[<PIPE_{source}>, <PIPE_{target}>, <EVENT_ID{slot}>]")

    for slot in range(depth):
        lines.extend(
            [
                f"    %slot{slot} = arith.constant {slot} : index",
                f"    %ia{slot} = arith.constant {slot * 512} : i64",
                f"    %oa{slot} = arith.constant {(slot + depth) * 512} : i64",
                f"    %in{slot} = pto.alloc_tile addr = %ia{slot} : {tile}",
                f"    %out{slot} = pto.alloc_tile addr = %oa{slot} : {tile}",
            ]
        )
        flag("set", "V", "MTE2", slot)
        flag("set", "MTE3", "V", slot)
    lines += [
        "    scf.for %iv = %c0 to %trip step %c1 {",
        "      %selector = arith.remui %iv, %depth : index",
        "      %row = arith.muli %iv, %c16 : index",
        "      %input = pto.partition_view %source, offsets = [%row, %c0], "
        "sizes = [%c16, %c16] : !pto.tensor_view<?x?xf16> -> "
        + view,
    ]
    for slot in range(depth):
        lines += [
            f"      %active{slot} = arith.cmpi eq, %selector, %slot{slot} : index",
            f"      scf.if %active{slot} {{",
        ]
        flag("wait", "V", "MTE2", slot, "        ")
        lines += [f"        pto.tload ins(%input : {view}) outs(%in{slot} : {tile})"]
        flag("set", "MTE2", "V", slot, "        ")
        flag("wait", "MTE2", "V", slot, "        ")
        for use in range(uses):
            lines += [
                f"        %u{slot}_{use} = arith.constant {use} : index",
                f"        %base{slot}_{use} = arith.muli %iv, %uses : index",
                f"        %oi{slot}_{use} = arith.addi %base{slot}_{use}, %u{slot}_{use} : index",
                f"        %or{slot}_{use} = arith.muli %oi{slot}_{use}, %c16 : index",
                f"        %output{slot}_{use} = pto.partition_view %target, offsets = [%or{slot}_{use}, %c0], "
                f"sizes = [%c16, %c16] : !pto.tensor_view<?x?xf16> -> {view}",
            ]
            flag("wait", "MTE3", "V", slot, "        ")
            lines += [f"        pto.tabs ins(%in{slot} : {tile}) outs(%out{slot} : {tile})"]
            if use == uses - 1:
                flag("set", "V", "MTE2", slot, "        ")
            flag("set", "V", "MTE3", slot, "        ")
            flag("wait", "V", "MTE3", slot, "        ")
            lines += [f"        pto.tstore ins(%out{slot} : {tile}) outs(%output{slot}_{use} : {view})"]
            flag("set", "MTE3", "V", slot, "        ")
        lines += ["      }"]
    lines += ["    }"]
    for slot in range(depth):
        flag("wait", "V", "MTE2", slot)
        flag("wait", "MTE3", "V", slot)
    lines += ["    pto.barrier <PIPE_ALL>", "    return", "  }", "}"]
    license_lines = Path(__file__).read_text().split('"""', 1)[0].splitlines()[1:]
    prefix = "\n".join("//" + line[1:] for line in license_lines) + "\n"
    manual = prefix + "\n".join(lines) + "\n"
    automatic = (
        "\n".join(
            line
            for line in manual.splitlines()
            if not any(op in line for op in ("pto.set_flag", "pto.wait_flag", "pto.barrier"))
        )
        + "\n"
    )
    return automatic, manual


if __name__ == "__main__":
    root = Path(__file__).parent / "inputs"
    for name, depth, uses in (("one_buffer", 1, 1), ("two_buffer", 2, 1), ("three_buffer", 3, 1), ("four_use", 2, 4)):
        for arm, source in zip(("auto", "manual"), generate(depth, uses)):
            (root / f"{name}.{arm}.pto").write_text(source)
