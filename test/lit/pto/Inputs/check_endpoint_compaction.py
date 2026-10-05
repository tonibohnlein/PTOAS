# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare compacted IR execution against literal cyclic endpoint emission."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from check_logical_insertion import invoke, recognized
from check_physical_allocation import check_physical, opt



def synthetic_plan(positions, records, barrier=False):
    """Exercise emission fallback only; this is not a storage-analysis certificate."""
    lines = ["module {", "func.func @emission() attributes {pto.cyclic_allocation = {",
             "version = 1 : i64, plan = 0 : i64, directions = [{source = 4 : i64,",
             "target = 3 : i64, budget = 2 : i64, records = array<i64: " +
             ", ".join(map(str, records)) + ">}]}} {",
             "%zero = arith.constant 0 : index", "%one = arith.constant 1 : index",
             "%end = arith.constant 5 : index", "scf.for %k = %zero to %end step %one {"]
    for record, position in enumerate(positions):
        lines += [f"%c{record} = arith.constant {position} : index",
                  f"%g{record} = arith.cmpi eq, %k, %c{record} : index"]
    for kind, cut in [("set", 0), ("wait", 1)]:
        for record in range(len(positions)):
            lines += [f"scf.if %g{record} {{",
                      f"pto.logical_{kind} [<PIPE_MTE2>, <PIPE_MTE1>] "
                      f"plan 0 record {record} ordinal %zero",
                      f"}} {{pto.endpoint_cut = {cut} : i64}}"]
            if barrier and kind == "set" and record == 0:
                lines.append("pto.barrier <PIPE_MTE2>")
    return "\n".join(lines + ["}", "return", "}", "}"])


def fallback_checks(optimizer, path):
    """Declining compaction must leave all literal commands at the affected cut."""
    passes = ["--pto-frontier-allocate=eligible-ids=1,3"]
    path.write_text(synthetic_plan([0, 1, 2], [0, 2, 1]))
    failed_fit = opt(optimizer, path, passes).stdout
    assert failed_fit.count("pto.set_flag") == 3
    assert failed_fit.count("pto.wait_flag") == 3
    path.write_text(synthetic_plan([0, 1, 2], [0, 1, 2], barrier=True))
    barrier = opt(optimizer, path, passes).stdout
    assert barrier.count("pto.set_flag") == 3, "compaction crossed a local barrier"
    assert barrier.count("pto.wait_flag") == 1, "unobstructed cut should still compact"
    assert barrier.count("pto.barrier") == 1
    # Affine phases over a point set with a hole must retain a union of boxes.
    path.write_text(synthetic_plan([0, 1, 3, 4], [0, 1, 3, 2]))
    holes = opt(optimizer, path, passes).stdout
    assert holes.count("pto.set_flag") == 1 and holes.count("pto.wait_flag") == 1
    assert "arith.ori" in holes and "arith.cmpi ule" in holes and "arith.cmpi uge" in holes


def main():
    tool, optimizer = shutil.which(sys.argv[1]), shutil.which(sys.argv[2])
    source = Path(sys.argv[3]).read_text()
    checked = 0
    with tempfile.TemporaryDirectory(prefix="endpoint-compaction-") as scratch:
        path = Path(scratch) / "case.pto"
        for inner_end in [1, 2, 5, 8, 9, 17]:
            for outer_end in [1, 2, 8]:
                text = source.replace("constant 9 : index", f"constant {inner_end} : index")
                text = text.replace("array<i64: 8>", f"array<i64: {outer_end}>")
                path.write_text(text)
                compact = json.loads(invoke(tool, "--physical-trace", path))
                check_physical(recognized(tool, path), compact, {1, 3})
                path.write_text(text.replace("test.eligible_ids", "test.uncompact_endpoints, test.eligible_ids"))
                literal = json.loads(invoke(tool, "--physical-trace", path))
                for field in ["error", "events", "payloads", "outer_trips"]:
                    assert compact["physical"][field] == literal["physical"][field], field
                checked += 1
        # The same static payload cut is visited at 1, 3 and 7, leaving a hole at 5.
        # Preserve selected-arm execution as well as the sparse coordinates.
        hole = source.replace("%inner_end = arith.constant 9 : index",
                              "%inner_end = arith.constant 9 : index\n    %five = arith.constant 5 : index")
        first = "        pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile0 : !tile)"
        last = "        pto.textract ins(%tile1, %zero, %zero : !tile, index, index) outs(%left1 : !left)"
        hole = hole.replace(first, "        %keep = arith.cmpi ne, %k, %five : index\n        scf.if %keep {\n" + first)
        hole = hole.replace(last, last + "\n        }")
        path.write_text(hole)
        compact = json.loads(invoke(tool, "--physical-trace", path))
        check_physical(recognized(tool, path), compact, {1, 3})
        path.write_text(hole.replace("test.eligible_ids", "test.uncompact_endpoints, test.eligible_ids"))
        literal = json.loads(invoke(tool, "--physical-trace", path))
        assert compact["physical"]["events"] == literal["physical"]["events"]
        checked += 1
        # Typed, nonzero coordinates must be normalized without narrowing.
        typed = source.replace("%inner_end = arith.constant 9 : index", """%inner_end = arith.constant 10 : i32
    %inner_low = arith.constant 1 : i32
    %inner_step = arith.constant 2 : i32""")
        typed = typed.replace("%k = %one to %inner_end step %two {",
                              "%k = %inner_low to %inner_end step %inner_step : i32 {")
        path.write_text(typed)
        compact = json.loads(invoke(tool, "--physical-trace", path))
        check_physical(recognized(tool, path), compact, {1, 3})
        path.write_text(typed.replace("test.eligible_ids", "test.uncompact_endpoints, test.eligible_ids"))
        literal = json.loads(invoke(tool, "--physical-trace", path))
        assert compact["physical"]["events"] == literal["physical"]["events"]
        checked += 1
        fallback_checks(optimizer, path)
        path.write_text(source)
        logical = opt(optimizer, path, ["--pto-frontier-analysis"]).stdout
        physical = opt(optimizer, path, ["--pto-frontier-analysis", "--pto-frontier-allocate=eligible-ids=1,3"]).stdout
        assert physical.count("pto.set_flag") < logical.count("pto.logical_set"), "no static reduction"
        assert "pto.endpoint_cut" not in physical
        assert physical.count("<PIPE_ALL>") == 1
        for name in ["scf.for", "pto.tload", "pto.textract"]:
            assert physical.count(name) == source.count(name)
    print(f"endpoint compaction: {checked} identical command traces, compact sites, terminal completion")


if __name__ == "__main__":
    main()
