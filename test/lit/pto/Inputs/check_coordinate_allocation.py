# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check code growth and exact IDs independently of record numbering."""
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

from check_logical_insertion import invoke
from check_physical_allocation import opt


def fixture(size, threshold=False, sparse=False):
    """Saved certificates isolate lowering; no recognizer chooses these labels."""
    entries, members, records, even, odd, expected = [], [], [], [], [], []
    for row in range(size):
        for column in range(4):
            if sparse and column == 2:
                continue
            record = (4 * row + column) ** 2 + 13
            lane = int(row >= size // 2) if threshold else (row + column) % 2
            records.append(record)
            (even if column % 2 == 0 else odd).append(record)
            expected.append(["set", [1, 5][lane]])
            expected.append(["wait", [1, 5][lane]])
            entries.append(f"{{lane_begin = {lane} : i64, lane_count = 1 : i64, offset = 0 : i64, "
                           f"record = {record} : i64, source = 4 : i64, target = 3 : i64}}")
            point = f"[array<i64: 0, {1 + 2 * row}>, array<i64: 1, {column}>]"
            members.append(f"{{record = {record} : i64, source = {point}, target = {point}}}")
    pieces = []
    for piece, kind, cut, selected in [(0, 0, 0, records), (1, 2, 1, even), (2, 2, 1, odd)]:
        pieces.append(f"{{id = {piece} : i64, family = 13 : i64, kind = {kind} : i64, cut = {cut} : i64, "
                      "source_pipe = 4 : i64, target_pipe = 3 : i64, displacement = 0 : i64, "
                      "records = array<i64: " + ", ".join(map(str, selected)) + ">}")
    text = """module {
  func.func @coordinate_ids(%end: index) attributes {
    test.trace_existing, test.trace_arguments = array<i64: END>,
    pto.cyclic_allocation = {version = 3 : i64, plan = 0 : i64,
      strategy = "directed-cycle-cover", order_exact = true, budget = 2 : i64, entries = [ENTRIES]},
    pto.endpoint_families = {version = 3 : i64, plan = 0 : i64, families = [
      {id = 13 : i64, local = false, source_pipe = 4 : i64, target_pipe = 3 : i64,
       displacement = 0 : i64, source_cut = 0 : i64, target_cut = 1 : i64,
       source_order = 0 : i64, target_order = 0 : i64, members = [MEMBERS]}], pieces = [PIECES]}
  } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %thirteen = arith.constant 13 : index
    %true = arith.constant true
    scf.for %i = %one to %end step %two {
      %offset = arith.subi %i, %one : index
      %row = arith.divui %offset, %two : index
      %base = arith.muli %row, %four : index
      scf.for %j = %zero to %four step %one {
        %flat = arith.addi %base, %j : index
        %square = arith.muli %flat, %flat : index
        %label = arith.addi %square, %thirteen : index
        %parity = arith.remui %j, %two : index
        %even = arith.cmpi eq, %parity, %zero : index
        %odd = arith.cmpi ne, %parity, %zero : index
        %keep = arith.cmpi ne, %j, %two : index
        scf.if KEEP {
          scf.if %true {
            pto.logical_set [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 13 ordinal %zero members(%label)
              {pto.endpoint_piece = 0 : i64}
          } {pto.endpoint_cut = 0 : i64}
          scf.if %even {
            pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 13 ordinal %zero members(%label)
              {pto.endpoint_piece = 1 : i64}
          } {pto.endpoint_cut = 1 : i64}
          scf.if %odd {
            pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 13 ordinal %zero members(%label)
              {pto.endpoint_piece = 2 : i64}
          } {pto.endpoint_cut = 1 : i64}
        }
      } {pto.family_loop = 1 : i64}
    } {pto.family_loop = 0 : i64}
    return
  }
}
"""
    for token, replacement in [("END", str(1 + 2 * size)), ("ENTRIES", ", ".join(entries)),
                               ("MEMBERS", ", ".join(members)), ("PIECES", ", ".join(pieces)),
                               ("KEEP", "%keep" if sparse else "%true")]:
        text = text.replace(token, replacement)
    return text, expected


def commands(tool, path):
    trace = json.loads(invoke(tool, "--structured-trace", path))["trace"]
    assert not trace["error"], trace["error"]
    return [[event["kind"], event["physical_id"]] for event in trace["events"]
            if event["kind"] in ("set", "wait")]


def main():
    optimizer, interpreter = map(shutil.which, sys.argv[1:3])
    scratch = Path(sys.argv[3])
    scratch.mkdir(parents=True, exist_ok=True)
    observed = []
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        path = Path(directory) / "case.pto"
        for threshold, sparse in [(False, False), (False, True), (True, False)]:
            sizes = []
            for size in (8, 32, 128):
                source, expected = fixture(size, threshold, sparse)
                path.write_text(source)
                physical = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,5"]).stdout
                sizes.append(len(re.findall(r" = arith\.", physical)))
                assert physical.count("pto.set_flag") == 1
                assert physical.count("pto.wait_flag") == 2
                path.write_text(physical)
                assert commands(interpreter, path) == expected
                # Each actual command is checked, including noncontiguous IDs,
                # independent consumer pieces, nonunit coordinates and holes.
                path.write_text(physical.replace(f"array<i64: {1 + 2 * size}>", "array<i64: 1>"))
                assert commands(interpreter, path) == []
            assert max(sizes) < 150, sizes
            assert max(sizes) - min(sizes) <= 8, sizes
            observed.append(sizes)
    print("coordinate allocation: exact traces and bounded generated arithmetic", observed)


if __name__ == "__main__":
    main()
