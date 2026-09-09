#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Qualify native slot maps against original planned/explicit physical layout.

The driver uses the actual translator and independent Cartesian overlap oracle;
these tests additionally pin original slot order and lowering-stride admission.
They do not assume that a dynamic selector is in range or implicitly modulo N.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

TILE = "!pto.tile_buf<vec, 16x16xf16>"
LEFT = "!pto.tile_buf<loc=left, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>"


def program(*, tile=TILE, addresses=(0, 512), explicit=False, literal=False,
            views="", second=None, base=0, runtime_base=False):
    multi = f"!pto.multi_tile_buf<{tile}, count={len(addresses)}>"
    allocation = ("addr = %runtime_base" if runtime_base else "addr = %base" if explicit else
                  "{pto.multi_buffer_addrs = array<i64: " + ", ".join(map(str, addresses)) + ">}")
    extra = ""
    if second is not None:
        extra = f'''%other = pto.alloc_multi_tile {{pto.multi_buffer_addrs = array<i64: {", ".join(map(str, second))}>}} : {multi}
    %picked_other = pto.multi_tile_get %other[%selector] : {multi} -> {tile}'''
    return f'''module attributes {{pto.target_arch = "a3"}} {{
  func.func @f(%selector: index{", %runtime_base: i64" if runtime_base else ""}) attributes {{pto.kernel_kind = #pto.kernel_kind<{"cube" if tile == LEFT else "vector"}>}} {{
    %base = arith.constant {base} : i64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %slots = pto.alloc_multi_tile {allocation} : {multi}
    %picked = pto.multi_tile_get %slots[{"%one" if literal else "%selector"}] : {multi} -> {tile}
    {views}
    {extra}
    return
  }}
}}'''


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    lowerings = []

    def evaluate_lowered(text, selector, runtime_base=0):
        """Evaluate only the actual scalar address selection emitted by this pass.

        The expected physical table is supplied separately from the native
        mapping/translator; unknown operations fail this observation.
        """
        arguments = re.search(r"func.func @f\(([^)]*)\)", text).group(1)
        names = re.findall(r"(%[\w.$]+)\s*:", arguments)
        values = dict(zip(names, (selector, runtime_base)))
        addresses = []
        for line in text.splitlines():
            assignment = re.search(r"(%[\w.$]+) = (.+)", line)
            if not assignment:
                continue
            result, operation = assignment.groups()
            if operation.startswith("arith.constant "):
                values[result] = int(operation.split()[1])
            elif operation.startswith("arith.addi "):
                a, b = re.findall(r"%[\w.$]+", operation)
                values[result] = values[a] + values[b]
            elif operation.startswith("arith.cmpi eq, "):
                a, b = re.findall(r"%[\w.$]+", operation)
                values[result] = values[a] == values[b]
            elif operation.startswith("arith.select "):
                p, a, b = re.findall(r"%[\w.$]+", operation)
                values[result] = values[a] if values[p] else values[b]
            elif operation.startswith(("arith.index_cast ", "arith.index_castui ",
                                       "arith.extsi ", "arith.extui ", "arith.trunci ")):
                source = re.search(r"%[\w.$]+", operation).group(0)
                source_type, target_type = re.search(r": (index|i\d+) to (index|i\d+)", operation).groups()
                source_bits = 64 if source_type == "index" else int(source_type[1:])
                target_bits = 64 if target_type == "index" else int(target_type[1:])
                raw = values[source] % (1 << source_bits)
                unsigned = operation.startswith(("arith.extui ", "arith.index_castui "))
                if not unsigned and raw >= 1 << (source_bits - 1):
                    raw -= 1 << source_bits
                raw %= 1 << target_bits
                values[result] = raw - (1 << target_bits) if raw >= 1 << (target_bits - 1) else raw
            elif operation.startswith("pto.alloc_tile addr = "):
                address = re.search(r"addr = (%[\w.$]+)", operation).group(1)
                addresses.append(values[address])
            else:
                raise AssertionError(("unmodeled emitted address operation", operation))
        return addresses

    def lower(name, source, expected, *, literal=False, runtime_base=0, error=None):
        path = args.output / (name + ".pto")
        path.write_text(source)
        p = subprocess.run([str(args.driver.resolve()), "--lower", str(path)],
                           capture_output=True, text=True, timeout=30)
        (args.output / (name + ".stdout")).write_text(p.stdout)
        (args.output / (name + ".stderr")).write_text(p.stderr)
        if error is not None:
            assert p.returncode != 0 and error in p.stderr, (name, p.returncode, p.stdout, p.stderr)
        else:
            assert p.returncode == 0, (name, p.stderr)
            assert "pto.multi_tile_get" not in p.stdout and "pto.alloc_multi_tile" not in p.stdout
            for selector in (-1, 0, 1, len(expected) - 1, len(expected), len(expected) + 1):
                # Literal invalid slots are rejected by IR verification.
                # Dynamic invalid slots preserve the lowerer's slot-zero default.
                slot = 1 if literal else selector if 0 <= selector < len(expected) else 0
                assert evaluate_lowered(p.stdout, selector, runtime_base) == [expected[slot]], (name, selector, p.stdout)
        lowerings.append({"name": name, "rejected": error is not None})
        (args.output / "lowerings.json").write_text(json.dumps(lowerings, indent=2) + "\n")

    def run(name, source, expected, *, complete=True, reason="", layout=(512, 32, 512)):
        path = args.output / (name + ".pto")
        path.write_text(source)
        p = subprocess.run([str(args.driver.resolve()), str(path)], capture_output=True, text=True, timeout=30)
        (args.output / (name + ".stdout")).write_text(p.stdout)
        (args.output / (name + ".stderr")).write_text(p.stderr)
        assert p.returncode == 0, (name, p.stderr, p.stdout)
        result = json.loads(p.stdout)
        assert result["passed"], (name, result)
        function = result["functions"][0]
        assert function["physical_complete"] is complete, (name, function)
        assert reason in function["reason"], (name, function["reason"])
        for native in function["layouts"]:
            assert native["qualified"] and tuple(native[k] for k in ("bytes", "alignment", "stride")) == layout, (name, native)
        candidates = [r for r in function["records"] if r["operation"] in
                      {"pto.multi_tile_get", "pto.treshape", "pto.bitcast", "pto.subview"}]
        assert len(candidates) == len(expected), (name, candidates)
        for native, wanted in zip(candidates, expected):
            assert native["qualified"] is (wanted is not None), (name, native, wanted)
            if wanted is not None:
                assert native["selector"] and list(map(int, native["bases"])) == wanted, (name, native)
                assert native["bytes"] == layout[0]
        rows.append({"name": name, "checks": result["checks"], "function": function})
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        return function

    dynamic = run("planned_dynamic", program(), [[0, 512]])
    literal = run("planned_literal_keeps_slot_ids", program(literal=True), [[0, 512]])
    dynamic_get = next(r for r in dynamic["records"] if r["operation"] == "pto.multi_tile_get")
    literal_get = next(r for r in literal["records"] if r["operation"] == "pto.multi_tile_get")
    # Both validate and sort the complete two-slot root. Only the original
    # access-range validation changes when literal selection narrows its union.
    assert dynamic_get["qualification_work"] == literal_get["qualification_work"] + 2
    # Sixteen is the native multi_tile_buf verifier's maximum slot count.
    large_addresses = [512 * i for i in range(16)]
    large = run("literal_full_root_accounting", program(addresses=large_addresses, literal=True), [large_addresses])
    large_get = next(r for r in large["records"] if r["operation"] == "pto.multi_tile_get")
    assert large_get["qualification_work"] > 2 * literal_get["qualification_work"], large_get
    run("planned_reversed_order", program(addresses=(512, 0)), [[512, 0]])
    run("explicit_aligned_base", program(explicit=True), [[0, 512]])
    run("unaligned_planned_interval", program(addresses=(16, 528)), [], complete=False,
        reason="unaligned planned multi-tile physical interval")
    run("signed_address_end_overflow", program(addresses=(0, 2**63-32)), [], complete=False,
        reason="planned multi-tile interval exceeds")
    run("aligned_left_planned", program(tile=LEFT), [[0, 512]], layout=(32, 512, 512))
    run("aligned_left_explicit_dynamic", program(tile=LEFT, explicit=True), [[0, 512]], layout=(32, 512, 512))
    run("aligned_left_explicit_literal", program(tile=LEFT, explicit=True, literal=True), [[0, 512]],
        layout=(32, 512, 512))
    run("aligned_left_nonzero_base", program(tile=LEFT, explicit=True, base=512), [[512, 1024]],
        layout=(32, 512, 512))
    truncated_base = program(tile=LEFT, explicit=True).replace(
        "%base = arith.constant 0 : i64",
        "%wide = arith.constant 4294967808 : i64\n"
        "    %narrow = arith.trunci %wide : i64 to i32\n"
        "    %base = arith.extui %narrow : i32 to i64")
    run("aligned_left_truncated_base", truncated_base, [[512, 1024]], layout=(32, 512, 512))
    run("transparent_reshape", program(views=f"%view = pto.treshape %picked : {TILE} -> {TILE}"),
        [[0, 512], [0, 512]], complete=False, reason="physical phase has no qualified adapter")
    run("transparent_bitcast", program(views=f"%view = pto.bitcast %picked : {TILE} -> !pto.tile_buf<vec, 16x16xi16>"),
        [[0, 512], [0, 512]])
    run("conservative_subview", program(views=f"%view = pto.subview %picked[%zero, %zero] sizes [16, 16] : {TILE} -> {TILE}"),
        [[0, 512], None])
    overlap = run("partial_cross_root_overlap", program(second=(256, 768)), [[0, 512], [256, 768]])
    ids = [i for i, record in enumerate(overlap["records"]) if record["operation"] == "pto.multi_tile_get"]
    relation = next(row for row in overlap["overlaps"] if row["left"] == ids[0] and row["right"] == ids[1])
    assert relation["pairs"] == [[0, 0], [1, 0], [1, 1]], relation
    reversed_ = run("unequal_indices_same_bytes", program(second=(512, 0)), [[0, 512], [512, 0]])
    ids = [i for i, record in enumerate(reversed_["records"]) if record["operation"] == "pto.multi_tile_get"]
    relation = next(row for row in reversed_["overlaps"] if row["left"] == ids[0] and row["right"] == ids[1])
    assert relation["pairs"] == [[0, 1], [1, 0]], relation
    lower("lower_aligned_left_dynamic", program(tile=LEFT, explicit=True), [0, 512])
    lower("lower_aligned_left_literal", program(tile=LEFT, explicit=True, literal=True), [0, 512], literal=True)
    lower("lower_aligned_left_nonzero", program(tile=LEFT, explicit=True, base=512), [512, 1024])
    lower("lower_runtime_base", program(tile=LEFT, runtime_base=True), [1024, 1536], runtime_base=1024)
    lower("lower_planned_reversed", program(addresses=(512, 0)), [512, 0])
    lower("lower_truncated_base", truncated_base, [512, 1024])
    lower("lower_known_interval_overflow", program(explicit=True, base=2**63-512), [],
          error="slot address or footprint end overflows")
    lower("lower_known_planned_overflow", program(addresses=(0, 2**63-32)), [],
          error="planned slot address or footprint end overflows")
    cast_overflow = program(explicit=True).replace(
        "%base = arith.constant 0 : i64",
        f"%wide = arith.constant {2**63-512} : i64\n"
        "    %index = arith.index_cast %wide : i64 to index\n"
        "    %base = arith.index_cast %index : index to i64")
    lower("lower_cast_interval_overflow", cast_overflow, [], error="slot address or footprint end overflows")
    cast_negative = program(explicit=True).replace(
        "%base = arith.constant 0 : i64",
        "%negative = arith.constant -512 : i32\n"
        "    %base = arith.extsi %negative : i32 to i64")
    lower("lower_cast_negative", cast_negative, [], error="slot base is outside nonnegative i64")
    lower("lower_unaligned_planned", program(addresses=(16, 528)), [],
          error="planned slot address is not aligned")
    cast_unaligned = program(explicit=True).replace(
        "%base = arith.constant 0 : i64",
        "%unaligned = arith.constant 16 : i32\n"
        "    %base = arith.extui %unaligned : i32 to i64")
    lower("lower_cast_unaligned", cast_unaligned, [], error="slot base is not aligned")
    print(f"Native slot mapping: {len(rows)} cases; {sum(row['checks'] for row in rows)} sweep/corruption checks passed")
    print(f"Native slot address lowering: {len(lowerings)} cases passed")


if __name__ == "__main__":
    main()
