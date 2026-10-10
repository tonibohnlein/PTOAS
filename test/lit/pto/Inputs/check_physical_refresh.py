# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent absolute-byte closure for overlapping finite address schedules."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_compact_endpoints import check_commands
from check_periodic_demands import closure, native
from check_varying_rotating import attempts, report


def trace_check(document, lower, step, n, threshold, common_family, distance=None):
    assert document["accepted"] and document["bounded_demands_retained"], document
    assert not document["trace"]["error"], document
    payloads = [event for event in document["trace"]["events"] if event["kind"] == "payload"]
    expected = []
    accesses = []
    for induction in range(lower, n, step):
        # Four independent 512-byte atoms; allocation names never distinguish
        # overlapping bytes. The fixed view intersects both rotating banks.
        bank = {("left", atom) for atom in (2 * (induction % 2), 2 * (induction % 2) + 1)}
        fixed = {("left", atom) for atom in ((0, 1) if common_family else (1, 2))}
        if distance is None or induction + distance < n:
            expected.append(("rotating", [induction]))
            accesses.append(({("mat", 0)}, bank))
        expected.extend((("then" if induction < threshold else "else", [induction]),
                         ("compute", [induction])))
        accesses.extend((({("mat", 0)}, fixed), (bank | {("right", 0)}, {("acc", 0)})))
    assert [(event["label"], event["coordinates"]) for event in payloads] == expected
    edges = native([event["pipe"] for event in payloads])
    for a, (reads, writes) in enumerate(accesses):
        for b in range(a + 1, len(accesses)):
            later_reads, later_writes = accesses[b]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    check_commands(document, bounded=True, expected=closure(2 * len(payloads), edges)[0])


def check_suffix(tool, path, bounded_source, writer):
    for distance, step in ((0, 1), (1, 1), (4, 3)):
        case = bounded_source.replace("%one = arith.constant 1 : index",
                                      f"%one = arith.constant {step} : index")
        prefetch = (f"      %distance = arith.constant {distance} : index\n"
                    "      %next = arith.addi %i, %distance : index\n"
                    "      %interior = arith.cmpi slt, %next, %n : index\n"
                    "      scf.if %interior {\n" + writer + "\n      }")
        case = case.replace(writer, prefetch)
        for form in ("direct", "swapped", "inverse"):
            variant = case
            if form == "swapped":
                variant = variant.replace("slt, %next, %n", "sgt, %n, %next")
            if form == "inverse":
                variant = variant.replace("slt, %next, %n", "sge, %next, %n")
                variant = variant.replace("scf.if %interior {\n" + writer,
                                          "scf.if %interior {\n      } else {\n" + writer)
            path.write_text(variant)
            bounded = attempts(report(tool, path), "bounded-lifetime")
            span = 2 + (distance + step - 1) // step
            assert any(item["state"] == "applicable" and item["refresh_span"] == span
                       for item in bounded), (distance, step, form, bounded)
            for n in (0, 1, 2, 3, 4, 5, 7, 10, 13):
                path.write_text(variant.replace("array<i64: 3, 2>", f"array<i64: {n}, 2>"))
                result = subprocess.run([tool, "--structured-trace", str(path)], check=True,
                                        capture_output=True, text=True, timeout=60)
                trace_check(json.loads(result.stdout), 0, step, n, 2, False, distance)
    floating = bounded_source.replace("%m: index", "%m: index, %x: f32, %y: f32").replace(
        "arith.cmpi slt, %i, %m : index", "arith.cmpf ogt, %x, %y : f32")
    floating = floating.replace(writer,
        "      %next = arith.addi %i, %one : index\n"
        "      %interior = arith.cmpi slt, %next, %n : index\n"
        "      scf.if %interior {\n" + writer + "\n      }")
    path.write_text(floating)
    bounded = attempts(report(tool, path), "bounded-lifetime")
    assert any(item["state"] == "applicable" and item["refresh_span"] == 3
               for item in bounded), bounded


def check_refusals(tool, path, source, bounded_source, writer):
    # A similar spelling is not a proof: a different upper SSA, a
    # negative displacement, or a possibly wrapping addition must fail.
    for name, base, delta, bound in (("unrelated", bounded_source, 1, "%m"),
                                     ("negative", bounded_source, -1, "%n"),
                                     ("wrapping", source, 2, "%n")):
        guarded = (f"      %distance = arith.constant {delta} : index\n"
                   "      %next = arith.addi %i, %distance : index\n"
                   f"      %interior = arith.cmpi slt, %next, {bound} : index\n"
                   "      scf.if %interior {\n" + writer + "\n      }")
        path.write_text(base.replace(writer, guarded))
        bounded = attempts(report(tool, path), "bounded-lifetime")
        assert bounded and not any(item["state"] == "applicable" for item in bounded), (name, bounded)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    variants = [("overlap", source, 0, 1, False)]
    # Same family, unequal selector strides. This needs physical projection
    # even though its nominal family bounding intervals are disjoint.
    common = source.replace("    %overlap = pto.alloc_tile addr = %middle : !left",
                            "    %overlap = pto.multi_tile_get %ring[%zero] : !ring -> !left")
    variants.append(("unequal_strides", common, 0, 1, True))
    shifted = source.replace("    %two = arith.constant 2 : index",
                             "    %two = arith.constant 2 : index\n"
                             "    %origin = arith.constant 1 : index\n"
                             "    %stride = arith.constant 3 : index")
    shifted = shifted.replace("%i = %zero to %n step %one", "%i = %origin to %n step %stride")
    variants.append(("nonunit", shifted, 1, 3, False))
    with tempfile.TemporaryDirectory(prefix="physical-refresh-") as scratch:
        path = Path(scratch) / "case.pto"
        for name, case, lower, step, common_family in variants:
            path.write_text(case)
            bounded = attempts(report(tool, path), "bounded-lifetime")
            accepted = [item for item in bounded if item["state"] == "applicable"]
            assert len(accepted) == 1 and accepted[0]["refresh_span"] == 2, (name, bounded)
            for visits, threshold in ((0, 0), (1, 1), (2, 0), (3, 2), (4, 1), (5, 4)):
                n = lower + visits * step
                path.write_text(case.replace("array<i64: 3, 2>", f"array<i64: {n}, {threshold}>"))
                result = subprocess.run([tool, "--structured-trace", str(path)], check=True,
                                        capture_output=True, text=True, timeout=60)
                trace_check(json.loads(result.stdout), lower, step, n, threshold, common_family)
        # A next-visit prefetch may stop before the loop exits. Derive its
        # suffix from literal IR, then compare all actual events and hazards.
        writer = ('      pto.textract ins(%mat, %zero, %zero : !mat, index, index) '
                  'outs(%left : !left) {test.label = "rotating"}')
        bounded_source = source.replace("%n: index", "%count: i32").replace(
            "    %base =", "    %n = arith.index_cast %count : i32 to index\n    %base =")
        check_suffix(tool, path, bounded_source, writer)
        check_refusals(tool, path, source, bounded_source, writer)
        missing = source.replace(writer, "      scf.if %take {\n" + writer + "\n      }")
        path.write_text(missing)
        bounded = attempts(report(tool, path), "bounded-lifetime")
        assert bounded and not any(item["state"] == "applicable" for item in bounded), bounded
        assert any("writable physical atom" in item.get("certificate_error", "") for item in bounded), bounded
    print("physical refresh: common-byte coverage, unequal strides, original coordinates and full closure passed")


if __name__ == "__main__":
    main()
