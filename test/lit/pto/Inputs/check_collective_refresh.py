# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Prove collective refresh on guarded variants of the existing matmul fixture."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_compact_endpoints import check_commands
from check_periodic_demands import closure, native
from check_varying_rotating import attempts, report


WRITER = "      pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%left : !left)"
COMPUTE = "      pto.tmatmul ins(%left, %right : !left, !right) outs(%acc : !acc)"


def labelled_writer(label, target="%left"):
    return WRITER.replace("outs(%left", f"outs({target}") + f' {{test.label = "{label}"}}'


def variants(source):
    window = source.replace("    scf.for %visit = %zero to %visits step %one {\n"
                            "      %length = arith.addi %visit, %two : index\n"
                            "      scf.for %i = %zero to %length step %one {",
                            "    scf.for %i = %zero to %n step %one {")
    window = window.replace("      }\n    }\n", "    }\n")
    window = window.replace("    pto.textract ins(%mat, %zero, %zero : !mat, index, index) "
                            "outs(%first : !left)\n", "")
    window = window.replace("attributes {test.trace_arguments",
                            "attributes {test.bounded_lifetime_insertion, test.trace_arguments")
    window = window.replace(COMPUTE, COMPUTE + ' {test.label = "compute"}')
    condition = "      %take = arith.cmpi ult, %i, %m : index\n"
    then = labelled_writer("writer_then")
    other = labelled_writer("writer_other")
    final = labelled_writer("writer_else")
    pair = condition + "      scf.if %take {\n" + then + "\n      } else {\n" + final + "\n      }"
    nested = condition + "      %split = arith.cmpi eq, %slot, %zero : index\n" + (
        "      scf.if %take {\n        scf.if %split {\n" + then +
        "\n        } else {\n" + other + "\n        }\n      } else {\n" + final + "\n      }")
    separate = condition + "      scf.if %take {\n" + then + "\n      }\n" + (
        "      scf.if %take {\n      } else {\n" + final + "\n      }")
    missing = condition + "      scf.if %take {\n" + then + "\n      }"
    shifted = ("      %next = arith.addi %slot, %one : index\n"
               "      %nextslot = arith.remui %next, %two : index\n"
               "      %otherleft = pto.multi_tile_get %ring[%nextslot] : !ring -> !left\n"
               "      %take = arith.cmpi eq, %slot, %zero : index\n"
               "      scf.if %take {\n" + then + "\n      } else {\n" +
               labelled_writer("writer_else", "%otherleft") + "\n      }")
    return [(name, window.replace(WRITER, replacement), accepted) for name, replacement, accepted in (
        ("complementary", pair, True), ("nested", nested, True), ("separate_branches", separate, True),
        ("missing_arm", missing, False), ("same_orbit_different_offsets", shifted, False))]


def check_trace(document, n, threshold, nested):
    assert document["accepted"] and not document["trace"]["error"], document
    payloads = [item for item in document["trace"]["events"] if item["kind"] == "payload"]
    expected_payloads = []
    accesses = []
    for visit in range(n):
        writer = "writer_else" if visit >= threshold else (
            "writer_other" if nested and visit % 2 else "writer_then")
        expected_payloads.extend(((writer, [visit]), ("compute", [visit])))
        accesses.extend((({("mat", 0)}, {("left", visit % 2)}),
                         ({("left", visit % 2), ("right", 0)}, {("acc", 0)})))
    assert [(item["label"], item["coordinates"]) for item in payloads] == expected_payloads
    edges = native([item["pipe"] for item in payloads])
    for a, (reads, writes) in enumerate(accesses):
        for b in range(a + 1, len(accesses)):
            next_reads, next_writes = accesses[b]
            if writes & (next_reads | next_writes) or reads & next_writes:
                edges.add((2 * a + 1, 2 * b))
    expected = closure(2 * len(payloads), edges)[0]
    check_commands(document, bounded=True, expected=expected)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="collective-refresh-") as scratch:
        path = Path(scratch) / "case.pto"
        for name, case, accepted in variants(source):
            path.write_text(case)
            bounded = attempts(report(tool, path), "bounded-lifetime")
            good = [item for item in bounded if item["state"] == "applicable"]
            if not accepted:
                assert not good and any(item.get("certificate_error") for item in bounded), (name, bounded)
                continue
            assert len(good) == 1 and good[0]["refresh_span"] == 2, (name, bounded)
            assert not good[0]["entry_guards_available"], (name, good)
            # The same loop can be analyzed as a proper region. Its owned
            # demand circuit survives even when a parent needs further exports.
            prologue = "    pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%first : !left)\n"
            regional = case.replace("    scf.for %i = %zero to %n step %one {",
                                    prologue + "    scf.for %i = %zero to %n step %one {")
            path.write_text(regional)
            sequence = subprocess.run([tool, "--sequence-analysis", str(path)], capture_output=True,
                                      text=True, check=True, timeout=90)
            exported = json.loads(sequence.stdout)
            assert exported["bounded_cached_regions"] == 1 and exported["bounded_cache_stable"], exported
            assert exported["bounded_cached_demands"] > 0 and exported["unchanged"], exported
            for n, threshold in ((0, 0), (1, 0), (1, 1), (3, 1), (5, 4), (6, 2)):
                path.write_text(case.replace("array<i64: 2, 3>", f"array<i64: {n}, {threshold}>"))
                result = subprocess.run([tool, "--structured-trace", str(path)], capture_output=True,
                                        text=True, check=True, timeout=90)
                check_trace(json.loads(result.stdout), n, threshold, name == "nested")
    print("collective refresh: complementary/nested guards, exact covers and map-offset rejection passed")


if __name__ == "__main__":
    main()
