# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check phased persistent banks together with true visit-owned GM output."""
import json
from pathlib import Path
import sys
import tempfile

from check_composed_phases import check, invoke


def owned_source(source, probe, stride=16):
    source = source.replace("%n: index, %m: index)", "%n: index, %m: index, %dst: !pto.ptr<f32, gm>)")
    source = source.replace("array<i64: 3, 2>", "array<i64: 3, 2, 65536>")
    source = source.replace("    %ring = pto.alloc_multi_tile",
        "    %eight = arith.constant 8 : index\n"
        f"    %stride = arith.constant {stride} : index\n"
        f"    %probe = arith.constant {probe} : index\n"
        "    %columns = arith.constant 1073741824 : index\n"
        "    %dv = pto.make_tensor_view %dst, shape = [%one, %columns], "
        "strides = [%columns, %one] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>\n"
        "    %ring = pto.alloc_multi_tile")
    epilogue = '      pto.tsetval ins(%zero, %value : index, f32) outs(%selected : !cell) {test.label = "epilogue"}'
    source = source.replace(epilogue, epilogue + "\n"
        "      %column = arith.muli %visit, %stride overflow<nsw> : index\n"
        "      %output = pto.partition_view %dv, offsets = [%zero, %column], "
        "sizes = [%one, %eight] : !pto.tensor_view<?x?xf32>\n"
        "      pto.tstore ins(%selected : !cell) outs(%output : !pto.partition_tensor_view<1x8xf32>) "
        '{layout = #pto.layout<nd>, test.label = "owned"}')
    return source.replace("    return",
        "    %consumer_address = arith.constant 4096 : i64\n"
        "    %consumer = pto.alloc_tile addr = %consumer_address : !cell\n"
        "    %input = pto.partition_view %dv, offsets = [%zero, %probe], "
        "sizes = [%one, %eight] : !pto.tensor_view<?x?xf32>\n"
        "    pto.tload ins(%input : !pto.partition_tensor_view<1x8xf32>) outs(%consumer : !cell) "
        '{layout = #pto.layout<nd>, test.label = "consume"}\n'
        "    return")


def two_consumers(source):
    """Keep a second owned-byte query outside the first consumer's support."""
    source = owned_source(source, 0)
    return source.replace("    return",
        "    %second_probe = arith.constant 16 : index\n"
        "    %second_address = arith.constant 4128 : i64\n"
        "    %second_consumer = pto.alloc_tile addr = %second_address : !cell\n"
        "    %second_input = pto.partition_view %dv, offsets = [%zero, %second_probe], "
        "sizes = [%one, %eight] : !pto.tensor_view<?x?xf32>\n"
        "    pto.tload ins(%second_input : !pto.partition_tensor_view<1x8xf32>) "
        "outs(%second_consumer : !cell) "
        '{layout = #pto.layout<nd>, test.label = "consume1"}\n'
        "    return")


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="phase-owned-storage-") as scratch:
        path = Path(scratch) / "case.pto"
        # The middle probe is an untouched reservation hole; the other two
        # select owned bytes in distinct outer phases, including absent owners.
        for probe in (0, 8, 16):
            case = owned_source(source, probe)
            for n, m in ((0, 0), (1, 0), (2, 1), (3, 2), (5, 1)):
                path.write_text(case.replace("array<i64: 3, 2, 65536>", f"array<i64: {n}, {m}, 65536>"))
                check(json.loads(invoke(tool, "--structured-trace", path)), n, m,
                      owned_stride=64, owned_probe=4 * probe)
            path.write_text(case)
            report = json.loads(invoke(tool, "--sequence-analysis", path))
            assert not report["error"] and report["prepared"], report
            assert report["phase_descriptions"] > 0 and report["numeric_visits"] == 0, report
        case = two_consumers(source)
        for n, m in ((0, 0), (1, 0), (2, 1), (3, 2), (5, 1)):
            path.write_text(case.replace("array<i64: 3, 2, 65536>", f"array<i64: {n}, {m}, 65536>"))
            check(json.loads(invoke(tool, "--structured-trace", path)), n, m,
                  owned_stride=64, owned_probes=(0, 64))
        path.write_text(case)
        report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert not report["error"] and report["prepared"] and report["phase_descriptions"] > 0, report
        # Consecutive phases now overlap physically. Exact alternative routes
        # may handle the program, but the owned-family phase proof must refuse.
        path.write_text(owned_source(source, 0, stride=4))
        report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert report.get("phase_descriptions", 0) == 0, report
    print("phase-owned storage: exact closures, zero/partial periods, holes and alias rejection passed")


if __name__ == "__main__":
    main()
