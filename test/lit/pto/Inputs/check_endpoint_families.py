# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check exact endpoint-family membership, paired maps, cut order and persistence."""
from collections import defaultdict
from pathlib import Path
import shutil
import sys
import tempfile

from check_logical_insertion import recognized
from check_physical_allocation import opt


def coordinates(anchor):
    return tuple((item["induction"], item["value"]) for item in anchor["coordinates"])


def coexecute(left, right):
    known = dict(left)
    return all(loop not in known or known[loop] == value for loop, value in right)


def check_families(plan):
    assert not plan["error"] and not plan["family_error"], plan
    recipes, anchors, families = plan["recipes"], plan["anchors"], plan["families"]
    by_record = defaultdict(list)
    for position, recipe in enumerate(recipes):
        by_record[recipe["record"]].append((position, recipe))
    owners = {}
    assert len({family["id"] for family in families}) == len(families)
    for family in families:
        assert family["members"]
        assert family["id"] == min(member["record"] for member in family["members"])
        source_points, target_points = set(), set()
        for member in family["members"]:
            record = member["record"]
            assert record not in owners, ("duplicate family member", record)
            owners[record] = family
            assert record in by_record, ("unknown family member", record)
            kinds = [recipe["kind"] for _, recipe in by_record[record]]
            assert sorted(kinds) in (["set", "wait"], ["barrier"]), kinds
            for _, recipe in by_record[record]:
                assert (recipe["source"], recipe["target"]) == (member["source"], member["target"])
            source, target = anchors[member["source"]], anchors[member["target"]]
            assert source["after_cut"] == family["source_cut"]
            assert target["before_cut"] == family["target_cut"]
            for anchor, seen in [(source, source_points), (target, target_points)]:
                point = coordinates(anchor)
                assert point not in seen, ("ambiguous endpoint member", family, point)
                seen.add(point)
    assert owners.keys() == by_record.keys(), "family partition lost a record"

    # Existing insertion specifies SET, then barriers by pipe, then WAIT at a
    # shared cut. Within each category it preserves the original recipe order.
    kind_order = {"set": 0, "barrier": 1, "wait": 2}
    cuts = defaultdict(list)
    for position, recipe in enumerate(recipes):
        side = "source" if recipe["kind"] == "set" else "target"
        family = owners[recipe["record"]]
        anchor = anchors[recipe[side]]
        pipe_order = recipe["pipe"] if recipe["kind"] == "barrier" else 0
        cuts[family[side + "_cut"]].append((
            (kind_order[recipe["kind"]], pipe_order, position),
            family[side + "_order"], coordinates(anchor), (family["id"], recipe["kind"])))
    for entries in cuts.values():
        entries.sort()
        for index, left in enumerate(entries):
            for right in entries[index + 1:]:
                if left[3] != right[3] and coexecute(left[2], right[2]):
                    assert left[1] < right[1], ("coexecuting command order changed", left, right)
    return max((len(family["members"]) for family in families), default=0)


def with_hole(source):
    source = source.replace("%inner_end = arith.constant 9 : index",
                            "%inner_end = arith.constant 9 : index\n    %five = arith.constant 5 : index")
    first = "        pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile0 : !tile)"
    last = "        pto.textract ins(%tile1, %zero, %zero : !tile, index, index) outs(%left1 : !left)"
    source = source.replace(first, "        %keep = arith.cmpi ne, %k, %five : index\n        scf.if %keep {\n" + first)
    return source.replace(last, last + "\n        }")


def main():
    tool, optimizer = shutil.which(sys.argv[1]), shutil.which(sys.argv[2])
    assert tool and optimizer, "test tools must be available"
    source = Path(sys.argv[3]).read_text()
    cases = [(f"inner-{end}", source.replace("constant 9 : index", f"constant {end} : index"))
             for end in [1, 2, 5, 8, 9]]
    cases.append(("hole", with_hole(source)))
    grouped = 0
    with tempfile.TemporaryDirectory(prefix="endpoint-families-") as directory:
        path = Path(directory) / "case.pto"
        emitted_path = Path(directory) / "logical.pto"
        for name, text in cases:
            path.write_text(text)
            plan = recognized(tool, path)["logical_endpoints"]
            members = check_families(plan)
            grouped += members > 1
            if name in ["inner-5", "inner-8", "inner-9", "hole"]:
                assert members > 1, (name, "repeated coordinate recipes were not grouped")
            emitted = opt(optimizer, path, ["--pto-frontier-analysis"]).stdout
            assert ("pto.endpoint_families" in emitted) == bool(plan["families"])
            assert "pto.set_flag" not in emitted and "pto.wait_flag" not in emitted
            for operation in ["scf.for", "pto.tload", "pto.textract"]:
                assert emitted.count(operation) == text.count(operation)
            emitted_path.write_text(emitted)
            assert opt(optimizer, emitted_path, []).stdout == emitted, "family metadata changed on roundtrip"
    print(f"endpoint families: {len(cases)} exact partitions and order checks, {grouped} grouped cases")


if __name__ == "__main__":
    main()
