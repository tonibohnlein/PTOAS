# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independently check finite byte boundaries against concrete IR accesses."""
from pathlib import Path
import sys
import tempfile
from check_finite_expansion import invoke


def active(choices):
    assert all(present in (0, 1) for _, present in choices)
    return [site for site, present in choices if present]


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text().replace("test.finite_expansion_session", "test.expanded_storage")
    cases = [("base", source),
             ("zero", source.replace("to %five step", "to %one step")),
             ("fixed", source.replace("%first = arith.cmpi eq, %outer, %i : index",
                                      "%first = arith.constant false")),
             ("parity", source.replace("@finite(%outer: index)", "@finite(%outer: index, %second: index)")
              .replace("%slot = arith.remui %i, %two", "%slot = arith.remui %second, %two")),
             ("rmw", source.replace("} else {\n        pto.tmatmul ins(%left, %right : !left, !right)",
                                    "} else {\n        pto.tmatmul.acc "
                                    "ins(%acc, %left, %right : !acc, !left, !right)"))]
    translated = source.replace("test.expanded_storage", "test.expanded_storage, test.sample_signed")
    translated = translated.replace("    %base = arith.constant 0 : i64\n", "")
    translated = translated.replace("%banks = pto.alloc_multi_tile addr = %base",
                                    "%banks = pto.alloc_multi_tile addr = %bankbase")
    translated = translated.replace("    %mat = pto.alloc_tile",
                                    "    %bankbase = arith.constant 0 : i64\n    %mat = pto.alloc_tile")
    direct = translated.replace("    %mat = pto.alloc_tile",
                                "    %base = arith.index_cast %outer : index to i64\n    %mat = pto.alloc_tile")
    cases.append(("translated-direct", direct))
    cases.append(("translated-fixed", direct.replace("%first = arith.cmpi eq, %outer, %i : index",
                                                     "%first = arith.constant false")))
    cases.append(("translated-parity", direct.replace("@finite(%outer: index)",
                                                      "@finite(%outer: index, %second: index)")
                  .replace("%slot = arith.remui %i, %two", "%slot = arith.remui %second, %two")))
    divrem = translated.replace("    %mat = pto.alloc_tile",
                                "    %block = pto.get_block_idx\n"
                                "    %index = arith.index_cast %block : i64 to index\n"
                                "    %quotient = arith.divui %index, %three : index\n"
                                "    %remainder = arith.remui %index, %three : index\n"
                                "    %origin = arith.addi %quotient, %remainder : index\n"
                                "    %base = arith.index_cast %origin : index to i64\n    %mat = pto.alloc_tile")
    cases.append(("translated-divrem", divrem))
    with tempfile.TemporaryDirectory(prefix="finite-storage-") as directory:
        path = Path(directory) / "case.pto"
        for name, rendered in cases:
            path.write_text(rendered)
            for policy in ["may-not-alias", "may-alias"]:
                form, expanded = invoke(tool, path, policy)
                assert expanded["exact_selectors"], expanded
                assert expanded["invalid_storage_rejected"], expanded
                assert expanded["query_snapshot_unchanged"], expanded
                assert expanded["preflight_failure_retained"], expanded
                if name.startswith("translated"):
                    assert expanded["storage_translation_families"] == (4 if name == "translated-parity" else 3), name
                    assert expanded["selector_checks"] > 0, expanded
                for sample in expanded["samples"]:
                    sites = form["sites"]
                    present = [i for i, value in enumerate(sample["presence"]) if value]
                    pipes = {sites[i]["pipe"] for i in present}
                    assert {pipe: active(choices) for pipe, choices in sample["first_payloads"]} == {
                        pipe: [next(i for i in present if sites[i]["pipe"] == pipe)] for pipe in pipes}
                    assert {pipe: active(choices) for pipe, choices in sample["last_payloads"]} == {
                        pipe: [next(i for i in reversed(present) if sites[i]["pipe"] == pipe)] for pipe in pipes}
                    assert {site: active(choices) for site, choices in sample["first_sites"]} == {
                        i: [i] if i in present else [] for i in range(len(sites))}
                    expected_addresses = {(space, byte) for space in (2, 3, 4, 5)
                                          for byte in (-1, 0, 511, 512, 1023, 1024, 2047, 2048)}
                    assert len(sample["storage"]) == 32
                    assert {(item["space"], item["byte"]) for item in sample["storage"]} == expected_addresses
                    for boundary in sample["storage"]:
                        space, byte = boundary["space"], boundary["byte"]
                        reads, writes = {}, {}
                        origin = 0
                        if name.startswith("translated"):
                            parameter = form["parameters"].index(-1 if name == "translated-divrem" else 0)
                            origin = sample["parameters"][parameter]
                            if name == "translated-divrem":
                                origin = origin // 3 + origin % 3
                        selected_byte = byte - origin if space in (2, 4, 5) else byte
                        for i in present:
                            site = sites[i]
                            bank = (sample["parameters"][form["parameters"].index(1)] % 2
                                    if name in ("parity", "translated-parity") else site["fixed"][-1] % 2)
                            extract = site["op"] == "pto.textract"
                            r = ((extract and space == 2 and 0 <= selected_byte < 512) or
                                 (not extract and space == 3 and 512 * bank <= byte < 512 * (bank + 1)) or
                                 (not extract and space == 4 and 0 <= selected_byte < 512) or
                                 (site["op"] == "pto.tmatmul.acc" and space == 5 and 0 <= selected_byte < 1024))
                            w = ((extract and space == 3 and 512 * bank <= byte < 512 * (bank + 1)) or
                                 (not extract and space == 5 and 0 <= selected_byte < 1024))
                            if r: reads[i] = site["pipe"]
                            if w: writes[i] = site["pipe"]
                        assert boundary["member"] == bool(reads or writes), (name, boundary)
                        assert active(boundary["first_writers"]) == list(writes)[:1], (name, boundary)
                        assert active(boundary["last_writers"]) == list(writes)[-1:], (name, boundary)
                        before = {pipe: next((i for i in reads if reads[i] == pipe and i not in writes
                                             and (not writes or i < min(writes))), None)
                                  for pipe in set(reads.values())}
                        after = {pipe: next((i for i in reversed(reads) if reads[i] == pipe and i not in writes
                                            and (not writes or i > max(writes))), None)
                                 for pipe in set(reads.values())}
                        actual_first = {pipe: active(choices) for pipe, choices in boundary["first_readers"]
                                        if active(choices)}
                        actual_last = {pipe: active(choices) for pipe, choices in boundary["last_readers"]
                                       if active(choices)}
                        expected_first = {pipe: [i] for pipe, i in before.items() if i is not None}
                        assert actual_first == expected_first, (name, boundary)
                        expected_last = {pipe: [i] for pipe, i in after.items() if i is not None}
                        assert actual_last == expected_last, (name, boundary)
                assert path.read_text() == rendered
        alias = source.replace("@finite(%outer: index)",
                               "@finite(%outer: index, %p: !pto.ptr<f32, gm>, %q: !pto.ptr<f32, gm>)")
        alias = alias.replace("    // BODY", "    %scalar = arith.constant 1.0 : f32\n    // BODY")
        alias = alias.replace("      %slot =", "      pto.store %scalar, %p[%zero] : !pto.ptr<f32, gm>, f32\n"
                              "      %loaded = pto.load %p[%zero] : !pto.ptr<f32, gm> -> f32\n      %slot =")
        path.write_text(alias)
        for policy in ["may-not-alias", "may-alias"]:
            _, expanded = invoke(tool, path, policy)
            assert expanded["exact_selectors"] and expanded["alias_queries_checked"], expanded
            assert expanded["alias_query_count"] > 0, expanded
    print("finite-storage: exact byte support, writer/reader extrema, native extrema and original coordinates checked")


if __name__ == "__main__":
    main()
