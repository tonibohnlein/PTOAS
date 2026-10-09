# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check demand-only common translations without changing physical footprints."""
from pathlib import Path
import sys
import tempfile
from check_finite_expansion import invoke
from check_fixed_branches import check_graph


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text().replace("test.finite_expansion_session",
                                             "test.translation_compare, test.sample_signed")
    source = source.replace("    %base = arith.constant 0 : i64\n", "")
    source = source.replace("%banks = pto.alloc_multi_tile addr = %base",
                            "%banks = pto.alloc_multi_tile addr = %bankbase")
    source = source.replace("    %mat = pto.alloc_tile",
                            "    %bankbase = arith.constant 0 : i64\n    %mat = pto.alloc_tile")
    source = source.replace("%slot = arith.remui %i, %two : index", "%slot = arith.constant 0 : index")
    origins = ["%base = arith.index_cast %outer : index to i64",
        "%block = pto.get_block_idx\n    %index = arith.index_cast %block : i64 to index\n"
        "    %quotient = arith.divui %index, %three : index\n"
        "    %remainder = arith.remui %index, %three : index\n"
        "    %origin = arith.addi %quotient, %remainder : index\n"
        "    %base = arith.index_cast %origin : index to i64"]
    with tempfile.TemporaryDirectory(prefix="finite-translation-") as directory:
        path = Path(directory) / "case.pto"
        for origin in origins:
            for guarded, periodic in [(False, False), (True, False), (True, True)]:
                rendered = source.replace("    %mat = pto.alloc_tile", "    " + origin + "\n    %mat = pto.alloc_tile")
                if not guarded:
                    rendered = rendered.replace("%first = arith.cmpi eq, %outer, %i : index",
                                                "%first = arith.constant false")
                if periodic:
                    rendered = rendered.replace("@finite(%outer: index)", "@finite(%outer: index, %second: index)")
                    rendered = rendered.replace("%slot = arith.constant 0 : index",
                                                "%slot = arith.remui %second, %two : index")
                path.write_text(rendered)
                for policy in ["may-not-alias", "may-alias"]:
                    form, expanded = invoke(tool, path, policy)
                    assert expanded["translation_fragments"] > 0, expanded
                    assert expanded["translation_checked"], expanded
                    assert expanded["pruned_arms"] == (0 if guarded else 2)
                    if periodic:
                        assert form["period"] == 2, form
                    check_graph(form, expanded)
                    # The retained absolute primitive still contains parameter
                    # coefficients; relative domains do not overwrite it.
                    physical = [relation for relation in form["relations"] if relation["kind"] in (4, 5)]
                    assert any(any(coefficient for coefficient in row["coefficients"][1:])
                               for relation in physical for piece in relation["pieces"] for row in piece["rows"])
                    assert path.read_text() == rendered
    print("finite-translation: signed/div-rem origins, complete families, physical retention and graphs checked")


if __name__ == "__main__":
    main()
