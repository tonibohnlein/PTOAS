# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Concrete/possible-alias graph from independent original access intervals.

Different GM roots under may-alias admit possible overlap, without treating
interval offsets relative to different roots as absolute addresses. This is an
upper selected model, not a claim that all possible overlaps coexist physically.
The matrix accumulator protection is the existing supplied model contract.
"""

from check_periodic_demands import closure, native


def reference(items, scalar=(), possible_alias=False):
    edges = native([item["pipe"] for item in items])
    edges.update((2 * a + 1, 2 * b) for a, b in scalar)
    groups, group = [], 0
    for item in items:
        if item["operation"] == "pto.tmatmul":
            group += 1
        elif (
            any(e["space"] == 5 for e in item["effects"])
            and item["operation"] != "pto.tmatmul.acc"
        ):
            group += 1
        groups.append(group)
    for b, second in enumerate(items):
        for a, first in enumerate(items[:b]):
            if first["pipe"] == second["pipe"] == 0:
                continue
            for x in first["effects"]:
                for y in second["effects"]:
                    if x["space"] != y["space"] or (
                        x["mode"] != "write" and y["mode"] != "write"
                    ):
                        continue
                    protected = (
                        x["space"] == 5
                        and groups[a] == groups[b]
                        and all(
                            i["operation"].startswith("pto.tmatmul")
                            for i in (first, second)
                        )
                    )
                    if protected:
                        continue
                    overlap = (
                        possible_alias and x["space"] == 1 and x["base"] != y["base"]
                    ) or (
                        x["base"] == y["base"]
                        and any(
                            lo < end and start < hi
                            for lo, hi in x["intervals"]
                            for start, end in y["intervals"]
                        )
                    )
                    if overlap:
                        edges.add((2 * a + 1, 2 * b))
    reach, covers = closure(2 * len(items), edges)
    return reach, {(a // 2, b // 2) for a, b in covers} - set(scalar)


def main():
    import argparse
    import json
    from pathlib import Path
    from check_whole_kernel_access import inputs, independent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for name, label, text, strategy, trips in inputs(Path.cwd()):
        for alias in ("may-not-alias", "may-alias"):
            directory = (
                "semantics-pypto"
                if strategy == "pypto"
                else "semantics-cumsum"
                if name == "gdn_chunk_cumsum"
                else "semantics-bounded"
            )
            path = args.receipts / directory / name / label / alias / "contract.json"
            row = {
                "kernel": name,
                "configuration": label,
                "alias": alias,
                "raw": str(path),
            }
            try:
                doc = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError) as error:
                row.update(status="no-export", diagnostic=str(error))
                results.append(row)
                continue
            if (
                doc.get("error")
                or doc.get("retained_demand_error")
                or not all(doc.get("present", []))
            ):
                row.update(
                    status="export-or-occurrence-unavailable",
                    diagnostic=doc.get("error")
                    or doc.get("retained_demand_error")
                    or "presence mapping unqualified",
                )
                results.append(row)
                continue
            items, _, _, scalar = independent(text, strategy, trips)
            reach, covers = reference(items, scalar, alias == "may-alias")
            actual = {tuple(pair) for pair in doc["retained_demands"]}
            expected = [
                "".join("1" if bits >> j & 1 else "0" for j in range(len(reach)))
                for bits in reach
            ]
            row.update(
                status="compared",
                covers_equal=actual == covers,
                closure_equal=doc["event_rows"] == expected,
                selected_model_covers=len(covers),
                actual_covers=len(actual),
                extra=sorted(actual - covers)[:5],
                missing=sorted(covers - actual)[:5],
            )
            results.append(row)
    (args.receipts / "selected-model-results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    if any(
        row["status"] == "compared"
        and not (row["covers_equal"] and row["closure_equal"])
        for row in results
    ):
        raise RuntimeError(
            "Selected-model closure or cover comparison failed; see receipt"
        )
    for row in results:
        if row["status"] == "compared":
            print(
                row["kernel"],
                row["configuration"],
                row["alias"],
                row["covers_equal"],
                row["closure_equal"],
            )


if __name__ == "__main__":
    main()
