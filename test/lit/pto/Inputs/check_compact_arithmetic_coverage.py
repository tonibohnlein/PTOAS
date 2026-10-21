# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare existing regional strategies; all expansion is diagnostic/reference only."""

import argparse
import json
from pathlib import Path
import re
from assess_whole_kernel_access import annotate, digest, invoke
from check_compact_persistent_contract import annotated, run
from compact_persistent_oracle import boundary, merge
from finite_indexed_reference import check_queries, constants, source, words

CASES = (
    ("lossless_block_cast", "reduce", 1),
    ("lossless_block_cast", "correct", 5),
    ("lossless_block_cast", "expand", 7),
    ("gdn_chunk_cumsum", "cumsum", 0),
)


def block(text, ordinal):
    begin = list(re.finditer(r"scf.for\s", text))[ordinal].start()
    opening = text.index("{", begin)
    depth, end = 1, opening + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise RuntimeError("unbalanced reference loop")
    return text[begin:end]


def access_reference(args, text, loop, items, folder):
    # Slice only for the independent access oracle. Source reservations and
    # typed reinterpretations are retained; memory is assumed initialized on
    # entry to this slice. Production derivatives retain the full initializer.
    constants_and_allocs = [
        line
        for line in text.splitlines()
        if " = arith.constant " in line or " = pto.alloc_tile " in line
    ]
    selected = block(text, loop)
    closes = []
    for occurrence in re.finditer(r"scf.for\s", selected):
        start = selected.index("{", occurrence.start())
        depth, end = 1, start + 1
        while depth:
            depth += (selected[end] == "{") - (selected[end] == "}")
            end += 1
        closes.append(end)
    for end in sorted(closes, reverse=True):
        selected = selected[:end] + ' {pto.unroll = "full"}' + selected[end:]
    fragment = (
        'module attributes {pto.target_arch = "a3"} {\n'
        "func.func @access_slice() attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {\n"
        + "\n".join(constants_and_allocs)
        + "\n"
        + selected
        + "\nreturn\n}}\n"
    )
    path = folder / "access-slice.pto"
    path.write_text(fragment)
    for level in range(len(closes) + 1):
        target = folder / f"access-flat-{level}.pto"
        run(
            [
                args.optimizer,
                "--canonicalize",
                "--pto-unroll-loops",
                path,
                "-o",
                target,
            ],
            folder / f"unroll-{level}.stdout",
            timeout=35,
        )
        path = target
    output = folder / "accesses.json"
    run([args.trace_tool, "--step1-json", path], output, timeout=35)
    recovered = json.loads(output.read_text())["access_records"]
    expected = [effect for item in items for effect in item["effects"]]
    if len(recovered) != len(expected):
        raise RuntimeError("reference access count differs")
    for actual, reference in zip(recovered, expected):
        intervals = merge((x["begin"], x["end"]) for x in actual["ranges"])
        if (
            actual["space"] != reference["space"]
            or actual["mode"] != reference["mode"]
            or not actual["ranges_materialized"]
            or intervals != reference["intervals"]
            or actual["access_description"] != "declared-region"
        ):
            raise RuntimeError(f"independent scalar access mismatch: {actual}")
    return len(recovered)


def request(
    args, text, label, loop, strategy, exports, folder, sample=None, cells=None
):
    path = folder / (label + ".pto")
    attrs = (
        f'test.contract_strategy = "{strategy}", test.contract_loop = {loop} : i64, '
        f"test.contract_exports = {exports} : i64"
    )
    if strategy == "arithmetic" and sample is not None:
        attrs += ", test.contract_decode_checks"
    if sample is None:
        attrs += ", test.contract_math_only"
    else:
        text = annotated(text, sample, cells)
        attrs += ", " + constants(text)
    path.write_text(annotate(text, attrs))
    command = [args.trace_tool, "--compact-region-contract", path]
    result = invoke(command, path.with_suffix(".json"), 35)
    data = (
        json.loads(Path(result["stdout"]).read_text())
        if result["status"] == "success"
        else {}
    )
    return {"label": label, "run": result, "input_sha256": digest(path), "result": data}


def qualify(args, folder):
    rows = []
    for name, kind, loop in CASES:
        text = source(args.root, name, 1)
        items = words(kind, 1, 8)
        cells = sorted(
            {(6, -1, e["intervals"][0][0]) for x in items for e in x["effects"]}
        )[:8]
        cells.append((6, -1, 999999))
        sample = (
            items
            if kind != "expand"
            else [
                x
                for x in items
                if x["coordinates"][1] in (0, 1, 31, 32, 63, 64, 95, 96, 127)
            ]
        )
        # Boundary checks may select an occurrence outside the query sample.
        # Include those independent reference occurrences before querying.
        needed = {items.index(item) for item in sample}
        for cell in cells:
            selected = boundary([dict(item, active=True) for item in items], cell)
            needed.update(selected["first_writers"] + selected["last_writers"])
            for key in ("first_readers", "last_readers"):
                for values in selected[key].values():
                    needed.update(values)
        sample = [item for i, item in enumerate(items) if i in needed]
        directory = folder / kind
        directory.mkdir(exist_ok=True)
        access_count = access_reference(args, text, loop, items, directory)
        for strategy in ("production", "arithmetic", "numerical", "expanded"):
            row = request(
                args, text, strategy, loop, strategy, 3, directory, sample, cells
            )
            if strategy == "arithmetic" and not row["result"].get("decode_checks", {}).get(
                "unsupported_frame_unavailable"
            ):
                raise RuntimeError("unsupported occurrence frame became an absent answer")
            row["region"], row["access_comparisons"] = kind, access_count
            checked = check_queries(row["result"], sample, items, cells)
            row["independent"] = checked
            if checked["status"] == "checked":
                if not checked["all_four_queries_equal"] or not checked.get(
                    "selectors_equal"
                ):
                    raise RuntimeError(
                        f"query/selector mismatch: {kind}/{strategy}: {checked}"
                    )
                if (
                    strategy in ("arithmetic", "expanded")
                    and not checked["retained_covers_equal"]
                ):
                    raise RuntimeError(
                        f"actual retained-cover expansion unavailable: {checked}"
                    )
            elif strategy != "numerical":
                raise RuntimeError(f"unexpected unavailable export: {row}")
            rows.append(row)
            rows.append(
                request(
                    args, text, strategy + "-endpoints", loop, strategy, 7, directory
                )
            )
    return rows


def scaling(args, folder):
    cases = [
        (name, kind, loop, n, 8)
        for name, kind, loop in (CASES[0], CASES[2])
        for n in (1, 2, 8, 64)
    ]
    cases += [
        ("gdn_chunk_cumsum", "cumsum", 0, n, width)
        for n, width in ((1, 8), (2, 8), (8, 8), (8, 32), (8, 128))
    ]
    rows = []
    for name, kind, loop, count, width in cases:
        text = source(args.root, name, count, width)
        for strategy in ("production", "arithmetic", "expanded"):
            label = f"{kind}-{count}-{width}-{strategy}"
            row = request(args, text, label, loop, strategy, 0, folder)
            row.update(region=kind, count=count, width=width, strategy=strategy)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scaling", action="store_true")
    args = parser.parse_args()
    args.root = Path(__file__).resolve().parents[4]
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "modeled scalar bytes; no device qualification",
        "qualified": qualify(args, args.output),
    }
    if args.scaling:
        result["scaling"] = scaling(args, args.output)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print("indexed arithmetic regional checks passed")


if __name__ == "__main__":
    main()
