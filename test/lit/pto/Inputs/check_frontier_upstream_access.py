# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise shared upstream access refinements through real Frontier requests."""

import argparse
import json
from pathlib import Path
import re

from assess_whole_kernel_access import annotate, digest, invoke, inventory, profile
from check_compact_persistent_contract import annotated
from check_expanded_endpoint_adapter import fixture as expanded_fixture
from compact_persistent_oracle import boundary, merge, probes
from compact_vector_oracle import command_order
from reaching_access_research import require
from upstream_access_reference import l1_case, acc_case, row_case, required


def run(command, output):
    result = invoke([str(x) for x in command], output, 35)
    require(result["status"] == "success", str(result) + Path(result["stderr"]).read_text()[-2000:])
    return result, Path(result["stdout"]).read_text()


def unroll(text):
    ends = []
    for loop in re.finditer(r"\bscf.for\s", text):
        opening = text.index("{", loop.start())
        depth, end = 1, opening + 1
        while depth and end < len(text):
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        require(depth == 0, "unbalanced validation fixture")
        ends.append(end)
    for end in sorted(ends, reverse=True):
        text = text[:end] + ' {pto.unroll = "full"}' + text[end:]
    return text


def check_accesses(args, source, word, folder):
    # Only the independent validation slice is unfolded. The production input
    # below retains its original loops, coordinates and branch structure.
    path = folder / "access.input.pto"
    path.write_text(unroll(source))
    for index in range(4):
        target = folder / f"access.flat{index}.pto"
        run([args.optimizer, "--canonicalize", "--pto-unroll-loops", path, "-o", target],
            folder / f"unroll{index}.stdout")
        path = target
    _, output = run([args.trace_tool, "--step1-json", path], folder / "access.json")
    data = json.loads(output)
    records = data["access_records"]
    expected = [access for event in word if event["active"] for access in event["effects"]]
    require(len(records) == len(expected), f"access inventory mismatch: {len(records)} != {len(expected)}")
    exact = enclosed = 0
    for actual, reference in zip(records, expected):
        require(actual["space"] == reference["space"] and actual["mode"] == reference["mode"],
                "changed effect role/space")
        if reference["space"] == 1:
            continue  # This fixture does not qualify complete GM selectors.
        spans = merge((entry["begin"], entry["end"]) for entry in actual["ranges"])
        require(actual["ranges_materialized"], "missing concrete local bound")
        require(all(any(lo <= begin and end <= hi for lo, hi in spans)
                    for begin, end in reference["intervals"]), "uncovered original local footprint")
        if actual["access_description"] == "declared-region":
            require(spans == reference["intervals"], "declared exact footprint differs")
            exact += 1
        else:
            enclosed += 1
    return dict(records=len(records), exact_local=exact, enclosing_local=enclosed)


def regional(args, source, word, folder):
    cells = probes(word)
    cells.append((6, -1, 999999))  # Useful proven absence remains available.
    path = folder / "regional.pto"
    path.write_text(annotate(annotated(source, word, cells), "test.contract_exports = 3 : i64"))
    result, output = run([args.trace_tool, "--compact-region-contract", path], folder / "regional.json")
    data = json.loads(output)
    require(not data.get("error"), str(data))
    rows, covers = required(word)
    strings = ["".join("1" if row >> j & 1 else "0" for j in range(len(rows))) for row in rows]
    require(data["event_rows"] == strings, "all-event required closure differs")
    demand_error = data.get("retained_demand_error", "")
    if demand_error:
        # This existing test adapter cannot unfold symbolic arithmetic records.
        # Keep the missing check explicit; queries/selection/commands still run.
        require(demand_error == "nonempty arithmetic record expansion unavailable in this test adapter",
                "unexpected retained-demand test capability refusal: " + demand_error)
    else:
        require({tuple(pair) for pair in data["retained_demands"]} == covers, "actual retained covers differ")
    boundaries = data["boundaries"]
    require(not boundaries.get("error"), str(boundaries))
    expected = [boundary(word, cell) for cell in cells]
    for collection in (expected, boundaries["rows"]):
        for row in collection:
            for key in ("first_readers", "last_readers"):
                row[key] = {pipe: values for pipe, values in row[key].items() if values}
    require(boundaries["rows"] == expected, "local boundary selectors differ")
    return dict(backend=data["backend"], occurrences=len(word), covers=len(covers),
                retained_cover_check=demand_error or "passed",
                all_event_comparisons=len(rows)**2, selectors=len(cells),
                seconds=result.get("seconds"), input_sha256=digest(path))


def emitted(args, source, word, folder):
    path = folder / "production.input.pto"
    path.write_text(source)
    options = "--pto-frontier-analysis=physical-output=true gm-alias=may-not-alias profile-stages=true"
    compiled, text = run(
        [args.optimizer, options, path],
        folder / "physical.pto")
    trace_input = folder / "physical.trace.pto"
    trace_input.write_text(annotate(text, "test.trace_existing"))
    _, output = run([args.trace_tool, "--structured-trace", trace_input], folder / "physical.trace.json")
    trace = json.loads(output)
    require(trace.get("accepted") and not trace["trace"].get("error"), str(trace))
    trace = trace["trace"]
    payloads = [event for event in trace["events"] if event["kind"] == "payload"]
    active = [event for event in word if event["active"]]
    require([(event["type"], event["coordinates"], event["pipe"]) for event in payloads] ==
            [(event["phase"], event["coordinates"], event["pipe"]) for event in active],
            "payload/control sequence changed")
    reach, reuse = command_order(trace, include_commands=True)
    require(not any(row >> i & 1 for i, row in enumerate(reach)), "emitted cycle")
    if any(event["kind"] == "set" for event in trace["events"]):
        require(reuse["status"] == "passed", str(reuse))
    rows, _ = required(active)
    require(all(not (row & ~(reach[i] | (1 << i))) for i, row in enumerate(rows)),
            "emitted prerequisite missing")
    require(all(event["physical_id"] in range(6) for event in trace["events"]
                if event["kind"] in ("set", "wait")), "ineligible ID")
    commands = [event for event in trace["events"] if event["kind"] != "payload"]
    excess = sum(bool(reach[2*a+1] >> (2*b) & 1) and not bool(rows[2*a+1] >> (2*b) & 1)
                 for a in range(len(active)) for b in range(len(active)))
    ci = [[a, b] for a in range(len(active)) for b in range(len(active)) if reach[2*a+1] >> (2*b) & 1]
    return dict(payloads=len(active), commands=len(commands), excess_ci=excess, reuse=reuse,
                ci_pairs=ci, static=inventory(folder / "physical.pto"), seconds=compiled["seconds"],
                profiles=profile(Path(compiled["stderr"])))


def negative_views(args):
    result = []
    original = row_case()[1]
    unknown_base = original.replace("%out: !pto.ptr<f32, gm>)",
                                    "%out: !pto.ptr<f32, gm>, %base: i64)").replace(
                                        "%parent = pto.alloc_tile addr = %zero",
                                        "%parent = pto.alloc_tile addr = %base")
    unknown_row = original.replace("%out: !pto.ptr<f32, gm>)",
                                   "%out: !pto.ptr<f32, gm>, %index: index)").replace(
                                       "%parent[%row, %c0]", "%parent[%index, %c0]")
    short_row = original.replace("128", "32")
    for label, source in (("nonunit-step", row_case(step=2)[1]),
                          ("unknown-base", unknown_base), ("unknown-row", unknown_row),
                          ("unaligned-vector-base", row_case(base=32)[1]), ("short-vector-row", short_row)):
        path = args.output / (label + ".pto")
        path.write_text(source)
        _, output = run([args.trace_tool, "--step1-json", path], args.output / (label + ".json"))
        writes = [entry for entry in json.loads(output)["access_records"]
                  if entry["operation"] == "pto.tload" and entry["space"] == 6 and entry["mode"] == "write"]
        require(len(writes) == 1 and writes[0]["access_description"] == "allocation-enclosure" and
                writes[0]["declared_selections"] == 0, "unsupported row view gained an exact footprint")
        result.append(dict(case=label, declaration="unqualified; conservative enclosure retained",
                           native_execution_claimed=False))
    invalid = args.output / "unaligned-base.pto"
    invalid.write_text(row_case(base=16)[1])
    refused = invoke([str(args.trace_tool), "--step1-json", str(invalid)],
                     args.output / "unaligned-base.json", 35)
    require(refused["status"] == "failure" and
            "addr must be aligned to 32 bytes" in Path(refused["stderr"]).read_text(),
            "invalid allocation alignment escaped the existing verifier")
    result.append(dict(case="unaligned-base", declaration="invalid IR; verifier refusal preserved"))
    return result


def original_row_effects(args, source, word, folder):
    path = folder / "original-effects.pto"
    path.write_text(source)
    _, output = run([args.trace_tool, "--step1-json", path], folder / "original-effects.json")
    records = json.loads(output)["access_records"]
    writes = [entry for entry in records if entry["operation"] in ("pto.tload", "pto.tmov") and
              entry["space"] == 6 and entry["mode"] == "write"]
    if any(event["active"] and event["phase"] in (2, 3) for event in word):
        require(len(writes) == 2 and all(entry["access_description"] == "declared-region" and
                                       len(entry["regions"]) == 1 for entry in writes),
                "original loop still lost its exact row access; unrolling must not hide the gap")
    return dict(original_records=len(records), row_regions=sum(len(entry["regions"]) for entry in writes),
                descriptions=[entry["access_description"] for entry in writes])


def allocation_frames(args):
    rows = []
    for selected, closed in ((0, True), (1, False)):
        attrs = ('test.contract_math_only, test.contract_strategy = "expanded", '
                 f'test.contract_loop = {selected} : i64, test.contract_exports = 1 : i64, '
                 'test.contract_allocation_probe')
        path = args.output / f"allocation-frame-{selected}.pto"
        path.write_text(annotate(expanded_fixture(2, 1), attrs))
        _, output = run([args.trace_tool, "--compact-region-contract", path],
                        args.output / f"allocation-frame-{selected}.json")
        data = json.loads(output)
        require(data.get("backend") == "expanded-finite-guarded" and data.get("prepared") and
                data.get("retained_owner_reusable") and data.get("queries_retained"), str(data))
        require(data.get("allocated") == closed, "unbound enclosing visits gained an allocation certificate")
        rows.append(dict(selected_loop=selected, closed_invocation=closed, result=data))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("optimizer", "trace-tool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [l1_case(count, later) for count, later in ((0, 64), (1, 64), (2, 64), (3, 64), (1, 16))]
    cases.append(l1_case(second_window=True))
    cases += [acc_case(False), acc_case(True)]
    cases += [row_case(count, resets, same) for count, resets, same in
              ((0, 1, False), (1, 1, False), (3, 1, False), (7, 1, False), (3, 2, False), (3, 1, True))]
    rows = []
    for name, source, word in cases:
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        access_result = check_accesses(args, source, word, folder)
        queries = regional(args, source, word, folder)
        protocol = emitted(args, source, word, folder)
        original = original_row_effects(args, source, word, folder) if name.startswith("rows_") else {}
        rows.append(dict(name=name, accesses=access_result, regional=queries, physical=protocol,
                         original_effects=original))
    unavailable_covers = {row["name"] for row in rows if row["regional"]["retained_cover_check"] != "passed"}
    require(unavailable_covers <= {"l1_2_64", "l1_3_64"},
            "retained-cover test capability regressed for " + str(sorted(unavailable_covers)))
    negatives = negative_views(args)
    frames = allocation_frames(args)
    (args.output / "receipt.json").write_text(
        json.dumps(dict(cases=rows, negatives=negatives, allocation_frames=frames), indent=2) + "\n")
    cover_checks = sum(row["regional"]["retained_cover_check"] == "passed" for row in rows)
    print(f"Frontier upstream access benefits: {len(rows)} access/query/selector/physical cases passed; "
          f"{cover_checks} actual retained-cover checks; "
          f"{len(rows) - cover_checks} explicit arithmetic test-adapter limitations")


if __name__ == "__main__":
    main()
