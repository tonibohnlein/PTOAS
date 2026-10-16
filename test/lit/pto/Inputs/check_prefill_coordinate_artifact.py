# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Audit one retained prefill coordinate export without assigning physical IDs.

The original payload and exact logical protocol are checked at explicit scalar
valuations. N=0 is only a structural zero-trip control: its original state-update
loads require a padding premise. Other cases assume valid original storage.
Unobserved families and sharing edges remain covered only by producer proofs.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from check_fixed_protocol_capacity import capacity, same_fixed_protocol
from check_logical_insertion import closure_with_commands
from check_prefill_protocol_capacity import bind_prefill, payloads

COORDINATE_RECORDS = {37, 38, 39, 40, 48, 53, 61, 66, 74, 79, 87, 92, 93, 94, 95, 96}
SOURCE_SHA256 = "4a2da46ad42fbded8d15dd4b64cdabc93ae0df9570ea6d5d165712361ce16831"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_fields(text):
    """Split this certificate's printer vocabulary, never arbitrary MLIR."""
    stack, quoted, escaped, start, fields = [], False, False, 0, []
    closing = {"}": "{", "]": "[", ">": "<", ")": "("}
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[<(":
            stack.append(char)
        elif char in "}]>)":
            assert stack and stack.pop() == closing[char], "unbalanced certificate"
        elif char == "," and not stack:
            fields.append(text[start:index].strip())
            start = index + 1
    assert not stack and not quoted, "unterminated certificate"
    tail = text[start:].strip()
    if tail:
        fields.append(tail)
    return fields


def parse_attribute(text):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        result = {}
        for field in split_fields(text[1:-1]):
            pair = field.split(" = ", 1)
            assert pair[0] not in result, "duplicate certificate field"
            result[pair[0]] = parse_attribute(pair[1]) if len(pair) == 2 else True
        return result
    if text.startswith("[") and text.endswith("]"):
        return [parse_attribute(field) for field in split_fields(text[1:-1])]
    if text == "array<i64>":
        return []
    if re.fullmatch(r"array<i64: [-0-9, ]+>", text):
        return [int(value) for value in text[11:-1].split(",")]
    if re.fullmatch(r"-?[0-9]+ : i64", text):
        return int(text.split()[0])
    if text in ("true", "false"):
        return text == "true"
    if text.startswith('"') and text.endswith('"'):
        return json.loads(text)
    raise AssertionError("unsupported certificate vocabulary: " + text[:100])


def function_attribute(text, name):
    prefix = name + " = "
    assert text.count(prefix) == 1, "missing or duplicate owning attribute"
    start = text.index(prefix) + len(prefix)
    assert text[start] == "{"
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return parse_attribute(text[start:index + 1])
    raise AssertionError("unterminated owning attribute")


def metadata(text):
    certificate = function_attribute(text, "pto.finite_allocation")
    directory = function_attribute(text, "pto.endpoint_families")
    assert certificate["version"] == 2 and certificate["kind"] == "finite"
    assert certificate["strategy"] == "regional-palettes" and certificate["macro_reservations"]
    assert certificate["plan"] == directory["plan"]
    groups = certificate["groups"]
    assert len(groups) == 55 and sum(group["budget"] == 2 for group in groups) == 16
    assert all(group["budget"] in (1, 2) for group in groups)
    families = {family["id"]: family for family in directory["families"]}
    assert len(families) == len(directory["families"]) == 120
    assert sum(family["local"] for family in families.values()) == 65
    records, coordinate = {}, set()
    for number, group in enumerate(groups):
        assert len(group["records"]) == len(group["tuple_rules"]) == 1
        record = group["records"][0]
        assert record not in records and record in families
        records[record] = number
        family = families[record]
        assert not family["local"] and [member["record"] for member in family["members"]] == [record]
        assert family["source_pipe"] == group["source"] and family["target_pipe"] == group["target"]
        assert group["sources"] == [group["source"]] and group["targets"] == [group["target"]]
        assert all(0 <= prior < number for prior in group["conflicts"])
        assert len(group["conflicts"]) == len(set(group["conflicts"]))
        if group["budget"] == 2:
            coordinate.add(record)
            assert group["tuple_rules"] == [{"base": 0, "coordinate_count": 2,
                                              "terms": [[1, 1, 0, 2, 1]]}]
            assert family["source_choices"] == family["target_choices"] == []
            assert family["source_cut"] >= 0 and family["target_cut"] >= 0
    assert coordinate == COORDINATE_RECORDS
    lanes = certificate["lane_evidence"]["lanes"]
    assert len(lanes) == sum(group["budget"] for group in groups) == 71
    assert {(lane["group"], lane["local"]) for lane in lanes} == {
        (number, slot) for number, group in enumerate(groups) for slot in range(group["budget"])}
    for index, lane in enumerate(lanes):
        assert all(0 <= prior < index for prior in lane["conflicts"])
        assert all(0 <= later < len(lanes) and later != index for later in lane["successors"])
        assert all(len(lane[key]) == len(set(lane[key])) for key in ("conflicts", "successors"))
    assert len(certificate["certified_candidates"]) == 1
    assert certificate["certified_candidates"][0]["groups"] == groups
    return certificate, families, records


def command_graph(report):
    # This independent existing oracle validates matching, cuts and causal DAGs;
    # it adds no automatic textual SET-to-WAIT edge.
    capacity(report)
    events = report["trace"]["events"]
    pipes = [event["pipe"] for event in events if event["kind"] == "payload"]
    commands, sets, waits, originals = [], {}, {}, {}
    recent = {}
    for event in events:
        if event["kind"] == "payload":
            recent[event["pipe"]] = event
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            assert event.get("record_label") is True and event.get("members"), "missing original record tuple"
            identity = (event["plan"], event["record"], event["source_pipe"], event["target_pipe"],
                        event["source_ordinal"], tuple(event["members"]))
            command["identity"] = identity
            table = sets if event["kind"] == "set" else waits
            table[identity] = 2 * len(pipes) + len(commands)
            if event["kind"] == "set":
                originals[identity] = recent.get(event["source_pipe"])
        commands.append(command)
    return closure_with_commands(pipes, commands, include_commands=True), sets, waits, originals


def inspect_trace(report, certificate, families, records):
    reach, sets, waits, originals = command_graph(report)
    by_group, by_lane = defaultdict(list), defaultdict(list)
    transported, reused = defaultdict(int), defaultdict(int)
    groups = certificate["groups"]
    for identity in sets:
        record, *coordinates = identity[-1]
        assert record in records, "observed notification absent from certificate"
        group_id = records[record]
        group = groups[group_id]
        assert identity[2:4] == (group["source"], group["target"])
        slot = 0
        if record in COORDINATE_RECORDS:
            assert identity[4] == 0 and len(coordinates) == 1
            source = originals[identity]
            assert source and source["coordinates"] == coordinates, "SET lost original source IV"
            assert coordinates[0] >= 0
            slot = coordinates[0] % 2
            transported[record] += 1
        by_group[group_id].append(identity)
        by_lane[(group_id, slot)].append(identity)

    def precedes(left, right):
        return bool(reach[waits[left]] >> sets[right] & 1)

    for (group_id, slot), stream in by_lane.items():
        stream.sort(key=lambda identity: sets[identity])
        for previous, following in zip(stream, stream[1:]):
            assert precedes(previous, following), ("unsafe same-slot reuse", group_id, slot)
            reused[groups[group_id]["records"][0]] += 1
    sharing_checks, successor_checks, lane_sharing_checks = 0, 0, 0
    active_group_pairs, active_lane_edges, active_lane_pairs = set(), set(), set()
    for current, group in enumerate(groups):
        for prior in range(current):
            other = groups[prior]
            if (group["source"], group["target"]) != (other["source"], other["target"]):
                continue
            if prior in group["conflicts"]:
                continue  # A conservative conflict is permission to keep IDs distinct.
            for left in by_group[prior]:
                for right in by_group[current]:
                    assert precedes(left, right) or precedes(right, left), ("unproved parent sharing", prior, current)
                    sharing_checks += 1
                    active_group_pairs.add((prior, current))
    lanes = certificate["lane_evidence"]["lanes"]
    for index, lane in enumerate(lanes):
        group = groups[lane["group"]]
        for prior in range(index):
            earlier = lanes[prior]
            other = groups[earlier["group"]]
            if prior in lane["conflicts"] or (group["source"], group["target"]) != (
                    other["source"], other["target"]):
                continue
            for left in by_lane[(earlier["group"], earlier["local"])]:
                for right in by_lane[(lane["group"], lane["local"])]:
                    assert precedes(left, right) or precedes(right, left), ("unproved lane sharing", prior, index)
                    lane_sharing_checks += 1
                    active_lane_pairs.add((prior, index))
        for target in lane["successors"]:
            later = lanes[target]
            for left in by_lane[(lane["group"], lane["local"])]:
                for right in by_lane[(later["group"], later["local"])]:
                    assert precedes(left, right), ("unproved lane successor", index, target)
                    successor_checks += 1
                    active_lane_edges.add((index, target))
    return {"transported": dict(transported), "same_slot_reuses": dict(reused),
            "parent_sharing_checks": sharing_checks, "lane_successor_checks": successor_checks,
            "lane_sharing_checks": lane_sharing_checks,
            "observed_parent_pairs": sorted(active_group_pairs), "observed_lane_edges": sorted(active_lane_edges),
            "observed_lane_pairs": sorted(active_lane_pairs),
            "unobserved_coordinate_records": sorted(COORDINATE_RECORDS - transported.keys()),
            "cleanup_records": sorted(record for record, family in families.items()
                                      if family["source_choices"] or family["target_choices"])}


def must_refuse(action, label):
    try:
        action()
    except AssertionError:
        return
    raise AssertionError("accepted damaged coordinate evidence: " + label)


def controls(text, report, certificate, families, records):
    for old, new in (("coordinate_count = 2 : i64", "coordinate_count = 3 : i64"),
                     ("array<i64: 1, 1, 0, 2, 1>", "array<i64: 0, 1, 0, 2, 1>")):
        assert old in text
        must_refuse(lambda: metadata(text.replace(old, new)), "coordinate-rule")
    damaged = deepcopy(report)
    events = damaged["trace"]["events"]
    chosen = next(event for event in events if event["kind"] == "set" and
                  event["members"][0] in COORDINATE_RECORDS)
    identity = (chosen["record"], chosen["source_ordinal"], tuple(chosen["members"]))
    changed = 0
    for event in events:
        if event["kind"] not in ("set", "wait"):
            continue
        if (event["record"], event["source_ordinal"], tuple(event["members"])) == identity:
            # Keep pairing and the modulo-two color unchanged: only original
            # source-coordinate transport can detect this forged identity.
            event["members"][1] += 2
            changed += 1
    assert changed == 2
    must_refuse(lambda: inspect_trace(damaged, certificate, families, records), "source-IV-transport")
    damaged = deepcopy(report)
    events = damaged["trace"]["events"]
    events.pop(next(index for index, event in enumerate(events) if event["kind"] == "wait"))
    must_refuse(lambda: inspect_trace(damaged, certificate, families, records), "missing-WAIT")


def trace(tracer, path, bound, output):
    path.write_text(bound)
    command = [tracer, "--structured-trace", str(path)]
    run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
    output.with_suffix(".stdout").write_text(run.stdout)
    output.with_suffix(".stderr").write_text(run.stderr)
    assert run.returncode == 0, (command, run.returncode, run.stderr)
    report = json.loads(run.stdout)
    assert report["accepted"] and not report["trace"]["error"], report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "logical", "coordinate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tracer", required=True)
    args = parser.parse_args()
    executable = shutil.which(args.tracer)
    assert executable, "tracer executable unavailable"
    tracer = str(Path(executable).resolve(strict=True))
    inputs = {name: getattr(args, name).resolve(strict=True) for name in ("source", "logical", "coordinate")}
    assert digest(inputs["source"]) == SOURCE_SHA256
    args.output.mkdir(parents=True, exist_ok=True)
    texts = {name: path.read_text() for name, path in inputs.items()}
    certificate, families, records = metadata(texts["coordinate"])
    helper_dir = Path(__file__).resolve().parent
    pins = {str(path): digest(path) for path in [*inputs.values(), Path(tracer), Path(__file__).resolve(),
            helper_dir / "check_prefill_protocol_capacity.py", helper_dir / "check_fixed_protocol_capacity.py",
            helper_dir / "check_logical_insertion.py"]}
    receipt = {"scope": "observed logical protocol and typed export; no physical assignment/device claim",
               "transport_scope": "Observed tuple equals latest original same-source-pipe payload IV; "
                                  "symbolic family source-cut identity remains producer evidence.",
               "pins": pins, "coordinate_records": sorted(COORDINATE_RECORDS), "cases": []}
    for length in (0, 3, 4, 12):
        for initial in (0, 1):
            folder = args.output / f"n{length}-initial{initial}"
            folder.mkdir(exist_ok=True)
            reports, bindings = {}, {}
            for name, text in texts.items():
                bound, bindings[name] = bind_prefill(text, length, initial)
                reports[name] = trace(tracer, folder / (name + ".bound.pto"), bound, folder / name)
            original, logical, coordinate = (reports[name] for name in ("source", "logical", "coordinate"))
            assert all(event["kind"] == "payload" for event in original["trace"]["events"])
            assert payloads(original) == payloads(logical) == payloads(coordinate), "original payload change"
            assert same_fixed_protocol(logical["trace"], coordinate["trace"]), "logical protocol change"
            checks = inspect_trace(coordinate, certificate, families, records)
            if length == 4 and initial == 1:
                controls(texts["coordinate"], coordinate, certificate, families, records)
                receipt["corruption_controls"] = ["arity", "tuple-term", "source-IV", "missing-WAIT"]
            receipt["cases"].append({"length": length, "initial": initial, "bindings": bindings,
                                     "scope": "structural-only; padding unproved" if length == 0 else
                                              "assumes valid original storage/scalar valuation",
                                     "report_hashes": {name: digest(folder / (name + ".stdout")) for name in texts},
                                     "payloads": len(payloads(original)), **checks})
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    observed = {int(record) for case in receipt["cases"] for record in case["transported"]}
    receipt["unobserved_coordinate_records"] = sorted(COORDINATE_RECORDS - observed)
    reused = {int(record) for case in receipt["cases"] for record in case["same_slot_reuses"]}
    receipt["coordinate_records_without_observed_reuse"] = sorted(COORDINATE_RECORDS - reused)
    observed_edges = {tuple(edge) for case in receipt["cases"] for edge in case["observed_lane_edges"]}
    declared_edges = {(index, target) for index, lane in enumerate(certificate["lane_evidence"]["lanes"])
                      for target in lane["successors"]}
    receipt["unobserved_declared_lane_edges"] = sorted(declared_edges - observed_edges)
    observed_pairs = {tuple(pair) for case in receipt["cases"] for pair in case["observed_parent_pairs"]}
    groups = certificate["groups"]
    declared_pairs = {(prior, current) for current, group in enumerate(groups) for prior in range(current)
                      if prior not in group["conflicts"] and
                      (groups[prior]["source"], groups[prior]["target"]) == (group["source"], group["target"])}
    receipt["unobserved_declared_parent_pairs"] = sorted(declared_pairs - observed_pairs)
    lanes = certificate["lane_evidence"]["lanes"]
    observed_pairs = {tuple(pair) for case in receipt["cases"] for pair in case["observed_lane_pairs"]}
    declared_pairs = {(prior, current) for current, lane in enumerate(lanes) for prior in range(current)
                      if prior not in lane["conflicts"] and
                      (groups[lanes[prior]["group"]]["source"], groups[lanes[prior]["group"]]["target"]) ==
                      (groups[lane["group"]]["source"], groups[lane["group"]]["target"])}
    receipt["unobserved_declared_lane_pairs"] = sorted(declared_pairs - observed_pairs)
    receipt["symbolic_scope"] = "All-input transport, finite domains and unobserved sharing rely on producer proofs."
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(receipt["cases"]), "receipt": str(args.output / "receipt.json"),
                      "unobserved_coordinate_records": receipt["unobserved_coordinate_records"],
                      "coordinate_records_without_observed_reuse":
                          receipt["coordinate_records_without_observed_reuse"]}, sort_keys=True))


if __name__ == "__main__":
    main()
