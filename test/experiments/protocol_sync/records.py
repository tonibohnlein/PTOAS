# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Parse stable diagnostic records; retain raw frontier text and reject lost counts."""

import json
import re


FIELDS = re.compile(r'([\w-]+)=("(?:[^"\\]|\\.)*"|\[[^\[\]]*\]|\S+)')
BLOCK = re.compile(r"^PROTOCOL-SYNC ([\w-]+) function=@(\S+)")
NUMBERED = re.compile(
    r"^\s+(region|phase|storage-family|storage-track|storage-transition|lane-pattern|raw-pair|obligation) #(\d+)")
COLLECTIONS = {
    "region": "regions", "phase": "phases", "storage-family": "families",
    "storage-track": "tracks", "storage-transition": "transitions",
    "lane-pattern": "candidates", "raw-pair": "raw_pairs", "obligation": "obligations",
}


def fields(line):
    """Read scalar/list fields without interpreting diagnostic expressions as code."""
    result = {}
    for key, value in FIELDS.findall(line):
        result[key] = json.loads(value) if value.startswith('"') else value
    return result


def ids(value):
    return [int(number) for number in re.findall(r"#(\d+)", value)]


def new_function(name):
    result = {collection: [] for collection in COLLECTIONS.values()}
    result.update(name=name, accesses=[], unprojected=[], occurrences=[],
                  failures=[], e_rejections=[], direct_rejections=[], lifecycles=[])
    return result


def read_detail(function, current, phase, track, line):
    stripped = line.strip()
    if stripped.startswith("access #"):
        match = re.match(r"access #(\d+) (\S+)", stripped)
        if not match:
            raise ValueError(f"malformed access: {line}")
        function["accesses"].append(dict(fields(line), id=int(match[1]), mode=match[2], phase=phase))
    elif stripped.startswith("occurrence access="):
        function["occurrences"].append(dict(fields(line), track=track))
    elif stripped.startswith("unprojected-access a#"):
        access = int(re.search(r"a#(\d+)", line)[1])
        function["unprojected"].append(dict(fields(line), access=access))
    elif stripped.startswith("rejected reason="):
        function["failures"].append(fields(line))
    elif stripped.startswith("rejection channel="):
        function["e_rejections"].append(fields(line))
    elif stripped.startswith("rejection obligation="):
        function["direct_rejections"].append(fields(line))
    elif stripped.startswith("independent-lifecycle component="):
        function["lifecycles"].append(fields(line))
    elif stripped.startswith("canonical-local "):
        function.setdefault("canonical_local", []).append(fields(line))
    elif stripped.startswith(("projection-audit ", "transition-audit ", "independent-e-differential ")):
        function[stripped.split()[0]] = fields(line)
    elif current is not None and stripped.startswith((
            "source-frontier=", "ready-source=", "release-source=", "completion-cut=")):
        current.setdefault("frontier_lines", []).append(stripped)
    elif current is not None and stripped.startswith((
            "reference-placement=", "target-query=", "generations=")):
        current.update(fields(line))


def validate_function(function):
    counts = function.get("statistics", {}).get("counts", {})
    checks = (("transitions", "storage_transition_frontiers", "storage-tracks"),
              ("candidates", "lane_pattern_candidates", "lane-patterns"),
              ("unprojected", "storage_track_accesses_unprojected", "storage-tracks"))
    for collection, counter, block in checks:
        if block in function.get("blocks", []) and len(function[collection]) != counts.get(counter):
            raise ValueError(f"lost {collection} in {function['name']}: expected {counts.get(counter)}")


def parse_diagnostics(text):
    """Return records grouped by function, preserving separate static occurrences."""
    functions = {}
    active = None
    current = None
    phase = None
    track = None
    for line in text.splitlines():
        if line.startswith("{") and '"kind":"protocol-sync-statistics"' in line:
            record = json.loads(line)
            function = functions.setdefault(record["function"], new_function(record["function"]))
            if "statistics" in function:
                raise ValueError(f"duplicate statistics for {record['function']}")
            function["statistics"] = record
            continue
        match = BLOCK.match(line)
        if match:
            block, name = match.groups()
            active = functions.setdefault(name, new_function(name))
            active.setdefault("blocks", []).append(block)
            current = None
            phase = None
            track = None
            if block == "concrete-verification":
                active["verdict"] = fields(line)
            elif block == "ready-release-plan":
                active["e_plan"] = fields(line)
            continue
        if active is None:
            continue
        match = NUMBERED.match(line)
        if match:
            kind, number = match.groups()
            current = dict(fields(line), id=int(number))
            active[COLLECTIONS[kind]].append(current)
            if kind == "phase":
                phase = int(number)
            if kind == "storage-track":
                track = int(number)
            continue
        read_detail(active, current, phase, track, line)
    for function in functions.values():
        validate_function(function)
    return {"functions": list(functions.values())}
