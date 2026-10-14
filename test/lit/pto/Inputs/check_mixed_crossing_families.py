# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check mixed regular/guarded production crossings against an unfolded graph."""
import json
from pathlib import Path
import re
import sys
import tempfile
from check_frontier_stage_profile import run as compile_profile, split_report
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native
from check_repeated_region import run


def check(document, outer, inner, guard):
    assert document["accepted"] and not document["trace"]["error"], document
    events = document["trace"]["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    expected = [(0, [])] + [(1, [i]) for i in range(outer)] + [(2, [])]
    if guard:
        expected.append((3, []))
    expected += [(4, [i]) for i in range(inner)] + [(5, [])]
    assert [(item["type"], item["coordinates"]) for item in payloads] == expected
    effect_by_type = [({"mat"}, {"left"}), ({"vec"}, {"vec"}),
                      ({"left", "right"}, {"acc"}), ({"mat"}, {"right"}),
                      ({"vec"}, {"vec"}), ({"left", "right"}, {"acc"})]
    effects = [effect_by_type[kind] for kind, _ in expected]
    pipes = [item["pipe"] for item in payloads]
    edges = native(pipes)
    for source, (reads, writes) in enumerate(effects):
        for target in range(source + 1, len(effects)):
            later_reads, later_writes = effects[target]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * source + 1, 2 * target))
    required = closure(2 * len(effects), edges)[0]
    commands = []
    for event in events:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(payloads), "unexpected regional drain"
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    assert closure_with_commands(pipes, commands) == [row & ~(1 << i) for i, row in enumerate(required)]


def main():
    tool, optimizer, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    count = 0
    mixed = 0
    with tempfile.TemporaryDirectory(prefix="mixed-crossing-families-") as scratch:
        path = Path(scratch) / "case.pto"
        for outer, inner in ((0, 3), (2, 0), (1, 2), (4, 4), (8, 16)):
            for guard in (0, 1):
                current = source.replace("%nend = arith.constant 4", f"%nend = arith.constant {outer}")
                current = current.replace("%mend = arith.constant 4", f"%mend = arith.constant {inner}")
                current = current.replace("array<i64: 1>", f"array<i64: {guard}>")
                compiled = compile_profile(optimizer, current, False, True)
                if compiled.returncode:
                    raise RuntimeError(compiled.stderr)
                _, control, _ = split_report(compiled.stderr)
                mixed += control[0].get("recipe_forms", {}).get("mixed_leaves", 0)
                traced, replacements = re.subn(r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
                    r"\1attributes {test.trace_existing, ", compiled.stdout, count=1)
                assert replacements == 1
                path.write_text(traced)
                check(json.loads(run(tool, "--structured-trace", path)), outer, inner, guard)
                count += 1
    assert mixed > 0, "fixture did not exercise mixed production crossing families"
    print(f"mixed production families: {count} guarded/zero-trip unfolded closures; {mixed} mixed leaves")


if __name__ == "__main__":
    main()
