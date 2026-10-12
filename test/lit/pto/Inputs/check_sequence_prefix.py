# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate retained sequence prefix commands against independent byte conflicts."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
from check_repeated_region import run
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def check(document, p, q, read):
    assert document["accepted"] and not document["trace"]["error"], document
    events = document["trace"]["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    expected = ([0] if p else []) + ([1] if q else []) + ([2] if read else [])
    assert [event["type"] for event in payloads] == expected
    assert all(not event["coordinates"] for event in payloads)
    pipes = [event["pipe"] for event in payloads]
    effects = [({"mat"}, {"left"}) if site != 2 else ({"left", "right"}, {"acc"}) for site in expected]
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            later_reads, later_writes = effects[b]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    required = closure(2 * len(payloads), edges)[0]
    commands = []
    for event in events:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(payloads), "unexpected intermediate drain"
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], document


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="sequence-prefix-") as scratch:
        path = Path(scratch) / "case.pto"
        for p, q, read in itertools.product((0, 1), repeat=3):
            path.write_text(source.replace("array<i64: 1, 1, 1>", f"array<i64: {p}, {q}, {read}>"), encoding="utf-8")
            check(json.loads(run(tool, "--structured-trace", path)), p, q, read)
    print("sequence prefix: eight guarded payload and command closures passed")


if __name__ == "__main__":
    main()
