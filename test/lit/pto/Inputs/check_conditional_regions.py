# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify branch-selected compact regions against the independent event oracle."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_repeated_region import check
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def run(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return json.loads(result.stdout)


def loaded_closure(original, inserted, enabled, trips, prefix_reader):
    """Enumerate original byte hazards; interpret only emitted command causality."""
    assert original["accepted"] and inserted["accepted"], (original, inserted)
    assert not original["trace"]["error"] and not inserted["trace"]["error"], (original, inserted)
    raw = [event for event in original["trace"]["events"] if event["kind"] == "payload"]
    payloads = [event for event in inserted["trace"]["events"] if event["kind"] == "payload"]
    assert payloads == raw, "insertion changed or replayed an original payload/load"
    expected = [("E", [])] + ([("R", [])] if prefix_reader else []) + [("Q", [])]
    if enabled:
        for visit in range(trips):
            for inner in range(3):
                expected.extend([("L", [visit, inner]), ("C", [visit, inner])])
    expected.append(("E", []))
    assert [(event.get("label"), event["coordinates"]) for event in raw] == expected, raw
    effects = []
    for label, coordinates in expected:
        slot = coordinates[-1] % 2 if coordinates else 0
        if label == "Q":
            effects.append(({("flags", 0)}, set()))
        elif label in ("R", "C"):
            effects.append(({("left", slot), ("right", 0)}, {("acc", 0)}))
        else:
            effects.append(({("mat", 0)}, {("left", slot)}))
    pipes = [event["pipe"] for event in raw]
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            later_reads, later_writes = effects[b]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    required = closure(2 * len(effects), edges)[0]
    commands, published, acquired = [], {}, {}
    for event in inserted["trace"]["events"]:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(raw), "nonterminal invocation drain"
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            identity = (event["plan"], event["record"], event["source_pipe"], event["target_pipe"],
                        event["source_ordinal"], tuple(event.get("members", [])))
            command["identity"] = identity
            table = published if event["kind"] == "set" else acquired
            assert identity not in table, "duplicate observed endpoint"
            table[identity] = event["gap"]
        commands.append(command)
    assert published.keys() == acquired.keys(), "unmatched publication/cleanup"
    assert closure_with_commands(pipes, commands) == [row & ~(1 << i) for i, row in enumerate(required)]
    load_position = next(index for index, item in enumerate(expected) if item[0] == "Q")
    return {identity: (gap, acquired[identity], len(raw)) for identity, gap in published.items()
            if gap <= load_position}


def observed_load_source(source):
    source = source.replace("outs(%first : !left)", 'outs(%first : !left) {test.label = "E"}')
    source = source.replace("outs(%left : !left)", 'outs(%left : !left) {test.label = "L"}')
    source = source.replace("ins(%first, %right : !left, !right) outs(%acc : !acc)",
                            'ins(%first, %right : !left, !right) outs(%acc : !acc) {test.label = "R"}')
    source = source.replace("ins(%left, %right : !left, !right) outs(%acc : !acc)",
                            'ins(%left, %right : !left, !right) outs(%acc : !acc) {test.label = "C"}')
    return source.replace("pto.load %flags[%zero] :", 'pto.load %flags[%zero] '
                          '{test.label = "Q", test.trace_boolean_argument = 1 : i64} :')


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    tested = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "conditional.pto"
        for guard in (0, 1):
            for n, m in ((0, 3), (2, 0), (1, 1), (2, 3)):
                candidate = source.replace("array<i64: 1, 2, 3>", f"array<i64: {guard}, {n}, {m}>")
                path.write_text(candidate)
                report = run(tool, "--sequence-analysis", path)
                assert report["prepared"] and not report["error"], report
                check(run(tool, "--structured-trace", path), [n if guard else 0, m])
                tested += 1
        # Swapping the active arm must preserve its occurrence semantics.
        candidate = source.replace("scf.if %g {", "scf.if %g {\n    } else {")
        for guard in (0, 1):
            path.write_text(candidate.replace("array<i64: 1, 2, 3>", f"array<i64: {guard}, 2, 3>"))
            check(run(tool, "--structured-trace", path), [0 if guard else 2, 3])
            tested += 1
        start = source.index("    scf.for %visit")
        end = source.index("    }\n    pto.textract", start)
        body = source[start:end]
        both = source[:end] + "    } else {\n" + body + source[end:]
        for guard in (0, 1):
            path.write_text(both.replace("array<i64: 1, 2, 3>", f"array<i64: {guard}, 2, 3>"))
            document = run(tool, "--structured-trace", path)
            for event in document.get("trace", {}).get("events", []):
                if event["kind"] == "payload":
                    if event["coordinates"]:
                        event["type"] = 1 if event["pipe"] == 3 else 2
                    elif event["type"] != 0:
                        event["type"] = 3
            check(document, [2, 3])
            tested += 1
        # Shared interfaces must remain economical under repeated branch wrapping.
        nested = source
        for depth in range(2, 5):
            nested = nested.replace("    scf.for %visit", "    scf.if %g {\n    scf.for %visit", 1)
            marker = "    }\n    pto.textract"
            nested = nested.replace(marker, "    }\n" + marker, 1)
            path.write_text(nested)
            report = run(tool, "--sequence-analysis", path)
            assert report["prepared"], report
            check(run(tool, "--structured-trace", path), [2, 3])
            print("conditional depth", depth, "expressions", report["expressions"], "emitted", report["emitted"])
            tested += 1
        # Explicit runs inside a selected arm retain their original block cuts,
        # including a nonempty internal cross-pipe demand before a compact child.
        prefix = """    pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%first : !left)
    pto.tmatmul ins(%first, %right : !left, !right) outs(%acc : !acc)
"""
        for guard in (0, 1):
            candidate = source.replace("scf.if %g {", "scf.if %g {\n" + prefix, 1)
            path.write_text(candidate.replace("array<i64: 1, 2, 3>", f"array<i64: {guard}, 2, 3>"))
            report = run(tool, "--structured-trace", path)
            assert report["accepted"] and not report["trace"]["error"], report
            payloads = [event for event in report["trace"]["events"] if event["kind"] == "payload"]
            assert len(payloads) == (16 if guard else 2), payloads
            commands = []
            for event in report["trace"]["events"]:
                if event["kind"] == "payload" or (event["kind"] == "barrier" and event["pipe"] == 6):
                    continue
                command = dict(event)
                if event["kind"] != "barrier":
                    command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                           tuple(event.get("members", [])))
                commands.append(command)
            actual = closure_with_commands([event["pipe"] for event in payloads], commands)
            if guard:
                # Payloads 1 and 2 are the inserted TEXTRACT and TMATMUL:
                # the latter reads the tile written by the former.
                assert actual[3] & (1 << 4), "missing explicit prefix readiness"
        loaded = source.replace("%g: i1,", "%flags: !pto.ptr<i1, gm>, %g: i1,")
        loaded = loaded.replace("test.check_region_scope", "test.materialize_sequence_recipe, test.check_region_scope")
        loaded = loaded.replace("    scf.if %g {",
            "    %predicate = pto.load %flags[%zero] : !pto.ptr<i1, gm> -> i1\n    scf.if %predicate {", 1)
        # Prefix and branch writers share a pipe: the incoming requirement is
        # a consumer-side barrier. Exit handoffs can use the already loaded
        # predicate, so no endpoint needs to replay this load before its cut.
        path.write_text(loaded)
        report = run(tool, "--sequence-analysis", path)
        assert report["prepared"] and report["prepared_scalar_free"] and report["unchanged"], report
        assert report["materialized"] and not report["materialization_error"], report
        # A prefix reader needs publication before the load. Exact early
        # publication plus terminal cleanup now handles the unavailable selective
        # predicate; retain the load and verify both observed values independently.
        unavailable = loaded.replace("    %predicate = pto.load",
            "    pto.tmatmul ins(%first, %right : !left, !right) outs(%acc : !acc)\n"
            "    %predicate = pto.load", 1)
        path.write_text(unavailable)
        report = run(tool, "--sequence-analysis", path)
        assert report["prepared"] and report["prepared_scalar_free"] and report["unchanged"], report
        assert report["materialized"] and not report["materialization_error"] and report["cleanup_families"], report
        early = {}
        for prefix_reader, program in ((False, loaded), (True, unavailable)):
            for trips in (0, 2):
                for enabled in (0, 1):
                    observed = observed_load_source(program).replace(
                        "array<i64: 1, 2, 3>", f"array<i64: {enabled}, {trips}, 3>")
                    path.write_text(observed.replace("test.check_region_scope", "test.trace_existing"))
                    original = run(tool, "--structured-trace", path)
                    path.write_text(observed)
                    inserted = run(tool, "--structured-trace", path)
                    endpoints = loaded_closure(original, inserted, enabled, trips, prefix_reader)
                    if prefix_reader and trips == 2:
                        early[enabled] = endpoints
                    tested += 1
        assert any(identity in early[1] and inactive[1] == inactive[2] and
                   early[1][identity][1] < early[1][identity][2]
                   for identity, inactive in early[0].items()), "missing active/terminal WAIT partition"
    print(f"conditional compact regions: {tested} emitted closures passed")


if __name__ == "__main__":
    main()
