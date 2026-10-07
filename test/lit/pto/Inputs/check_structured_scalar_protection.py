# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare scalar/vector command profiles across structured analysis routes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def main():
    tool, fixture = sys.argv[1:]
    template = Path(fixture).read_text()
    scalar_write = "pto.tsetval ins(%zero, %scalar : index, f32) outs(%tile : !cell)\n"
    scalar_read = "%value = pto.tgetval ins(%tile, %zero : !cell, index) outs : f32\n"
    vector = "pto.tadds ins(%tile, %scalar : !cell, f32) outs(%tile : !cell)\n"
    word = scalar_write + scalar_read + vector + scalar_write

    def loop(iv, body, bound="%n"):
        return f"scf.for %{iv} = %zero to {bound} step %one {{\n{body}}}\n"

    def branch(body):
        return f"scf.if %flag {{\n{body}}} else {{\n{body}}}\n"

    bodies = [word, branch(word), loop("i", word),
              loop("i", branch(word)), loop("i", loop("j", word)),
              scalar_write + loop("i", vector) + scalar_read,
              loop("i", scalar_write + scalar_read),
              loop("i", loop("j", word, "%i"))]
    count = 0
    with tempfile.TemporaryDirectory(prefix="scalar-structure-") as directory:
        path = Path(directory) / "case.pto"
        for n in (0, 1, 2, 3):
            for flag in (0, 1):
                for body in bodies:
                    source = template.replace("// BODY", body).replace("array<i64: 2, 1>",
                                                                         f"array<i64: {n}, {flag}>")
                    path.write_text(source)
                    result = subprocess.run([tool, "--structured-trace", str(path)],
                                            capture_output=True, text=True, check=True)
                    doc = json.loads(result.stdout)
                    assert doc["accepted"], (body, doc)
                    trace = doc["trace"]
                    assert not trace["error"], trace
                    payloads = [e for e in trace["events"] if e["kind"] == "payload"]
                    commands = []
                    for event in trace["events"]:
                        if event["kind"] == "payload":
                            continue
                        command = dict(event)
                        if event["kind"] != "barrier":
                            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                                   tuple(event.get("members", [])))
                        commands.append(command)
                    pipes = [e["pipe"] for e in payloads]
                    # Every vector accesses the whole cell containing the scalar
                    # element. Scalars have read or write effects on that element.
                    edges = native(pipes)
                    for b, target in enumerate(payloads):
                        for a, source in enumerate(payloads[:b]):
                            scalar_a = source["pipe"] == 0
                            scalar_b = target["pipe"] == 0
                            if not (scalar_a and scalar_b):
                                edges.add((2*a+1, 2*b))
                    expected, _ = closure(2*len(pipes), edges)
                    actual = closure_with_commands(pipes, commands)
                    assert actual == [row & ~(1 << i) for i, row in enumerate(expected)], (body, trace)
                    count += 1
    # Select both periodic routes with tile arguments of unknown geometry.
    # These scalar effects are represented by uniform may-alias relationships,
    # not the rotating fragment extractor. Fallback must not hide an omission.
    base = Path(fixture).parent
    with tempfile.TemporaryDirectory(prefix="scalar-residual-") as directory:
        path = Path(directory) / "case.pto"
        for guarded in (False, True):
            name = "sync_guarded_rotating_analysis.pto" if guarded else "sync_rotating_analysis.pto"
            source = (base / name).read_text()
            if not guarded:
                prefix = source[:source.index("  func.func @")]
                function = source.split("  func.func @ring_many", 1)[1].split("\n  func.func", 1)[0]
                source = prefix + "  func.func @ring_many" + function + "\n}\n"
            source = "!scalar_tile = !pto.tile_buf<vec, 4x8xf32>\n" + source
            source = source.replace("(%n: index)", "(%sa: !scalar_tile, %sb: !scalar_tile, %n: index)")
            source = source.replace("(%g: i1,", "(%sa: !scalar_tile, %sb: !scalar_tile, %g: i1,")
            source = source.replace("array<i64: 0>", "array<i64: 0, 0, 3>")
            source = source.replace("array<i64: 9>", "array<i64: 3>")
            source = source.replace("    scf.for", "    %value = arith.constant 1.0 : f32\n    scf.for", 1)
            operations = ("pto.tsetval ins(%zero, %value : index, f32) outs(%sa : !scalar_tile)\n"
                          "%unused = pto.tgetval ins(%sb, %zero : !scalar_tile, index) outs : f32\n")
            marker = "// LOCAL_GUARD" if guarded else "      %slot ="
            source = source.replace(marker, operations + marker, 1)
            path.write_text(source)
            if guarded:
                recognized = subprocess.run([tool, "--recognize", str(path)],
                                            capture_output=True, text=True, check=True)
                assert "recognize guarded-rotating: applicable" in recognized.stdout, recognized.stdout
            mode = "--structured-trace" if guarded else "--rotating-analysis"
            result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            if guarded:
                assert report["accepted"], report
            else:
                assert not report["error"] and report["prepared"], report
                pipes = report["periodic"]["payloads"]
                assert not any(pipes[e["source"]] == pipes[e["target"]] == 0
                               for e in report["periodic"]["generators"]), report
            assert not report["trace"]["error"], report
            assert any(e["kind"] == "payload" and
                       (e.get("pipe") == 0 if guarded else pipes[e["type"]] == 0)
                       for e in report["trace"]["events"]), report
            assert not any(e["kind"] == "barrier" and e["pipe"] == 0
                           for e in report["trace"]["events"]), report
    print(f"structured scalar protection: {count} exact profile comparisons passed")


if __name__ == "__main__":
    main()
