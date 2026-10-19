# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise accepted finite children through completion and physical allocation."""
import itertools
import json
from pathlib import Path
import re
import sys

from check_deferred_family_output import executable, invoke
from check_finite_allocation import physical_check
from check_finite_guarded_analysis import required
from check_logical_insertion import closure_with_commands


def replay(checker, text, trips, take):
    text = re.sub(r"test.trace_arguments = array<i64: [^>]+>",
                  f"test.trace_existing, test.trace_arguments = array<i64: {trips}, {take}>", text)
    return json.loads(invoke([checker, "--structured-trace", "-"], text).stdout)


def validate(logical, physical, labels, child_visits):
    events = logical["trace"]["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    assert [event["label"] for event in payloads] == labels
    effects = {"WX": (set(), {"x"}), "WY": (set(), {"y"}),
               "RX": ({"x"}, set()), "RY": ({"y"}, set()), "FINAL": ({"x"}, set())}
    commands = []
    for event in events:
        if event["kind"] == "payload":
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands([event["pipe"] for event in payloads], commands)
    expected = required(payloads, effects)
    assert all((needed & ~(1 << i)) & ~actual[i] == 0 for i, needed in enumerate(expected))
    # Imported children preserve their exact internal payload order. Parent
    # boundary completions may strengthen cross-visit order only.
    for begin, end in child_visits:
        expected_child = required(payloads[begin:end], effects)
        mask = (1 << (2 * (end - begin))) - 1
        for i, needed in enumerate(expected_child):
            assert (actual[2 * begin + i] >> (2 * begin)) & mask == needed & ~(1 << i)
        assert not any(command["kind"] == "barrier" and command["pipe"] == 6
                       and begin < command["gap"] < end for command in commands)
    physical_check({**logical, "allocated": True, "physical": physical["trace"]}, {2, 5})


def separated_children(optimizer, source):
    header = source[:source.index("    scf.for %i")]
    tail = """    scf.while : () -> () {
      scf.condition(%false)
    } do { scf.yield }
    return
  }
}
"""
    first = """    scf.if %take {
      pto.texpands ins(%one : f32) outs(%x : !tile)
      %rx = pto.tgetval ins(%x, %zero : !tile, index) outs : f32
    }
"""
    second = first.replace("%x", "%y").replace("%rx", "%ry")
    for boundary, accepted in (("", False),
                               ("    scf.if %take { pto.barrier <PIPE_ALL> }\n", False),
                               ("    pto.barrier <PIPE_ALL>\n", True)):
        program = header + first + boundary + second + tail
        logical = invoke([optimizer, "--pto-frontier-analysis", "-"], program).stdout
        if accepted:
            assert "eventful_inline_children = 2" in logical
            assert logical.count("pto.logical_set") == 2
        else:
            # The CFG proof does not assume a correlation between two visits of
            # a branch predicate. A bypass path keeps resources pending. Reject
            # provisional imports, then use the existing primitive fallback.
            assert "eventful_inline_children = 0" in logical
            assert "pto.logical_set" not in logical
            assert logical.count("pto.barrier <PIPE_ALL>") == 3 + bool(boundary)
    print("completion resources: no boundary / bypass refused; existing unconditional boundary accepted")


def independent_directions(optimizer, source):
    header = source[:source.index("    scf.for %i")]
    program = header + """
    scf.if %take {
      pto.texpands ins(%one : f32) outs(%x : !tile)
      %rx = pto.tgetval ins(%x, %zero : !tile, index) outs : f32
    }
    scf.if %take {
      pto.tsetval ins(%zero, %one : index, f32) outs(%y : !tile)
      pto.texp ins(%y : !tile) outs(%y : !tile)
    }
    scf.while : () -> () { scf.condition(%false) } do { scf.yield }
    return
  }
}
"""
    logical = invoke([optimizer, "--pto-frontier-analysis", "-"], program).stdout
    assert "eventful_inline_children = 2" in logical
    assert logical.count("pto.logical_set") == logical.count("pto.logical_wait") == 2
    assert logical.count("pto.barrier <PIPE_ALL>") == 1
    physical = invoke([optimizer, "--pto-frontier-analysis=physical-output=true allocation-ids=5", "-"],
                      program).stdout
    assert physical.count("pto.set_flag") == physical.count("pto.wait_flag") == 2
    assert physical.count("pto.barrier <PIPE_ALL>") == 1
    print("completion resources: independent directions share one numeric ID without an extra completion")


def threaded_combined(optimizer, source):
    start = source.index("  func.func")
    end = source.rfind("\n}")
    body = source[start:end]
    program = source[:start] + body + body.replace("@pending_children", "@second_children") + "\n}\n"
    flags = ["--pto-frontier-analysis=physical-output=true allocation-ids=2,5", "-"]
    threaded = invoke([optimizer, *flags], program).stdout
    serial = invoke([optimizer, "--mlir-disable-threading", *flags], program).stdout
    assert threaded == serial and "pto.logical_" not in threaded
    assert threaded.count("pto.completion_protocol") == 2
    print("completion combined: serial/threaded multi-function outputs agree")


def assembly_scaling(optimizer, source):
    start, end = source.index("    scf.for %i"), source.index("    scf.while")
    child = """    scf.if %take {
      pto.texpands ins(%one : f32) outs(%x : !tile)
      %rx = pto.tgetval ins(%x, %zero : !tile, index) outs : f32
    }
"""
    for count in (8, 32, 128):
        program = source[:start] + child * count + source[end:]
        physical = invoke([optimizer, "--pto-frontier-analysis",
                           "--pto-frontier-allocate=eligible-ids=2,5", "-"], program).stdout
        assert physical.count("pto.set_flag") == physical.count("pto.wait_flag") == count
        assert physical.count("pto.barrier <PIPE_ALL>") == count + 1
        assert physical.count("arith.") <= 2 * count + 8
    print("completion assembly: 8/32/128 retained children, linear emitted arithmetic and commands")


def main():
    optimizer, checker, fixture = sys.argv[1:]
    optimizer, checker = map(executable, (optimizer, checker))
    source = Path(fixture).read_text()
    separated_children(optimizer, source)
    independent_directions(optimizer, source)
    assembly_scaling(optimizer, source)
    threaded_combined(optimizer, source)
    tested = 0
    for nested in (False, True):
        program = source
        if nested:
            program = program.replace("      scf.if %take {",
                                      "      scf.for %j = %zero to %n step %one_i {\n      scf.if %take {")
            program = program.replace("    }\n    pto.texpands", "      }\n    }\n    pto.texpands")
        logical = invoke([optimizer, "--mlir-disable-threading", "--pto-frontier-analysis", "-"], program).stdout
        assert "eventful_inline_children = 2" in logical and 'strategy = "regional-palettes"' in logical
        assert logical.count("pto.logical_set") == logical.count("pto.logical_wait") == 3
        physical = invoke([optimizer, "--pto-frontier-allocate=eligible-ids=2,5", "-"], logical).stdout
        assert "pto.logical_" not in physical
        # Reparsed logical output uses the requested pool. Resource failure is
        # distinct from matching closure; no barrier is added to repair it.
        refused = invoke([optimizer, "--pto-frontier-allocate=eligible-ids=2", "-"], logical, succeeds=False)
        assert refused.returncode and not refused.stdout
        combined = invoke([optimizer, "--pto-frontier-analysis=physical-output=true allocation-ids=2,5", "-"],
                          program).stdout
        assert "pto.logical_" not in combined
        refused_combined = invoke(
            [optimizer, "--pto-frontier-analysis=physical-output=true allocation-ids=2", "-"],
            program, succeeds=False)
        assert refused_combined.returncode and not refused_combined.stdout
        for trips, take in itertools.product((0, 1, 3), (0, 1)):
            count = (trips * trips if nested else trips) if take else 0
            labels = ["WX", "WY", "RX", "RY"] * count + ["WX", "RX", "FINAL"]
            children = [(4 * visit, 4 * visit + 4) for visit in range(count)]
            children.append((4 * count, 4 * count + 2))
            logical_trace = replay(checker, logical, trips, take)
            physical_trace = replay(checker, physical, trips, take)
            combined_trace = replay(checker, combined, trips, take)
            validate(logical_trace, physical_trace, labels, children)
            validate(logical_trace, combined_trace, labels, children)
            assert physical_trace["trace"]["events"] == combined_trace["trace"]["events"]
            tested += 1
        # A saved/reparsed module has the same output with pass threading on.
        assert invoke([optimizer, "--pto-frontier-analysis", "-"], program).stdout == logical
    print(f"completion eventful: {tested} pending-child traces, nested/zero/skipped visits, "
          "final-false while and causal reuse")


if __name__ == "__main__":
    main()
