# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check composed phase bodies using independently authored byte conflicts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def invoke(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return result.stdout


def check(document, n, m, repeats=1, bank_stride=32, initial_load=False, active_prefix=False,
          moving_extract=False, external_value=False, moving_stride=32, phase_lengths=(),
          owned_stride=0, owned_probe=0, owned_probes=(), owned_consumers=True, owned_origin=0,
          read_square=False, visit_lower=0, visit_step=1, bank_sequence=(0, 1)):
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    expected = [("entry", [], set(range(2048, 2052)), set())] if external_value else []
    for outer in range(repeats):
        prefix = [outer] if repeats != 1 else []
        for visit in range(max(0, min(n, m) if active_prefix else n)):
            # Distinct physical bank bytes, with a nonzero allocation origin.
            coordinate = visit_lower + visit_step * visit
            base = 512 + bank_stride * bank_sequence[visit % len(bank_sequence)]
            cell = set(range(base, base + 4))
            tile = set(range(base, base + 32))
            if initial_load:
                gm_base = 1000000000 + 65536 * 4 * visit
                expected.append(("load", prefix + [coordinate], set(range(gm_base, gm_base + 32)), tile))
            expected.append(("prologue", prefix + [coordinate], cell, set()))
            inner_length = phase_lengths[visit % len(phase_lengths)] if phase_lengths else m
            for inner in range(max(0, inner_length)):
                if moving_extract:
                    source_begin = 4096 + inner * moving_stride
                    expected.append(("extract", prefix + [coordinate, inner],
                                     set(range(source_begin, source_begin + 32)), tile))
                expected.append(("compute", prefix + [coordinate, inner], tile, tile))
                read_begin = base + 4 * inner * inner if read_square else base
                expected.append(("read", prefix + [coordinate, inner], set(range(read_begin, read_begin + 4)), set()))
            expected.append(("epilogue", prefix + [coordinate], set(), cell))
            if owned_stride:
                owned_base = 1000000000 + 65536 * 4 + owned_origin + visit * owned_stride
                expected.append(("owned", prefix + [coordinate], tile, set(range(owned_base, owned_base + 32))))
            if moving_extract:
                expected.append(("sourcewrite", prefix + [coordinate], set(), set(range(4096, 4100))))
    if owned_stride and owned_consumers:
        for consumer, offset in enumerate(owned_probes or (owned_probe,)):
            probe = 1000000000 + 65536 * 4 + offset
            output = 4096 + 32 * consumer
            label = "consume" if consumer == 0 else f"consume{consumer}"
            expected.append((label, [], set(range(probe, probe + 32)), set(range(output, output + 32))))
    events = trace["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    assert [(p["label"], p["coordinates"]) for p in payloads] == [x[:2] for x in expected]
    pipes = [p["pipe"] for p in payloads]
    edges = native(pipes)
    for source, (_, _, reads, writes) in enumerate(expected):
        for target in range(source + 1, len(expected)):
            # Same-scalar storage hazards are protected by hardware. Keep
            # scalar SSA prerequisites below; they are a separate native rule.
            if pipes[source] == 0 and pipes[target] == 0:
                continue
            next_reads, next_writes = expected[target][2:]
            if writes & (next_reads | next_writes) or reads & next_writes:
                edges.add((2 * source + 1, 2 * target))
    # Native scalar prerequisites: every consumer of the prologue scalar must
    # wait for its producer; this also holds when the inner loop has zero trips.
    first = None
    for target, (label, _, _, _) in enumerate(expected):
        if label == "prologue":
            first = target
        elif label in ("compute", "epilogue", "sourcewrite"):
            producer = 0 if external_value and label == "compute" else first
            edges.add((2 * producer + 1, 2 * target))
    _, covers = closure(2 * len(payloads), edges)
    for source, target in covers:
        if source % 2 and not target % 2 and pipes[source // 2] == pipes[target // 2]:
            edges.update((2 * previous + 1, target) for previous in range(target // 2)
                         if pipes[previous] == pipes[target // 2])
    required = closure(2 * len(payloads), edges)[0]
    commands = []
    for event in events:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(payloads)
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    # Native scalar completion edges have no command; add their transitive
    # consequences to the interpreted synchronization/native graph as well.
    for source, target in edges:
        source_label, target_label = expected[source // 2][0], expected[target // 2][0]
        entry_edge = external_value and source_label == "entry" and target_label == "compute"
        local_edge = source_label == "prologue" and target_label in ("compute", "epilogue", "sourcewrite") and \
            not (external_value and target_label == "compute") and \
            expected[source // 2][1] == expected[target // 2][1][:len(expected[source // 2][1])]
        if source % 2 and target % 2 == 0 and (entry_edge or local_edge):
            actual[source] |= 1 << target
    actual_edges = {(i, j) for i, row in enumerate(actual) for j in range(len(actual)) if row & (1 << j)}
    actual = closure(len(actual), actual_edges)[0]
    assert [row & ~(1 << i) for i, row in enumerate(actual)] == \
        [row & ~(1 << i) for i, row in enumerate(required)], (n, m, document)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="composed-phases-") as scratch:
        path = Path(scratch) / "case.pto"
        for n, m in ((0, 0), (0, 3), (1, 0), (1, 2), (2, 1), (3, 0), (3, 2), (5, 3)):
            path.write_text(source.replace("array<i64: 3, 2>", f"array<i64: {n}, {m}>"))
            check(json.loads(invoke(tool, "--structured-trace", path)), n, m)
        compute = ('        pto.tadds ins(%selected, %value : !cell, f32) '
                   'outs(%selected : !cell) {test.label = "compute"}')
        epilogue = '      pto.tsetval ins(%zero, %value : index, f32) outs(%selected : !cell) {test.label = "epilogue"}'
        sliced = source.replace(compute,
            '        %first_outer = arith.cmpi eq, %visit, %zero : index\n'
            '        %first_inner = arith.cmpi eq, %i, %zero : index\n'
            '        %initial = arith.andi %first_outer, %first_inner : i1\n'
            '        scf.if %initial {\n' + compute + '\n        } else {\n' + compute + '\n        }')
        sliced = sliced.replace(epilogue,
            '      %next = arith.addi %visit, %one : index\n'
            '      %last = arith.cmpi eq, %next, %n : index\n'
            '      scf.if %last {\n' + epilogue + '\n      } else {\n' + epilogue + '\n      }')
        for n, m in ((0, 2), (1, 0), (1, 2), (2, 2), (3, 2), (5, 1)):
            path.write_text(sliced.replace("array<i64: 3, 2>", f"array<i64: {n}, {m}>"))
            check(json.loads(invoke(tool, "--structured-trace", path)), n, m)
        sliced_report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert sliced_report["repeated_regions"] > 0 and sliced_report["phase_descriptions"] > 0, sliced_report
        # An outer conditional encloses a complete compact child and its
        # explicit siblings. Select the arm before dispatching the child, rather
        # than sending the entire nest to the arithmetic fallback. The false
        # suffix is an empty body, including all visits when m is zero.
        conditional = source.replace("      %value = pto.tgetval",
            "      %active = arith.cmpi slt, %visit, %m : index\n"
            "      scf.if %active {\n      %value = pto.tgetval")
        conditional = conditional.replace(epilogue, epilogue + "\n      }")
        for n, m in ((0, 2), (3, 0), (1, 2), (3, 1), (5, 3)):
            path.write_text(conditional.replace("array<i64: 3, 2>", f"array<i64: {n}, {m}>"))
            check(json.loads(invoke(tool, "--structured-trace", path)), n, m, active_prefix=True)
        conditional_report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert conditional_report["repeated_regions"] > 0, conditional_report
        assert conditional_report["phase_descriptions"] > 0, conditional_report
        # Finite inner visits carry genuinely moving partial source ranges.
        # The two source subregions must remain distinct; overwriting the first
        # one after the body exposes their different WAR boundaries.
        moving = sliced.replace("!ring =", "!large = !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16, "
                                "v_row=1, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>\n!ring =")
        moving = moving.replace("    %ring = pto.alloc_multi_tile",
            "    %eight = arith.constant 8 : index\n"
            "    %source_address = arith.constant 4096 : i64\n"
            "    %source = pto.alloc_tile addr = %source_address : !large\n"
            "    %ring = pto.alloc_multi_tile")
        moving = moving.replace("to %m step", "to %two step")
        moving = moving.replace("        %first_outer =",
            "        %column = arith.muli %i, %eight : index\n"
            "        pto.textract ins(%source, %zero, %column : !large, index, index) "
            'outs(%selected : !cell) {test.label = "extract"}\n'
            "        %first_outer =")
        moving = moving.replace("    }\n    return",
            '      pto.tsetval ins(%zero, %value : index, f32) outs(%source : !large) '
            '{test.label = "sourcewrite"}\n    }\n    return')
        for n in (0, 1, 2, 5):
            path.write_text(moving.replace("array<i64: 3, 2>", f"array<i64: {n}, 2>"))
            check(json.loads(invoke(tool, "--structured-trace", path)), n, 2, moving_extract=True)
        external = moving.replace("    scf.for %visit",
            "    %entry_address = arith.constant 2048 : i64\n"
            "    %entry_buffer = pto.alloc_tile addr = %entry_address : !cell\n"
            '    %entry_value = pto.tgetval ins(%entry_buffer, %zero : !cell, index) outs : f32 '
            '{test.label = "entry"}\n    scf.for %visit')
        external = external.replace("pto.tadds ins(%selected, %value", "pto.tadds ins(%selected, %entry_value")
        path.write_text(external)
        check(json.loads(invoke(tool, "--structured-trace", path)), 3, 2,
              moving_extract=True, external_value=True)
        moving_report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert moving_report["repeated_regions"] > 0 and moving_report["prepared"], moving_report
        # The nonlinear spelling has the same concrete byte footprints for
        # these two visits: i*i is 0 and 1 for i in [0,2). Check the independently
        # specified ordering without requiring a particular accepted backend.
        finite = external.replace("        %column = arith.muli %i, %eight : index",
            "        %square = arith.muli %i, %i : index\n"
            "        %column = arith.muli %square, %eight : index")
        # Also cover the unsliced inner body, retaining only the outer boundary
        # conditional. Recognition may exploit the supplied access model.
        guard_begin = finite.index("        %first_outer =")
        guard_end = finite.index("        %unused =", guard_begin)
        finite = finite[:guard_begin] + compute.replace("%value", "%entry_value") + "\n" + finite[guard_end:]
        for n in (0, 1, 2, 5):
            path.write_text(finite.replace("array<i64: 3, 2>", f"array<i64: {n}, 2>"))
            check(json.loads(invoke(tool, "--structured-trace", path)), n, 2,
                  moving_extract=True, external_value=True)
        finite_report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert finite_report["repeated_regions"] > 0 and finite_report["prepared"], finite_report
        # The shared scalar normalizer also accepts a commuted low-bit mask;
        # physical banks and all ordering requirements are unchanged.
        masked = source.replace("arith.remui %visit, %two", "arith.andi %one, %visit")
        path.write_text(masked)
        check(json.loads(invoke(tool, "--structured-trace", path)), 3, 2)
        # Large physical spacing is a constant after storage normalization.
        # It must not count as a variable arithmetic coefficient or enumerate
        # the unused gap between the two 32-byte accessed tiles.
        spaced = source.replace(
            "      %selected = pto.multi_tile_get %ring[%slot] : !ring -> !cell",
            "      %stride = arith.constant 65536 : index\n"
            "      %offset = arith.muli %slot, %stride : index\n"
            "      %offset64 = arith.index_cast %offset : index to i64\n"
            "      %address = arith.addi %base, %offset64 : i64\n"
            "      %selected = pto.alloc_tile addr = %address : !cell")
        path.write_text(spaced)
        check(json.loads(invoke(tool, "--structured-trace", path)), 3, 2, bank_stride=65536)
        spaced_report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert spaced_report["repeated_regions"] > 0, spaced_report
        # The same phase provider handles a read-only GM stream on an explicit
        # sibling. Its evolving address is irrelevant only because the shared
        # input proves that no writer can touch that storage, including re-entry.
        streamed = source.replace("%m: index)", "%m: index, %gm: !pto.ptr<f32>)")
        streamed = streamed.replace("    %ring = pto.alloc_multi_tile",
            "    %eight = arith.constant 8 : index\n"
            "    %columns = arith.constant 1073741824 : index\n"
            "    %stride = arith.constant 65536 : index\n"
            "    %stream = pto.make_tensor_view %gm, shape = [%one, %columns], "
            "strides = [%columns, %one] : !pto.tensor_view<?x?xf32>\n"
            "    %ring = pto.alloc_multi_tile")
        streamed = streamed.replace("      %value = pto.tgetval",
            "      %column = arith.muli %visit, %stride : index\n"
            "      %part = pto.partition_view %stream, offsets = [%zero, %column], "
            "sizes = [%one, %eight] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x8xf32>\n"
            "      pto.tload ins(%part : !pto.partition_tensor_view<1x8xf32>) "
            'outs(%selected : !cell) {test.label = "load"}\n'
            "      %value = pto.tgetval")
        path.write_text(streamed)
        check(json.loads(invoke(tool, "--structured-trace", path)), 3, 2, initial_load=True)
        stream_report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert stream_report["repeated_regions"] > 0, stream_report
        # A compact SSA DAG must stay compact during phase normalization.
        # The repeated additions have exponentially many unfolded paths; the
        # final subtraction cancels them and preserves the original bank.
        shared = ["      %phase = arith.remui %visit, %two : index"]
        previous = "%phase"
        for level in range(36):
            value = f"%shared{level}"
            shared.append(f"      {value} = arith.addi {previous}, {previous} : index")
            previous = value
        shared += [f"      %cancel = arith.subi {previous}, {previous} : index",
                   "      %slot = arith.addi %phase, %cancel : index"]
        dag = source.replace("      %slot = arith.remui %visit, %two : index", "\n".join(shared))
        path.write_text(dag)
        check(json.loads(invoke(tool, "--structured-trace", path)), 3, 2)
        path.write_text(source)
        report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert not report["error"] and report["prepared"] and report["numeric_visits"] == 0, report
        assert report["repeated_regions"] > 0, report
        small = invoke(tool, "--insert-logical-library", path)
        path.write_text(source.replace("array<i64: 3, 2>", "array<i64: 1000000000, 1000000001>"))
        large = invoke(tool, "--insert-logical-library", path)
        assert small == large.replace("1000000000, 1000000001", "3, 2")
        # Recognition diagnostics can name scf.for in string attributes.
        assert sum(line.lstrip().startswith("scf.for ") for line in small.splitlines()) == 2
        triple = source.replace("    scf.for %visit",
                                "    scf.for %outer = %zero to %two step %one {\n    scf.for %visit")
        triple = triple.replace("    return", "    }\n    return")
        path.write_text(triple)
        check(json.loads(invoke(tool, "--structured-trace", path)), 3, 2, repeats=2)
        print("composed phases: byte closures, zero trips, partial periods, scalar inputs and three levels passed")


if __name__ == "__main__":
    main()
