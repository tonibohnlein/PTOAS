# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Original-SSA exclusive-arm oracle, independent of retained guard labels."""
import itertools
import sys
import tempfile
from pathlib import Path

from ptoas.mlir.ir import Context, DenseI64ArrayAttr, DictAttr, IntegerAttr, Module
from ptoas.mlir.dialects import pto
from check_banked_xor import command_graph, execute, expected, independent_covers, run, rejected_original
from check_sequential_boundaries import rows_hold, guard_holds


def actual_choice(function):
    for view in function.regions[0].blocks[0].operations:
        op = view.operation
        if op.name == "scf.if" and all(len(region.blocks) and any(
                item.operation.name == "scf.for" for item in region.blocks[0].operations)
                for region in op.regions):
            return op
    raise ValueError("original exclusive loop choice absent")


def occurrences(body_sites, n1, n2, outcome):
    selected = n1 if outcome else n2
    body = body_sites[0 if outcome else 1]
    exit_site = 1 + sum(map(len, body_sites))
    return [(0, [])] + [(site, [i]) for i in range(max(0, selected)) for site in body] + (
        [(exit_site, [])] if selected > 0 else [])


def covers(summary, body_sites, n1, n2, outcome):
    result = set()
    events = occurrences(body_sites, n1, n2, outcome)
    for raw in summary["minimum"]:
        piece = DictAttr(raw)
        source = IntegerAttr(piece["source"]).value
        target = IntegerAttr(piece["consumer"]).value
        for a, (site_a, coords_a) in enumerate(events):
            for b, (site_b, coords_b) in enumerate(events):
                if site_a == source and site_b == target and rows_hold(
                        piece, coords_a + coords_b + [n1, n2, int(outcome), 1]):
                    result.add((a, b))
    return result


def ports(summary, body_sites, reference, n1, n2, outcome):
    identities = [0, 1]
    selected = n1 if outcome else n2
    count = max(0, selected) * len(body_sites[0 if outcome else 1])
    for child in range(2):
        present = count and outcome == (child == 0)
        identities += [2, 3, 2 * count, 2 * count + 1] if present else [None] * 4
    identities += [2 * (count + 1), 2 * (count + 1) + 1] if count else [None] * 2
    for raw in summary["port_closure"]:
        piece = DictAttr(raw)
        a = identities[IntegerAttr(piece["source_port"]).value]
        b = identities[IntegerAttr(piece["consumer_port"]).value]
        reachable = a is not None and b is not None and (a == b or b in reference[a])
        assert guard_holds(piece["guard"], [n1, n2, int(outcome), 1]) == reachable, (n1, n2, outcome, a, b)


def main():
    tool, source = sys.argv[1:]
    compiled = run(tool, source, "frontier-synch")
    assert compiled.returncode == 0, compiled.stderr
    existing = run(tool, source, "existing")
    assert existing.returncode == 0, existing.stderr
    shared = lambda text: [line for line in text.splitlines() if line.startswith("Shared")]
    assert shared(existing.stderr) == shared(compiled.stderr)
    source_text = Path(source).read_text(encoding="utf-8")
    with Context() as context:
        pto.register_dialect(context)
        original_module = Module.parse(source_text)
        output_module = Module.parse(compiled.stdout)
        original = list(original_module.body.operations)[0]
        output = list(output_module.body.operations)[0]
        choice = actual_choice(original)
        # Original lexical identities, independent of retained planner metadata.
        body_sites = []
        next_site = 1
        for region in choice.regions:
            loop = next(view.operation for view in region.blocks[0].operations
                        if view.operation.name == "scf.for")
            size = sum(view.operation.name == "pto.txor" for view in loop.regions[0].blocks[0].operations)
            body_sites.append(list(range(next_site, next_site + size)))
            next_site += size
        summary = DictAttr(DictAttr(output.attributes["pto.frontier.analysis"])["single_stream"])
        assert IntegerAttr(summary["static_endpoint_templates"]).value == 8
        assert len(summary["children"]) == 2
        for index, raw in enumerate(summary["children"]):
            child = DictAttr(raw)
            assert str(child["activation_outcome"]) == ("true" if index == 0 else "false")
        direct = str(original.regions[0].blocks[0].arguments[4].type) == "i1"
        bounds = (-2, 0, 1, 2, 3, 5)
        valuations = list(itertools.product(bounds, repeat=2)) + [(1, -(1 << 63)), (-(1 << 63), 1)]
        for n1, n2 in valuations:
            for value in (False, True):
                inputs = {2: n1, 3: n2, 4: 1 if value else -1}
                boolean = value if direct else None
                values = {}
                execute(original, n1, active=boolean, scalar_inputs=inputs, values_out=values)
                outcome = values[choice.operands[0]]
                commands = execute(output, n1, active=boolean, scalar_inputs=inputs)
                reference = expected(original, n1, False, boolean, inputs)
                projected = command_graph(commands)
                assert projected == reference, (n1, n2, value, "modeled closure")
                assert covers(summary, body_sites, n1, n2, outcome) == independent_covers(original, n1, boolean, inputs)
                correlated = expected(original, n1, True, boolean, inputs)
                assert all(row <= projected[i] for i, row in enumerate(correlated))
                ports(summary, body_sites, reference, n1, n2, outcome)
                selected = n1 if outcome else n2
                assert sum(kind == "set" for kind, _ in commands) == (2 if selected > 0 else 0)
                for kind in ("wait", "set", "barrier"):
                    candidates = [i for i, (name, arg) in enumerate(commands) if name == kind and arg != "PIPE_ALL"]
                    if not candidates:
                        continue
                    changed = list(commands)
                    if kind == "set":
                        changed.insert(candidates[0], changed[candidates[0]])
                    else:
                        changed.pop(candidates[0])
                    try:
                        equal = command_graph(changed) == reference
                    except ValueError:
                        equal = False
                    assert not equal, (n1, n2, value, kind)
    with tempfile.TemporaryDirectory(prefix="exclusive-boundaries-") as directory:
        candidate = Path(directory) / "wrong-selected.pto"
        guard = "%gate" if direct else "%active"
        candidate.write_text(source_text.replace(guard + ", %n1, %n2", guard + ", %n2, %n1"), encoding="utf-8")
        rejected_original(tool, candidate, "wrong-selected-arm")
        if len(body_sites[0]) > 1:
            # Break a real lexical/wrap link without changing target-native
            # eligibility: disjoint scratch and output, common read-only input.
            broken = source_text.replace("    %input =", "    %broken_tmp_addr = arith.constant 90112 : i64\n"
                                         "    %broken_out_addr = arith.constant 100352 : i64\n"
                                         "    %broken_tmp = pto.alloc_tile addr = %broken_tmp_addr : !vec\n"
                                         "    %broken_banks = pto.alloc_multi_tile addr = %broken_out_addr : "
                                         "!pto.multi_tile_buf<!vec, count=2>\n    %input =", 1)
            # Match original bank cardinality, not the planner's annotations.
            if "count=3" in source_text:
                broken = broken.replace("%broken_out_addr : !pto.multi_tile_buf<!vec, count=2>",
                                        "%broken_out_addr : !pto.multi_tile_buf<!vec, count=3>")
            bank_type = "!pto.multi_tile_buf<!vec, count=" + ("3" if "count=3" in source_text else "2") + ">"
            payload = "pto.txor ins(%input, %input, %tmp : !vec, !vec, !vec) outs(%output1 : !vec)"
            broken = broken.replace(payload,
                "%broken_output = pto.multi_tile_get %broken_banks[%slot1] : " + bank_type + " -> !vec\n"
                "      pto.txor ins(%input, %input, %broken_tmp : !vec, !vec, !vec) outs(%broken_output : !vec)", 1)
            candidate.write_text(broken, encoding="utf-8")
            rejected_original(tool, candidate, "missing-original-body-chain")
    print("verified actual exclusive contexts, F*/all-kind ports, shared-union closure, "
          "physical coverage and mutations")


if __name__ == "__main__":
    main()
