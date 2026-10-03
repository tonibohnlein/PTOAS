# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent finite command-contract oracle for the actual compiled loop."""
import itertools
import sys
import tempfile
from pathlib import Path

from ptoas.mlir.ir import Context, DenseI64ArrayAttr, DictAttr, IntegerAttr, Module
from ptoas.mlir.dialects import pto
from check_banked_xor import command_graph, execute, expected, independent_covers, run, rejected_original


def rows_hold(piece, values):
    for name in ("equalities", "inequalities"):
        rows = list(DenseI64ArrayAttr(piece[name]))
        assert len(rows) % len(values) == 0
        for start in range(0, len(rows), len(values)):
            result = sum(a * b for a, b in zip(rows[start:start + len(values)], values))
            valid = result == 0 if name == "equalities" else result >= 0
            if not valid:
                return False
    return True


def guard_holds(guards, values):
    return any(rows_hold(DictAttr(piece), values) for piece in guards)


def advertised_covers(summary, n1, n2, active):
    occurrences = [(0, [])]
    if active:
        occurrences += [(1, [i]) for i in range(max(0, n1))]
        occurrences += [(2, [i]) for i in range(max(0, n2))]
        if n1 > 0 or n2 > 0:
            occurrences += [(3, [])]
    activated = str(DictAttr(summary["activation_binding"])["origin"]) != '\"none\"'
    result = set()
    for raw in summary["minimum"]:
        piece = DictAttr(raw)
        source = IntegerAttr(piece["source"]).value
        target = IntegerAttr(piece["consumer"]).value
        for a, (site_a, coord_a) in enumerate(occurrences):
            for b, (site_b, coord_b) in enumerate(occurrences):
                if site_a == source and site_b == target and rows_hold(
                        piece, coord_a + coord_b + [n1, n2] + ([int(active)] if activated else []) + [1]):
                    result.add((a, b))
    return result


def check_ports(summary, reference, n1, n2, active):
    # Original dynamic identities, coalesced naturally at n=1. No absent dummy.
    ports = [0, 1]
    count = max(0, n1) if active else 0
    for first, length in ((1, count), (1 + count, max(0, n2) if active else 0)):
        last = first + length - 1
        ports += [2 * first, 2 * first + 1, 2 * last, 2 * last + 1] if length else [None] * 4
    last = 1 + count + (max(0, n2) if active else 0)
    ports += [2 * last, 2 * last + 1] if active and (n1 > 0 or n2 > 0) else [None] * 2
    activated = str(DictAttr(summary["activation_binding"])["origin"]) != '\"none\"'
    for raw in summary["port_closure"]:
        piece = DictAttr(raw)
        a = IntegerAttr(piece["source_port"]).value
        b = IntegerAttr(piece["consumer_port"]).value
        valid = ports[a] is not None and ports[b] is not None
        required = valid and (ports[a] == ports[b] or ports[b] in reference[ports[a]])
        # The parent interface is admitted under original common activation;
        # even the external entry is not an advertised active-context port when inactive.
        required &= active
        values = [n1, n2] + ([int(active)] if activated else []) + [1]
        assert guard_holds(piece["guard"], values) == required, (n1, n2, a, b)


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
        summary = DictAttr(DictAttr(output.attributes["pto.frontier.analysis"])["single_stream"])
        assert list(DenseI64ArrayAttr(summary["parameter_arguments"])) == [2, 3]
        assert IntegerAttr(summary["static_pair_alternatives"]).value == 4
        assert IntegerAttr(summary["static_endpoint_templates"]).value == 8
        assert IntegerAttr(summary["closure_maximum_pieces"]).value <= 8
        activated = len(original.regions[0].blocks[0].arguments) == 5
        bounds = (-2, 0, 1, 2, 3, 5)
        pairs = list(itertools.product(bounds, repeat=2)) + [(-(1 << 63), 1), (1, -(1 << 63))]
        for n1, n2 in pairs:
            for gate in ((-1, 1) if activated else (1,)):
                inputs = {2: n1, 3: n2, 4: gate}
                active = gate > 0
                commands = execute(output, n1, scalar_inputs=inputs)
                reference = expected(original, n1, False, scalar_inputs=inputs)
                projected = command_graph(commands)
                assert projected == reference, (n1, n2, gate, "modeled closure")
                assert advertised_covers(summary, n1, n2, active) == independent_covers(
                    original, n1, scalar_inputs=inputs), (n1, n2, gate, "F* rows")
                correlated = expected(original, n1, True, scalar_inputs=inputs)
                assert all(row <= projected[i] for i, row in enumerate(correlated))
                check_ports(summary, reference, n1, n2, active)
                count = sum(kind == "set" for kind, _ in commands)
                assert count == (2 if active and (n1 > 0 or n2 > 0) else 0)
                if count:
                    for mutation in ("missing-wait", "duplicate-set", "missing-local"):
                        changed = list(commands)
                        kind = {"missing-wait": "wait", "duplicate-set": "set", "missing-local": "barrier"}[mutation]
                        candidates = [i for i, (name, value) in enumerate(changed)
                                      if name == kind and value != "PIPE_ALL"]
                        if not candidates:
                            continue
                        index = candidates[0]
                        if mutation == "duplicate-set":
                            changed.insert(index, changed[index])
                        else:
                            changed.pop(index)
                        try:
                            equal = command_graph(changed) == reference
                        except ValueError:
                            equal = False
                        assert not equal, (n1, n2, mutation)
    # Valid original arithmetic selecting an absent/earlier child must not be
    # relabelled as the supported last-active source merely by an annotation.
    with tempfile.TemporaryDirectory(prefix="sequential-boundaries-") as directory:
        candidate = Path(directory) / "wrong-last.pto"
        candidate.write_text(source_text.replace("%positive2, %n2, %n1", "%positive2, %n1, %n2"), encoding="utf-8")
        rejected_original(tool, candidate, "wrong-last-active")
    print("verified two-bound shared-port queries/F*, union closure, physical coverage, "
          "exclusive IDs, zero/negative/short trips and mutations")


if __name__ == "__main__":
    main()
