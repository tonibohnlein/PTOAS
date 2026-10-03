# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Finite independent model, original-command and causal flag-rearm oracle."""
from pathlib import Path
import re
import sys
import tempfile
from ptoas.mlir.ir import Context, Module, DictAttr, IntegerAttr, DenseI64ArrayAttr, BoolAttr, StringAttr
from ptoas.mlir.dialects import pto
from check_symbolic_physical import selected_closure
from check_periodic_physical import rearm as symbolic_rearm
from check_banked_xor import (close, command_graph, execute, expected, independent_covers,
                             rejected_original, retained_covers, run, activation_certificate)


def original_model(function, n, correlated, active=None, scalar_inputs=None):
    effects = []
    commands = execute(function, n, correlated, effects, active, scalar_inputs)
    graph = command_graph(commands)
    for source, (reads, writes) in enumerate(effects):
        for target in range(source + 1, len(effects)):
            target_reads, target_writes = effects[target]
            if writes & target_reads or reads & target_writes or writes & target_writes:
                graph[2 * source + 1].add(2 * target)
    return close(graph)


def rearm(commands, n):
    details = {}
    command_graph(commands, details)
    prior_wait = {}
    sets = {}
    waits = {}
    for kind, direction, finish in details["events"]:
        if kind == "set":
            if direction in prior_wait:
                assert finish in details["reach"][prior_wait[direction]], "lexical reuse without causal rearm"
            sets[direction] = sets.get(direction, 0) + 1
        elif kind == "wait":
            prior_wait[direction] = finish
            waits[direction] = waits.get(direction, 0) + 1
    assert sets == waits
    counts = {direction[:2]: count for direction, count in sets.items()}
    if n <= 0:
        assert not counts
    else:
        assert counts[("PIPE_MTE2", "PIPE_V")] == n
        assert counts.get(("PIPE_V", "PIPE_MTE2"), 0) == n - 1
        assert counts[("PIPE_V", "PIPE_MTE3")] == 1




def main():
    tool, source = sys.argv[1:]
    original_text = Path(source).read_text(encoding="utf-8")
    compiled = run(tool, source, "frontier-synch")
    existing = run(tool, source, "existing")
    assert compiled.returncode == 0, compiled.stderr
    assert existing.returncode == 0, existing.stderr
    shared = lambda text: [line for line in text.splitlines() if line.startswith("Shared")]
    assert shared(compiled.stderr) == shared(existing.stderr)
    detailed = run(tool, source, "frontier-synch", " dump-frontier-demands=true report-frontier-costs=true")
    assert detailed.returncode == 0 and detailed.stdout == compiled.stdout, detailed.stderr
    assert "original_barrier_chain" in compiled.stdout
    assert "certified-repeated-ready-release" in compiled.stdout
    banks = int(re.search(r"count=(\d+)", original_text).group(1))
    with Context() as context:
        pto.register_dialect(context)
        original_module = Module.parse(original_text)
        original = list(original_module.body.operations)[0]
        # Keep both modules alive while borrowing their parsed function values.
        output_module = Module.parse(compiled.stdout)
        output = list(output_module.body.operations)[0]
        activation_certificate(original, output)
        chain = DictAttr(DictAttr(DictAttr(output.attributes["pto.frontier.analysis"])[
            "shared_target"])["original_barrier_chain"])
        computes = list(DenseI64ArrayAttr(chain["compute_sources"]))
        assert IntegerAttr(chain["consumer_source"]).value == computes[0]
        assert IntegerAttr(chain["release_producer_source"]).value == computes[-1]
        assert len(chain["lexical_witnesses"]) + 1 == len(computes)
        # Source type determines the Boolean parameter; analysis annotations do not.
        arguments = list(original.regions[0].blocks[0].arguments)
        conditional = any(str(argument.type) == "i1" for argument in arguments)
        computed = not conditional and len(arguments) > 3 and str(arguments[3].type) == "index"
        valuations = [(value, None) for value in (False, True)] if conditional else (
            [(None, {3: value}) for value in (-1, 0, 1)] if computed else [(None, None)])
        for supplied_active, scalar_inputs in valuations:
            for n in sorted({-2, 0, 1, 2, banks - 1, banks, banks + 1, 2 * banks + 1}):
                active = supplied_active
                if computed:
                    values = {}
                    execute(original, n, scalar_inputs=scalar_inputs, values_out=values)
                    common = next(operation for operation in original.regions[0].blocks[0].operations
                                  if operation.operation.name == "scf.if")
                    active = values[common.operands[0]]
                    assert isinstance(active, bool)
                graph = expected(original, n, False, active, scalar_inputs)
                assert original_model(original, n, False, active, scalar_inputs) == graph, \
                    "source barrier not model-redundant"
                assert retained_covers(output, n, active) == independent_covers(original, n, active, scalar_inputs)
                commands = execute(output, n, active=active, scalar_inputs=scalar_inputs)
                assert command_graph(commands) == graph
                physical = original_model(original, n, True, active, scalar_inputs)
                assert all(row <= graph[i] for i, row in enumerate(physical))
                rearm(commands, n if active is not False else 0)
                if active is False:
                    assert not any(kind in ("set", "wait") for kind, _ in commands)
                    escaped = list(commands) + [("wait", ("PIPE_V", "PIPE_MTE3", "EVENT_ID0"))]
                    try:
                        command_graph(escaped)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("inactive escaped acquisition evaded balance oracle")
                if n > 1 and active is not False:
                    missing = list(commands)
                    wait = next(i for i, command in enumerate(missing)
                                if command[0] == "wait" and command[1][:2] == ("PIPE_V", "PIPE_MTE2"))
                    missing.pop(wait)
                    try:
                        equal = command_graph(missing) == graph
                    except ValueError:
                        equal = False
                    assert not equal, "missing release acquisition escaped oracle"
                    early = list(commands)
                    publication = next(i for i, command in enumerate(early)
                                       if command[0] == "set" and command[1][:2] == ("PIPE_V", "PIPE_MTE2"))
                    moved = early.pop(publication)
                    payload = next(i for i, command in enumerate(early) if command == ("payload", "PIPE_V"))
                    early.insert(payload, moved)
                    try:
                        equal = command_graph(early) == graph
                    except ValueError:
                        equal = False
                    assert not equal, "publication before source completion escaped oracle"
    negative_sources = {
            "missing-original-barrier": original_text.replace("      pto.barrier <PIPE_MTE2>\n", ""),
            "wrong-original-pipe": original_text.replace("pto.barrier <PIPE_MTE2>", "pto.barrier <PIPE_V>"),
        }
    payloads = re.findall(r"pto.txor[^\n]+", original_text)
    if len(payloads) > 1:
        # Only final Q's own loaded-input read supplies the captured WAR.
        final = payloads[-1]
        assert ", %input," in final
        negative_sources["missing-last-release"] = original_text.replace(
            final, final.replace(", %input,", ", %mid,"), 1)
        extra = original_text.replace("    %zero =", "    %extra_addr = arith.constant 180224 : i64\n"
                                      "    %extra = pto.alloc_tile addr = %extra_addr : !vec\n    %zero =", 1)
        negative_sources["missing-first-ready"] = extra.replace(
            payloads[0], payloads[0].replace("%input, %input", "%extra, %extra"), 1)
        negative_sources["missing-lexical-link"] = extra.replace(
            payloads[1], payloads[1].replace("%mid, %input, %tmp", "%input, %input, %extra"), 1)
    if "%initial =" in original_text:
        negative_sources["missing-prefix-waw"] = original_text.replace("outs(%initial : !vec)", "outs(%tmp : !vec)")
    if "%active: i1" in original_text or "%gate: index" in original_text:
        negative_sources["independent-exit-guard"] = original_text.replace("scf.if %nonempty {", "scf.if %active {")
        negative_sources["extra-else-payload"] = original_text.replace(
            "    }\n    return", "    } else {\n      pto.tload ins(%gm0 : !pto.partition_tensor_view<16x128xi32>) "
            "outs(%tmp : !vec)\n    }\n    return")
    if "%gate: index" in original_text:
        negative_sources["unsigned-comparison"] = original_text.replace(
            "arith.cmpi sgt, %gate", "arith.cmpi ugt, %gate").replace(
            "arith.cmpi sle, %threshold", "arith.cmpi ule, %threshold")
        negative_sources["unsupported-guard-expression"] = original_text.replace(
            "    scf.if %active {", "    %combined = arith.ori %active, %active : i1\n    scf.if %combined {", 1)
    with tempfile.TemporaryDirectory(prefix="repeated-handoff-") as directory:
        for name, text in negative_sources.items():
            candidate = Path(directory) / (name + ".pto")
            candidate.write_text(text, encoding="utf-8")
            alternative = run(tool, candidate, "frontier-synch")
            if alternative.returncode:
                rejected_original(tool, candidate, name)
                continue
            # These mutations disqualify the specialized saved-boundary route,
            # not the shared SyncIR. A newly supported general plan is legal
            # only when its own selected R and physical protocol check out.
            with Context() as context:
                pto.register_dialect(context)
                source_module = Module.parse(text)
                alternative_module = Module.parse(alternative.stdout)
                source_function = source_module.body.operations[0].operation
                function = alternative_module.body.operations[0].operation
                status = StringAttr(function.attributes["pto.frontier.physical_status"]).value
                assert status == "certified-symbolic-pools", name + " used unqualified specialized allocation"
                metadata = DictAttr(function.attributes["pto.frontier.analysis"])
                for active, scalars in valuations:
                    for n in (0, 1, 2, 3):
                        inputs = {2: n, **(scalars or {})}
                        arguments = source_function.regions[0].blocks[0].arguments
                        for index, argument in enumerate(arguments):
                            if str(argument.type) == "i1":
                                inputs[index] = int(active)
                        occurrences = []
                        execute(source_function, n, active=active, scalar_inputs=inputs,
                                occurrences_out=occurrences)
                        selected = selected_closure(metadata, occurrences, inputs)
                        required = original_model(source_function, n, False, active, inputs)
                        commands = execute(function, n, active=active, scalar_inputs=inputs)
                        assert all(row <= selected[i] for i, row in enumerate(required)), name + " misses storage"
                        assert command_graph(commands) == selected, name + " changes selected payload order"
                        symbolic_rearm(commands)
    print("verified source-barrier equivalence, modeled F*, correlated coverage, zero/short/incomplete "
          "trips, directed causal rearm and source/protocol mutations")


if __name__ == "__main__":
    main()
