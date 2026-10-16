# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Qualify the unified finite service through existing physical output routes."""
import itertools
import json
from pathlib import Path
import sys
import tempfile

from check_deferred_family_output import arithmetic, executable, invoke
from check_finite_allocation import physical_check
from check_finite_guarded_analysis import compute, load, validate
from check_guarded_publication import publication_counts


def render(template, body, values=(1, 1, 1, 3)):
    begin = template.index("    // BODY")
    end = template.index("    // END_BODY", begin)
    text = template[:begin] + body + "\n" + template[end:]
    return text.replace("array<i64: 1, 1, 1, 3>", "array<i64: " + ", ".join(map(str, values)) + ">")


def trace(tool, source, values):
    if source.count("test.trace_arguments = array<i64: 1, 1, 1, 3>") != 1:
        raise RuntimeError("missing original finite invocation binding")
    bound = source.replace("test.trace_arguments = array<i64: 1, 1, 1, 3>",
                           "test.trace_existing, test.trace_arguments = array<i64: " +
                           ", ".join(map(str, values)) + ">")
    report = json.loads(invoke([tool, "--structured-trace", "-"], bound).stdout)
    assert report["accepted"] and not report["trace"]["error"], report
    return report["trace"]


def main():
    optimizer, tool, compiler, fixture = sys.argv[1:]
    optimizer, tool, compiler = map(executable, (optimizer, tool, compiler))
    template = Path(fixture).read_text(encoding="utf-8")
    command = [optimizer, "--mlir-disable-threading"]  # Serial semantics matrix; threading has its own gate.
    flags = "gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5"
    allocate = "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"
    late = "%choice = scf.if %g -> i1 { scf.yield %h : i1 } else { scf.yield %g : i1 }"
    branches = "scf.if %choice { " + compute("C") + " } else { " + compute("D") + " }"
    effects = {label: ({"mat"}, {"left"}) for label in ("L", "E")}
    effects.update({label: ({"left", "right"}, {"acc"}) for label in ("C", "D")})
    cases = [
        ("", lambda s, g, h, n: []),
        (load() + load("E"), lambda s, g, h, n: ["L", "E"]),
        (load() + compute() + load("E") + compute("D"), lambda s, g, h, n: ["L", "C", "E", "D"]),
        ("scf.if %s { " + load() + " }" + late + branches,
         lambda s, g, h, n: (["L"] if s else []) + ["C" if g and h else "D"]),
        (load() + late + "scf.if %choice { " + compute() + " }",
         lambda s, g, h, n: ["L"] + (["C"] if g and h else [])),
        ("scf.for %i = %zero to %n step %one { " + load() +
         "scf.if %g { " + compute() + " } }",
         lambda s, g, h, n: (["L"] + (["C"] if g else [])) * n),
    ]
    checked = 0
    for body, labels in cases:
        source = render(template, body)
        logical = invoke([*command, "--pto-frontier-analysis=" + flags, "-"], source).stdout
        saved = invoke([*command, "--pto-frontier-analysis=" + flags + " deferred-output=true", "-"], source).stdout
        assert "pto.logical_set" not in saved and "pto.logical_wait" not in saved
        assert arithmetic(saved) == arithmetic(source), "saved finite recipe materialized scalar control"
        split = invoke([*command, allocate, "-"], saved).stdout
        combined = invoke([*command, "--pto-frontier-analysis=" + flags + " physical-output=true", "-"], source).stdout
        ordinary = invoke([*command, allocate, "-"], logical).stdout
        for physical in (split, combined, ordinary):
            assert "pto.logical_" not in physical and "pto.deferred_family" not in physical
        emitted = invoke([compiler, "--pto-arch=a3", "--pto-level=level3",
                          "--enable-insert-sync=false", "--enable-plan-memory=false", "-"], combined).stdout
        assert "publication(" in emitted and "pto.logical_" not in emitted, "finite C++ emission failed"
        # Save/reparse through separate processes above; neither can borrow the
        # original finite owner, publication callbacks or detached SSA.
        for values in itertools.product((0, 1), (0, 1), (0, 1), (0, 1, 3)):
            logical_trace = trace(tool, logical, values)
            document = {"accepted": True, "trace": logical_trace}
            validate(document, labels(*values), effects)
            publication_counts(document)
            for physical in (split, combined, ordinary):
                physical_check({**document, "allocated": True, "physical": trace(tool, physical, values)},
                               set(range(6)))
            checked += 1
    # Independent publication identities whose consumers are all skipped remain
    # live together until the terminal epilogue. This is a fixed-protocol witness,
    # not a claim that another realization could never use fewer IDs.
    with tempfile.TemporaryDirectory(prefix="finite-terminal-capacity-") as scratch:
        path = Path(scratch) / "input.pto"
        for count in (2, 4, 7):
            definitions, publications, consumers = [], [], []
            for i in range(count):
                definitions += [f"%address{i} = arith.constant {512 * i} : i64",
                                f"%bank{i} = pto.alloc_tile addr = %address{i} : !left"]
                publications.append(load(f"L{i}", f"bank{i}"))
                consumers.append(compute(f"C{i}").replace("%left,", f"%bank{i},"))
            body = "\n".join(definitions + publications + [late, "scf.if %choice {", *consumers, "}"])
            source = render(template, body, (1, 0, 0, 3)).replace(
                "test.trace_arguments =",
                "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>, test.trace_arguments =")
            path.write_text(source, encoding="utf-8")
            report = json.loads(invoke([tool, "--gm-alias=may-not-alias", "--structured-trace", str(path)], "").stdout)
            assert report["accepted"], report
            publication_counts(report)
            events = report["trace"]["events"]
            sets = [e for e in events if e["kind"] == "set"]
            waits = [e for e in events if e["kind"] == "wait"]
            assert len(sets) == count and len(waits) == count, report
            assert all(e["gap"] == count for e in waits), report
            assert max(events.index(e) for e in sets) < min(events.index(e) for e in waits), report
            if count > 6:
                assert not report["allocated"] and report["allocation_unchanged_on_failure"], report
            else:
                physical_check(report, set(range(6)))
    print(f"finite family pipeline: {checked} valuations across saved, combined and ordinary output; "
          "terminal lifetime capacity and unchanged refusal checked")


if __name__ == "__main__":
    main()
