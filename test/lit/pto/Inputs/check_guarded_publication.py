# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check shared publication/terminal cleanup across finite and sequence routes."""
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from check_finite_guarded_analysis import compute, load, validate
from check_finite_allocation import physical_check
from check_frontier_public_pipeline import run, unchanged_failure


def invoke(tool, path, mode="--structured-trace", policy="may-not-alias"):
    result = subprocess.run([tool, "--gm-alias=" + policy, mode, str(path)],
                            capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout if mode == "--insert-logical-library" else json.loads(result.stdout)


def reject_without_visit_filter(opt, path, source, reason):
    path.write_text(source)
    prepared = subprocess.run([opt, "--mlir-disable-threading", "--pto-frontier-analysis", str(path)],
                              capture_output=True, text=True, timeout=90, check=False)
    assert prepared.returncode == 0, prepared.stderr
    logical = prepared.stdout
    induction = re.search(r"scf.for (%\w+) =", logical).group(1)
    iv = re.escape(induction)
    pattern = r"(%\w+) = arith.cmpi eq, (?:" + iv + r", %\w+|%\w+, " + iv + r") : index"
    broken, count = re.subn(pattern, r"\1 = arith.constant true", logical)
    assert count, logical
    path.write_text(broken)
    rejected = subprocess.run([opt, "--mlir-disable-threading",
        "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5", str(path)],
        capture_output=True, text=True, timeout=90, check=False)
    assert rejected.returncode != 0 and reason in rejected.stderr, rejected.stderr


def main():
    tool, opt, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    begin = source.index("    // BODY")
    end = source.index("    // END_BODY", begin)
    late = "%choice = scf.if %g -> i1 { scf.yield %h : i1 } else { scf.yield %g : i1 }"
    consumer = "scf.if %choice { " + compute() + " }"
    loop = ("scf.for %i = %zero to %n step %one { " +
            compute("U").replace("%left,", "%second,").replace("outs(%acc", "outs(%aux") + " }")
    effects = {x: ({"mat"}, {"left"}) for x in ("L", "E")}
    effects.update({"U": ({"second", "right"}, {"aux"}),
                    "C": ({"left", "right"}, {"acc"}),
                    "D": ({"left", "right"}, {"acc"})})

    def render(body, values=(1, 1, 1, 3), ids=(0, 1, 2, 3, 4, 5)):
        text = source[:begin] + body + "\n" + source[end:]
        text = text.replace("array<i64: 1, 1, 1, 3>",
                            "array<i64: " + ", ".join(map(str, values)) + ">")
        return text.replace("test.trace_arguments =",
                            "test.eligible_ids = array<i64: " + ", ".join(map(str, ids)) + ">, test.trace_arguments =")

    sources = [(load(), lambda s, g: True),
               ("scf.if %s { " + load() + " }", lambda s, g: bool(s)),
               ("scf.if %s { } else { " + load() + " }", lambda s, g: not s),
               ("scf.if %s { scf.if %g { " + load() + " } }", lambda s, g: bool(s and g))]
    checked = 0
    with tempfile.TemporaryDirectory(prefix="guarded-publication-") as scratch:
        path = Path(scratch) / "input.pto"
        for (prefix, present), expanded, policy in itertools.product(
                sources, (False, True), ("may-not-alias", "may-alias")):
            body = "\n".join([prefix, loop if expanded else "", late, consumer, load("E")])
            # A runtime loop forces the crossing adapter; the finite alternative
            # exercises the same publication policy with a different producer.
            path.write_text(render(body))
            report = invoke(tool, path, "--sequence-analysis", policy)
            assert not report["error"] and report["prepared"] and report["unchanged"], report
            assert report["queries_available"], report
            if expanded:
                assert report["crossing_preparation"] == 0, report
            for s, g, h, n in itertools.product((0, 1), (0, 1), (0, 1), (0, 1, 3)):
                path.write_text(render(body, (s, g, h, n)))
                report = invoke(tool, path, policy=policy)
                labels = (["L"] if present(s, g) else []) + (["U"] * n if expanded else [])
                labels += (["C"] if g and h else []) + ["E"]
                validate(report, labels, effects)
                physical_check(report, set(range(6)))
                # Surplus notification is consumed after every original payload,
                # even when the consumer is skipped or the intervening loop empty.
                if present(s, g) and not (g and h):
                    waits = [e for e in report["trace"]["events"] if e["kind"] == "wait"]
                    assert any(e["gap"] == len(labels) for e in waits), report
                checked += 1
        # A selected first consumer is a singleton even when its static cut
        # executes repeatedly. Every enclosing loop must have a fixed point.
        selected_ir = None
        for depth, step, policy in itertools.product((1, 2), (1, 2), ("may-not-alias", "may-alias")):
            stride = "%one" if step == 1 else "%two"
            repeated = "scf.for %i = %zero to %n step " + stride + " { " + consumer + " }"
            if depth == 2:
                repeated = "scf.for %j = %zero to %n step " + stride + " { " + repeated + " }"
            body = "\n".join(["%two = arith.constant 2 : index", load(), late, repeated, load("E")])
            for g, h, n in itertools.product((0, 1), (0, 1), (0, 1, 3)):
                path.write_text(render(body, (1, g, h, n)))
                report = invoke(tool, path, policy=policy)
                count = len(range(0, n, step)) ** depth if g and h else 0
                validate(report, ["L"] + ["C"] * count + ["E"], effects)
                physical_check(report, set(range(6)))
                checked += 1
            if depth == 1 and step == 1 and policy == "may-not-alias":
                selected_ir = path.read_text()
        # Serialized terminal alternatives are checked again against actual
        # guards. Removing visit selection must not admit repeated consumers.
        reject_without_visit_filter(opt, path, selected_ir, "endpoint alternatives")
        # Two future guards can leave both publications outstanding. A consumer
        # bound is not a lifetime bound when its WAIT may be skipped.
        second = "%next = scf.if %h -> i1 { scf.yield %g : i1 } else { scf.yield %h : i1 }"
        body = "\n".join([load(), loop, late, consumer, load("E"), second,
                           "scf.if %next { " + compute("D") + " }"])
        for g, h in itertools.product((0, 1), repeat=2):
            path.write_text(render(body, (1, g, h, 3), (0,)))
            report = invoke(tool, path)
            assert report["accepted"] and not report["allocated"], report
            assert report["allocation_unchanged_on_failure"], report
            path.write_text(render(body, (1, g, h, 3), (0, 1)))
            report = invoke(tool, path)
            validate(report, ["L"] + ["U"] * 3 + (["C"] if g and h else []) +
                     ["E"] + (["D"] if g and h else []), effects)
            physical_check(report, {0, 1})
        # The last source of a repetition is also one selected occurrence.
        # This needs no per-visit history: the exact presence and final ordinal
        # are invariant and the consumer's opaque outcome remains at return.
        body = "\n".join(["scf.for %i = %zero to %n step %one { " + load() + " }", late, consumer])
        for g, h, n in itertools.product((0, 1), (0, 1), (0, 1, 3)):
            path.write_text(render(body, (1, g, h, n)))
            report = invoke(tool, path)
            validate(report, ["L"] * n + (["C"] if g and h else []), effects)
            physical_check(report, set(range(6)))
            checked += 1
        source_ir = path.read_text()
        reject_without_visit_filter(opt, path, source_ir, "one selected publication")
        # Original coordinates can start below zero and cross it with a
        # non-unit step. The selected consumer/source still executes once;
        # unsigned distance division must recover its original visit ordinal.
        for source_loop, step, policy in itertools.product(
                (False, True), (1, 2), ("may-not-alias", "may-alias")):
            stride = "%one" if step == 1 else "%two"
            repeated = "scf.for %i = %lower to %upper step " + stride + " { "
            # Supply the existing counted-domain proof a bounded upper value;
            # unrestricted signed %n could exceed its representable trip count.
            body = ["%lower = arith.constant -3 : index", "%two = arith.constant 2 : index",
                    "%limit = arith.constant 3 : index", "%upper = arith.minsi %n, %limit : index"]
            if source_loop:
                body += [repeated + load() + " }", late, consumer]
            else:
                body += [load(), late, repeated + consumer + " }", load("E")]
            for g, h, n in itertools.product((0, 1), (0, 1), (-4, -3, -2, 0, 1, 3)):
                path.write_text(render("\n".join(body), (1, g, h, n)))
                report = invoke(tool, path, policy=policy)
                visits = len(range(-3, n, step))
                if source_loop:
                    labels = ["L"] * visits + (["C"] if g and h else [])
                else:
                    labels = ["L"] + ["C"] * (visits if g and h else 0) + ["E"]
                validate(report, labels, effects)
                physical_check(report, set(range(6)))
                checked += 1
        # A late condition private to an arm cannot be replayed in the epilogue.
        negatives = ["scf.if %s { " + "\n".join([load(), loop, late, consumer]) + " }"]
        for body in negatives:
            path.write_text(render(body))
            report = invoke(tool, path, "--sequence-analysis")
            assert not report["error"] and not report["prepared"], report
            assert report["queries_available"] and report["unchanged"], report
        # Recovered source presence must mask a condition from an inactive arm.
        # The deliberately overflowing original scalar is never evaluated when
        # %s is false; replayed poison must not reach a cleanup guard either.
        prefix = ("scf.if %s { %sum = arith.addi %x, %one64 overflow<nsw> : i64\n"
                  "%nested = arith.cmpi slt, %sum, %one64 : i64\n"
                  "scf.if %nested { " + load() + " } }")
        body = "\n".join(["%one64 = arith.constant 1 : i64", prefix, loop, late, consumer, load("E")])
        text = render(body, (0, 0, 0, 3)).replace("%n: index)", "%n: index, %x: i64)")
        text = text.replace("array<i64: 0, 0, 0, 3>", "array<i64: 0, 0, 0, 3, 9223372036854775807>")
        path.write_text(text)
        emitted = invoke(tool, path, "--insert-logical-library")
        assert re.search(r"arith.select %arg0, %\w+, %false", emitted), emitted
        # Public physical-context projection transfers every endpoint-choice
        # proof with its original cuts and guards. Cleanup at the function
        # epilogue acquires a context-only section wrapper during insertion.
        for repeated, policy in itertools.product((False, True), ("may-not-alias", "may-alias")):
            target = "scf.for %i = %zero to %n step %one { " + consumer + " }" if repeated else consumer
            body = "\n".join(["pto.section.cube { scf.if %s { " + load() + " } }", late,
                               "pto.section.cube { " + target + "\n" + load("E") + " }"])
            program = render(body)
            flags = ["--pto-frontier-analysis=gm-alias=" + policy]
            logical = run(opt, program, flags).stdout
            assert "pto.sync_context_plans" in logical and "target_choices = array<i64: -1," in logical, logical
            physical = run(opt, logical, ["--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"]).stdout
            assert "pto.logical_" not in physical and "pto.set_flag" in physical and "pto.wait_flag" in physical
            # Branch-private outcomes remain unavailable after projection.
            private = "scf.if %s {\n" + body + "\n}"
            unchanged_failure(opt, render(private), flags)
        # Vector sections use the same transferred proof and terminal wrapper,
        # including the A5 context-only wrapper's emission semantics.
        vector_load = "pto.tload ins(%part : !pto.partition_tensor_view<32xf32>) outs(%vec : !vec)"
        vector_compute = "pto.tabs ins(%vec : !vec) outs(%vec : !vec)"
        prelude = ("%size = arith.constant 32 : index\n"
                   "%view = pto.make_tensor_view %ptr, shape = [%size], strides = [%one] : !pto.tensor_view<32xf32>\n"
                   "%part = pto.partition_view %view, offsets = [%zero], sizes = [%size] : !pto.tensor_view<32xf32>\n"
                   "%vec = pto.alloc_tile addr = %base : !vec\n")
        for arch, repeated, policy in itertools.product(
                ("a3", "a5"), (False, True), ("may-not-alias", "may-alias")):
            target = "scf.if %choice { " + vector_compute + " }"
            if repeated:
                target = "scf.for %i = %zero to %n step %one { " + target + " }"
            body = "\n".join(["pto.section.vector { scf.if %s { " + vector_load + " } }", late,
                               "pto.section.vector { " + target + "\n" + vector_load + " }"])
            program = render(prelude + body).replace('%n: index)', '%n: index, %ptr: !pto.ptr<f32, gm>)')
            program = program.replace('module attributes', '!vec = !pto.tile_buf<vec, 1x32xf32>\nmodule attributes', 1)
            program = program.replace('pto.target_arch = "a3"', 'pto.target_arch = "' + arch + '"')
            logical = run(opt, program, ["--pto-frontier-analysis=gm-alias=" + policy]).stdout
            assert "pto.sync_context_plans" in logical and "target_choices = array<i64: -1," in logical
            physical = run(opt, logical, ["--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"]).stdout
            assert "pto.logical_" not in physical and "pto.set_flag" in physical and "pto.wait_flag" in physical
    print(f"shared publication: {checked} independent logical/physical closures; "
          "terminal lifetimes, projected sections, missing history and unavailable cleanup checked")


if __name__ == "__main__":
    main()
