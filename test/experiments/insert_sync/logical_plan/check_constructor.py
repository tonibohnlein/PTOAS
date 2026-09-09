#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Serial native logical-constructor acceptance; no device performance claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
from observations import SERIAL_DRIVER, analyze, population, boundary_evidence
from measure import measure
from compare_boundaries import run as observe


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-root", required=True, type=Path)
    parser.add_argument("--native-driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", dest="cases")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--focused", action="store_true",
                      help="Small automatic gate; excludes milestone buffering and four-kernel coverage")
    mode.add_argument("--buffers-only", action="store_true",
                      help="Strict two/three-buffer scenarios, boundaries, C++ and guard mutations")
    parser.add_argument("--buffer-case", action="append", choices=("two_buffer", "three_buffer"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows = []
    library = args.python_root / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so"
    provenance = {"native_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(args.native_driver.read_bytes()).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "checkpoint_manifest_sha256": hashlib.sha256((HERE / "checkpoint/manifest.json").read_bytes()).hexdigest(),
        "device": "not-run"}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    def run(name, command, expected=0):
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
        elapsed = time.monotonic() - start
        (args.output / (name + ".stdout")).write_text(result.stdout)
        (args.output / (name + ".stderr")).write_text(result.stderr)
        (args.output / (name + ".command.json")).write_text(json.dumps(
            {"command": command, "exit": result.returncode, "seconds": elapsed}, indent=2) + "\n")
        assert (result.returncode == 0) == (expected == 0), (name, result.returncode, result.stderr[-2000:])
        return result, elapsed

    prefix = [sys.executable, "-c", SERIAL_DRIVER, str(args.python_root.resolve()),
              "--pto-arch=a3", "--pto-level=level3", "--enable-insert-sync"]
    selected = [] if args.buffers_only else (args.cases or (["one_buffer"] if args.focused else
                              ["one_buffer", "online_softmax", "q_proj", "qk_matmul"]))
    cases = {case["case_id"]: case for case in population()}
    result, _ = run("guard-growth", [str(args.native_driver.resolve()),
        str(HERE / "inputs/emission_contract.pto"), "guard-growth"])
    assert json.loads(result.stdout) == {"passed": True, "checks": 6}
    # No legacy seed is supplied to the logical arm. Each invocation reparses
    # the same frozen unsynchronized input independently.
    for case_id in selected:
        case = cases[case_id]
        scenarios = case.get("scenarios", [])
        assert scenarios and len({item["name"] for item in scenarios}) == len(scenarios), case_id
        source = case["source"].resolve()
        row = {"case": case_id, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "arms": {}}
        arms = {
            "upstream": [],
            "logical": ["--insert-sync-planner=logical", "--insert-sync-gm-alias=assume-disjoint-arguments"],
        }
        for arm, flags in arms.items():
            name = case_id + "." + arm
            output = args.output / (name + ".pto")
            _, elapsed = run(name, [*prefix, *flags, "--emit-pto-ir", str(source), "-o", str(output.resolve())])
            projection = analyze(output)
            row["arms"][arm] = {"seconds": elapsed, "projection": projection,
                                 "metrics": measure(output, case.get("scenarios", []))}
            if arm == "logical":
                assert any(attrs.get("pto.insert_sync.producer") == '"logical"'
                           for attrs in projection["status_attributes"]), row
        for frozen_arm in ("revised", "refiner", "logical"):
            artifact = case["artifacts"][frozen_arm]
            frozen = HERE / artifact["path"]
            output = args.output / (case_id + "." + ("checkpoint" if frozen_arm == "logical" else frozen_arm) + ".pto")
            output.write_bytes(frozen.read_bytes())
            row["arms"]["checkpoint" if frozen_arm == "logical" else frozen_arm] = {
                "origin": "frozen accepted M2 artifact", "sha256": artifact["sha256"],
                "projection": analyze(output), "metrics": measure(output, case.get("scenarios", []))}
        checkpoint = row["arms"]["checkpoint"]
        for key in ("payload", "allocations", "views", "abi"):
            assert checkpoint["projection"][key] == row["arms"]["logical"]["projection"][key], (case_id, "checkpoint " + key)
        # Frozen outputs retain the old exit policy. The current explicit
        # retirement drain and endpoint normalization intentionally change the
        # command/scalar inventory, never the payload or useful boundaries.
        now = row["arms"]["logical"]
        assert now["projection"]["mechanisms"]["PIPE_ALL"] == 1
        row["checkpoint_costs"] = []
        for scenario in case["scenarios"]:
            name = scenario["name"]
            old_metric = checkpoint["metrics"]["scenarios"][name]
            new_metric = now["metrics"]["scenarios"][name]
            assert old_metric["payload_sha256"] == new_metric["payload_sha256"]
            for action in ("pto.set_flag", "pto.wait_flag"):
                assert new_metric["counts"].get(action, 0) <= old_metric["counts"].get(action, 0), (case_id, name, action)
            old_scalar = sum(old_metric["scalar_counts"].values())
            new_scalar = sum(new_metric["scalar_counts"].values())
            assert new_scalar <= old_scalar, (case_id, name, "scalar regression", old_scalar, new_scalar)
            row["checkpoint_costs"].append({"scenario": name, "old_scalar": old_scalar, "new_scalar": new_scalar,
                "old_commands": old_metric["counts"], "new_commands": new_metric["counts"]})
        baseline = row["arms"]["refiner"]
        current = row["arms"]["logical"]
        for key in ("payload", "allocations", "views", "abi"):
            assert baseline["projection"][key] == current["projection"][key], (case_id, "changed " + key)
        for scenario in case.get("scenarios", []):
            name = scenario["name"]
            assert baseline["metrics"]["scenarios"][name]["payload_sha256"] == current["metrics"]["scenarios"][name]["payload_sha256"]
        row["boundaries"] = boundary_evidence(args.output / (case_id + ".refiner.pto"),
                                                args.output / (case_id + ".logical.pto"), case.get("scenarios", []))
        assert all(value["status"] == "checked" for value in row["boundaries"]), row["boundaries"]
        if case_id == "online_softmax":
            # Mandatory effectiveness, not acceptance by fallback or a lower
            # static pair inventory. Preserve the demonstrated looping gain.
            new = current["metrics"]["scenarios"]["16"]["counts"]
            old = baseline["metrics"]["scenarios"]["16"]["counts"]
            for action in ("pto.set_flag", "pto.wait_flag"):
                assert new[action] <= old[action] and new[action] <= 96
            assert current["projection"]["mechanisms"]["PIPE_ALL"] == 1
        if case_id == "qk_matmul":
            row["first_panel_readiness"] = []
            for scenario in case["scenarios"]:
                observation, metric = observe(args.output / (case_id + ".logical.pto"), scenario)
                assert metric["counts"].get("pto.set_flag", 0) == metric["counts"].get("pto.wait_flag", 0)
                if scenario["name"] == "0":
                    continue
                loads = [p for p, v in enumerate(observation.before) if v["lane"] == "PIPE_MTE2"]
                first_reader = next(p for p, v in enumerate(observation.before) if v["lane"] == "PIPE_MTE1")
                acquired = observation.before[first_reader]["completed"].get("PIPE_MTE2", -1)
                # The unchanged input has Q0 and Q1 preloads before its loop.
                # Its first Q0 extract must acquire Q0 without capturing Q1.
                assert len(loads) >= 2 and acquired == loads[0] and acquired < loads[1], (scenario, acquired, loads)
                row["first_panel_readiness"].append({"scenario": scenario["name"],
                    "first_preload": loads[0], "independent_preload": loads[1],
                    "first_reader": first_reader, "acquired_mte2_prefix": acquired})
                new = current["metrics"]["scenarios"][scenario["name"]]["counts"]
                old = baseline["metrics"]["scenarios"][scenario["name"]]["counts"]
                assert new["pto.set_flag"] < old["pto.set_flag"] and new["pto.wait_flag"] < old["pto.wait_flag"]
            assert row["first_panel_readiness"], "QK readiness observations missing"
        if case_id == "q_proj":
            for scenario in case["scenarios"]:
                new = current["metrics"]["scenarios"][scenario["name"]]["counts"]
                old = baseline["metrics"]["scenarios"][scenario["name"]]["counts"]
                assert new["pto.set_flag"] == new["pto.wait_flag"]
                assert new["pto.set_flag"] < old["pto.set_flag"] and new["pto.wait_flag"] < old["pto.wait_flag"]
        # Exercise the actual C++ emission pipeline, keeping compilation time
        # distinct from static/executed mechanisms and scalar command overhead.
        _, row["cpp_seconds"] = run(case_id + ".logical.cpp", [*prefix, *arms["logical"], str(source),
                    "-o", str((args.output / (case_id + ".logical.cpp")).resolve())])
        rows.append(row)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(case_id, {arm: value["projection"]["mechanisms"] for arm, value in row["arms"].items()}, flush=True)

    structured = []
    for fixture in (() if args.buffers_only else ("skipped_reader", "emission_contract", "first_else_reader")):
        source = HERE / "inputs" / (fixture + ".pto")
        outputs = {}
        for arm, flags in (("existing", []), ("logical", ["--insert-sync-planner=logical"])):
            output = args.output / (fixture + "." + arm + ".pto")
            run(fixture + "." + arm, [*prefix, "--insert-sync-gm-alias=assume-disjoint-arguments",
                *flags, "--emit-pto-ir", str(source), "-o", str(output.resolve())])
            outputs[arm] = output
        baseline = analyze(outputs["existing"])
        current = analyze(outputs["logical"])
        for key in ("payload", "allocations", "views", "abi"):
            assert baseline[key] == current[key], (fixture, key)
        for trips in (0, 1, 2, 5):
            for take in (0, 1):
                scenario = {"name": f"{trips}/{take}", "arguments": ["src", "dst", trips, take]}
                _, new = observe(outputs["logical"], scenario)
                old = measure(outputs["existing"], [scenario])["scenarios"][scenario["name"]]
                assert old["payload_sha256"] == new["payload_sha256"], (fixture, scenario)
                assert new["counts"].get("pto.set_flag", 0) == new["counts"].get("pto.wait_flag", 0)
                structured.append({"fixture": fixture, "scenario": scenario, "logical": new})
    (args.output / "structured.json").write_text(json.dumps(structured, indent=2) + "\n")

    # Relation-derived guards must independently construct both physical slot
    # counts. Negative/empty bounds execute no payload or remainder; every
    # residue class, partial fill and subsequent reuse receives matched events.
    buffer_rows = []
    for case_id in (() if args.focused else (args.buffer_case or ("two_buffer", "three_buffer"))):
        source = cases[case_id]["source"].resolve()
        common = [*prefix, "--insert-sync-gm-alias=assume-disjoint-arguments", "--emit-pto-ir", str(source)]
        outputs = {}
        for arm, flags in (("existing", []), ("logical", ["--insert-sync-planner=logical"])):
            output = args.output / (case_id + "." + arm + ".pto")
            _, elapsed = run(case_id + "." + arm, [*common, *flags, "-o", str(output.resolve())])
            outputs[arm] = output
        a, b = (analyze(outputs[arm]) for arm in ("existing", "logical"))
        for key in ("payload", "allocations", "views", "abi"):
            assert a[key] == b[key], (case_id, "changed", key)
        assert any(attrs.get("pto.insert_sync.producer") == '\"logical\"'
                   for attrs in b["status_attributes"]), b
        scenarios = [{"name": str(n), "arguments": ["src", "dst", n]}
                     for n in (-1, 0, 1, 2, 3, 4, 5, 7, 16)]
        old = measure(outputs["existing"], scenarios)
        new = measure(outputs["logical"], scenarios)
        for scenario in scenarios:
            name = scenario["name"]
            _, metric = observe(outputs["logical"], scenario)
            assert metric["payload_sha256"] == old["scenarios"][name]["payload_sha256"]
            assert metric["counts"].get("pto.set_flag", 0) == metric["counts"].get("pto.wait_flag", 0)
            if int(name) <= 0:
                assert metric["scalar_counts"].get("arith.remsi", 0) == 0
                assert metric["counts"].get("pto.set_flag", 0) == 0
        boundaries = boundary_evidence(outputs["existing"], outputs["logical"], scenarios)
        assert all(row["status"] == "checked" for row in boundaries), boundaries
        # Compile the checked emitted plan to C++; no second construction is
        # needed to challenge its actual newly introduced guards in codegen.
        run(case_id + ".cpp", [sys.executable, "-c", SERIAL_DRIVER, str(args.python_root.resolve()),
            "--pto-arch=a3", "--pto-level=level3", str(outputs["logical"].resolve()),
            "-o", str((args.output / (case_id + ".cpp")).resolve())])
        buffer_rows.append({"case": case_id, "strict": "applied", "seconds": elapsed,
            "upstream": {"mechanisms": a["mechanisms"], "metrics": old},
            "logical": {"mechanisms": b["mechanisms"], "metrics": new}, "boundaries": boundaries})
        (args.output / "buffering.json").write_text(json.dumps(buffer_rows, indent=2) + "\n")
    for mutation in (() if args.focused else
                     ("none", "change-residue", "unguard-remainder", "change-slot-distance")):
        result, _ = run("guard-reconstruction." + mutation, [str(args.native_driver.resolve()),
            str(cases["two_buffer"]["source"].resolve()), mutation])
        answer = json.loads(result.stdout)
        assert answer["invoked"], answer
        if mutation == "none":
            assert answer["applied"], answer
        else:
            assert answer["changed"] and answer["counts_preserved"] and not answer["applied"] and answer["original_preserved"], answer

    if args.buffers_only:
        print("Strict buffering acceptance passed; device validation remains separate")
        return

    # Positive construction is required before each native mutation challenge.
    fixture = HERE / "inputs/emission_contract.pto"
    for mutation in ("none", "erase-wait", "widen-barrier", "swap-loads", "change-rounding", "add-allocation"):
        result, _ = run("reconstruction." + mutation, [str(args.native_driver.resolve()), str(fixture), mutation])
        answer = json.loads(result.stdout)
        assert answer["invoked"], answer
        if mutation == "none":
            assert answer["applied"] and answer["status"] == "applied", answer
        else:
            assert answer["changed"] and not answer["applied"] and answer["original_preserved"], answer
    result, _ = run("reconstruction.softmax-none", [str(args.native_driver.resolve()),
        str(cases["online_softmax"]["source"].resolve()), "none"])
    answer = json.loads(result.stdout)
    assert answer["invoked"] and answer["applied"] and answer["status"] == "applied", answer
    result, _ = run("reconstruction.swap-wait-keys", [str(args.native_driver.resolve()),
        str(cases["online_softmax"]["source"].resolve()), "swap-wait-keys"])
    answer = json.loads(result.stdout)
    assert answer["changed"] and answer["counts_preserved"] and not answer["applied"] and answer["original_preserved"], answer

    # Low budgets must remain explicit strict failures and safe whole-function
    # fallback. Compare the same fallback configuration, not the revised flags.
    source = HERE / "inputs/skipped_reader.pto"
    common = [*prefix, "--insert-sync-gm-alias=assume-disjoint-arguments", "--emit-pto-ir", str(source)]
    run("limit.strict", [*common, "--insert-sync-planner=logical", "--insert-sync-logical-work-budget=0",
                         "-o", str((args.output / "limit.strict.pto").resolve())], expected=1)
    for arm, flags in (("existing", []), ("fallback", ["--insert-sync-planner=logical-or-existing",
                                                     "--insert-sync-logical-work-budget=0"])):
        output = args.output / ("limit." + arm + ".pto")
        run("limit." + arm, [*common, *flags, "-o", str(output.resolve())])
    a, b = (analyze(args.output / ("limit." + arm + ".pto")) for arm in ("existing", "fallback"))
    for key in ("payload", "allocations", "views", "abi", "placements", "mechanisms", "sync_control"):
        assert a[key] == b[key], ("fallback changed", key)
    assert any(attrs.get("pto.insert_sync.logical_status") == '"analysis-limit"'
               and attrs.get("pto.insert_sync.producer") == '"existing-fallback"'
               for attrs in b["status_attributes"]), b
    print(("Focused constructor gate" if args.focused else "Native constructor acceptance") +
          " passed; device validation remains separate")


if __name__ == "__main__":
    main()
