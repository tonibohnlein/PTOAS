#!/usr/bin/env python3
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
sys.path.insert(0, str(HERE.parent / "performance"))
from run_qwen_additions import REVISED_FLAGS, SERIAL_DRIVER, analyze
from run_native_handoffs import population, boundary_evidence
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
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows = []

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
    selected = args.cases or ["one_buffer", "online_softmax"]
    cases = {case["case_id"]: case for case in population()}
    # No legacy seed is supplied to the logical arm. Each invocation reparses
    # the same frozen unsynchronized input independently.
    for case_id in selected:
        case = cases[case_id]
        source = case["source"].resolve()
        row = {"case": case_id, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "arms": {}}
        arms = {
            "revised": REVISED_FLAGS,
            "refiner": [*REVISED_FLAGS, "--insert-sync-handoff-planning"],
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
            assert current["projection"]["mechanisms"]["PIPE_ALL"] == 0
        # Exercise the actual C++ emission pipeline, keeping compilation time
        # distinct from static/executed mechanisms and scalar command overhead.
        _, row["cpp_seconds"] = run(case_id + ".logical.cpp", [*prefix, *arms["logical"], str(source),
                    "-o", str((args.output / (case_id + ".logical.cpp")).resolve())])
        rows.append(row)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(case_id, {arm: value["projection"]["mechanisms"] for arm, value in row["arms"].items()}, flush=True)

    structured = []
    for fixture in ("skipped_reader", "emission_contract"):
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

    # Positive construction is required before each native mutation challenge.
    fixture = HERE / "inputs/emission_contract.pto"
    for mutation in ("none", "erase-wait", "widen-barrier", "swap-loads", "change-rounding", "add-allocation"):
        result, _ = run("reconstruction." + mutation, [str(args.native_driver.resolve()), str(fixture), mutation])
        answer = json.loads(result.stdout)
        assert answer["invoked"], answer
        if mutation != "none":
            assert answer["changed"] and not answer["applied"] and answer["original_preserved"], answer
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
    print("Native constructor acceptance passed; device validation remains separate")


if __name__ == "__main__":
    main()
