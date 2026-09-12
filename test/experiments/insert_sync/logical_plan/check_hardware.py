#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""S7 native contract gate. Executes real import/emission/reconstruction.

This is not device verification. The explicit contract must separately be
qualified against the selected runtime/toolchain before deploying its output.
No compilation-time measurements include diagnostic proof audits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CONTRACT = "a2a3-mmad-acc-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The parent structured gate intentionally reuses its diagnostics directory
    # across local/CTest invocations. Every artifact below is rewritten and the
    # summary is rebuilt from this invocation, so stale directory existence must
    # not turn an otherwise reproducible rerun into a harness failure.
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.python_root.resolve()))
    from observations import analyze
    from compare_boundaries import run as observe
    from s6_report import parse_reports, validate_report

    def invoke(name, command, audit=False, expected=0):
        env = dict(os.environ)
        for key in ("PTOAS_STRUCTURED_PLAN_JSON", "PTOAS_LOGICAL_TRACE"):
            env.pop(key, None)
        if audit:
            env["PTOAS_STRUCTURED_PLAN_JSON"] = "1"
        result = subprocess.run([str(x) for x in command], env=env,
                                capture_output=True, text=True, timeout=120)
        (args.output / (name + ".stdout")).write_text(result.stdout)
        (args.output / (name + ".stderr")).write_text(result.stderr)
        (args.output / (name + ".command.json")).write_text(json.dumps(
            dict(command=list(map(str, command)), returncode=result.returncode), indent=2) + "\n")
        if (result.returncode == 0) != (expected == 0):
            raise RuntimeError(f"{name}: unexpected exit {result.returncode}: {result.stderr[-3000:]}")
        return result

    specs = json.loads((HERE / "hardware_inputs/manifest.json").read_text())["cases"]
    specs = [dict(spec, name=spec["name"] + "_" + arch, arch=arch)
             for spec in specs for arch in ("a2", "a3")]
    rows = []
    for spec in specs:
        source = HERE / "hardware_inputs" / spec["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError("changed hardware fixture " + spec["name"])
        if spec["arch"] == "a2":
            text = source.read_text()
            if text.count('pto.target_arch = "a3"') != 1:
                raise RuntimeError("ambiguous fixture architecture binding")
            source = args.output / (spec["name"] + ".input.pto")
            source.write_text(text.replace('pto.target_arch = "a3"', 'pto.target_arch = "a2"'))
        original = analyze(source)
        plans = {}
        for contract in ("conservative", CONTRACT):
            name = spec["name"] + "." + contract
            output = args.output / (name + ".pto")
            invocation = invoke(name, [args.driver, source, "none", output, contract], audit=True)
            verdict = json.loads(invocation.stdout)
            if not all(verdict.get(k) for k in ("accepted", "atomic", "expected")):
                raise RuntimeError(f"{name}: constructor did not accept: {verdict}")
            reports = parse_reports(invocation.stderr)
            if len(reports) != 1:
                raise RuntimeError("one kernel report required")
            accounting = validate_report(reports[0])
            eligible = accounting["intrinsic_accumulator_requirements"]
            if bool(eligible) != (contract == CONTRACT and spec["expect_intrinsic"]):
                raise RuntimeError(f"{name}: actual native rule usage differs: {eligible}")
            projection = analyze(output)
            for field in ("payload", "allocations", "views", "abi"):
                if projection[field] != original[field]:
                    raise RuntimeError(f"{name}: changed {field}")
            executions = []
            for trips in (0, 1, 2, 3, 7):
                scenario = {"name": str(trips), "arguments": ["A", "B", "C", trips]}
                before, _ = observe(source, scenario)
                after, metrics = observe(output, scenario)
                if before.payload != after.payload or after.tokens:
                    raise RuntimeError(f"{name}: payload/participation mismatch at {trips}")
                executions.append(dict(trips=trips, mechanisms=metrics))
            plans[contract] = dict(output=str(output), mechanisms=projection["mechanisms"],
                                   scalar=projection["sync_control"], accounting=accounting,
                                   executions=executions)
        base = plans["conservative"]["mechanisms"]
        hardware = plans[CONTRACT]["mechanisms"]
        if base["PIPE_ALL"] != 1 or hardware["PIPE_ALL"] != 1:
            raise RuntimeError("retirement policy changed")
        if spec["expect_intrinsic"]:
            if hardware["named"].get("PIPE_M", 0) >= base["named"].get("PIPE_M", 0):
                raise RuntimeError(spec["name"] + ": rule did not eliminate a named M barrier")
            for mutation in ("drop-m-to-mte1", "drop-m-to-fix", "wrong-key", "drop-retirement"):
                name = spec["name"] + "." + mutation
                result = invoke(name, [args.driver, source, mutation,
                                       args.output / (name + ".pto"), CONTRACT])
                verdict = json.loads(result.stdout)
                if not (verdict["mutation_applied"] and verdict["expected"] and verdict["atomic"]):
                    raise RuntimeError(f"required concrete protection was not challenged: {verdict}")
        elif hardware != base:
            raise RuntimeError(spec["name"] + ": unsupported geometry changed its conservative plan")
        rows.append(dict(spec=spec, arms=plans))

    # The unchanged benchmark population is also compared under BOTH contracts.
    # Synthetic geometry success is not a substitute for real q_proj import.
    from observations import population
    corpus_rows = []
    for case in population():
        # Bind architecture once, before either arm and before projection.
        # The frozen input and all its payload/control remain unchanged.
        raw = case['source'].read_text()
        bound = raw.replace('pto.target_arch = "a2a3"', 'pto.target_arch = "a3"')
        # Some frozen inputs carry no module architecture at all. The pass binds
        # one, which would otherwise read as a changed module attribute in the
        # projection below. Bind it here so both arms are compared against the
        # same architecture, not against an unbound module.
        if 'pto.target_arch' not in bound:
            bound = bound.replace('module {', 'module attributes {pto.target_arch = "a3"} {', 1)
        source = args.output / ('original.' + case['case_id'] + '.input.a3.pto')
        source.write_text(bound)
        original = analyze(source); arms = {}
        for contract in ('conservative', CONTRACT):
            name = 'original.' + case['case_id'] + '.' + contract
            output = args.output / (name + '.pto')
            result = invoke(name, [args.driver, source, 'none', output, contract], audit=True)
            verdict = json.loads(result.stdout)
            if not (verdict['accepted'] and verdict['expected'] and verdict['atomic']):
                raise RuntimeError(f'{name}: actual original input refused: {verdict}')
            projection = analyze(output)
            for field in ('payload', 'allocations', 'views', 'abi'):
                if projection[field] != original[field]:
                    raise RuntimeError(f'{name}: changed original {field}')
            report = validate_report(parse_reports(result.stderr)[0])
            executions = []
            for scenario in case['scenarios']:
                before, _ = observe(source, scenario); after, metrics = observe(output, scenario)
                if before.payload != after.payload or after.tokens:
                    raise RuntimeError(f'{name}: changed payload or outstanding notification')
                if case['case_id'] == 'qk_matmul' and scenario['arguments'][4] > 0:
                    first = next(i for i, payload in enumerate(after.payload) if payload[0] == 'pto.textract')
                    if after.before[first]['completed'].get('PIPE_MTE2', -1) != 0:
                        raise RuntimeError('hardware contract broadened original first Q-panel acquisition')
                executions.append(dict(scenario=scenario['name'], mechanisms=metrics))
            arms[contract] = dict(mechanisms=projection['mechanisms'], accounting=report, executed=executions)
        if case['case_id'] == 'q_proj':
            hw, old = arms[CONTRACT], arms['conservative']
            if hw['accounting']['intrinsic_accumulator_requirements'] == 0:
                raise RuntimeError('unchanged q_proj did not use the native hardware rule')
            if hw['mechanisms']['named'].get('PIPE_M', 0) >= old['mechanisms']['named'].get('PIPE_M', 0):
                raise RuntimeError('unchanged q_proj has no actual named-M-barrier improvement')
        corpus_rows.append(dict(case=case['case_id'], source_sha256=case['sha256'],
                                pass_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                architecture_binding='a2a3 -> a3 only', arms=arms))

    # Exercise the actual existing pass option, not just the test-driver API.
    source = HERE / "hardware_inputs/mmad_qproj_16.pto"
    output = args.output / "pass-option.pto"
    argument = "-pto-insert-sync=planner=structured hardware-contract=" + CONTRACT + " gm-alias=assume-disjoint-arguments"
    result = invoke("pass-option", [args.opt, source, argument, "-o", output], audit=True)
    if not validate_report(parse_reports(result.stderr)[0])["intrinsic_accumulator_requirements"]:
        raise RuntimeError("public pass option did not reach hardware-qualified construction")
    invoke("reject-existing-profile", [args.opt, source,
           "-pto-insert-sync=planner=existing hardware-contract=" + CONTRACT], expected=1)
    invoke("reject-misspelled-profile", [args.opt, source,
           "-pto-insert-sync=planner=structured hardware-contract=a2a3-mmad-ac-v1"], expected=1)
    summary = dict(status="passed", cases=rows, original_population=corpus_rows, native_driver_sha256=hashlib.sha256(args.driver.read_bytes()).hexdigest(),
                   device="NOT_RUN", timing="audit invocations; not performance evidence")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="passed", cases=len(rows), device="NOT_RUN")))


if __name__ == "__main__":
    main()
