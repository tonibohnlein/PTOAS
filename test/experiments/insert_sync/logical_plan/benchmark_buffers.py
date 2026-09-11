# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""Serial, repeated, paired compilation; seven-case coverage is always explicit.

Mechanisms are deliberately not combined into a score. C++ generation starts
from already synchronized PTO and is never reported as synchronization time.
A harness timeout is a failed observation, not a faster compiler result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARMS = ("existing", "structured", "logical")
POPULATION = ("one_buffer", "two_buffer", "three_buffer", "four_use",
              "online_softmax", "qk_matmul", "q_proj")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def ratio_summary(trials, candidate, repeats):
    """Use matched successful rounds only; incompleteness cannot pass a gate."""
    groups = {}
    for row in trials:
        if row["round"] >= 0:
            group = groups.setdefault(row["round"], {})
            if row["arm"] in group:
                raise ValueError("duplicate arm in a paired round")
            group[row["arm"]] = row
    ratios = []
    for repeat in range(repeats):
        pair = groups.get(repeat, {})
        left, right = pair.get("existing"), pair.get(candidate)
        if left and right and left["status"] == right["status"] == "applied":
            if left["seconds"] <= 0 or right["seconds"] < 0:
                raise ValueError("invalid measured interval")
            ratios.append(right["seconds"] / left["seconds"])
    return dict(paired_samples=len(ratios), expected_samples=repeats,
                complete=len(ratios) == repeats,
                ratios=ratios, median_ratio=statistics.median(ratios) if ratios else None,
                max_ratio=max(ratios) if ratios else None)


def provenance(python_root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
        return result.stdout if result.returncode == 0 else None
    head = git("rev-parse", "HEAD")
    changes = git("diff", "HEAD", "--")
    status = git("status", "--porcelain")
    # Record the actually imported compiler/runtime path as well as candidate
    # binary hashes. A source checkout alone is not binary provenance.
    from ptoas import _loader
    native = _loader.ensure_core()
    native_path = Path(native.__file__).resolve()
    binaries = {str(native_path): digest(native_path)}
    for path in sorted(python_root.rglob("libPTOASCompiler*.so*")):
        if path.is_file():
            binaries[str(path.resolve())] = digest(path)
    build = python_root.parent
    cache = build / "CMakeCache.txt"
    return dict(head=head.decode().strip() if head else None,
                dirty=status.decode() if status is not None else None,
                tracked_diff_sha256=hashlib.sha256(changes).hexdigest() if changes is not None else None,
                runner_sha256=digest(__file__), python=sys.executable,
                python_root=str(python_root), binary_sha256=binaries,
                cmake_cache_sha256=digest(cache) if cache.exists() else None)


def invoke(command, prefix, env, timeout):
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                capture_output=True, timeout=timeout)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
        status = "applied" if code == 0 else ("signal" if code < 0 else "refused-or-error")
    except subprocess.TimeoutExpired as error:
        code, status = None, "harness-timeout"
        stdout, stderr = error.stdout or "", error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
    seconds = time.perf_counter() - started
    Path(str(prefix) + ".stdout").write_text(stdout)
    Path(str(prefix) + ".stderr").write_text(stderr)
    row = dict(command=command, returncode=code, status=status, seconds=seconds,
               stdout=str(prefix) + ".stdout", stderr=str(prefix) + ".stderr")
    dump(str(prefix) + ".command.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["existing", "structured"])
    parser.add_argument("--cases", nargs="+", choices=POPULATION, default=list(POPULATION))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="External harness deadline; not a compiler analysis quota")
    parser.add_argument("--emit-cpp", action="store_true")
    parser.add_argument("--require-ratio", type=float,
                        help="Gate each candidate's median paired whole-compilation ratio (at least 3 rounds)")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("benchmark validation requires Python assertions")
    if args.repeats < 1 or args.warmups < 0 or args.timeout <= 0:
        parser.error("invalid repeat, warmup or timeout")
    if len(set(args.arms)) != len(args.arms) or len(set(args.cases)) != len(args.cases):
        parser.error("duplicate arms or cases")
    if args.require_ratio is not None and ("existing" not in args.arms or len(args.arms) < 2
                                         or args.repeats < 3 or args.require_ratio <= 0):
        parser.error("a ratio gate needs existing, a candidate, at least 3 repeats, and a positive threshold")
    python_root = args.python_root.resolve()
    sys.path.insert(0, str(python_root))
    from observations import SERIAL_DRIVER, analyze, population
    from measure import measure
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # S6 diagnostics deliberately perform extra per-handoff verification. They
    # are not part of normal compilation and cannot contaminate paired timings.
    disabled_diagnostics = {name: env.pop(name) for name in
                            ("PTOAS_STRUCTURED_PLAN_JSON", "PTOAS_LOGICAL_TRACE") if name in env}
    cases = {item["case_id"]: item for item in population()}
    if set(cases) != set(POPULATION):
        raise RuntimeError("review the declared population before changing its coverage denominator")
    output = dict(provenance=provenance(python_root), disabled_diagnostics=disabled_diagnostics,
                  population=list(POPULATION),
                  requested_cases=args.cases, requested_arms=args.arms,
                  omitted_cases=[c for c in POPULATION if c not in args.cases],
                  repeats=args.repeats, warmups=args.warmups, rows=[],
                  device="NOT_RUN", timed_stage="whole PTO emission, synchronization included")
    all_ok = True
    for case_id in args.cases:
        case = cases[case_id]
        row = dict(case=case_id, source_sha256=digest(case["source"]), trials=[], ratios={})
        reference = None
        last_success = {}
        for repeat in range(-args.warmups, args.repeats):
            # Rotate all arms each round; with two arms this alternates order.
            # No concurrent compilers.
            offset = (repeat + args.warmups) % len(args.arms)
            order = args.arms[offset:] + args.arms[:offset]
            for arm in order:
                stem = args.output / f"{case_id}.{repeat}.{arm}"
                pto = Path(str(stem) + ".pto")
                command = [sys.executable, "-c", SERIAL_DRIVER, str(python_root),
                           "--pto-arch=a3", "--pto-level=level3", "--enable-insert-sync",
                           f"--insert-sync-planner={arm}",
                           "--insert-sync-gm-alias=assume-disjoint-arguments",
                           "--emit-pto-ir", str(case["source"].resolve()), "-o", str(pto.resolve())]
                if arm == "structured":
                    command.insert(-4, "--insert-sync-logical-work-budget=0")
                result = invoke(command, stem, env, args.timeout)
                result.update(round=repeat, arm=arm)
                if result["status"] == "applied":
                    try:
                        report = analyze(pto)
                        projection = {key: report[key] for key in ("payload", "allocations", "views", "abi")}
                        if arm in ("logical", "structured") and not any(
                                attrs.get("pto.insert_sync.producer") == f'"{arm}"'
                                for attrs in report["status_attributes"]):
                            raise ValueError("the requested strict planner did not produce this output")
                        if reference is None:
                            reference = projection
                        if projection != reference:
                            raise ValueError("original payload/control/geometry differs between arms")
                        result.update(output_sha256=digest(pto), mechanisms=report["mechanisms"],
                                      scalar_sites=report["sync_control"],
                                      keys_by_direction=report["event_ids_by_direction"],
                                      status_attributes=report["status_attributes"])
                        last_success[arm] = pto
                    except Exception as error:
                        result.update(status="validation-error", validation_error=str(error))
                if result["status"] != "applied":
                    all_ok = False
                row["trials"].append(result)
                # Retain partial evidence even on interruption/refusal.
                dump(args.output / "summary.json", {**output, "in_progress": row})
        for arm, pto in last_success.items():
            # Outside the timed region; these are replay/observation results.
            try:
                row.setdefault("executed", {})[arm] = measure(pto, case.get("scenarios", []))
            except Exception as error:
                all_ok = False
                row.setdefault("replay_errors", {})[arm] = str(error)
            if args.emit_cpp:
                prefix = args.output / f"{case_id}.{arm}.cpp"
                command = [sys.executable, "-c", SERIAL_DRIVER, str(python_root),
                           "--pto-arch=a3", "--pto-level=level3", str(pto.resolve()), "-o", str(prefix)]
                row.setdefault("cpp_from_synchronized_pto", {})[arm] = invoke(
                    command, prefix, env, args.timeout)
                all_ok &= row["cpp_from_synchronized_pto"][arm]["status"] == "applied"
        if "existing" in args.arms:
            for arm in args.arms:
                if arm == "existing":
                    continue
                ratios = ratio_summary(row["trials"], arm, args.repeats)
                if args.require_ratio is not None:
                    ratios["passed"] = (ratios["complete"] and
                                         ratios["median_ratio"] <= args.require_ratio)
                    all_ok &= ratios["passed"]
                row["ratios"][arm] = ratios
        output["rows"].append(row)
        dump(args.output / "summary.json", output)
        print(json.dumps({"case": case_id, "ratios": row["ratios"],
                          "outcomes": {arm: [t["status"] for t in row["trials"]
                                             if t["arm"] == arm and t["round"] >= 0]
                                       for arm in args.arms}}))
    output["status"] = "passed" if all_ok else "incomplete-or-failed"
    output["ratio_gate"] = args.require_ratio
    dump(args.output / "summary.json", output)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
