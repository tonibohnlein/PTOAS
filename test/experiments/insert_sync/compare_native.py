#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run one frozen-input population on one immutable compiler arm, serially.

Accept the frozen TSV's case_id/source/sha256/level columns. Preserve original
inputs, outputs, pass-boundary dumps and diagnostics. A separate --compare mode
produces per-kernel IR diffs; neither compilation nor an IR diff proves safety.
"""

import argparse
import csv
import difflib
import hashlib
import json
import os
import signal
import sys
import traceback
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    """Hash a native artifact without loading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, timeout):
    """Preserve diagnostics even when a compiler times out."""
    start = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode, result.stdout, result.stderr, time.monotonic() - start
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        return None, stdout.decode(errors="replace"), stderr.decode(errors="replace"), time.monotonic() - start


def warm_runtime(runtime):
    """Preload one serial native runtime before forking isolated invocations."""
    sys.path.insert(0, str(runtime))
    from ptoas import _cli
    from ptoas.mlir import ir
    class SerialContext(ir.Context):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_multithreading(False)
    ir.Context = SerialContext
    _cli._load_native_module()
    # Forking a process with active native threads is not this execution mode.
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("warm-process mode requires a single-threaded parent")
    return _cli


def run_warm(cli, arguments, timeout, directory, kind):
    """Avoid repeated imports; every compiler call still gets a fresh process.

    Native option state, crashes and timeouts cannot contaminate the next call.
    The parent performs no compilation while its sole child is running.
    """
    stdout_path, stderr_path = directory / f"{kind}.stdout", directory / f"{kind}.stderr"
    start = time.monotonic()
    sys.stdout.flush(); sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            os.dup2(stdout.fileno(), 1); os.dup2(stderr.fileno(), 2)
            try:
                code = cli.launch(arguments, wrapper=Path(cli.__file__))
            except BaseException:
                traceback.print_exc()
                code = 1
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(int(code))
    status = None
    while time.monotonic() - start < timeout:
        finished, raw = os.waitpid(pid, os.WNOHANG)
        if finished:
            status = os.waitstatus_to_exitcode(raw)
            break
        time.sleep(0.01)
    else:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    return (status, stdout_path.read_text(errors="replace") if stdout_path.exists() else "",
            stderr_path.read_text(errors="replace") if stderr_path.exists() else "", time.monotonic() - start)


def participation(stderr):
    """Attribute bypasses from pre-pass IR, not comments in the source file."""
    entries = []
    for chunk in stderr.split("// -----// IR Dump Before PTOInsertSync")[1:]:
        before = chunk.split("// -----// IR Dump After", 1)[0]
        # Debug dumps follow the pre-pass function. Do not mistake newly
        # planned flags in those dumps for fixed synchronization in the input.
        before = before.split("// === [PTOInsertSync Debug]", 1)[0]
        names = re.findall(r"func.func(?:\s+private)?\s+@([^\s(]+)", before)
        entries.append({"function": names[0] if names else "unknown",
                        "lifecycle_input_marker": 'pto.insert_sync.status = "lifecycle-plus-residuals"' in before,
                        "explicit_sync_bypass": bool(re.search(
                            r"pto\.(?:set_flag|wait_flag|record_event|wait_event)\b", before))})
    committed = set()
    for chunk in stderr.split("// -----// IR Dump After PTOInsertSync")[1:]:
        after = chunk.split("// -----// IR Dump Before", 1)[0]
        for line in after.splitlines():
            name = re.search(r"func.func\s+@([^\s(]+)", line)
            if (name and 'pto.insert_sync.status = "lifecycle-plus-residuals"' in line
                    and re.search(r"pto\.insert_sync\.lifecycle_channels = [1-9][0-9]*", line)):
                committed.add(name[1])
    return {"functions": entries,
            "lifecycle_committed": bool(entries) and all(
                entry["function"] in committed and not entry["lifecycle_input_marker"]
                for entry in entries),
            "analysis_executed": "[PTOInsertSync Debug] After Analysis" in stderr,
            "allocation_executed": "[PTOInsertSync Debug] After EventId Allocation" in stderr,
            "pass_invoked": bool(entries)}


def read_rows(manifest, root):
    """Validate the complete population before launching any compiler."""
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    seen = set()
    for row in rows:
        name = row["case_id"]
        if name in seen or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid or duplicate case_id")
        seen.add(name)
        source = (root / row["source"]).resolve()
        if not source.is_relative_to(root) or digest(source) != row["sha256"]:
            raise ValueError(f"input identity mismatch: {name}")
        if row["level"] not in ("level1", "level2", "level3"):
            raise ValueError(f"missing original compilation level: {name}")
    return rows


def compile_arm(args):
    """Compile PTO and C++ independently and retain every row, including failures."""
    if args.warm_process and Path(sys.executable).resolve() != args.python.resolve():
        raise ValueError("warm-process mode must run with the interpreter specified by --python")
    root = args.input_root.resolve()
    rows = read_rows(args.manifest, root)
    runtime = args.python_root.resolve() if args.python_root else args.build.resolve() / "python"
    native = runtime / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so"
    entry = runtime / "ptoas/_cli.py"
    command_prefix = [str(args.python.resolve()), "-c",
                      "import sys; sys.path.insert(0, sys.argv.pop(1)); "
                      "from pathlib import Path; from ptoas import _cli; from ptoas.mlir import ir; "
                      "exec('class SerialContext(ir.Context):\\n "
                      "def __init__(self, *a, **kw):\\n  super().__init__(*a, **kw); self.enable_multithreading(False)'); "
                      "ir.Context = SerialContext; "
                      "raise SystemExit(_cli.launch(sys.argv[1:], wrapper=Path(_cli.__file__)))", str(runtime)]
    cli = warm_runtime(runtime) if args.warm_process else None
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"population": args.population, "manifest_sha256": digest(args.manifest),
              "native_sha256": digest(native), "cli_sha256": digest(entry),
              "arm": args.arm, "arch": args.arch, "rows": [],
              "gm_alias": args.gm_alias, "audit_mode": args.audit,
              "buffer_generations": args.buffer_generations, "mmad_chains": args.mmad_chains,
              "effect_coverage": args.effect_coverage,
              "execution_mode": "isolated-fork-with-preloaded-runtime" if cli else "fresh-process",
              "semantic_verification": "not-run", "device_execution": "not-run"}
    for row in rows:
        directory = args.output / row["case_id"]
        directory.mkdir()
        record = {"case_id": row["case_id"], "input_sha256": row["sha256"], "runs": {}}
        for kind in ("pto", "cpp"):
            output = directory / f"output.{kind}"
            command = command_prefix + [f"--pto-arch={args.arch}",
                       f"--pto-level={row['level']}", "--enable-insert-sync"]
            if args.arm == "staged":
                command.append("--insert-sync-defer-same-pipe")
            if args.gm_alias:
                command.append(f"--insert-sync-gm-alias={args.gm_alias}")
            if args.buffer_generations:
                command.append("--insert-sync-buffer-generations")
            if args.mmad_chains:
                command.append("--insert-sync-mmad-chains")
            if args.effect_coverage:
                command.append(f"--insert-sync-effect-coverage={args.effect_coverage}")
            if args.audit and kind == "pto":
                command.append(f"--insert-sync-audit={args.audit}")
            if kind == "pto":
                command.extend(["--emit-pto-ir", "--pto-insert-sync-debug=1",
                                "--mlir-print-ir-before=pto-insert-sync",
                                "--mlir-print-ir-after=pto-insert-sync"])
            command.extend([str(root / row["source"]), "-o", str(output)])
            code, stdout, stderr, elapsed = (run_warm(cli, command[len(command_prefix):], args.timeout, directory, kind)
                                            if cli else run(command, args.timeout))
            (directory / f"{kind}.stdout").write_text(stdout)
            (directory / f"{kind}.stderr").write_text(stderr)
            result = {"command": command, "returncode": code, "seconds": elapsed,
                      "status": "pass" if code == 0 and output.is_file() else "failure"}
            if kind == "pto":
                result.update(participation(stderr))
                result["planning_diagnostics"] = [line for line in stderr.splitlines()
                    if "remark: InsertSync" in line or "error:" in line]
                result["audit_diagnostics"] = [
                    line for line in stderr.splitlines()
                    if "remark: InsertSync audit:" in line
                ]
                if output.is_file():
                    result["local_audit_verdicts"] = re.findall(
                        r'pto\.insert_sync\.audit = "([^"]+)"', output.read_text()
                    )
            record["runs"][kind] = result
        report["rows"].append(record)
        if len(report["rows"]) % 25 == 0:
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{len(report['rows'])}/{len(rows)} {row['case_id']} "
              f"pto={record['runs']['pto']['status']} cpp={record['runs']['cpp']['status']}", flush=True)
    report["native_unchanged"] = digest(native) == report["native_sha256"]
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["native_unchanged"]:
        raise RuntimeError("compiler changed during campaign")


def compare(arms, output):
    """Keep placement differences, not just counts, for every original kernel."""
    reports = [json.loads((path / "results.json").read_text()) for path in arms]
    if len({report["manifest_sha256"] for report in reports}) != 1:
        raise ValueError("cannot compare different populations")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for row in reports[0]["rows"]:
        name = row["case_id"]
        left = arms[0] / name / "output.pto"
        for arm in arms[1:]:
            right = arm / name / "output.pto"
            entry = {"case_id": name, "comparison": arm.name,
                     "both_compiled": left.is_file() and right.is_file()}
            if entry["both_compiled"]:
                difference = "".join(difflib.unified_diff(
                    left.read_text().splitlines(keepends=True), right.read_text().splitlines(keepends=True),
                    fromfile=str(left), tofile=str(right)))
                entry["ir_changed"] = bool(difference)
                if difference:
                    (output / f"{name}.{arm.name}.diff").write_text(difference)
            results.append(entry)
    (output / "differences.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--python-root", type=Path, help="Frozen complete Python/native runtime instead of --build")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--population")
    parser.add_argument("--arm", choices=("main", "combined", "staged"))
    parser.add_argument("--arch", choices=("a2", "a3"), default="a3")
    parser.add_argument("--warm-process", action="store_true",
                        help="Preload the runtime and fork each isolated invocation (Linux, one worker)")
    parser.add_argument("--buffer-generations", action="store_true")
    parser.add_argument("--mmad-chains", action="store_true")
    parser.add_argument("--effect-coverage", choices=("report", "strict"))
    parser.add_argument("--gm-alias", choices=("may-alias", "assume-disjoint-arguments"))
    parser.add_argument("--audit", choices=("report", "strict"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.output)
    elif all((args.manifest, args.input_root, args.build or args.python_root,
              args.python, args.population, args.arm)):
        compile_arm(args)
    else:
        parser.error("supply all compiler/population arguments or --compare")


if __name__ == "__main__":
    main()
