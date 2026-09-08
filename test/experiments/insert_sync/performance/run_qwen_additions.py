#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Serial A3 comparison of eight unchanged Qwen kernels; no device execution.

Retain PTO/C++, native pass diagnostics, payload/ABI/allocation projections,
and synchronization placements anchored to non-sync program points.
"""

import argparse
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_native import digest, participation, run
from measure import SYNC, attrs, children, static_metrics


CASES = {
    "rmsnorm": "aiv/rmsnorm.pto",
    "softmax": "aiv/softmax.pto",
    "online_softmax": "aiv/online_softmax.pto",
    "silu": "aiv/silu.pto",
    "rope_kv_cache": "aiv/rope_kv_cache.pto",
    "qk_matmul": "aic/qk_matmul.pto",
    "sv_matmul": "aic/sv_matmul.pto",
    "q_proj": "aic/q_proj.pto",
}
REVISED_FLAGS = [
    "--insert-sync-gm-alias=assume-disjoint-arguments",
    "--insert-sync-effect-coverage=report", "--insert-sync-defer-same-pipe",
    "--insert-sync-mmad-chains", "--insert-sync-buffer-generations",
]
SERIAL_DRIVER = """import sys
sys.path.insert(0, sys.argv.pop(1))
from pathlib import Path
from ptoas import _cli
from ptoas.mlir import ir
class SerialContext(ir.Context):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_multithreading(False)
ir.Context = SerialContext
raise SystemExit(_cli.launch(sys.argv[1:], wrapper=Path(_cli.__file__)))
"""


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def project(module):
    """Exact non-sync structure, attrs, types and SSA wiring (no algebraic rewrites).

Compiler status attrs and the explicitly recorded GM contract are removed.
Synchronization-only branches and comparisons used exclusively by those branches
are recorded separately. Payload control, constants, views and effects remain.
"""
    values, terms, records, placements, allocations, views, abi = {}, {}, [], [], [], [], []
    sync_control = Counter()

    def only_sync(op):
        if op.name in SYNC or op.name == "scf.yield":
            return not op.results
        return (op.name == "scf.if" and not op.results and
                all(only_sync(view.operation) for region in op.regions
                    for block in region.blocks for view in block.operations))

    def walk(op):
        for child in children(op):
            yield child
            yield from walk(child)

    excluded = {op for op in walk(module.operation) if op.name == "scf.if" and only_sync(op)}
    control_arithmetic = {"arith.cmpi", "arith.subi", "arith.addi", "arith.andi", "arith.ori", "arith.xori"}
    while True:
        found = {op for op in walk(module.operation) if op not in excluded and op.name in control_arithmetic
                 and op.results and any(value.uses for value in op.results)
                 and all(use.owner in excluded for value in op.results for use in value.uses)}
        if not found:
            break
        excluded.update(found)

    def visit(op, path, guards=()):
        properties = attrs(op)
        if op in excluded:
            sync_control[op.name] += 1
            if op.name in control_arithmetic:
                for i, value in enumerate(op.results):
                    terms[value] = hashed([op.name, properties, [terms[v] for v in op.operands], i])
            else:
                for ri, region in enumerate(op.regions):
                    branch = guards + ((terms[op.operands[0]], ri == 0),)
                    for block in region.blocks:
                        for view in block.operations:
                            child = view.operation
                            if child.name != "scf.yield":
                                visit(child, path, branch)
            return
        if op.name in SYNC:
            placements.append({"point": path, "op": op.name, "attrs": properties, "guards": guards})
            return
        for key in list(properties):
            if key.startswith("pto.insert_sync."):
                del properties[key]
            elif key == "pto.gm_alias":
                if properties[key] != '"assume-disjoint-arguments"':
                    raise ValueError("unexpected GM allocation contract")
                del properties[key]
        record = {"point": path, "op": op.name, "attrs": properties,
                  "operands": [values[v] for v in op.operands],
                  "types": [str(v.type) for v in op.results]}
        operand_terms = [terms[v] for v in op.operands]
        for i, value in enumerate(op.results):
            values[value] = [path, "result", i]
            terms[value] = hashed([op.name, properties, record["types"], operand_terms, i])
        records.append(record)
        if op.name in {"pto.alloc_tile", "pto.alloc", "memref.alloc", "memref.alloca"}:
            allocations.append({**record, "operand_expression_sha256": operand_terms})
        if op.name in {"pto.make_tensor_view", "pto.partition_view", "pto.subview",
                       "pto.multi_tile_get", "pto.bind_tile"}:
            views.append({**record, "operand_expression_sha256": operand_terms})
        if op.name == "func.func":
            abi.append(properties)
        for ri, region in enumerate(op.regions):
            for bi, block in enumerate(region.blocks):
                block_path = path + ["region", ri, "block", bi]
                records.append({"point": block_path, "block_types": [str(v.type) for v in block.arguments]})
                for ai, value in enumerate(block.arguments):
                    values[value] = [block_path, "argument", ai]
                    terms[value] = hashed([values[value], str(value.type)])
                index = 0
                for view in block.operations:
                    child = view.operation
                    visit(child, block_path + [index], guards)
                    index += child.name not in SYNC and child not in excluded
    visit(module.operation, [])
    return {"payload": records, "placements": placements, "allocations": allocations,
            "views": views, "abi": abi, "sync_control": dict(sync_control)}


def analyze(path):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        module = ir.Module.parse(path.read_text())
        if not module.operation.verify():
            raise ValueError(f"invalid emitted IR: {path}")
        metrics = static_metrics(module.operation)
        result = project(module)
        counts = metrics["counts"]
        barriers = Counter()
        for key, value in counts.items():
            if key.startswith("barrier:"):
                barriers[re.search(r"PIPE_[A-Z0-9]+", key)[0]] += value
        result["mechanisms"] = {"sets": counts.get("pto.set_flag", 0),
                                "waits": counts.get("pto.wait_flag", 0),
                                "PIPE_ALL": barriers.pop("PIPE_ALL", 0),
                                "named": dict(barriers)}
        result["event_ids_by_direction"] = metrics["event_ids_by_direction"]
        result["status_attributes"] = [
            {k: v for k, v in attrs(op).items() if k.startswith("pto.insert_sync.")}
            for op in children(module.operation) if op.name == "func.func"
        ]
        return result


def pass_entry(stderr):
    chunks = stderr.split("// -----// IR Dump Before PTOInsertSync (pto-insert-sync) //----- //")
    if len(chunks) != 2:
        raise ValueError("expected exactly one InsertSync function invocation")
    # The top-level function closes at column zero. Native repair witnesses
    # can follow it before the usual debug marker and are not MLIR input.
    end = re.search(r"(?m)^}\s*$", chunks[1])
    if not end:
        raise ValueError("unterminated pass-entry function dump")
    function = chunks[1][:end.end()].strip()
    return 'module attributes {pto.target_arch = "a3"} {\n' + function + '\n}\n'


def execute(argv, directory, kind, timeout):
    code, stdout, stderr, seconds = run(argv, timeout)
    (directory / f"{kind}.stdout").write_text(stdout)
    (directory / f"{kind}.stderr").write_text(stderr)
    return {"argv": argv, "returncode": code, "seconds": seconds}, stderr


def write_diff(path, left, right):
    path.write_text("".join(difflib.unified_diff(
        json.dumps(left, indent=2, sort_keys=True).splitlines(keepends=True),
        json.dumps(right, indent=2, sort_keys=True).splitlines(keepends=True),
        fromfile="original", tofile="revised")))


def payload_cuts(data):
    """Describe cuts before payload/control, ignoring intervening scalar setup.

    This is a placement index, not an asynchronous completion proof.
    """
    nodes = {tuple(p["point"]): p for p in data["payload"] if "op" in p}
    result = []
    for action in data["placements"]:
        point = action["point"]
        candidates = [k for k, op in nodes.items() if k[:-1] == tuple(point[:-1])
            and isinstance(k[-1], int) and k[-1] >= point[-1]
            and (op["op"].startswith("pto.t") or op["op"] in
                 {"scf.for", "scf.if", "scf.yield", "func.return"})]
        anchor = min(candidates, key=lambda k: k[-1]) if candidates else tuple(point)
        result.append({"before": list(anchor), "next_op": nodes.get(anchor, {}).get("op", "end"),
            "op": action["op"], "guards": action.get("guards", ()),
            "attrs": {k: v for k, v in action["attrs"].items() if k != "event_id"}})
    return result


def summarize(output):
    """Re-analyze retained outputs without rerunning any compiler invocation."""
    report = json.loads((output / "results.json").read_text())
    runtime = Path(report["runtime"]["revised"]["python_root"])
    if digest(runtime / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so") != report["runtime"]["revised"]["native_sha256"]:
        raise ValueError("measurement parser no longer matches the recorded revised binary")
    sys.path.insert(0, report["runtime"]["revised"]["python_root"])
    checkpoint = {k: v for k, v in report.items() if k != "rows"}
    checkpoint["evidence_directory"] = str(output)
    checkpoint["validation"] = {"compiler_emissions": 32, "native_diagnostic_runs": 8,
        "summed_process_seconds": sum(run["seconds"] for row in report["rows"].values()
            for arm in ("original", "revised") for run in row[arm]["runs"].values()),
        "device_execution": "not-run"}
    checkpoint["rows"] = {}
    checkpoint["previous_analysis_failures"] = report.get("failures", [])
    # Projection-only failures are recomputed below, with the current guarded
    # sync-control projection. Preserve the original report as an audit trail.
    checkpoint["failures"] = [failure for failure in report.get("failures", [])
                               if not failure.endswith(": non-sync identity check failed")]
    lines = ["# Qwen A3 additions: original versus shared-generation InsertSync", "",
             "Static inventories. Sets, waits, named barriers and PIPE_ALL remain separate.", "",
             "| Kernel | Original set / wait | Revised set / wait | Named barriers (both) | PIPE_ALL original → revised |",
             "| --- | ---: | ---: | --- | ---: |"]
    for case in CASES:
        row = report["rows"][case]
        compact = checkpoint["rows"][case] = {}
        if any(row[a]["status"] != "pass" for a in ("original", "revised")):
            compact.update(row)
            lines.append(f"| {case} | compile failure | compile failure | unavailable | unavailable |")
            continue
        for arm in ("original", "revised"):
            directory = output / case / arm
            row[arm]["analysis"] = analyze(directory / "output.pto")
            data = row[arm]["analysis"]
            compact[arm] = {k: v for k, v in row[arm].items() if k not in ("analysis", "before")}
            compact[arm]["analysis"] = {k: v for k, v in data.items()
                if k not in ("payload", "placements", "allocations", "views", "abi")}
            compact[arm]["identities"] = {k: {"sha256": hashed(data[k]), "records": len(data[k])}
                for k in ("payload", "placements", "allocations", "views", "abi")}
        left, right = (row[a]["analysis"] for a in ("original", "revised"))
        for key in ("payload", "allocations", "views", "abi"):
            row["comparison"][key + "_identical"] = left[key] == right[key]
            if not row["comparison"][key + "_identical"]:
                raise ValueError(f"{case}: {key} changed")
        native = analyze(output / case / "revised/native.pto")
        row["comparison"]["native_diagnostic_plan_matches_cli"] = all(
            native[k] == right[k] for k in ("mechanisms", "placements"))
        if not row["comparison"]["native_diagnostic_plan_matches_cli"]:
            raise ValueError(f"{case}: native diagnostic plan differs from CLI output")
        def bag(data, ignore_key):
            return Counter(json.dumps({**p, "attrs": {
                k: v for k, v in p["attrs"].items() if not (ignore_key and k == "event_id")}},
                sort_keys=True) for p in data["placements"])
        row["comparison"]["same_boundary_inventory_including_keys"] = bag(left, False) == bag(right, False)
        row["comparison"]["same_boundary_inventory_ignoring_keys"] = bag(left, True) == bag(right, True)
        compact["comparison"] = row["comparison"]
        cuts = {a: payload_cuts(row[a]["analysis"]) for a in ("original", "revised")}
        bags = {a: Counter(json.dumps(p, sort_keys=True) for p in cuts[a]) for a in cuts}
        compact["payload_cut_changes"] = {
            "removed": [json.loads(x) for x in (bags["original"] - bags["revised"]).elements()],
            "added": [json.loads(x) for x in (bags["revised"] - bags["original"]).elements()],
        }
        save(output / case / "payload-cut-changes.json", compact["payload_cut_changes"])
        for key in ("payload", "allocations", "views", "abi", "placements"):
            write_diff(output / case / f"{key}.diff", left[key], right[key])
        a, b = left["mechanisms"], right["mechanisms"]
        named = ", ".join(f"{p.removeprefix('PIPE_')}={n}" for p, n in a["named"].items()) or "none"
        if a["named"] != b["named"]:
            named += " → " + str(b["named"])
        lines.append(f"| {case} | {a['sets']} / {a['waits']} | {b['sets']} / {b['waits']} | {named} | {a['PIPE_ALL']} → {b['PIPE_ALL']} |")
    save(output / "results.json", report)
    save(output / "checkpoint.json", checkpoint)
    (output / "counts.md").write_text("\n".join(lines) + "\n")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-python-root", type=Path, required=True)
    parser.add_argument("--revised-python-root", type=Path, required=True)
    parser.add_argument("--pto-test-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--analyze-existing", action="store_true", help="reuse existing outputs; no compilation")
    args = parser.parse_args()
    if args.analyze_existing:
        summarize(args.output.resolve())
        return 0
    repo = Path(__file__).resolve().parents[4]
    source_root = repo / "test/samples/Qwen3DecodeA3"
    manifest_path = Path(__file__).with_name("qwen-additions-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    expected = {row["case_id"]: row for row in manifest["cases"]}
    if set(expected) != set(CASES):
        raise ValueError("frozen manifest has a different population")
    for name, relative in CASES.items():
        source = source_root / "kernels" / relative
        if str(source.relative_to(repo)) != expected[name]["source"] or digest(source) != expected[name]["sha256"]:
            raise ValueError(f"frozen source identity mismatch: {name}")
        if digest(repo / expected[name]["golden"]) != expected[name]["golden_sha256"]:
            raise ValueError(f"frozen golden identity mismatch: {name}")
    for entry in manifest["support_files"]:
        if digest(repo / entry["path"]) != entry["sha256"]:
            raise ValueError(f"frozen support identity mismatch: {entry['path']}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.revised_python_root.resolve()))
    report = {"arch": "a3", "level": "level3", "device_execution": "not-run",
              "gm_contract": "distinct GM argument allocations for both arms; original legacy default",
              "revised_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
              "revised_build_note": "current worktree build; separate one-shot handoff experiment disabled",
              "manifest_sha256": digest(manifest_path),
              "runtime": {}, "inputs": {}, "rows": {}, "failures": []}
    patch = subprocess.check_output(["git", "-C", str(repo), "diff", "--", "include", "lib", "tools", "python"])
    (output / "revised-worktree.patch").write_bytes(patch)
    (output / "manifest.json").write_bytes(manifest_path.read_bytes())
    report["revised_worktree_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    report["optimizer_sha256"] = digest(args.pto_test_opt)
    for arm, runtime in (("original", args.original_python_root), ("revised", args.revised_python_root)):
        runtime = runtime.resolve()
        report["runtime"][arm] = {"python_root": str(runtime),
            "native_sha256": digest(runtime / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so"),
            "cli_sha256": digest(runtime / "ptoas/_cli.py")}
    if report["runtime"]["original"]["native_sha256"] != "0fa0a43290cb0a43f48755ded11fe9879323698b86ea8aa63cf127bb3844b7b3":
        raise ValueError("original compiler does not match the recorded 7e2ec3e29 baseline")
    report["original_source"] = "7e2ec3e29420e297dcf5b3c59ba4d90841464821"
    for case, relative in CASES.items():
        directory = output / case
        directory.mkdir()
        source = source_root / "kernels" / relative
        frozen = directory / "input.pto"
        frozen.write_bytes(source.read_bytes())
        input_info = analyze(frozen)
        if input_info["placements"]:
            raise ValueError(f"{case}: input contains synchronization")
        report["inputs"][case] = {"source": str(source.relative_to(repo)), "sha256": digest(source),
            "golden": str((source_root / f"{case}_golden.py").relative_to(repo))}
        case_row = report["rows"][case] = {}
        for arm in ("original", "revised"):
            dest = directory / arm
            dest.mkdir()
            row = case_row[arm] = {"status": "pass", "runs": {}}
            prefix = [sys.executable, "-c", SERIAL_DRIVER, report["runtime"][arm]["python_root"],
                      "--pto-arch=a3", "--pto-level=level3", "--enable-insert-sync"]
            if arm == "revised":
                prefix += REVISED_FLAGS
            for kind in ("pto", "cpp"):
                argv = prefix.copy()
                if kind == "pto":
                    argv += ["--emit-pto-ir", "--pto-insert-sync-debug=1",
                             "--mlir-print-ir-before=pto-insert-sync", "--mlir-print-ir-after=pto-insert-sync"]
                argv += [str(frozen), "-o", str(dest / f"output.{kind}")]
                result, stderr = execute(argv, dest, kind, args.timeout)
                row["runs"][kind] = result
                if result["returncode"] != 0:
                    row["status"] = "compile-failure"
                    report["failures"].append(f"{case}/{arm}/{kind}: compile failure")
                    continue
                result["output_sha256"] = digest(dest / f"output.{kind}")
                if kind == "pto":
                    row["participation"] = participation(stderr)
                    if not row["participation"]["pass_invoked"] or any(
                            f["explicit_sync_bypass"] for f in row["participation"]["functions"]):
                        raise ValueError(f"{case}/{arm}: automatic insertion not established")
                    (dest / "before.pto").write_text(pass_entry(stderr))
                    row["analysis"] = analyze(dest / "output.pto")
                    row["before"] = analyze(dest / "before.pto")
            print(f"{case}/{arm}: {row['status']}", flush=True)
        if all(case_row[a]["status"] == "pass" for a in ("original", "revised")):
            left, right = (case_row[a]["analysis"] for a in ("original", "revised"))
            checks = {key + "_identical": left[key] == right[key] for key in ("payload", "allocations", "views", "abi")}
            checks["pass_entry_identical"] = case_row["original"]["before"]["payload"] == case_row["revised"]["before"]["payload"]
            identity_ok = all(checks.values())
            case_row["comparison"] = checks
            for key in ("payload", "allocations", "views", "abi", "placements"):
                write_diff(directory / f"{key}.diff", left[key], right[key])
            case_row["comparison"]["placement_identical"] = left["placements"] == right["placements"]
            case_row["comparison"]["cpp_identical"] = digest(directory / "original/output.cpp") == digest(directory / "revised/output.cpp")
            if not identity_ok:
                report["failures"].append(f"{case}: non-sync identity check failed")
            native_argv = [str(args.pto_test_opt.resolve()), str(directory / "revised/before.pto"),
                "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false",
                "--pto-insert-sync=buffer-generations=true defer-same-pipe=true "
                "mmad-chains=true gm-alias=assume-disjoint-arguments effect-coverage=report",
                "-o", str(directory / "revised/native.pto")]
            result, diagnostic = execute(native_argv, directory / "revised", "native", args.timeout)
            case_row["revised"]["runs"]["native"] = result
            case_row["revised"]["diagnostics"] = [line.split("remark: ", 1)[1]
                for line in diagnostic.splitlines() if "remark: " in line]
            if result["returncode"] != 0:
                report["failures"].append(f"{case}: native diagnosis failed")
        save(output / "results.json", report)
    report["binaries_unchanged"] = digest(args.pto_test_opt) == report["optimizer_sha256"] and all(
        digest(Path(data["python_root"]) / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so") == data["native_sha256"]
        for data in report["runtime"].values())
    if not report["binaries_unchanged"]:
        report["failures"].append("compiler binary changed during campaign")
    save(output / "results.json", report)
    if not report["failures"]:
        summarize(output)
    print(f"8 cases; {len(report['failures'])} failures; results: {output}")
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
