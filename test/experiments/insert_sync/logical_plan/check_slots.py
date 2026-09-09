#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native rotating-storage requirements, emitted ordering and readiness gate.

The bounded oracle interprets original SSA selectors and actual slot addresses,
then enumerates conflicting concrete accesses independently of native selection.
libisl challenges membership in the exported symbolic requirements. Unknown or
out-of-range selectors must retain original may-conflicts. This is finite
semantic/command evidence, not numerical execution or device qualification.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from check_relations import ISL, isl_text
from compare_boundaries import Boundaries
from measure import children, fingerprint, replay, static_metrics
from observations import project

HERE = Path(__file__).resolve().parent
MANDATORY = {"rotating_d2", "rotating_mask_d2", "rotating_explicit_base_d2"}
HAZARDS = (("RAW", "writes", "reads"), ("WAR", "reads", "writes"), ("WAW", "writes", "writes"))


def walk(op):
    yield op
    for child in children(op):
        yield from walk(child)


def overlap(left, right):
    return left["scope"] == right["scope"] and max(left["begin"], right["begin"]) < min(
        left["begin"] + left["bytes"], right["begin"] + right["bytes"])


class OriginalAccesses:
    def __init__(self, function, data, spec, flip):
        self.options, self.flip = spec["options"], flip
        self.count = self.options.get("count", 2)
        self.maps = {"a": [512 * i for i in range(self.count)],
                     "b": self.options.get("mapping", [512 * i for i in range(self.count)])}
        self.phases = [op for op in walk(function) if op.name in {"pto.tload", "pto.tabs", "pto.tadd"}]
        names = ["pto.tload", "pto.tload", "pto.tadd" if self.options.get("common_consumer") else "pto.tabs", "pto.tabs"]
        if self.options.get("extra_reader"):
            names.append("pto.tabs")
        assert [op.name for op in self.phases] == names
        assert [p["op"] for p in data["phases"]] == names
        assert len([op for op in walk(function) if op.name == "scf.for"]) == 1
        self.phase_ids = {op: p for p, op in enumerate(self.phases)}
        self.native = {a["id"]: a for a in data["accesses"]}
        assert len(self.native) == len(data["accesses"])
        self.roles, self.effects = {}, {}
        for p, phase in enumerate(data["phases"]):
            assert phase["id"] == p
            reads = [] if p < 2 else (["a", "b"] if p == 2 and self.options.get("common_consumer") else
                                    ["b" if p == 3 else "a"])
            writes = ["a" if p == 0 else "b"] if p < 2 else ["out" + str(p - 2)]
            self.effects[p] = {}
            for kind, wanted in (("reads", reads), ("writes", writes)):
                ids = [a for a in phase[kind] if self.native[a]["scope"] == "vec"]
                assert len(ids) == len(wanted), (p, kind, ids, wanted)
                self.effects[p][kind] = ids
                for aid, role in zip(ids, wanted):
                    assert self.roles.setdefault(aid, role) == role
                    access = self.native[aid]
                    assert access["physical"] and not access["unknown_range"] and int(access["allocation_bytes"]) == 512
                    assert list(map(int, access["base_addresses"])) == self.bases(role), (aid, role, access)
        self.allocations, self.events = {}, []
        self.iteration = -1

    def bases(self, role):
        return self.maps[role] if role in self.maps else [4096 + 512 * int(role[3:])]

    def exact(self, aid):
        role = self.roles[aid]
        return not (role in ("a", "b") and self.options.get("out_of_range")) and not (
            role == "a" and self.options.get("unknown"))

    def concrete(self, role, iteration):
        if role not in self.maps:
            return self.bases(role)[0]
        modulus = 3 if self.options.get("out_of_range") else self.count
        selector = (iteration + (role == "b")) % modulus
        if role == "a" and self.options.get("unknown") and not self.flip:
            selector = (iteration + 1) % modulus
        if self.options.get("out_of_range", 0) < 0:
            selector -= 1
        if not 0 <= selector < self.count:
            selector = 0
        return self.maps[role][selector]

    def observe(self, op, point, signature):
        if op.name == "pto.alloc_tile":
            assert str(op.results[0].type) == "!pto.tile_buf<vec, 16x16xf16>"
            base = signature[1][0]
            assert isinstance(base, int)
            self.allocations[fingerprint(signature)] = {"scope": "vec", "begin": base, "bytes": 512}
        if op not in self.phase_ids:
            return
        p = self.phase_ids[op]
        if p == 0:
            self.iteration += 1
        event = {"phase": p, "iv": self.iteration, "lane": "PIPE_MTE2" if p < 2 else "PIPE_V",
                 "reads": [], "writes": []}
        # Complete ins operands followed by one outs operand. GM load input is
        # outside this local-memory oracle, not silently called independent.
        local = [v for v, operand in zip(signature[1], op.operands) if "!pto.tile_buf<vec," in str(operand.type)]
        ids = self.effects[p]["reads"] + self.effects[p]["writes"]
        assert len(local) == len(ids)
        for index, (aid, value) in enumerate(zip(ids, local)):
            region = value["local_tile"] if isinstance(value, dict) else self.allocations[value]
            assert region == {"scope": "vec", "begin": self.concrete(self.roles[aid], self.iteration), "bytes": 512}
            kind = "reads" if index < len(self.effects[p]["reads"]) else "writes"
            event[kind].append((aid, region))
        self.events.append(event)

    def requirements(self):
        expected, actual, universe = defaultdict(set), [], set()
        for p, left in self.effects.items():
            for q, right in self.effects.items():
                for kind, src, dst in HAZARDS:
                    for a in left[src]:
                        for b in right[dst]:
                            universe.add((kind, p, q, a, b))
        for j, right in enumerate(self.events):
            for i, left in enumerate(self.events[:j]):
                for kind, src, dst in HAZARDS:
                    for a, first in left[src]:
                        for b, second in right[dst]:
                            real = overlap(first, second)
                            coarse = any(max(x, y) < min(x + 512, y + 512)
                                         for x in self.bases(self.roles[a]) for y in self.bases(self.roles[b]))
                            if real:
                                actual.append((i, j, kind))
                            if (real if self.exact(a) and self.exact(b) else coarse):
                                expected[kind, left["phase"], right["phase"], a, b].add((left["iv"], right["iv"]))
        return expected, actual, universe

    def evidence(self):
        expected, actual, _ = self.requirements()
        counts = Counter(real_conflicts=len(actual), retained_occurrences=sum(map(len, expected.values())))
        coarse = 0
        for j, target in enumerate(self.events):
            for source in self.events[:j]:
                for kind, read_kind, write_kind in HAZARDS:
                    for a, first in source[read_kind]:
                        for b, second in target[write_kind]:
                            if any(max(x, y) < min(x + 512, y + 512)
                                   for x in self.bases(self.roles[a]) for y in self.bases(self.roles[b])):
                                coarse += 1
                            if overlap(first, second):
                                width = min(first["begin"] + first["bytes"], second["begin"] + second["bytes"]) - max(first["begin"], second["begin"])
                                counts["partial_byte_conflicts"] += width < min(first["bytes"], second["bytes"])
                if source["phase"] == 1 and target["phase"] == 2 and source["iv"] == target["iv"]:
                    required = any(overlap(a, b) for _, a in source["writes"] for _, b in target["reads"])
                    counts["second_readiness_required" if required else "second_readiness_absent"] += 1
                if source["phase"] == 4 and target["iv"] == source["iv"] + 1 and target["phase"] < 2:
                    required = any(overlap(a, b) for _, a in source["reads"] for _, b in target["writes"])
                    counts["extra_reader_reuse" if required else "extra_reader_other_slot_absent"] += 1
        counts["omitted_may_conflicts"] = coarse - counts["retained_occurrences"]
        counts["conservative_extra"] = counts["retained_occurrences"] - counts["real_conflicts"]
        counts["first_readers"] = sum(event["phase"] == 2 for event in self.events)
        counts["skipped_first_readers"] = sum(event["phase"] == 0 for event in self.events) - counts["first_readers"]
        assert counts["omitted_may_conflicts"] >= 0 and counts["conservative_extra"] >= 0
        return counts


def check_evidence(spec, evidence):
    options = spec["options"]
    assert evidence["real_conflicts"] > 0, (spec["name"], evidence)
    if options.get("unknown") or options.get("out_of_range"):
        assert evidence["conservative_extra"] > 0, ("unknown occurrence precision erased its conservative obligations", evidence)
    elif not options.get("common_consumer") and not options.get("mapping"):
        assert evidence["omitted_may_conflicts"] > 0, ("native slot precision was never exercised", evidence)
    if options.get("mapping") or options.get("common_consumer"):
        assert evidence["second_readiness_required"] > 0, ("adverse second-preload dependency was not exercised", evidence)
    if options.get("partial_overlap"):
        assert evidence["partial_byte_conflicts"] > 0, ("no genuinely partial byte overlap", evidence)
    if options.get("extra_reader"):
        assert evidence["extra_reader_reuse"] > 0 and evidence["extra_reader_other_slot_absent"] > 0, evidence
    if options.get("skipped_reader"):
        assert evidence["first_readers"] > 0 and evidence["skipped_first_readers"] > 0, evidence


def check_domains(isl, data, oracle, n, flip):
    assert data["export_complete"] and data["points"]
    symbols = data["points"][0]["s"]
    fixed = oracle.options.get("fixed_trip_count")
    assert symbols == ((1 if oracle.options.get("known_parameter") else 0) if fixed else
                       (2 if oracle.options.get("skipped_reader") else 1)), symbols
    params = ",".join(f"p{s}" for s in range(symbols))
    bindings = ([f"p0={n}"] if symbols else []) + ([f"p1={flip}"] if symbols == 2 else [])
    extent = fixed or n
    empty = isl.map(f"[{params}] -> {{[p,i] -> [q,j]: false}}")
    native = {}
    for item in data["requirements"]:
        if item["kind"] == "retirement":
            continue
        key = tuple(item[k] for k in ("kind", "source", "target", "source_access", "target_access"))
        value = isl.map(isl_text(item["occurrences"]))
        native[key] = native.get(key, empty) | value
    expected, actual, universe = oracle.requirements()
    assert set(native) <= universe, set(native) - universe
    for key in universe:
        _, p, q, _, _ = key
        conditions = " and ".join([*bindings, f"0<=i<{extent}", f"0<=j<{extent}"])
        scope = isl.map(f"[{params}] -> {{[{p},i] -> [{q},j]: {conditions}}}")
        pairs = expected.get(key, set())
        suffix = (": " + " and ".join(bindings)) if bindings else ""
        clauses = "; ".join(f"[{p},{i}] -> [{q},{j}]{suffix}" for i, j in sorted(pairs))
        wanted = isl.map(f"[{params}] -> {{{clauses}}}") if pairs else empty
        observed = native.get(key, empty) & scope
        assert observed.equal(wanted), ("slot requirement mismatch", n, flip, key, str(observed), str(wanted))
    return actual, len(universe), sum(map(len, expected.values()))


def inspect(data, source, spec, isl, ir, pto, require_quality):
    emitted = 'module attributes {pto.target_arch = "a3"} {\n' + data["emitted_ir"] + '\n}'
    rows, independent, broadened, evidence = [], 0, 0, Counter()
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        original, candidate = ir.Module.parse(source), ir.Module.parse(emitted)
        projections = [project(m) for m in (original, candidate)]
        for key in ("payload", "allocations", "views", "abi"):
            assert projections[0][key] == projections[1][key], key
        functions = [next(op for op in children(m.operation) if op.name == "func.func") for m in (original, candidate)]
        for n in spec["replay_bounds"]:
            for flip in spec["boolean_values"] or [None]:
                arguments = ["source", n] + ([flip] if flip is not None else [])
                oracle = OriginalAccesses(functions[0], data, spec, flip)
                original_metric = replay(functions[0], arguments, observer=oracle.observe)
                evidence.update(oracle.evidence())
                if require_quality:
                    actual, comparisons, requirements = check_domains(isl, data, oracle, n, flip)
                else:
                    _, actual, _ = oracle.requirements()
                    comparisons, requirements = 0, len(actual)
                observed = Boundaries()
                metric = replay(functions[1], arguments, observer=observed.observe)
                assert original_metric["payload_sha256"] == metric["payload_sha256"]
                assert len(observed.before) == len(oracle.events) and not observed.tokens
                for left, right, kind in actual:
                    lane = oracle.events[left]["lane"]
                    assert observed.before[right]["completed"].get(lane, -1) >= left, ("missing actual local order", n, flip, left, right, kind)
                for lane, last in observed.issued.items():
                    assert observed.drained.get(lane, -1) >= last, ("missing terminal completion", lane, last)
                for target, event in enumerate(oracle.events):
                    if event["phase"] != 2:
                        continue
                    loads = [(i, e) for i, e in enumerate(oracle.events[:target]) if e["iv"] == event["iv"] and e["phase"] < 2]
                    assert len(loads) == 2
                    first, second = loads[0][0], loads[1][0]
                    needs_second = any(overlap(a, b) for _, a in loads[1][1]["writes"] for _, b in event["reads"])
                    if not needs_second and not spec["options"].get("unknown") and not spec["options"].get("out_of_range"):
                        independent += 1
                        prefix = observed.before[target]["completed"].get("PIPE_MTE2", -1)
                        broadened += prefix >= second
                        if require_quality:
                            assert first <= prefix < second, ("independent second preload recaptured", n, flip, event["iv"], prefix, first, second)
                rows.append({"bound": n, "flip": flip, "physical_occurrences": len(oracle.events),
                             "symbolic_relation_comparisons": comparisons, "required_occurrence_pairs": requirements,
                             "counts": metric["counts"], "scalar_counts": metric["scalar_counts"],
                             "scalar_steps": metric["scalar_steps"], "completion_prefixes": observed.before})
        check_evidence(spec, evidence)
        return {"static": static_metrics(candidate.operation), "scenarios": rows, "evidence": dict(evidence),
                "independent_first_readers": independent, "broadened_first_readers": broadened}


def inspect_discovery(data, source, spec, isl, ir, pto):
    """A failed realization may still export useful, independently checked facts.

    These checks never turn an unavailable strict construction into effectiveness.
    """
    rows, evidence = [], Counter()
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        module = ir.Module.parse(source)
        function = next(op for op in children(module.operation) if op.name == "func.func")
        for n in spec["replay_bounds"]:
            for flip in spec["boolean_values"] or [None]:
                oracle = OriginalAccesses(function, data, spec, flip)
                replay(function, ["source", n] + ([flip] if flip is not None else []), observer=oracle.observe)
                evidence.update(oracle.evidence())
                _, comparisons, pairs = check_domains(isl, data, oracle, n, flip)
                rows.append({"bound": n, "flip": flip, "relation_comparisons": comparisons,
                             "required_occurrence_pairs": pairs})
    assert rows and any(row["required_occurrence_pairs"] for row in rows)
    check_evidence(spec, evidence)
    return {"scenarios": rows, "evidence": dict(evidence)}


def compare_prefixes(old_result, new_result):
    assert len(old_result["scenarios"]) == len(new_result["scenarios"])
    differences = 0
    for old, new in zip(old_result["scenarios"], new_result["scenarios"]):
        assert (old["bound"], old["flip"]) == (new["bound"], new["flip"])
        assert len(old["completion_prefixes"]) == len(new["completion_prefixes"])
        for target, (before, after) in enumerate(zip(old["completion_prefixes"], new["completion_prefixes"])):
            assert before["lane"] == after["lane"]
            for lane in before["completed"].keys() | after["completed"].keys():
                prior, current = before["completed"].get(lane, -1), after["completed"].get(lane, -1)
                assert current <= prior, ("new mandatory payload completion", old["bound"], target, lane, prior, current)
                differences += current < prior
    assert differences > 0
    return differences


def main():
    if not __debug__:
        raise RuntimeError("Slot acceptance requires assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs/slots")
    parser.add_argument("--case", action="append", help="Restrict probing to named cases; default all twelve")
    parser.add_argument("--baseline-json", type=Path, help="Frozen Step3 rotating_d2 retirement output; no extra compiler invocation")
    parser.add_argument("--mutations", action="store_true")
    parser.add_argument("--focused", action="store_true", help="Small strict binding seeds and fast conservative negatives")
    parser.add_argument("--stop-file", type=Path, help="Finish the current case, then pause before another native invocation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.python_root.resolve()))
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    isl = ISL()
    driver_hash = hashlib.sha256(args.driver.read_bytes()).hexdigest()
    baseline_hash = hashlib.sha256(args.baseline_json.read_bytes()).hexdigest() if args.baseline_json else None
    (args.output / "provenance.json").write_text(json.dumps({
        "driver": str(args.driver.resolve()), "driver_sha256": driver_hash,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_json_sha256": baseline_hash}, indent=2) + "\n")
    specs = json.loads((args.inputs / "manifest.json").read_text())
    if args.focused:
        specs = [spec for spec in specs if spec["options"].get("unknown") or spec["options"].get("out_of_range")]
        for name, options, mandatory in (
            ("slot_guard_binding", {}, True),
            ("slot_guard_known_parameter", {"known_parameter": True}, True),
            ("slot_partial_overlap", {"mapping": [256, 768], "partial_overlap": True}, True),
        ):
            source = args.inputs / (name + ".pto")
            specs.append({"name": name, "file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "options": {"fixed_trip_count": 4, **options}, "mandatory": mandatory,
                          "replay_bounds": [-1, 0, 1, 4] if not options.get("partial_overlap") else [0],
                          "boolean_values": []})
        for spec in specs:
            spec["required_application"] = True
        args.mutations = True
    if args.case:
        assert set(args.case) <= {s["name"] for s in specs}
        specs = [s for s in specs if s["name"] in args.case]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows, commands, paused = [], [], False

    def invoke(path, mutation, label, trace=False):
        discovery = args.output / (label + ".discovery.json")
        command = [str(args.driver.resolve()), str(path.resolve()), mutation, "--discovery-output=" + str(discovery.resolve())]
        start = time.monotonic()
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=90,
                                    env={**env, **({"PTOAS_LOGICAL_TRACE": "1"} if trace else {})})
        except subprocess.TimeoutExpired as error:
            for extension, value in (("stdout.json", error.stdout), ("stderr", error.stderr)):
                text = value.decode() if isinstance(value, bytes) else (value or "")
                (args.output / (label + "." + extension)).write_text(text)
            commands.append({"command": command, "seconds": time.monotonic() - start, "exit": "timeout"})
            (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            facts = json.loads(discovery.read_text()) if discovery.exists() else {}
            assert not facts or facts.get("discovery_only") is True
            return "timeout", {"status": "timeout", "reason": "strict native invocation exceeded 90 seconds",
                               "work": None, "applied": False, "export_complete": False, **facts}
        (args.output / (label + ".stdout.json")).write_text(result.stdout)
        (args.output / (label + ".stderr")).write_text(result.stderr)
        commands.append({"command": command, "seconds": time.monotonic() - start, "exit": result.returncode})
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        assert result.returncode in (0, 1), (label, result.returncode, result.stderr[-4000:])
        return result.returncode, json.loads(result.stdout)

    def evaluate(spec, row):
        path = args.inputs / spec["file"]
        source = path.read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == spec["sha256"], spec["name"]
        code, data = invoke(path, "retirement", spec["name"])
        row.update({"strict_status": data["status"], "reason": data["reason"], "work": data["work"],
                    "applied": data["applied"]})
        if code == "timeout":
            row["validation_status"] = "not-established"
            row["effectiveness"] = "not established; timed-out process supplies no proof or rollback evidence"
            if data.get("export_complete") and data.get("phases") and data.get("points"):
                row["discovery_only"] = inspect_discovery(data, source, spec, isl, ir, pto)
                row["validation_status"] = "discovery-only"
            print(spec["name"], "timed out; continuing remaining cases", flush=True)
            return
        if not data["applied"]:
            assert code == 1 and data["original_preserved"] and not data["emitted_ir"], data
            assert data["status"] in {"unsupported", "analysis-limit", "allocation-failure", "unproved"}, data
            row["effectiveness"] = "not established; no fallback counted"
            if data["export_complete"] and data["phases"] and data["points"]:
                row["discovery_only"] = inspect_discovery(data, source, spec, isl, ir, pto)
            if spec["options"].get("unknown") or spec["options"].get("out_of_range"):
                assert row.get("discovery_only"), ("conservative negative has no checked obligations", row)
            row["validation_status"] = "discovery-only" if row.get("discovery_only") else "not-established"
            print(spec["name"], "strict unavailable:", data["reason"], flush=True)
            return
        assert code == 0 and data["status"] == "applied" and data["invoked"] and data["export_complete"], data
        sidecar = json.loads((args.output / (spec["name"] + ".discovery.json")).read_text())
        assert data["discovery_written"] and sidecar["discovery_only"] and sidecar["export_complete"]
        for field in ("accesses", "phases", "requirements", "points", "orders"):
            assert sidecar[field] == data[field], (spec["name"], "discovery sidecar differs", field)
        row["observations"] = inspect(data, source, spec, isl, ir, pto, True)
        if spec["name"] in MANDATORY or spec.get("mandatory"):
            assert row["observations"]["independent_first_readers"] > 0
        (args.output / (spec["name"] + ".logical.pto")).write_text(
            'module attributes {pto.target_arch = "a3"} {\n' + data["emitted_ir"] + '\n}')
        if spec["name"] == "rotating_d2" and args.baseline_json:
            baseline = json.loads(args.baseline_json.read_text())
            assert baseline["applied"] and baseline["invoked"] and baseline["export_complete"]
            row["baseline"] = inspect(baseline, source, spec, isl, ir, pto, False)
            assert row["baseline"]["broadened_first_readers"] > 0
            assert row["observations"]["broadened_first_readers"] == 0
            row["removed_payload_prefix_requirements"] = compare_prefixes(row["baseline"], row["observations"])
        mutations = {"rotating_d2": ("erase-wait", "swap-loads", "swap-wait-keys"),
                     "slot_guard_binding": ("slot-guard-selector-binding", "slot-guard-unbound-parameter"),
                     "slot_guard_known_parameter": ("slot-guard-known-parameter",)}
        if spec["name"] in mutations and args.mutations:
            row["mutations"] = []
            for mutation in mutations[spec["name"]]:
                status, rejected = invoke(path, mutation, spec["name"] + "." + mutation)
                assert status == 0 and rejected["invoked"] and rejected["changed"]
                assert not rejected["applied"] and rejected["original_preserved"], rejected
                if mutation.startswith("slot-guard-"):
                    assert rejected["counts_preserved"] and rejected["original_scalar_uses_preserved"], rejected
                    assert "original payload" not in rejected["reason"], rejected
                    assert rejected["original_parameter_count"] == int(bool(spec["options"].get("known_parameter")))
                row["mutations"].append({"mutation": mutation, "reason": rejected["reason"]})
        row["validation_status"] = "passed"
        row["accepted"] = True
        print(spec["name"], "strict applied;", row["observations"]["independent_first_readers"], "independent readiness checks", flush=True)
    for spec in specs:
        if args.stop_file and args.stop_file.exists():
            paused = True
            break
        row = {"case": spec["name"], "source_sha256": spec["sha256"], "applied": False, "accepted": False,
               "strict_status": "not-run", "validation_status": "not-established"}
        rows.append(row)
        try:
            evaluate(spec, row)
        except KeyboardInterrupt:
            paused = True
            row["strict_status"] = "interrupted"
            row["validation_status"] = "not-established"
            row["effectiveness"] = "not established; interrupted at user request"
        except Exception as error:
            row["validation_status"] = "failed"
            row["validation_error"] = str(error)
            row["validation_error_type"] = type(error).__name__
            print(spec["name"], "FAILED:", str(error)[-2000:], flush=True)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        if paused:
            break
    trace_refusal, boundary_checks = None, []
    if args.focused and not paused:
        boundary_inputs = [args.inputs / (name + ".pto")
                           for name in ("slot_guard_binding", "slot_guard_known_parameter")]
        negative = (args.inputs / "slot_guard_binding.pto").read_text().replace(
            "%n = arith.constant 4 : index", "%n = arith.constant 4 : index\n    %lower = arith.constant -3 : index\n    %step = arith.constant 2 : index").replace(
            "scf.for %i = %c0 to %n step %c1", "scf.for %i = %lower to %n step %step").replace(
            "%slots[%cur_mod]", "%slots[%c0]").replace("%slots[%next_mod]", "%slots[%c1]")
        # Keep the payload's descriptors fixed: this challenges loop-domain
        # complements at negative/non-unit occurrences, independently of slots.
        negative = "\n".join(line for line in negative.splitlines()
                             if "%cur_mod =" not in line and "%next_mod =" not in line) + "\n"
        path = args.output / "boundary-negative-nonunit.pto"
        path.write_text(negative)
        boundary_inputs.append(path)
        for path in boundary_inputs:
            code, data = invoke(path, "boundary-conditions", path.stem + ".boundary-conditions")
            assert code == 0 and data["boundary_conditions_invoked"] and data["boundary_conditions_passed"], data
            assert data["invoked"] and data["applied"] and data["status"] == "applied", data
            boundary_checks.append({"case": path.stem, "passed": True,
                                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        from check_physical_addresses import cases as address_cases
        source = next(case["source"] for case in address_cases() if case["name"] == "runtime_cross_root")
        path = args.output / "trace-early-refusal.pto"
        path.write_text(source)
        code, data = invoke(path, "none", "trace-early-refusal", trace=True)
        assert code == 1 and not data["applied"] and data["original_preserved"]
        assert data["status"] == "unsupported" and "footprint" in data["reason"], data
        trace_refusal = {"status": data["status"], "reason": data["reason"], "trace_enabled": True}
    assert hashlib.sha256(args.driver.read_bytes()).hexdigest() == driver_hash, "native driver changed during campaign"
    accepted = {row["case"] for row in rows if row["accepted"]}
    mandatory = {spec["name"] for spec in specs
                 if spec["name"] in MANDATORY or spec.get("mandatory") or spec.get("required_application")}
    missing_mandatory = sorted(mandatory - accepted)
    failed = [row["case"] for row in rows if row["validation_status"] == "failed"]
    timed_out = [row["case"] for row in rows if row["strict_status"] == "timeout"]
    checks_passed = not (paused or missing_mandatory or failed or timed_out)
    summary = {"status": "paused" if paused else ("passed" if len(accepted) == len(rows) else ("partial" if checks_passed else "failed")),
               "campaign_complete": not paused, "checks_passed": checks_passed,
               "cases": rows, "applied": sum(r["applied"] for r in rows), "accepted": len(accepted),
               "requested": len(specs), "completed_results": len(rows), "missing_mandatory": missing_mandatory,
               "not_run": [spec["name"] for spec in specs if spec["name"] not in {row["case"] for row in rows}],
               "validation_failed": failed, "timed_out": timed_out,
               "mandatory_applied": sorted(mandatory & accepted),
               "driver_sha256": driver_hash, "baseline_json_sha256": baseline_hash, "device": "not run",
               "traced_early_refusal": trace_refusal,
               "boundary_conditions": boundary_checks,
               "trust_boundary": "bounded original-input scalar/descriptor replay and libisl membership; no device or exact last-definition claim"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
