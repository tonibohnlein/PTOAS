#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Discover/freeze and compare A2/A3 inputs without conflating preparation stages.

Discovery is deliberately a RAW compatibility population, not a production
admission claim. For production admission, supply a manifest of snapshots
captured immediately before InsertSync after the real preparation pipeline.
Every arm consumes the identical immutable snapshot. No legacy seed is used.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCHEMA = "oahs.s7.corpus.v1"
LOCK_SCHEMA = "oahs.s7.corpus-lock.v1"
PORTABLE_ROOT = "$REPO"
ARMS = ("existing", "structured", "structured-hardware")
CONTRACT = "a2a3-mmad-acc-v1"
REGRESSIONS = ("q_proj", "online_softmax", "qk_matmul", "hc_head_linear", "hc_pre_linear", "weights_proj")
STAGES = ("raw-addressed-compatibility", "prepared-pass-entry", "historical-archive")
CLASSES = ("production", "focused", "intentional-refusal", "historical")
FROZEN_SCHEMA = "oahs.frozen-corpus.record.v1"
FROZEN_LOCK_SCHEMA = "oahs.frozen-corpus.lock.v1"
FROZEN_RECORDS = 363
SHA256 = re.compile(r"[0-9a-f]{64}")
FROZEN_ROOT_KINDS = {
    "PTOAS": "ptoas-source",
    "PYPTO_GENERATED": "generated-pto-snapshot",
}
FROZEN_ARMS = {
    "baseline": {
        "label": "conservative structured planner",
        "argv": [
            "--mlir-disable-threading",
            "--pto-insert-sync=planner=structured structured-precision=false "
            "logical-work-budget=0 gm-alias={gm_contract}",
            "{input}",
        ],
    },
    "current": {
        "label": "bc95 deferred-wrap production baseline",
        "argv": [
            "--mlir-disable-threading",
            "--pto-insert-sync=planner=composition structured-precision=true "
            "logical-work-budget=0 gm-alias={gm_contract}",
            "{input}",
        ],
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def discover(repo, all_samples=False, portable=False):
    """Exact filenames/contents, no inference of missing inputs or stage fixes."""
    repo = Path(repo).resolve()
    choices = {}
    for path in sorted((repo / "test/samples").rglob("*.pto")):
        if not all_samples and path.stem not in REGRESSIONS:
            continue
        text = path.read_text()
        arch = re.search(r'pto\.target_arch\s*=\s*"(a2a3|a2|a3)"', text)
        if arch:
            choices[path] = ("production", "raw-addressed-compatibility")
    # Preserve the existing seven immutable benchmark sources when available.
    frozen = repo / "test/experiments/insert_sync/logical_plan/checkpoint/manifest.json"
    if frozen.exists():
        for case in json.loads(frozen.read_text())["cases"]:
            path = repo / case["source"]
            if not path.is_file() or digest(path.read_bytes()) != case["sha256"]:
                raise ValueError("missing/changed frozen population source: " + case["case_id"])
            choices[path] = ("production" if "test/samples/" in case["source"] else "focused",
                             "raw-addressed-compatibility")
    for path in sorted((repo / "test/experiments/insert_sync/logical_plan/hardware_inputs").glob("*.pto")):
        choices[path] = ("focused", "raw-addressed-compatibility")
    archives = sorted((repo / "test/experiments/insert_sync/logical_plan/archive/historical_gemm").glob("*.pto"))
    for path in archives:
        choices[path] = ("historical", "historical-archive")
    cases = [dict(id=str(p.relative_to(repo)), path=str(p.relative_to(repo)), sha256=digest(p.read_bytes()),
                  classification=c, stage=s, gm_contract="may-alias")
             for p, (c, s) in sorted(choices.items())]
    present = {p.stem for p in choices}
    missing = [name for name in REGRESSIONS if name not in present]
    if not archives:
        missing.append("historical_gemm: run the pinned recover_historical_gemm.py first")
    return dict(schema=SCHEMA, root=PORTABLE_ROOT if portable else str(repo),
                discovery=dict(all_a2a3=bool(all_samples),
                    preparation="identical source bytes; bind module target a2a3 to a3 only"),
                cases=cases, missing_requested=missing,
                note="Raw compatibility only. Classify real pre-pass snapshots separately. GM defaults to may-alias; change only for a qualified caller contract.")


def validate_manifest(manifest):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("wrong corpus schema")
    ids = set()
    for case in manifest["cases"]:
        if not case.get("id") or case["id"] in ids:
            raise ValueError("missing/duplicate input identity")
        ids.add(case["id"])
        if case.get("stage") not in STAGES or case.get("classification") not in CLASSES:
            raise ValueError("explicit stage/classification required")
        if case.get("gm_contract") not in ("may-alias", "assume-disjoint-arguments"):
            raise ValueError("unknown GM contract; no all-access nonalias mode")
        if not re.fullmatch(r"[0-9a-f]{64}", case.get("sha256", "")):
            raise ValueError("input fingerprint required")
        if case["stage"] == "prepared-pass-entry" and not case.get("preparation"):
            raise ValueError("prepared snapshot requires pipeline/build provenance")
    if not ids:
        raise ValueError("empty corpus is not an admission study")


def manifest_digest(manifest):
    return digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


def verify_discovery(manifest, repo):
    """Require the checked-in identity population to equal fresh discovery."""
    validate_manifest(manifest)
    discovery = manifest.get("discovery")
    if not isinstance(discovery, dict) or not isinstance(discovery.get("all_a2a3"), bool):
        raise ValueError("frozen manifest requires exact discovery parameters")
    actual = discover(repo, discovery["all_a2a3"], portable=True)
    if manifest != actual:
        expected = {case["id"]: case["sha256"] for case in manifest["cases"]}
        found = {case["id"]: case["sha256"] for case in actual["cases"]}
        missing = sorted(set(expected) - set(found))
        added = sorted(set(found) - set(expected))
        changed = sorted(case for case in set(expected) & set(found)
                         if expected[case] != found[case])
        raise ValueError("frozen corpus differs from discovery: " + json.dumps(
            dict(missing=missing, added=added, changed=changed,
                 expected_missing=manifest.get("missing_requested", []),
                 actual_missing=actual.get("missing_requested", []))))
    return dict(inputs=len(actual["cases"]), manifest_sha256=manifest_digest(manifest),
                missing_requested=actual["missing_requested"])


def compact_lock(result, manifest, implementation="same source tree as this lock"):
    """Stable evidence only: no host paths, commands, timings, or binary hashes."""
    expected = {case["id"]: case["sha256"] for case in manifest["cases"]}
    observed = {case["id"]: case["sha256"] for case in result["cases"]}
    if observed != expected:
        raise ValueError("result population does not match the frozen manifest")
    rows = []
    refusal_reasons = sorted({item.get("reason", "")
        for case in result["cases"] for item in case["arms"].values()
        if item["status"] != "applied"})
    reason_ids = {reason: index for index, reason in enumerate(refusal_reasons)}
    for case in result["cases"]:
        arms = {}
        for arm in ARMS:
            item = case["arms"][arm]
            evidence = dict(status=item["status"])
            if item["status"] == "applied":
                mechanisms = item["mechanisms"]
                evidence["mechanism_counts"] = {
                    key: mechanisms[key] for key in
                    ("sets", "waits", "named_barriers", "pipe_all")}
                evidence["mechanisms_sha256"] = digest(json.dumps(
                    mechanisms, sort_keys=True, separators=(",", ":")).encode())
                evidence["output_sha256"] = item["output_sha256"]
            else:
                evidence["reason_id"] = reason_ids[item.get("reason", "")]
            arms[arm] = evidence
        rows.append(dict(id=case["id"], source_sha256=case["sha256"],
                         pre_sync_ir_sha256=case["input_sha256"],
                         adaptations=case["adaptations"], arms=arms))
    aggregate = {stage: dict(pools=data["pools"],
        existing_successful_hardware_refusals=data["existing_successful_hardware_refusals"])
        for stage, data in result["summary"].items()}
    return dict(schema=LOCK_SCHEMA, manifest_sha256=manifest_digest(manifest),
                implementation=implementation,
                preparation=manifest.get("discovery", {}).get("preparation"),
                refusal_reasons=refusal_reasons,cases=rows, summary=aggregate,
                missing_requested=result["missing_requested"], device="NOT_RUN")


def validate_lock(lock, manifest):
    validate_manifest(manifest)
    if lock.get("schema") != LOCK_SCHEMA or \
       lock.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("corpus lock does not name the exact manifest")
    if not isinstance(lock.get("implementation"), str) or not lock["implementation"]:
        raise ValueError("corpus lock requires implementation provenance")
    expected = {case["id"]: case["sha256"] for case in manifest["cases"]}
    rows = lock.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("corpus lock population differs from manifest")
    reasons=lock.get("refusal_reasons")
    if not isinstance(reasons,list) or any(not isinstance(reason,str) for reason in reasons):
        raise ValueError("corpus lock refusal dictionary is invalid")
    seen = set()
    for row in rows:
        if row.get("id") in seen or expected.get(row.get("id")) != row.get("source_sha256"):
            raise ValueError("corpus lock input identity differs from manifest")
        seen.add(row["id"])
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("pre_sync_ir_sha256", "")):
            raise ValueError("corpus lock requires a pre-sync IR hash")
        if set(row.get("arms", {})) != set(ARMS) or any(
                row["arms"][arm].get("status") not in
                ("applied", "refused", "measurement-failed") for arm in ARMS):
            raise ValueError("corpus lock has an invalid arm result")
        for arm in ARMS:
            item=row["arms"][arm]
            if item["status"]!="applied" and (not isinstance(item.get("reason_id"),int) or
                                               item["reason_id"]<0 or item["reason_id"]>=len(reasons)):
                raise ValueError("corpus lock has an invalid refusal reference")
    if seen != set(expected):
        raise ValueError("corpus lock omits a manifest input")
    return dict(inputs=len(rows), manifest_sha256=lock["manifest_sha256"])


def prepare_bytes(data, stage):
    # This is target selection only. Do not repair geometry, inject temporaries,
    # erase manual synchronization, or change payload/control to admit an input.
    text = data.decode()
    changes = []
    if stage == "prepared-pass-entry":
        if re.search(r'pto\.target_arch\s*=\s*"a2a3"', text):
            raise ValueError("prepared snapshot must already select a concrete target")
    else:
        text, n = re.subn(r'(pto\.target_arch\s*=\s*)"a2a3"', r'\1"a3"', text)
        if n:
            changes.append("explicit target binding a2a3 -> a3; no payload edits")
    return text.encode(), changes


def composition_outcome(returncode, stdout, stderr):
    """Canonical stable fields shared with the composition corpus contract."""
    status = "pass" if returncode == 0 else "refused" if returncode > 0 else "crashed"
    first = None
    if returncode != 0:
        reason = re.search(r"structured construction: [^;]+; (.*?); work=", stderr)
        first = reason.group(1) if reason else next(
            (line.split("error: ", 1)[-1] for line in stderr.splitlines() if "error: " in line), status)
    return {
        "status": status,
        "returncode": returncode,
        "first_refusal": first,
        "mechanisms": {
            "sets": stdout.count("pto.set_flag["),
            "waits": stdout.count("pto.wait_flag["),
            "barriers": stdout.count("pto.barrier"),
        },
    }


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_sha(value, what):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(what + " requires a lowercase SHA-256")


def cas_address(sha256):
    """Canonical address shared by original and prepared byte populations."""
    _validate_sha(sha256, "CAS object")
    return f"sha256/{sha256[:2]}/{sha256}"


def _validate_outcome(value, what):
    required = {"status", "returncode", "first_refusal", "mechanisms"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(what + " has noncanonical outcome fields")
    if value["status"] not in ("pass", "refused") or not isinstance(value["returncode"], int):
        raise ValueError(what + " has an unsupported frozen status")
    if (value["status"] == "pass" and value["returncode"] != 0) or \
       (value["status"] == "refused" and value["returncode"] <= 0):
        raise ValueError(what + " status and return code disagree")
    refusal = value["first_refusal"]
    if value["status"] == "pass":
        if refusal is not None:
            raise ValueError(what + " successful outcome has a refusal")
    elif not isinstance(refusal, str) or not refusal:
        raise ValueError(what + " refused outcome lacks its exact first refusal")
    mechanisms = value["mechanisms"]
    if not isinstance(mechanisms, dict) or set(mechanisms) != {"sets", "waits", "barriers"} or any(
            not isinstance(count, int) or count < 0 for count in mechanisms.values()):
        raise ValueError(what + " has invalid mechanism counts")


def validate_frozen_record(record, index):
    required = {
        "schema", "index", "id", "root", "path", "family", "classification",
        "compatibility_stage", "gm_contract", "original", "preparation", "prepared",
        "baseline", "current",
    }
    if not isinstance(record, dict) or set(record) != required or record.get("schema") != FROZEN_SCHEMA:
        raise ValueError(f"record {index} has noncanonical fields or schema")
    if record["index"] != index or not isinstance(record["id"], str) or not record["id"]:
        raise ValueError(f"record {index} has an invalid stable identity")
    if not isinstance(record["family"], str) or not record["family"]:
        raise ValueError(f"record {index} has an invalid corpus family")
    if record["root"] not in FROZEN_ROOT_KINDS:
        raise ValueError(f"record {index} has an invalid named root")
    path = record["path"]
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError(f"record {index} has a nonportable path")
    pure = Path(path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != path:
        raise ValueError(f"record {index} path escapes or is not normalized")
    if record["classification"] not in CLASSES or record["compatibility_stage"] not in STAGES:
        raise ValueError(f"record {index} lacks an exact stage/classification")
    if record["gm_contract"] not in ("may-alias", "assume-disjoint-arguments"):
        raise ValueError(f"record {index} has an unknown GM contract")
    original = record["original"]
    if not isinstance(original, dict) or set(original) != {"kind", "sha256"}:
        raise ValueError(f"record {index} has invalid original-byte provenance")
    if original["kind"] not in set(FROZEN_ROOT_KINDS.values()) or \
       original["kind"] != FROZEN_ROOT_KINDS[record["root"]]:
        raise ValueError(f"record {index} original kind disagrees with its named root")
    _validate_sha(original["sha256"], f"record {index} original")
    preparation = record["preparation"]
    if not isinstance(preparation, dict) or set(preparation) != {"recipe", "adaptations"} or \
       preparation["recipe"] != "bind-a2a3-to-a3-v1" or not isinstance(preparation["adaptations"], list) or \
       any(not isinstance(item, str) for item in preparation["adaptations"]):
        raise ValueError(f"record {index} has invalid preparation provenance")
    prepared = record["prepared"]
    if not isinstance(prepared, dict) or set(prepared) != {"sha256", "cas"}:
        raise ValueError(f"record {index} has invalid prepared-byte provenance")
    _validate_sha(prepared["sha256"], f"record {index} prepared")
    if prepared["cas"] != cas_address(prepared["sha256"]):
        raise ValueError(f"record {index} has a noncanonical CAS address")
    _validate_outcome(record["baseline"], f"record {index} baseline")
    _validate_outcome(record["current"], f"record {index} current")


def load_frozen_manifest(path):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("frozen manifest is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("frozen manifest requires canonical LF-terminated JSONL")
    lines = text.splitlines()
    if len(lines) != FROZEN_RECORDS or any(not line for line in lines):
        raise ValueError(f"frozen manifest requires exactly {FROZEN_RECORDS} records")
    records = []
    ids = set()
    locations = set()
    for index, line in enumerate(lines):
        record = json.loads(line)
        validate_frozen_record(record, index)
        if line != _canonical_json(record):
            raise ValueError(f"record {index} is not canonical JSON")
        if record["id"] in ids or (record["root"], record["path"]) in locations:
            raise ValueError(f"record {index} duplicates an identity or source path")
        ids.add(record["id"])
        locations.add((record["root"], record["path"]))
        records.append(record)
    return records, digest(raw)


def load_frozen_lock(path, manifest_sha256):
    lock = json.loads(Path(path).read_text())
    required = {
        "schema", "record_schema", "record_count", "manifest_sha256", "roots",
        "cas_layout", "preparation_recipes", "arms", "source_manifests", "evidence",
        "summary", "scope",
    }
    if not isinstance(lock, dict) or set(lock) != required or lock.get("schema") != FROZEN_LOCK_SCHEMA or \
       lock.get("record_schema") != FROZEN_SCHEMA or lock.get("record_count") != FROZEN_RECORDS:
        raise ValueError("invalid frozen lock schema or population")
    if lock.get("manifest_sha256") != manifest_sha256:
        raise ValueError("frozen lock does not name the exact manifest bytes")
    roots = lock["roots"]
    if not isinstance(roots, dict) or not roots or any(
            not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or not isinstance(note, str) or not note
            for name, note in roots.items()):
        raise ValueError("frozen lock has invalid named roots")
    if lock.get("preparation_recipes") != {
            "bind-a2a3-to-a3-v1": "exact source bytes; only bind module target a2a3 to a3"}:
        raise ValueError("frozen lock has an unknown preparation recipe")
    if lock.get("cas_layout") != {
            "algorithm": "sha256",
            "path": "sha256/{first2}/{sha256}",
            "populations": ["original", "prepared"]}:
        raise ValueError("frozen lock has an unknown or incomplete CAS layout")
    if lock.get("arms") != FROZEN_ARMS:
        raise ValueError("frozen lock commands differ from the exact reviewed public-option templates")
    sources = lock["source_manifests"]
    if not isinstance(sources, list) or len(sources) != len(roots):
        raise ValueError("frozen lock must name one source manifest per root")
    source_roots = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"root", "sha256", "role"}:
            raise ValueError("invalid source-manifest provenance")
        if source["root"] not in roots or source["root"] in source_roots or \
           not isinstance(source["role"], str) or not source["role"]:
            raise ValueError("invalid source-manifest root or role")
        source_roots.add(source["root"])
        _validate_sha(source["sha256"], "source manifest")
    evidence = lock["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"baseline", "current"}:
        raise ValueError("frozen lock requires exact baseline/current evidence")
    for arm in ("baseline", "current"):
        item = evidence[arm]
        if not isinstance(item, dict) or set(item) != {
                "reference_revision", "campaign_base_revision",
                "campaign_dirty_listing_sha256", "compiler_sha256",
                "summary_sha256", "role"} or \
           not isinstance(item["campaign_base_revision"], str) or \
           not re.fullmatch(r"[0-9a-f]{40}", item["campaign_base_revision"]) or \
           (item["reference_revision"] is not None and
            (not isinstance(item["reference_revision"], str) or
             not re.fullmatch(r"[0-9a-f]{40}", item["reference_revision"]))) or \
           not isinstance(item["role"], str) or not item["role"]:
            raise ValueError("invalid frozen evidence provenance")
        _validate_sha(item["campaign_dirty_listing_sha256"], "campaign dirty listing")
        _validate_sha(item["compiler_sha256"], "campaign compiler")
        _validate_sha(item["summary_sha256"], "summary evidence")
    summary = lock["summary"]
    if not isinstance(summary, dict) or set(summary) != {"baseline", "current"}:
        raise ValueError("invalid frozen aggregate summary")
    for arm in ("baseline", "current"):
        if summary[arm] != {"pass": 197, "refused": 166}:
            raise ValueError("frozen aggregate does not match the qualified 363-record population")
    if not isinstance(lock["scope"], str) or not lock["scope"]:
        raise ValueError("frozen lock requires an evidence-scope statement")
    return lock


def load_root_map(path, required):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != set(required) or any(
            not isinstance(path, str) or not path for path in value.values()):
        raise ValueError("root map must define exactly the lock's named roots")
    roots = {name: Path(path).expanduser().resolve() for name, path in value.items()}
    for name, root in roots.items():
        if not root.is_dir():
            raise ValueError(f"named root {name} is not a directory: {root}")
    return roots


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path, stop=None):
    """Reject an existing symlink at or above path, optionally stopping at stop."""
    path = Path(path).absolute()
    stop = Path(stop).absolute() if stop is not None else None
    cursor = path
    while True:
        if cursor.is_symlink():
            raise ValueError("symlink path component is not permitted: " + str(cursor))
        if cursor == stop or cursor.parent == cursor:
            return
        cursor = cursor.parent


def _require_external(path, roots, what):
    requested = Path(path).expanduser().absolute()
    _reject_symlink_components(requested)
    resolved = requested.resolve()
    forbidden = [ROOT.resolve(), *(root.resolve() for root in roots.values())]
    if any(_inside(resolved, root) for root in forbidden):
        raise ValueError(what + " must be outside Git/source roots")
    return resolved


def _read_cas_object(cas, sha256, identity, role):
    object_path = (cas / cas_address(sha256)).absolute()
    _reject_symlink_components(object_path, cas)
    resolved = object_path.resolve()
    if not _inside(resolved, cas) or not object_path.is_file():
        raise ValueError(f"missing or escaping {role} CAS object: {identity}")
    data = object_path.read_bytes()
    if digest(data) != sha256:
        raise ValueError(f"{role} CAS object hash mismatch: {identity}")
    return data


def preflight_frozen(records, roots, cas_root=None):
    """Validate every byte population before any external compiler invocation."""
    cas = _require_external(cas_root, roots, "CAS root") if cas_root else None
    if not roots and cas is None:
        raise ValueError("a root map or complete original/prepared CAS is required")
    if cas is not None:
        if not cas.is_dir():
            raise ValueError("CAS root is not a directory: " + str(cas))
    prepared_rows = []
    originals = {}
    for record in records:
        source = None
        original = None
        if roots:
            if record["root"] not in roots:
                raise ValueError("missing named root for input: " + record["id"])
            root = roots[record["root"]]
            source = (root / record["path"]).resolve()
            if not _inside(source, root) or not source.is_file():
                raise ValueError("missing or escaping original input: " + record["id"])
            original = source.read_bytes()
            if digest(original) != record["original"]["sha256"]:
                raise ValueError("original byte hash mismatch: " + record["id"])
            originals[source] = record["original"]["sha256"]
        if cas is not None:
            stored_original = _read_cas_object(
                cas, record["original"]["sha256"], record["id"], "original")
            if original is not None and stored_original != original:
                raise ValueError("original source and CAS bytes disagree: " + record["id"])
            original = stored_original
        if original is None:
            raise ValueError("original bytes are unavailable: " + record["id"])
        prepared, adaptations = prepare_bytes(original, record["compatibility_stage"])
        if adaptations != record["preparation"]["adaptations"] or \
           digest(prepared) != record["prepared"]["sha256"]:
            raise ValueError("prepared byte hash/provenance mismatch: " + record["id"])
        if cas is not None:
            object_bytes = _read_cas_object(
                cas, record["prepared"]["sha256"], record["id"], "prepared")
            if object_bytes != prepared:
                raise ValueError("prepared CAS object mismatch: " + record["id"])
            prepared = object_bytes
        prepared_rows.append((record, source, original, prepared))
    return prepared_rows, originals


def verify_originals_unchanged(originals):
    for source, expected in originals.items():
        if not source.is_file() or digest(source.read_bytes()) != expected:
            raise RuntimeError("original input changed during replay: " + str(source))


def export_frozen_cas(records, roots, output):
    if not roots:
        raise ValueError("CAS export requires a root map")
    output = _require_external(output, roots, "CAS output")
    if output.exists() and not output.is_dir():
        raise ValueError("CAS output exists and is not a directory")
    rows, originals = preflight_frozen(records, roots)
    objects = {}
    for record, source, original, prepared in rows:
        for sha256, data in ((record["original"]["sha256"], original),
                             (record["prepared"]["sha256"], prepared)):
            prior = objects.setdefault(sha256, data)
            if prior != data:
                raise RuntimeError("SHA-256 collision in frozen byte populations")
    targets = []
    for sha256, data in objects.items():
        target = output / cas_address(sha256)
        _reject_symlink_components(target, output)
        resolved = target.resolve()
        if not _inside(resolved, output) or any(
                _inside(resolved, root.resolve()) for root in roots.values()):
            raise ValueError("CAS object path escapes output or enters a source root")
        if target.exists():
            if target.is_symlink() or not target.is_file() or digest(target.read_bytes()) != sha256:
                raise ValueError("existing CAS object has different bytes: " + str(target))
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        _reject_symlink_components(temporary, output)
        if temporary.exists():
            raise ValueError("CAS temporary path already exists: " + str(temporary))
        targets.append((sha256, data, target, temporary))
    # No CAS byte is written until every original/prepared object target has
    # passed the symlink, containment, and existing-object checks above.
    created = reused = 0
    for sha256, data, target, temporary in targets:
        if target.exists():
            reused += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(target, output)
        resolved = target.resolve()
        if not _inside(resolved, output) or any(
                _inside(resolved, root.resolve()) for root in roots.values()):
            raise ValueError("CAS object path changed after target preflight")
        _reject_symlink_components(temporary, output)
        with temporary.open("xb") as stream:
            stream.write(data)
        if digest(temporary.read_bytes()) != sha256:
            temporary.unlink()
            raise RuntimeError("CAS write verification failed: " + sha256)
        try:
            try:
                # Publish without replacing a path that appeared after preflight.
                os.link(temporary, target)
            except FileExistsError:
                if target.is_symlink() or not target.is_file() or digest(target.read_bytes()) != sha256:
                    raise ValueError("CAS object changed during publication: " + str(target))
                reused += 1
            else:
                created += 1
        finally:
            if temporary.exists():
                temporary.unlink()
    verify_originals_unchanged(originals)
    return {
        "records": len(rows),
        "original_references": len(rows),
        "prepared_references": len(rows),
        "unique_objects": len(objects),
        "created": created,
        "reused": reused,
    }


def replay_frozen(records, lock, roots, opt, output, arm, timeout, cas_root=None):
    # Keep this API safe when called directly rather than through main(), which
    # has already validated the complete lock.
    if lock.get("arms") != FROZEN_ARMS:
        raise ValueError("replay commands differ from the exact reviewed public-option templates")
    opt = Path(opt).expanduser().resolve()
    if not opt.is_file():
        raise ValueError("compiler executable is missing")
    forbidden = dict(roots)
    if cas_root is not None:
        forbidden["CAS"] = Path(cas_root).expanduser().resolve()
    output = _require_external(output, forbidden, "replay output")
    if output.exists():
        raise ValueError("replay output already exists")
    compiler_sha256 = digest(opt.read_bytes())
    # Critical ordering: all 363 originals, preparations, and optional CAS
    # objects are checked before output creation or the first subprocess.
    rows, originals = preflight_frozen(records, roots, cas_root)
    output.mkdir(parents=True)
    selected = ("baseline", "current") if arm == "both" else (arm,)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PTOAS_STRUCTURED_PLAN_JSON", None)
    env.pop("PTOAS_LOGICAL_TRACE", None)
    results = []
    for record, source, original, prepared in rows:
        snapshot = output / f"{record['index']:04d}-{record['prepared']['sha256']}.pto"
        snapshot.write_bytes(prepared)
        for name in selected:
            argv = [item.replace("{input}", str(snapshot)).replace("{gm_contract}", record["gm_contract"])
                    for item in lock["arms"][name]["argv"]]
            command = [str(opt), *argv]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
                outcome = composition_outcome(completed.returncode, completed.stdout, completed.stderr)
            except subprocess.TimeoutExpired:
                outcome = {"status": "timeout", "returncode": None, "first_refusal": "timeout",
                           "mechanisms": {"sets": 0, "waits": 0, "barriers": 0}}
            if digest(snapshot.read_bytes()) != record["prepared"]["sha256"]:
                raise RuntimeError("prepared snapshot changed during replay: " + record["id"])
            if source is not None and digest(source.read_bytes()) != record["original"]["sha256"]:
                raise RuntimeError("original input changed during replay: " + record["id"])
            expected = record[name]
            row = {
                "index": record["index"],
                "id": record["id"],
                "arm": name,
                "argv": argv,
                "expected": expected,
                "actual": outcome,
                "exact_match": outcome == expected,
            }
            results.append(row)
            with (output / "results.jsonl").open("a") as stream:
                stream.write(_canonical_json(row) + "\n")
    verify_originals_unchanged(originals)
    if cas_root is not None:
        # A candidate process receives only snapshots, but re-read the complete
        # transported evidence so mutation cannot go unreported.
        _, checked_originals = preflight_frozen(records, roots, cas_root)
        verify_originals_unchanged(checked_originals)
    if digest(opt.read_bytes()) != compiler_sha256:
        raise RuntimeError("compiler executable changed during replay")
    report = {
        "schema": "oahs.frozen-corpus.replay.v1",
        "manifest_sha256": lock["manifest_sha256"],
        "candidate_compiler_sha256": compiler_sha256,
        "reference_compiler_sha256": {
            name: lock["evidence"][name]["compiler_sha256"] for name in selected
        },
        "arm_argv": {name: lock["arms"][name]["argv"] for name in selected},
        "records": len(rows),
        "arms": list(selected),
        "comparisons": len(results),
        "exact_matches": sum(row["exact_match"] for row in results),
        "deviations": sum(not row["exact_match"] for row in results),
        "deviations_by_arm": {
            name: sum(row["arm"] == name and not row["exact_match"] for row in results)
            for name in selected
        },
        "scope": lock["scope"],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def summarize(rows):
    out = {}
    for stage in STAGES:
        selected = [r for r in rows if r["stage"] == stage]
        if not selected:
            continue
        pools = {}
        for classification in CLASSES:
            population = [r for r in selected if r["classification"] == classification]
            if not population:
                continue
            pools[classification] = dict(inputs=len(population),
                accepted={a: sum(r["arms"][a]["status"] == "applied" for r in population) for a in ARMS})
        failures = Counter()
        common = []
        for r in selected:
            if r["arms"]["existing"]["status"] == "applied":
                if r["arms"]["structured-hardware"]["status"] != "applied":
                    failures[r["arms"]["structured-hardware"].get("reason", "unclassified failure")] += 1
                else:
                    common.append(r["id"])
        out[stage] = dict(pools=pools, existing_successful_hardware_refusals=dict(failures),
                          common_success_ids=common)
    return out


def metrics(path, python_root):
    # Parse actual IR; no grep over comments, and no summing different mechanisms.
    sys.path.insert(0, str(python_root))
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    from measure import static_metrics
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        module = ir.Module.parse(path.read_text())
        if not module.operation.verify():
            raise ValueError("invalid emitted IR")
        data = static_metrics(module.operation)
    counts = data["counts"]
    named, retirement = Counter(), 0
    for key, count in counts.items():
        if key.startswith("barrier:"):
            pipe = re.search(r"PIPE_[A-Z0-9]+", key).group(0)
            if pipe == "PIPE_ALL":
                retirement += count  # reported ALL sites, not assumed all terminal
            else:
                named[pipe] += count
    return dict(sets=counts.get("pto.set_flag", 0), waits=counts.get("pto.wait_flag", 0),
                named_barriers=dict(named), pipe_all=retirement,
                event_ids_by_direction=data["event_ids_by_direction"],
                placements_sha256=data["placement_sha256"])


def sweep(manifest, opt, python_root, output, timeout, require_historical,
          repo=ROOT, lock_output=None):
    validate_manifest(manifest)
    root = Path(repo).resolve() if manifest["root"] == PORTABLE_ROOT else Path(manifest["root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key in ("PTOAS_STRUCTURED_PLAN_JSON", "PTOAS_LOGICAL_TRACE"):
        env.pop(key, None)
    rows = []
    for number, case in enumerate(manifest["cases"]):
        source = Path(case["path"])
        if not source.is_absolute():
            source = root / source
        data = source.read_bytes()
        if digest(data) != case["sha256"]:
            raise ValueError("changed corpus input " + case["id"])
        folder = output / f"{number:04d}"
        folder.mkdir()
        prepared, changes = prepare_bytes(data, case["stage"])
        snapshot = folder / "input.pto"
        snapshot.write_bytes(prepared)
        row = dict(case, input_sha256=digest(prepared), adaptations=changes, arms={})
        # Rotate order: this remains one diagnostic round, NOT a timing gate.
        arms = ARMS[number % 3:] + ARMS[:number % 3]
        for arm in arms:
            planner = "existing" if arm == "existing" else "structured"
            contract = CONTRACT if arm == "structured-hardware" else "conservative"
            path = folder / (arm + ".pto")
            option = f"-pto-insert-sync=planner={planner} hardware-contract={contract} gm-alias={case['gm_contract']}"
            command = [str(opt), str(snapshot), option, "-o", str(path)]
            start = time.monotonic()
            try:
                result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
                code, stdout, stderr = result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired as error:
                code = "harness-timeout"
                def text(value):
                    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
                stdout, stderr = text(error.stdout), text(error.stderr)
            seconds = time.monotonic() - start
            (folder / (arm + ".stdout")).write_text(stdout)
            (folder / (arm + ".stderr")).write_text(stderr)
            item = dict(status="applied" if code == 0 else "refused", exit=code, command=command, seconds=seconds)
            if code == 0:
                try:
                    item["mechanisms"] = metrics(path, python_root)
                except Exception as error:
                    item["status"] = "measurement-failed"
                    item["reason"] = type(error).__name__ + ": " + str(error)
                item["output_sha256"] = digest(path.read_bytes())
            else:
                diagnostic = next((line.strip() for line in stderr.splitlines() if "error:" in line), stderr[-1000:])
                item["reason"] = (diagnostic.partition("error:")[2].strip() if "error:" in diagnostic else diagnostic) or str(code)
            row["arms"][arm] = item
        rows.append(row)
        (output / "results.json").write_text(json.dumps(dict(cases=rows), indent=2) + "\n")
    result = dict(status="completed", cases=rows, summary=summarize(rows),
                  missing_requested=manifest.get("missing_requested", []),
                  opt_sha256=digest(opt.read_bytes()), device="NOT_RUN",
                  timing="single diagnostic round, no matched performance acceptance claim")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lock = compact_lock(result, manifest)
    (output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    if lock_output:
        with Path(lock_output).open("x") as stream:
            json.dump(lock, stream, indent=2)
            stream.write("\n")
    if require_historical and not any(r["classification"] == "historical" for r in rows):
        raise RuntimeError("historical GEMM missing: results retained, coverage request incomplete")
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--repo", type=Path, default=ROOT)
    d.add_argument("--all-a2a3", action="store_true")
    d.add_argument("--portable", action="store_true",
                   help="write a repository-relative manifest suitable for version control")
    d.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--repo", type=Path, default=ROOT)
    v.add_argument("--lock", type=Path)
    l = sub.add_parser("lock")
    l.add_argument("--manifest", type=Path, required=True)
    l.add_argument("--summary", type=Path, required=True)
    l.add_argument("--implementation", required=True)
    l.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("run")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--opt", type=Path, required=True)
    s.add_argument("--python-root", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--repo", type=Path, default=ROOT,
                   help="repository root for a portable $REPO manifest")
    s.add_argument("--lock-output", type=Path,
                   help="also create a compact version-controlled result lock")
    s.add_argument("--timeout", type=float, default=120, help="external process timeout; NOT an analysis quota")
    s.add_argument("--require-historical", action="store_true")
    for name in ("frozen-verify", "frozen-export", "frozen-replay"):
        frozen = sub.add_parser(name)
        frozen.add_argument("--manifest", type=Path, required=True)
        frozen.add_argument("--lock", type=Path, required=True)
        frozen.add_argument("--root-map", type=Path, required=name == "frozen-export")
        if name in ("frozen-verify", "frozen-replay"):
            frozen.add_argument("--cas-root", type=Path)
        if name == "frozen-export":
            frozen.add_argument("--output", type=Path, required=True)
        if name == "frozen-replay":
            frozen.add_argument("--opt", type=Path, required=True)
            frozen.add_argument("--output", type=Path, required=True)
            frozen.add_argument("--arm", choices=("baseline", "current", "both"), default="both")
            frozen.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.command == "discover":
        report = discover(args.repo, args.all_a2a3, args.portable)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps(dict(inputs=len(report["cases"]), missing=report["missing_requested"])))
    elif args.command == "verify":
        manifest = json.loads(args.manifest.read_text())
        report = verify_discovery(manifest, args.repo)
        if args.lock:
            report["lock"] = validate_lock(json.loads(args.lock.read_text()), manifest)
        print(json.dumps(report, indent=2))
    elif args.command == "lock":
        manifest = json.loads(args.manifest.read_text())
        lock = compact_lock(json.loads(args.summary.read_text()), manifest,
                            args.implementation)
        validate_lock(lock, manifest)
        with args.output.open("x") as stream:
            json.dump(lock, stream, indent=2)
            stream.write("\n")
        print(json.dumps(dict(inputs=len(lock["cases"]),
                              manifest_sha256=lock["manifest_sha256"])))
    elif args.command.startswith("frozen-"):
        records, manifest_sha256 = load_frozen_manifest(args.manifest)
        lock = load_frozen_lock(args.lock, manifest_sha256)
        roots = load_root_map(args.root_map, lock["roots"]) if args.root_map else {}
        if {record["root"] for record in records} != set(lock["roots"]):
            raise ValueError("manifest and lock name different root populations")
        if not roots and getattr(args, "cas_root", None) is None:
            raise ValueError("rootless verification/replay requires --cas-root")
        if args.command == "frozen-verify":
            rows, originals = preflight_frozen(records, roots, args.cas_root)
            verify_originals_unchanged(originals)
            report = {"records": len(rows), "manifest_sha256": manifest_sha256,
                      "roots": sorted(roots), "cas_verified": args.cas_root is not None,
                      "original_source_crosscheck": bool(roots)}
        elif args.command == "frozen-export":
            report = export_frozen_cas(records, roots, args.output)
            report["manifest_sha256"] = manifest_sha256
        else:
            if args.timeout <= 0:
                raise ValueError("timeout must be positive")
            report = replay_frozen(records, lock, roots, args.opt, args.output,
                                   args.arm, args.timeout, args.cas_root)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        sweep(json.loads(args.manifest.read_text()), args.opt.resolve(), args.python_root.resolve(),
              args.output.resolve(), args.timeout, args.require_historical,
              args.repo.resolve(), args.lock_output)


if __name__ == "__main__":
    main()
