#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Lightweight tests for the frozen OAHS corpus evidence contract."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
SPEC = importlib.util.spec_from_file_location("s7_corpus", HERE / "s7_corpus.py")
s7_corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s7_corpus)


def record_for(data, root="PTOAS", index=0, identity="case"):
    prepared, adaptations = s7_corpus.prepare_bytes(data, "raw-addressed-compatibility")
    success = {
        "status": "pass",
        "returncode": 0,
        "first_refusal": None,
        "mechanisms": {"sets": 0, "waits": 0, "barriers": 0},
    }
    prepared_hash = s7_corpus.digest(prepared)
    return {
        "schema": s7_corpus.FROZEN_SCHEMA,
        "index": index,
        "id": identity,
        "root": root,
        "path": identity + ".pto",
        "family": "unit",
        "classification": "focused",
        "compatibility_stage": "raw-addressed-compatibility",
        "gm_contract": "may-alias",
        "original": {"kind": "ptoas-source", "sha256": s7_corpus.digest(data)},
        "preparation": {"recipe": "bind-a2a3-to-a3-v1", "adaptations": adaptations},
        "prepared": {
            "sha256": prepared_hash,
            "cas": f"sha256/{prepared_hash[:2]}/{prepared_hash}",
        },
        "baseline": copy.deepcopy(success),
        "current": copy.deepcopy(success),
    }


def replay_lock():
    return {
        "manifest_sha256": "0" * 64,
        "arms": copy.deepcopy(s7_corpus.FROZEN_ARMS),
        "evidence": {
            "baseline": {"compiler_sha256": "1" * 64},
            "current": {"compiler_sha256": "2" * 64},
        },
        "scope": "unit replay",
    }


class FrozenArtifactsTest(unittest.TestCase):
    def test_checked_in_population_and_lock_are_exact(self):
        records, manifest_hash = s7_corpus.load_frozen_manifest(CORPUS / "frozen-363.jsonl")
        lock = s7_corpus.load_frozen_lock(CORPUS / "frozen-363.lock.json", manifest_hash)
        self.assertEqual(len(records), 363)
        self.assertEqual({row["index"] for row in records}, set(range(363)))
        self.assertEqual({row["root"] for row in records}, set(lock["roots"]))
        self.assertEqual(lock["summary"]["baseline"], {"pass": 197, "refused": 166})
        self.assertEqual(lock["summary"]["current"], {"pass": 197, "refused": 166})
        schema = json.loads((CORPUS / "frozen-363.schema.json").read_text())
        self.assertEqual(schema["$id"], s7_corpus.FROZEN_SCHEMA)

    def test_record_validator_rejects_path_and_status_corruption(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        record = record_for(data)
        s7_corpus.validate_frozen_record(record, 0)
        escaped = copy.deepcopy(record)
        escaped["path"] = "../case.pto"
        with self.assertRaisesRegex(ValueError, "escapes"):
            s7_corpus.validate_frozen_record(escaped, 0)
        inconsistent = copy.deepcopy(record)
        inconsistent["baseline"]["first_refusal"] = "not successful"
        with self.assertRaisesRegex(ValueError, "successful outcome"):
            s7_corpus.validate_frozen_record(inconsistent, 0)
        unknown_kind = copy.deepcopy(record)
        unknown_kind["original"]["kind"] = "unknown"
        with self.assertRaisesRegex(ValueError, "kind disagrees"):
            s7_corpus.validate_frozen_record(unknown_kind, 0)
        wrong_root_kind = copy.deepcopy(record)
        wrong_root_kind["root"] = "PYPTO_GENERATED"
        with self.assertRaisesRegex(ValueError, "kind disagrees"):
            s7_corpus.validate_frozen_record(wrong_root_kind, 0)

    def test_lock_rejects_every_command_template_mutation(self):
        _, manifest_hash = s7_corpus.load_frozen_manifest(CORPUS / "frozen-363.jsonl")
        original = json.loads((CORPUS / "frozen-363.lock.json").read_text())
        mutations = {}
        unknown_placeholder = copy.deepcopy(original)
        unknown_placeholder["arms"]["current"]["argv"][-1] = "{unreviewed_input}"
        mutations["unknown placeholder"] = unknown_placeholder
        hardcoded_input = copy.deepcopy(original)
        hardcoded_input["arms"]["current"]["argv"][-1] = "/tmp/input.pto"
        mutations["hard-coded input"] = hardcoded_input
        arbitrary_option = copy.deepcopy(original)
        arbitrary_option["arms"]["current"]["argv"].insert(0, "--load-pass-plugin=/tmp/plugin.so")
        mutations["arbitrary option"] = arbitrary_option
        with tempfile.TemporaryDirectory() as directory:
            for name, lock in mutations.items():
                with self.subTest(name=name):
                    path = Path(directory) / (name.replace(" ", "-") + ".json")
                    path.write_text(json.dumps(lock))
                    with self.assertRaisesRegex(ValueError, "exact reviewed public-option"):
                        s7_corpus.load_frozen_lock(path, manifest_hash)

    def test_replay_validates_all_inputs_before_any_subprocess(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "sources"
            source_root.mkdir()
            first = record_for(data, index=0, identity="first")
            second = record_for(data, index=1, identity="second")
            (source_root / first["path"]).write_bytes(data)
            (source_root / second["path"]).write_bytes(data)
            second["original"]["sha256"] = "f" * 64
            compiler = base / "ptoas-opt"
            compiler.write_bytes(b"unit compiler")
            output = base / "results"
            with mock.patch.object(s7_corpus.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "original byte hash mismatch: second"):
                    s7_corpus.replay_frozen(
                        [first, second], replay_lock(), {"PTOAS": source_root},
                        compiler, output, "baseline", 1)
            run.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual((source_root / first["path"]).read_bytes(), data)

    def test_replay_invokes_compiler_only_on_exact_snapshot(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        prepared, _ = s7_corpus.prepare_bytes(data, "raw-addressed-compatibility")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "sources"
            source_root.mkdir()
            record = record_for(data)
            source = source_root / record["path"]
            source.write_bytes(data)
            compiler = base / "ptoas-opt"
            compiler.write_bytes(b"unit compiler")
            output = base / "results"

            def invoke(command, **unused):
                self.assertEqual(Path(command[0]), compiler)
                snapshot = Path(command[-1])
                self.assertNotEqual(snapshot, source)
                self.assertEqual(snapshot.read_bytes(), prepared)
                stdout = "pto.set_flag[" if any(
                    "planner=composition" in item for item in command) else ""
                return subprocess.CompletedProcess(command, 0, stdout, "")

            with mock.patch.object(s7_corpus.subprocess, "run", side_effect=invoke) as run:
                report = s7_corpus.replay_frozen(
                    [record], replay_lock(), {"PTOAS": source_root},
                    compiler, output, "both", 1)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(report["comparisons"], 2)
            self.assertEqual(report["exact_matches"], 1)
            self.assertEqual(report["deviations"], 1)
            self.assertEqual(report["deviations_by_arm"], {"baseline": 0, "current": 1})
            self.assertEqual(source.read_bytes(), data)
            snapshots = list(output.glob("*.pto"))
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(s7_corpus.digest(snapshots[0].read_bytes()), record["prepared"]["sha256"])
            result_rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
            self.assertTrue(result_rows[0]["exact_match"])
            self.assertFalse(result_rows[1]["exact_match"])

    def test_direct_replay_rejects_command_bypass_before_writes(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "sources"
            source_root.mkdir()
            record = record_for(data)
            (source_root / record["path"]).write_bytes(data)
            compiler = base / "ptoas-opt"
            compiler.write_bytes(b"unit compiler")
            output = base / "results"
            lock = replay_lock()
            lock["arms"]["current"]["argv"].insert(0, "--load-pass-plugin=/tmp/plugin.so")
            with mock.patch.object(s7_corpus.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "exact reviewed public-option"):
                    s7_corpus.replay_frozen(
                        [record], lock, {"PTOAS": source_root},
                        compiler, output, "current", 1)
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_small_content_addressed_export_is_exact(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "sources"
            source_root.mkdir()
            record = record_for(data)
            source = source_root / record["path"]
            source.write_bytes(data)
            cas = base / "cas"
            report = s7_corpus.export_frozen_cas([record], {"PTOAS": source_root}, cas)
            prepared, _ = s7_corpus.prepare_bytes(data, "raw-addressed-compatibility")
            self.assertEqual(report, {
                "records": 1,
                "original_references": 1,
                "prepared_references": 1,
                "unique_objects": 2,
                "created": 2,
                "reused": 0,
            })
            original_object = cas / s7_corpus.cas_address(record["original"]["sha256"])
            self.assertEqual(original_object.read_bytes(), data)
            self.assertEqual((cas / record["prepared"]["cas"]).read_bytes(), prepared)
            rows, originals = s7_corpus.preflight_frozen([record], {"PTOAS": source_root}, cas)
            self.assertEqual(rows[0][2], data)
            self.assertEqual(rows[0][3], prepared)
            s7_corpus.verify_originals_unchanged(originals)

            rootless, no_sources = s7_corpus.preflight_frozen([record], {}, cas)
            self.assertEqual(rootless[0][2:], (data, prepared))
            self.assertEqual(no_sources, {})

            source.write_bytes(data + b"// changed\n")
            with self.assertRaisesRegex(ValueError, "original byte hash mismatch"):
                s7_corpus.preflight_frozen([record], {"PTOAS": source_root}, cas)

            source.write_bytes(data)
            original_object.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "original CAS object hash mismatch"):
                s7_corpus.preflight_frozen([record], {}, cas)

    def test_rootless_cas_replay_uses_only_prepared_snapshot(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        prepared, _ = s7_corpus.prepare_bytes(data, "raw-addressed-compatibility")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "sources"
            source_root.mkdir()
            record = record_for(data)
            (source_root / record["path"]).write_bytes(data)
            cas = base / "cas"
            s7_corpus.export_frozen_cas([record], {"PTOAS": source_root}, cas)
            compiler = base / "ptoas-opt"
            compiler.write_bytes(b"unit compiler")
            output = base / "results"

            def invoke(command, **unused):
                snapshot = Path(command[-1])
                self.assertEqual(snapshot.read_bytes(), prepared)
                self.assertFalse(str(snapshot).startswith(str(cas)))
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(s7_corpus.subprocess, "run", side_effect=invoke) as run:
                report = s7_corpus.replay_frozen(
                    [record], replay_lock(), {}, compiler, output,
                    "current", 1, cas)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(report["deviations"], 0)

    def test_replay_rechecks_complete_cas_after_candidate(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "sources"
            source_root.mkdir()
            record = record_for(data)
            (source_root / record["path"]).write_bytes(data)
            cas = base / "cas"
            s7_corpus.export_frozen_cas([record], {"PTOAS": source_root}, cas)
            original_object = cas / s7_corpus.cas_address(record["original"]["sha256"])
            compiler = base / "ptoas-opt"
            compiler.write_bytes(b"unit compiler")
            output = base / "results"

            def invoke(command, **unused):
                original_object.write_bytes(b"candidate mutation")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(s7_corpus.subprocess, "run", side_effect=invoke) as run:
                with self.assertRaisesRegex(ValueError, "original CAS object hash mismatch"):
                    s7_corpus.replay_frozen(
                        [record], replay_lock(), {}, compiler, output,
                        "current", 1, cas)
            self.assertEqual(run.call_count, 1)

    def test_cas_must_be_external_to_every_known_source_root(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "sources"
            source_root.mkdir()
            record = record_for(data)
            (source_root / record["path"]).write_bytes(data)
            internal_cas = source_root / "cas"
            internal_cas.mkdir()
            with self.assertRaisesRegex(ValueError, "outside Git/source roots"):
                s7_corpus.preflight_frozen(
                    [record], {"PTOAS": source_root}, internal_cas)
            with self.assertRaisesRegex(ValueError, "outside Git/source roots"):
                s7_corpus.export_frozen_cas(
                    [record], {"PTOAS": source_root}, internal_cas)

    def test_cas_export_rejects_symlinked_target_directory(self):
        data = b'module attributes {pto.target_arch = "a2a3"} {}\n'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "sources"
            source_root.mkdir()
            record = record_for(data)
            (source_root / record["path"]).write_bytes(data)
            cas = base / "cas"
            (cas / "sha256").mkdir(parents=True)
            outside = base / "outside"
            outside.mkdir()
            prefix = record["prepared"]["sha256"][:2]
            (cas / "sha256" / prefix).symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink path component"):
                s7_corpus.export_frozen_cas([record], {"PTOAS": source_root}, cas)
            self.assertEqual(list(outside.iterdir()), [])
            original_object = cas / s7_corpus.cas_address(record["original"]["sha256"])
            self.assertFalse(original_object.exists())


if __name__ == "__main__":
    unittest.main()
