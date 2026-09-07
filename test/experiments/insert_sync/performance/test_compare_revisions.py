#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Campaign failure propagation without invoking a compiler or loading MLIR."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compare_revisions


class ComparisonFailureTests(unittest.TestCase):
    def run_campaign(self, *, outcome="pass", allow_rejections=False, fail_replay=False):
        with tempfile.TemporaryDirectory(prefix="insertsync-comparison-test-") as temporary:
            root = Path(temporary)
            source = root / "input.pto"
            source.write_text("fixture")
            expected = "InsertSync unsupported effect: missing semantic summary"
            case = {"case_id": "fixture", "level": "level2", "scenarios": [{"name": "one"}],
                    "expected_autosync_rejection": expected,
                    "sources": {"auto": {"path": source.name, "sha256": compare_revisions.digest(source)}}}
            (root / "manifest.json").write_text(json.dumps({"cases": [case]}))
            (root / "kernel-pairs-manifest.json").write_text(json.dumps({"cases": []}))
            args = SimpleNamespace(output=root / "results", revised_python_root=root, original_python_root=root,
                                   original_source="original", revised_source="revised", arch=["a3"],
                                   timeout=1, allow_rejections=allow_rejections)

            def compile_arm(options):
                directory = options.output / "fixture"
                directory.mkdir(parents=True)
                runs = {}
                for kind in ("pto", "cpp"):
                    code, diagnostic = 0, ""
                    if outcome == "rejection":
                        code, diagnostic = 1, "error: " + expected
                    elif outcome == "timeout":
                        code, diagnostic = None, "error: " + expected
                    elif outcome == "crash":
                        code, diagnostic = -11, "error: " + expected
                    elif outcome == "mixed":
                        code, diagnostic = 1, "error: " + (expected if kind == "pto" else "emitter failed")
                    (directory / f"{kind}.stderr").write_text(diagnostic)
                    if code == 0:
                        (directory / f"output.{kind}").write_text("output")
                    runs[kind] = {"returncode": code, "status": "pass" if code == 0 else "failure",
                                  "analysis_executed": outcome != "bypass", "allocation_executed": True,
                                  "functions": [{"explicit_sync_bypass": False}]}
                report = {"native_sha256": "native", "cli_sha256": "cli", "native_unchanged": True,
                          "rows": [{"case_id": "fixture", "runs": runs}]}
                (options.output / "results.json").write_text(json.dumps(report))

            def measure(path, scenarios, *unused):
                if scenarios and fail_replay:
                    raise ValueError("unsupported loop bounds")
                counts = {} if path.parent.name == "inputs" else {"pto.barrier": 1}
                return {"static": {"counts": counts}, "scenarios": {}}

            with patch.object(compare_revisions, "__file__", str(root / "compare_revisions.py")), \
                    patch.object(compare_revisions, "compile_arm", side_effect=compile_arm), \
                    patch.object(compare_revisions, "measure", side_effect=measure), \
                    patch.object(compare_revisions, "compare"), contextlib.redirect_stdout(io.StringIO()):
                failed = compare_revisions.campaign(args)
            return (failed, json.loads((args.output / "counts.json").read_text()),
                    (args.output / "counts.md").read_text())

    def test_success_has_no_failures(self):
        failed, report, _ = self.run_campaign()
        self.assertFalse(failed)
        self.assertEqual(report["failures"], [])

    def test_replay_failure_is_visible_even_when_static_counts_exist(self):
        failed, report, markdown = self.run_campaign(fail_replay=True)
        self.assertTrue(failed)
        self.assertEqual(len(report["failures"]), 3)
        row = report["architectures"]["a3"]["combined"]["rows"]["fixture"]
        self.assertEqual(row["status"], "measurement-failure")
        self.assertEqual(row["inserted_static"]["total"], 1)
        self.assertIn("ValueError: unsupported loop bounds", row["measurement_error"])
        self.assertIn("1 (measurement-failure; see counts.json)", markdown)

    def test_declared_rejections_require_explicit_opt_in(self):
        self.assertTrue(self.run_campaign(outcome="rejection")[0])
        failed, report, _ = self.run_campaign(outcome="rejection", allow_rejections=True)
        self.assertFalse(failed)
        self.assertEqual(report["architectures"]["a3"]["combined"]["rows"]["fixture"]["status"], "rejected")

    def test_opt_in_does_not_accept_timeout_crash_or_mixed_failure(self):
        for outcome in ("timeout", "crash", "mixed"):
            with self.subTest(outcome=outcome):
                failed, report, _ = self.run_campaign(outcome=outcome, allow_rejections=True)
                self.assertTrue(failed)
                self.assertEqual(report["architectures"]["a3"]["combined"]["rows"]["fixture"]["status"],
                                 "compile-failure")

    def test_unconfirmed_insertion_fails(self):
        self.assertTrue(self.run_campaign(outcome="bypass")[0])


if __name__ == "__main__":
    unittest.main()
