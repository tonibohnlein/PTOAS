#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for exact third-party license exceptions."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import check_license_headers as checker


REPO_ROOT = Path(__file__).resolve().parents[2]


class LicenseHeadersTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ptoas-license-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.benchmark = self.root / checker.BENCHMARK_ROOT
        (self.benchmark / "references").mkdir(parents=True)
        self.entries = []
        self.reference = (checker.BENCHMARK_ROOT / "references/topk_kernel.cpp").as_posix()
        self.add_reference("topk_kernel.cpp", b"// Original upstream notice\nint reference;\n")
        license_data = (REPO_ROOT / checker.BENCHMARK_ROOT / "references/pto-kernels-LICENSE").read_bytes()
        self.add_reference("pto-kernels-LICENSE", license_data)
        notice = "\n".join(f"# {line}" if line else "#" for line in license_data.decode().splitlines())
        self.generator = self.root / checker.DERIVED_BSD_PATH
        self.generator.write_text(f"#!/usr/bin/env python3\n{notice}\n\nprint('derived')\n", encoding="utf-8")

    def add_reference(self, name, data):
        (self.benchmark / "references" / name).write_bytes(data)
        self.entries.append({"path": f"references/{name}", "sha256": hashlib.sha256(data).hexdigest()})
        self.write_manifest()

    def write_manifest(self):
        (self.benchmark / "kernel-pairs-manifest.json").write_text(
            json.dumps({"reference_files": self.entries}), encoding="utf-8"
        )

    def test_current_approved_sources(self):
        for path in sorted(checker.THIRD_PARTY_PATHS | {checker.DERIVED_BSD_PATH}):
            with self.subTest(path=path):
                self.assertTrue(checker.has_expected_header(path, checker.comment_style_for(path), REPO_ROOT))

    def test_ordinary_header_policy_is_unchanged(self):
        path = self.root / "ordinary.py"
        path.write_text("#!/usr/bin/env python3\n" + "\n".join(checker.HASH_HEADER) + "\n", encoding="utf-8")
        self.assertTrue(checker.has_expected_header("ordinary.py", "#", self.root))
        path.write_text("# An unrelated notice\n", encoding="utf-8")
        self.assertFalse(checker.has_expected_header("ordinary.py", "#", self.root))

    def test_reference_requires_exact_bytes_even_with_oat_header(self):
        self.assertTrue(checker.has_expected_header(self.reference, "//", self.root))
        path = self.root / self.reference
        path.write_text("\n".join(checker.SLASH_HEADER) + "\n", encoding="utf-8")
        self.assertFalse(checker.has_expected_header(self.reference, "//", self.root))

    def test_manifest_does_not_exempt_additional_files(self):
        self.add_reference("unapproved.cpp", b"// Original upstream notice\n")
        path = (checker.BENCHMARK_ROOT / "references/unapproved.cpp").as_posix()
        self.assertFalse(checker.has_expected_header(path, "//", self.root))

    def test_missing_or_malformed_manifest_rejects_exception(self):
        manifest = self.benchmark / "kernel-pairs-manifest.json"
        manifest.unlink()
        self.assertFalse(checker.has_expected_header(self.reference, "//", self.root))
        for content in ("invalid JSON", "[]", '{"reference_files": null}', '{"reference_files": []}'):
            with self.subTest(content=content):
                manifest.write_text(content, encoding="utf-8")
                self.assertFalse(checker.has_expected_header(self.reference, "//", self.root))

    def test_duplicate_reference_pin_rejects_exception(self):
        self.entries.append(self.entries[0].copy())
        self.write_manifest()
        self.assertFalse(checker.has_expected_header(self.reference, "//", self.root))

    def test_derived_generator_requires_full_notice(self):
        self.assertTrue(checker.has_expected_header(checker.DERIVED_BSD_PATH, "#", self.root))
        text = self.generator.read_text(encoding="utf-8")
        self.generator.write_text(text.replace("# All rights reserved.\n", ""), encoding="utf-8")
        self.assertFalse(checker.has_expected_header(checker.DERIVED_BSD_PATH, "#", self.root))

    def test_derived_generator_requires_pinned_license(self):
        license_path = self.benchmark / "references/pto-kernels-LICENSE"
        license_path.write_text("The Clear BSD License\n", encoding="utf-8")
        self.assertFalse(checker.has_expected_header(checker.DERIVED_BSD_PATH, "#", self.root))
        license_path.unlink()
        self.assertFalse(checker.has_expected_header(checker.DERIVED_BSD_PATH, "#", self.root))


if __name__ == "__main__":
    unittest.main()
