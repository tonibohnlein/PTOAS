#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for the changed-code compliance prefilter."""

from pathlib import Path
import tempfile
import unittest

from check_changed_code import ChangedLine, scan_cpp_control_braces


class ControlBraceTest(unittest.TestCase):
    def scan(self, source: str, changed_line: int = 1):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = Path("input.cpp")
            (repo / path).write_text(source, encoding="utf-8")
            line = source.splitlines()[changed_line - 1]
            return scan_cpp_control_braces(
                repo, [ChangedLine(path, changed_line, line)])

    def test_accepts_nested_multiline_condition_with_braces(self):
        findings = self.scan(
            "if (value == items.end() ||\n"
            "    other()) {\n"
            "  return;\n"
            "}\n")
        self.assertEqual(findings, [])

    def test_accepts_brace_on_following_line(self):
        findings = self.scan("while (ready())\n{\n  run();\n}\n")
        self.assertEqual(findings, [])

    def test_rejects_inline_unbraced_body(self):
        findings = self.scan("if (ready()) return;\n")
        self.assertEqual(len(findings), 1)

    def test_rejects_next_line_unbraced_body(self):
        findings = self.scan("for (int value : values)\n  use(value);\n")
        self.assertEqual(len(findings), 1)

    def test_checks_bare_else_and_do(self):
        self.assertEqual(self.scan("else\n{\n  recover();\n}\n"), [])
        self.assertEqual(len(self.scan("do\n  run();\n")), 1)


if __name__ == "__main__":
    unittest.main()
