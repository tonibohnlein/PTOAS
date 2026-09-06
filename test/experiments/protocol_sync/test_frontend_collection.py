# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Check source inventory and artifact accounting without loading a frontend."""

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from collect_frontend import harvest_inputs, run_entry, worker_environment
from frontend_inventory import discover_entries
from frontend_worker import load_entry_module, runtime_scalars


class FrontendCollectionTests(unittest.TestCase):
    def test_source_driver_sibling_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "collection_fixture"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "entry.py").write_text("from collection_sibling import value\n", encoding="utf-8")
            (package / "collection_sibling.py").write_text("value = 7\n", encoding="utf-8")
            with patch.object(sys, "path", list(sys.path)), patch.object(sys, "argv", list(sys.argv)):
                with patch.dict(sys.modules):
                    module = load_entry_module(root, Path("collection_fixture/entry.py"))
                    self.assertEqual(module.value, 7)
                    self.assertEqual(sys.argv, [str(package / "entry.py")])

    def test_codegen_and_math_workers_are_bounded(self):
        args = SimpleNamespace(pypto_root=Path("pypto"), lib_root=Path("lib"), results=Path("results"))
        environment = worker_environment(args)
        for key in ("PYPTO_CODEGEN_MAX_WORKERS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(environment[key], "1")

    def test_decorators_are_inspected_not_executed(self):
        source = "@pl.jit(side_effect())\ndef kernel(x):\n    return x\n"
        self.assertEqual(discover_entries(source), [
            {"entry": "kernel", "kind": "jit", "line": 2, "top_level": True}])

    def test_nested_entries_remain_unresolved_inventory_seeds(self):
        source = "def factory(n):\n    @pl.jit\n    def kernel(x):\n        return x\n    return kernel\n"
        entry = discover_entries(source)[0]
        self.assertEqual(entry["entry"], "factory.kernel")
        self.assertFalse(entry["top_level"])
        row = run_entry(None, dict(entry, draft=False))
        self.assertEqual(row["status"], "needs-construction-adapter")
        self.assertEqual(row["inputs"], [])

    def test_helpers_are_not_independent_entry_points(self):
        source = "@pl.jit.inline\ndef helper(x):\n    return x\n@pl.jit.host\ndef host(x):\n    return helper(x)\n"
        self.assertEqual([entry["entry"] for entry in discover_entries(source)], ["host"])

    def test_drafts_remain_explicit(self):
        row = run_entry(None, {"draft": True, "top_level": True})
        self.assertEqual(row["status"], "declared-draft")

    def test_syntax_errors_are_not_factory_failures(self):
        row = run_entry(None, {"kind": "syntax-error", "draft": False, "top_level": False})
        self.assertEqual(row["status"], "source-syntax-error")

    def test_runtime_scalars_do_not_invent_specialization_values(self):
        def kernel(x: "pl.Tensor", count: "pl.Scalar[pl.INT32]", offset: "pl.Scalar" = 3):
            return x, count, offset

        marker = object()
        self.assertEqual(runtime_scalars(SimpleNamespace(_func=kernel), SimpleNamespace(RUNTIME=marker)),
                         {"count": marker})

    def test_partial_generated_outputs_keep_parent_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            output = root / "outputs/seed"
            output.mkdir(parents=True)
            (output / "kernel.pto").write_text("module {}\n", encoding="utf-8")
            rows = harvest_inputs(root, output, {"case_id": "seed"})
            self.assertEqual(rows[0]["parent_seed"], "seed")
            self.assertEqual(rows[0]["level"], "level3")
            self.assertEqual((root / rows[0]["source"]).read_text(encoding="utf-8"), "module {}\n")

    def test_generated_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            output = root / "outputs"
            output.mkdir()
            (root / "outside.pto").write_text("module {}\n", encoding="utf-8")
            (output / "escape.pto").symlink_to(root / "outside.pto")
            with self.assertRaisesRegex(ValueError, "escapes"):
                harvest_inputs(root, output, {"case_id": "seed"})


if __name__ == "__main__":
    unittest.main()
