# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Check source-derived samples without importing torch or executing drivers."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from frontend_samples import discover_driver_calls
from frontend_worker import materialize_samples


def calls(body, prefix="import torch\nN = 16\n"):
    text = prefix + 'if __name__ == "__main__":\n'
    text += "\n".join("    " + line for line in body.splitlines())
    return discover_driver_calls(text, {"kernel"})


class FrontendSampleTests(unittest.TestCase):
    def test_call_order_reassignment_and_alias_identity(self):
        rows = calls("a = torch.zeros((N, N), dtype=torch.float32)\n"
                     "b = a\nkernel(a, b)\n"
                     "a = torch.zeros((N // 2, N), dtype=torch.float16)\nkernel(a, b)")
        self.assertEqual(rows[0]["arguments"][0], rows[0]["arguments"][1])
        self.assertEqual(rows[0]["arguments"][0]["shape"], [16, 16])
        self.assertEqual(rows[1]["arguments"][0]["shape"], [8, 16])
        self.assertEqual(rows[1]["arguments"][1]["shape"], [16, 16])

    def test_samples_are_fixed_meta_tensors_not_source_constructor_calls(self):
        row = calls("a = torch.full((N, N), 7, dtype=torch.float32)\nkernel(a, a, 3)")[0]
        seen = []

        def empty(shape, *, dtype, device):
            seen.append((shape, dtype, device))
            return object()

        with patch.dict(sys.modules, {"torch": SimpleNamespace(empty=empty, float32="f32")}):
            arguments, keywords = materialize_samples(row)
        self.assertIs(arguments[0], arguments[1])
        self.assertEqual(arguments[2], 3)
        self.assertEqual(keywords, {})
        self.assertEqual(seen, [((16, 16), "f32", "meta")])

    def test_unknown_reassignment_and_mutation_stop_segment(self):
        for statement in ("a = arbitrary()", "a.resize_(7)", "arbitrary(a)"):
            with self.subTest(statement=statement):
                rows = calls("a = torch.zeros((N, N), dtype=torch.float32)\n" + statement + "\nkernel(a)")
                self.assertEqual(rows[0]["status"], "needs-driver-adapter")

    def test_no_implicit_dtype_layout_device_or_expansion(self):
        for constructor in ("torch.zeros((N, N))", "torch.zeros((N, N), dtype=torch.float32, device='npu')",
                            "torch.zeros((N, N), **kwargs)", "torch.empty((N, N), dtype=dynamic_dtype)"):
            with self.subTest(constructor=constructor):
                self.assertEqual(calls(f"a = {constructor}\nkernel(a)")[0]["status"], "needs-driver-adapter")

    def test_import_identity_and_rebinding(self):
        good = calls("a = th.zeros((16, 16), dtype=th.float32)\nkernel(a)", "import torch as th\n")
        self.assertEqual(good[0]["status"], "source-samples")
        for prefix in ("import other as torch\n", "import torch\ntorch = 1\n",
                       "import torch\nimport other as torch\n", "import torch\ndef torch(): pass\n"):
            self.assertEqual(calls("a = torch.zeros((16, 16), dtype=torch.float32)\nkernel(a)", prefix)[0]["status"],
                             "needs-driver-adapter")
        self.assertEqual(calls("kernel = 1\nkernel(3)")[0]["status"], "needs-driver-adapter")

    def test_conditional_and_loop_calls_remain_unresolved(self):
        rows = calls("if condition:\n    kernel(1)\nfor i in range(3):\n    kernel(i)\nkernel(2)")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "needs-driver-adapter" for row in rows))

    def test_arithmetic_rank_and_size_limits(self):
        for shape in ("(2**1000, 1)", "(2147483648 * 2147483648, 1)", "(1 // 0, 1)", "(-1, 16)",
                      "(0, 16)", "(16777217, 1)", "(65536, 65536)", "(1,1,1,1,1,1,1,1,1)"):
            self.assertEqual(calls(f"a = torch.empty({shape}, dtype=torch.float32)\nkernel(a)")[0]["status"],
                             "needs-driver-adapter")

    def test_config_override_is_explicit_and_not_executed(self):
        rows = calls("cfg = RunConfig()\nkernel.lower(3, config=cfg)\nkernel.compile(4, config=RunConfig())",
                     "from pypto.runtime import RunConfig\n")
        self.assertEqual([row["config_override"] for row in rows], ["cfg", "RunConfig()"])
        self.assertTrue(all(row["keywords"] == {} for row in rows))
        self.assertEqual(calls("kernel(3, config=arbitrary())")[0]["status"], "needs-driver-adapter")

    def test_literal_scalars_preserved_not_runtime_markers(self):
        rows = calls("kernel.lower(3, scale=0.5)\nkernel.specialize(5, scale=1.0)")
        self.assertEqual([row["arguments"] for row in rows], [[3], [5]])
        self.assertEqual([row["keywords"]["scale"] for row in rows], [0.5, 1.0])

    def test_assert_success_path_does_not_execute_tensor_computation(self):
        rows = calls("a = torch.ones((N, N), dtype=torch.float32)\n"
                     "assert torch.allclose(a, a + a)\nkernel(a)")
        self.assertEqual(rows[0]["status"], "source-samples")
        self.assertEqual(calls("assert arbitrary()\nkernel(1)")[0]["status"], "needs-driver-adapter")

    def test_module_mutations_and_unknown_bindings_poison_samples(self):
        for mutation in ("a.resize_(4,4)", "N += 1", "N: int = 17", "if True:\n    N = 17",
                         "from other import fake as torch", "x, y = (1,2)", "a.shape = (4,4)"):
            prefix = "import torch\nN=16\na=torch.zeros((2,2), dtype=torch.float32)\n" + mutation + "\n"
            rows = calls("kernel(torch.zeros((N,), dtype=torch.float32))", prefix)
            self.assertEqual(rows[0]["status"], "needs-driver-adapter", mutation)

    def test_effectful_call_arguments_invalidate_following_calls(self):
        rows = calls("a=torch.zeros((2,2), dtype=torch.float32)\nkernel(a.resize_(7))\nkernel(a)")
        self.assertTrue(all(row["status"] == "needs-driver-adapter" for row in rows))

    def test_nested_driver_calls_are_all_recorded(self):
        rows = calls("kernel(kernel(3))\nkernel(4)")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "needs-driver-adapter" for row in rows))

    def test_indirect_returned_callable_is_not_a_direct_entry(self):
        rows = calls("kernel(1)(2)\nkernel(3)")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "needs-driver-adapter" for row in rows))
        self.assertEqual(rows[0]["invocation"], "kernel(1)")
        row = calls("kernel(3, config=RunConfig()())", "from pypto.runtime import RunConfig\n")[0]
        self.assertEqual(row["status"], "needs-driver-adapter")

    def test_unknown_assertion_objects_are_not_assumed_readonly(self):
        self.assertEqual(calls("assert imported_flag\nkernel(1)")[0]["status"], "needs-driver-adapter")

    def test_destructured_call_result_retains_both_calls(self):
        rows = calls("a, b = kernel(1)\nkernel(2)")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "source-samples")
        self.assertEqual(rows[1]["status"], "needs-driver-adapter")

    def test_definition_time_effects_are_not_ignored(self):
        for definition in ("def helper(x=a.resize_(7)): pass", "class C:\n    a.resize_(7)",
                           "@arbitrary(a)\ndef helper(): pass", "def helper(x: arbitrary(a)): pass"):
            prefix = "import torch\na=torch.zeros((2,2),dtype=torch.float32)\n" + definition + "\n"
            self.assertEqual(calls("kernel(a)", prefix)[0]["status"], "needs-driver-adapter")
        prefix = "import torch\ndef helper():\n    arbitrary()\n"
        self.assertEqual(calls("kernel(1)", prefix)[0]["status"], "source-samples")


if __name__ == "__main__":
    unittest.main()
