# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mutation checks for the payload/allocation evidence, without compilation."""
import unittest

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from run_qwen_additions import pass_entry, project


SOURCE = """module {
  func.func @sample() {
    %zero = arith.constant 0 : i64
    %address = arith.constant 64 : i64
    %a = pto.alloc_tile addr = %zero : !pto.tile_buf<vec, 1x16xf32>
    %b = pto.alloc_tile addr = %address : !pto.tile_buf<vec, 1x16xf32>
    pto.tabs ins(%a : !pto.tile_buf<vec, 1x16xf32>) outs(%b : !pto.tile_buf<vec, 1x16xf32>)
    return
  }
}
"""


class QwenEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.context = ir.Context()
        self.context.__enter__()
        self.context.enable_multithreading(False)
        pto.register_dialect(self.context, load=True)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def projection(self, text):
        module = ir.Module.parse(text)
        module.operation.verify()
        return project(module)

    def test_sync_and_ssa_renaming_preserve_payload(self):
        changed = SOURCE.replace("    return", "    pto.barrier <PIPE_V>\n    return")
        changed = changed.replace("%a ", "%renamed ").replace("%a :", "%renamed :")
        left, right = self.projection(SOURCE), self.projection(changed)
        self.assertEqual(left["payload"], right["payload"])
        self.assertEqual(left["allocations"], right["allocations"])
        self.assertNotEqual(left["placements"], right["placements"])

    def test_address_change_reaches_allocation_contract(self):
        left = self.projection(SOURCE)
        right = self.projection(SOURCE.replace("constant 64 :", "constant 128 :"))
        self.assertNotEqual(left["allocations"], right["allocations"])

    def test_payload_operand_change_is_detected(self):
        left = self.projection(SOURCE)
        right = self.projection(SOURCE.replace("ins(%a :", "ins(%b :"))
        self.assertNotEqual(left["payload"], right["payload"])

    def test_sync_only_control_is_separate_and_guard_sensitive(self):
        control = """    %ready = arith.cmpi slt, %zero, %address : i64
    scf.if %ready {
      pto.barrier <PIPE_V>
    }
"""
        original = self.projection(SOURCE)
        changed = self.projection(SOURCE.replace("    return", control + "    return"))
        opposite = self.projection(SOURCE.replace("    return", control.replace("slt", "sge") + "    return"))
        self.assertEqual(original["payload"], changed["payload"])
        self.assertEqual(original["allocations"], changed["allocations"])
        self.assertEqual(changed["sync_control"], {"arith.cmpi": 1, "scf.if": 1})
        self.assertNotEqual(changed["placements"], opposite["placements"])

    def test_payload_guard_cannot_be_hidden_as_sync_control(self):
        payload = next(line for line in SOURCE.splitlines() if "pto.tabs" in line)
        guarded = SOURCE.replace(payload, "    %ready = arith.cmpi slt, %zero, %address : i64\n"
                                 "    scf.if %ready {\n" + payload + "\n    }")
        self.assertNotEqual(self.projection(SOURCE)["payload"], self.projection(guarded)["payload"])

    def test_sync_only_constant_is_recorded_without_hiding_payload_constants(self):
        control = """    %one = arith.constant 1 : i64
    %ready = arith.cmpi ne, %zero, %one : i64
    scf.if %ready {
      pto.barrier <PIPE_V>
    }
"""
        changed = SOURCE.replace("    return", control + "    return")
        original, guarded = self.projection(SOURCE), self.projection(changed)
        self.assertEqual(original["payload"], guarded["payload"])
        self.assertEqual(guarded["sync_control"], {"arith.constant": 1, "arith.cmpi": 1, "scf.if": 1})
        self.assertNotEqual(guarded["placements"], self.projection(changed.replace("constant 1 :", "constant 2 :"))["placements"])
        shared = changed.replace("    %one = arith.constant 1 : i64\n", "")
        shared = shared.replace("    %zero =", "    %one = arith.constant 32 : i64\n    %zero =")
        shared = shared.replace("addr = %address", "addr = %one")
        self.assertNotEqual(original["allocations"], self.projection(shared)["allocations"])

    def test_diagnostic_text_is_not_pass_entry_ir(self):
        function = "func.func @sample() {\n  return\n}"
        log = ("// -----// IR Dump Before PTOInsertSync (pto-insert-sync) //----- //\n" +
               function + "\n[InsertSync retained repair] group=1\nsource: diagnostic\n")
        result = pass_entry(log)
        self.assertNotIn("retained repair", result)
        ir.Module.parse(result).operation.verify()


if __name__ == "__main__":
    unittest.main()
