# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Existing boundary and Boolean-replay challenges, extracted without planners."""
import unittest
from measure import children, replay
from compare_boundaries import Boundaries, compare as compare_boundaries
from observations import project
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto

class BoundaryTests(unittest.TestCase):
    @staticmethod
    def flag(source, target, key="0"):
        return {"src_pipe": source, "dst_pipe": target, "event_id": key}

    def test_delayed_publication_blocks_independent_work(self):
        schedules = []
        for late in (False, True):
            b = Boundaries()
            b.action("load", {}, "PIPE_MTE2", ["load A"])
            if not late:
                b.action("pto.set_flag", self.flag("PIPE_MTE2", "PIPE_V"))
            b.action("load", {}, "PIPE_MTE2", ["load B"])
            if late:
                b.action("pto.set_flag", self.flag("PIPE_MTE2", "PIPE_V", "7"))
            b.action("pto.wait_flag", self.flag("PIPE_MTE2", "PIPE_V", "7" if late else "0"))
            b.action("compute", {}, "PIPE_V", ["consume A"])
            schedules.append(b)
        difference = compare_boundaries(*schedules)
        self.assertEqual(len(difference), 1)
        self.assertEqual((difference[0]["target"], difference[0]["manual_prefix"],
                          difference[0]["automatic_prefix"]), (2, 0, 1))

    def test_prefixes_follow_transitive_handoffs_and_token_consumption(self):
        b = Boundaries()
        b.action("load", {}, "PIPE_MTE2", ["load"])
        b.action("pto.set_flag", self.flag("PIPE_MTE2", "PIPE_V"))
        b.action("pto.wait_flag", self.flag("PIPE_MTE2", "PIPE_V"))
        b.action("compute", {}, "PIPE_V", ["compute"])
        b.action("pto.set_flag", self.flag("PIPE_V", "PIPE_MTE3"))
        b.action("pto.wait_flag", self.flag("PIPE_V", "PIPE_MTE3"))
        b.action("store", {}, "PIPE_MTE3", ["store"])
        self.assertEqual(b.before[-1]["completed"], {"PIPE_MTE2": 0, "PIPE_V": 1})
        with self.assertRaisesRegex(ValueError, "without publication"):
            b.action("pto.wait_flag", self.flag("PIPE_V", "PIPE_MTE3"))

    def test_rearmed_key_captures_the_new_generation(self):
        b = Boundaries()
        key = self.flag("PIPE_MTE2", "PIPE_V")
        for generation in range(2):
            b.action("load", {}, "PIPE_MTE2", ["load", generation])
            b.action("pto.set_flag", key)
            with self.assertRaisesRegex(ValueError, "overwritten"):
                b.action("pto.set_flag", key)
            b.action("pto.wait_flag", key)
        self.assertEqual(b.completed["PIPE_V"]["PIPE_MTE2"], 1)

    def test_all_drain_reaches_a_lane_that_has_not_issued_yet(self):
        b = Boundaries()
        b.action("load", {}, "PIPE_MTE2", ["load"])
        b.action("pto.barrier", {"pipe": "PIPE_ALL"})
        b.action("compute", {}, "PIPE_V", ["compute"])
        self.assertEqual(b.before[-1]["completed"], {"PIPE_MTE2": 0})

class BooleanReplayTests(unittest.TestCase):
    def test_nested_sync_arithmetic_is_private(self):
        source = '''module { func.func @f(%n: index) {
          %z = arith.constant 0 : index
          %two = arith.constant 2 : index
          %ok = arith.cmpi sge, %n, %z : index
          scf.if %ok {
            %r = arith.remsi %n, %two : index
            %even = arith.cmpi eq, %r, %z : index
            scf.if %even { pto.barrier <PIPE_V> }
          }
          return
        } }'''
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(source)
            bare = ir.Module.parse('module { func.func @f(%n: index) { return } }')
            actual = project(module)
            self.assertEqual(actual["payload"], project(bare)["payload"])
            self.assertEqual(actual["sync_control"]["arith.remsi"], 1)
            # A result escaping the region is payload even when the region
            # also contains synchronization and otherwise pure arithmetic.
            escapes = ir.Module.parse('''module { func.func @f(%n: index, %ok: i1) -> index {
              %v = scf.if %ok -> (index) {
                %two = arith.constant 2 : index
                %r = arith.remsi %n, %two : index
                pto.barrier <PIPE_V>
                scf.yield %r : index
              } else { scf.yield %n : index }
              return %v : index
            } }''')
            payload = [row.get("op") for row in project(escapes)["payload"]]
            self.assertIn("arith.remsi", payload)
            self.assertIn("scf.if", payload)

    def test_signed_boolean_minmax(self):
        source = '''module { func.func @f(%p: i1) {
          %true = arith.constant true
          %false = arith.constant false
          %min = arith.minsi %p, %false : i1
          %max = arith.maxsi %p, %true : i1
          %equal = arith.cmpi eq, %min, %max : i1
          scf.if %equal { pto.barrier <PIPE_V> }
          scf.if %min { pto.barrier <PIPE_MTE2> }
          return
        } }'''
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(source)
            function = next(children(module.operation))
            for argument in (0, 1, -1):
                result = replay(function, [argument])
                self.assertEqual(result["counts"].get("pto.barrier", 0), 2 if argument else 1)
                pipes = [key for key in result["counts"] if key.startswith("barrier:")]
                self.assertEqual(any("PIPE_MTE2" in pipe for pipe in pipes), bool(argument))

    def test_negated_first_iteration_has_no_unmatched_wait(self):
        source = '''module { func.func @f() {
          %z = arith.constant 0 : index
          %one = arith.constant 1 : index
          %two = arith.constant 2 : index
          %true = arith.constant true
          scf.for %i = %z to %two step %one {
            %first = arith.cmpi eq, %i, %z : index
            %later = arith.xori %first, %true : i1
            scf.if %later { pto.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>] }
            scf.if %first { pto.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>] }
          }
          return
        } }'''
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(source)
            boundaries = Boundaries()
            result = replay(next(children(module.operation)), [], observer=boundaries.observe)
            self.assertEqual(result["counts"]["pto.set_flag"], 1)
            self.assertEqual(result["counts"]["pto.wait_flag"], 1)
            self.assertEqual(boundaries.tokens, {})

    def test_boolean_arguments_comparisons_and_signed_extension(self):
        source = '''module { func.func @f(%p: i1) {
          %true = arith.constant true
          %false = arith.constant false
          %minus = arith.constant -1 : i64
          %eq = arith.cmpi eq, %p, %true : i1
          %negative = arith.cmpi slt, %p, %false : i1
          %wide = arith.extsi %p : i1 to i64
          %extended = arith.cmpi eq, %wide, %minus : i64
          %a = arith.andi %eq, %negative : i1
          %b = arith.andi %a, %extended : i1
          scf.if %b { pto.barrier <PIPE_V> } else { pto.barrier <PIPE_MTE2> }
          return
        } }'''
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(source)
            function = next(children(module.operation))
            for argument in (0, 1, -1):
                result = replay(function, [argument])
                pipes = [key for key in result["counts"] if key.startswith("barrier:")]
                self.assertEqual(len(pipes), 1)
                self.assertIn("PIPE_V" if argument else "PIPE_MTE2", pipes[0])


class MultiTileReplayTests(unittest.TestCase):
    @staticmethod
    def source(allocation, shape="16x16xf16", selection="%slot"):
        multi = f"!pto.multi_tile_buf<vec, {shape}, count=2>"
        tile = f"!pto.tile_buf<vec, {shape}>"
        return f'''module {{ func.func @f(%slot: index) {{
          %base = arith.constant 1024 : i64
          %bad = arith.constant 3 : index
          %out_addr = arith.constant 4096 : i64
          %out = pto.alloc_tile addr = %out_addr : {tile}
          %slots = pto.alloc_multi_tile {allocation} : {multi}
          %selected = pto.multi_tile_get %slots[{selection}] : {multi} -> {tile}
          pto.tabs ins(%selected : {tile}) outs(%out : {tile})
          return
        }} }}'''

    def selected(self, source, slot):
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(source)
            observer = Boundaries()
            result = replay(next(children(module.operation)), [slot], observer=observer.observe)
            # Allocation and selection are descriptors, not asynchronous phases.
            self.assertEqual(len(observer.before), 1)
            self.assertEqual(observer.before[0]["lane"], "PIPE_V")
            self.assertEqual(result["counts"]["pto.multi_tile_get"], 1)
            return observer.payload[0][1][0]["local_tile"]

    def test_planned_order_and_dynamic_default_slot(self):
        source = self.source("{pto.multi_buffer_addrs = array<i64: 1024, 0>}")
        for slot, base in ((0, 1024), (1, 0), (2, 1024), (-1, 1024), (3, 1024)):
            self.assertEqual(self.selected(source, slot), {"scope": "vec", "begin": base, "bytes": 512})

    def test_explicit_aligned_stride(self):
        source = self.source("addr = %base")
        self.assertEqual(self.selected(source, 0)["begin"], 1024)
        self.assertEqual(self.selected(source, 1)["begin"], 1536)

    def test_unknown_geometry_is_not_replayed(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.selected(self.source("addr = %base", shape="8x16xf16"), 0)

    def test_constant_out_of_range_is_not_dynamic_fallback(self):
        # Native verification rejects this before replay can select a fallback.
        with self.assertRaisesRegex(ir.MLIRError, "constant slot 3 is out of range"):
            self.selected(self.source("addr = %base", selection="%bad"), 0)

if __name__ == "__main__":
    unittest.main()
