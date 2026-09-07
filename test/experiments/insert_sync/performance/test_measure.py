#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mutation controls for benchmark accounting, independent of InsertSync output."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from generate_controls import generate
from generate_kernel_pairs import conv2d, flash_cube, strip_local_sync, topk
from generate_recurrence_pairs import GENERATORS, triangular_inverse, wy
from measure import children, normalize_gm_pipe_assembly, replay, static_metrics
from run import compare
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto


class AccountingTests(unittest.TestCase):
    def test_kernel_pairs_reproduce_and_only_remove_local_sync(self):
        root = Path(__file__).parent
        manifest = json.loads((root / "kernel-pairs-manifest.json").read_text())
        generated = {"topk_128": topk(), "conv2d_interior": conv2d(), "flash_attention_cube": flash_cube()}
        generated.update({name: generate() for name, generate in GENERATORS.items()})
        for case in manifest["cases"]:
            manual = (root / case["sources"]["manual"]["path"]).read_text()
            automatic = (root / case["sources"]["auto"]["path"]).read_text()
            self.assertEqual(generated[case["case_id"]], manual)
            self.assertEqual(strip_local_sync(manual), automatic)
            with ir.Context() as context:
                pto.register_dialect(context, load=True)
                for source in (manual, automatic):
                    self.assertTrue(ir.Module.parse(source).operation.verify())
            for source in case["sources"].values():
                self.assertEqual(hashlib.sha256((root / source["path"]).read_bytes()).hexdigest(), source["sha256"])
        for source in manifest["reference_files"] + manifest["support_files"]:
            self.assertEqual(hashlib.sha256((root / source["path"]).read_bytes()).hexdigest(), source["sha256"])

    def test_fifo_operations_survive_local_sync_removal(self):
        manual = flash_cube()
        automatic = strip_local_sync(manual)
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            before = static_metrics(ir.Module.parse(manual).operation)
            after = static_metrics(ir.Module.parse(automatic).operation)
            expected = {"pto.initialize_l2g2l_pipe": 3, "pto.talloc": 2, "pto.tpush": 2, "pto.tpop": 1, "pto.tfree": 1}
            self.assertEqual(before["fixed_queue_operations"], expected)
            self.assertEqual(after["fixed_queue_operations"], expected)
            self.assertEqual(after["counts"], {})
            # A slot-entry release selects the device credit-return overload;
            # the entry-less TFREE implementation in the pinned ISA is a no-op.
            module = ir.Module.parse(automatic)
            def descendants(op):
                yield op
                for child in children(op):
                    yield from descendants(child)
            free = next(op for op in descendants(module.operation) if op.name == "pto.tfree")
            self.assertEqual(len(free.operands), 2)
            self.assertIn("tensor_view", str(free.operands[0].type))

    def test_fifo_assembly_bridge_preserves_operation_attributes(self):
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            fixture = flash_cube().replace("operandSegmentSizes =", "nosplit = false, operandSegmentSizes =")
            source = ir.Module.parse(fixture).operation.get_asm()
            converted, count = normalize_gm_pipe_assembly(source)
            self.assertEqual(count, 3)
            self.assertEqual(ir.Module.parse(converted).operation.get_asm(), source)
            unknown = source.replace("nosplit = false}", "nosplit = false, unknown = 1}")
            self.assertEqual(normalize_gm_pipe_assembly(unknown), (unknown, 0))

    def test_conv2d_release_follows_third_extraction(self):
        source = conv2d()
        for slot in range(2):
            arm = source.split(f"scf.if %pactive{slot} {{", 1)[1]
            release = arm.index(f"pto.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID{slot}>]")
            self.assertEqual(arm[:release].count("func.call @benchmark_conv_img2col"), 3)
            self.assertEqual(arm[:release].count("func.call @benchmark_conv_extract_weight"), 3)
        self.assertIn("func.call @benchmark_conv_setfmatrix", strip_local_sync(source))

    def test_wy_peer_counts_and_empty_stripe_participation(self):
        # Counts are per peer. Hardware's two-vector quorum is not simulated.
        for kda in (False, True):
            ready = (10, 11) if kda else (2, 1)
            free = (12, 13) if kda else (3, 4)
            source = wy(kda)
            with ir.Context() as context:
                pto.register_dialect(context, load=True)
                manual_module = ir.Module.parse(source)
                automatic_module = ir.Module.parse(strip_local_sync(source))
                manual = list(children(manual_module.operation))
                automatic = list(children(automatic_module.operation))
                for chunks in (0, 1, 3):
                    for tail in (False, True):
                        args = ["k", "v", "beta", "g", "a", "ws2", "ws1", "u", "w", "ffts", chunks, tail]
                        cube = replay(manual[1], args)
                        self.assertEqual(cube["payload_sha256"], replay(automatic[1], args)["payload_sha256"])
                        for event in ready:
                            self.assertEqual(cube["counts"].get(f"pto.sync.wait:{event} : i32", 0), chunks)
                        for event in free:
                            self.assertEqual(cube["counts"].get(f"pto.sync.set:{event} : i32", 0), chunks)
                        for stripe in (0, 1):
                            vector = replay(manual[0], args + [stripe])
                            self.assertEqual(
                                vector["payload_sha256"], replay(automatic[0], args + [stripe])["payload_sha256"]
                            )
                            for event in ready:
                                self.assertEqual(vector["counts"].get(f"pto.sync.set:{event} : i32", 0), chunks)
                            for event in free:
                                expected = chunks if kda else max(chunks - 1, 0)
                                self.assertEqual(vector["counts"].get(f"pto.sync.wait:{event} : i32", 0), expected)
                            self.assertEqual(vector["counts"].get("pto.tstore", 0), 2 * chunks)
                            if chunks and tail and stripe == 1:
                                full = replay(manual[0], args[:-1] + [False, stripe])
                                self.assertEqual(
                                    full["counts"]["pto.tload"] - vector["counts"]["pto.tload"], 3 if kda else 1
                                )

    def test_fixed_cross_signal_mutation_is_visible(self):
        source = strip_local_sync(wy(True))
        changed = source.replace("pto.sync.set <PIPE_MTE3>, 10 {ffts_mode = 2 : i32}", "", 1)
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            before = ir.Module.parse(source)
            after = ir.Module.parse(changed)
            self.assertNotEqual(
                static_metrics(before.operation)["fixed_protocol_sha256"],
                static_metrics(after.operation)["fixed_protocol_sha256"],
            )
            args = ["k", "v", "beta", "g", "a", "ws2", "ws1", "u", "w", "ffts", 1, True, 1]
            self.assertNotEqual(
                replay(next(children(before.operation)), args)["payload_sha256"],
                replay(next(children(after.operation)), args)["payload_sha256"],
            )

    def test_triangular_scalar_payload_and_recurring_credits(self):
        source = triangular_inverse()
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            manual_module = ir.Module.parse(source)
            automatic_module = ir.Module.parse(strip_local_sync(source))
            manual = next(children(manual_module.operation))
            automatic = next(children(automatic_module.operation))
            for matrices in (0, 1, 3):
                before = replay(manual, ["src", "dst", matrices])
                after = replay(automatic, ["src", "dst", matrices])
                self.assertEqual(before["payload_sha256"], after["payload_sha256"])
                self.assertEqual(before["counts"]["pto.set_flag"], before["counts"]["pto.wait_flag"])
                # 16 basis vectors; 15 + sum(0..14) scalar-dependent AXPYs.
                self.assertEqual(before["counts"].get("pto.taxpy", 0), 120 * matrices)
                self.assertEqual(before["counts"].get("pto.tgetval", 0), 120 * matrices)
                self.assertEqual(before["counts"].get("pto.tload", 0), matrices)
                self.assertEqual(before["counts"].get("pto.tstore", 0), matrices)
                self.assertEqual(before["counts"].get("pto.barrier", 0), 105 * matrices)

    def evaluate(self, source, trips):
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(source)
            function = next(children(module.operation))
            return replay(function, ["src", "dst", trips]), static_metrics(module.operation)

    def test_generated_sources_are_frozen(self):
        root = Path(__file__).parent / "inputs"
        for name, depth, uses in (
            ("one_buffer", 1, 1),
            ("two_buffer", 2, 1),
            ("three_buffer", 3, 1),
            ("four_use", 2, 4),
        ):
            for arm, source in zip(("auto", "manual"), generate(depth, uses)):
                self.assertEqual(source, (root / f"{name}.{arm}.pto").read_text())

    def test_prime_steady_drain_formulas_and_matched_payload(self):
        for depth in (1, 2, 3):
            for uses in (1, 4):
                auto, manual = generate(depth, uses)
                for trips in (0, 1, depth, depth + 1, 9):
                    result, _ = self.evaluate(manual, trips)
                    expected = 2 * depth + trips * (2 + 2 * uses)
                    self.assertEqual(result["counts"]["pto.set_flag"], expected)
                    self.assertEqual(result["counts"]["pto.wait_flag"], expected)
                    self.assertEqual(result["counts"].get("pto.tabs", 0), trips * uses)
                    self.assertEqual(result["payload_sha256"], self.evaluate(auto, trips)[0]["payload_sha256"])

    def test_missing_wait_changes_actions_but_not_payload(self):
        source = generate(2)[1]
        mutated = source.replace("pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]", "", 1)
        original, _ = self.evaluate(source, 3)
        changed, _ = self.evaluate(mutated, 3)
        self.assertEqual(original["payload_sha256"], changed["payload_sha256"])
        self.assertNotEqual(original["action_trace_sha256"], changed["action_trace_sha256"])
        self.assertEqual(original["counts"]["pto.wait_flag"] - changed["counts"]["pto.wait_flag"], 2)

    def test_moving_wait_detected_with_equal_counts(self):
        source = generate(1)[1]
        line = "        pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]"
        mutated = source.replace(line + "\n", "", 1)
        anchor = next(line for line in mutated.splitlines() if "pto.tabs" in line)
        mutated = mutated.replace(anchor, anchor + "\n" + line, 1)
        original, static = self.evaluate(source, 2)
        changed, moved = self.evaluate(mutated, 2)
        self.assertEqual(original["counts"], changed["counts"])
        self.assertEqual(original["payload_sha256"], changed["payload_sha256"])
        self.assertNotEqual(original["action_trace_sha256"], changed["action_trace_sha256"])
        self.assertNotEqual(static["placement_sha256"], moved["placement_sha256"])

    def test_changed_allocation_detected(self):
        source = generate(2)[0]
        changed = source.replace("arith.constant 512 : i64", "arith.constant 4096 : i64")
        self.assertNotEqual(
            self.evaluate(source, 3)[0]["payload_sha256"], self.evaluate(changed, 3)[0]["payload_sha256"]
        )

    def test_unknown_scalar_rejected(self):
        source = generate(1)[0].replace("arith.remui", "arith.maxsi")
        with self.assertRaisesRegex(ValueError, "unsupported operation"):
            self.evaluate(source, 1)

    def test_baseline_gate_rejects_changes_and_population_mismatch(self):
        baseline = {
            "arch": "a3",
            "manifest_sha256": "input",
            "rows": {"case/staged": {"status": "pass", "metrics": {"static": {"counts": 3}}, "audit_verdicts": []}},
        }
        self.assertEqual(compare(baseline, copy.deepcopy(baseline)), [])
        mutation = copy.deepcopy(baseline)
        mutation["rows"]["case/staged"]["metrics"]["static"]["counts"] = 2
        self.assertTrue(compare(baseline, mutation))
        mutation["manifest_sha256"] = "different"
        with self.assertRaises(ValueError):
            compare(baseline, mutation)


if __name__ == "__main__":
    unittest.main()
