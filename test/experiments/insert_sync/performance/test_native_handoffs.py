# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Mutation tests for experiment acceptance, including unsupported observations."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

from run_native_handoffs import boundary_evidence, compare_evidence, population, require_effectiveness
from compare_boundaries import ObserverUnsupported


class HandoffAcceptance(unittest.TestCase):
    def test_population_keeps_all_frozen_cases(self):
        cases = population()
        self.assertEqual(len(cases), 19)
        for name in ("qk_matmul", "sv_matmul"):
            case = next(c for c in cases if c["case_id"] == name)
            self.assertEqual([len(s["arguments"]) for s in case["scenarios"]], [7] * 4)

    def test_contract_and_scenario_mutations_fail(self):
        evidence = {"projection": {k: ["original"] for k in ("payload", "allocations", "views", "abi", "sync_control")},
                    "metrics": {"scenarios": {"empty": {"payload_sha256": "same", "scalar_counts": {"arith.cmpi": 1}}}}}
        compare_evidence(evidence, copy.deepcopy(evidence))
        for key in evidence["projection"]:
            mutated = copy.deepcopy(evidence)
            mutated["projection"][key] = ["changed"]
            with self.assertRaises(ValueError):
                compare_evidence(evidence, mutated)
        for key in ("payload_sha256", "scalar_counts"):
            mutated = copy.deepcopy(evidence)
            mutated["metrics"]["scenarios"]["empty"][key] = "changed"
            with self.assertRaises(ValueError):
                compare_evidence(evidence, mutated)
        mutated = copy.deepcopy(evidence)
        mutated["metrics"]["scenarios"] = {}
        with self.assertRaises(ValueError):
            compare_evidence(evidence, mutated)

    def test_successful_fallback_is_not_effectiveness(self):
        row = {"case": "online_softmax", "arms": {a: {"metrics": {"scenarios": {"16": {
            "counts": {"pto.set_flag": 188, "pto.wait_flag": 188}}}}} for a in ("seed", "handoff")}}
        with self.assertRaisesRegex(ValueError, "improvement lost"):
            require_effectiveness(row)
        row["arms"]["handoff"]["metrics"]["scenarios"]["16"]["counts"] = {"pto.set_flag": 96, "pto.wait_flag": 96}
        require_effectiveness(row)

    def test_unknown_observer_is_explicit_but_tokens_are_hard_failure(self):
        scenario = [{"name": "one", "arguments": []}]
        with patch("compare_boundaries.run", side_effect=ObserverUnsupported("no qualified physical lane: fixture")):
            result = boundary_evidence(Path("seed"), Path("trial"), scenario)
        self.assertEqual(result[0]["status"], "unsupported")
        for reason in ("wait without publication", "undrained event tokens", "physical payload traces differ"):
            with patch("compare_boundaries.run", side_effect=ValueError(reason)):
                with self.assertRaises(ValueError):
                    boundary_evidence(Path("seed"), Path("trial"), scenario)

    def test_added_blocking_is_rejected(self):
        with patch("compare_boundaries.run", return_value=(object(), {})), patch("compare_boundaries.compare",
                return_value=[{"automatic_requires_later_prefix": True}]):
            with self.assertRaisesRegex(ValueError, "new blocking"):
                boundary_evidence(Path("seed"), Path("trial"), [{"name": "one"}])

    def test_unknown_in_one_arm_does_not_mask_hard_failure_in_other(self):
        for errors in ([ObserverUnsupported("unknown lane"), ValueError("wait without publication")],
                       [ValueError("physical payload differs"), ObserverUnsupported("unknown lane")]):
            with patch("compare_boundaries.run", side_effect=errors):
                with self.assertRaises(ValueError) as failure:
                    boundary_evidence(Path("seed"), Path("trial"), [{"name": "one"}])
                self.assertNotIsInstance(failure.exception, ObserverUnsupported)

    def test_named_function_selection_and_peer_refusal(self):
        from compare_boundaries import run
        source = '''module {
          func.func @first(%unused: index) { return }
          func.func @second() { return }
        }'''
        with tempfile.TemporaryDirectory(prefix="handoff-observer-") as scratch:
            path = Path(scratch) / "input.pto"
            path.write_text(source)
            run(path, {"function": "second", "arguments": []})
            with self.assertRaisesRegex(ValueError, "explicit scenario"):
                run(path, {"arguments": []})
            path.write_text(source.replace("func.func @second() { return }", "func.func @second() {\n"
                "call @first(%unused) : (index) -> ()\nreturn }").replace("func.func @second()", "func.func @second(%unused: index)"))
            with self.assertRaises(ObserverUnsupported):
                run(path, {"function": "second", "arguments": [0]})


if __name__ == "__main__":
    unittest.main()
