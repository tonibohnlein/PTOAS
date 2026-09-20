#!/usr/bin/env python3
"""Discriminating tests for the benchmark's protocol checker."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from check import run_ir
from generate import CASES, generate

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("oracle",here.parents[1]/"oahs/check_carried_slot_trace.py")
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)

class Checks(unittest.TestCase):
    def run_text(self,text):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"input.pto";path.write_text(text)
            return run_ir(path,0,oracle,True,1)

    def test_original_rearming_is_not_certified(self):
        _,trace,_=self.run_text(generate(CASES["smoke"],True))
        self.assertEqual(len(trace.rearming_failures),2)

    def test_banked_keys_supply_rearming(self):
        _,trace,_=self.run_text(generate(CASES["smoke"],True,True))
        self.assertFalse(trace.rearming_failures)
        self.assertGreater(trace.required_checks,0)

    def test_removing_readiness_is_rejected(self):
        text=generate(CASES["smoke"],True,True)
        text="\n".join(s for s in text.splitlines() if not
                       ("pto." in s and "_flag[<PIPE_MTE2>, <PIPE_MTE1>" in s))
        with self.assertRaisesRegex(AssertionError,"missing physical conflict"):
            self.run_text(text)

    def test_extra_drain_adds_payload_order(self):
        original=generate(CASES["smoke"],True,True)
        changed=original.replace("    pto.textract", "    pto.barrier <PIPE_ALL>\n    pto.textract")
        def relation(text):
            _,trace,issues=self.run_text(text)
            return {(i,j) for j,issue in enumerate(issues) for i,(_,done,_,_) in enumerate(trace.payloads)
                    if trace.ancestors[issue] & (1<<done)}
        a,b=relation(original),relation(changed)
        self.assertTrue(a < b)

if __name__=="__main__":unittest.main()
