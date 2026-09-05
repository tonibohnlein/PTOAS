# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Regression tests for evidence accounting, not a hardware correctness oracle."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acceptance import classify_strict, summarize_acceptance
from campaign import inputs, probe_command
from census import classify_track, first_control_gate, population, summarize, topology_hash
from records import fields, new_function, parse_diagnostics
from provenance import untracked_sources
from native_followup import checked_path, successful_probe


def statistics(name, **counts):
    return json.dumps({"kind": "protocol-sync-statistics", "function": name, "counts": counts},
                      separators=(",", ":"))


class EvidenceTests(unittest.TestCase):
    """Do not let a missing dump or ambiguous scope become positive evidence."""

    def test_native_followup_requires_concrete_verdict(self):
        functions = [{"verdict": {"status": "rejected"}}]
        self.assertFalse(successful_probe("concrete", 0, functions, True))
        self.assertFalse(successful_probe("concrete", 0, [], True))
        functions[0]["verdict"]["status"] = "accepted"
        self.assertTrue(successful_probe("concrete", 0, functions, True))
        self.assertFalse(successful_probe("concrete", 0, functions, False))

    def test_native_followup_rejects_fallback_producer(self):
        functions = [{"statistics": {"producer": "protocol-plus-direct-residuals"}}]
        for kind in ("direct-cpp", "mixed-cpp"):
            self.assertTrue(successful_probe(kind, 0, functions, True))
        functions[0]["statistics"]["producer"] = "legacy-fallback"
        self.assertFalse(successful_probe("mixed-cpp", 0, functions, True))

    def test_followup_artifact_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "allowed").mkdir()
            (root / "outside").write_text("not an input", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escaped"):
                checked_path(root / "allowed", "../outside", "unused")

    def test_fields_preserve_lists_and_quoted_reasons(self):
        self.assertEqual(fields('families=[#1:slot=none, #2:slot=none] detail="two words"'),
                         {"families": "[#1:slot=none, #2:slot=none]", "detail": "two words"})

    def test_half_open_range_does_not_consume_next_list(self):
        self.assertEqual(fields("range=[0,32) families=[#0, #1]"),
                         {"range": "[0,32)", "families": "[#0, #1]"})

    def test_multi_function_records_and_verdicts(self):
        text = "\n".join([
            "PROTOCOL-SYNC concrete-verification function=@first status=accepted first-failed-stage=none",
            statistics("first", concrete_verifier_accepted=1),
            "PROTOCOL-SYNC concrete-verification function=@second status=rejected first-failed-stage=schedule",
            statistics("second", concrete_verifier_rejected=1),
        ])
        row = dict(parse_diagnostics(text), case_id="one", return_code=0)
        summary = summarize([row], "concrete")
        self.assertEqual(summary["totals"]["functions"], 2)
        self.assertEqual(summary["concrete_first_failed_stage"], {"none": 1, "schedule": 1})
        self.assertEqual(summary["programs_all_concrete_accepted"], 0)

    def test_missing_statistics_is_not_success(self):
        row = dict(parse_diagnostics("PROTOCOL-SYNC schedule function=@f"), case_id="one", return_code=0)
        self.assertEqual(summarize([row], "concrete")["totals"]["missing_statistics_rows"], 1)

    def test_lost_records_rejected(self):
        text = "PROTOCOL-SYNC storage-tracks function=@f\n" + statistics(
            "f", storage_transition_frontiers=1, storage_track_accesses_unprojected=0)
        with self.assertRaisesRegex(ValueError, "lost transitions"):
            parse_diagnostics(text)

    def test_duplicate_statistics_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate statistics"):
            parse_diagnostics(statistics("f") + "\n" + statistics("f"))

    def test_frontier_and_projection_provenance(self):
        text = "\n".join([
            "PROTOCOL-SYNC schedule function=@f",
            "  phase #0 op=pto.tload space=gm",
            "    access #1 read root=%arg0 space=gm",
            "PROTOCOL-SYNC storage-tracks function=@f",
            "  unprojected-access a#1 reason=non-physical-storage",
            "  storage-transition #0 raw-pair-members=[r#2, r#3] tracks=[t#0]",
            "    source-frontier=[pp0 guard=[]] target-frontier=[pp1 guard=[]] iteration-distance=0",
            "    target-query=directed-event checkpoint-e=not-applicable checkpoint-e-reason=none",
            statistics("f", storage_transition_frontiers=1, storage_track_accesses_unprojected=1),
        ])
        record = parse_diagnostics(text)["functions"][0]
        self.assertEqual(record["accesses"][0]["phase"], 0)
        self.assertEqual(record["unprojected"][0]["access"], 1)
        self.assertEqual(record["transitions"][0]["raw-pair-members"], "[r#2, r#3]")
        self.assertIn("guard=[]", record["transitions"][0]["frontier_lines"][0])

    def test_control_gate_uses_scan_order(self):
        function = new_function("f")
        function["regions"] = [{"kind": "loop", "cardinality": "zero-or-more", "op": "scf.for"},
                               {"kind": "loop", "cardinality": "exactly-once", "op": "scf.for"},
                               {"kind": "choice"}]
        self.assertEqual(first_control_gate(function), "control.multiple-loops")
        function["regions"][0]["op"] = "scf.while"
        self.assertEqual(first_control_gate(function), "control.loop-operation")

    def test_topology_excludes_function_and_operation_names(self):
        left, right = new_function("left"), new_function("right")
        left["phases"] = [{"id": 0, "core": "vector", "op": "pto.tabs"}]
        right["phases"] = [{"id": 0, "core": "vector", "op": "pto.tadd"}]
        self.assertEqual(topology_hash(left), topology_hash(right))
        right["phases"][0]["core"] = "cube"
        self.assertNotEqual(topology_hash(left), topology_hash(right))

    def test_repeated_function_names_are_distinct_instances(self):
        records = [{"case_id": case, "function": "f", "topology_hash": "same", "reason": "control"}
                   for case in ("one", "two")]
        self.assertEqual(population(records, "reason")["control"],
                         {"records": 2, "function_instances": 2, "function_names": 1, "topologies": 1})

    def test_unknown_roots_are_not_equivalent_views(self):
        self.assertEqual(classify_track(new_function("f"), {"id": 0, "families": "[#0, #1]"}),
                         "unresolved-root-or-scope")

    def test_sequential_and_cross_scope_are_only_candidates(self):
        function = new_function("f")
        function["families"] = [{"id": index, "root": f"%{index}", "root-lexical-scope": "#0"}
                                for index in (0, 1)]
        function["occurrences"] = [{"track": 0, "family": f"#{index}", "phase": f"#{index}",
                                    "guard": "[]", "loops": "[]"} for index in (0, 1)]
        track = {"id": 0, "families": "[#0, #1]"}
        self.assertEqual(classify_track(function, track), "sequential-use-candidate")
        function["families"][1]["root-lexical-scope"] = "#1"
        self.assertEqual(classify_track(function, track), "cross-scope-numeric-coincidence-candidate")


class AcceptanceTests(unittest.TestCase):
    """Admission is not inferred from absent diagnostics or aggregate failures."""

    def test_all_probes_share_architecture_and_alias_contract(self):
        args = SimpleNamespace(python=Path("python"), ptoas=Path("ptoas"),
                               input_root=Path("inputs"), results=Path("results"), arch="a2", gm_alias=None)
        for alias in (None, "may-alias", "assume-disjoint-arguments"):
            args.gm_alias = alias
            for strict in (False, True):
                command = probe_command(args, {"case_id": "one", "source": "kernel.pto"}, "plan", strict)
                self.assertIn("--pto-arch=a2", command)
                overrides = [flag for flag in command if flag.startswith("--protocol-sync-gm-alias=")]
                self.assertEqual(overrides, [] if alias is None else [f"--protocol-sync-gm-alias={alias}"])
                self.assertEqual("--protocol-sync-fallback=fail" in command, strict)

    def test_missing_statistics_or_crash_is_not_rejection(self):
        for return_code in (0, 1, -11, 124):
            self.assertEqual(classify_strict({"functions": [], "return_code": return_code}),
                             "incomplete-or-invocation-error")

    def test_noop_is_native_only_with_native_producer(self):
        function = {"statistics": {"planner_result": "no-op", "producer": "protocol-plus-direct-residuals"}}
        probe = {"return_code": 0, "functions": [function]}
        self.assertEqual(classify_strict(probe), "admitted")
        function["statistics"]["producer"] = "legacy-fallback-unsupported"
        self.assertEqual(classify_strict(probe), "incomplete-or-invocation-error")

    def test_canonical_local_precision_boundary_is_retained(self):
        text = ('PROTOCOL-SYNC selected-world function=@f\n'
                '  canonical-local atoms=0 requirements=0 covered=0 boundary="crosses a block"\n')
        record = parse_diagnostics(text)["functions"][0]["canonical_local"][0]
        self.assertEqual(record["boundary"], "crosses a block")

    def test_untracked_source_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory(prefix="protocol-sync-source-") as directory:
            root = Path(directory)
            (root / "lib").mkdir()
            source = root / "lib/new.cpp"
            source.write_text("first\n", encoding="utf-8")
            with patch("provenance.checked_output", return_value="lib/new.cpp\0"):
                first = untracked_sources(root)
                source.write_text("second\n", encoding="utf-8")
                self.assertNotEqual(first, untracked_sources(root))

    def test_overlapping_blockers_survive_semantic_gate(self):
        function = new_function("f")
        function["failures"] = [{"reason": "unsupported-effectful-operation", "op": "pto.set_validshape"}]
        function["statistics"] = {"generation_rejections": {"conflicting-physical-ranges": 2},
                                  "residual_obligations_by_kind": {"unknown-alias": 3}}
        strict_function = dict(new_function("f"), statistics={"producer": "fail-closed-policy"})
        strict = {"return_code": 1, "functions": [strict_function]}
        row = {"case_id": "one", "source": "kernel.pto", "input_sha256": "hash", "functions": [function],
               "strict_mixed": strict, "empty_world": {"return_code": 0}}
        summary = summarize_acceptance([row])
        self.assertEqual(summary["strict_status"], {"rejected": 1})
        self.assertEqual(len(summary["kernels_per_observed_blocker"]), 3)
        self.assertEqual(sum(summary["observed_blocker_occurrences"].values()), 6)


class ManifestTests(unittest.TestCase):
    """A corpus identity mismatch must fail before compilation."""

    def check_manifest(self, body, expected_error):
        with tempfile.TemporaryDirectory(prefix="protocol-sync-manifest-") as directory:
            root = Path(directory)
            (root / "kernel.pto").write_text("module {}\n", encoding="utf-8")
            manifest = root / "manifest.tsv"
            manifest.write_text(body, encoding="utf-8")
            args = SimpleNamespace(mode="storage", manifest=manifest, input_root=root, expected_rows=1)
            with self.assertRaisesRegex(ValueError, expected_error):
                inputs(args)

    def test_hash_mismatch(self):
        self.check_manifest("case_id\tsource\tsha256\nk0\tkernel.pto\twrong\n", "manifest hash")

    def test_duplicate_case(self):
        self.check_manifest("case_id\tsource\nk0\tkernel.pto\nk0\tkernel.pto\n", "duplicate case")

    def test_invalid_case_path(self):
        self.check_manifest("case_id\tsource\n../k0\tkernel.pto\n", "invalid or duplicate")

    def test_invalid_level(self):
        self.check_manifest("case_id\tsource\tlevel\nk0\tkernel.pto\tguessed\n", "unsupported input pipeline")


if __name__ == "__main__":
    unittest.main()
