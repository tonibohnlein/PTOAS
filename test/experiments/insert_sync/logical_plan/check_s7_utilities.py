#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Test metadata/accounting/corpus helpers only; not native or hardware proof."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from s6_report import validate_report
from s7_corpus import (SCHEMA, ARMS, compact_lock, discover, manifest_digest,
                       prepare_bytes, summarize, validate_lock,
                       validate_manifest, verify_discovery)


def main():
    checks = 0
    def expect_bad(function, *args):
        nonlocal checks
        try:
            function(*args)
        except (ValueError, KeyError):
            checks += 1
            return
        raise AssertionError("invalid record accepted")
    lane = dict(core="AIC", pipe="M")
    report = dict(schema="oahs.s6.plan.v1", producer="structured", new_hardware_elision=True,
                  intrinsic_accumulator_requirements="1", retirement_sites=1,
                  generated_static_sites_including_retirement="1",
                  units=[dict(unit=0, period="1", wrapper_count=0,
                              hardware_contract="a2a3-mmad-acc-v1",
                              atoms=[dict(atom=0, lane=lane), dict(atom=1, lane=lane)],
                              actions=[], sites=[], deletion_audit=[], requirements=[dict(requirement=0,
                              source=0, target=1, property="accumulator-update", intrinsic_accumulator_order=True)])])
    assert validate_report(report)["intrinsic_accumulator_requirements"] == 1; checks += 1
    for key, value in (("property", "completion"), ("intrinsic_accumulator_order", "true"), ("target", 999)):
        bad = copy.deepcopy(report); bad["units"][0]["requirements"][0][key] = value
        expect_bad(validate_report, bad)
    bad = copy.deepcopy(report); bad["units"][0]["hardware_contract"] = "conservative"
    expect_bad(validate_report, bad)
    bad = copy.deepcopy(report); bad["units"][0]["atoms"][1]["lane"] = dict(core="AIC", pipe="MTE1")
    expect_bad(validate_report, bad)
    bad = copy.deepcopy(report); bad["intrinsic_accumulator_requirements"] = "2"
    expect_bad(validate_report, bad)
    manifest = dict(schema=SCHEMA, root=".", discovery=dict(all_a2a3=False,
                       preparation="test"), cases=[dict(id="one", path="x.pto", sha256="0"*64,
                       classification="production", stage="raw-addressed-compatibility", gm_contract="may-alias")])
    validate_manifest(manifest); checks += 1
    for key, value in (("stage", "unknown"), ("classification", "kernel-maybe"),
                       ("gm_contract", "all-gm-disjoint"), ("sha256", "bad")):
        bad = copy.deepcopy(manifest); bad["cases"][0][key] = value
        expect_bad(validate_manifest, bad)
    bad = copy.deepcopy(manifest); bad["cases"] *= 2
    expect_bad(validate_manifest, bad)
    bad = copy.deepcopy(manifest); bad["cases"][0]["stage"] = "prepared-pass-entry"
    expect_bad(validate_manifest, bad)
    bad["cases"][0]["preparation"] = dict(command=["actual", "pipeline"], compiler_sha256="a"*64)
    validate_manifest(bad); checks += 1
    text = b'module attributes {pto.target_arch = "a2a3"} { }'
    bound, changes = prepare_bytes(text, "raw-addressed-compatibility")
    assert bound == b'module attributes {pto.target_arch = "a3"} { }' and len(changes) == 1; checks += 1
    expect_bad(prepare_bytes, text, "prepared-pass-entry")
    assert prepare_bytes(bound, "prepared-pass-entry") == (bound, []); checks += 1
    rows = []
    for i in range(3):
        rows.append(dict(id=str(i), stage="raw-addressed-compatibility", classification="production",
                    arms={a:dict(status="applied") for a in ARMS}))
    rows[0]["arms"]["structured-hardware"] = dict(status="refused", reason="known")
    result = summarize(rows)["raw-addressed-compatibility"]
    assert result["pools"]["production"]["accepted"]["structured-hardware"] == 2
    assert result["existing_successful_hardware_refusals"] == {"known":1}; checks += 2
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); folder = root / "test/samples"; folder.mkdir(parents=True)
        (folder / "q_proj.pto").write_bytes(text)
        (folder / "weights_proj.pto").write_bytes(text.replace(b'a2a3', b'a5'))
        report = discover(root)
        assert len(report["cases"]) == 1 and "weights_proj" in report["missing_requested"]
        assert any(x.startswith("historical_gemm:") for x in report["missing_requested"])
        checks += 2
        portable = discover(root, portable=True)
        verified = verify_discovery(portable, root)
        assert verified["inputs"] == 1 and verified["manifest_sha256"] == manifest_digest(portable)
        checks += 2
        row = dict(portable["cases"][0], input_sha256="1"*64, adaptations=[],
                   arms={arm:dict(status="refused", reason="test") for arm in ARMS})
        result = dict(cases=[row], summary=summarize([row]),
                      missing_requested=portable["missing_requested"])
        lock = compact_lock(result, portable)
        assert validate_lock(lock, portable)["inputs"] == 1; checks += 1
        bad_lock = copy.deepcopy(lock); bad_lock["manifest_sha256"] = "f"*64
        expect_bad(validate_lock, bad_lock, portable)
    print(json.dumps(dict(status="passed", checks=checks, native="NOT_RUN", device="NOT_RUN")))


if __name__ == "__main__":
    main()
