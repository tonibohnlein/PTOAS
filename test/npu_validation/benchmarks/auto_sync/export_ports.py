#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reproduce the nine benchmark ports, verify IR and synchronization-only delta.

Default: check registered files without modifying them. --install writes pairs
and updates their registry hashes after all pairs verify. This is an explicit
benchmark translation, not a TileLang importer or a production recognizer.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from port_conv import causal_conv1d_decode, causal_conv1d_prefill
from port_gating import fused_gating_delta
from port_groupnorm import group_norm
from port_lossless import lossless_block_cast
from port_persistent import persistent_gemm
from port_vector import elementwise_pipeline, gdn_chunk_cumsum, mhc_head_mix
from prepare import ROOT, executable, verify


FACTORIES = {
    "persistent_gemm": persistent_gemm, "group_norm": group_norm,
    "causal_conv1d_prefill": causal_conv1d_prefill, "causal_conv1d_decode": causal_conv1d_decode,
    "gdn_chunk_cumsum": gdn_chunk_cumsum, "mhc_head_mix": mhc_head_mix,
    "elementwise_pipeline": elementwise_pipeline, "gated_delta_rule": fused_gating_delta,
    "lossless_block_cast": lossless_block_cast,
}
SYNC = re.compile(r"\bpto\.(set_flag_dyn|wait_flag_dyn|barrier)\b")
ADAPTATIONS = {
    "persistent_gemm": ["Intrinsic GEMM's audited NZ L1 layout; full persistent source task mapping."],
    "group_norm": ["Serial no-padding FP32 factory only, not channel pipeline; compact column aliases for reductions."],
    "causal_conv1d_prefill": ["One sequence chunk per batch and four-token groups; positive lengths divisible by four.",
                              "PTO MulAddDst/Silu helpers expanded with A3 barriers and scratch storage."],
    "causal_conv1d_decode": ["PTO MulAddDst/Silu helpers expanded with A3 barriers and scratch storage.",
                             "Authored event ID 6 preserved, not certified for a six-ID target."],
    "gdn_chunk_cumsum": ["Original vector/scalar handoffs, eight chunks and one final store retained."],
    "mhc_head_mix": ["Forward only; PTO TSIGMOID mutates its source and has three V barriers."],
    "elementwise_pipeline": ["Source sub-M=32 schedule, two vector lanes and two UB banks."],
    "gated_delta_rule": ["Positive runtime sequence lengths; original state branch and scalar softplus branch.",
                         "PTO A3 reciprocal requires distinct storage: temporary, V barrier, copy back.",
                         "MulAddDst expanded; authored ID 6 preserved; upstream automatic-sync default not assumed."],
    "lossless_block_cast": ["Generic 64x64 output-scale blocks, not the default max4 optimized path.",
                            "A3 i32 AND: i32 mask, i16 aliases, TAND, V barrier, copy back.",
                            "Scale tile row padding to 32 bytes; reinterpretation uses shared physical storage."],
}


def without_sync(text):
    return "\n".join(line for line in text.splitlines() if not SYNC.search(line))


def make_record(case):
    port, config = FACTORIES[case["id"]]()
    pair = port if isinstance(port, dict) else {
        variant: port.render(variant == "expert") for variant in ("input", "expert")}
    if without_sync(pair["expert"]) != pair["input"].rstrip("\n"):
        raise ValueError(f"Payload/control/storage mismatch: {case['id']}")
    meta = {"source_commit": case["commit"], "source_files": case["sources"], "configuration": config,
            "adaptations": ADAPTATIONS[case["id"]],
            "baseline": "ported authored protocol; native upstream timing is a separate baseline",
            "invocation": "Terminal ALL completion added; input removes all replaceable local synchronization.",
            "gm_alias": ("may-not-alias; pointer arguments denote disjoint allocations, "
                         "except repeated uses of one argument"),
            "device_qualified": False, "launch_wrapper": "not yet provided",
            "comparison_gate": ("Validate expert, existing and frontier against "
                                "the independent reference before timing."),
            "source_reference": case["reference"]}
    return pair, json.dumps(meta, indent=2) + "\n"


def check_ir(opt, text, name):
    result = subprocess.run([opt, "--mlir-disable-threading", "-"], input=text, text=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60, check=False)
    if result.returncode:
        raise ValueError(f"IR verification failed for {name}:\n{result.stderr}")


def record_hash(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def apply_records(manifest, records, install):
    for case in manifest["benchmarks"]:
        name = case["id"]
        if name not in records:
            continue
        pair, meta = records[name]
        directory = ROOT / "prepared" / name
        files = {variant + ".pto": text for variant, text in pair.items()}
        files["port.json"] = meta
        for filename, text in files.items():
            path = directory / filename
            if install:
                directory.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            elif not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise ValueError(f"Regeneration mismatch: {path}")
        if install:
            case["status"] = "prepared_pto"
            case["prepared"] = {name: record_hash(directory / name) for name in ("input.pto", "expert.pto")}
            case["port_metadata"] = record_hash(directory / "port.json")
            case["device_qualified"] = False
    if install:
        (ROOT / "benchmarks.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pto-test-opt", default="pto-test-opt")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    try:
        manifest = json.loads((ROOT / "benchmarks.json").read_text(encoding="utf-8"))
        verify(manifest)
        opt = executable(args.pto_test_opt)
        records = {}
        for case in manifest["benchmarks"]:
            if case["id"] not in FACTORIES:
                continue
            pair, meta = make_record(case)
            for variant, text in pair.items():
                check_ir(opt, text, case["id"] + "/" + variant)
            records[case["id"]] = pair, meta
            print(f"Verified {case['id']}: IR and synchronization-only delta")
        apply_records(manifest, records, args.install)
        verify(manifest)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Port export failed: {error}\n")


if __name__ == "__main__":
    main()
