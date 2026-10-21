# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Standalone production policy, resources and transactional refusal checks."""

import argparse
import json
import re
from pathlib import Path
from assess_whole_kernel_access import invoke, digest
from check_current_cut_services import fixture
from check_operation_current_cut_services import fixture as scalar_fixture
from check_service_pruning import profile
from reaching_access_research import require


BASE = "conservative-prefix=true physical-output=true"


def run(args, name, text, options, *, failure=None):
    source = args.output / (name + ".input.pto")
    source.write_text(text)
    flags = (
        ["--mlir-disable-threading", "--mlir-print-ir-after-failure"] if failure else []
    )
    result = invoke(
        [args.production_tool, "--pto-frontier-analysis=" + options, *flags, source],
        args.output / (name + ".pto"),
        35,
    )
    error = Path(result["stderr"]).read_text()
    if failure:
        require(result["status"] != "success" and failure in error, str(result) + error)
        require(
            "frontier-prefix-result " not in error, "published an uncommitted success"
        )
        # IR dump is the original owned module even if an earlier staged
        # function was successfully prepared/emitted before this refusal.
        dump = error.split("IR Dump After", 1)[-1] if "IR Dump After" in error else ""
        require("module" in dump, "missing failure atomicity dump")
        for command in ("pto.set_flag", "pto.wait_flag", "pto.barrier"):
            require(command not in dump, "partial synchronization escaped staging")
        require(
            dump.count("func.func") == text.count("func.func"),
            "function lost on refusal",
        )
    else:
        require(result["status"] == "success", str(result) + error)
        work = profile(result)
        require(work["protocol_certified"], "missing physical check")
        require("conservative-modeled-storage-coverage" in error, "missing quality")
        require("covers-selected-prefix-order" in error, "unsupported equality")
    return result


def diagnostic_variation(args):
    rows = []
    for pipes in (2, 3, 4):
        for classes in (1, 4, 8):
            sites = [dict(pipe=p, facts=[2] * classes) for p in range(pipes)]
            sites += [dict(pipe=p, facts=[1] * classes) for p in range(pipes)]
            data = dict(
                classes=classes,
                sites=sites,
                tree=[dict(kind="payload", site=i) for i in range(len(sites))],
                prune=True,
            )
            source = args.output / f"pipe-class-{pipes}-{classes}.json"
            source.write_text(json.dumps(data))
            result = invoke(
                [args.trace_tool, "--source-pipe-json", source],
                source.with_suffix(".out.json"),
                35,
            )
            require(result["status"] == "success", str(result))
            value = json.loads(Path(result["stdout"]).read_text())
            require(value["pruning"]["accepted"], str(value))
            rows.append(
                dict(
                    pipes=pipes,
                    classes=classes,
                    result=result,
                    metrics=value["metrics"],
                    pruning=value["pruning"],
                    roles=value["role_ids"],
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-tool", type=Path, required=True)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    text = re.sub(
        r"test.current_cut_ids = array<i64: [^>]*>, ", "", fixture(2, 3, "first", 4)
    )
    rows = []
    for pool in ("0,1", "3,5"):
        rows.append(run(args, "pool-" + pool, text, BASE + " allocation-ids=" + pool))
    rows.append(
        run(
            args,
            "insufficient",
            text,
            BASE + " allocation-ids=3",
            failure="resource-reservation",
        )
    )
    # Invalid options leave the module untouched before any preparation.
    for index, options in enumerate(
        (
            "prune-prefix-services=true",
            "conservative-prefix=true",
            BASE + " completion-only=true",
            BASE + " analysis-only=true",
            BASE + " deferred-output=true",
            BASE + " normalize-two-trip-loops=true",
        )
    ):
        rows.append(
            run(
                args,
                f"invalid-{index}",
                text,
                options,
                failure="conservative-prefix requires",
            )
        )
    rows.append(
        run(
            args,
            "unsupported-target",
            text.replace('"a3"', '"a5"'),
            BASE,
            failure="current-cut",
        )
    )
    scalar = scalar_fixture(2)
    rows.append(
        run(
            args,
            "reserved",
            scalar,
            BASE + " allocation-ids=7",
            failure="resource-reservation",
        )
    )
    # Same physical protocol and alias contract as operation-admission regression.
    rows.append(run(args, "scalar", scalar, BASE + " gm-alias=may-not-alias"))
    distinct = (
        scalar_fixture(1, transfer=True)
        .replace(
            "(%bounds: !pto.ptr<f32, gm>)",
            "(%bounds: !pto.ptr<f32, gm>, %output: !pto.ptr<f32, gm>)",
        )
        .replace("pto.make_tensor_view %bounds", "pto.make_tensor_view %output")
    )
    rows.append(
        run(
            args,
            "visibility-may-alias",
            distinct,
            BASE + " gm-alias=may-alias",
            failure="scalar-gm-visibility",
        )
    )
    rows.append(
        run(args, "visibility-disjoint", distinct, BASE + " gm-alias=may-not-alias")
    )
    rows.append(run(args, "readonly-may-alias", scalar, BASE + " gm-alias=may-alias"))
    # First function succeeds inside staging, second has unqualified control.
    second = text[text.index("func.func") : text.rfind("}")].replace(
        "@points", "@other"
    )
    second = second.replace(
        " return",
        " %unready = scf.execute_region -> index {\n scf.yield %one : index\n }\n return",
    )
    multi = text[: text.rfind("}")] + second + "\n}\n"
    rows.append(run(args, "late-refusal", multi, BASE, failure="structured-control"))
    # Production must not consume the diagnostic-only synthetic pool attribute.
    ignored = text.replace(
        "attributes {pto.kernel_kind",
        "attributes {test.current_cut_ids = array<i64: 99>, pto.kernel_kind",
    )
    rows.append(run(args, "test-attribute-ignored", ignored, BASE))
    result = dict(
        runs=rows,
        variable_pipes_classes=diagnostic_variation(args),
        tool_sha256=digest(args.production_tool),
        trace_tool_sha256=digest(args.trace_tool),
    )
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "production prefix: policy, pools, transactional refusals and variable-class checks passed"
    )


if __name__ == "__main__":
    main()
