# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Keep overlapping observed blockers, independently of the first emission gate."""

from collections import Counter


def classify_strict(probe):
    """A normal fail-closed rejection is distinct from a crash or invocation error."""
    stats = [function["statistics"] for function in probe["functions"] if "statistics" in function]
    if probe["return_code"] == 0 and stats and all(
            record.get("planner_result") in {"materialized-mixed", "no-op"}
            and record.get("producer") == "protocol-plus-direct-residuals" for record in stats):
        return "admitted"
    if probe["return_code"] == 1 and any(record.get("producer") == "fail-closed-policy" for record in stats):
        return "rejected"
    return "incomplete-or-invocation-error"


def observed_blockers(row):
    """Do not infer invisible downstream proofs from an aborted strict invocation."""
    blockers = Counter()
    for function in row["functions"]:
        for failure in function["failures"]:
            blockers[f"extraction.{failure['reason']}:{failure['op']}"] += 1
        stats = function.get("statistics", {})
        for histogram in ("generation_rejections", "channel_rejections", "residual_obligations_by_kind"):
            for reason, count in stats.get(histogram, {}).items():
                if count:
                    blockers[f"{histogram}.{reason}"] += count
    for function in row["empty_world"].get("functions", []):
        for kind, count in function.get("statistics", {}).get("residual_obligations_by_kind", {}).items():
            if count:
                # Separate probe identity avoids adding repeated observations
                # as though they were distinct semantic obligations.
                blockers[f"empty-world.{kind}"] += count
    for function in row["strict_mixed"]["functions"]:
        for rejection in function["direct_rejections"]:
            blockers[f"direct-repair.{rejection['reason']}"] += 1
    return dict(blockers)


def summarize_acceptance(rows):
    """Return a per-program matrix; counts overlap and cannot predict admission."""
    records = [{"case_id": row["case_id"], "source": row["source"],
                "target_arch": row.get("target_arch", "a3"),
                "gm_alias_override": row.get("gm_alias_override"),
                "input_sha256": row["input_sha256"], "strict_status": classify_strict(row["strict_mixed"]),
                "blockers": observed_blockers(row),
                "empty_world_return_code": row["empty_world"]["return_code"]} for row in rows]
    kernels = Counter()
    occurrences = Counter()
    for record in records:
        kernels.update(record["blockers"].keys())
        occurrences.update(record["blockers"])
    return {"records": records, "strict_status": dict(Counter(r["strict_status"] for r in records)),
            "kernels_per_observed_blocker": dict(kernels), "observed_blocker_occurrences": dict(occurrences),
            "scope": "All exposed diagnostics, not a complete proof universe behind failed extraction; "
                     "empty-world residual kinds can also appear in admitted programs."}
