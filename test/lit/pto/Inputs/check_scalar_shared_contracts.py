# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check shared scalar contracts without introducing a helper-name registry."""
import json
import shutil
import subprocess
import sys


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing synchronization test tool")
    result = subprocess.run([tool, "--recognize", sys.argv[2]], check=True,
                            capture_output=True, text=True, timeout=60)
    reports = {d["function"]: d for line in result.stdout.splitlines()
               if line.startswith("{") for d in [json.loads(line)]}
    report = reports["scalar_protocol"]
    operations = {p["operation"]: p for p in report["payloads"]}
    for name in ("pto.declare_local_array", "pto.local_array_get", "pto.local_array_set",
                 "func.call", "pto.comm.tnotify", "pto.comm.twait"):
        assert operations[name]["pipe"] == 0, (name, operations)
    assert {e["mode"] for e in operations["func.call"]["effects"]} == {"read", "write"}
    assert any(i["issue"] == "unmodeled-operation" for n in reports["unaccounted_storage"]["nodes"]
               for a in n["attempts"] for i in a["issues"])
    assert {e["mode"] for e in operations["pto.comm.twait"]["effects"]} == {"read", "write"}
    assert all(not p["effects"] for p in report["payloads"] if "local_array" in p["operation"])
    assert all(a["state"] == "applicable" for n in report["nodes"] for a in n["attempts"])
    assert any(i["issue"] == "unmodeled-operation" for n in reports["unresolved_call"]["nodes"]
               for a in n["attempts"] for i in a["issues"])
    assert all(i["category"] == "unmet-obligation" for n in reports["unresolved_call"]["nodes"]
               for a in n["attempts"] for i in a["issues"] if i["issue"] == "unmodeled-operation")
    release = next(p for p in reports["fifo_release"]["payloads"] if p["operation"] == "pto.tfree")
    assert {e["mode"] for e in release["effects"]} == {"read", "write"}, release
    boundaries = reports["completion_boundaries"]["payloads"]
    assert len(boundaries) == 1 and boundaries[0]["operation"] == "pto.syncall", boundaries
    assert {e["mode"] for e in boundaries[0]["effects"]} == {"read", "write"}, boundaries
    quant = {p["operation"]: p for p in reports["quantized_publication"]["payloads"]}
    assert not quant["pto.set_quant_vector"]["effects"], quant
    assert {(e["space"], e["mode"]) for e in quant["pto.tpush"]["effects"]} == {
        (5, "read"), (8, "read")}, quant
    print("shared scalar contracts: native results, signal accesses, stack state and unresolved callee passed")


if __name__ == "__main__":
    main()
