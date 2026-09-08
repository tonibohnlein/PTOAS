# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Read-only export, honest unsupported status and stable source identities."""
import json
from pathlib import Path
import subprocess
import sys

binary, source, prefix = sys.argv[1:]
original = Path(source).read_text()


def run(name, text, export, nested=False):
    path = Path(prefix + "." + name + ".pto")
    path.write_text(text)
    directory = Path(prefix + "." + name + ".facts")
    options = "buffer-generations=true defer-same-pipe=true"
    if export:
        options += " handoff-facts-dir=" + str(directory)
    pipeline = ("--pass-pipeline=builtin.module(builtin.module(func.func(pto-insert-sync{" + options + "})))"
                if nested else "--pto-insert-sync=" + options)
    result = subprocess.run([binary, str(path), "--mlir-disable-threading",
        "--mlir-print-op-on-diagnostic=false", pipeline],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    records = [json.loads(p.read_text()) for p in directory.glob("*.json")] if export else []
    return result.stdout, records


baseline, _ = run("off", original, False)
enabled, records = run("on", original, True)
assert enabled == baseline, "export changed compiled output"
assert len(records) == 1, records
record = records[0]
assert record["status"] == "supported-local-projection", record
assert record["native_transformation_proof"] is False
assert record["native"]["abstract_nodes_are_occurrences"] is False
assert record["gm_contract"] == "may-alias"
assert record["native"]["omitted_effects"]
assert len(record["model"]["statements"]) == 3
assert any(s["iterators"] for s in record["model"]["statements"])
assert all(a["definite"] is False for s in record["model"]["statements"] for a in s["writes"])
again, repeated = run("repeat", original, True)
assert again == enabled and repeated == records, "nondeterministic export"

# The scalar adapter does not guess a loop-local unknown predicate, or export
# wrapping arithmetic as mathematical arithmetic.
unsupported = original.replace("scf.if %read", "scf.if %unknown").replace(
    "      scf.if", "      %unknown = arith.trunci %i : index to i1\n      scf.if")
# Use a valid index-to-i64 cast followed by truncation.
unsupported = unsupported.replace("%unknown = arith.trunci %i : index to i1",
    "%wide = arith.index_cast %i : index to i64\n      %unknown = arith.trunci %wide : i64 to i1")
off, _ = run("unknown.off", unsupported, False)
on, rejected = run("unknown.on", unsupported, True)
assert off == on
assert rejected[0]["status"] == "unsupported" and "model" not in rejected[0]
unsigned = original.replace("    }\n    return", "    } {unsignedCmp}\n    return")
off, _ = run("unsigned.off", unsigned, False)
on, rejected = run("unsigned.on", unsigned, True)
assert off == on
assert rejected[0]["status"] == "unsupported" and "model" not in rejected[0]
assert "a3" in record["parent_module_attributes"]
manual = original.replace("    return", "    pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]\n"
    "    pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]\n    return")
off, _ = run("manual.off", manual, False)
on, bypass = run("manual.on", manual, True)
assert off == on and bypass[0]["status"] == "bypassed"
assert bypass[0]["reason"] == "explicit synchronization" and "model" not in bypass[0]
# Two identical symbols in distinct nested modules need two distinct records,
# including when the contexts otherwise have the same attributes.
body = original.split('module attributes {pto.target_arch = "a3"} {', 1)[1].rsplit("}", 1)[0]
aliases = original.split('module attributes {pto.target_arch = "a3"} {', 1)[0]
nested = aliases + 'module { module @left attributes {pto.target_arch = "a3"} {\n' + body + (
    '} module @right attributes {pto.target_arch = "a3"} {\n') + body + '} }\n'
off, _ = run("nested.off", nested, False, nested=True)
on, nested_records = run("nested.on", nested, True, nested=True)
assert off == on and len(nested_records) == 2
assert len({item["context_md5"] for item in nested_records}) == 2
descriptor = (Path(source).parent / "insert_sync_descriptor_flow.pto").read_text()
for label, value, full in (("full", "%size", True), ("partial", "%partial", False), ("unknown", "%trips", False)):
    text = descriptor.replace("pto.set_validshape %a, %size, %size", "pto.set_validshape %a, " + value + ", %size")
    off, _ = run("descriptor." + label + ".off", text, False)
    on, captures = run("descriptor." + label + ".on", text, True)
    assert off == on and captures[0]["status"] == "supported-local-projection"
    writes = [access for statement in captures[0]["model"]["statements"] for access in statement["writes"]
              if access["address"] == "0"]
    assert writes and all(access["definite"] is False for access in writes)
    assert all(access["native_whole_productions"] == full for access in writes)
helper = '''module {
  func.func private @opaque(!pto.tile_buf<vec, 16x16xf16>)
      attributes {pto.tileop.kind = "vector"}
  func.func @missing_helper(%a: !pto.tile_buf<vec, 16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    func.call @opaque(%a) : (!pto.tile_buf<vec, 16x16xf16>) -> ()
    return
  }
}'''
off, _ = run("helper.off", helper, False)
on, captures = run("helper.on", helper, True)
assert off == on
caller = next(r for r in captures if r["function"] == "missing_helper")
assert caller["status"] == "unsupported" and "model" not in caller
assert "helper" in caller["reason"]
print("Native handoff facts: output parity, looping/skipped-reader occurrences, "
      "explicit GM boundary, conservative writes, deterministic IDs, target provenance, "
      "unsupported unsigned loop and predicate, explicit-sync bypass, nested contexts "
      "full/partial/unknown descriptors and unmodeled helper rejection passed")
