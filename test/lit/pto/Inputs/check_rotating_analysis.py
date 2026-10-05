# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Execute emitted guards and compare with physical conflicts, without unfolding in production."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native
from check_rotating_extraction import require


def validate(document, slots, lower, step, upper):
    require(not document["error"] and document["prepared"], document)
    require(document["unchanged_preparation"], "preparation modified IR")
    trace = document["trace"]
    require(not trace["error"], trace)
    count = max(0, (upper-lower+step-1)//step)
    require(trace["outer_trips"] == count and trace["payloads"] == count*2, trace)
    pipes = document["periodic"]["payloads"]
    require(len(pipes) == 2 and pipes[0] != pipes[1], pipes)
    rotating = [a for a in document["fragments"] if a["slots"] == slots]
    require(len(rotating) == 2, rotating)
    for item in rotating:
        require(item["stride"] == step%slots and item["offset"] == lower%slots, item)
    dynamic_pipes = pipes*count
    edges = native(dynamic_pipes)
    effects = []
    for iteration in range(count):
        slot = (lower + step*iteration)%slots
        effects.extend([({("mat",0)}, {("left",slot)}),
                        ({("left",slot),("right",0)}, {("acc",0)})])
    for a, (reads, writes) in enumerate(effects):
        for b in range(a+1,len(effects)):
            later_reads,later_writes = effects[b]
            if writes & (later_reads|later_writes) or reads & later_writes:
                edges.add((2*a+1,2*b))
    required,_ = closure(2*len(effects),edges)
    commands=[]
    seen=0
    for event in trace["events"]:
        if event["kind"] == "payload":
            require((event["type"],event["ordinal"]) == (seen%2,seen//2), event)
            seen+=1
        elif event["kind"] == "barrier" and event["pipe"] == 6:
            require(event["gap"] == count*2, event)
        else:
            require(event["gap"] == seen,event)
            command=dict(event)
            if event["kind"] != "barrier":
                command["identity"]=(event["plan"],event["record"],event["source_ordinal"])
            commands.append(command)
    actual=closure_with_commands(dynamic_pipes,commands)
    require(actual == [row & ~(1<<i) for i,row in enumerate(required)],
            (document["function"],"emitted synchronization changed required order"))


def main():
    tool, fixture = sys.argv[1:]
    source=Path(fixture).read_text()
    output=subprocess.run([tool,"--rotating-analysis",fixture],check=True,capture_output=True,text=True)
    documents=[json.loads(line) for line in output.stdout.splitlines() if line.startswith("{")]
    configs=[(2,0,1,0),(2,0,1,1),(2,0,1,2),(2,0,1,9),(3,3,2,20),(3,3,2,2),(2,3,2,14)]
    require(len(documents)==len(configs),output.stdout)
    for document,config in zip(documents,configs):
        validate(document,*config)
    # The production pass retains loop/payload structure and leaves IDs logical.
    inserted=subprocess.run([tool,"--insert-logical",fixture],check=True,capture_output=True,text=True)
    for name in ("scf.for", "pto.textract", "pto.tmatmul"):
        require(inserted.stdout.count(name)==source.count(name),name)
    require("pto.logical_set" in inserted.stdout and "pto.logical_wait" in inserted.stdout,inserted.stdout)
    require("pto.set_flag" not in inserted.stdout, "physical allocation ran unexpectedly")
    prefix=source[:source.index("module attributes")]
    one=re.search(r"  func.func @ring_many.*?(?=\n  func.func)",source,re.S).group()
    boundary=one.replace("    scf.for", "    %outside = pto.alloc_tile addr = %base : !left\n"
        "    pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%outside : !left)\n    scf.for")
    overlap=one.replace("    scf.for", "    %alias = pto.alloc_tile addr = %base : !left\n    scf.for")
    overlap=overlap.replace("      pto.tmatmul",
        "      pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%alias : !left)\n"
        "      pto.tmatmul")
    with tempfile.TemporaryDirectory(prefix="rotating-reject-") as directory:
        for text in (boundary,overlap):
            path=Path(directory)/"case.pto"
            path.write_text(prefix+'module attributes {pto.target_arch = "a3"} {\n'+text+'\n}\n')
            result=subprocess.run([tool,"--rotating-analysis",str(path)],check=True,capture_output=True,text=True)
            doc=json.loads(result.stdout)
            require(not doc["prepared"] and doc["unchanged_preparation"],doc)
            rejected=subprocess.run([tool,"--insert-logical",str(path)],capture_output=True,text=True)
            require(rejected.returncode != 0,"unsupported whole-region obligation accepted")
    print("rotating analysis:7 runtime-bound loops, exact command closure and matching guards passed")


if __name__ == "__main__":
    main()
