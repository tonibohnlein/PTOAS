# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check shared-cell adaptation, exact covers, boundary state, and actual insertion."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from check_explicit_reduction import occurrence, access, conflicts, closure, require


def check_graph(document):
    occurrences = [occurrence(i, item["pipe"], [access(*a) for a in item["accesses"]])
                   for i, item in enumerate(document["occurrences"])]
    edges = conflicts(occurrences)
    reachable = closure(occurrences, edges)
    for source, row in enumerate(document["event_reachable"]):
        expected = [source == target or target in reachable[source] for target in range(2 * len(occurrences))]
        require(row == expected, (document["function"], source, row, expected))
    require(document["invalid_event_rejected"], document)
    require(not document["span_error"] and document["span_retained"] == document["retained"], document)
    require(document.get("gap_rejected", True) and document.get("reverse_rejected", True), document)
    covers = {(a, b) for a, b in edges
              if not any(2*b in reachable[z] for z in reachable[2*a+1])}
    require(covers == set(map(tuple, document["retained"])), document)
    labels = list(dict.fromkeys(item["pipe"] for item in occurrences))
    totals = dict.fromkeys(labels, 0)
    ranks = []
    for item in occurrences:
        totals[item["pipe"]] += 1
        ranks.append(totals[item["pipe"]])
    for b, row in enumerate(document["start_ranks"]):
        expected = [max([ranks[a] for a, item in enumerate(occurrences)
                         if item["pipe"] == p and 2*b in reachable[2*a+1]] or [0]) for p in labels]
        require(row == expected, (row, expected))
    for cell in document["boundary"]:
        atom = cell["atom"]
        writes, readers = [], []
        for i, item in enumerate(occurrences):
            same = [a for a in item["accesses"] if a["atom"] == atom]
            if any(a["write"] for a in same):
                writes.append(i)
            elif any(a["read"] for a in same):
                readers.append(i)
        require(cell["first_writer"] == (writes[0] if writes else None), cell)
        require(cell["last_writer"] == (writes[-1] if writes else None), cell)
        first, last = {}, {}
        for i in readers:
            if not writes or i < writes[0]:
                first.setdefault(occurrences[i]["pipe"], i)
            if not writes or i > writes[-1]:
                last[occurrences[i]["pipe"]] = i
        require(dict(cell["first_readers"]) == first and dict(cell["last_readers"]) == last, cell)


def main():
    tool, fixture = sys.argv[1:]
    run = subprocess.run([tool, "--explicit-analysis", fixture], check=True, capture_output=True, text=True)
    documents = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    require(len(documents) == 9, run.stdout)
    text = Path(fixture).read_text()
    prefix = text[:text.index("module attributes")]
    functions = re.findall(r"  func.func @\w+.*?(?=\n  func.func|\n}\n)", text, re.S)
    with tempfile.TemporaryDirectory(prefix="explicit-analysis-") as tmp:
        for document, function in zip(documents, functions):
            name = document["function"]
            if name != "reject_symbolic":
                require(not document["error"], document)
                check_graph(document)
            else:
                require(bool(document["error"]), document)
            path = Path(tmp)/"case.pto"
            path.write_text(prefix + 'module attributes {pto.target_arch = "a3"} {\n' + function + '\n}\n')
            inserted = subprocess.run([tool, "--insert-logical", str(path)], capture_output=True, text=True)
            require((inserted.returncode != 0) == name.startswith("reject_"), (name, inserted.stderr))
            if inserted.returncode:
                continue
            cross = sum(document["occurrences"][a]["pipe"] != document["occurrences"][b]["pipe"]
                        for a,b in document["retained"])
            require(inserted.stdout.count("pto.logical_set") == cross, inserted.stdout)
            require(inserted.stdout.count("pto.logical_wait") == cross, inserted.stdout)
            local = len(document["retained"]) - cross
            require(inserted.stdout.count("pto.barrier") == local + bool(document["occurrences"]), inserted.stdout)
            seen = 0
            local_targets = {b for a,b in document["retained"]
                             if document["occurrences"][a]["pipe"] == document["occurrences"][b]["pipe"]}
            for line in inserted.stdout.splitlines():
                command = re.search(r"pto.logical_(set|wait).* record (\d+) ordinal", line)
                if command:
                    a, b = document["retained"][int(command[2])]
                    require(seen == (a + 1 if command[1] == "set" else b), (name, seen, line))
                if "pto.barrier" in line and "PIPE_ALL" not in line:
                    require(seen in local_targets, (name, seen, line))
                if re.search(r"pto\.(tsetval|tgetval|textract|tmatmul|load)([ .]|$)", line):
                    seen += 1
            require(seen == len(document["occurrences"]), (name, seen))
            for opname in ("pto.tsetval", "pto.tgetval", "pto.textract", "pto.tmatmul"):
                require(inserted.stdout.count(opname) == function.count(opname), (name, opname))
    require(documents[1]["retained"] == [[0,1],[1,2]], documents[1])
    require(documents[2]["retained"] == [], documents[2])
    require(documents[3]["retained"] == [[0,1]], documents[3])
    protected = next(d for d in documents if d["function"] == "accumulation")
    require(any(a[3] for item in protected["occurrences"] for a in item["accesses"]), protected)
    require(protected["retained"] == [[1, 2]], protected)
    print("explicit analysis: shared ranges, views, roots, boundaries, hardware protection and insertion passed")


if __name__ == "__main__":
    main()
