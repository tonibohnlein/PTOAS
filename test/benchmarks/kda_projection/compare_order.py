"""Finite explicit-command order diagnostic, not a memory or coupled FIFO oracle.

Payload identity must already pass native reconstruction against the unchanged
input. The caller supplies the imported engine mapping. Queue operations remain
opaque payload nodes: no peer, lowering-internal or target ACC edges are assumed.
All start/completion relations, including outward queue payloads, are compared.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from count_trace import execute, parse
from check_carried_slot_trace import Trace


class OrderTrace(Trace):
    def __init__(self, pipes):
        super().__init__()
        self.pipes = pipes
        self.order = []
        self.nodes = []
        self.names = []

    def vertex(self, parents):
        parents = list(parents)
        node = super().vertex(parents)
        bits = 0
        for parent in parents:
            bits |= self.order[parent]
        self.order.append(bits)
        return node

    def label(self, node):
        self.order[node] |= 1 << len(self.nodes)
        self.nodes.append(node)

    def operation(self, name):
        if name not in self.pipes:
            assert name in {
                "alloc_tile", "make_tensor_view", "partition_view",
                "set_validshape", "initialize_l2g2l_pipe",
            }, ("missing imported payload engine", name)
            return
        pipe = self.pipes[name]
        issued = self.issue(pipe)
        self.label(issued)
        done = self.vertex([issued])
        self.label(done)
        self.finishes.setdefault(pipe, []).append(done)
        self.names.append((name, pipe))

    def relations(self):
        return [self.order[node] & ~(1 << i) for i, node in enumerate(self.nodes)]


def run(path, function, bindings, pipes):
    lines = re.sub(r"%c-(\d+)", r"%cneg\1", path.read_text()).splitlines()
    start = next(i for i, line in enumerate(lines) if f"func.func @{function}(" in line) + 1
    nodes, _ = parse(lines, start)
    trace = OrderTrace(pipes)
    execute(nodes, dict(bindings), Counter(), trace)
    assert not trace.live, "unconsumed events at exit"
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--pipes", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-no-added", action="store_true",
                        help="Fail the host acceptance gate if any payload relation is added")
    args = parser.parse_args()
    bindings, pipes = json.loads(args.bindings), json.loads(args.pipes)
    before, after = [run(path, args.function, bindings, pipes)
                     for path in (args.baseline, args.candidate)]
    assert before.names == after.names, "payload sequence or engines changed"
    a, b = before.relations(), after.relations()
    examples = []
    for target, (old, new) in enumerate(zip(a, b)):
        added = new & ~old
        while added and len(examples) < 12:
            source = (added & -added).bit_length() - 1
            examples.append(dict(source_vertex=source, target_vertex=target,
                                 source_payload=before.names[source // 2],
                                 target_payload=before.names[target // 2]))
            added &= added - 1
    report = dict(
        scope=__doc__, function=args.function, bindings=bindings, pipes=pipes,
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        payloads=len(before.names), before=sum(x.bit_count() for x in a),
        after=sum(x.bit_count() for x in b),
        removed=sum((x & ~y).bit_count() for x, y in zip(a, b)),
        added=sum((y & ~x).bit_count() for x, y in zip(a, b)),
        added_examples=examples)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("payloads", "before", "after", "removed", "added")}))

    if args.require_no_added and report["added"]:
        raise SystemExit("ordering gate failed: candidate adds payload prerequisites")


if __name__ == "__main__":
    main()
