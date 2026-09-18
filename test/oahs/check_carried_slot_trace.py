# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent concrete ordering check for the unchanged Shenggan fixture.

This test interpreter supports only the fixture's scalar syntax. It checks the
actual emitted event words and concrete local physical intervals, not constructor
claims. It does not execute numerical payloads or model device latency.
GM inputs/output are distinct in this fixture; numerical addressing is not an
additional device-visibility premise or a general PTO interpreter.
"""
import re
import sys


def parse(lines, at):
    nodes = []
    while at < len(lines):
        line = lines[at].strip()
        if line.startswith("}"):
            return nodes, at
        if line.endswith("{"):
            body, end = parse(lines, at + 1)
            other = []
            if lines[end].strip() == "} else {":
                other, end = parse(lines, end + 1)
            nodes.append((line, body, other))
            at = end + 1
        else:
            nodes.append((line, [], []))
            at += 1
    raise AssertionError("unterminated original region")


class Trace:
    def __init__(self, serialize=False, check_outer=False, check_bank=False, serialize_parent=False):
        self.ancestors = []
        self.launch = {}
        self.gate = {}
        self.finishes = {}
        self.live = {}
        self.consumed = {}
        self.payloads = []
        self.overlap_checks = 0
        self.required_checks = 0
        self.serialize = serialize
        self.native_acc = {}
        self.native_acc_checks = 0
        self.check_outer = check_outer
        self.outer_checks = 0
        self.sync_counts = {}
        self.check_bank = check_bank
        self.serialize_parent = serialize_parent
        self.bank_checks = 0
        self.early_ready_checks = 0

    def vertex(self, parents):
        bits = 0
        for p in parents:
            bits |= self.ancestors[p] | (1 << p)
        self.ancestors.append(bits)
        return len(self.ancestors) - 1

    def before(self, pipe):
        return [self.launch[pipe]] if pipe in self.launch else []

    def issue(self, pipe):
        parents = self.before(pipe)
        if pipe in self.gate:
            parents.append(self.gate[pipe])
        issue = self.vertex(parents)
        self.launch[pipe] = issue
        return issue

    def sync(self, kind, source, observer=None, key=None):
        identity = (kind, source, observer)
        self.sync_counts[identity] = self.sync_counts.get(identity, 0) + 1
        if kind == "barrier" and source == "ALL":
            issued = self.vertex(list(self.launch.values()) + list(self.gate.values()))
            done = self.vertex([issued] + [x for fs in self.finishes.values() for x in fs])
            for pipe in list(self.launch):
                self.launch[pipe] = issued
                self.gate[pipe] = done
                self.finishes[pipe].append(done)
            return
        pipe = observer if kind == "wait_flag" else source
        issued = self.issue(pipe)
        parents = [issued]
        identity = (source, observer, key)
        if kind in ("set_flag", "barrier"):
            parents += self.finishes.get(source, [])
        else:
            assert identity in self.live, ("unseeded wait", identity)
            parents.append(self.live.pop(identity))
        done = self.vertex(parents)
        if kind == "set_flag":
            assert identity not in self.live, ("occupied publication", identity)
            if identity in self.consumed:
                assert self.ancestors[done] & (1 << self.consumed[identity]), ("unproved rearm", identity)
            self.live[identity] = done
        else:
            self.gate[pipe] = done
            if kind == "wait_flag":
                self.consumed[identity] = done
        self.finishes.setdefault(pipe, []).append(done)

    def payload(self, name, effects, context, acc_order=False):
        pipe = {"tload": "MTE2", "textract": "MTE1", "tmatmul": "M",
                "tmatmul.acc": "M", "tstore": "FIX"}[name]
        if self.serialize and name == "textract":
            self.sync("barrier", "ALL")
        if self.serialize_parent and name == "tload" and context[-1][1] > 0:
            self.sync("barrier", "ALL")
        issued = self.issue(pipe)
        done = self.vertex([issued])
        for old in self.payloads:
            for a, aw in old[2]:
                for b, bw in effects:
                    if (aw or bw) and a[0] == b[0] and a[1] < b[1] + b[2] and b[1] < a[1] + a[2]:
                        # Target-qualified ACC access ordering is not a graph
                        # completion edge. Never propagate it to operand reuse
                        # or FIX readiness. Both original operations must match.
                        if (acc_order and name == "tmatmul.acc" and self.native_acc.get(old[1]) == acc_order and
                                a[0] == "acc" and a == b):
                            self.native_acc_checks += 1
                            continue
                        assert self.ancestors[issued] & (1 << old[1]), ("missing physical conflict", old, name, effects, context)
                        self.required_checks += 1
        if self.check_outer and name == "tload":
            last_compute = next((p for p in reversed(self.payloads) if p[0].startswith("tmatmul")), None)
            for old in self.payloads:
                if (old is last_compute and old[3][:-2] == context[:-1] and
                        old[3][-2][0] == context[-1][0] and old[3][-2][1] + 1 == context[-1][1]):
                    assert not self.ancestors[issued] & (1 << old[1]), "child compute gates parent DMA"
                    self.outer_checks += 1
                if old[0] == "tload" and old[3] == context:
                    assert not self.ancestors[issued] & (1 << old[1]), "disjoint MAT pools serialized"
        if name == "textract":
            # Adjacent inner iterations use distinct banks. Restrict the
            # forbidden edge to one invocation of that inner loop; region-exit
            # cleanup is allowed after all its payload has issued.
            prior = next((p for p in reversed(self.payloads) if p[0].startswith("tmatmul")), None)
            if prior and prior[3][:-1] == context[:-1] and prior[3][-1][0] == context[-1][0] and prior[3][-1][1] + 1 == context[-1][1]:
                assert not self.ancestors[issued] & (1 << prior[1]), ("current-bank compute gates next-bank preparation", context)
                self.overlap_checks += 1
        if self.check_bank:
            if name == "tload":
                for old in self.payloads:
                    if (old[0].startswith("tmatmul") and old[3][:-2] == context[:-1] and
                            old[3][-2][0] == context[-1][0] and old[3][-2][1] + 1 == context[-1][1]):
                        assert not self.ancestors[issued] & (1 << old[1]), (
                            "previous-group compute gates a different MAT bank", old[3], context)
                        self.bank_checks += 1
            if (name == "textract" and context[-1][1] == 0 and
                    any(write and cell[0] == "left" for cell, write in effects)):
                loads = [old for old in self.payloads if old[0] == "tload" and old[3] == context[:-1]]
                assert len(loads) == 2, "fixture MAT producers changed"
                assert not self.ancestors[issued] & (1 << loads[1][1]), "B load gates early A readiness"
                self.early_ready_checks += 1
        self.payloads.append((name, done, effects, context))
        if acc_order:
            self.native_acc[done] = acc_order
        self.finishes.setdefault(pipe, []).append(done)


def execute(nodes, env, trace, context=()):
    for line, body, other in nodes:
        if not line or line.startswith("//"):
            continue
        for_match = re.search(r"scf.for (%\w+) = (%\w+) to (%\w+) step (%\w+)(.*) \{", line)
        if for_match:
            iv, lb, ub, step, rest = for_match.groups()
            bindings = re.findall(r"(%\w+) = (%\w+)", rest)
            carried = [env[init] for _, init in bindings]
            for i in range(env[lb], env[ub], env[step]):
                local = dict(env)
                local[iv] = i
                local.update((arg, val) for (arg, _), val in zip(bindings, carried))
                carried = execute(body, local, trace, context + ((iv, i),)) or []
            if " = scf.for" in line:
                assert len(carried) == 1
                env[line.split(" = ")[0]] = carried[0]
            continue
        if_match = re.search(r"scf.if (%\w+)", line)
        if if_match:
            result = execute(body if env[if_match[1]] else other, dict(env), trace, context)
            if " = scf.if" in line:
                assert result and len(result) == 1
                env[line.split(" = ")[0]] = result[0]
            continue
        if line.startswith("scf.yield"):
            return [env[v] for v in re.findall(r"%\w+", line.split(":")[0])]
        flag = re.search(r"pto.(set_flag|wait_flag)\[<PIPE_(\w+)>, <PIPE_(\w+)>, <EVENT_ID(\d+)>\]", line)
        if flag:
            trace.sync(flag[1], flag[2], flag[3], int(flag[4]))
            continue
        barrier = re.search(r"pto.barrier <PIPE_(\w+)>", line)
        if barrier:
            trace.sync("barrier", barrier[1])
            continue
        payload = re.match(r"pto.(tload|textract|tmatmul.acc|tmatmul|tstore)\b", line)
        if payload:
            ins, outs = line.split(" outs(")
            read = re.findall(r"%\w+", ins.split("ins(")[1].split(":")[0])
            written = re.findall(r"%\w+", outs.split(":")[0])
            effects = [(env[v], write) for values, write in ((read, False), (written, True))
                       for v in values if isinstance(env.get(v), tuple)]
            qualified = False
            if payload[1] in ("tmatmul", "tmatmul.acc") and "acc_phase" not in line:
                a, b = (env.get(("shape", v)) for v in read[-2:])
                c = env.get(("shape", written[0]))
                if a and b and c:
                    qualified = (a[2] == b[2] == 16 and c[2] == 32 and
                                 a[0] == c[0] and b[1] == c[1] and a[1] == b[0] and
                                 all(0 < n <= 4095 and n % 16 == 0 for n in (*a[:2], *b[:2])) and
                                 (a[0] // 16) * (b[1] // 16) >= 10 and
                                 (payload[1] == "tmatmul" or read[0] == written[0]))
            trace.payload(payload[1], effects, context, (*c, a[1]) if qualified else False)
            continue
        if " = " not in line:
            assert line == "return", line
            continue
        name, expression = line.split(" = ", 1)
        operands = re.findall(r"%\w+", expression.split(":")[0])
        const = re.match(r"arith.constant (-?\d+)", expression)
        if const:
            env[name] = int(const[1])
        elif expression.startswith("arith.index_cast"):
            env[name] = env[operands[0]]
        elif expression.startswith("arith.cmpi"):
            pred = expression.split()[1].rstrip(",")
            a, b = (env[v] for v in operands)
            env[name] = {"eq": a == b, "ne": a != b, "slt": a < b, "sle": a <= b,
                         "sgt": a > b, "sge": a >= b}[pred]
        elif expression.startswith("arith."):
            a, b = (env[v] for v in operands)
            op = expression.split()[0]
            if op == "arith.addi": env[name] = a + b
            elif op == "arith.subi": env[name] = a - b
            elif op == "arith.muli": env[name] = a * b
            elif op == "arith.divsi": env[name] = a // b
            elif op in ("arith.remsi", "arith.remui"): env[name] = a % b
            elif op == "arith.maxsi": env[name] = max(a, b)
            elif op == "arith.andi": env[name] = a & b
            elif op == "arith.ori": env[name] = a | b
            else: raise AssertionError(op)
        elif expression.startswith("pto.alloc_tile"):
            shape = re.search(r"!pto.tile_buf<(\w+), (\d+)x(\d+)xf(\d+)", expression)
            assert shape, expression
            addr = re.search(r"addr = (%\w+)", expression)[1]
            env[name] = (shape[1], env[addr], int(shape[2]) * int(shape[3]) * int(shape[4]) // 8)
            valid = [re.search(r"valid_" + d + r" = (%\w+)", expression) for d in ("row", "col")]
            dimensions = tuple(env[v[1]] if v else int(shape[i+2]) for i, v in enumerate(valid))
            if dimensions == (int(shape[2]), int(shape[3])):
                env[("shape", name)] = (*dimensions, int(shape[4]))
        else:
            assert expression.startswith(("pto.make_tensor_view", "pto.partition_view")), expression
    return None


def main():
    lines = open(sys.argv[1]).read().splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func @hpgemm_hpgemm" in line) + 1
    nodes, _ = parse(lines, start)
    for step in (256, 128, 64):
        trace = Trace(check_outer=True, check_bank=True)
        execute(nodes, {"%arg3": 0, "%arg4": step}, trace)
        assert not trace.live, "unconsumed events at return"
        for kind in ("set_flag", "wait_flag"):
            for source, observer in (("FIX", "M"), ("M", "FIX")):
                assert trace.sync_counts.get((kind, source, observer), 0) == 256 // step, (
                    "invariant enclosing completion or its acknowledgment repeats in a child",
                    trace.sync_counts)
        assert all(count == 0 for (kind, source, _), count in trace.sync_counts.items()
                   if kind == "barrier" and source != "ALL"), trace.sync_counts
        assert trace.sync_counts.get(("barrier", "ALL", None), 0) == 1, trace.sync_counts
        assert trace.overlap_checks == (256 // step) * 16 * 3 * 2, trace.overlap_checks
        assert trace.outer_checks == (256 // step) * 15 * 2, trace.outer_checks
        entries = 256 // step
        assert trace.bank_checks == entries * 15 * 4 * 2, trace.bank_checks
        assert trace.early_ready_checks == entries * 16, trace.early_ready_checks
        pairs = {("M", "MTE1"): 64 * entries + 2,
                 ("MTE1", "MTE2"): 16 * entries + 2,
                 ("MTE2", "MTE1"): 32 * entries,
                 ("MTE1", "M"): 64 * entries,
                 ("FIX", "M"): entries, ("M", "FIX"): entries}
        for kind in ("set_flag", "wait_flag"):
            actual = {(source, observer): count
                      for (command, source, observer), count in trace.sync_counts.items() if command == kind}
            assert actual == pairs, ("bank-qualified event population changed", actual)
        print("outer entries", 256 // step, "required edges", trace.required_checks,
              "native ACC access checks", trace.native_acc_checks,
              "forbidden inner/outer overlap edges absent", trace.overlap_checks, trace.outer_checks,
              "bank prefetch", trace.bank_checks, "early A readiness", trace.early_ready_checks,
              "event pairs", sum(pairs.values()), "named barriers", 0, "terminal ALL", 1)
    # A safety-preserving drain must FAIL the quality gate. This distinguishes
    # the overlap assertion from an acceptance-only synchronization test.
    try:
        execute(nodes, {"%arg3": 0, "%arg4": 256}, Trace(serialize=True))
    except AssertionError as error:
        assert "current-bank compute gates next-bank preparation" in str(error), error
    else:
        raise AssertionError("ordering oracle accepted an inserted whole-pipeline drain")
    try:
        execute(nodes, {"%arg3": 0, "%arg4": 256}, Trace(check_bank=True, serialize_parent=True))
    except AssertionError as error:
        assert "previous-group compute gates a different MAT bank" in str(error), error
    else:
        raise AssertionError("ordering oracle accepted a drain before next-bank prefetch")


if __name__ == "__main__":
    main()
