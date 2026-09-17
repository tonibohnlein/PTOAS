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
    def __init__(self, serialize=False):
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

    def payload(self, name, effects, context):
        pipe = {"tload": "MTE2", "textract": "MTE1", "tmatmul": "M",
                "tmatmul.acc": "M", "tstore": "FIX"}[name]
        if self.serialize and name == "textract":
            self.sync("barrier", "ALL")
        issued = self.issue(pipe)
        done = self.vertex([issued])
        for old in self.payloads:
            for a, aw in old[2]:
                for b, bw in effects:
                    if (aw or bw) and a[0] == b[0] and a[1] < b[1] + b[2] and b[1] < a[1] + a[2]:
                        assert self.ancestors[issued] & (1 << old[1]), ("missing physical conflict", old, name, effects, context)
                        self.required_checks += 1
        if name == "textract":
            # Adjacent inner iterations use distinct banks. Restrict the
            # forbidden edge to one invocation of that inner loop; region-exit
            # cleanup is allowed after all its payload has issued.
            prior = next((p for p in reversed(self.payloads) if p[0].startswith("tmatmul")), None)
            if prior and prior[3][:-1] == context[:-1] and prior[3][-1][0] == context[-1][0] and prior[3][-1][1] + 1 == context[-1][1]:
                assert not self.ancestors[issued] & (1 << prior[1]), ("current-bank compute gates next-bank preparation", context)
                self.overlap_checks += 1
        self.payloads.append((name, done, effects, context))
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
            trace.payload(payload[1], effects, context)
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
        else:
            assert expression.startswith(("pto.make_tensor_view", "pto.partition_view")), expression
    return None


def main():
    lines = open(sys.argv[1]).read().splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func @hpgemm_hpgemm" in line) + 1
    nodes, _ = parse(lines, start)
    for step in (256, 128):
        trace = Trace()
        execute(nodes, {"%arg3": 0, "%arg4": step}, trace)
        assert not trace.live, "unconsumed events at return"
        assert trace.overlap_checks == (256 // step) * 16 * 3 * 2, trace.overlap_checks
        print("outer entries", 256 // step, "required edges", trace.required_checks,
              "forbidden overlap edges absent", trace.overlap_checks)
    # A safety-preserving drain must FAIL the quality gate. This distinguishes
    # the overlap assertion from an acceptance-only synchronization test.
    try:
        execute(nodes, {"%arg3": 0, "%arg4": 256}, Trace(serialize=True))
    except AssertionError as error:
        assert "current-bank compute gates next-bank preparation" in str(error), error
    else:
        raise AssertionError("ordering oracle accepted an inserted whole-pipeline drain")


if __name__ == "__main__":
    main()
