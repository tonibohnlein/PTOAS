#!/usr/bin/env python3
"""Source/PTO trace correspondence plus bounded local event/conflict checks.

Reuses the independent test command graph, never the constructor's analysis.
GM shapes, strides, offsets and tile layouts enter payload identity. The causal
check covers local storage; numerical GM coverage is checked on device.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path


def run_ir(path, core, oracle, graph=True, tile_limit=None):
    text = path.read_text().splitlines()
    start = next(i for i, s in enumerate(text) if "func.func @reference_gemm" in s)
    args = re.findall(r"(%\w+):", text[start])
    nodes, _ = oracle.parse(text, start + 1)
    env = dict(zip(args, ("out", "a", "b", core)))
    class DiagnosticTrace(oracle.Trace):
        def __init__(self):
            super().__init__()
            self.rearming_failures = []
        def sync(self, kind, source, observer=None, key=None):
            try:
                super().sync(kind, source, observer, key)
            except AssertionError as error:
                # Preserve the rejected manual reference as evidence. Continue
                # its concrete matching graph WITHOUT inventing a rearm edge.
                # Such a graph is diagnostic, never a protocol certificate.
                assert error.args[0] == ("unproved rearm", (source,observer,key)), error
                assert kind == "set_flag"
                self.rearming_failures.append(dict(source=source,observer=observer,key=key,
                                                   payloads_before=len(self.payloads)))
                done = len(self.ancestors)-1
                self.live[(source,observer,key)] = done
                self.finishes.setdefault(source,[]).append(done)
    trace = DiagnosticTrace() if graph else None
    records, issues = [], []

    def visit(nodes, env, depth=0):
        for line, body, other in nodes:
            if not line or line.startswith("//"): continue
            loop = re.search(r"scf.for (%\w+) = (%\w+) to (%\w+) step (%\w+)(.*) \{", line)
            if loop:
                iv, lo, hi, step, rest = loop.groups()
                bindings = re.findall(r"(%\w+) = (%\w+)", rest)
                carried = [env[v] for _, v in bindings]
                upper = min(env[hi], tile_limit) if tile_limit and depth < 2 else env[hi]
                for value in range(env[lo], upper, env[step]):
                    child = dict(env)
                    child[iv] = value
                    child.update((a, v) for (a, _), v in zip(bindings, carried))
                    carried = visit(body, child, depth + 1) or []
                if " = scf.for" in line:
                    assert len(carried) == 1, line
                    env[line.split(" = ")[0]] = carried[0]
                continue
            branch = re.search(r"scf.if (%\w+)", line)
            if branch:
                ret = visit(body if env[branch[1]] else other, dict(env), depth)
                if " = scf.if" in line:
                    assert len(ret) == 1
                    env[line.split(" = ")[0]] = ret[0]
                continue
            if line.startswith("scf.yield"):
                return [env[v] for v in re.findall(r"%\w+", line.split(":")[0])]
            flag = re.search(r"pto.(set_flag|wait_flag)\[<PIPE_(\w+)>, <PIPE_(\w+)>, <EVENT_ID(\d+)>\]", line)
            if flag:
                word = [flag[1], flag[2], flag[3], int(flag[4])]
                records.append({"sync": word})
                if graph: trace.sync(*word)
                continue
            bar = re.search(r"pto.barrier <PIPE_(\w+)>", line)
            if bar:
                records.append({"sync": ["barrier", bar[1]]})
                if graph: trace.sync("barrier", bar[1])
                continue
            op = re.match(r"pto.(tload|textract|tmatmul.acc|tmatmul|tstore)\b", line)
            if op:
                ins, outs = line.split(" outs(")
                read = re.findall(r"%\w+", ins.split("ins(")[1].split(":")[0])
                write = re.findall(r"%\w+", outs.split(":")[0])
                tiles = [env[v] for v in read + write if isinstance(env[v], list) and len(env[v]) == 7]
                record = {"op": op[1], "tiles": tiles}
                if op[1] in ("tload", "tstore"):
                    record["gm"] = env[(read if op[1] == "tload" else write)[0]]
                if op[1] == "textract": record["slice"] = [env[v] for v in read[1:]]
                records.append(record)
                if graph:
                    effects = [(tuple((env[v][0], env[v][1], env[v][2]*env[v][3]*env[v][4]//8)), w)
                               for vs, w in ((read, False), (write, True)) for v in vs
                               if isinstance(env[v], list) and len(env[v]) == 7]
                    pipe = {"tload":"MTE2", "textract":"MTE1", "tmatmul":"M", "tmatmul.acc":"M", "tstore":"FIX"}[op[1]]
                    # Disable Shenggan-specific forbidden-order assertions. We
                    # measure ordering below instead of requiring a better plan.
                    context = ((str(len(records)), 0),)
                    issue = len(trace.ancestors)
                    qualified = (128, 256, 32, 64) if op[1].startswith("tmatmul") else False
                    trace.payload(op[1], effects, context, qualified)
                    issues.append(issue)
                    assert trace.launch[pipe] == issue
                continue
            if line == "return": continue
            assert " = " in line, line
            name, expr = line.split(" = ", 1)
            operands = re.findall(r"%\w+", expr.split(":")[0])
            const = re.match(r"arith.constant (-?\d+)", expr)
            if const: env[name] = int(const[1]); continue
            if expr.startswith("arith.index_cast"):
                env[name] = env[operands[0]]; continue
            if expr.startswith("arith.cmpi"):
                a, b = (env[v] for v in operands)
                pred = expr.split()[1].rstrip(",")
                env[name] = {"eq": a==b, "ne": a!=b, "slt": a<b, "sle": a<=b,
                             "sgt": a>b, "sge": a>=b, "ult":a<b, "ule":a<=b}[pred]
                continue
            if expr.startswith("arith."):
                a, b = (env[v] for v in operands)
                op = expr.split()[0].split(".")[1]
                if op in ("divsi", "divui"): val = a // b
                elif op in ("remsi", "remui"): val = a % b
                elif op == "addi": val = a + b
                elif op == "subi": val = a - b
                elif op == "muli": val = a * b
                elif op == "andi": val = a & b
                elif op == "ori": val = a | b
                else: raise AssertionError(expr)
                env[name] = val; continue
            if expr.startswith("pto.alloc_tile"):
                s = re.search(r"!pto.tile_buf<(\w+), (\d+)x(\d+)xf(\d+)", expr)
                assert s, expr
                addr = re.search(r"addr = (%\w+)", expr)[1]
                bl = re.search(r"blayout=(\w+)", expr)
                sl = re.search(r"slayout=(\w+)", expr)
                bl = bl[1] if bl else "row_major"
                sl = sl[1] if sl else "row_major"
                env[name] = [s[1], env[addr], int(s[2]), int(s[3]), int(s[4]),
                             int(bl == "row_major"), int(sl == "col_major")]
                continue
            if expr.startswith("pto.make_tensor_view"):
                ptr, rows, cols, rs, cs = operands
                dtype = re.search(r"xf(\d+)", expr)[1]
                env[name] = [env[ptr], 0, env[rows], env[cols], env[rs]*int(dtype)//8, env[cs]*int(dtype)//8]
                continue
            if expr.startswith("pto.partition_view"):
                view, row, col, rows, cols = operands
                g, offset, _, _, rs, cs = env[view]
                env[name] = [g, offset+env[row]*rs+env[col]*cs, env[rows], env[cols], rs, cs]
                continue
            raise AssertionError(expr)
    visit(nodes, env)
    if graph: assert not trace.live, (path, trace.live)
    return records, trace, issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plans", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("oracle", args.oracle)
    oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
    here = Path(__file__).resolve().parent
    results = {}
    for case in ("smoke", "reference"):
        folder = args.plans / case
        cfg = json.loads((folder / "config.json").read_text())
        binary = folder / "source-probe"
        cmd = ["g++", "-std=c++17", "-O1", "-I"+str(here/"probe"),
               '-DORIGINAL_SOURCE="'+str(args.source.resolve())+'"',
               *[f"-DBENCH_{k.upper()}={v}" for k,v in cfg.items() if k != "cores"],
               str(here/"probe/main.cpp"), "-o", str(binary)]
        subprocess.run(cmd, check=True)
        identity_checks = []
        # Full payload/event identity for every launch core, including boundaries.
        for core in range(cfg["cores"]):
            original = [json.loads(x) for x in subprocess.check_output([str(binary),str(core)], text=True).splitlines()]
            manual, _, _ = run_ir(folder/"manual.pto", core, oracle, False)
            assert original == manual, (case, core, next((i,a,b) for i,(a,b) in enumerate(zip(original,manual)) if a != b))
            expected = [x for x in original if "op" in x]
            for arm in ("input", "manual_banked_keys", "handoff", "existing"):
                generated, _, _ = run_ir(folder/(arm+".pto"), core, oracle, False)
                assert [x for x in generated if "op" in x] == expected, (case, core, arm, "payload changed")
            identity_checks.append(dict(core=core,payloads=len(expected),sha256=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()))
        cases = []
        for limit in (1,2):
            graphs = {}
            for arm in ("manual", "manual_banked_keys", "handoff", "existing"):
                records, trace, issues = run_ir(folder/(arm+".pto"), 0, oracle, True, limit)
                relations = {(i,j) for j,issue in enumerate(issues) for i,(_,done,_,_) in enumerate(trace.payloads)
                             if trace.ancestors[issue] & (1<<done)}
                graphs[arm] = relations
                counts = trace.sync_counts
                cases.append(dict(tile_limit_per_dimension=limit,arm=arm,payloads=len(trace.payloads),
                                  pairs=sum(v for (kind,_,_),v in counts.items() if kind=="set_flag"),
                                  named=sum(v for (kind,pipe,_),v in counts.items() if kind=="barrier" and pipe!="ALL"),
                                  terminal_all=counts.get(("barrier","ALL",None),0),
                                  local_conflicts=trace.required_checks, native_acc=trace.native_acc_checks,
                                  rearming_failures=trace.rearming_failures,
                                  protocol_check_passed=not trace.rearming_failures,
                                  finish_issue_relations=len(relations)))
            for r in cases[-4:]:
                rel=graphs[r["arm"]]; manual=graphs["manual"]
                r.update(diagnostic_added_vs_manual=len(rel-manual),diagnostic_removed_vs_manual=len(manual-rel))
                if r["arm"] != "manual": assert r["protocol_check_passed"], r
            assert graphs["manual_banked_keys"] == graphs["manual"]
        results[case] = dict(source_identity=identity_checks,bounded_graphs=cases)
        print(case, "source identity checked on", len(identity_checks), "cores; bounded protocols checked", flush=True)
    (args.plans / "checks.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__": main()
