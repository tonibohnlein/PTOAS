#!/usr/bin/env python3
"""Reproduce all host arms using one compiler revision, without device access."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument("--opt",required=True)
p.add_argument("--driver",required=True)
p.add_argument("--ptoas",nargs="+",required=True)
p.add_argument("--source",type=Path,required=True)
p.add_argument("--oracle",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
here=Path(__file__).resolve().parent
args.output.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(here/"generate.py"),str(args.output)],check=True)
records=[]
def run(cmd,log):
    start=time.monotonic()
    with log.open("w") as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
    records.append(dict(argv=cmd,seconds=time.monotonic()-start,log=str(log)))
for case in ("smoke","reference"):
    d=args.output/case
    for arm in ("handoff","existing"):
        run([args.opt,"--pto-insert-sync=algorithm="+arm,str(d/"input.pto"),"-o",str(d/(arm+".pto"))],d/(arm+".log"))
    # This driver constructs and reconstructs in the native integration path.
    with (d/"constructed.pto").open("w") as output, (d/"reconstruct.log").open("w") as log:
        subprocess.run([args.driver,"--construct",str(d/"input.pto")],stdout=output,stderr=log,check=True,timeout=1200)
    for arm in ("manual","manual_banked_keys","handoff","existing"):
        run([*args.ptoas,"--pto-level=level3","--pto-arch=a3",str(d/(arm+".pto")),"-o",str(d/(arm+".cpp"))],d/(arm+".lower.log"))
run([sys.executable,str(here/"check.py"),str(args.output),"--source",str(args.source),"--oracle",str(args.oracle)],args.output/"checks.log")
(args.output/"host-runs.json").write_text(json.dumps(records,indent=2)+"\n")
