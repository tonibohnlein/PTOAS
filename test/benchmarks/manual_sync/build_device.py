#!/usr/bin/env python3
"""Remote build: exact same explicit device flags and PTO headers for all arms."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--plans", type=Path, required=True)
p.add_argument("--pto-isa", type=Path, required=True)
p.add_argument("--bisheng", required=True)
p.add_argument("--flags", type=Path, required=True, help="JSON argv list from the target's validated compilation flags")
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
flags = json.loads(args.flags.read_text())
assert isinstance(flags,list) and all(isinstance(x,str) for x in flags)
assert "-O2" in flags and all(x == "-O2" for x in flags if x.startswith("-O")), flags
assert not any("auto-enable" in x for x in flags), "Manual comparison must disable downstream autosync"
assert "--cce-pto-enable" in flags
assert not any(x.startswith("-I") and "pto" in x.lower() for x in flags), "Supply PTO headers via --pto-isa only"
args.output.mkdir(parents=True,exist_ok=True)
here = Path(__file__).resolve().parent
manifest=[]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for case in ("smoke","reference"):
    cfg=json.loads((args.plans/case/"config.json").read_text())
    for arm in ("original_cpp","manual","manual_banked_keys","handoff","existing"):
        source=(args.pto_isa/"kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp" if arm=="original_cpp"
                else args.plans/case/(arm+".cpp")).resolve()
        compile_source=source
        if arm=="original_cpp":
            # Remove only the upstream host launcher so the common wrapper can
            # launch either fixed configuration. Do NOT define __COSTMODEL for
            # a device build: it changes the PTO headers' implementation path.
            text=source.read_text();marker="\n#ifndef __COSTMODEL\n"
            assert text.count(marker)==1
            compile_source=(args.output/"original_device_only.hpp").resolve()
            compile_source.write_text(text.split(marker)[0]+"\n")
        dest=(args.output/f"{case}-{arm}.so").resolve()
        definitions=[f"-DBENCH_{k.upper()}={v}" for k,v in cfg.items()]
        definitions += ['-DARM_SOURCE="'+str(compile_source)+'"',f"-DKERNEL_ENTRY=bench_{case}_{arm}"]
        if arm=="original_cpp": definitions += ["-DORIGINAL_ARM"]
        cmd=[args.bisheng,*flags,"-I"+str((args.pto_isa/"include").resolve()),*definitions,
             "-shared",str(here/"device_wrapper.cpp"),"-o",str(dest),"--cce-fatobj-link"]
        with (args.output/f"{case}-{arm}.build.log").open("w") as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        # Driver -### trace preserves the actual cc1 optimization options.
        audit=subprocess.run([*cmd,"-###"],capture_output=True,text=True,check=True)
        (args.output/f"{case}-{arm}.cc1.log").write_text(audit.stdout+audit.stderr)
        manifest.append(dict(case=case,arm=arm,source=str(source),source_sha256=sha(source),
                             compile_source=str(compile_source),compile_source_sha256=sha(compile_source),
                             library=str(dest),library_sha256=sha(dest),argv=cmd))
        print(case,arm,sha(dest),flush=True)
(args.output/"build.json").write_text(json.dumps(manifest,indent=2)+"\n")
