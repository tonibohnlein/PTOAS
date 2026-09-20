#!/usr/bin/env python3
"""Remote ACL correctness/timing harness; one cached FP64 oracle per seed.

The supplied C++ source uses column-major B. Never transpose the uploaded bytes
to row-major. Device timings are gated by every arm's completed numerical checks.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

p=argparse.ArgumentParser()
p.add_argument("--plans",type=Path,required=True)
p.add_argument("--binaries",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--case",choices=("smoke","reference"),required=True)
p.add_argument("--acl",default="libascendcl.so")
p.add_argument("--device",type=int,default=0)
p.add_argument("--host-threads",type=int,required=True)
p.add_argument("--seeds",type=int,nargs="+",default=[1,7,19])
p.add_argument("--rounds",type=int,default=6)
p.add_argument("--samples",type=int,default=30)
args=p.parse_args()
assert args.host_threads > 0
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]=str(args.host_threads)
import numpy as np

args.output.mkdir(parents=True,exist_ok=True)
cfg=json.loads((args.plans/args.case/"config.json").read_text())
m,n,k=(cfg[x] for x in ("m","n","k"))
arms=("original_cpp","manual","manual_banked_keys","handoff","existing")
acl=C.CDLL(args.acl)
ptr=C.c_void_p
size=C.c_size_t
signatures={"aclInit":[C.c_char_p],"aclrtSetDevice":[C.c_int],"aclrtCreateStream":[C.POINTER(ptr)],
 "aclrtMalloc":[C.POINTER(ptr),size,C.c_int],"aclrtFree":[ptr],
 "aclrtMemcpy":[ptr,size,ptr,size,C.c_int],"aclrtSynchronizeStream":[ptr],
 "aclrtCreateEvent":[C.POINTER(ptr)],"aclrtRecordEvent":[ptr,ptr],
 "aclrtEventElapsedTime":[C.POINTER(C.c_float),ptr,ptr],"aclrtDestroyEvent":[ptr],
 "aclrtDestroyStream":[ptr],"aclrtResetDevice":[C.c_int],"aclFinalize":[]}
for name,types in signatures.items():
    fn=getattr(acl,name);fn.argtypes=types;fn.restype=C.c_int
def call(name,*a):
    rc=getattr(acl,name)(*a)
    if rc: raise RuntimeError((name,rc))
def copy_to(dst,src): call("aclrtMemcpy",dst,src.nbytes,ptr(src.ctypes.data),src.nbytes,1)
def copy_from(dst,src): call("aclrtMemcpy",ptr(dst.ctypes.data),dst.nbytes,src,dst.nbytes,2)
allocations=[]
def alloc(count):
    out=ptr();call("aclrtMalloc",C.byref(out),count,0);allocations.append(out);return out
def offset(x,n): return ptr(x.value+n)
libs={};launch={};hashes={}
build=json.loads((args.binaries/"build.json").read_text())
for arm in arms:
    path=(args.binaries/f"{args.case}-{arm}.so").resolve()
    hashes[arm]=hashlib.sha256(path.read_bytes()).hexdigest()
    entry=next(x for x in build if x["case"]==args.case and x["arm"]==arm)
    assert hashes[arm]==entry["library_sha256"], (arm,"loaded binary differs from build manifest")
    libs[arm]=C.CDLL(str(path),mode=C.RTLD_LOCAL)
    fn=libs[arm].LaunchReference;fn.argtypes=[ptr,ptr,ptr,ptr];fn.restype=None;launch[arm]=fn
call("aclInit",None);call("aclrtSetDevice",args.device)
stream=ptr();call("aclrtCreateStream",C.byref(stream))
guard=256;sentinel=np.float32(-7777);guard_value=np.float32(-1234.5)
da=alloc((m*k+2*guard)*2);db=alloc((n*k+2*guard)*2)
outputs=[alloc((m*n+2*guard)*4) for _ in range(4)]
initial=np.full(m*n+2*guard,guard_value,dtype=np.float32);initial[guard:-guard]=sentinel
got=np.empty_like(initial)
results={"config":cfg,"hashes_loaded":hashes,"correctness":[],"timings":[],
         "bound":"gamma_K * sum(abs(a*b)), u=2^-24, exact FP16 products; no overflow/underflow for generated dyadics",
         "manual_rearming":"Upstream manual fails the stricter local causal-rearming certificate; timing is empirical, not proof of that contract."}
def save():
    tmp=args.output/"results.json.tmp";tmp.write_text(json.dumps(results,indent=2)+"\n");tmp.replace(args.output/"results.json")

try:
    for seed in args.seeds:
        started=time.monotonic();rng=np.random.default_rng(seed)
        # Multiples of 1/2048 within FP16's exact integer-significand range.
        a=(rng.integers(-2047,2048,size=(m,k),dtype=np.int16).astype(np.float32)/2048).astype(np.float16)
        b=(rng.integers(-2047,2048,size=(n,k),dtype=np.int16).astype(np.float32)/2048).astype(np.float16)
        ha=np.pad(a.ravel(),(guard,guard));hb=np.pad(b.ravel(),(guard,guard))
        copy_to(da,ha);copy_to(db,hb)
        # Disk-backed references are shared across arms and four queued slots.
        gold=np.lib.format.open_memmap(args.output/f"gold-{seed}.npy",mode="w+",dtype=np.float64,shape=(m,n))
        magnitude=np.lib.format.open_memmap(args.output/f"abs-{seed}.npy",mode="w+",dtype=np.float64,shape=(m,n))
        bf=b.astype(np.float64);ab=np.abs(bf)
        for row in range(0,m,128):
            aa=a[row:row+128].astype(np.float64)
            gold[row:row+128]=aa@bf.T
            magnitude[row:row+128]=np.abs(aa)@ab.T
        gold.flush();magnitude.flush();del bf,ab
        oracle_seconds=time.monotonic()-started
        gamma=(k*2.0**-24)/(1-k*2.0**-24)
        baseline=None
        for arm in arms:
            # Every seed tests four launches queued without intervening sync.
            for out in outputs:copy_to(out,initial)
            for out in outputs:launch[arm](offset(out,guard*4),offset(da,guard*2),offset(db,guard*2),stream)
            call("aclrtSynchronizeStream",stream)
            for slot,out in enumerate(outputs):
                copy_from(got,out)
                value=got[guard:-guard].reshape(m,n)
                bad=untouched=nonfinite=0;worst=0.0
                for row in range(0,m,128):
                    v=value[row:row+128];err=np.abs(v.astype(np.float64)-gold[row:row+128]);bound=gamma*magnitude[row:row+128]
                    bad+=int(np.count_nonzero(err>bound));untouched+=int(np.count_nonzero(v==sentinel));nonfinite+=int(np.count_nonzero(~np.isfinite(v)))
                    worst=max(worst,float(np.max(np.divide(err,bound,out=np.zeros_like(err),where=bound!=0))))
                guards=int(np.count_nonzero(got[:guard]!=guard_value)+np.count_nonzero(got[-guard:]!=guard_value))
                identity=hashlib.sha256(value.tobytes()).hexdigest()
                if baseline is None:baseline=value.copy()
                entry=dict(seed=seed,arm=arm,slot=slot,bad=bad,untouched=untouched,nonfinite=nonfinite,guards=guards,
                           worst_err_over_bound=worst,output_sha256=identity,
                           bitwise_equal_original=bool(np.array_equal(value,baseline)),
                           max_abs_difference_original=float(np.max(np.abs(value-baseline))),oracle_seconds=oracle_seconds)
                entry["pass"]=not(bad or untouched or nonfinite or guards)
                results["correctness"].append(entry);save()
                print(json.dumps(entry),flush=True)
                if not entry["pass"]:raise RuntimeError("Correctness failed; no timing permitted")
            # Inputs, including both guard regions, must remain byte-identical.
            for expected,device in ((ha,da),(hb,db)):
                actual=np.empty_like(expected);copy_from(actual,device)
                if not np.array_equal(actual.view(np.uint16),expected.view(np.uint16)):
                    raise RuntimeError((arm,"input or input guard modified"))
        del gold,magnitude,baseline
    # All arms/seeds have passed before any timed experiment starts.
    events=[ptr(),ptr()]
    for e in events:call("aclrtCreateEvent",C.byref(e))
    for arm in arms:
        for _ in range(20):launch[arm](offset(outputs[0],guard*4),offset(da,guard*2),offset(db,guard*2),stream)
        call("aclrtSynchronizeStream",stream)
    for rnd in range(args.rounds):
        # Rotate each sample too: equal first/last exposure across the campaign.
        for sample in range(args.samples):
            shift=(rnd+sample)%len(arms)
            for arm in arms[shift:]+arms[:shift]:
                call("aclrtRecordEvent",events[0],stream)
                launch[arm](offset(outputs[0],guard*4),offset(da,guard*2),offset(db,guard*2),stream)
                call("aclrtRecordEvent",events[1],stream);call("aclrtSynchronizeStream",stream)
                ms=C.c_float();call("aclrtEventElapsedTime",C.byref(ms),events[0],events[1])
                results["timings"].append(dict(round=rnd,sample=sample,arm=arm,us=ms.value*1000))
        save()
    results["summary"]={arm:dict(zip(("q25_us","median_us","q75_us"),map(float,np.percentile(
        [r["us"] for r in results["timings"] if r["arm"]==arm],[25,50,75])))) for arm in arms}
    save();print(json.dumps(results["summary"],indent=2))
    for e in events:call("aclrtDestroyEvent",e)
finally:
    for x in allocations:call("aclrtFree",x)
    call("aclrtDestroyStream",stream);call("aclrtResetDevice",args.device);call("aclFinalize")
