#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run an adapter library: check EVERY launch, then emit synchronized host samples.

One invocation owns one device and one arm. A campaign controller must schedule
arms in balanced blocks on the same device. This is not a paired-ratio analysis.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
# Numerical reference work uses one BLAS worker; NPU launch geometry is unrelated.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from goldens import CASES, make_fixture


def checked(code):
    if code != 0:
        raise RuntimeError(f'Device runtime returned {code}')


class Device:
    def __init__(self, library, card):
        self.acl = ct.CDLL('libascendcl.so', mode=ct.RTLD_GLOBAL)
        self.runtime = ct.CDLL('libruntime.so', mode=ct.RTLD_GLOBAL)
        self.library = ct.CDLL(str(library.resolve()))
        signatures = {
            'aclInit': [ct.c_char_p], 'aclrtSetDevice': [ct.c_int32],
            'aclrtCreateStream': [ct.POINTER(ct.c_void_p)],
            'aclrtSynchronizeStream': [ct.c_void_p],
            'aclrtMalloc': [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_int],
            'aclrtFree': [ct.c_void_p],
            'aclrtMemcpy': [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int],
            'aclrtDestroyStream': [ct.c_void_p], 'aclrtResetDevice': [ct.c_int32],
            'aclFinalize': [],
        }
        for name, args in signatures.items():
            fn = getattr(self.acl, name)
            fn.argtypes, fn.restype = args, ct.c_int
        self.library.RunCase.argtypes = [ct.POINTER(ct.c_void_p), ct.c_uint64, ct.c_int64,
                                         ct.c_bool, ct.c_void_p, ct.c_int64, ct.POINTER(ct.c_double)]
        self.library.RunCase.restype = ct.c_int
        self.runtime.rtGetC2cCtrlAddr.argtypes = [ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint32)]
        self.runtime.rtGetC2cCtrlAddr.restype = ct.c_int
        self.card, self.allocations = card, []
        checked(self.acl.aclInit(None))
        checked(self.acl.aclrtSetDevice(card))
        self.stream = ct.c_void_p()
        checked(self.acl.aclrtCreateStream(ct.byref(self.stream)))
        self.ffts, length = ct.c_uint64(), ct.c_uint32()
        checked(self.runtime.rtGetC2cCtrlAddr(ct.byref(self.ffts), ct.byref(length)))

    def allocate(self, array):
        # All nine input/output/workspace allocations are disjoint and guarded.
        guard = 256
        ptr = ct.c_void_p()
        checked(self.acl.aclrtMalloc(ct.byref(ptr), array.nbytes + 2*guard, 0))
        self.allocations.append((ptr, array.nbytes))
        raw = np.full(array.nbytes + 2*guard, 0xA5, np.uint8)
        raw[guard:-guard] = array.view(np.uint8).ravel()
        checked(self.acl.aclrtMemcpy(ptr, raw.nbytes, ct.c_void_p(raw.ctypes.data), raw.nbytes, 1))
        return ptr.value + guard

    def upload(self, pointer, array):
        checked(self.acl.aclrtMemcpy(pointer, array.nbytes, ct.c_void_p(array.ctypes.data), array.nbytes, 1))

    def read(self, pointer, array):
        got = np.empty_like(array)
        checked(self.acl.aclrtMemcpy(ct.c_void_p(got.ctypes.data), got.nbytes, pointer, got.nbytes, 2))
        return got

    def check_guards(self):
        for ptr, size in self.allocations:
            for offset in (0, 256+size):
                guard = np.empty(256, np.uint8)
                checked(self.acl.aclrtMemcpy(ct.c_void_p(guard.ctypes.data), 256, ptr.value+offset, 256, 2))
                if not np.all(guard == 0xA5):
                    raise AssertionError(f'Buffer guard corrupted at {ptr.value+offset:#x}')

    def run(self, pointers, count, tail, launches):
        us = ct.c_double()
        checked(self.library.RunCase(pointers, self.ffts, count, tail, self.stream, launches, ct.byref(us)))
        return us.value

    def close(self):
        for ptr, _ in self.allocations:
            checked(self.acl.aclrtFree(ptr))
        checked(self.acl.aclrtDestroyStream(self.stream))
        checked(self.acl.aclrtResetDevice(self.card))
        checked(self.acl.aclFinalize())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--tail-half', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--correctness-launches', type=int, default=50)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1,
                        help='Use 1 for literal wall time per launch; >1 measures batched throughput')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.correctness_launches, args.batch) < 1 or args.samples < 0:
        parser.error('positive launch and batch counts, nonnegative samples required')
    if args.output.exists():
        parser.error('output already exists')
    fixture = make_fixture(args.case, args.count, args.tail_half, args.seed)
    report = dict(case=args.case, count=args.count, tail_half=args.tail_half, seed=args.seed,
                  device=args.device, arm=args.arm, block=args.block, batch=args.batch,
                  metric='synchronized_host_wall_us_per_launch',
                  mode='single_launch' if args.batch == 1 else 'batched_throughput',
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  atol=fixture.atol, rtol=fixture.rtol, correctness=[], timing_samples=[], status='NOT_RUN')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = None
    try:
        device = Device(args.library,args.device)
        pointers = [device.allocate(a) for a in fixture.buffers]
        report['buffers'] = [{'base':p, 'bytes':a.nbytes, 'dtype':str(a.dtype), 'shape':a.shape,
                              'initial_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
                             for p,a in zip(pointers,fixture.buffers)]
        pointers = (ct.c_void_p*9)(*pointers)
        for launch in range(args.correctness_launches):
            # Poison all outputs afresh; persistent workspace/queue state is
            # retained so repeated-launch credit leaks are exercised.
            for index in fixture.outputs:
                device.upload(pointers[index], fixture.buffers[index])
            device.run(pointers,args.count,args.tail_half,1)
            errors = {}
            for index,expected in fixture.outputs.items():
                got = device.read(pointers[index],expected)
                np.testing.assert_allclose(got,expected,rtol=fixture.rtol,atol=fixture.atol)
                untouched = expected == -123.0
                np.testing.assert_array_equal(got[untouched],expected[untouched])
                errors[index] = float(np.max(np.abs(got.astype(np.float64)-expected)))
            device.check_guards()
            report['correctness'].append(dict(launch=launch,max_abs_error=errors))
        for _ in range(10):
            device.run(pointers,args.count,args.tail_half,1)
        for sample in range(args.samples):
            report['timing_samples'].append(dict(sample=sample, host_us=device.run(
                pointers,args.count,args.tail_half,args.batch)))
        device.check_guards()
        report['status'] = 'PASS'
    except Exception as error:
        report['status'], report['error'] = 'FAIL', repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        if device is not None:
            device.close()
    print(json.dumps({'status':report['status'], 'output':str(args.output)}))

if __name__ == '__main__':
    main()
