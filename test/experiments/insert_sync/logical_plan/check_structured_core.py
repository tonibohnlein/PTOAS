# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile the actual solver-free core. No mock adapter or extracted algorithm."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sanitize',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent; root=here.parents[3]
    core=root/'lib/PTO/Transforms/InsertSync/StructuredSyncCore.cpp'
    header=root/'include/PTO/Transforms/InsertSync/StructuredSyncCore.h'
    source=here/'structured_core_test.cpp'; binary=args.output/'structured-core-test'
    command=[args.compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-fno-exceptions',
             '-I'+str(root/'include'),str(source),str(core),'-o',str(binary)]
    if args.sanitize: command[2:2]=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
    subprocess.run(command,check=True)
    env=dict(os.environ)
    if args.sanitize: env['ASAN_OPTIONS']='detect_leaks=0'; env['UBSAN_OPTIONS']='halt_on_error=1'
    result=subprocess.run([str(binary)],text=True,capture_output=True,env=env,check=True)
    summary=json.loads(result.stdout)
    summary.update(command=command,compiler_version=subprocess.check_output([args.compiler,'--version'],text=True),
                   source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (core,header,source)},
                   native_adapter='NOT_RUN',device='NOT_RUN')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'stderr.txt').write_text(result.stderr)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
