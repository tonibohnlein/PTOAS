#!/usr/bin/env python3
"""Build the production summary-index/projection tests, serially, offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', default='clang++')
    parser.add_argument('--sanitizers', action='store_true')
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    compiler = shutil.which(args.compiler)
    if not compiler:
        parser.error('compiler unavailable: '+args.compiler)
    args.output.mkdir(parents=True, exist_ok=False)
    source = root/'test/experiments/insert_sync/logical_plan/lifetime_revision_index_test.cpp'
    executable = args.output/'lifetime-index-test'
    command = [compiler, '-std=c++17', '-O1' if args.sanitizers else '-O2',
               '-Wall','-Wextra','-Werror','-Wpedantic','-fno-exceptions',
               '-I'+str(root/'include'), str(source), '-o',str(executable)]
    if args.sanitizers:
        command += ['-fsanitize=address,undefined','-fno-omit-frame-pointer']
    record = dict(native_PTOAS_tested=False, device_tested=False, commands=[])
    for invocation in [command, [str(executable.resolve())]]:
        start=time.monotonic()
        p=subprocess.run(invocation, capture_output=True, text=True, timeout=120)
        record['commands'].append(dict(argv=invocation,returncode=p.returncode,
                                      seconds=time.monotonic()-start,stdout=p.stdout,stderr=p.stderr))
        if p.returncode: break
    record['sources']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [source, root/'include/PTO/Transforms/InsertSync/StructuredSyncLifetimeSummary.h',
                                 root/'include/PTO/Transforms/InsertSync/StructuredSyncStorageEffects.h']}
    record['status']='passed' if len(record['commands'])==2 and all(c['returncode']==0 for c in record['commands']) else 'failed'
    (args.output/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record['commands'][-1]['stdout'])
    print(record['commands'][-1]['stderr'])
    print(record['status'])
    return 0 if record['status']=='passed' else 1

if __name__=='__main__':
    raise SystemExit(main())
