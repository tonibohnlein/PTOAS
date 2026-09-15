#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Transparent capture shim for an existing frontend's PTOAS invocations.

Usage: capture_compiler.py --real /absolute/ptoas --records DIR -- <original args>
For a frontend expecting an executable named ptoas, install a small shell launcher
that passes these two explicit arguments. The real executable cannot be this shim.
No command/IR rewriting, extra synchronization flags, architecture conversion,
backend fallback, or kernel-name filtering. Captures the complete module, including
helpers. Compile stdout/stderr are forwarded byte-for-byte after the command exits.
Capture is serialized to avoid unbounded workers; lock waiting is NOT compiler time.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from evidence import EvidenceError, canonical, require, sha, stable_read, write_new, file_identity


def infer_paths(args, cwd):
    output = None
    inputs = []
    skip = False
    for i, arg in enumerate(args):
        if skip:
            skip = False
            continue
        if arg in ('-o', '--output'):
            require(i + 1 < len(args), 'Output flag has no value')
            output = args[i + 1]
            skip = True
        elif arg.startswith('--output='):
            output = arg.split('=', 1)[1]
        elif not arg.startswith('-') and Path(arg).suffix in ('.pto', '.mlir'):
            path = Path(arg)
            inputs.append(path if path.is_absolute() else cwd / path)
    # Unknown invocation forms still compile and produce an incomplete record;
    # do not silently remove them from the frontend population.
    source = inputs[0] if len(inputs) == 1 else None
    target = Path(output) if output and output != '-' else None
    if target and not target.is_absolute():
        target = cwd / target
    return source, target


def invoke(real, records, args, context_path=None):
    real, records = Path(real).resolve(), Path(records).absolute()
    require(real.is_file() and real != Path(__file__).resolve(), 'Invalid real compiler')
    require(args, 'Original compiler arguments are required')
    records.mkdir(parents=True, exist_ok=True)
    require(not records.is_symlink(), 'Record-directory symlink refused')
    # One active compiler through this shim. This does not control external
    # processes started outside the shim; do not claim a global worker limit.
    with (records / '.capture.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        folder = records / ('invocation-' + uuid.uuid4().hex)
        folder.mkdir()
        cwd = Path.cwd()
        source, output = infer_paths(args, cwd)
        report = dict(schema='oahs.capture-invocation.v1', cwd=str(cwd), argv=[str(real)] + args,
                      input_stage='frontend-emitted PTOAS input; not asserted pre-InsertSync IR',
                      tool_sha256=file_identity(real)['sha256'], input=None, output=None,
                      capture_complete=True, fallback_by_shim=False)
        if context_path:
            data = stable_read(context_path)
            write_new(folder / 'context.json', data)
            report['context_sha256'] = sha(data)
        else:
            report['capture_complete'] = False
            report['context_missing'] = True
        if source:
            try:
                data = stable_read(source)
                write_new(folder / 'input.pto', data)
                report['input'] = dict(original_path=str(source), path='input.pto', sha256=sha(data), bytes=len(data))
            except (OSError, EvidenceError) as error:
                report['input_error'] = str(error)
                report['capture_complete'] = False
        else:
            report['capture_complete'] = False
            report['input_error'] = 'Expected exactly one file input; stdin/ambiguous form must be captured separately'
        prior = output.stat() if output and output.exists() else None
        started = time.monotonic()
        try:
            with (folder / 'stdout').open('xb') as out, (folder / 'stderr').open('xb') as err:
                process = subprocess.run([str(real)] + args, stdout=out, stderr=err, check=False)
            report['returncode'] = process.returncode
        except OSError as error:
            report['returncode'] = 127
            report['launch_error'] = str(error)
        report['compiler_seconds'] = time.monotonic() - started
        if output and output.is_file():
            now = output.stat()
            changed = prior is None or (prior.st_ino, prior.st_size, prior.st_mtime_ns, prior.st_ctime_ns) != (now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns)
            if changed:
                data = stable_read(output)
                write_new(folder / 'output.bin', data)
                report['output'] = dict(original_path=str(output), path='output.bin', sha256=sha(data), bytes=len(data))
            else:
                report['capture_complete'] = False
                report['output_error'] = 'Output predates the invocation; not accepted as produced evidence'
        elif report['returncode'] == 0:
            report['capture_complete'] = False
            report['output_error'] = 'Successful command produced no identifiable output file'
        if file_identity(real)['sha256'] != report['tool_sha256']:
            report['capture_complete'] = False
            report['tool_changed'] = True
        for stream in ('stdout', 'stderr'):
            path = folder / stream
            if path.exists():
                data = stable_read(path)
                report[stream + '_sha256'] = sha(data)
                getattr(sys, stream).buffer.write(data)
                getattr(sys, stream).buffer.flush()
        write_new(folder / 'invocation.json', canonical(report))
        return report['returncode'] if report['returncode'] >= 0 else 128 - report['returncode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--context', type=Path)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    original = args.args[1:] if args.args and args.args[0] == '--' else args.args
    return invoke(args.real, args.records, original, args.context)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, EvidenceError) as error:
        print(f'Capture infrastructure error: {error}', file=sys.stderr)
        raise SystemExit(125)
