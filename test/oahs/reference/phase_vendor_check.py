#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check the immutable phase reference in a scratch copy, never rewrite the pin."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys, tempfile
VENDOR=Path(__file__).resolve().parent/'vendor/v010'
def verify_pin(root=VENDOR):
    pin=json.loads((root/'PIN.json').read_text())
    for name,want in pin['files'].items():
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=want:
            raise ValueError('phase reference integrity failure: '+name)
    return pin

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
    pin=verify_pin()
    with tempfile.TemporaryDirectory(prefix='oahs-phase-reference-') as d:
        root=Path(d)/'v010';shutil.copytree(VENDOR,root)
        phase=root/'checks/phases'
        sys.path.insert(0,str(phase))
        import phase_interface as pi
        checked=[pi.check_certificate(p) for p in sorted((phase/'certificates').glob('*.json'))]
        result={'certificates':len(checked),'states':sum(x['states'] for x in checked),'successors':sum(x['successor_checks'] for x in checked),'pinned_files':len(pin['files'])}
        if args.full:
            for name in ['run_checks.py','operational_checks.py','population_checks.py']:
                p=subprocess.run([sys.executable,str(phase/name)],capture_output=True,text=True)
                if p.returncode:raise RuntimeError(name+'\n'+p.stdout+'\n'+p.stderr)
            result['fresh_phase_campaign']=json.loads((phase/'phase_results.json').read_text())
            result['operational']=json.loads((phase/'operational_results.json').read_text())
            result['population']=json.loads((phase/'population_results.json').read_text())
        # Corrupted source cannot be silently accepted as the reference.
        bad=root/'checks/phases/phase_interface.py';bad.write_bytes(bad.read_bytes()+b'\n# corruption probe\n')
        try:verify_pin(root)
        except ValueError:result['corruption_rejected']=True
        else:raise AssertionError('corruption not detected')
        print(json.dumps(result,indent=2))
    verify_pin()
if __name__=='__main__':main()
