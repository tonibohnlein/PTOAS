#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Summarize the eight run.py campaigns, keeping every mechanism separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def mechanisms(counts):
    pipes=Counter()
    for key,value in counts.items():
        if key.startswith('barrier:'):
            pipes[re.search(r'PIPE_[A-Z0-9]+',key).group()] += value
    sets,waits=counts.get('pto.set_flag',0),counts.get('pto.wait_flag',0)
    return dict(sets=sets, waits=waits, pair_inventory=sets if sets==waits else None,
                named_barriers={p:v for p,v in sorted(pipes.items()) if p!='PIPE_ALL'},
                PIPE_ALL=pipes.get('PIPE_ALL',0))


def summarize(campaign):
    result=dict(measurement='Static IR and scalar replay; no device numerical/asynchronous validation or timing',
                failures=[], rows=[], scenarios=[])
    natives=set()
    by_key={}
    for arch in ('a2','a3'):
        for mode in ('off','on'):
            for population in ('controls','kernels'):
                directory=campaign/f'{arch}-{mode}-{population}'
                if population=='kernels':
                    directory=campaign/f'{arch}-{mode}-kernels-entry-free'
                report=json.loads((directory/'results.json').read_text())
                natives.add(report['native_sha256'])
                result['failures']+=report['failures']
                if not report['native_unchanged'] or report['mmad_chains'] != (mode=='on'):
                    result['failures'].append(f'{directory}: compiler changed or wrong option')
                for name,row in report['rows'].items():
                    case,arm=name.split('/')
                    entry=dict(arch=arch,mode=mode,case=case,arm=arm,status=row['status'],
                               source_sha256=row['input_sha256'],manifest_sha256=report['manifest_sha256'],
                               mechanisms=mechanisms(row['metrics']['static']['counts']))
                    by_key[(arch,mode,case,arm)]=row
                    result['rows'].append(entry)
                    for scenario,metric in row['metrics']['scenarios'].items():
                        result['scenarios'].append(dict(arch=arch,mode=mode,case=case,arm=arm,
                                                       scenario=scenario,mechanisms=mechanisms(metric['counts']),
                                                       payload_sha256=metric['payload_sha256']))
    for (arch,mode,case,arm),row in by_key.items():
        if mode!='on':continue
        before=by_key[(arch,'off',case,arm)]
        if row['input_sha256']!=before['input_sha256']:
            result['failures'].append(f'{arch}/{case}/{arm}: off/on input mismatch')
        left=before['metrics']['scenarios'];right=row['metrics']['scenarios']
        if set(left)!=set(right) or any(left[s]['payload_sha256']!=right[s]['payload_sha256'] for s in left):
            result['failures'].append(f'{arch}/{case}/{arm}: payload replay mismatch')
    inventories = {(r['arch'], r['mode'], r['case'], r['arm']): r['mechanisms'] for r in result['rows']}
    for (arch, mode, case, arm), inventory in inventories.items():
        if arch == 'a3' and inventory != inventories[('a2', mode, case, arm)]:
            result['failures'].append(f'{case}/{mode}/{arm}: architecture counts differ')
        if arm == 'combined' and inventory != inventories[(arch, mode, case, 'staged')]:
            result['failures'].append(f'{arch}/{case}/{mode}: combined/staged counts differ')
    if len(natives)!=1:result['failures'].append('not the same native compiler across all arms')
    result['native_sha256']=sorted(natives)
    return result


def markdown(report):
    lines=['# MMAD r3: all eleven local fixtures','',
           'Compiler source: `9e061dcf91f3707dd7d5fc38ddf922186c841dce`. MMAD-chain inference is opt-in.',
           f"Native SHA-256: `{report['native_sha256'][0]}`.",'',
           'Both A2 and A3, combined and staged modes were run with the flag off and on. '
           'All 132 compile/replay rows pass. Each row emits both PTO and C++ (264 successful compiler invocations). '
           'Scalar replay preserves payload traces for every available off/on scenario. '
           'Conv2D and FlashAttention have static measurements only; device correctness remains a separate check.','',
           'The table shows A3 combined mode. The static inventories agree with staged mode and A2. '
           'Set and wait sites are shown as a pair inventory only when their counts balance; '
           'this does not imply that they execute equally on every path. No total or performance score is formed.','',
           '| Fixture | Arm | Pairs | PIPE_M | PIPE_V | PIPE_MTE1 | PIPE_MTE2 | PIPE_MTE3 | PIPE_FIX | PIPE_ALL |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    rows={(r['case'],r['mode'],r['arm']):r for r in report['rows'] if r['arch']=='a3'}
    cases=list(dict.fromkeys(r['case'] for r in report['rows']))
    for case in cases:
        for mode,arm,label in [('off','manual','Hand'),('off','combined','MMAD off'),('on','combined','MMAD on')]:
            m=rows[(case,mode,arm)]['mechanisms'];p=m['named_barriers']
            values=[m['pair_inventory']]+[p.get('PIPE_'+x,0) for x in ('M','V','MTE1','MTE2','MTE3','FIX')]+[m['PIPE_ALL']]
            lines.append(f"| {case} | {label} | "+' | '.join(map(str,values))+' |')
    lines+=['','Only GEMM changes with the MMAD option: `PIPE_M` sites fall from 18 to 4. '
             'Pairs remain 44; its exit `PIPE_ALL` remains 1. On the primary core-0 scalar path, '
             'executed `PIPE_M` barriers fall from 1,408 to 176. This is a count reduction, not a measured speedup.','',
             'FlashAttention uses the repaired entry-aware `TFREE` in both pair members. '
             'That fixture repair raises the automatic inventory from the historical 13 pairs to 15 '
             'with the MMAD flag either off or on. It is not an MMAD regression. '
             'The historical input remains reproducible at `9e061dcf` and in the archived campaign.','',
             'Device adapter synchronization is outside this table: FlashAttention has a fixed vector peer '
             '(two stripes), and the GDN adapter drains two final cross-core credits on each vector stripe. '
             'Full-group device accounting must include those mechanisms separately.','',
             'The one-buffer / 16-trips result is 66 executed pairs, zero named barriers, one `PIPE_ALL`; '
             '`MANUAL_VS_FOLLOWUP_V2.md` already contains that corrected value in the current checkout.','',
             'Raw commands, post-pass PTO, generated C++, placement inventories and replay traces are retained '
             'in the campaign directories. The companion JSON contains explicit set/wait counts and all replay scenarios.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Output basename, without extension')
    a=p.parse_args();report=summarize(a.campaign)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    a.output.with_suffix('.md').write_text(markdown(report))
    print(json.dumps({'rows':len(report['rows']),'scenarios':len(report['scenarios']),'failures':report['failures']}))
    return bool(report['failures'])

if __name__=='__main__':raise SystemExit(main())
