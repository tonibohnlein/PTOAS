#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Audit every publication candidate; never equate a Cartesian pair with causality.

Consumes a pinned replay without invoking a compiler or changing alias contracts.
Reports every pair, including nonblocking pairs and mutually exclusive sites.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from coverage_campaign import witness_classes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_relation(nodes, writer, reader):
    parents = {}
    for node in nodes.values():
        for position, child in enumerate(node['children']):
            if child in parents:
                raise ValueError('witness control is not a tree')
            parents[child] = (node['id'], position)

    def path(node):
        result = []
        while node in parents:
            parent, position = parents[node]
            result.append((parent, position))
            node = parent
        return dict(result)

    left, right = path(writer), path(reader)
    common_loops = [n for n in left if n in right and nodes[n]['kind'] in (4, 5)]
    conflicts = [n for n in left if n in right and nodes[n]['kind'] == 3 and left[n] != right[n]]
    if conflicts:
        order = 'mutually-exclusive-within-one-iteration'
    elif writer == reader:
        order = 'same-operation-macro-phase-order-required'
    else:
        forks = [n for n in left if n in right and left[n] != right[n]]
        if not forks:
            order = 'unresolved-structural-order'
        else:
            lca = forks[0]
            order = ('writer-before-reader' if left[lca] < right[lca] else 'reader-before-writer')
    return dict(same_iteration_order=order, common_loops=common_loops,
                exclusive_choices=conflicts, possible_loop_backedge=bool(common_loops),
                possible_cross_invocation=True,
                causal_refusal_proven=False,
                missing_causal_evidence='failing consumer/cell and reaching write generation not recorded')


def operation_kind(text):
    found = re.search(r'\bpto\.[\w.]+', text)
    return found.group() if found else 'unknown'


def explain(pair, writer, reader):
    classification = pair['classification']
    if not pair['native_overlap']:
        return 'nonblocking', 'Already separated; this pair does not explain the refusal.'
    if classification == 'lost-provenance-or-range-precision':
        return 'available-fact-not-preserved', 'Import the recorded contract/range into the affected cell group.'
    if classification == 'genuine-possible-overlap':
        macro = any(kind in ('pto.comm.tput', 'pto.comm.tget') for kind in (writer, reader))
        return ('overlapping-macro-communication' if macro else 'overlapping-GM-communication',
                'Qualify the exact macro phases/private resources separately.' if macro else
                'Qualify the exact load/store cache recipe, generation and consumer context.')
    if set(pair['writer_access']['roots']).isdisjoint(pair['reader_access']['roots']):
        return 'insufficient-cross-root-alias-evidence', (
            'A matching caller allocation/ABI proof is needed. Distinct arguments, Out annotations, '
            'or legacy pointer inequality are not noalias guarantees.')
    return 'insufficient-range-or-causal-evidence', (
        'Preserve footprint independently of dynamic position, then establish which write reaches '
        'the reader. Geometry alone cannot eliminate an actual store/reload.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.campaign.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    cases, counts = [], Counter()
    for row in summary['rows']:
        if 'publication_reference' not in row:
            continue
        reports_path = args.campaign.parent / f'{row["index"]:04d}.witness.json'
        reports = json.loads(reports_path.read_text())
        legacy = Path(row['arms']['existing']['output'])
        source = Path(row['arms']['existing']['command'][-1])
        if digest(source) != row['prepared_sha256'] or digest(legacy) != row['arms']['existing']['output_sha256']:
            raise ValueError('input/output identity changed: ' + row['id'])
        pairs = []
        for report in reports:
            nodes = {node['id']: node for node in report['nodes']}
            for pair in witness_classes([report]):
                w, r = pair['writer_access'], pair['reader_access']
                writer, reader = nodes[w['node']], nodes[r['node']]
                wk, rk = operation_kind(writer['operation']), operation_kind(reader['operation'])
                category, missing = explain(pair, wk, rk)
                counts[category] += 1
                pairs.append(dict(**pair, category=category, missing_fact_or_realization=missing,
                                  writer_operation=writer['operation'], reader_operation=reader['operation'],
                                  writer_kind=wk, reader_kind=rk,
                                  control=control_relation(nodes, w['node'], r['node']),
                                  shared_cells=sorted(set(w['cells']) & set(r['cells'])),
                                  legacy_decision='MemAlias overlap query only; emitted dependency not attributed',
                                  range_note='legacy bytes=0 is unknown, not an empty access',
                                  fixed_protocol_nodes=[n for n in report['nodes'] if any(
                                      op in n['operation'] for op in ('pto.cmo.', 'pto.fence.', 'pto.comm.tnotify', 'pto.comm.twait'))]))
        text = source.read_text()
        case = dict(id=row['id'], index=row['index'], family=row['family'],
                    prepared_sha256=row['prepared_sha256'], input=str(source),
                    gm_alias=row['gm_alias'], hardware_contract=row['hardware_contract'],
                    ownership_contract=row['ownership_contract'], actual=row['actual'],
                    source_locations=sorted(set(re.findall(r'loc\("([^"]+)"', text))),
                    frontend_abi='Only frozen alias/ABI metadata is assumed; source locations are provenance hints.',
                    lowering='Default TLOAD/TSTORE and macro variants remain separately unqualified for GM publication.',
                    legacy_output=str(legacy), legacy_output_sha256=digest(legacy),
                    legacy_sync=[dict(line=i, operation=line.strip()) for i, line in
                                 enumerate(legacy.read_text().splitlines(), 1) if re.search(
                                     r'pto\.(set_flag|wait_flag|barrier|cmo\.|fence\.|comm\.tnotify|comm\.twait)', line)],
                    report_sha256=digest(reports_path),
                    report_exhausted=any(r['report_exhausted'] for r in reports),
                    categories=dict(Counter(p['category'] for p in pairs)), pairs=pairs)
        cases.append(case)
        (args.output / f'{row["index"]:04d}.json').write_text(json.dumps(case, indent=2) + '\n')
    if len(cases) != 86:
        raise ValueError('expected all 86 frozen publication refusals')
    result = dict(schema='oahs.publication.audit.v1', campaign=str(args.campaign),
                  campaign_sha256=digest(args.campaign), measurement='analysis-of-pinned-replay',
                  cases=len(cases), distinct_inputs=len({c['prepared_sha256'] for c in cases}),
                  pair_categories=dict(counts), causal_witness_limit='Candidate pairs, not a proof of the first refusal.',
                  rows=[{k: v for k, v in c.items() if k not in ('pairs', 'legacy_sync')} for c in cases])
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('cases', 'distinct_inputs', 'pair_categories')}, indent=2))


if __name__ == '__main__':
    main()
