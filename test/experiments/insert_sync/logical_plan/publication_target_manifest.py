#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Freeze and compare the current GM-publication first-refusal population."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from coverage_campaign import witness_classes
from audit_publication_refusals import control_relation, operation_kind
from reaching_writers import WriterFlow


PUBLICATION = 'MTE3-to-MTE2 GM publication has no qualified compositional realization'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reports(campaign, row):
    path = campaign.parent / f'{row["index"]:04d}.witness.json'
    return path, json.loads(path.read_text())


def classified(campaign, row):
    path, raw = reports(campaign, row)
    return path, witness_classes(raw)


def matching_pairs(witnesses):
    return [pair for pair in witnesses if pair['failure_relation'] == 'matching-consumer-cell']


def enrich_pairs(raw, witnesses):
    reports = {report['function']: report for report in raw}
    flows = {name: WriterFlow(report) for name, report in reports.items()}
    enriched = []
    for original in witnesses:
        pair = dict(original)
        report = reports[pair['function']]
        flow = flows[pair['function']]
        nodes = {node['id']: node for node in report['nodes']}
        writer = pair['writer_access']['node']
        reader = pair['reader_access']['node']
        if writer in nodes and reader in nodes:
            writer_text = nodes[writer]['operation']
            reader_text = nodes[reader]['operation']
            pair.update(writer_operation=writer_text, reader_operation=reader_text,
                        writer_kind=operation_kind(writer_text), reader_kind=operation_kind(reader_text),
                        control=control_relation(nodes, writer, reader),
                        writer_flow_relation=flow.relation(pair['writer_access'], pair['reader_access']),
                        shared_cells=sorted(set(pair['writer_access']['cells']) &
                                            set(pair['reader_access']['cells'])))
            if pair['failure_relation'] == 'matching-consumer-cell':
                pair['writer_flow_witness'] = flow.witness(pair['writer_access'], pair['reader_access'])
        else:
            pair.update(writer_operation=None, reader_operation=None,
                        writer_kind='unmapped', reader_kind='unmapped', control=None,
                        shared_cells=sorted(set(pair['writer_access']['cells']) &
                                            set(pair['reader_access']['cells'])))
        enriched.append(pair)
    return enriched


def pair_site(pair):
    def access_site(access, operation, kind):
        return (access['node'], access['native_phase_id'], access['macro_phase'],
                access['pipeline'], access['write'], kind, operation)
    return (pair['function'],
            access_site(pair['writer_access'], pair['writer_operation'], pair['writer_kind']),
            access_site(pair['reader_access'], pair['reader_operation'], pair['reader_kind']))


def analysis_hashes():
    names = ('publication_target_manifest.py', 'coverage_campaign.py',
             'audit_publication_refusals.py', 'reaching_writers.py')
    return {name: digest(Path(__file__).with_name(name)) for name in names}


def legacy_sync(row):
    output = Path(row['arms']['existing']['output'])
    return [dict(line=line_number, operation=line.strip())
            for line_number, line in enumerate(output.read_text().splitlines(), 1)
            if re.search(r'pto\.(set_flag|wait_flag|barrier|cmo\.|fence\.|comm\.tnotify|comm\.twait)', line)]


def identity(row):
    prepared = Path(row['arms']['composition']['command'][-1])
    existing = Path(row['arms']['existing']['output'])
    if digest(prepared) != row['prepared_sha256'] or digest(existing) != row['arms']['existing']['output_sha256']:
        raise ValueError('prepared input or InsertSync output identity changed: ' + row['id'])
    return dict(input_sha256=row['input_sha256'], prepared_input=str(prepared),
                prepared_sha256=row['prepared_sha256'],
                alias_contract=row['gm_alias'], hardware_contract=row['hardware_contract'],
                ownership_contract=row['ownership_contract'],
                frontend_abi='Only the frozen caller/ABI and explicit alias metadata are assumed.',
                existing_output=str(existing), existing_output_sha256=digest(existing),
                existing_synchronization=row['arms']['existing']['synchronization'],
                existing_emitted_sync=legacy_sync(row))


def remaining_category(row, pairs):
    if any(not p['writer_access']['origins_complete'] or not p['reader_access']['origins_complete'] for p in pairs):
        return 'unsupported-provenance-transformation'
    if any(p['writer_access']['roots'] == p['reader_access']['roots'] and not p['exact_ranges'] for p in pairs):
        return 'unsupported-range-or-layout-precision'
    if row['gm_alias'] == 'may-alias' and any(
            p['writer_access']['roots'] != p['reader_access']['roots'] for p in pairs):
        return 'missing-external-alias-guarantee'
    return 'genuine-or-unresolved-overlapping-GM-communication'


def freeze(args):
    hashes = analysis_hashes()
    campaign = args.campaign.resolve()
    summary = json.loads(campaign.read_text())
    selected = [row for row in summary['rows'] if PUBLICATION in (row['actual'].get('first_refusal') or '')]
    if len(selected) != args.expected:
        raise ValueError(f'expected {args.expected} current GM-publication first refusals, found {len(selected)}')
    rows = []
    for row in selected:
        report_path, raw = reports(campaign, row)
        witnesses = enrich_pairs(raw, witness_classes(raw))
        matching = matching_pairs(witnesses)
        rows.append(dict(id=row['id'], index=row['index'], family=row['family'],
                         **identity(row), gm_alias=row['gm_alias'],
                         before=row['actual'], report=str(report_path), report_sha256=digest(report_path),
                         matching_consumer_cell_candidates=matching,
                         enumerated_candidate_witnesses=[p for p in witnesses
                                                         if p['failure_relation'] != 'matching-consumer-cell']))
    if analysis_hashes() != hashes:
        raise ValueError('manifest analysis implementation changed during freeze')
    result = dict(schema='oahs.gm-publication-targets.v5', measurement='frozen-current-first-refusals',
                  analysis_source_sha256=hashes,
                  source_campaign=str(campaign), source_campaign_sha256=digest(campaign),
                  revision=summary['revision'], source_diff_sha256=summary['source_diff_sha256'],
                  binary_sha256=summary['binary_sha256'], manifest_sha256=summary['manifest_sha256'],
                  rows=rows, counts=dict(rows=len(rows), distinct_inputs=len({r['prepared_sha256'] for r in rows}),
                                         contracts=dict(Counter(r['gm_alias'] for r in rows)),
                                         matching_candidates=sum(len(r['matching_consumer_cell_candidates']) for r in rows)))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['counts'], indent=2))


def compare(args):
    hashes = analysis_hashes()
    target = json.loads(args.targets.read_text())
    campaign = args.campaign.resolve()
    summary = json.loads(campaign.read_text())
    current = {row['id']: row for row in summary['rows']}
    if set(current) != {row['id'] for row in json.loads(Path(target['source_campaign']).read_text())['rows']}:
        raise ValueError('comparison campaign does not contain the same 363-row population')
    rows = []
    for before in target['rows']:
        after = current[before['id']]
        if (after['prepared_sha256'], after['gm_alias'], after['hardware_contract'], after['ownership_contract']) != (
                before['prepared_sha256'], before['gm_alias'], before['hardware_contract'], before['ownership_contract']):
            raise ValueError('target identity or contract changed: ' + before['id'])
        report_path, raw = reports(campaign, after)
        witnesses = enrich_pairs(raw, witness_classes(raw))
        matching = matching_pairs(witnesses)
        refusal = after['actual'].get('first_refusal') or ''
        still_publication = PUBLICATION in refusal
        before_groups = {}
        for pair in before['matching_consumer_cell_candidates']:
            before_groups.setdefault(pair_site(pair), []).append(pair)
        current_groups = {}
        for pair in witnesses:
            current_groups.setdefault(pair_site(pair), []).append(pair)
        proofs = []
        for site, prior in before_groups.items():
            now = current_groups.get(site, [])
            report_complete = all(not report.get('report_exhausted', False)
                                  for report in raw if report['function'] == site[0])
            if report_complete and now and all(not pair['native_overlap'] and
                           (pair['proven_disjoint_ranges'] or pair['proven_disjoint_contract']) for pair in now):
                proofs.append(dict(site=site, original_candidates=prior, current_candidates=now,
                                   proof=('caller-disjointness-contract' if
                                          all(pair['proven_disjoint_contract'] for pair in now)
                                          else 'same-root-disjoint-ranges')))
        rows.append(dict(id=before['id'], index=before['index'], family=before['family'],
                         prepared_sha256=before['prepared_sha256'], gm_alias=before['gm_alias'],
                         before=before['before'], after=after['actual'], still_publication=still_publication,
                         admitted=after['actual']['returncode'] == 0,
                         downstream_refusal=None if after['actual']['returncode'] == 0 or still_publication else refusal,
                         remaining_category=remaining_category(after, matching) if still_publication else None,
                         disjointness_proofs=proofs, report=str(report_path), report_sha256=digest(report_path),
                         matching_consumer_cell_candidates=matching,
                         enumerated_candidate_witnesses=[p for p in witnesses
                                                         if p['failure_relation'] != 'matching-consumer-cell']))
    if analysis_hashes() != hashes:
        raise ValueError('manifest analysis implementation changed during comparison')
    result = dict(schema='oahs.gm-publication-comparison.v5', measurement='fresh-target-comparison',
                  analysis_source_sha256=hashes,
                  targets=str(args.targets.resolve()), targets_sha256=digest(args.targets),
                  campaign=str(campaign), campaign_sha256=digest(campaign), revision=summary['revision'],
                  source_diff_sha256=summary['source_diff_sha256'], binary_sha256=summary['binary_sha256'],
                  rows=rows,
                  counts=dict(targets=len(rows), gm_refusals_removed=sum(not r['still_publication'] for r in rows),
                              newly_admitted=sum(r['admitted'] for r in rows),
                              downstream_refusals=sum(r['downstream_refusal'] is not None for r in rows),
                              admission_regressions=len(summary['lost']),
                              remaining=dict(Counter(r['remaining_category'] for r in rows
                                                     if r['remaining_category']))))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['counts'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--campaign', type=Path, required=True)
    freeze_parser.add_argument('--output', type=Path, required=True)
    freeze_parser.add_argument('--expected', type=int, default=89)
    freeze_parser.set_defaults(action=freeze)
    compare_parser = sub.add_parser('compare')
    compare_parser.add_argument('--targets', type=Path, required=True)
    compare_parser.add_argument('--campaign', type=Path, required=True)
    compare_parser.add_argument('--output', type=Path, required=True)
    compare_parser.set_defaults(action=compare)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
