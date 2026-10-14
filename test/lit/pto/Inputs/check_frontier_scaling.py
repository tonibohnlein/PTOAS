# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check represented work; elapsed times are measurements, never class criteria."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


def require(value, message):
    if not value:
        raise RuntimeError(message)


class Measurements:
    """Own one private output directory and its serial child-process records."""

    def __init__(self, tool, prefix):
        self.tool = tool
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.output = Path(tempfile.mkdtemp(prefix=prefix.name + '-', dir=prefix.parent))
        self.records = []
        print('measurement artifacts:', self.output, flush=True)

    def invoke(self, name, flag, source=None):
        """Run one child and preserve its evidence before checking success."""
        argv = [self.tool, flag] + ([str(source)] if source else [])
        start = time.monotonic()
        try:
            result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=120)
            stdout, stderr = result.stdout, result.stderr
            code = result.returncode
            status = 'success' if code == 0 else 'process_failure'
        except subprocess.TimeoutExpired as error:
            def text(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
            stdout, stderr = text(error.stdout), text(error.stderr)
            code, status = None, 'measurement_timeout'
        (self.output / (name + '.out')).write_text(stdout)
        (self.output / (name + '.err')).write_text(stderr)
        record = dict(name=name, command=argv, seconds=time.monotonic() - start,
                      returncode=code, status=status)
        self.records.append(record)
        # Retain the terminal process record even when validation must stop.
        (self.output / 'partial-results.json').write_text(json.dumps(self.records, indent=2) + '\n')
        require(status == 'success', f'{name}: {status}: {stderr}')
        return stdout, record


def descriptor_checks(run):
    """Check shared stable descriptor lookup work over accesses and nesting."""
    for count, depth in [(8, 0), (32, 0), (128, 0), (512, 0), (8, 1), (8, 4), (8, 16)]:
        name = f'descriptor_{count}_{depth}'
        bound = 16 * (count + 2 * depth + 12)
        lines = ['!tile = !pto.tile_buf<vec, 8x8xf32, valid=?x?>', 'module {',
                 f'func.func @{name}(%n: index) attributes {{test.max_descriptor_visits = {bound} : i64}} {{',
                 '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index',
                 '%c8 = arith.constant 8 : index', '%addr = arith.constant 0 : i64',
                 '%t = pto.alloc_tile addr = %addr valid_row = %c8 valid_col = %c8 : !tile']
        lines += [f'scf.for %i{i} = %c0 to %n step %c1 {{' for i in range(depth)]
        lines += [f'%a{i} = "pto.tgetval"(%t, %c0) {{test.shape = array<i64: 8, 8>}} : (!tile, index) -> f32'
                  for i in range(count)]
        lines += ['}'] * depth + ['return', '}', '}']
        source = run.output / (name + '.pto')
        source.write_text('\n'.join(lines) + '\n')
        text, record = run.invoke(name, '--descriptor-state-checks', source)
        match = re.search(r'checked=(\d+) visits=(\d+) stable-lookups', text)
        require(match and int(match[1]) == count and int(match[2]) <= bound, text)
        record.update(accesses=count, depth=depth, repeated_queries=64 * count,
                      source_visits=int(match[2]), source_visit_bound=bound)



def quotient_checks(run):
    """Separate sparse supplied graph size from explicitly requested query rows."""
    # Supplied sparse certificates isolate quotient construction from source
    # extraction. The diagnostic helper also explicitly exports m-by-2m queries;
    # their size is reported rather than charged to the sparse graph alone.
    for count in (8, 16, 32):
        name = f'quotient_{count}'
        case = dict(pipes=[i % 2 for i in range(count)],
                    records=[[i, (i + 1) % count, int(i == count - 1)] for i in range(count)])
        source = run.output / (name + '.json')
        source.write_text(json.dumps([case]))
        text, record = run.invoke(name, '--periodic-checks', source)
        result = json.loads(text)[0]
        require(not result['error'], result['error'])
        require(len(result['thresholds']) == count and
                all(len(row) == 2 * count for row in result['thresholds']),
                'query directory lost source/target sites')
        # One payload edge and two native successor edges per site, plus
        # exactly one distinct supplied generator per site.
        require(result['graph_edges'] == 4 * count, 'sparse quotient graph was expanded')
        record.update(sites=count, supplied_generators=count, graph_edges=result['graph_edges'],
                      explicit_query_entries=sum(map(len, result['thresholds'])),
                      retained_demands=len(result['retained']))



def guarded_checks(run):
    """Check the sparse guarded construction against its stated work counters."""
    # Core chains close all native paths between k demand rounds.
    # These counters exclude separately requested all-pair query exports.
    for count in (6, 12, 24):
        name = f'guarded_{count}'
        case = dict(pipes=[i % 3 for i in range(count)], records=[
            dict(source=i, target=(i + 1) % count, bound=3) for i in range(count)],
            samples=[], export_thresholds=False)
        source = run.output / (name + '.json')
        source.write_text(json.dumps([case]))
        text, record = run.invoke(name, '--guarded-periodic-checks', source)
        result = json.loads(text)[0]
        require(not result['error'], result['error'])
        require(result['edges'] == count and result['frontier_entries'] == 12 * count,
                'sparse guarded input was expanded')
        require(result['relaxations'] == 18 * result['edges'],
                'guarded relaxation count differs from supplied graph bound')
        require(result['native_scan_visits'] == 102 * count and result['demand_rounds'] == 18,
                'core native closure work is not linear')
        require(result['threshold_queries'] == 0, 'unrequested query export constructed')
        record.update(sites=count, supplied_generators=count, **{key: result[key] for key in
                      ['edges', 'frontier_entries', 'relaxations', 'expressions', 'threshold_queries']})



def embedded_checks(run):
    """Run existing independent byte and graph oracles with aggregate timing."""
    fixtures = Path(__file__).resolve().parent.parent
    # Existing independent graph/byte oracles contain parameter sweeps and
    # counter assertions. Record these as checked families, not invented
    # per-parameter timing samples or counters unavailable from their exports.
    checks = [('memberships', '--numeric-analysis', 'sync_numeric_template.pto'),
              ('control_views', '--certify-regions', 'sync_normalized_control_session.pto'),
              ('encoded_slots', '--rotating-region-checks', 'sync_rotating_region.pto'),
              ('fixed_ports', '--numerical-repeated-squaring-checks', None)]
    for name, flag, fixture in checks:
        _, record = run.invoke(name, flag, fixtures / fixture if fixture else None)
        record['measurement_scope'] = 'existing embedded parameter/counter oracle; aggregate process time'

def main():
    require(len(sys.argv) == 3, "expected TOOL OUTPUT_PREFIX")
    tool = shutil.which(sys.argv[1])
    require(tool, "test tool not found")
    run = Measurements(tool, Path(sys.argv[2]))
    descriptor_checks(run)
    quotient_checks(run)
    guarded_checks(run)
    embedded_checks(run)
    manifest = dict(tool_sha256=hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
                    limits_are_class_criteria=False, records=run.records)
    (run.output / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('frontier scaling: descriptor, incidences, control, quotient, guarded, slots and fixed-port checks passed')


if __name__ == '__main__':
    main()
