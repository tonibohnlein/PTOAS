#!/usr/bin/env python3
"""Native acceptance for the e74 lifetime hardening patch. No network or device use.

This runs the actual PTOAS reconstruction driver. A missing driver or fixture
is a failed/incomplete run, never a passing skip. No compiler result is inferred
from the standalone index tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrap_section(text: str) -> str:
    begin = text.index('    %z = arith.constant')
    end = text.rindex('    return')
    body = text[begin:end]
    return text[:begin] + '    pto.section.cube {\n' + ''.join(
        '  ' + line if line.strip() else line for line in body.splitlines(keepends=True)
    ) + '    }\n' + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    driver = args.driver.resolve(strict=True)
    root = args.source_root.resolve(strict=True)
    fixture = root / 'test/experiments/insert_sync/logical_plan/structured_inputs/unitflag_paired.pto'
    original = fixture.read_text()
    section = wrap_section(original)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    cases = []
    for label, text in [('top', original), ('section', section)]:
        for credit in ['true', 'false']:
            cases.append((label + '-' + credit, text, credit, True))
    store_line = next(line for line in section.splitlines(keepends=True) if 'pto.tstore ins' in line)
    cases.extend([
        ('section-missing-store', section.replace(store_line, '', 1), 'true', False),
        ('section-partial-store', section.replace('#pto<st_phase final>', '#pto<st_phase partial>', 1), 'true', False),
        ('section-partial-producer', section.replace('#pto<acc_phase final>', '#pto<acc_phase partial>', 1), 'true', False),
        ('section-invalid-entry', section.replace('array<i64: 0, 1024>', 'array<i64: 512, 1024>', 1), 'false', False),
    ])
    mismatched = root / 'test/experiments/insert_sync/logical_plan/structured_inputs/unitflag_mismatched_geometry.pto'
    cases.append(('mismatched-geometry', mismatched.read_text(), 'true', False))
    for label, source_text, credit, expected in cases:
        source = args.output / (label + '.input.pto')
        output = args.output / (label + '.output.pto')
        source.write_text(source_text)
        mode = 'demands:none' if expected else 'demands:expect-unsupported'
        command = [str(driver), str(source.resolve()), mode, str(output.resolve()),
                   'conservative', 'assume-disjoint-arguments', 'a2a3-unitflag-paired-v1', credit]
        start = time.monotonic()
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=120)
            verdict = json.loads(p.stdout)
            accepted = bool(verdict['accepted'])
            passed = p.returncode == 0 and accepted == expected
            stderr, stdout, code = p.stderr, p.stdout, p.returncode
        except (subprocess.TimeoutExpired, ValueError, KeyError) as exc:
            verdict, passed, stderr, stdout, code = None, False, str(exc), '', None
        row = dict(case=label, command=command, source_sha256=digest(source),
                   expected_accepted=expected, passed=passed, verdict=verdict,
                   returncode=code, seconds=time.monotonic()-start)
        (args.output / (label + '.stdout')).write_text(stdout)
        (args.output / (label + '.stderr')).write_text(stderr)
        if output.exists():
            row['output_sha256'] = digest(output)
        rows.append(row)
    record = dict(driver_sha256=digest(driver), fixture_sha256=digest(fixture),
                  rows=rows, native_tested=True, device_tested=False,
                  status='passed' if all(r['passed'] for r in rows) else 'failed')
    (args.output/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    return 0 if record['status']=='passed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
