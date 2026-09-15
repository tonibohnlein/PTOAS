#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""M1 evidence capture, existing-compiler replay, and strict comparison.

No synchronization construction, semantic opcode admission, or IR rewriting.
Python 3.10+ standard library. External commands run serially without a shell.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import statistics
import subprocess
import sys
import time

SCHEMA = 'oahs.evidence.v1'
COHORTS = ('ptoas', 'pypto_ops', 'pypto_lib', 'model_kernels', 'device_regressions')
CONTRACTS = ('architecture', 'hardware', 'alias', 'abi', 'memory_planner')
HANDLING = ('generated', 'authored_preserved', 'noop_verified', 'fallback', 'refused', 'unclassified')
SHA = re.compile(r'^[a-f0-9]{64}$')
PRODUCTION_REVISION = 'ad63e67a79d47e35a76692dd534df13bd4bd6b23'
REV = re.compile(r'^[a-f0-9]{40}$')
IDENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$')


class EvidenceError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise EvidenceError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable_read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Not a regular non-symlink file: {path}')
    with path.open('rb') as stream:
        a = os.fstat(stream.fileno())
        require(a.st_size <= 512 * 1024 * 1024, f'Artifact exceeds 512 MiB capture bound: {path}')
        data = stream.read()
        b = os.fstat(stream.fileno())
    now = path.stat()
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(key(a) == key(b) == key(now) and len(data) == a.st_size, f'File changed during read: {path}')
    return data


def file_identity(path):
    """Hash potentially large compiler/device binaries without reading them into RAM."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Not a regular artifact: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        size = 0
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = path.stat()
    key = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    require(key(before) == key(after) == key(current) and size == before.st_size,
            f'Artifact changed during hashing: {path}')
    return dict(sha256=digest.hexdigest(), bytes=size)


def load(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, f'Duplicate JSON key: {key}')
            out[key] = value
        return out
    def bad(value):
        raise EvidenceError(f'Nonfinite JSON number: {value}')
    return json.loads(stable_read(path), object_pairs_hook=unique, parse_constant=bad)


def safe(root, relative):
    rel = PurePosixPath(relative)
    require(relative and not rel.is_absolute() and '..' not in rel.parts, f'Unsafe relative path: {relative}')
    target = Path(root)
    for part in rel.parts:
        target = target / part
        require(not target.is_symlink(), f'Symlink refused: {target}')
    require(target.resolve().is_relative_to(Path(root).resolve()), f'Escaping path: {relative}')
    return target


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def fresh_directory(path):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), f'Output must not exist: {path}')
    path.mkdir(parents=True)
    return path


def resolve_source(root, value):
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def text_inventory(data):
    """Lexical diagnostics only. Never used to prove semantics or handling.

    Strip comments and strings, retaining quoted generic operation names only
    immediately before '('. Count instruction-looking operation positions, not
    enum attributes or opcode text inside diagnostics. May undercount unusual
    formatting; dynamic costs and terminal retirement are deliberately unknown.
    """
    text = data.decode('utf-8')
    token = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.S)
    def mask(match):
        s = match.group()
        if s.startswith('"') and re.fullmatch(r'"(?:pto|scf|func|arith)\.[\w.]+"', s):
            if text[match.end():].lstrip().startswith('('):
                return s[1:-1]
        return ''.join('\n' if c == '\n' else ' ' for c in s)
    clean = token.sub(mask, text)
    pattern = re.compile(r'(?m)^\s*(?:%[\w.$:#-]+(?:\s*,\s*%[\w.$:#-]+)*\s*=\s*)?((?:pto|scf|func|arith)\.[\w.]+)\b')
    ops = Counter(m.group(1) for m in pattern.finditer(clean))
    barriers = Counter()
    for m in re.finditer(r'\bpto\.barrier\s*(?:<\s*)?(PIPE_[A-Z0-9]+)', clean):
        barriers[m.group(1)] += 1
    return dict(scope='lexical diagnostics, not semantic/effect verification', operations=dict(sorted(ops.items())),
                set_sites=ops['pto.set_flag'], wait_sites=ops['pto.wait_flag'],
                dynamic_set_sites=ops['pto.set_flag_dyn'], dynamic_wait_sites=ops['pto.wait_flag_dyn'],
                named_barriers=sum(n for p, n in barriers.items() if p != 'PIPE_ALL'),
                pipe_all_sites=barriers['PIPE_ALL'], barrier_pipes=dict(barriers),
                terminal_retirement='unclassified', executed_mechanisms=None,
                acknowledgment_roles='unclassified', acquired_prefixes=None)


def check_identifier(value):
    require(isinstance(value, str) and IDENT.fullmatch(value), f'Invalid identifier: {value!r}')
    return value


def capture(spec_path, output):
    spec_path = Path(spec_path).absolute()
    spec = load(spec_path)
    require(spec.get('schema') == SCHEMA, 'Wrong inventory schema')
    require(isinstance(spec.get('inputs'), list), 'Missing expected input inventory')
    require(set(spec.get('cohorts', {})) == set(COHORTS), 'All five required cohorts must be declared')
    require(set(spec.get('regressions', {})) == {'small_gemm', 'attention_refusals', 'prefill'},
            'Small GEMM, attention failures, and prefill must be declared, even when missing')
    for repo, revision in spec.get('source_revisions', {}).items():
        require(isinstance(repo, str) and REV.fullmatch(revision or ''), 'Source revisions must be full Git IDs')
    require(spec.get('source_revisions'), 'Source provenance is required')
    seen = set()
    for case in spec['inputs']:
        cid = check_identifier(case.get('id'))
        require(cid not in seen, f'Duplicate case: {cid}')
        seen.add(cid)
        require(case.get('cohort') in COHORTS, 'Unknown cohort')
        require(case.get('expectation') in ('supported', 'invalid', 'unclassified'), 'Explicit expectation required')
        require(case.get('authored') in ('none', 'protocol', 'unclassified'), 'Explicit authored-protocol classification required')
        require(isinstance(case.get('contracts'), dict) and set(CONTRACTS) <= case['contracts'].keys(), 'Missing input contracts')
        require(all(case['contracts'][key] is not None and case['contracts'][key] != '' for key in CONTRACTS), 'Empty input contract')
        require(case.get('expectation') != 'invalid' or (bool(case.get('invalid_reason')) and bool(case.get('invalid_diagnostic'))),
                'Invalid input needs a documented reason and expected diagnostic')
        require(all(isinstance(a, str) for a in case.get('compile_args', [])), 'compile_args must be strings')
        require(case.get('preparation', {}).get('kind') in ('identity', 'external'), 'Preparation must be explicit')
        require(case['preparation']['kind'] != 'external' or case['preparation'].get('argv'), 'External preparation command is missing')
        # Never execute per-case commands while freezing inputs.
    out = fresh_directory(output)
    files = []
    missing = [label + ': source revision missing' for label in ('PTOAS', 'PyPTO', 'pypto-lib') if label not in spec['source_revisions']]
    def keep(relative, data):
        write_new(safe(out, relative), data)
        info = dict(path=relative, sha256=sha(data), bytes=len(data))
        files.append(info)
        return info
    def copy(value, label):
        if not value:
            missing.append(label)
            return None
        try:
            data = stable_read(resolve_source(spec_path.parent, value))
        except (OSError, EvidenceError) as error:
            missing.append(f'{label}: {error}')
            return None
        return keep(f'objects/{sha(data)}', data) if not safe(out, f'objects/{sha(data)}').exists() else dict(path=f'objects/{sha(data)}', sha256=sha(data), bytes=len(data))
    keep('source-inventory.json', canonical(spec))
    cohorts = {}
    for cohort in COHORTS:
        source = spec['cohorts'][cohort]
        require(isinstance(source.get('collected'), bool), 'cohort.collected must be Boolean')
        evidence = copy(source.get('enumeration_log'), cohort + ': enumeration evidence')
        cohorts[cohort] = dict(collected=source['collected'], enumeration=evidence,
                               collection_argv=source.get('collection_argv', []), note=source.get('note', ''))
        if not source['collected'] or not source.get('collection_argv'):
            missing.append(cohort + ': collection incomplete')
        if not any(c['cohort'] == cohort for c in spec['inputs']):
            missing.append(cohort + ': no inventory entries')
    regressions = {}
    for name, source in spec['regressions'].items():
        regressions[name] = dict(case_ids=source.get('case_ids', []),
                                evidence=copy(source.get('evidence'), name + ': regression evidence'),
                                note=source.get('note', ''), status='captured' if source.get('evidence') else 'missing')
        minimum = source.get('minimum_cases', 1)
        require(type(minimum) is int and minimum > 0, 'Regression minimum must be positive')
        require(len(set(source.get('case_ids', []))) == len(source.get('case_ids', [])), 'Duplicate regression case IDs')
        if len(source.get('case_ids', [])) < minimum or not set(source['case_ids']) <= seen:
            missing.append(name + ': missing regression cases')
    cases = []
    for source in spec['inputs']:
        case = dict(source)
        case['raw'] = copy(source.get('raw'), case['id'] + ': raw')
        case['prepared'] = copy(source.get('prepared'), case['id'] + ': prepared')
        case['frontend_invocation_evidence'] = copy(source.get('frontend_invocation'), case['id'] + ': frontend invocation') if source.get('frontend_invocation') else None
        if source.get('capture_complete') is False:
            missing.append(case['id'] + ': interrupted/incomplete compiler capture')
        case['preparation_evidence'] = copy(source['preparation'].get('evidence'), case['id'] + ': preparation evidence') if source['preparation']['kind'] == 'external' else None
        if case['raw'] and case['prepared']:
            require(source['preparation']['kind'] != 'identity' or case['raw']['sha256'] == case['prepared']['sha256'],
                    f'Identity preparation changed bytes: {case["id"]}')
            case['lexical_inventory'] = text_inventory(stable_read(safe(out, case['prepared']['path'])))
            case['state'] = 'captured'
        else:
            case['state'] = 'missing'
        if case['expectation'] == 'unclassified' or case['authored'] == 'unclassified' or any(
                case['contracts'][key] == 'not-recorded' for key in CONTRACTS):
            missing.append(case['id'] + ': classification incomplete')
        cases.append(case)
    # Missing expected members remain in this manifest and in all later reports.
    result = dict(schema=SCHEMA, kind='capture', source_revisions=spec['source_revisions'], cohorts=cohorts,
                  regressions=regressions, cases=cases, files=files, missing=missing,
                  inventory_complete=not missing, evidence_claim='producer-declared inventory with retained evidence; not universal coverage')
    result['capture_id'] = sha(canonical(result))
    write_new(out / 'capture.json', canonical(result))
    return result


def verify_capture(root):
    root = Path(root)
    value = load(root / 'capture.json')
    require(value.get('schema') == SCHEMA and value.get('kind') == 'capture', 'Not a capture manifest')
    expected = value['capture_id']
    unsigned = dict(value)
    del unsigned['capture_id']
    require(SHA.fullmatch(expected) and sha(canonical(unsigned)) == expected, 'Capture manifest changed')
    paths = set()
    for entry in value['files']:
        require(entry['path'] not in paths, 'Duplicate captured path')
        paths.add(entry['path'])
        data = stable_read(safe(root, entry['path']))
        require(len(data) == entry['bytes'] and sha(data) == entry['sha256'], 'Captured bytes changed: ' + entry['path'])
    for case in value['cases']:
        for key in ('raw', 'prepared', 'preparation_evidence', 'frontend_invocation_evidence'):
            entry = case.get(key)
            require(not entry or entry['path'] in paths, 'Uninventoried input object')
    return value


def toolchain_identity(config, base):
    required = {'argv', 'artifacts', 'environment', 'versions', 'source_revision'}
    require(required <= config.keys(), 'Incomplete toolchain specification')
    require(config['source_revision'] == PRODUCTION_REVISION, 'M1 baseline must use the pinned production revision, not the experimental branch')
    require(set(config['argv']) == {'sync', 'lower'}, 'Both synchronization and final C++ lowering commands are required')
    artifacts = {}
    for label, supplied in config['artifacts'].items():
        check_identifier(label)
        path = resolve_source(base, supplied).resolve()
        artifacts[label] = dict(path=str(path), **file_identity(path))
    require('compiler' in artifacts, 'Compiler executable must be inventoried')
    require(isinstance(config['environment'], dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in config['environment'].items()), 'Environment must be explicit strings')
    require(all(isinstance(config['versions'].get(k), str) and config['versions'][k] and
                not config['versions'][k].lower().startswith(('record ', 'unknown', 'not-recorded'))
                for k in ('llvm_mlir', 'pto_isa', 'cann', 'runtime', 'device_profile')),
            'Record exact dependencies or not-applicable with reason, not template placeholders')
    for stage, argv in config['argv'].items():
        require(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv), 'argv must be an argument array')
        require(argv.count('{input}') == 1 and argv.count('{output}') == 1 and argv.count('{flags}') <= 1, 'Use one input/output and at most one flags placeholder')
        require(argv[0].startswith('{') and argv[0].endswith('}') and argv[0][1:-1] in artifacts, 'Executable must be a pinned artifact placeholder')
        for arg in argv:
            if arg.startswith('{'):
                require(arg in ('{input}', '{output}', '{flags}') or arg[1:-1] in artifacts, 'Unknown command placeholder')
    require('--enable-insert-sync' in config['argv']['sync'] and '--emit-pto-ir' in config['argv']['sync'],
            'Production baseline must explicitly insert synchronization and emit PTO')
    require(not any('insert-sync' in arg or arg == '--emit-pto-ir' for arg in config['argv']['lower']),
            'Lower the actual synchronized output without inserting synchronization again')
    # No implicit shell, no automatic arch rebinding, no fallback command.
    return dict(artifacts=artifacts, source_revision=config['source_revision'], versions=config['versions'],
                argv=config['argv'], environment=config['environment'])


def launch(argv, cwd, environment, stdout, stderr, timeout):
    started = time.monotonic()
    with Path(stdout).open('xb') as out, Path(stderr).open('xb') as err:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=out, stderr=err,
                                   start_new_session=(os.name == 'posix'))
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            code = process.wait()
    return dict(returncode=code, timed_out=timed_out, seconds=time.monotonic() - started,
                stdout_sha256=sha(stable_read(stdout)), stderr_sha256=sha(stable_read(stderr)))


def run_baseline(capture_root, config_path, output, timeout=120.0):
    require(math.isfinite(timeout) and timeout > 0, 'Timeout must be positive and finite')
    capture_root, config_path = Path(capture_root).absolute(), Path(config_path).absolute()
    cap, config = verify_capture(capture_root), load(config_path)
    out = fresh_directory(output)
    report = dict(schema=SCHEMA, kind='compiler_run', planner='existing', capture_id=cap['capture_id'],
                  inventory_complete=cap['inventory_complete'], status='incomplete', cases=[],
                  correctness='not_executed', device='not_executed')
    try:
        identity = toolchain_identity(config, config_path.parent)
        report['toolchain'] = identity
        report['toolchain_id'] = sha(canonical(identity))
        for case in cap['cases']:
            row = dict(id=case['id'], cohort=case['cohort'], expectation=case['expectation'],
                       authored=case['authored'], prepared_sha256=case['prepared']['sha256'] if case['prepared'] else None,
                       contracts_sha256=sha(canonical(case['contracts'])), handling='unclassified', stages=[])
            report['cases'].append(row)
            if case['state'] != 'captured':
                row['status'] = 'missing_input'
                continue
            if case['expectation'] == 'unclassified' or case['authored'] == 'unclassified' or any(
                case['contracts'][key] == 'not-recorded' for key in CONTRACTS):
                row['status'] = 'unclassified_input'
                continue
            require(not any('insert-sync' in arg or arg.startswith(('--emit-', '-o', '--output')) or arg in ('--', '-') for arg in case.get('compile_args', [])),
                    'Case arguments cannot override synchronization/output or introduce another input')
            require(all(arg.startswith('-') and ('=' in arg or arg.startswith('--')) for arg in case.get('compile_args', [])),
                    'Case flags must be named options, with values supplied using =')
            case_dir = out / case['id']
            case_dir.mkdir()
            source = safe(capture_root, case['prepared']['path'])
            # A private copy prevents compiler wrappers from mutating captured inputs.
            input_copy = case_dir / 'prepared.pto'
            write_new(input_copy, stable_read(source))
            current = input_copy
            row['status'] = 'compiled'
            for stage, extension in (('sync', 'pto'), ('lower', 'cpp')):
                destination = case_dir / (stage + '.' + extension)
                substitutes = {key: a['path'] for key, a in identity['artifacts'].items()}
                substitutes.update(input=str(current), output=str(destination))
                argv = []
                for arg in identity['argv'][stage]:
                    if arg == '{flags}':
                        argv.extend(case.get('compile_args', []))
                    else:
                        argv.append(substitutes[arg[1:-1]] if arg.startswith('{') and arg.endswith('}') else arg)
                record = dict(stage=stage, argv=argv, input_sha256=sha(stable_read(current)))
                row['stages'].append(record)
                try:
                    record.update(launch(argv, case_dir, identity['environment'], case_dir / (stage + '.stdout'),
                                         case_dir / (stage + '.stderr'), timeout))
                except OSError as error:
                    record.update(error=str(error), returncode=None, timed_out=False)
                if record.get('timed_out'):
                    row['status'] = 'timeout'
                    break
                if record.get('returncode') != 0:
                    row['status'] = 'compiler_error' if record.get('returncode') is not None and record['returncode'] > 0 else 'infrastructure_error'
                    row['handling'] = 'refused' if row['status'] == 'compiler_error' else 'unclassified'
                    break
                if not destination.is_file() or destination.stat().st_size == 0:
                    row['status'] = 'missing_output'
                    break
                require(sha(stable_read(current)) == record['input_sha256'], 'Compiler mutated its input')
                data = stable_read(destination)
                record.update(output=str(destination.relative_to(out)), output_sha256=sha(data), bytes=len(data))
                if stage == 'sync':
                    row['synchronized_sha256'] = sha(data)
                    row['static'] = text_inventory(data)
                current = destination
            if row['status'] == 'compiled':
                row['handling'] = 'authored_unverified' if case['authored'] == 'protocol' else 'existing_completed'
            row['expectation_met'] = row['status'] == 'compiled' if case['expectation'] == 'supported' else (
                row['status'] == 'compiler_error' and row['stages'][-1]['stage'] == case.get('invalid_stage', 'sync') and
                case['invalid_diagnostic'] in stable_read(case_dir / (row['stages'][-1]['stage'] + '.stderr')).decode('utf-8', errors='replace'))
        # Detect replacement of compiler binaries/dependencies while the serial campaign runs.
        require(toolchain_identity(config, config_path.parent) == identity, 'Toolchain changed during campaign')
        verify_capture(capture_root)
        report['status'] = 'completed' if all(row.get('expectation_met', False) for row in report['cases']) else 'incomplete'
        if not report['cases'] or not cap['inventory_complete']:
            report['status'] = 'incomplete'
        report['totals'] = dict(Counter(row['status'] for row in report['cases']))
    except (OSError, EvidenceError, UnicodeError) as error:
        report['error'] = str(error)
    finally:
        recorded = {row['id']: row for row in report['cases']}
        for case in cap['cases']:
            if case['id'] not in recorded:
                recorded[case['id']] = dict(id=case['id'], cohort=case['cohort'], expectation=case['expectation'],
                    authored=case['authored'], prepared_sha256=case['prepared']['sha256'] if case['prepared'] else None,
                    contracts_sha256=sha(canonical(case['contracts'])), handling='unclassified', stages=[])
            recorded[case['id']].setdefault('status', 'not_run')
            recorded[case['id']].setdefault('expectation_met', False)
        report['cases'] = [recorded[case['id']] for case in cap['cases']]
        report['totals'] = dict(Counter(row['status'] for row in report['cases']))
        report['run_id'] = sha(canonical(report))
        write_new(out / 'run.json', canonical(report))
    return report


def verify_run(root):
    root = Path(root)
    report = load(root / 'run.json')
    expected = report['run_id']
    unsigned = dict(report)
    del unsigned['run_id']
    require(sha(canonical(unsigned)) == expected, 'Run manifest changed')
    for case in report['cases']:
        for stage in case['stages']:
            for stream in ('stdout', 'stderr'):
                digest = stage.get(stream + '_sha256')
                if digest:
                    path = safe(root, case['id'] + '/' + stage['stage'] + '.' + stream)
                    require(sha(stable_read(path)) == digest, 'Command log changed')
            if stage.get('output'):
                require(sha(stable_read(safe(root, stage['output']))) == stage['output_sha256'], 'Compiler output changed')
    return report


def compare(capture_root, baseline_root, candidate_path):
    """Read a future pass's explicit handling receipts; never infer them from counts.

    Receipt fields are claims by a pinned producer, not cryptographic proofs of
    algorithm identity. Compiler mutation/device checks remain separate gates.
    """
    cap, baseline = verify_capture(capture_root), verify_run(baseline_root)
    candidate_path = Path(candidate_path).absolute()
    candidate = load(candidate_path)
    require(candidate.get('capture_id') == baseline['capture_id'] == cap['capture_id'], 'Different input populations')
    require(candidate.get('schema') == SCHEMA and candidate.get('kind') == 'candidate_receipts', 'Not explicit candidate receipts')
    require(candidate.get('planner') == 'oahs' and REV.fullmatch(candidate.get('source_revision', '')), 'Candidate planner/revision not pinned')
    require(SHA.fullmatch(candidate.get('toolchain_id', '')), 'Candidate toolchain identity missing')
    supplied = {}
    for entry in candidate.get('cases', []):
        cid = check_identifier(entry.get('id'))
        require(cid not in supplied, 'Duplicate candidate receipt')
        supplied[cid] = entry
    original = {c['id']: c for c in cap['cases']}
    require(set(supplied) <= set(original), 'Candidate introduces a different population')
    base = {c['id']: c for c in baseline['cases']}
    require(set(base) == set(original), 'Baseline silently dropped expected inputs')
    rows = []
    for cid, case in original.items():
        b, entry = base[cid], supplied.get(cid)
        result = dict(id=cid, cohort=case['cohort'], handling='missing', covered=False)
        if entry:
            require(entry.get('handling') in HANDLING, 'Unknown handling classification')
            require(entry.get('prepared_sha256') == b['prepared_sha256'] and entry.get('contracts_sha256') == b['contracts_sha256'], 'Changed source or contracts')
            result['handling'] = entry['handling']
            if entry['handling'] in ('generated', 'authored_preserved', 'noop_verified', 'fallback'):
                for kind in ('output', 'verification'):
                    artifact = entry.get(kind, {})
                    require(SHA.fullmatch(artifact.get('sha256', '')), 'Missing receipt artifact identity')
                    require(sha(stable_read(resolve_source(candidate_path.parent, artifact['path']))) == artifact['sha256'], 'Receipt artifact changed')
                require(entry.get('verification_status') in ('passed', 'failed', 'not_run'), 'Verification outcome missing')
                accepted = entry['verification_status'] == 'passed' and entry['handling'] != 'fallback'
                accepted &= not (entry['handling'] == 'authored_preserved' and case['authored'] != 'protocol')
                result['covered'] = accepted
        result['negative_passed'] = None
        if case['expectation'] == 'invalid':
            result['negative_passed'] = False
            if entry and entry['handling'] == 'refused' and entry.get('diagnostic'):
                artifact = entry['diagnostic']
                data = stable_read(resolve_source(candidate_path.parent, artifact['path']))
                require(sha(data) == artifact['sha256'], 'Negative diagnostic changed')
                result['negative_passed'] = case['invalid_diagnostic'] in data.decode('utf-8', errors='replace')
        result['baseline_supported'] = case['expectation'] == 'supported' and b['status'] == 'compiled'
        result['baseline_unresolved'] = case['expectation'] == 'supported' and b['status'] != 'compiled'
        rows.append(result)
    denominator = sum(r['baseline_supported'] for r in rows)
    numerator = sum(r['baseline_supported'] and r['covered'] for r in rows)
    complete = cap['inventory_complete'] and baseline['status'] == 'completed' and denominator > 0 and numerator == denominator and all(r['negative_passed'] is not False for r in rows)
    return dict(schema=SCHEMA, kind='coverage_comparison', status='passed' if complete else 'incomplete',
                capture_id=cap['capture_id'], baseline_run_id=baseline['run_id'],
                numerator=numerator, denominator=denominator,
                automatic_synthesis=sum(r['baseline_supported'] and r['covered'] and
                    r['handling'] in ('generated', 'noop_verified') for r in rows),
                authored_protocol_handling=sum(r['baseline_supported'] and r['covered'] and
                    r['handling'] == 'authored_preserved' for r in rows),
                baseline_unresolved=sum(r['baseline_unresolved'] for r in rows),
                handling=dict(Counter(r['handling'] for r in rows)), rows=rows,
                device_quality='not_established', evidence='explicit producer receipts; not inferred from process exit or synchronization count')


def device_summary(capture_root, run_root, metrics_path):
    cap, run = verify_capture(capture_root), verify_run(run_root)
    require(cap['capture_id'] == run['capture_id'], 'Device run/capture mismatch')
    metrics_path = Path(metrics_path).absolute()
    metrics = load(metrics_path)
    require(metrics.get('schema') == SCHEMA and metrics.get('kind') == 'device_measurements', 'Wrong device evidence schema')
    require(metrics.get('run_id') == run['run_id'] and metrics.get('device') and metrics.get('measurement_protocol'), 'Device protocol/run binding missing')
    raw = metrics.get('raw_log', {})
    require(SHA.fullmatch(raw.get('sha256', '')), 'Raw timing log identity missing')
    require(sha(stable_read(resolve_source(metrics_path.parent, raw['path']))) == raw['sha256'], 'Raw timing log changed')
    byid = {case['id']: case for case in run['cases']}
    expected = metrics.get('expected_cases', [])
    require(expected and len(set(expected)) == len(expected) and set(expected) <= set(byid), 'Explicit measured workload inventory is invalid')
    rows, seen = [], set()
    for sample in metrics.get('kernels', []):
        cid = sample['id']
        require(cid not in seen and cid in expected, 'Unexpected or duplicate kernel timing')
        seen.add(cid)
        require(sample.get('synchronized_sha256') == byid[cid].get('synchronized_sha256'), 'Timing belongs to another emitted kernel')
        binary, build = sample.get('binary', {}), sample.get('build_record', {})
        for artifact in (binary, build):
            require(SHA.fullmatch(artifact.get('sha256', '')), 'Executed binary/build record is missing')
            require(file_identity(resolve_source(metrics_path.parent, artifact['path']))['sha256'] == artifact['sha256'],
                    'Device binary/build evidence changed')
        build_data = load(resolve_source(metrics_path.parent, build['path']))
        require(build_data.get('synchronized_sha256') == sample['synchronized_sha256'] and
                build_data.get('binary_sha256') == binary['sha256'] and build_data.get('argv'),
                'Build record does not link synchronized PTO to executed binary')
        times = sample.get('microseconds', [])
        require(times and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in times), 'Invalid timing samples')
        require(sample.get('correctness') in ('passed', 'failed', 'not_run'), 'Correctness outcome missing')
        require(type(sample.get('invocations')) is int and sample['invocations'] > 0, 'Invocation count missing')
        rows.append(dict(id=cid, median_us=statistics.median(times), samples=len(times),
                         invocations=sample['invocations'], correctness=sample['correctness']))
    rows.sort(key=lambda r: r['median_us'] * r['invocations'], reverse=True)
    missing = sorted(set(expected) - seen)
    # A measured subset cannot establish all model-kernel baseline timings.
    required = {c['id'] for c in cap['cases'] if c['cohort'] in ('model_kernels', 'device_regressions') and c['expectation'] == 'supported'}
    missing_inventory = sorted(required - set(expected))
    model = metrics.get('model', {})
    model_ok = isinstance(model, dict) and model.get('correctness') == 'passed' and model.get('microseconds') and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in model['microseconds'])
    complete = cap['inventory_complete'] and run['status'] == 'completed' and not missing and not missing_inventory and model_ok and all(r['correctness'] == 'passed' for r in rows)
    return dict(schema=SCHEMA, kind='device_summary', status='recorded' if complete else 'incomplete',
                run_id=run['run_id'], device=metrics['device'], measurement_protocol=metrics['measurement_protocol'],
                missing_cases=missing, missing_workload_inventory=missing_inventory, ranked_kernels=rows,
                model=model, provenance='externally measured evidence imported, not executed by this tool',
                caution='Per-kernel weighted time is an attribution aid, not the model critical path or a speedup proof.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('capture'); p.add_argument('--spec', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('verify'); p.add_argument('capture', type=Path)
    p = sub.add_parser('baseline'); p.add_argument('--capture', type=Path, required=True); p.add_argument('--toolchain', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--timeout', type=float, default=120)
    p = sub.add_parser('compare'); p.add_argument('--capture', type=Path, required=True); p.add_argument('--baseline', type=Path, required=True); p.add_argument('--candidate', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('device'); p.add_argument('--capture', type=Path, required=True); p.add_argument('--run', type=Path, required=True); p.add_argument('--metrics', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        report = capture(args.spec, args.out)
        status = report['inventory_complete']
    elif args.command == 'verify':
        report = verify_capture(args.capture)
        status = report['inventory_complete']
    elif args.command == 'baseline':
        report = run_baseline(args.capture, args.toolchain, args.out, args.timeout)
        status = report['status'] == 'completed'
    elif args.command == 'compare':
        report = compare(args.capture, args.baseline, args.candidate)
        write_new(args.out, canonical(report)); status = report['status'] == 'passed'
    else:
        report = device_summary(args.capture, args.run, args.metrics)
        write_new(args.out, canonical(report)); status = report['status'] == 'recorded'
    print(canonical(report).decode(), end='')
    return 0 if status else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Evidence error: {exc}', file=sys.stderr)
        raise SystemExit(1)
