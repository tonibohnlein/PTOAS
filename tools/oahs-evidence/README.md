# OAHS coverage and measurement tools

Python 3.10+ standard library; Linux/POSIX for the transparent compiler shim.
No compiler modifications or dependencies on any old OAHS implementation.
Start with `pins.json` and the inventory/toolchain templates. The live pass is
described in [selected-plan construction](../../docs/designs/oahs-selected-plan.md).

## Coverage denominator

Enumerate PTOAS regressions, PyPTO operations, pypto-lib, model-generated kernels,
and device cases at exact revisions under their original target, alias, and ABI
contracts. Reconcile frontend collection logs with captured compiler invocations;
failures before compiler invocation remain missing-input rows. Keep authored
protocols, intentional invalids, refusals, and timeouts explicit. Repeated source
bytes under different runtime contracts remain separate cases.

Report analysis admission, automatic synthesis, authored-protocol preservation,
and device correctness separately. Compilation does not establish numerical
correctness; command counts do not establish speedup. A partially enumerated
cohort cannot support a complete-coverage claim.

## 1. Test the infrastructure

```sh
python3 -m unittest discover -s tools/oahs-evidence -p 'test_*.py' -v
```

These tests execute a **synthetic compiler** to validate the runner and failure
classification. They are not PTOAS, IR-verifier, kernel numerical, or device tests.

## 2. Capture the actual frontend inventory

Run the real PyPTO/pypto-lib/model suites with a compiler launcher such as:

```sh
#!/bin/sh
exec /absolute/python3 /absolute/PTOAS/tools/oahs-evidence/capture_compiler.py \
  --real /absolute/pinned-production/ptoas \
  --records /absolute/captures/pypto-ops \
  --context /absolute/context.json -- "$@"
```

Select the launcher through the frontend's existing compiler-executable setting.
Do not point `--real` back to the launcher or put it in a directory that shadows
itself. The shim preserves argv, input, stdout/stderr and exit code. It does not
synthesize a fallback. The complete module is retained, including helper symbols.

The per-workload `context.json` should contain:

```json
{
  "source_revisions": {"PyPTO": "REPLACE_WITH_40_HEX_COMMIT", "pypto-lib": "REPLACE_WITH_40_HEX_COMMIT"},
  "expectation": "supported",
  "authored": "none",
  "contracts": {
    "architecture": "a3",
    "hardware": "exact production target contract/version",
    "alias": "original caller contract; no added disjointness",
    "abi": "original scalar bounds, shapes, launch and argument contract",
    "memory_planner": "the actual frontend planner"
  },
  "compile_args": ["--pto-arch=a3", "--pto-level=level3"]
}
```

Those are example values, not permission to rebind an A5 input to A3 or strengthen
its alias/ABI contract. Intentional negative tests need `expectation=invalid`,
`invalid_reason`, `invalid_diagnostic` (literal), and optionally `invalid_stage`.
Authored-protocol tests use `authored=protocol`.

```sh
python3 tools/oahs-evidence/collect_records.py \
  --group pypto_ops=/absolute/captures/pypto-ops \
  --group model_kernels=/absolute/captures/prefill \
  --out /absolute/work/inventory.json
```

The collector does not certify frontend completeness. Reconcile every upstream
collected test/callsite, including failures before compiler invocation, against the
inventory. Add rows with null raw/prepared paths for unavailable expected inputs.
Provide the enumeration logs and exact collection commands, then set each cohort's
`collected` flag only when that reconciliation is done. Record original runtime
parameter cases, rather than treating equal source bytes as equal executions.

`inventory.template.json` starts with all five cohorts and the three reported
regression populations explicitly missing. It is expected to fail qualification.
The minimum of 44 prefill captures is not a cap; also include refused/fallback
kernels. Do not replace these cases with the old frozen historical benchmark set.

## 3. Freeze exact inputs

```sh
python3 tools/oahs-evidence/evidence.py capture \
  --spec /absolute/work/inventory.json --out /absolute/work/frozen
python3 tools/oahs-evidence/evidence.py verify /absolute/work/frozen
```

Exit 0 means the declared inventory is captured and reconciled; exit 2 means
incomplete evidence retained in the manifest; exit 1 means malformed/tampered
inputs or infrastructure error. This is not semantic correctness or OAHS coverage.
Output directories must be new. No silent reuse of earlier artifacts.

Source `raw`/`prepared` paths are relative to the spec or absolute. Preparation is
identity unless an external command and its provenance are explicitly recorded.
The tools do not run arbitrary preparation commands or modify source bytes.

## 4. Run the pinned InsertSync baseline

Copy and fill `toolchain.template.json`. Pin the actual compiler executable,
compiler/MLIR libraries, interpreter/launcher where relevant, build configuration
and any other toolchain files in `artifacts`. Record exact versions and explicit
runtime environment. The example paths are intentionally nonexistent. A placeholder
configuration must not be reported as a successful baseline.

```sh
python3 tools/oahs-evidence/evidence.py baseline \
  --capture /absolute/work/frozen \
  --toolchain /absolute/work/toolchain.json \
  --out /absolute/work/existing-baseline
```

The runner invokes `--enable-insert-sync --emit-pto-ir`, then compiles that exact
synchronized output to C++ without reinserting synchronization. Commands are serial;
there is no parallel build and no more than one compiler process from this runner.
It does not control work launched outside it. No source-level `planner` option is
introduced. Input flags cannot override synchronization selection or output paths.

Actual native binary construction, device launch, warmup and timing remain with the
existing deployment harness; do not infer them from successful host compilation.
Compiler times here are per-stage wall-clock telemetry, not device latency.

## 5. Attach device results and candidate receipts

`device` expects a `device_measurements` JSON bound to `run_id`, with a raw log
reference `{path,sha256}`, exact device/protocol description, `expected_cases`,
per-kernel records (`id`, `synchronized_sha256`, `microseconds`, `invocations`,
`correctness`), and separately measured `model` results. Every kernel record also
needs `binary` and `build_record` hashed file references; the build record must
name `synchronized_sha256`, `binary_sha256` and the actual build `argv`.
The source of these measurements is explicitly external.

```sh
python3 tools/oahs-evidence/evidence.py device \
  --capture /absolute/work/frozen --run /absolute/work/existing-baseline \
  --metrics /absolute/work/device-measurements.json --out /absolute/work/device-summary.json
```

`compare` consumes `candidate_receipts` with the same
`capture_id`, a pinned source/toolchain ID, and one explicit handling record per
case. Allowed handling: `generated`, `authored_preserved`, `noop_verified`,
`fallback`, `refused`, `unclassified`. Successful records require hashed output
and verifier evidence plus a verifier status. Contract and prepared-input hashes
must match the baseline. Fallback never counts as OAHS coverage. Deliberate invalids
need a hashed diagnostic with the expected error. These receipts bind external compiler and verifier evidence; the evidence tools
do not synthesize synchronization or establish correctness themselves.

```sh
python3 tools/oahs-evidence/evidence.py compare \
  --capture /absolute/work/frozen --baseline /absolute/work/existing-baseline \
  --candidate /absolute/work/oahs-receipts.json --out /absolute/work/coverage.json
```

Use `test_evidence.py` for executable schema examples. Its times and artifacts are
explicitly synthetic; never import them as actual baseline/device evidence.
