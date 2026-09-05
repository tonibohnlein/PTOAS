# ProtocolSync evidence campaigns

These are host-only, read-only diagnostic campaigns, not synchronization
selection or hardware correctness tests. Keep campaign outputs in a disk-backed
build directory, not `/tmp`. A campaign never overwrites an existing result
directory and uses at most two compiler subprocesses. Do not overlap it with
other resource-intensive jobs on this machine.

## Reproduction

Build and test a committed source revision first. The runner rejects tracked
edits and uncommitted experiment scripts unless `--allow-dirty` is explicitly
used for debugging. Debug runs are not clean baselines.

```bash
cmake --build build --parallel 2
.venv/bin/python test/experiments/protocol_sync/campaign.py \
  --mode storage --workers 2 --expected-rows 394 \
  --input-root build/protocol-sync-lane-c6-corpus/inputs \
  --manifest build/protocol-sync-lane-c6-corpus/inputs/build/device-corpus-d99520397/manifest.tsv \
  --results build/protocol-sync-c71/storage
```

Use `--mode lane` and a different fresh result directory for C.6. Manifest rows
are tab-separated and require `case_id` and `source`; `source` is relative to
`--input-root`. Supplied `sha256` and `bytes` fields are verified before any
compiler invocation. Case IDs must be unique and cannot contain path separators.

For the frozen old emitted programs:

```bash
.venv/bin/python test/experiments/protocol_sync/campaign.py \
  --mode concrete --workers 2 --expected-rows 199 \
  --input-root build/protocol-sync-old-emitted/canonical-frontier-394-a3-19674e06-final/artifacts \
  --results build/protocol-sync-c71/concrete
```

Concrete mode discovers `CASE/frontier_on/out.pto`, reparses already physically
assigned IR at level 3, and invokes F's concrete verifier. It does not run legacy
selection or interpret empty-world residual counts as concrete coverage. A
rejected concrete verdict is a completed audit, not a runner failure. A compiler
error, missing statistics, corrupt input hash, or changed source/compiler during
the run makes the campaign fail.

The default compiler is the configured Python-package launcher under `build/`.
`--build-root`, `--ptoas`, and `--python` allow another compatible configured
tree. Provenance collection requires its CMake cache, compiler shared library,
LLVM build and source checkout. It records host-only scope explicitly; this
does not certify CANN/PTO-ISA runtime versions or device behavior.

## Artifacts and interpretation

- `run.json`: source commit, dirty-diff hash, experiment-script hashes, ordered
  input-set hash, manifest hash, exact invocation, compiler/cache/toolchain
  fingerprints, relevant environment, start/end times and stability check.
- `rows.jsonl`: every row's command, input and diagnostic hashes, return status,
  per-function statistics, access/root/region records, transitions, costs,
  target queries, E admission and rejection records, and concrete verdicts.
- `diagnostics/`: deterministic gzip archives of complete per-row output.
- `summary.json`: additive counts separately from maxima; rejected recurring
  records by function instance, name, structured-topology hash and first proof;
  unprojected access categories; multi-family use shapes; atom expansion.
- `hashes.json`: hashes of every generated artifact, excluding itself.

Archive the whole directory alongside the input manifest/archive. Commit the
compact aggregates and an archive digest to the evidence ledger. A digest alone
does not make missing raw artifacts reproducible: retain the actual archive.

`archive_results.py --campaign RESULTS ... --include INPUT_ARCHIVE ...
--output FRESH_SUMMARY_DIRECTORY --archive FRESH_ARCHIVE.tar.gz` validates the
completion manifests, archives raw evidence and emits compact summaries with
per-row hashes. Run it serially. It explicitly records whether the campaigns
were clean committed runs; archiving does not upgrade a debug/dirty run into
a clean baseline. Executed scripts are also snapshotted inside each campaign.

Topology hashes describe ordered structured regions and phase resources. They
exclude operation/function names and addresses, but are not a semantic
equivalence or graph-isomorphism test. E control refinements follow the current
recognizer's region-scan order; they describe the first failed admission gate,
not every reason a generalized protocol could be invalid.

Multi-family categories are deliberately qualified:

- `equivalent-root-views` means the recorded allocation root agrees; it is not
  a proof that the views' access ranges or valid shapes agree.
- `sequential-use-candidate` and `overlapping-use-candidate` describe lexical
  access spans. Neither establishes concurrent hardware liveness.
- `cross-scope-numeric-coincidence-candidate` requires different recorded
  lexical allocation scopes. Lexical scope is not yet the storage alias domain.
- Missing provenance or guarded/looping accesses stay explicitly unresolved.

The PyPTO/PyPTO-Lib 152-kernel acceptance population is separate from these 394
differential rows. The supplied `c4f924114` diagnosis is an externally reported
baseline, not a locally reproduced result. The user selected a fresh population
from current-main PyPTO/PyPTO-Lib. Freeze those revisions and generated inputs;
do not label the new population as the historical 152 or infer admission
improvements from blocker counts. Retain collection failures as explicit rows.

Use `--mode acceptance --expected-rows COUNT` with the frozen population's
manifest and its actual row count.
For each row this collects a storage/schedule snapshot, a separate empty-world
residual snapshot, and actual strict mixed emission with fallback disabled.
Optional manifest `level` is `level2` or `level3` (default: already physically
assigned level 3). It never guesses physical-core ownership or GM noalias.

The acceptance matrix preserves *all exposed* extraction, generation/channel,
residual-kind, and direct-repair diagnostics even when strict emission stops at
an earlier gate. Missing extraction still limits the downstream universe; an
absent diagnostic is not a successful proof. Empty-world obligations can occur
in accepted programs too. Strict admission/rejection is therefore recorded
separately from overlapping blocker occurrences. Per-row raw logs remain the
authority for unparsed or newly introduced diagnostics.
