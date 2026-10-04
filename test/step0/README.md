# Step 0: shared synchronization input

Step 0 collects the information used by the analysis routes. It does not select
an exact route, compute demands, insert synchronization, or allocate event IDs.

Its reusable components are `SyncInput`, `PhaseIndex`, and `SyncStorageEffects`:

- Reuse InsertSync's translator, pipe assignments and buffer records.
- Preserve every shared phase read/write record and every operation-interface
  effect declaration, including declarations without a translated pipe phase.
- Preserve source operation anchors, original regions, loop bounds and branch
  conditions. Unsupported expressions retain their SSA identity.
- Retain planned addresses and view ranges. Reuse the existing alias/range query
  for possible conflicts, with the selected GM alias policy.
- Attach symbolic descriptor maps and slot selectors where available. A missing
  map is an unresolved description, not an empty access or an extraction failure.

The read-only `pto-sync-input-test --step0-json INPUT` audit checks declarations
against the original MLIR interfaces, records against the shared phase vectors,
loop domains, SSA availability and unchanged source IR. It reports declarations
without a pipe phase separately. These include descriptor metadata and existing
pipe/communication operations; preserving them does not assign new semantics to
them. No cross-core synchronization analysis is added.

`--existing-check INPUT` independently runs the existing InsertSync pass on a
prepared module. This verifies pass acceptance at the same pipeline boundary;
it is not a C++ code-generation or device check.

## Corpus runner

Run serially with the selected workspace Python and freshly built test tool:

```sh
.venv/bin/python test/step0/run_corpus.py \
  --tool /path/to/pto-sync-input-test \
  --manifest /path/to/manifest.json \
  --output /path/to/new-results-directory
```

The manifest contains `cases`, each with `id`, `family`, `input` (prepared IR),
`sha256` and the expected number of `functions`. Optional `source` and
`source_sha256` bind the snapshot to the original frontend input. Record snapshot
preparation provenance alongside the manifest. Paths are local files selected by
the operator, not remote downloads or shell fragments.

For each input, the runner checks hashes, runs existing InsertSync, then audits
Step 0 with both GM alias policies. It requires identical collected information
under both policies and preserves commands, logs, timing, hashes and failures.
The output directory must be new. The tool disables MLIR multithreading; the
runner launches one child at a time and kills its process group on timeout.

## Claim supported by this test

Success establishes preservation of the shared compiler input for the tested
snapshots. It does not establish completeness of native instruction effect
semantics, exact accessed-byte sets, a compact normal form for every expression,
or acceptance by the tractable-case recognizers. Those stronger requirements
remain explicit inputs to later analysis routes. In particular, buffer-range
information never establishes a complete overwrite by itself.
