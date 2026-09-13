# OAHS frozen corpus reproducibility

This milestone freezes 363 PTO inputs and their exact baseline/current compiler
outcomes. It is a compiler-compatibility record, not a device-correctness,
performance, or upstream-source-regeneration claim.

The version-controlled evidence consists of:

- `frozen-363.jsonl`: one canonical record per input, in stable order;
- `frozen-363.lock.json`: named roots, exact command arms, source/evidence
  provenance, aggregate outcomes, and the JSONL byte hash;
- `frozen-363.schema.json`: the portable record schema; and
- `root-map.example.json`: the host-specific mapping for source-backed replay;
  a complete transported CAS needs no root map.

The lock's `source_manifests` entries are hashes of external provenance
manifests used to assemble this record. Those external files are not included
or located by this package, so normal frozen verification checks the 363
per-record identities rather than claiming to reverify the external manifest
artifacts themselves.

PTO payloads are deliberately not committed. Original and prepared bytes share
the canonical address `sha256/xx/<digest>` derived from their respective hashes;
identical populations share one object. An optional external content-addressed
store can therefore transport the complete frozen input population.

## Evidence boundary

`PTOAS` records identify hash-frozen PTO files in a PTOAS source tree.
`PYPTO_GENERATED` records identify hash-frozen generated PTO snapshots. The
latter root is not a PyPTO or PyPTO-lib source checkout, and this corpus cannot
reproduce those snapshots from upstream Python sources.

For every record, `original` identifies the exact bytes under its named root.
`prepared` identifies the exact bytes passed to the compiler. The only permitted
preparation recipe is `bind-a2a3-to-a3-v1`: replace an `a2a3` module target with
`a3`, without changing payload or control-flow IR. Records already selecting a
concrete target remain byte-identical.

Reproducing upstream generation is a separate campaign. It needs pinned PyPTO
and PyPTO-lib source revisions, environment/toolchain provenance, the exact
generation command, and a generated-output hash that matches the corresponding
`PYPTO_GENERATED` original. None of that is inferred by this lock.

The frozen baseline/current outcomes were captured from the campaign base,
dirty-listing hash, compiler hash, and summary hash named in the lock. The
`current` reference implementation is committed revision
`bc95ab98ee965ae8978ee44ade8bcd80193bb9f1`, the production baseline for later
candidates. The separate campaign fields, including the saved compiler hash,
preserve how its evidence was
originally captured; they do not claim that a newly rebuilt binary has the old
binary hash or retroactively relabel a dirty campaign as a clean run.

The lock accepts only the two checked-in, reviewed public-option argv templates.
It rejects unknown placeholders, hard-coded input paths, and additional options
instead of executing arbitrary lock-provided arguments. Replay uses the
executable supplied explicitly by `--opt`; the saved reference compiler is a
hash, not an executable path and is never invoked.

## Root map

A root map is required to export a CAS and may be used to verify or replay
against the original source locations. It is optional for verification and
replay when a complete original/prepared CAS is supplied. To use source
locations, copy `root-map.example.json` outside the repository and replace both
values:

```json
{
  "PTOAS": "/disk/checkouts/PTOAS",
  "PYPTO_GENERATED": "/disk/corpora/pypto-generated-pto"
}
```

The generated root must contain the frozen `inputs/...` paths, not upstream
Python files. A root map must define exactly the named roots in the lock. When a
complete CAS is supplied without a root map, the transported original snapshots
replace these host paths. They remain generated PTO evidence, not a reproduction
from upstream PyPTO/PyPTO-lib sources.

## Validate all source and prepared bytes

From the repository root:

```sh
python3 test/experiments/insert_sync/logical_plan/s7_corpus.py frozen-verify \
  --manifest test/experiments/insert_sync/logical_plan/corpus/frozen-363.jsonl \
  --lock test/experiments/insert_sync/logical_plan/corpus/frozen-363.lock.json \
  --root-map /disk/oahs-corpus/root-map.json
```

This reads all 363 originals, verifies each original SHA-256, derives the
prepared bytes in memory, and verifies each prepared SHA-256. No compiler is
invoked.

To require a previously exported external content-addressed store as well:

```sh
python3 test/experiments/insert_sync/logical_plan/s7_corpus.py frozen-verify \
  --manifest test/experiments/insert_sync/logical_plan/corpus/frozen-363.jsonl \
  --lock test/experiments/insert_sync/logical_plan/corpus/frozen-363.lock.json \
  --root-map /disk/oahs-corpus/root-map.json \
  --cas-root /disk/oahs-corpus/cas
```

Every addressed object must exist and equal the locally derived prepared bytes,
not merely have the expected filename. The same command also requires and
cross-checks each original CAS object against its named-root source.

A complete CAS can be verified on another host without the historical roots:

```sh
python3 test/experiments/insert_sync/logical_plan/s7_corpus.py frozen-verify \
  --manifest test/experiments/insert_sync/logical_plan/corpus/frozen-363.jsonl \
  --lock test/experiments/insert_sync/logical_plan/corpus/frozen-363.lock.json \
  --cas-root /disk/oahs-corpus/cas
```

For every record this verifies the original object hash, independently derives
the prepared bytes using the locked recipe, verifies the prepared hash, and
requires the prepared CAS object to equal those derived bytes.

## Export the optional payload store

The output must be outside the repository and every named source root:

```sh
python3 test/experiments/insert_sync/logical_plan/s7_corpus.py frozen-export \
  --manifest test/experiments/insert_sync/logical_plan/corpus/frozen-363.jsonl \
  --lock test/experiments/insert_sync/logical_plan/corpus/frozen-363.lock.json \
  --root-map /disk/oahs-corpus/root-map.json \
  --output /disk/oahs-corpus/cas
```

The export contains both original and prepared byte populations. Existing
objects are accepted only when their exact bytes hash correctly. Objects are
written under their content address and verified before publication. The
exporter rejects symlinked output components and checks every resolved object
path remains inside the external CAS and outside all source roots. CAS roots
must also be outside Git and every known source root in verify and replay mode.

## Replay serially

Replay is deliberately serial. It preflights every original, prepared byte
sequence, and requested CAS object before creating the result directory or
starting the first compiler process:

```sh
python3 test/experiments/insert_sync/logical_plan/s7_corpus.py frozen-replay \
  --manifest test/experiments/insert_sync/logical_plan/corpus/frozen-363.jsonl \
  --lock test/experiments/insert_sync/logical_plan/corpus/frozen-363.lock.json \
  --root-map /disk/oahs-corpus/root-map.json \
  --cas-root /disk/oahs-corpus/cas \
  --opt /disk/build/bin/ptoas-opt \
  --output /disk/oahs-corpus/replay-001 \
  --arm both \
  --timeout 120
```

After transporting a complete CAS, `--root-map` may be omitted:

```sh
python3 test/experiments/insert_sync/logical_plan/s7_corpus.py frozen-replay \
  --manifest test/experiments/insert_sync/logical_plan/corpus/frozen-363.jsonl \
  --lock test/experiments/insert_sync/logical_plan/corpus/frozen-363.lock.json \
  --cas-root /disk/oahs-corpus/cas \
  --opt /disk/build/bin/ptoas-opt \
  --output /disk/oahs-corpus/replay-002 \
  --arm both \
  --timeout 120
```

The result directory must not already exist and must be outside Git/source
roots and the CAS. Each invocation receives a newly materialized, hash-checked
prepared snapshot in that result directory; it never receives an original
source path. After every invocation the snapshot and corresponding original are
checked again when the record is source-backed, and all source-backed originals
are rechecked at completion. When a CAS is supplied, the complete
original/prepared CAS population is re-derived and rechecked after the replay.
The compiler executable is also hashed before replay and checked for change
afterward.

Outcome differences are evidence, not a harness error: replay records the exact
expected and actual outcomes, argv, and match bit for every selected arm and
continues through the population. The actual argv is retained in each result.
A completed replay writes `summary.json` with
the manifest hash, candidate and reference compiler hashes, and
exact-match/deviation counts. Input, prepared
byte, CAS, compiler-mutation, or source-mutation failures remain fatal. Neither
result file is qualification for device behavior or a compilation-time claim.

## Local validation of this milestone

The 11 serial unit tests pass, including corrupted provenance, command-allowlist
bypass, preflight-before-subprocess ordering, symlink refusal, rootless replay,
and mutation of CAS bytes during a candidate invocation. Export and rootless
verification cover all 363 records using 663 deduplicated objects for the 726
original/prepared references. No payload object is checked into Git.

The periodic-candidate native driver
`e166d5a8ac833e70527a13a47c2949e3797870af08f4494fcdf512208caf8442`
was then replayed rootlessly with both locked public-option arms. All **726/726**
comparisons match the frozen results, with zero deviations in statuses,
refusals, or mechanism counts. This validates the runner against the recorded
reference outcomes without changing their historical provenance. The report is
under the existing build's `test-results/oahs-periodic-frozen-replay`.

The separate cohorts remain **45/150 PTOAS**, **7/35 PyPTO**, and **145/178
PyPTO-lib** snapshot records. There are 336 unique original-byte hashes across
363 records. These denominators are neither unique production kernels nor an
end-to-end regeneration or device-qualification result.
