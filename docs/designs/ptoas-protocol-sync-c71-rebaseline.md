# C7.1 ProtocolSync rebaseline

## Status and reproducibility boundary

The host experiments below completed on 2026-09-05 and were rerun on clean
commit `1e911a4e00b62ad63846e531e88e11936a38ce16`. All three runs recorded a
clean, stable source revision, without `--allow-dirty`. Their semantic
aggregates exactly match the earlier patch-frozen runs; only timing fields
changed. No new protocol or repair was enabled in this freeze, and nothing
was pushed during this work.

The clean archive is disk-backed, 11,749,756 bytes:

```text
build/protocol-sync-c71-1e911a4e0-evidence.tar.gz
SHA256 1802eb8ac72e0d91f6b9a7fb3a0cbe268aa7842ee2be829ab4eedcb72ad9a461
```

Its [compact manifest and aggregates](protocol-sync-evidence/c71/clean-1e911a4e0/archive.json)
record the experiment/source commit, toolchain revisions, commands, input
hashes, per-row hashes and archive hash. The archive contains the inputs and
raw results, not just their hashes. It is retained locally, not uploaded to CI.

### Earlier patch-frozen snapshot (retained for comparison)

Source base: `8c3a0b58d755fe770cbe37bdc6401ab9d0c5b3a1`, whose diagnostic
commits sit above E/F and the A2/A3 capability merge `c4f924114`. The exact
tracked patch digest used by all three campaigns is:

```text
53c3876936d8295739be87fb54bc262f4e85b51f4c0fe10ba77a3f73bd96213b
```

The snapshot contains the patch itself, all executed experiment scripts,
commands, per-row JSONL, raw compressed diagnostics, input archives, CMake
cache, source/toolchain identities, and per-file/per-row hashes. Source and
compiler stability checks passed for every run.

The raw archive is disk-backed, 11,764,866 bytes:

```text
build/protocol-sync-c71-8c3a0b58d-evidence.tar.gz
SHA256 49e828ecc766f4a6812c006eedd38cf323d0eda59a9ace0b03cfce480fe3747b
```

Reviewable records are in
[`protocol-sync-evidence/c71/patch-frozen-8c3a0b58d/`](protocol-sync-evidence/c71/patch-frozen-8c3a0b58d/archive.json).
The archive retains the actual inputs, not only their hashes. It is local;
there is no claim that a versioned CI artifact has been uploaded.

The host uses Python 3.12.13, LLVM/MLIR 19.1.7 at
`fa2fd1f75d742fec91ecf40d98c9c7961647ec42`, GCC 15.3.1 and CMake 3.31.11.
The LLVM checkout has a two-line `<cstdint>` include adjustment in
`ValueBoundsOpInterfaceImpl.h`; the exact patch and hash are captured, so the
toolchain is not falsely described as a pristine upstream checkout. CANN and
PTO-ISA runtimes were not exercised.

The diagnostic infrastructure is committed and the clean rerun is complete.
The runner and archive commands are documented in
[`test/experiments/protocol_sync/README.md`](../../test/experiments/protocol_sync/README.md).

## Differential population: all 394 rows

Both C.6 and C.7 completed 394/394 rows, covering 405 function instances,
without compiler failures, missing statistics or timeouts.

- C.6: 1,050 diagnostic candidates: 1,020 completion cuts, 22 shared one-shot
  frontiers and eight balanced-choice round trips.
- C.7: 3,492 tracks; 18,025 projected of 18,151 attempted local accesses;
  29,770 track occurrences; 2,515 multi-family tracks.
- 37,261 transitions: 1,020 completion, 334 ready, 237 release and 35,670
  residual records.
- All 66,839 raw pairs occur exactly once. The projection census again checks
  1,276,033 access-pair relations with zero mask/overlap mismatches. Linear
  frontier containment covers 54,217 memberships; 12,622 remain outside that
  generally proven subset.
- Independent lifecycle reconstruction still finds 177 shapes in 76 function
  instances, with no corpus admission by strict E.

All 40 directly comparable storage counters match the earlier local
`44e3bb474` run. These are exactness/accounting claims relative to the current
extracted ranges, not proof that dynamic-subview or valid-shape extraction is
semantically exact. No C.6/C.7 candidate is selectable.

Atoms per projected access: p50 **1**, p95 **4**, maximum **33**. The current
storage analysis in the clean run takes 222 us p50, 3,097 us p95 and 25,170 us
maximum across 405 function instances (276,330 us aggregate). This is one host run, without a
memory-use/scaling or device-performance claim.

## First failed E proof for the 474 recurring records

| First failure | Records | Function instances | Function names | Topology hashes |
|---|---:|---:|---:|---:|
| Loop cardinality | 244 | 68 | 27 | 20 |
| Multiple loops | 88 | 20 | 4 | 3 |
| Choice | 50 | 11 | 5 | 4 |
| Unsupported visibility | 92 | 14 | 7 | 5 |

This follows E's actual first-failure scan order. Cardinality failures can
also contain nested loops or choices; the table does not predict admission
gains from relaxing cardinality alone. For example, `dispatch_gather` first
encounters a known-nonempty loop, then a nested loop.

The older 72-visibility/20-incomplete-channel split is not the current first
failure breakdown: those 92 records now stop at visibility. This same split
was already present in the `44e3bb474` local run. Topology hashes group ordered
control regions and physical phase resources, not semantic equivalence.

## The 126 unprojected accesses

All are non-physical local accesses in PyPTO-Lib: 96 in VEC and 30 in MAT.

| Operation | Accesses |
|---|---:|
| `pto.tpop` | 62 |
| `pto.tconcat` | 20 |
| `pto.tmuls` | 14 |
| `pto.tmov` | 12 |
| `pto.tcvt` | 8 |
| `pto.trowexpandmul` | 6 |
| `pto.textract` | 4 |

Each record retains its case, source file, function, access, operation, root,
address space and rejection. This points to queue-origin provenance and its
consumers. It does not identify 126 missing GM regions: the denominator here
is the local-storage projection experiment.

## Multi-family track classification

| First observable shape | Tracks |
|---|---:|
| Different lexical allocation scopes | 1,094 |
| Same recorded allocation root / equivalent-root views | 76 |
| Straight-line overlapping lexical use spans | 101 |
| Straight-line sequential lexical use spans | 52 |
| Remaining guarded or looping uses | 1,192 |

These are diagnostic shapes, not asynchronous lifetime proofs. Different
lexical scopes are only **numeric-address-coincidence candidates**; two scopes
can still access the same physical memory instance. Do not assign a different
alias domain to every allocation or lexical region. Likewise, sequential issue
does not by itself reclaim storage. The 52 sequential-use tracks are a bounded
starting population for exact reuse repair, not 52 already-safe reuses.

## F concrete audit: all 199 old emitted programs

All 199 programs parsed and received concrete-verifier verdicts, covering
204 functions. Seventeen programs passed completely. At function level:

| Concrete result / first failed stage | Functions |
|---|---:|
| Accepted | 17 |
| Schedule extraction | 58 |
| Recurring-event reconstruction | 65 |
| Static-event reconstruction | 34 |
| Barrier reconstruction | 8 |
| Unmodeled fixed synchronization | 6 |
| Residual obligations after reconstruction | 16 |

Thus 171 of the 187 rejected functions stop before obligation discharge.
This is evidence about the verifier's import surface, not a finding that 187
old functions contain races. Even the 16 residual-stage rejections need
localized witnesses before interpreting them as concrete legacy defects.
The result reproduces the earlier draft's 17 accepted / 187 rejected function
counts and now explains the first failed stages. An honest exact legacy
coverage percentage still requires canonical obligation IDs and broader fixed
supply import.

## Acceptance population and GM alias decision

The separate 152-kernel diagnosis is preserved as
[`acceptance-reported.json`](protocol-sync-evidence/c71/acceptance-reported.json):
four admitted, 148 rejected at `c4f924114`. Its exact inputs, manifest and
per-kernel raw results are unavailable locally. These historical counts remain
externally reported, not locally reproduced.

The user explicitly selected a new acceptance population from current-main
PyPTO and PyPTO-Lib, rather than recovery of those exact historical inputs.
Freeze frontend revisions and input hashes before measuring; report the actual
row count, with collection failures retained. The delivery goal is all valid
collected A2/A3 kernels compiling natively in both GM alias modes, without
legacy fallback. This fresh acceptance campaign is not yet complete.

The new runner's acceptance mode records actual strict mixed admission plus
independent schedule/storage and empty-world residual snapshots. All exposed
blocker classes remain per kernel even when strict emission fails earlier.
Unknown downstream facts are not counted as passed proofs.

A two-row smoke test admits the existing linear load/compute/store fixture
and rejects the unresolved GM write/read fixture. It is a runner test, not an
expansion of the 152-kernel population.

The user's two-mode GM policy is recorded in the
[GM alias contract](ptoas-protocol-sync-gm-alias-contract.md): safe may-alias by
default; opt-in disjoint **argument-root** promises; same-root views remain
checked. Safe-mode obligations must first be tested against existing transitive
order. Completion and GM visibility remain distinct effects. The mode switch
is not yet implemented in this diagnostic snapshot.

## Validation and next implementation slice

```text
cmake --build build --parallel 2
  passed; existing serial-LTO scheduling notices only
llvm-lit -j 2 build/test/lit
  1,892 passed, 1 unsupported, 0 failed
llvm-lit -j 1 -v --filter protocol_sync build/test/lit
  35 passed, including all four C++ ProtocolSync unit executables
.venv/bin/python test/experiments/protocol_sync/test_records.py
  18 passed
check_changed_code.py --repo . --base origin/codex/protocol-sync
  16 code files, 0 errors, 0 warnings
```

The next semantic slice remains access precision/provenance and the explicit
GM alias contract, then exact straight-line physical reuse through canonical
obligations. Keep ordinary loop repair and concrete loop verification together.
Model `set_validshape` as metadata affecting later accesses, not as a name to
whitelist. Preserve authoritative core ownership. Communication, queue
ownership and ACC effects still require separate qualified contracts. Stable
and alternating L1/ACC protocol recognizers remain postponed.
